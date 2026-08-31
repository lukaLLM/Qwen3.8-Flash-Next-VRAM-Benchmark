#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# CONC-01: unified vs non-unified KV, counterbalanced and repeated.
#
# WHAT THE ORIGINAL RUN GOT WRONG. One run per arm, fixed order, and the report
# then described the gap as possibly a context-budget difference. That reading
# was wrong and is corrected here: BOTH arms are configured with the same total
# context (131,072) and the same 16 slots. Unified KV reports the shared pool;
# non-unified reports roughly 8,192 per slot. Each request is ~3,600 prompt plus
# 256 output tokens and fits either way, so this compares TWO CACHE LAYOUTS, not
# two context budgets. Neither mode gives 131,072 tokens independently to all 16
# requests.
#
# WHAT THIS ADDS.
#   - order unified, non-unified, non-unified, unified, so each layout takes an
#     early and a late position. SPEC-01 showed how badly a fixed order can
#     mislead: there, one pair read +12% and the reverse pair read -1.7%.
#   - three repeats of the whole level sweep per server start.
#   - the per-slot and shared-pool context values recorded from /props, so the
#     cache-layout claim is evidence rather than assertion.
#   - per-arm GPU telemetry with a throttle verdict.
#
#   ./benchmark/conc01.sh --execute --plan-id CORRECTION-R1
# -----------------------------------------------------------------------------
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PLAN_ID_REQUIRED="CORRECTION-R1"
EXECUTE=0; PLAN_ID=""; REPEATS="${REPEATS:-3}"; LEVELS="${LEVELS:-1,2,4,8,16}"
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --plan-id) PLAN_ID="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done
if [ "$EXECUTE" != 1 ]; then
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo; echo "Nothing was run. Pass --execute --plan-id $PLAN_ID_REQUIRED."
  exit 0
fi
[ "$PLAN_ID" = "$PLAN_ID_REQUIRED" ] || { echo "refusing: --plan-id must be $PLAN_ID_REQUIRED" >&2; exit 64; }

source benchmark/gpu_settle.sh
source benchmark/dump_compose.sh
export COOLDOWN_MIN="${COOLDOWN_MIN:-120}" COOLDOWN_TEMP="${COOLDOWN_TEMP:-42}" COOLDOWN_MAX="${COOLDOWN_MAX:-900}"
IMG="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="results/corrections/${STAMP}_CONC-01"
mkdir -p "$(dirname "$OUT")"; mkdir "$OUT" || { echo "refusing: $OUT exists"; exit 1; }
log(){ echo "$*" | tee -a "$OUT/controller.log"; }
down(){ ./scripts/serve.sh --down >/dev/null 2>&1; sleep 2; }
cleanup(){ kill "${TELE_PID:-0}" 2>/dev/null; down; }
trap cleanup EXIT

log "CONC-01 | levels $LEVELS | $REPEATS repeats per server | order unified,non,non,unified"
log "evidence -> $OUT"
exec 9>/tmp/queue_rest.lock
log "acquiring the benchmark lock ..."; flock 9; log "lock acquired"
gmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
gtemp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
log "  preflight: GPU ${gmem} MiB / ${gtemp} C"
if [ "$gmem" -gt 2000 ] 2>/dev/null; then log "  STOP: GPU busy"; exit 1; fi
if [ "$gtemp" -gt 45 ] 2>/dev/null; then log "  STOP: GPU too hot"; exit 1; fi

run_arm(){   # $1 = true|false (kv unified)   $2 = position label
  local kvu="$1"
  local pos="$2"
  local name="${pos}_kv${kvu}"
  local d="$OUT/$name"
  mkdir -p "$d"
  log "  --- $name (KV_UNIFIED=$kvu) ---"
  down; gpu_settle 2>&1 | tee -a "$OUT/controller.log"
  # Record the resolved compose for this arm - same variables as the serve call.
  PARALLEL=16 KV_UNIFIED="$kvu" CTX=131072 LOAD_MODE=none LAZY=off \
  UBATCH=512 NGL=999 N_CPU_MOE=0 LLAMA_IMAGE="$IMG" \
    dump_compose "$d" || log "      WARN: compose.txt not recorded for $name"

  PARALLEL=16 KV_UNIFIED="$kvu" LLAMA_IMAGE="$IMG" LOAD_MODE=none LAZY=off \
    UBATCH=512 NGL=999 N_CPU_MOE=0 \
    ./scripts/serve.sh --quant UD-IQ4_XS --ctx 131072 --load-mode none \
      --kv-unified "$kvu" --lazy off --no-wait >/dev/null 2>&1
  local ok=0 i
  for i in $(seq 1 240); do
    curl -sf -m 5 http://localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }
    case "$(docker ps -a --filter name=q38n --format '{{.Status}}' | head -1)" in
      Exited*|Restarting*) break ;;
    esac
    sleep 5
  done
  [ "$ok" = 1 ] || { log "      server did not start"; echo fail > "$d/FAILED"; return 1; }
  curl -s http://localhost:8000/props > "$d/props.json"
  # the cache-layout evidence: what context each mode actually reports
  python3 - "$d/props.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); g=d.get("default_generation_settings") or {}
print(f"      slots {d.get('total_slots')}  n_ctx(reported) {g.get('n_ctx')}")
PY
  uv run benchmark/gpu_telemetry.py watch --seconds 3600 --interval 2 --max-temp 84 \
      --samples --out "$d/gpu.json" > "$d/gpu_telemetry.log" 2>&1 &
  TELE_PID=$!
  local r
  for r in $(seq 1 "$REPEATS"); do
    log "      repeat $r/$REPEATS"
    uv run benchmark/concurrency.py --levels "$LEVELS" --n-predict 256 \
        --out "$d/conc_r${r}.json" 2>&1 | grep -E '^\s+c=|efficiency|aggregate' \
        | sed 's/^/        /' | tee -a "$OUT/controller.log"
  done
  kill -TERM "${TELE_PID:-0}" 2>/dev/null; sleep 10; kill -KILL "${TELE_PID:-0}" 2>/dev/null
  TELE_PID=""
  cn=$(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1)
  [ -n "$cn" ] && docker logs "$cn" > "$d/server.log.after" 2>&1
  python3 -c "
import json
try:
    print('      telemetry verdict:', json.load(open('$d/gpu.json')).get('verdict'))
except Exception: print('      telemetry verdict: no file')"
}

run_arm true  p1
run_arm false p1
run_arm false p2
run_arm true  p2
down

log ""
log "########## unified vs non-unified KV, $REPEATS repeats, counterbalanced ##########"
python3 - "$OUT" <<'PY'
import json,pathlib,sys,statistics as st
o=pathlib.Path(sys.argv[1]); data={}; per={}
for d in sorted(p for p in o.iterdir() if p.is_dir()):
    mode="unified" if d.name.endswith("true") else "non-unified"
    for f in sorted(d.glob("conc_r*.json")):
        try: j=json.loads(f.read_text())
        except Exception: continue
        # concurrency.py writes a bare LIST of {concurrency, aggregate_tps, ...}
        rowsrc = j if isinstance(j, list) else (j.get("levels") or j.get("results") or [])
        for row in rowsrc:
            c=row.get("concurrency") or row.get("c")
            agg=row.get("aggregate_tps") or row.get("aggregate")
            pr=row.get("per_request_tps")
            if c and agg:
                data.setdefault(mode,{}).setdefault(int(c),[]).append(float(agg))
                per.setdefault(mode,{}).setdefault(int(c),[]).append(float(pr or 0))
if not data:
    print("  no concurrency rows parsed - inspect conc_r*.json"); raise SystemExit
lv=sorted({c for m in data.values() for c in m})
print(f"  {'conc':>6}{'unified':>22}{'non-unified':>22}{'ratio':>8}")
for c in lv:
    u=data.get("unified",{}).get(c,[]); n=data.get("non-unified",{}).get(c,[])
    us=f"{st.mean(u):.1f} +/- {(st.stdev(u) if len(u)>1 else 0):.1f} (n={len(u)})" if u else "-"
    ns=f"{st.mean(n):.1f} +/- {(st.stdev(n) if len(n)>1 else 0):.1f} (n={len(n)})" if n else "-"
    r=f"{st.mean(n)/st.mean(u):.2f}x" if u and n else "-"
    print(f"  {c:>6}{us:>22}{ns:>22}{r:>8}")
json.dump({"aggregate":{m:{str(c):v for c,v in d.items()} for m,d in data.items()},
           "per_request":{m:{str(c):v for c,v in d.items()} for m,d in per.items()}},
          open(o/"summary.json","w"),indent=2)
print(f"\n  wrote {o}/summary.json")
print("  Both arms ran ctx 131072 with 16 slots. This compares CACHE LAYOUTS,")
print("  not context budgets - see props.json per arm.")
PY
log "DONE -> $OUT"
