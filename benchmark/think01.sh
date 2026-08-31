#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# THINK-01: does preserving earlier turns' reasoning help or hurt?
#
# BOTH ARMS RUN IN THINKING MODE. The model generates a full reasoning trace on
# every turn either way. The flag only controls whether EARLIER turns' reasoning
# is sent back in the history:
#
#   preserve = on   history is append-only, so the prompt prefix never changes
#                   and the KV cache keeps hitting - but the prompt grows by a
#                   whole reasoning trace every turn.
#   preserve = off  prompts are far shorter, but dropping earlier thinking
#                   rewrites the middle of the history, invalidating the cached
#                   prefix.
#
# Fewer tokens against cacheable tokens. Which wins is empirical.
#
# WHY THE PREVIOUS RESULT IS ONLY EXPLORATORY, and what this fixes:
#   - both arms ran the NON-thinking sampler (temp 0.7 / top-p 0.80) while in
#     thinking mode. Corrected here to the model card's thinking values:
#     temp 1.0, top-p 0.95, top-k 20, min-p 0.0, presence 0.0, repeat 1.0.
#   - one run per arm. Here: two sessions per arm.
#   - preserve=on always ran first. Here: on, off, off, on.
#   - both arms shared one server. THIS TEST IS ABOUT KV CACHE BEHAVIOUR, so a
#     second arm running against a server warmed by the first contaminates
#     exactly what is being measured. Every arm gets a fresh server here.
#
# METRIC - and an earlier version of this header got it wrong.
# It claimed wall clock is meaningful here because both arms answer the same
# prompts. Measured 2026-08-30, that is false: at temperature 1.0 the arms
# generate different amounts of reasoning. The first pair gave 91,280 tokens on
# the ON arm against 148,214 on OFF - 62% more - so wall clock partly measures
# how much work was done, exactly the confound that rules it out for the
# speculation test.
#
# Compare instead:
#   decode tok/s at comparable prompt length  (the cost of a long prompt)
#   total tokens PREFILLED                    (the cache benefit)
#   final prompt size                         (how fast context is consumed)
# Tokens prefilled and served from cache are recorded per turn.
#
#   ./benchmark/think01.sh --execute --plan-id CORRECTION-R1
# -----------------------------------------------------------------------------
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PLAN_ID_REQUIRED="CORRECTION-R1"
EXECUTE=0; PLAN_ID=""; TURNS="${TURNS:-5}"; MAXTOK="${MAXTOK:-49152}"
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --plan-id) PLAN_ID="$2"; shift 2 ;;
    --turns)   TURNS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done
if [ "$EXECUTE" != 1 ]; then
  sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo; echo "Nothing was run. Pass --execute --plan-id $PLAN_ID_REQUIRED."
  exit 0
fi
[ "$PLAN_ID" = "$PLAN_ID_REQUIRED" ] || { echo "refusing: --plan-id must be $PLAN_ID_REQUIRED" >&2; exit 64; }

source benchmark/gpu_settle.sh
source benchmark/dump_compose.sh
export COOLDOWN_MIN="${COOLDOWN_MIN:-120}" COOLDOWN_TEMP="${COOLDOWN_TEMP:-42}" COOLDOWN_MAX="${COOLDOWN_MAX:-900}"
IMG="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"

# Model card, THINKING mode.
TEMP_V=1.0; TOPP_V=0.95; TOPK_V=20; MINP_V=0.0

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="results/corrections/${STAMP}_THINK-01"
mkdir -p "$(dirname "$OUT")"; mkdir "$OUT" || { echo "refusing: $OUT exists"; exit 1; }
log(){ echo "$*" | tee -a "$OUT/controller.log"; }
down(){ ./scripts/serve.sh --down >/dev/null 2>&1; sleep 2; }
cleanup(){ kill "${TELE_PID:-0}" 2>/dev/null; down; }
trap cleanup EXIT

log "THINK-01 | thinking mode both arms | $TURNS turns | cap $MAXTOK | order on,off,off,on"
log "  sampler: temp $TEMP_V top-p $TOPP_V top-k $TOPK_V min-p $MINP_V (model card THINKING)"
log "evidence -> $OUT"
exec 9>/tmp/queue_rest.lock
log "acquiring the benchmark lock ..."; flock 9; log "lock acquired"
gmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
gtemp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
log "  preflight: GPU ${gmem} MiB / ${gtemp} C"
if [ "$gmem" -gt 2000 ] 2>/dev/null; then log "  STOP: GPU busy"; exit 1; fi
if [ "$gtemp" -gt 45 ] 2>/dev/null; then log "  STOP: GPU too hot"; exit 1; fi

run_arm(){   # $1 = on|off   $2 = session label
  local arm="$1"
  local sess="$2"
  local name="${sess}_${arm}"
  local d="$OUT/$name"
  mkdir -p "$d"
  log "  --- $name (preserve_thinking=$arm) ---"
  down; gpu_settle 2>&1 | tee -a "$OUT/controller.log"
  # Record the resolved compose for this arm - same variables as the serve call.
  TEMP="$TEMP_V" TOP_P="$TOPP_V" TOP_K="$TOPK_V" MIN_P="$MINP_V" \
  CTX=262144 LOAD_MODE=none LAZY=off UBATCH=512 PARALLEL=1 NGL=999 \
  N_CPU_MOE=0 LLAMA_IMAGE="$IMG" \
    dump_compose "$d" || log "      WARN: compose.txt not recorded for $name"

  TEMP="$TEMP_V" TOP_P="$TOPP_V" TOP_K="$TOPK_V" MIN_P="$MINP_V" \
  LOAD_MODE=none LAZY=off UBATCH=512 PARALLEL=1 NGL=999 N_CPU_MOE=0 \
  LLAMA_IMAGE="$IMG" \
    ./scripts/serve.sh --quant UD-IQ4_XS --ctx 262144 --load-mode none \
      --lazy off --no-wait >/dev/null 2>&1
  local ok=0 i
  for i in $(seq 1 240); do
    curl -sf -m 5 http://localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }
    case "$(docker ps -a --filter name=q38n --format '{{.Status}}' | head -1)" in
      Exited*|Restarting*) break ;;
    esac
    sleep 5
  done
  if [ "$ok" != 1 ]; then log "      server did not start"; echo fail > "$d/FAILED"; return 1; fi
  curl -s http://localhost:8000/props > "$d/props.json"

  # model-card conformance, gated: this test depends on what the model writes
  uv run benchmark/check_config.py --mode thinking --max-tokens "$MAXTOK" \
      --json "$d/config_check.json" > "$d/config_check.log" 2>&1
  grep -E 'CONFIG CHECK|FAULT' "$d/config_check.log" | sed 's/^/      card: /' | tee -a "$OUT/controller.log"
  if grep -q 'CONFIG CHECK: FAIL' "$d/config_check.log"; then
    log "      FAIL: server does not match the model card for thinking mode"
    echo config_mismatch > "$d/FAILED"; return 1
  fi

  uv run benchmark/gpu_telemetry.py watch --seconds 7200 --interval 2 --max-temp 84 \
      --samples --out "$d/gpu.json" > "$d/gpu_telemetry.log" 2>&1 &
  TELE_PID=$!

  uv run benchmark/preserve_thinking.py --arm "$arm" --turns "$TURNS" \
      --max-tokens "$MAXTOK" --out "$d" 2>&1 | tee "$d/run.log" \
      | grep -E 'turn |single arm|error|Error|Traceback' | sed 's/^/      /' \
      | tee -a "$OUT/controller.log"

  kill -TERM "${TELE_PID:-0}" 2>/dev/null; sleep 10; kill -KILL "${TELE_PID:-0}" 2>/dev/null
  TELE_PID=""
  cn=$(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1)
  [ -n "$cn" ] && docker logs "$cn" > "$d/server.log.after" 2>&1
  python3 -c "
import json
try: print('      telemetry verdict:', json.load(open('$d/gpu.json')).get('verdict'))
except Exception: print('      telemetry verdict: no file')"
}

run_arm on  s1
run_arm off s1
run_arm off s2
run_arm on  s2
down

log ""
log "########## preserve_thinking, thinking sampler, counterbalanced ##########"
python3 - "$OUT" <<'PY'
import json,pathlib,sys,statistics as st
o=pathlib.Path(sys.argv[1]); arms={"on":[],"off":[]}
for d in sorted(p for p in o.iterdir() if p.is_dir()):
    f=d/"preserve_thinking.json"
    if not f.exists(): continue
    j=json.loads(f.read_text())
    for k in ("on","off"):
        if k in j: arms[k].append((d.name,j[k]))
print(f"  {'arm':<10}{'turns':>7}{'final prompt':>14}{'prefilled':>11}{'cached':>11}{'wall s':>10}")
for k in ("on","off"):
    for n,v in arms[k]:
        print(f"  {n:<10}{v.get('turns_completed','-'):>7}{v.get('final_prompt_tokens',0):>14,}"
              f"{v.get('total_prefilled',0):>11,}{v.get('total_cached',0):>11,}"
              f"{v.get('total_wall_s',0):>10,.1f}")
res={}
for k in ("on","off"):
    w=[v.get("total_wall_s",0) for _,v in arms[k] if v.get("total_wall_s")]
    pf=[v.get("total_prefilled",0) for _,v in arms[k]]
    ca=[v.get("total_cached",0) for _,v in arms[k]]
    if w: res[k]={"wall_mean":round(st.mean(w),1),
                  "wall_sd":round(st.stdev(w),1) if len(w)>1 else 0.0,
                  "n":len(w),"prefilled_mean":round(st.mean(pf)),
                  "cached_mean":round(st.mean(ca))}
for k,v in res.items(): print(f"\n  preserve={k}: {v}")
if "on" in res and "off" in res:
    a,b=res["on"]["wall_mean"],res["off"]["wall_mean"]
    print(f"\n  wall clock: on {a:.1f}s  off {b:.1f}s  -> off is {100*(1-b/a):+.1f}% vs on")
    spread=max(res['on']['wall_sd'],res['off']['wall_sd'])
    if spread > abs(a-b)/2:
        print("  WARNING: within-arm spread is large relative to the difference.")
        print("           Treat as indicative, not established.")
(o/"summary.json").write_text(json.dumps(res,indent=2))
print(f"\n  wrote {o}/summary.json")
PY
log "DONE -> $OUT"
