#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SPEC-01 smoke: is a deterministic n-gram A/B even possible on this model?
#
# WHY THIS IS ONLY A SMOKE TEST. The formal SPEC-01 is nine turns in four
# counterbalanced pairs - 11 to 15 hours. It is only worth scheduling if greedy
# decoding actually produces IDENTICAL output with and without speculation.
# Three turns answers that for a fraction of the cost, and the stop rule is the
# whole point: the moment a turn's visible text differs between arms, the formal
# run is pointless and must not be scheduled.
#
# WHAT WENT WRONG BEFORE, and what this fixes:
#   - The published "+16.4% at 37% acceptance" mixed two different runs. The
#     completed default-sampler run had 49.7% acceptance; 37% came from the
#     failed temp-0 run.
#   - The previous temp-0 baseline had NO per-request output cap. Its first
#     request generated 261,735 tokens and the connection dropped. Every turn
#     here is capped at 24,576 tokens.
#   - The arms generated different text, so their wall-clock gap was not an
#     isolated speculation effect. Hence hashing.
#
# This is n-gram speculation. It is NOT DFlash and not an MTP draft head -
# neither exists for this model.
#
#   ./benchmark/spec01_smoke.sh --execute --plan-id CORRECTION-R1
# -----------------------------------------------------------------------------
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PLAN_ID_REQUIRED="CORRECTION-R1"
EXECUTE=0; PLAN_ID=""; TURNS="${TURNS:-3}"; MAXTOK="${MAXTOK:-24576}"
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --plan-id) PLAN_ID="$2"; shift 2 ;;
    --turns)   TURNS="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done
if [ "$EXECUTE" != 1 ]; then
  sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo; echo "Nothing was run. Pass --execute --plan-id $PLAN_ID_REQUIRED."
  exit 0
fi
[ "$PLAN_ID" = "$PLAN_ID_REQUIRED" ] || { echo "refusing: --plan-id must be $PLAN_ID_REQUIRED" >&2; exit 64; }

source benchmark/gpu_settle.sh
export COOLDOWN_MIN="${COOLDOWN_MIN:-120}" COOLDOWN_TEMP="${COOLDOWN_TEMP:-42}" COOLDOWN_MAX="${COOLDOWN_MAX:-900}"
IMG="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="results/corrections/${STAMP}_SPEC-01-smoke"
mkdir -p "$(dirname "$OUT")"; mkdir "$OUT" || { echo "refusing: $OUT exists"; exit 1; }
log(){ echo "$*" | tee -a "$OUT/controller.log"; }
down(){ ./scripts/serve.sh --down >/dev/null 2>&1; sleep 2; }
cleanup(){ kill "${TELE_PID:-0}" 2>/dev/null; down; }
trap cleanup EXIT

log "SPEC-01 smoke | $TURNS turns | output cap $MAXTOK tok/turn | temp 0, top-k 1"
log "evidence -> $OUT"
exec 9>/tmp/queue_rest.lock
log "acquiring the benchmark lock ..."; flock 9; log "lock acquired"

gmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
gtemp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
log "  preflight: GPU ${gmem} MiB / ${gtemp} C"
[ "$gmem" -gt 2000 ] 2>/dev/null && { log "  STOP: GPU busy"; exit 1; }

run_arm(){   # $1 = none | ngram-mod
  # NOTE: two statements on purpose. `local a="$1" b="$OUT/$a"` declares both
  # names before assigning, so $a is still unset when b is evaluated and `set -u`
  # aborts the script.
  local spec="$1"
  local d="$OUT/$spec"
  mkdir -p "$d"
  log "  --- arm spec=$spec ---"
  down; gpu_settle 2>&1 | tee -a "$OUT/controller.log"
  TEMP=0 TOP_K=1 SPEC_TYPE="$spec" LOAD_MODE=none LAZY=off UBATCH=512 \
    PARALLEL=1 NGL=999 N_CPU_MOE=0 LLAMA_IMAGE="$IMG" \
    ./scripts/serve.sh --quant UD-IQ4_XS --ctx 262144 --load-mode none \
      --spec "$spec" --lazy off --no-wait >/dev/null 2>&1
  local ok=0 i
  for i in $(seq 1 240); do
    curl -sf -m 5 http://localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }
    case "$(docker ps -a --filter name=q38n --format '{{.Status}}' | head -1)" in
      Exited*|Restarting*) break ;;
    esac
    sleep 5
  done
  docker logs "$(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1)" \
    > "$d/server.log" 2>&1
  [ "$ok" = 1 ] || { log "      server did not start"; echo fail > "$d/FAILED"; return 1; }
  curl -s http://localhost:8000/props > "$d/props.json"

  uv run benchmark/gpu_telemetry.py watch --seconds 5400 --interval 2 --max-temp 84 \
      --samples --out "$d/gpu.json" > "$d/gpu_telemetry.log" 2>&1 &
  TELE_PID=$!

  # --max-tokens is the cap the previous attempt lacked.
  uv run benchmark/bench_ngram.py --url http://localhost:8000 --turns "$TURNS" \
      --tag "spec01smoke-$spec" --temperature 0 --top-k 1 --max-tokens "$MAXTOK" \
      2>&1 | tee "$d/bench.log" \
      | grep -E "overall decode|draft accept|total wall|abort|error|Error" | sed 's/^/      /' \
      | tee -a "$OUT/controller.log"

  kill -TERM "${TELE_PID:-0}" 2>/dev/null; sleep 10; kill -KILL "${TELE_PID:-0}" 2>/dev/null
  TELE_PID=""
  # bench_ngram writes the FULL generated text, which is what makes hashing possible
  # capture the server log AFTER the turns, not just after startup: the pre-run
  # copy only ever contains boot lines and cannot show a mid-run drop.
  cn=$(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1)
  [ -n "$cn" ] && docker logs "$cn" > "$d/server.log.after" 2>&1
  local run
  run=$(ls -dt benchmark/bench_results/*"spec01smoke-$spec" 2>/dev/null | head -1)
  if [ -n "$run" ]; then
    cp -r "$run" "$d/bench_results"
    log "      captured $(ls "$d/bench_results/responses" 2>/dev/null | wc -l) turn responses"
  fi
}

run_arm none
run_arm ngram-mod
down

log ""
log "########## per-turn visible-text hashes ##########"
python3 - "$OUT" <<'PY'
import hashlib,json,pathlib,sys
o=pathlib.Path(sys.argv[1])
def turns(arm):
    d=o/arm/"bench_results"/"responses"
    if not d.exists(): return {}
    out={}
    for f in sorted(d.glob("*.md")):
        t=f.read_text()
        # strip the harness's own header line so we hash only model output
        body=t.split("\n\n",1)[1] if "\n\n" in t else t
        out[f.stem]=(hashlib.sha256(body.encode()).hexdigest()[:16],len(body))
    return out
a,b=turns("none"),turns("ngram-mod")
keys=sorted(set(a)|set(b))
if not keys:
    print("  no responses captured - cannot compare"); raise SystemExit(1)
print(f"  {'turn':<6}{'baseline':<20}{'ngram-mod':<20}{'chars a/b':<18}match")
allmatch=True; first_div=None
for k in keys:
    ha,la=a.get(k,("-",0)); hb,lb=b.get(k,("-",0))
    m = ha==hb and ha!="-"
    if not m:
        allmatch=False; first_div=first_div or k
    print(f"  {k:<6}{ha:<20}{hb:<20}{f'{la}/{lb}':<18}{'YES' if m else 'NO'}")
res={"identical":allmatch,"first_divergent_turn":first_div,
     "baseline":{k:v[0] for k,v in a.items()},"ngram":{k:v[0] for k,v in b.items()}}
(o/"hashes.json").write_text(json.dumps(res,indent=2))
print()
if allmatch:
    print("  -> OUTPUT IS IDENTICAL. A deterministic A/B is valid, so the wall-clock")
    print("     difference IS the speculation effect. The formal 9-turn run is worth")
    print("     scheduling.")
else:
    print(f"  -> OUTPUT DIVERGES (first at turn {first_div}). Greedy decoding is NOT")
    print("     output-lossless here, so a wall-clock comparison cannot isolate the")
    print("     speculation effect. DO NOT schedule the formal 9-turn run; report the")
    print("     decode RATE only, and say the arms produced different text.")
PY
log "DONE -> $OUT"
