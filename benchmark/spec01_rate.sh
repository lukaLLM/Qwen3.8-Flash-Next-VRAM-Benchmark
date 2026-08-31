#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SPEC-01, rate-based: what does n-gram speculation give on a REAL coding
# workload, in non-thinking mode, with the sampler the model card recommends?
#
# WHY THIS REPLACES THE GREEDY DESIGN.
# The greedy smoke test (temp 0, top-k 1) was an attempt to make the two arms
# produce identical output so that WALL CLOCK would isolate the speculation
# effect. It failed: output diverged at turn 1 even though /props confirmed
# temperature 0.0 and top_k 1 on both arms. Speculative decoding verifies
# several tokens per forward pass, so the batch shape differs from
# one-at-a-time decoding; floating-point reductions differ with batch shape and
# a near-tie argmax can flip. One flipped token diverges the whole
# continuation.
#
# So greedy buys nothing here, and it is also not how anyone runs a coding
# model. This design instead:
#
#   1. Uses NON-THINKING mode, so turns finish instead of being truncated at an
#      output cap mid-reasoning. In the greedy run, 2 of 3 turns hit the 24,576
#      cap and the harness discarded their text entirely.
#   2. Uses the model card's NON-THINKING sampler, because mode and sampler
#      must match: temp 0.7, top-p 0.80, top-k 20, min-p 0.0,
#      presence-penalty 1.5, repeat-penalty 1.0.
#   3. Scores DECODE TOKENS PER SECOND, not wall clock. With sampling the two
#      arms generate different text and different token counts, so wall clock
#      partly measures how much work was done. Rate is the fair comparison and
#      stays valid however the text differs.
#   4. Counterbalances the arm order across two pairs, because sampled runs
#      vary and a single fixed order cannot separate the arm from its position.
#
# Reported: decode t/s per arm, draft count and acceptance, completion tokens.
# This is n-gram speculation - NOT DFlash, and not an MTP draft head.
#
#   ./benchmark/spec01_rate.sh --execute --plan-id CORRECTION-R1
# -----------------------------------------------------------------------------
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PLAN_ID_REQUIRED="CORRECTION-R1"
# THINKING MODE, 3 turns, high cap. See the mode note below: with thinking off
# this model answers a coding task with tool calls and writes no code at all.
EXECUTE=0; PLAN_ID=""; TURNS="${TURNS:-3}"; MAXTOK="${MAXTOK:-49152}"
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --plan-id) PLAN_ID="$2"; shift 2 ;;
    --turns)   TURNS="$2"; shift 2 ;;
    --max-tokens) MAXTOK="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done
if [ "$EXECUTE" != 1 ]; then
  sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo; echo "Nothing was run. Pass --execute --plan-id $PLAN_ID_REQUIRED."
  exit 0
fi
[ "$PLAN_ID" = "$PLAN_ID_REQUIRED" ] || { echo "refusing: --plan-id must be $PLAN_ID_REQUIRED" >&2; exit 64; }

source benchmark/gpu_settle.sh
source benchmark/dump_compose.sh
export COOLDOWN_MIN="${COOLDOWN_MIN:-120}" COOLDOWN_TEMP="${COOLDOWN_TEMP:-42}" COOLDOWN_MAX="${COOLDOWN_MAX:-900}"
IMG="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"

# Model card, THINKING mode. Do not mix these with the non-thinking values.
TEMP_V=1.0; TOPP_V=0.95; TOPK_V=20; MINP_V=0.0; PRESP_V=0.0; REPP_V=1.0

# WHY THINKING MODE, MEASURED NOT ASSUMED.
# An earlier version of this test disabled thinking, to stop turns being
# truncated at the output cap mid-reasoning. That broke the workload: with
# thinking off, this model answers these coding prompts with a <tool_call> block
# and stops, waiting for a tool result the harness cannot provide. Checked with
# benchmark/check_outputs.py across both arms and both attempts:
#
#   thinking off : 9 of 9 turns tool calls, 0 turns containing code -> DEGENERATE
#   thinking on  : 0 turns tool calls, code in every turn           -> USABLE
#
# A system prompt forbidding tools did not change it. So thinking stays ON and
# the truncation is fixed the other way, by raising the cap to 49,152 so turns
# finish. Turn 1 needed 20,132 tokens; the old 24,576 cap cut off turns 2 and 3,
# whose text the harness then discarded entirely.



STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="results/corrections/${STAMP}_SPEC-01-rate"
mkdir -p "$(dirname "$OUT")"; mkdir "$OUT" || { echo "refusing: $OUT exists"; exit 1; }
log(){ echo "$*" | tee -a "$OUT/controller.log"; }
down(){ ./scripts/serve.sh --down >/dev/null 2>&1; sleep 2; }
cleanup(){ kill "${TELE_PID:-0}" 2>/dev/null; down; }
trap cleanup EXIT

log "SPEC-01 rate | THINKING mode | $TURNS turns | cap $MAXTOK tok/turn"
log "  sampler: temp $TEMP_V top-p $TOPP_V top-k $TOPK_V min-p $MINP_V presence $PRESP_V repeat $REPP_V"
log "  metric: decode tokens/second. Wall clock is NOT comparable - the arms"
log "          generate different text and different token counts."
log "evidence -> $OUT"
exec 9>/tmp/queue_rest.lock
log "acquiring the benchmark lock ..."; flock 9; log "lock acquired"

gmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
gtemp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
log "  preflight: GPU ${gmem} MiB / ${gtemp} C"
if [ "$gmem" -gt 2000 ] 2>/dev/null; then log "  STOP: GPU busy"; exit 1; fi

run_arm(){   # $1 = none|ngram-mod   $2 = pair label
  local spec="$1"
  local pair="$2"
  local name="${pair}_${spec}"
  local d="$OUT/$name"
  mkdir -p "$d"
  log "  --- $name ---"
  down; gpu_settle 2>&1 | tee -a "$OUT/controller.log"
  # Record the resolved compose for this arm - same variables as the serve call.
  TEMP="$TEMP_V" TOP_P="$TOPP_V" TOP_K="$TOPK_V" MIN_P="$MINP_V" \
  SPEC_TYPE="$spec" CTX=262144 LOAD_MODE=none LAZY=off UBATCH=512 PARALLEL=1 \
  NGL=999 N_CPU_MOE=0 LLAMA_IMAGE="$IMG" \
    dump_compose "$d" || log "      WARN: compose.txt not recorded for $name"

  TEMP="$TEMP_V" TOP_P="$TOPP_V" TOP_K="$TOPK_V" MIN_P="$MINP_V" \
  SPEC_TYPE="$spec" LOAD_MODE=none LAZY=off UBATCH=512 PARALLEL=1 NGL=999 \
  N_CPU_MOE=0 LLAMA_IMAGE="$IMG" \
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
  docker logs "$(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1)" > "$d/server.log" 2>&1
  if [ "$ok" != 1 ]; then log "      server did not start"; echo fail > "$d/FAILED"; return 1; fi
  curl -s http://localhost:8000/props > "$d/props.json"

  uv run benchmark/gpu_telemetry.py watch --seconds 5400 --interval 2 --max-temp 84 \
      --samples --out "$d/gpu.json" > "$d/gpu_telemetry.log" 2>&1 &
  TELE_PID=$!

  uv run benchmark/bench_ngram.py --url http://localhost:8000 --turns "$TURNS" \
      --tag "spec01rate-$name" \
      --temperature "$TEMP_V" --top-p "$TOPP_V" --top-k "$TOPK_V" --min-p "$MINP_V" \
      --presence-penalty "$PRESP_V" --repeat-penalty "$REPP_V" --max-tokens "$MAXTOK" \
      2>&1 | tee "$d/bench.log" \
      | grep -E "overall decode|draft accept|end-to-end|abort|error|Error" | sed 's/^/      /' \
      | tee -a "$OUT/controller.log"

  kill -TERM "${TELE_PID:-0}" 2>/dev/null; sleep 10; kill -KILL "${TELE_PID:-0}" 2>/dev/null
  TELE_PID=""
  # capture the server log AFTER the turns, not just after startup: the pre-run
  # copy only ever contains boot lines and cannot show a mid-run drop.
  cn=$(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1)
  [ -n "$cn" ] && docker logs "$cn" > "$d/server.log.after" 2>&1
  local run
  run=$(ls -dt benchmark/bench_results/*"spec01rate-$name" 2>/dev/null | head -1)
  [ -n "$run" ] && cp -r "$run" "$d/bench_results"
  # A speed number is only meaningful if the model did the workload. Check the
  # TEXT, not just the throughput, and record the verdict beside the arm.
  if [ -n "$run" ]; then
    uv run benchmark/check_outputs.py "$d/bench_results" --expect-code \
        --json "$d/output_check.json" 2>&1 | tee "$d/output_check.log" \
        | grep -E "VERDICT|median |  - " | sed 's/^/      /' | tee -a "$OUT/controller.log"
  fi
}

# counterbalanced: each arm gets one first position and one second
run_arm none      p1
run_arm ngram-mod p1
run_arm ngram-mod p2
run_arm none      p2
down

log ""
log "########## decode rate, non-thinking, recommended sampler ##########"
python3 - "$OUT" <<'PY'
import json,pathlib,sys,statistics as st
o=pathlib.Path(sys.argv[1])
rows={}
for d in sorted(p for p in o.iterdir() if p.is_dir()):
    f=d/"bench_results"/"stats.json"
    if not f.exists(): continue
    s=json.loads(f.read_text())
    agg=s.get("aggregates") or {}
    turns=[r for r in s.get("rows",[]) if not r.get("error")]
    tok=sum(r.get("completion_tokens",0) for r in turns)
    dec=[r["predicted_per_second"] for r in turns if r.get("predicted_per_second")]
    dn=sum(r.get("draft_n") or 0 for r in turns)
    da=sum(r.get("draft_n_accepted") or 0 for r in turns)
    capped=sum(1 for r in turns if r.get("completion_tokens")==49152)
    rows[d.name]={"turns":len(turns),"tokens":tok,
                  "decode_tps_mean":round(st.mean(dec),2) if dec else None,
                  "decode_tps_median":round(st.median(dec),2) if dec else None,
                  "draft_n":dn,"draft_accept_pct":round(100*da/dn,1) if dn else None,
                  "turns_at_cap":capped}
print(f"  {'arm':<18}{'turns':>6}{'tokens':>9}{'decode t/s':>12}{'draft acc':>11}{'at cap':>8}")
for k,v in rows.items():
    print(f"  {k:<18}{v['turns']:>6}{v['tokens']:>9}{str(v['decode_tps_mean']):>12}"
          f"{str(v['draft_accept_pct']):>11}{v['turns_at_cap']:>8}")
bad=[]
for d in sorted(p for p in o.iterdir() if p.is_dir()):
    f=d/"output_check.json"
    if f.exists():
        c=json.loads(f.read_text())
        rows.setdefault(d.name,{})["output_verdict"]=c.get("verdict")
        rows[d.name]["output_flags"]=c.get("flags")
        if c.get("verdict")!="USABLE": bad.append(d.name)
if bad:
    print(f"\n  OUTPUT CHECK FAILED on: {', '.join(bad)}")
    print("  The model did not do the workload on these arms, so their speed")
    print("  numbers describe the wrong activity. NOT REPORTABLE.")
base=[v["decode_tps_mean"] for k,v in rows.items() if k.endswith("_none") and v.get("decode_tps_mean")]
ng=[v["decode_tps_mean"] for k,v in rows.items() if k.endswith("_ngram-mod") and v.get("decode_tps_mean")]
res={"per_arm":rows}
if base and ng:
    b,n=st.mean(base),st.mean(ng)
    res.update({"baseline_decode_tps":round(b,2),"ngram_decode_tps":round(n,2),
                "rate_gain_pct":round(100*(n/b-1),1),
                "baseline_arms":base,"ngram_arms":ng})
    print(f"\n  baseline mean {b:.2f} t/s  (arms {base})")
    print(f"  ngram-mod mean {n:.2f} t/s  (arms {ng})")
    print(f"  --> decode RATE gain: {100*(n/b-1):+.1f}%")
    spread=max(abs(base[0]-base[-1]),abs(ng[0]-ng[-1])) if len(base)>1 and len(ng)>1 else None
    if spread is not None:
        print(f"  within-arm spread across order positions: {spread:.2f} t/s")
        if spread > abs(n-b)/2:
            print("  WARNING: order/sampling spread is large relative to the effect.")
            print("           Treat the gain as indicative, not established.")
    print("\n  Wall clock is deliberately not reported as a speedup: the arms")
    print("  generate different text and different token counts under sampling.")
(o/"summary.json").write_text(json.dumps(res,indent=2))
print(f"\n  wrote {o}/summary.json")
PY
log "DONE -> $OUT"
