#!/usr/bin/env bash
# Queue 6: answer the two questions queue 5 left open, at the length that matters.
#
# WHY. Queue 5 screened every arm at ISL 2048 and gated promotion on DECODE.
# Both choices were wrong for the arm that survived:
#
#   - S2 is a PREFILL kernel (--linear-attn-prefill-backend flashinfer, proved
#     live by `extend=FlashInferGDNKernel`). At ISL 2048 prefill is 0.3 s of
#     work; at 261K it is 36 s, and it is the entire left panel of the chart.
#     Screening it at 2K and gating it on decode meant it could never reach the
#     length where it might pay. Its 2K result is not evidence about 262K.
#
#   - The decode gate itself was inside the noise: two effectively identical
#     queue-5 cells differed 16.6% on prefill and 2.7% on decode at n=1.
#
# AND the control has a bigger problem (B-31). A byte-identical re-run of the
# published full-window arm gave decode 154.03 against the published 126.91,
# +21.4%, while TTFT and prefill reproduced within 2.5%. Decode is NEXTN
# speculative decoding, so it depends on acceptance rate on the generated text,
# and fn_maxctx.conf sets CONTEXT_REQUESTS=1 - both figures are ONE sample of a
# stochastic quantity. Until that spread is characterised, no full-window decode
# claim is publishable, for his flags or for ours.
#
# SO THIS QUEUE DOES TWO THINGS, BOTH AT THE FULL WINDOW:
#
#   1. CONTEXT_REQUESTS=3 instead of 1, so each arm reports a spread rather than
#      a point. Three boots of the control, to separate within-boot generation
#      variance from boot-to-boot variance. That settles B-31 or shows it is
#      wider still.
#   2. The same treatment for S2, so the prefill kernel is finally measured
#      where prefill is the dominant cost.
#
# NOT INCLUDED, deliberately:
#   S1 (full FlashInfer GDN)  hangs on this build - SGLang's own 600s warmup
#                             timeout. A hang does not become a number by
#                             running it at a longer context.
#   S3/S4/S5                  inside noise at 2K and no mechanism that would
#                             make them length-dependent. Re-screen them only if
#                             the control's spread turns out to be tight.
#
# READING IT. Compare PREFILL and TTFT between control and S2 - that is the
# question. Decode is recorded but is the noisy axis; treat the control's own
# three-boot spread as the yardstick for whether any decode difference means
# anything at all.
#
# ~55 min: 6 arms, each 1 warmup + 3 measured 261K requests plus boot and settle.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"

REPO="$PWD"; LOG="$REPO/results/newconfig_queue6.log"
G="--execute --plan-id FLASHNEXT-R1"
export THINKING=off
export CONTEXT_REQUESTS=3     # sweepable, so it overrides fn_maxctx.conf's 1

say(){ echo "[nq6 $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say "START $n"; "$@" >>"$LOG" 2>&1; local rc=$?; say "END   $n exit=$rc"; return $rc; }
clean(){ for c in q38n flashnext freetoken q38n-mtp q38n-lazy; do docker rm -f "$c" >/dev/null 2>&1; done; sleep 20; }
dcgm(){ curl -fsS --max-time 3 localhost:9401/metrics 2>/dev/null | grep -q '^DCGM_FI_DEV_SM_CLOCK{' \
  || DCGM_NAME=dcgm-bench DCGM_PROF_PORT=9401 DCGM_COUNTERS=bench/dcgm_metrics.csv \
     DCGM_READY_FIELD=DCGM_FI_DEV_SM_CLOCK ./benchmark/dcgm_exporter.sh up >>"$LOG" 2>&1; }

dcgm; clean
say "CONTEXT_REQUESTS=$CONTEXT_REQUESTS at ISL 261504, 3 boots per arm"

# Interleaved A-B-A-B-A-B, not three of one then three of the other: if the box
# drifts over the hour, blocking the arms would put that drift entirely into the
# difference. Auto_Bench.md 8 asks for the control in both directions.
for rep in 1 2 3; do
  say "=== repetition $rep of 3 ==="
  run "C${rep}_control" ./bench/context.sh $G sglang fn_maxctx || say "WARN control rep$rep failed"
  clean
  run "C${rep}_prefill_fi" env SGLANG_LINEAR_ATTN_PREFILL_BACKEND=flashinfer \
      ./bench/context.sh $G sglang fn_maxctx || say "WARN prefill_fi rep$rep failed"
  clean
done

say "=== summary ==="
python3 bench/maxctx_spread.py 2>&1 | tee -a "$LOG"
say "=== DONE ==="
