#!/usr/bin/env bash
# ============================================================================
# Engine runner: one workload, one engine, fully recorded.
#
#   ./bench/run.sh --execute --plan-id FLASHNEXT-R1 <engine> <workload>
#   ./bench/run.sh <engine> <workload>        # prints this plan and exits
#
#   engines    llamacpp | sglang | freetoken   (vllm retained, unused here)
#   workloads  bench/workloads/*.conf          (fn_fast, fn_spec, _mocktest)
#
# Does the whole measured cycle in the order Auto_Bench.md 5 requires, so a run
# cannot silently skip a fairness step:
#
#   lock -> settle GPU -> boot engine -> preflight asserts -> record provenance
#     -> aiperf sweep -> runtime state -> post-run gate -> tear down
#
# EVERY run writes artifacts/<workload>_<engine>_<stamp>/ containing the aiperf
# exports, provenance.json (image digest, engine version read from the RUNNING
# container, model revision, every parity flag, driver, GPU clocks at start, and
# the full aiperf command line) and memory.json (restart count, OOM kills,
# cgroup peak, host swap). A number without provenance is not a measurement, it
# is an anecdote.
#
# ONE ENGINE AT A TIME. The three arms bind different host ports, so a stray
# second engine would NOT fail on the bind - it would quietly share VRAM and
# host RAM and corrupt both sets of numbers. bench_lock is what actually
# prevents that, and it is shared with this repo's own controllers.
#
# WORKING DIRECTORY MATTERS: aiperf runs from the repo root, which must hold no
# credential file. Both aiperf and its mock server use pydantic-settings, which
# auto-loads one from the working directory, rejects unknown keys such as an HF
# token, and prints the rejected value in the traceback. Compose and the
# credential file live in docker/ for that reason.
#
# This script was a 263-line copy of lib.sh carrying its own older settle_gpu,
# preflight, record_provenance and post_gate. Those four had drifted well behind
# lib.sh (34 vs 75 lines of preflight, and no thermal watchdog at all), and an
# engine table existing in two places is exactly how a parity flag goes stale.
# It is now a thin controller over lib.sh, like context.sh and validate.sh.
# ============================================================================
set -euo pipefail

# --- auto_run/Auto_Bench.md 1: nothing runs by accident --------------------
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/plan_gate.sh"
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

bench_init "${1:?usage: run.sh <engine> <workload>}" \
           "${2:?usage: run.sh <engine> <workload>}"

# The library's sampler/watchdog traps are intentionally generic and replace
# one another as they start. This controller owns a server too, so restore one
# composite cleanup path AFTER each helper that installs its own trap. A failed
# output gate must not strand 80+ GiB of VRAM or leave the next arm measuring a
# second engine.
cleanup_run() {
  # Evidence BEFORE teardown. engine_down deletes the container, and with it
  # State.OOMKilled, the exit code and the cgroup counters - i.e. exactly what a
  # capacity finding consists of. capture_failure is a no-op on a clean run
  # (memory.json already exists) and idempotent if the trap fires twice.
  [[ -f "${OUT:-}/memory.json" ]] || capture_failure "${OUT:-}" 2>/dev/null || true
  thermal_watchdog_stop 2>/dev/null || true
  thermal_stop 2>/dev/null || true
  engine_down 2>/dev/null || true
}
trap cleanup_run EXIT INT TERM

OUT="$REPO/artifacts/${WORKLOAD}_${ENGINE}_${STAMP}"
mkdir -p "$OUT"

log "run: $ENGINE / $WORKLOAD"
echo "  ISL/OSL     : $ISL / $OSL"
echo "  concurrency : $CONCURRENCY"
if [[ -n "${REQUEST_COUNT:-}" ]]; then
  echo "  requests    : ${REQUEST_COUNT}   warmup: ${WARMUP}"
else
  echo "  duration    : ${DURATION}s   warmup: ${WARMUP}"
fi
echo "  evidence    : $OUT"

bench_lock
thermal_start "${WORKLOAD}_${ENGINE}"
trap cleanup_run EXIT INT TERM
thermal_watchdog_start "$OUT"
trap cleanup_run EXIT INT TERM
settle_gpu
engine_up
preflight
# A requested cap that did not apply would produce an uncapped run that looks
# capped in its own evidence. Refuse rather than record that.
if [[ -n "${MEM_CAP_BYTES:-}" ]]; then
  verify_mem_cap || die "host-RAM cap was requested but not applied - see the warning above"
fi
sampling_args
workload_args          # also composes the per-engine flags via engine_aiperf_args
record_provenance "$OUT"

log "aiperf sweep -> $OUT"
# NOT the repo root: it carries a credential file, and pydantic-settings would
# auto-load it from the CWD and print rejected values into a traceback that lands
# in a published evidence directory. See aiperf_wd() in lib.sh.
cd "$(aiperf_wd)"
if [[ -n "${REQUEST_COUNT:-}" ]]; then
  LIMIT_ARGS=(--request-count "$REQUEST_COUNT")
else
  LIMIT_ARGS=(--benchmark-duration "$DURATION")
fi
OUTPUT_ARGS=()
[[ -n "${INPUT_FILE:-}" ]] && OUTPUT_ARGS=(--export-outputs-json)
timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile \
  --model "$MODEL" --url "$BASE" \
  --endpoint-type chat --streaming \
  ${WORKLOAD_ARGS[@]+"${WORKLOAD_ARGS[@]}"} \
  ${SAMPLING_ARGS[@]+"${SAMPLING_ARGS[@]}"} \
  --concurrency "$CONCURRENCY" \
  "${LIMIT_ARGS[@]}" --warmup-request-count "$WARMUP" \
  "${OUTPUT_ARGS[@]}" \
  --random-seed "$SEED" --parameter-sweep-same-seed \
  --gpu-telemetry "$DCGM_URL" "$REPO/bench/dcgm_metrics.csv" \
  --slice-duration 5 \
  --goodput "$GOODPUT" \
  --output-artifact-dir "$OUT" --ui simple \
  2>&1 | tee "$OUT/run.log"

# Capture the server log AFTER the requests: the boot-time capture cannot
# contain the run's errors (Auto_Bench.md 5).
if [[ -n "${CONTAINER:-}" ]]; then
  docker logs "$CONTAINER" > "$OUT/server.log.after" 2>&1 || true
fi

record_runtime_state "$OUT"
thermal_watchdog_stop
thermal_stop "$OUT"
post_gate "$OUT" "$OSL"
if [[ -n "${INPUT_FILE:-}" ]]; then
  log "real-code output gate"
  python3 "$BENCH_DIR/check_aiperf_code_outputs.py" "$OUT" \
    | tee "$OUT/output_check.log"
fi

log "tearing down"
engine_down
log "done -> $OUT"
echo "inspect with: uv run aiperf plot ${OUT#"$REPO"/} --dashboard"
