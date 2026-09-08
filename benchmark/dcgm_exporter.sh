#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# A dcgm-exporter with PROFILING metrics enabled, on port 9402.
#
# There is already a dcgm-exporter on 9401 on this box, but it runs the default
# counter set, which has no PROF_* fields. This starts a second one with
# benchmark/dcgm-counters.csv so we get tensor-pipe and DRAM activity - the two
# numbers that say whether decode is compute bound or memory bound.
#
# Separate container and port on purpose: any pre-existing exporter on this box
# is left alone.
#
#   ./benchmark/dcgm_exporter.sh up
#   ./benchmark/dcgm_exporter.sh check
#   ./benchmark/dcgm_exporter.sh down
#
# Note: profiling metrics sample the GPU. Start it BEFORE an arm and leave it
# running for the whole arm, rather than starting it mid-measurement.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Two exporters are wanted on this box and they are NOT interchangeable:
#
#   :9402  dcgm-prof   benchmark/dcgm-counters.csv   PROF_* profiling counters,
#                      what FINDINGS.md 4 was measured with
#   :9401  dcgm-bench  bench/dcgm_metrics.csv        the DEV_* superset the ported
#                      AIPerf harness asserts on (bench/lib.sh preflight dies
#                      without SM_CLOCK, and post_gate fails every cell)
#
# Same script, parameterised, rather than a second near-identical copy.
# Defaults are unchanged so existing callers keep working.
NAME="${DCGM_NAME:-dcgm-prof}"
PORT="${DCGM_PROF_PORT:-9402}"
COUNTERS="${DCGM_COUNTERS:-benchmark/dcgm-counters.csv}"
# The field whose presence proves the exporter is actually serving THIS set. A
# row in a counter file is not proof of a column: an unknown field makes the
# exporter exit 1, but a known-yet-unsupported one exports nothing SILENTLY.
READY_FIELD="${DCGM_READY_FIELD:-DCGM_FI_PROF_PIPE_TENSOR_ACTIVE}"
IMAGE=nvidia/dcgm-exporter:4.2.3-4.1.3-ubi9

# --- STABILITY GUARD, added after the 2026-09-02 hard lockup --------------
# Ten profiling exporters started and torn down in nine minutes preceded a hard
# lockup of this machine (results/incidents/2026-09-02_gpu_lockup.md). Repeated
# acquisition of DCGM DCP profiling watches appears to destabilise the driver,
# and the first symptom is PROF columns flapping between present and absent -
# which reads as a measurement puzzle and invites more probing. It is not a
# puzzle; it is the warning.
#
# So: refuse to start again within COOLDOWN seconds of the last start. Start one
# exporter, leave it up for the whole arm, stop it once.
STAMP_FILE="${TMPDIR:-/tmp}/.dcgm_exporter_last_start"
COOLDOWN="${DCGM_START_COOLDOWN:-120}"


case "${1:-up}" in
  up)
    if [ -f "$STAMP_FILE" ]; then
      last=$(cat "$STAMP_FILE" 2>/dev/null || echo 0)
      age=$(( $(date +%s) - last ))
      if [ "$age" -lt "$COOLDOWN" ]; then
        echo "refusing: last exporter start was ${age}s ago, cooldown is ${COOLDOWN}s." >&2
        echo "  Rapid profiling-watch cycling hard-locked this box on 2026-09-02." >&2
        echo "  See results/incidents/2026-09-02_gpu_lockup.md. Override: DCGM_START_COOLDOWN=0" >&2
        exit 75
      fi
    fi
    date +%s > "$STAMP_FILE"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --rm \
      --gpus all --cap-add SYS_ADMIN --runtime nvidia \
      -p "${PORT}:9400" \
      -v "$PWD/$COUNTERS:/etc/dcgm-exporter/custom.csv:ro" \
      "$IMAGE" -f /etc/dcgm-exporter/custom.csv >/dev/null
    echo "  started $NAME on :$PORT - waiting for first scrape"
    for _ in $(seq 1 30); do
      sleep 2
      if curl -s --max-time 3 "http://localhost:$PORT/metrics" | grep -q "^${READY_FIELD}{"; then
        echo "  live: $READY_FIELD is a real column"; exit 0
      fi
    done
    echo "  WARNING: exporter up but $READY_FIELD absent after 60s." >&2
    echo "  Profiling needs a driver/GPU that supports DCP and no other client" >&2
    echo "  holding the profiling watches. Logs:" >&2
    docker logs --tail 15 "$NAME" 2>&1 | sed 's/^/    /' >&2
    exit 1 ;;
  down) docker rm -f "$NAME" >/dev/null 2>&1 && echo "  stopped $NAME" || echo "  not running" ;;
  check)
    curl -s --max-time 5 "http://localhost:$PORT/metrics" \
      | grep -E "^DCGM_FI_(PROF|DEV_(GPU_TEMP|FB_USED|POWER))" | sed 's/{[^}]*}/ /' \
      | awk '{printf "  %-36s %s\n",$1,$2}' ;;
  *) echo "usage: $0 {up|down|check}" >&2; exit 64 ;;
esac
