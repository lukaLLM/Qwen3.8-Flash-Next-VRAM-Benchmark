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
NAME=dcgm-prof
PORT="${DCGM_PROF_PORT:-9402}"
IMAGE=nvidia/dcgm-exporter:4.2.3-4.1.3-ubi9

case "${1:-up}" in
  up)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --rm \
      --gpus all --cap-add SYS_ADMIN --runtime nvidia \
      -p "${PORT}:9400" \
      -v "$PWD/benchmark/dcgm-counters.csv:/etc/dcgm-exporter/custom.csv:ro" \
      "$IMAGE" -f /etc/dcgm-exporter/custom.csv >/dev/null
    echo "  started $NAME on :$PORT - waiting for first scrape"
    for _ in $(seq 1 30); do
      sleep 2
      if curl -s --max-time 3 "http://localhost:$PORT/metrics" | grep -q DCGM_FI_PROF_PIPE_TENSOR_ACTIVE; then
        echo "  profiling metrics live"; exit 0
      fi
    done
    echo "  WARNING: exporter up but no PROF_ metrics after 60s." >&2
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
