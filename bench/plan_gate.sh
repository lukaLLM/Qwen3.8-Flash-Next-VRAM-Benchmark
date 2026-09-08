#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# auto_run/Auto_Bench.md 1 - "Plan before execute" - for the ported AIPerf
# controllers.
#
#   A controller with no arguments prints its full plan (arms, order, expected
#   duration, evidence paths) and exits. Running requires an explicit --execute
#   plus a matching --plan-id. Nothing runs by accident.
#
# The controllers came from Inference_Engines/bench/, where this gate does not
# exist: run.sh, context.sh, discover.sh, validate.sh and specdecode.sh all
# begin work on the first invocation. That is fine in a repo where the operator
# types the command; it is not fine here, where a benchmark can hold the GPU
# for hours and this box has been OOM-crashed twice.
#
# One helper rather than six copies, so the gate cannot drift between scripts.
#
# Usage, immediately after `set -euo pipefail` and BEFORE any argument is read:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/plan_gate.sh"
#     plan_gate FLASHNEXT-R1 "$@"
#     set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}
#
# plan_gate consumes --execute/--plan-id/-h and leaves everything else in
# GATE_ARGV, so the controller's own positional arguments still work.
# -----------------------------------------------------------------------------

# The lock is DELIBERATELY the same file this repo's own controllers take
# (benchmark/correction_run_v2.sh:318). The two suites drive the same GPU, so
# they must serialise against each other, not merely against themselves.
BENCH_LOCK="${BENCH_LOCK:-/tmp/queue_rest.lock}"

plan_gate() {
  local required="${1:?plan_gate <PLAN-ID> \"\$@\"}"; shift
  local caller="${BASH_SOURCE[1]:-$0}"
  local execute=0 plan_id=""
  GATE_ARGV=()

  while [ $# -gt 0 ]; do
    case "$1" in
      --execute) execute=1; shift ;;
      --plan-id) plan_id="${2:-}"; shift 2 ;;
      -h|--help) execute=0; break ;;
      *)         GATE_ARGV+=("$1"); shift ;;
    esac
  done

  if [ "$execute" != 1 ]; then
    # The script's own header comment IS its plan. Printing anything else would
    # let the two drift apart.
    awk 'NR>1 && /^[^#]/{exit} NR>1{sub(/^#[[:space:]]?/,""); print}' "$caller"
    echo
    echo "Nothing was run."
    echo "To run:  $(basename "$caller") --execute --plan-id $required ${GATE_ARGV[*]:-<args>}"
    exit 0
  fi

  if [ "$plan_id" != "$required" ]; then
    echo "refusing: --plan-id must be $required (got '${plan_id:-<none>}')" >&2
    exit 64
  fi
}

# Serialise against every other controller on this box. Not FIFO - Auto_Bench 1
# is explicit that when stage order matters physically, the stages belong in one
# script rather than racing waiters.
# NEVER `rm -f "$BENCH_LOCK"`. flock is released automatically when the holding
# process exits, including on kill, so a stale lock file does not exist. Deleting
# the file while another process holds it gives the next process a FRESH INODE and
# its own lock - both then believe they hold the benchmark lock, and two engines
# run at once. That happened on 2026-09-03 and cost a contaminated SGLang run
# (artifacts/_quarantine/). If the lock looks stuck, something IS still running:
# find it, do not delete the file.
bench_lock() {
  exec 9>"$BENCH_LOCK"
  if ! flock -n 9; then
    echo "  benchmark lock is HELD - another run is in progress:" >&2
    fuser -v "$BENCH_LOCK" 2>&1 | sed 's/^/    /' >&2 || true
    echo "  waiting for it to finish (this is the 'one engine at a time' rule) ..." >&2
    flock 9
  fi
  echo "  lock acquired ($BENCH_LOCK)"
}
