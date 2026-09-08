#!/usr/bin/env bash
# ============================================================================
# Validation sweep. Runs BOTH comparison views against a confirmed ceiling.
#
#   ./bench/validate.sh <vllm|sglang> <workload>
#
# TWO VIEWS, because they answer different questions and neither substitutes
# for the other (plan.md "Two capacity views"):
#
#   View A - same absolute load: 1,4,8,16,32,64,128, IDENTICAL for both engines.
#     This is the primary head-to-head. A general "engine X is faster" claim
#     must rest on this view. An SLA violation here is a RECORDED RESULT, not a
#     stop condition - we keep climbing the ladder. Without View A, "SGLang is
#     better at 50% load" can silently mean c=20 vs c=60 and is not a
#     comparison at all.
#
#   View B - relative saturation: 25/50/75/100% of THIS engine's own confirmed
#     ceiling. Answers how gracefully each engine degrades toward its own limit.
#
# A concurrency in both views is run ONCE and referenced by both.
#
# Stops early only on hard failure/OOM. Remaining cells are then reported as
# unavailable and are no longer paired comparisons.
# ============================================================================
set -euo pipefail

# --- auto_run/Auto_Bench.md 1: nothing runs by accident --------------------
# Prints this script's header (its plan) and exits unless given
# --execute --plan-id FLASHNEXT-R1. Also serialises against every other
# controller on this box, including benchmark/correction_run_v2.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/plan_gate.sh"
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}


source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

bench_init "${1:?usage: validate.sh <vllm|sglang> <workload>}" \
           "${2:?usage: validate.sh <vllm|sglang> <workload>}"

DISCOVERED="$BENCH_DIR/workloads/.discovered/${WORKLOAD}_${ENGINE}.json"
[[ -f "$DISCOVERED" ]] || die "no confirmed ceiling: $DISCOVERED
Run: ./bench/discover.sh $ENGINE $WORKLOAD"

OUT="$REPO/artifacts/validate_${WORKLOAD}_${ENGINE}_${STAMP}"

: "${ABSOLUTE_LADDER:=1,4,8,16,32,64,128}"
: "${PRIMARY_REPEATS:=3}"
: "${ESCALATED_REPEATS:=5}"
: "${CV_THRESHOLD:=0.10}"
: "${PROMPT_CORPUS:=coding}"
: "${SEED_SENSITIVITY:=1}"   # run the secondary study; 0 to skip

# Written as a statement, not a conditional expression. `raise X if c else print(y)`
# parses as `raise (X if c else print(y))`, so the happy path raises None ->
# TypeError, which under set -e kills the script AFTER printing the right number.
CEILING=$(python3 - "$DISCOVERED" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
v = d.get("confirmed_sla_concurrency_ceiling")
if v is None:
    sys.exit(f"ceiling missing from {sys.argv[1]}")
print(int(v))
PY
)

# View B: percentage cells. Dedup below the point where percentages collapse
# onto the same integer - that collapse is itself a reported finding, not
# something to hide, so every label is kept in the manifest.
read -r -a VIEW_B <<<"$(python3 -c "
c=$CEILING
seen={}
for f in (0.25,0.50,0.75,1.00):
    v=max(1,round(f*c)); seen.setdefault(v,[]).append(f'{int(f*100)}%')
print(' '.join(str(v) for v in sorted(seen)))")"
read -r -a VIEW_A <<<"${ABSOLUTE_LADDER//,/ }"

# Union, run-once. sort -n -u is what makes a shared concurrency a single cell.
read -r -a CELLS <<<"$(printf '%s\n' "${VIEW_A[@]}" "${VIEW_B[@]}" | sort -n -u | paste -sd' ' -)"

log "validation: $ENGINE / $WORKLOAD"
echo "  confirmed ceiling : $CEILING"
echo "  View A (absolute) : ${VIEW_A[*]}"
echo "  View B (relative) : ${VIEW_B[*]}"
echo "  cells to run      : ${CELLS[*]}"

mkdir -p "$OUT"
thermal_start "validate_${WORKLOAD}_${ENGINE}"
thermal_watchdog_start "$OUT"
settle_gpu
engine_up
preflight
sampling_args
workload_args
record_provenance "$OUT"

common_args=(
  --model "$MODEL" --url "$BASE"
  --endpoint-type chat --streaming
  "${WORKLOAD_ARGS[@]}"
  "${SAMPLING_ARGS[@]}"
  # reset hook removed: it is engine-conditional and already in WORKLOAD_ARGS via
  # engine_aiperf_args. Hardcoding it passed an EMPTY --reset-kv-cache-path for
  # engines with no flush endpoint (llama.cpp, FreeToken), and aiperf rejects that
  # outright - it killed every rung of the first context ladder.
  --gpu-telemetry "$DCGM_URL" "$REPO/bench/dcgm_metrics.csv"
  --slice-duration 5
  --goodput "$GOODPUT"
  --benchmark-duration "$DURATION" --warmup-request-count "$WARMUP"
  --ui simple
)

# NOT the repo root: it carries a credential file, and pydantic-settings would
# auto-load it from the CWD and print rejected values into a traceback that lands
# in a published evidence directory. See aiperf_wd() in lib.sh.
cd "$(aiperf_wd)"

# ---------------------------------------------------------------------------
# Primary repeats: SAME seed across trials. This measures RUNTIME repeatability
# with input content held constant. --vary-seed-per-trial "captures end-to-end
# variance at the cost of conflating input noise with runtime noise in the
# resulting confidence statistics" (cli-options.md:1434-1436), so a CV computed
# that way could NOT be described as engine instability. Seed sensitivity is a
# separate study below, reported separately and never merged into this CV.
#
# NOTE: --parameter-sweep-same-seed is deliberately NOT passed. aiperf documents
# it as a silent no-op when no sweep is configured (troubleshooting/sweeps.md:213),
# and run.sh has been passing it on single-cell invocations where it does
# nothing. Cross-cell content correlation needs a real sweep or a persisted
# dataset; claiming it from this flag would be false. See FINDINGS F5.
# ---------------------------------------------------------------------------
run_cell() {
  local c="$1" repeats="$2" tag="$3"
  local cell="$OUT/${tag}_c${c}"
  log "cell c=$c  (${repeats} repeats, $tag)"
  gpu_cool
  if ! timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile "${common_args[@]}" \
      --concurrency "$c" \
      --num-profile-runs "$repeats" \
      --random-seed "$SEED" \
      --output-artifact-dir "$cell" 2>&1 | tee "$OUT/${tag}_c${c}.log" >/dev/null; then
    warn "cell c=$c FAILED to run - recorded as unavailable"
    echo "$c" >> "$OUT/.unavailable"
    return 1
  fi
  return 0
}

# cv_check — returns 1 when any SLO-relevant metric's CV exceeds the threshold.
cv_check() {
  python3 - "$1" "$CV_THRESHOLD" <<'PY'
import json, pathlib, statistics, sys
cell, thr = pathlib.Path(sys.argv[1]), float(sys.argv[2])
runs = sorted(cell.glob("**/profile_export_aiperf.json"))
if len(runs) < 2:
    print("    CV: <2 runs, skipped"); sys.exit(0)
series = {}
for r in runs:
    d = json.loads(r.read_text())
    for m in ("time_to_first_token", "inter_token_latency", "request_latency",
              "output_token_throughput", "request_throughput"):
        v = (d.get(m) or {}).get("avg")
        if v is not None: series.setdefault(m, []).append(float(v))
hot = []
for m, vals in sorted(series.items()):
    if len(vals) < 2 or statistics.fmean(vals) == 0: continue
    cv = statistics.stdev(vals) / statistics.fmean(vals)
    flag = "  <-- OVER" if cv > thr else ""
    print(f"    CV {m:26s} {cv:6.3f}{flag}")
    if cv > thr: hot.append(m)
sys.exit(1 if hot else 0)
PY
}

# run_cell_guarded — retries a cell the WATCHDOG killed, after a real rest.
# A thermally-killed cell is invalid, never partial-credit and never resumed:
# it is re-run whole. A cell that failed for any OTHER reason is not retried,
# because retrying a genuine error just burns GPU time twice.
run_cell_guarded() {
  local c="$1" repeats="$2" tag="$3" attempt=0
  while :; do
    # Clear BEFORE the attempt: the flag is evidence about THIS attempt only.
    # Left stale, an earlier firing would misattribute a later genuine failure
    # as thermal and retry it pointlessly.
    thermal_clear
    if run_cell "$c" "$repeats" "$tag"; then return 0; fi
    if thermal_aborted && (( attempt < THERMAL_RETRIES )); then
      attempt=$(( attempt + 1 ))
      warn "c=$c killed by thermal watchdog - resting, then retry $attempt/$THERMAL_RETRIES"
      cat "$THERMAL_FLAG" >> "$OUT/thermal_events.log" 2>/dev/null || true
      thermal_clear
      gpu_rest
      rm -rf "${OUT:?}/${tag}_c${c}"   # discard the partial cell
      continue
    fi
    return 1
  done
}

FAILED_HARD=0
for c in "${CELLS[@]}"; do
  if ! run_cell_guarded "$c" "$PRIMARY_REPEATS" primary; then FAILED_HARD=1; break; fi
  if ! cv_check "$OUT/primary_c${c}"; then
    # AIPerf cannot top up an existing run, so escalation is a FULL second
    # invocation. Both are kept; the 5-run result is the final estimate. They
    # are never presented as one accumulated 8-run set.
    warn "c=$c exceeded CV ${CV_THRESHOLD} - escalating to ${ESCALATED_REPEATS} repeats"
    run_cell_guarded "$c" "$ESCALATED_REPEATS" escalated || FAILED_HARD=1
    echo "$c" >> "$OUT/.escalated"
  fi
done

# ---------------------------------------------------------------------------
# Secondary study: seed sensitivity = WORKLOAD ROBUSTNESS, not runtime noise.
# Kept in its own directory so it cannot be accidentally pooled with the
# primary CV.
# ---------------------------------------------------------------------------
if [[ "$SEED_SENSITIVITY" == "1" && "$FAILED_HARD" == "0" ]]; then
  probe_c="${VIEW_B[$(( ${#VIEW_B[@]} / 2 ))]}"
  log "secondary study: seed sensitivity at c=$probe_c (workload robustness, reported separately)"
  gpu_cool
  timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile "${common_args[@]}" \
    --concurrency "$probe_c" \
    --num-profile-runs "$PRIMARY_REPEATS" \
    --random-seed "$SEED" --vary-seed-per-trial \
    --output-artifact-dir "$OUT/seed_sensitivity_c${probe_c}" \
    2>&1 | tee "$OUT/seed_sensitivity.log" >/dev/null || warn "seed-sensitivity study failed (non-fatal)"
  cv_check "$OUT/seed_sensitivity_c${probe_c}" || true
fi

engine_down
thermal_watchdog_stop
thermal_verdict || warn "thermal/host verdict flagged this run - see the trace before trusting it"
thermal_stop

post_gate "$OUT" "$OSL" || warn "post-run gate failed - see above; this run is not a result"

# ---------------------------------------------------------------------------
# Manifest. Records which view(s) each cell belongs to so a reader can never
# mistake a relative-saturation cell for an absolute-load one.
# ---------------------------------------------------------------------------
CEILING="$CEILING" OUT="$OUT" REPO="$REPO" WORKLOAD="$WORKLOAD" ENGINE="$ENGINE" \
STAMP="$STAMP" VIEW_A="${VIEW_A[*]}" VIEW_B="${VIEW_B[*]}" CELLS="${CELLS[*]}" \
PRIMARY_REPEATS="$PRIMARY_REPEATS" CV_THRESHOLD="$CV_THRESHOLD" \
SAMPLING="$SAMPLING_ARGS_STR" DISCOVERED="$DISCOVERED" \
python3 - "$OUT/validation_manifest.json" <<'PY'
import json, os, sys
e = os.environ
a = set(e["VIEW_A"].split()); b = set(e["VIEW_B"].split())
ceiling = int(e["CEILING"])
out = e["OUT"]
esc = set(open(f"{out}/.escalated").read().split()) if os.path.exists(f"{out}/.escalated") else set()
una = set(open(f"{out}/.unavailable").read().split()) if os.path.exists(f"{out}/.unavailable") else set()
cells = []
for c in e["CELLS"].split():
    views = [v for v, s in (("absolute_load", a), ("relative_saturation", b)) if c in s]
    row = {"concurrency": int(c), "views": views,
           "escalated_to_5": c in esc, "unavailable": c in una}
    if c in b:
        row["pct_of_ceiling"] = round(100 * int(c) / ceiling, 1)
    cells.append(row)
json.dump({
  "workload": e["WORKLOAD"], "engine": e["ENGINE"], "stamp": e["STAMP"],
  "confirmed_sla_concurrency_ceiling": ceiling,
  "discovery_source": e["DISCOVERED"].replace(e["REPO"] + "/", ""),
  "cells": cells,
  "repeat_policy": {
    "primary": {"repeats": int(e["PRIMARY_REPEATS"]), "seed": "same across trials",
                "measures": "runtime repeatability", "cv_threshold": float(e["CV_THRESHOLD"]),
                "escalation": "full second invocation at 5 repeats; NOT a top-up of the 3-run set"},
    "secondary": {"flag": "--vary-seed-per-trial", "measures": "workload robustness",
                  "note": "reported separately; never pooled into the primary CV"},
  },
  "sampling": e["SAMPLING"],
  "interpretation": {
    "absolute_load": "same workload+concurrency on both engines; the only view that supports a general engine-winner claim",
    "relative_saturation": "percentage of THIS engine's own ceiling; supports degradation comparisons only",
    "sla_violation": "a recorded result, not a stop condition",
    "unavailable": "hard failure/OOM only; these cells are not paired comparisons",
  },
}, open(sys.argv[1], "w"), indent=2)
print(open(sys.argv[1]).read())
PY

log "done -> ${OUT#"$REPO"/}"
