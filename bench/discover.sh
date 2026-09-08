#!/usr/bin/env bash
# ============================================================================
# Capacity discovery. Finds, then CONFIRMS, an engine's ceiling for a workload.
#
#   ./bench/discover.sh <vllm|sglang> <workload> [--recipe sla-concurrency|goodput]
#   ./bench/discover.sh vllm w1_chat
#   ./bench/discover.sh vllm w1_chat --recipe goodput
#
# Two DIFFERENT numbers, deliberately named apart (see plan.md "Terminology"):
#
#   sla-concurrency : highest CONFIRMED concurrency where every aggregate SLA
#                     filter passes. TTFT/TPOT are evaluated at p95.
#   goodput         : max requests/s under a per-request SLO attainment
#                     fraction. NOT the same question, and the two are allowed
#                     to disagree - that disagreement is itself a result.
#
# Sources the workload conf but IGNORES its CONCURRENCY/DURATION/WARMUP: those
# describe a fixed measurement sweep, and this script is searching instead.
#
# WHY TWO PHASES. The bracketing search runs at --request-count 100, which is
# cheap (an explicit --request-count overrides the recipe's 1000-request tier;
# _shared_warmup.py:81-99 "explicit user values always win"). But 100 requests
# is far too thin to DEFINE a boundary: at c=100 that is ~1 request per virtual
# slot and p95 rests on the worst ~5 observations. Every downstream percentage
# cell inherits this number, so the candidate is re-tested properly before it
# is written. See plan.md §B.
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

RECIPE="sla-concurrency"
ARGS=()
for a in "$@"; do
  case "$a" in
    --recipe=*) RECIPE="${a#*=}" ;;
    --recipe)   RECIPE="__next__" ;;
    *) if [[ "$RECIPE" == "__next__" ]]; then RECIPE="$a"; else ARGS+=("$a"); fi ;;
  esac
done
case "$RECIPE" in sla-concurrency|goodput) ;; *) die "--recipe must be sla-concurrency|goodput" ;; esac

bench_init "${ARGS[0]:?usage: discover.sh <vllm|sglang> <workload> [--recipe R]}" \
           "${ARGS[1]:?usage: discover.sh <vllm|sglang> <workload> [--recipe R]}"

OUT="$REPO/artifacts/discover_${WORKLOAD}_${ENGINE}_${RECIPE}_${STAMP}"
DISCOVERED_DIR="$BENCH_DIR/workloads/.discovered"
DISCOVERED="$DISCOVERED_DIR/${WORKLOAD}_${ENGINE}.json"

# Discovery knobs. PROBE_REQUESTS is for bracketing only; the confirmation
# floor scales with concurrency because a flat 300 at c=128 covers only ~2.3
# concurrency waves per repeat.
# MEASURED 2026-08-24, and the reason this is not 100: at c=1 this workload runs
# ~10.4s/request, so 100 requests is a 17-MINUTE probe, and with
# --num-profile-runs 2 a single low-concurrency search point cost ~35 minutes.
# An unbounded [1,1000] search at that price is hours of GPU time before the
# first useful answer.
#
# A fixed request count is the wrong unit for a concurrency search in BOTH
# directions: it is ruinously slow at c=1, and at c=64 the same 100 requests are
# only ~1.5 per slot, far too thin to judge p95. Bracketing is therefore cheap
# and approximate BY DESIGN - all the statistical weight lives in the
# confirmation phase below, which scales its request count with concurrency.
: "${PROBE_REQUESTS:=32}"
# Bracketing runs ONE profile run per point. Replication belongs in
# confirmation, where it is spent on the boundary that actually matters rather
# than on every rung of the climb.
: "${PROBE_PROFILE_RUNS:=1}"
# Bound the search. The default space is [1,1000]; nothing in this project is
# near 1000 (w1_chat's own note puts SGLang's ceiling around 43), and an
# oversized upper bound just adds probes.
: "${CONCURRENCY_MIN:=1}"
: "${CONCURRENCY_MAX:=128}"
: "${CONFIRM_REQUESTS_MIN:=300}"
: "${CONFIRM_REQUESTS_PER_SLOT:=5}"
: "${CONFIRM_REPEATS:=3}"
: "${SEARCH_STYLE:=monotonic}"
: "${SLO_ATTAINMENT:=0.95}"
: "${PROMPT_CORPUS:=coding}"

# ---------------------------------------------------------------------------
# Map the conf's GOODPUT string onto the recipe's SLA flags. FAIL LOUDLY on an
# unmapped metric: silently dropping one would search against a weaker SLA than
# the workload declares and hand back an inflated ceiling.
#
# Note these are RELATED BUT NOT IDENTICAL semantics, and the writeup must not
# conflate them: --ttft-sla-ms/--tpot-sla-ms are aggregate p95 thresholds, while
# --goodput is a per-request AND-of-both fraction.
# ---------------------------------------------------------------------------
TTFT_SLA=""; TPOT_SLA=""
for pair in $GOODPUT; do
  metric="${pair%%:*}"; value="${pair##*:}"
  case "$metric" in
    time_to_first_token) TTFT_SLA="$value" ;;
    inter_token_latency) TPOT_SLA="$value" ;;
    *) die "unmapped GOODPUT metric '$metric' in ${WORKLOAD}.conf - add a mapping, do not ignore it" ;;
  esac
done
[[ -n "$TTFT_SLA" && -n "$TPOT_SLA" ]] \
  || die "GOODPUT must define both time_to_first_token and inter_token_latency (got '$GOODPUT')"

# E2E budget. REQUIRED by the goodput recipe - it raises ValueError naming any
# missing flag (_max_goodput_under_slo.py:86-110), so it must be chosen, not
# inferred. This is a DISCLOSED ENGINEERING BUDGET, not a statistically derived
# p99: TTFT budget + (OSL-1) inter-token steps + 10% margin. The first token is
# the TTFT term, hence OSL-1. Recompute if the conf's SLOs or OSL change.
#
# Computed with exact rationals, NOT binary floats. `1.10 * 26550` evaluates to
# 29205.000000000004 in float, so math.ceil() returns 29206 and a documented
# constant silently drifts by 1 between the doc and the run. Fraction gives
# 29205 exactly and reproducibly.
E2E_SLA=$(python3 -c "
from fractions import Fraction as F
import math
print(math.ceil(F(11,10) * (F('$TTFT_SLA') + (F('$OSL') - 1) * F('$TPOT_SLA'))))")

log "discovery: $ENGINE / $WORKLOAD / recipe=$RECIPE"
echo "  SLA: TTFT p95 < ${TTFT_SLA}ms, TPOT p95 < ${TPOT_SLA}ms"
echo "  E2E budget (goodput recipe only): ${E2E_SLA}ms  [1.10 x ($TTFT_SLA + ($OSL-1) x $TPOT_SLA)]"
echo "  corpus: $PROMPT_CORPUS   probe: ${PROBE_REQUESTS} req x ${PROBE_PROFILE_RUNS} run, search c in [${CONCURRENCY_MIN},${CONCURRENCY_MAX}]"

mkdir -p "$OUT"
thermal_start "discover_${WORKLOAD}_${ENGINE}"
thermal_watchdog_start "$OUT"
settle_gpu
engine_up
preflight
sampling_args
workload_args
record_provenance "$OUT"

# Flags every phase shares. --reset-kv-cache isolates each probe so a previous
# probe's cache cannot flatter the next one's TTFT.
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
  --random-seed "$SEED"
  --ui simple
)

# NOT the repo root: it carries a credential file, and pydantic-settings would
# auto-load it from the CWD and print rejected values into a traceback that lands
# in a published evidence directory. See aiperf_wd() in lib.sh.
cd "$(aiperf_wd)"

# ---------------------------------------------------------------------------
# Phase 1 - bracketing search
# ---------------------------------------------------------------------------
probe_dir="$OUT/probe"
log "phase 1: bracketing search (cheap, $PROBE_REQUESTS requests/probe)"
if [[ "$RECIPE" == "sla-concurrency" ]]; then
  timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile "${common_args[@]}" \
    --search-recipe max-concurrency-under-sla \
    --search-style "$SEARCH_STYLE" \
    --ttft-sla-ms "$TTFT_SLA" --tpot-sla-ms "$TPOT_SLA" \
    --concurrency-min "$CONCURRENCY_MIN" --concurrency-max "$CONCURRENCY_MAX" \
    --num-profile-runs "$PROBE_PROFILE_RUNS" --request-count "$PROBE_REQUESTS" \
    --output-artifact-dir "$probe_dir" 2>&1 | tee "$OUT/probe.log"
else
  # All THREE SLA flags are mandatory for this recipe.
  timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile "${common_args[@]}" \
    --search-recipe max-goodput-under-slo \
    --ttft-sla-ms "$TTFT_SLA" --tpot-sla-ms "$TPOT_SLA" --e2e-sla-ms "$E2E_SLA" \
    --slo-attainment-fraction "$SLO_ATTAINMENT" \
    --concurrency-min "$CONCURRENCY_MIN" --concurrency-max "$CONCURRENCY_MAX" \
    --num-profile-runs "$PROBE_PROFILE_RUNS" --request-count "$PROBE_REQUESTS" \
    --output-artifact-dir "$probe_dir" 2>&1 | tee "$OUT/probe.log"
fi

# ---------------------------------------------------------------------------
# Read the candidate. boundary_summary.feasible_max is "the highest swept-dim
# value seen among feasible iterations" (search_history.py:41-45) - which is an
# OBSERVATION, not a proof of optimality. Only populated for 1D search spaces.
# ---------------------------------------------------------------------------
read_candidate() {
  python3 - "$probe_dir" "$RECIPE" <<'PY'
import json, pathlib, sys
d, recipe = pathlib.Path(sys.argv[1]), sys.argv[2]
hist = next(iter(d.glob("**/search_history.json")), None)
if hist is None:
    print("ERR no search_history.json produced"); sys.exit(1)
h = json.loads(hist.read_text())
if recipe == "sla-concurrency":
    bs = h.get("boundary_summary")
    if not bs or not bs.get("feasible_max"):
        print("ERR boundary_summary/feasible_max is null - no feasible point found"); sys.exit(1)
    fm = bs["feasible_max"]
    val = fm.get("value", fm.get(bs.get("swept_dim_path", ""), fm))
    while isinstance(val, dict):
        val = next(iter(val.values()))
    print(int(val))
else:
    bt = h.get("best_trials") or []
    if not bt:
        print("ERR best_trials empty"); sys.exit(1)
    if not bt[0].get("feasible_count", 0):
        print("ERR no feasible trial met the SLO attainment fraction"); sys.exit(1)
    vv = bt[0].get("variation_values") or {}
    val = next((v for k, v in vv.items() if "concurrency" in k.lower()), None)
    if val is None:
        print(f"ERR no concurrency in variation_values: {vv}"); sys.exit(1)
    print(int(val))
PY
}

CANDIDATE="$(read_candidate)" || { thermal_stop; die "$CANDIDATE"; }
log "phase 1 candidate: concurrency $CANDIDATE"

# Monotonic feasibility is ASSUMED by boundary_summary. If the run says
# otherwise, feasible_max is only "largest observed passing value".
MONOTONIC=$(python3 -c "
import json,pathlib,sys
h=next(iter(pathlib.Path('$probe_dir').glob('**/search_history.json')))
print(json.loads(h.read_text()).get('monotonicity_check', 'unreported'))" 2>/dev/null || echo unreported)
[[ "$MONOTONIC" == "False" || "$MONOTONIC" == "false" ]] && \
  warn "monotonicity_check=false - treat the ceiling as 'largest observed passing', not proven optimum"

# ---------------------------------------------------------------------------
# Phase 2 - boundary confirmation. Re-test C-D, C, C+D properly. A point is
# CONFIRMED only when every repeat passes; the largest confirmed passing point
# becomes the ceiling. Writes nothing at all if none confirms.
# ---------------------------------------------------------------------------
DELTA="${CONFIRM_DELTA:-1}"
confirm_one() {
  local c="$1" reqs cell
  (( c < 1 )) && return 1
  reqs=$(( CONFIRM_REQUESTS_MIN > c * CONFIRM_REQUESTS_PER_SLOT \
           ? CONFIRM_REQUESTS_MIN : c * CONFIRM_REQUESTS_PER_SLOT ))
  cell="$OUT/confirm_c${c}"
  log "confirming c=$c  (${reqs} requests x ${CONFIRM_REPEATS} repeats)"
  gpu_cool
  # Same seed across repeats: this measures RUNTIME repeatability with input
  # content held constant. Varying the seed here would conflate input noise
  # with runtime noise (cli-options.md:1434-1436). Seed sensitivity is a
  # separate study in validate.sh.
  timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile "${common_args[@]}" \
    --concurrency "$c" --request-count "$reqs" \
    --num-profile-runs "$CONFIRM_REPEATS" \
    --output-artifact-dir "$cell" 2>&1 | tee "$OUT/confirm_c${c}.log" >/dev/null

  python3 - "$cell" "$TTFT_SLA" "$TPOT_SLA" <<'PY'
import json, pathlib, sys
cell, ttft_sla, tpot_sla = pathlib.Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
runs = sorted(cell.glob("**/profile_export_aiperf.json"))
if not runs:
    print("    no runs produced"); sys.exit(1)
bad = []
for r in runs:
    d = json.loads(r.read_text())
    ttft = ((d.get("time_to_first_token") or {}).get("p95"))
    tpot = ((d.get("inter_token_latency") or {}).get("p95"))
    err  = (d.get("error_request_count") or {}).get("avg", 0) or 0
    tag = r.parent.name
    if err: bad.append(f"{tag}: {err} errors")
    if ttft is not None and ttft > ttft_sla: bad.append(f"{tag}: TTFT p95 {ttft:.0f} > {ttft_sla:.0f}")
    if tpot is not None and tpot > tpot_sla: bad.append(f"{tag}: TPOT p95 {tpot:.1f} > {tpot_sla:.1f}")
    print(f"    {tag}: TTFT p95={ttft} TPOT p95={tpot} errors={err}")
if bad:
    print("    FAIL: " + "; ".join(bad)); sys.exit(1)
print(f"    PASS ({len(runs)} repeats)")
PY
}

# Thermal-guarded confirmation: a cell the watchdog killed is rested and re-run
# whole, never partially credited.
confirm_guarded() {
  local c="$1" attempt=0
  while :; do
    thermal_clear   # flag is evidence about THIS attempt only (see validate.sh)
    if confirm_one "$c"; then return 0; fi
    if thermal_aborted && (( attempt < THERMAL_RETRIES )); then
      attempt=$(( attempt + 1 ))
      warn "c=$c killed by thermal watchdog - resting, then retry $attempt/$THERMAL_RETRIES"
      cat "$THERMAL_FLAG" >> "$OUT/thermal_events.log" 2>/dev/null || true
      thermal_clear; gpu_rest; rm -rf "${OUT:?}/confirm_c${c}"
      continue
    fi
    return 1
  done
}

CONFIRMED=""
if confirm_guarded "$CANDIDATE"; then
  CONFIRMED="$CANDIDATE"
  # C passed - probe upward until the first confirmed failure.
  probe_up=$(( CANDIDATE + DELTA ))
  while confirm_guarded "$probe_up"; do
    CONFIRMED="$probe_up"; probe_up=$(( probe_up + DELTA ))
  done
else
  # C failed - step down until something confirms.
  probe_down=$(( CANDIDATE - DELTA ))
  while (( probe_down >= 1 )); do
    if confirm_guarded "$probe_down"; then CONFIRMED="$probe_down"; break; fi
    probe_down=$(( probe_down - DELTA ))
  done
fi

engine_down
thermal_watchdog_stop
thermal_verdict || warn "thermal/host verdict flagged this run - see the trace before trusting it"
thermal_stop

[[ -n "$CONFIRMED" ]] || die "no confirmed passing concurrency - writing NOTHING. \
The probe suggested $CANDIDATE but it did not survive confirmation."

# ---------------------------------------------------------------------------
# Write the pointer validate.sh reads. Records the probe candidate alongside the
# confirmed value so the two can never be silently conflated.
# ---------------------------------------------------------------------------
mkdir -p "$DISCOVERED_DIR"
RECIPE="$RECIPE" WORKLOAD="$WORKLOAD" ENGINE="$ENGINE" STAMP="$STAMP" \
CONFIRMED="$CONFIRMED" CANDIDATE="$CANDIDATE" DELTA="$DELTA" MONOTONIC="$MONOTONIC" \
TTFT_SLA="$TTFT_SLA" TPOT_SLA="$TPOT_SLA" E2E_SLA="$E2E_SLA" SLO_ATTAINMENT="$SLO_ATTAINMENT" \
PROBE_REQUESTS="$PROBE_REQUESTS" CONFIRM_REPEATS="$CONFIRM_REPEATS" \
CONFIRM_REQUESTS_MIN="$CONFIRM_REQUESTS_MIN" CONFIRM_REQUESTS_PER_SLOT="$CONFIRM_REQUESTS_PER_SLOT" \
SEARCH_STYLE="$SEARCH_STYLE" PROMPT_CORPUS="$PROMPT_CORPUS" SEED="$SEED" \
MODEL="$MODEL" MODEL_REVISION="$MODEL_REVISION" OUT="$OUT" REPO="$REPO" \
SAMPLING="$SAMPLING_ARGS_STR" \
python3 - "$DISCOVERED" <<'PY'
import json, os, sys
e = os.environ
prov = {}
p = os.path.join(e["OUT"], "provenance.json")
if os.path.exists(p): prov = json.load(open(p))
json.dump({
  "workload": e["WORKLOAD"], "engine": e["ENGINE"], "recipe": e["RECIPE"],
  "stamp": e["STAMP"], "artifact_dir": e["OUT"].replace(e["REPO"] + "/", ""),
  "confirmed_sla_concurrency_ceiling": int(e["CONFIRMED"]),
  "probe_candidate": int(e["CANDIDATE"]),
  "search_step": int(e["DELTA"]),
  "monotonicity_check": e["MONOTONIC"],
  "sla_definitions": {
    # p95 for TTFT/TPOT, p99 if an E2E filter is ever added to this recipe
    # (_max_concurrency_under_sla.py:149-175). Do not restate these as "the SLA"
    # without the percentile - they are not the same as the per-request --goodput
    # fraction, which is an AND of per-request thresholds.
    "ttft_sla_ms": float(e["TTFT_SLA"]), "ttft_stat": "p95",
    "tpot_sla_ms": float(e["TPOT_SLA"]), "tpot_stat": "p95",
    "e2e_sla_ms": float(e["E2E_SLA"]), "e2e_stat": "p99",
    "e2e_basis": "disclosed engineering budget: 1.10 x (TTFT + (OSL-1) x TPOT), not a measured p99",
    "slo_attainment_fraction": float(e["SLO_ATTAINMENT"]),
  },
  "request_counts": {
    "probe": int(e["PROBE_REQUESTS"]),
    "confirm_formula": f"max({e['CONFIRM_REQUESTS_MIN']}, {e['CONFIRM_REQUESTS_PER_SLOT']} x concurrency)",
    "confirm_repeats": int(e["CONFIRM_REPEATS"]),
    "repeat_mode": "same-seed (runtime repeatability)",
  },
  "search_style": e["SEARCH_STYLE"], "prompt_corpus": e["PROMPT_CORPUS"],
  "seed": e["SEED"], "sampling": e["SAMPLING"],
  "model": e["MODEL"], "model_revision": e["MODEL_REVISION"],
  "engine_version": prov.get("engine_version"), "image_digest": prov.get("image_digest"),
  "source_commits": prov.get("source_commits"),
}, open(sys.argv[1], "w"), indent=2)
print(open(sys.argv[1]).read())
PY

log "done"
echo "  probe candidate : $CANDIDATE"
echo "  CONFIRMED ceiling: $CONFIRMED"
echo "  -> ${DISCOVERED#"$REPO"/}"
