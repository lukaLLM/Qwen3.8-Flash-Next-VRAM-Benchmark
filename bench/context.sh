#!/usr/bin/env bash
# ============================================================================
# Single-stream context ladder + engine capability probe.
#
#   ./bench/context.sh <vllm|sglang> [w3_context]
#
# TWO SEPARATE THINGS, deliberately not merged (plan.md §H):
#
#   1. LADDER - a FIXED absolute native-context ISL sweep at c=1, identical on
#      both engines. This is the comparison. Same ISL on both sides or it is not
#      a comparison at all.
#
#   2. CAPABILITY PROBE - one real request sized near the engine's OWN reported
#      pool. This is NOT part of the comparison; it is a per-engine capability
#      finding, and it exists because the boot log's token count is an
#      ALLOCATOR SIZE, not proof that a request that big is admitted and
#      completed. F8 asked "can SGLang serve one full-context request?" - this
#      answers it by trying, not by reading a number.
#
# Runs with MAX_RUNNING_REQUESTS=1 from the conf: see the conf for why that is
# a fairness requirement rather than an optimisation.
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

bench_init "${1:?usage: context.sh <vllm|sglang> [workload]}" "${2:-w3_context}"
[[ "${SWEEP_DIM:-}" == "isl" ]] || die "${WORKLOAD}.conf must set SWEEP_DIM=isl for context.sh"

OUT="$REPO/artifacts/context_${WORKLOAD}_${ENGINE}_${STAMP}"
read -r -a LADDER <<<"${ISL_LADDER//,/ }"

log "context ladder: $ENGINE / $WORKLOAD"
echo "  ISL rungs : ${LADDER[*]}"
echo "  OSL       : $OSL   concurrency: 1   admission slots: ${MAX_RUNNING_REQUESTS}"

mkdir -p "$OUT"
thermal_start "context_${WORKLOAD}_${ENGINE}"
# Was MISSING until 2026-08-25 (logs.md #25): this script logged thermals but had
# no ACTIVE protection, unlike discover.sh/validate.sh. Long-context prefill is
# the most power-dense work in the suite, so this is the last place it should
# have been omitted.
thermal_watchdog_start "$OUT"
settle_gpu
engine_up
preflight
sampling_args
record_provenance "$OUT"

# ---------------------------------------------------------------------------
# Read the engine's OWN reported pool size. UPPER-BOUND CANDIDATE ONLY.
# ---------------------------------------------------------------------------
pool_tokens() {
  if [[ "$ENGINE" == "vllm" ]]; then
    docker logs "$CONTAINER" 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' \
      | tail -1 | grep -oE '[0-9,]+' | tr -d ','
  else
    docker logs "$CONTAINER" 2>&1 | grep -oE 'max_total_num_tokens=[0-9]+' \
      | tail -1 | grep -oE '[0-9]+'
  fi
}
POOL="$(pool_tokens || true)"
[[ -n "$POOL" ]] || warn "could not read the engine's reported pool size from its log"
echo "  engine-reported pool: ${POOL:-unknown} tokens  (UPPER-BOUND CANDIDATE, not a proven limit)"

# NOT the repo root: it carries a credential file, and pydantic-settings would
# auto-load it from the CWD and print rejected values into a traceback that lands
# in a published evidence directory. See aiperf_wd() in lib.sh.
cd "$(aiperf_wd)"
# Engine-specific flags come from engine_aiperf_args in lib.sh, NOT from a list
# built here. This block used to duplicate them and got every one wrong:
#
#   --reset-kv-cache-path "$RESET_PATH"   llama.cpp has no reset endpoint, so this
#                                         passed an EMPTY path and aiperf refused
#                                         the whole run:
#                                           "endpoint.reset_kv_cache.path must be
#                                            a relative path starting with '/', got ''"
#                                         Every rung failed, including 2048, and the
#                                         controller dutifully recorded seven
#                                         "capacity findings" that were nothing of
#                                         the kind.
#   --prompt-corpus coding                the corpus CORPUS-0 disqualified: 67.2%
#                                         repeated-8gram at 262K
#   --extra-inputs min_tokens:$OSL        llama.cpp has no min_tokens field
#   (no --tokenizer)                      would hit the HF resolution failure
#   (no --cache-bust)                     no cold-prefill guarantee
#
# One composer, one place. If an engine needs a different flag, it belongs in
# engine_aiperf_args, not in a controller.
engine_aiperf_args
base_args=(
  --model "$MODEL" --url "$BASE"
  --endpoint-type chat --streaming
  --osl "$OSL" --osl-stddev 0
  ${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"}
  "${SAMPLING_ARGS[@]}"
  --gpu-telemetry "$DCGM_URL" "$REPO/bench/dcgm_metrics.csv"
  --slice-duration 5
  --concurrency 1
  --random-seed "$SEED"
  --ui simple
)
# OSL-dependent, and only where the engine has the field (llama.cpp does not).
if [[ "${ENGINE_MIN_TOKENS:-1}" == "1" ]]; then
  base_args+=(--extra-inputs "min_tokens:$OSL")
fi

# ---------------------------------------------------------------------------
# The ladder. A rung that fails is a RESULT (a capacity finding, rule 7), not a
# reason to abandon the sweep - later rungs still run so the failure boundary is
# located rather than merely encountered.
# ---------------------------------------------------------------------------
for isl in "${LADDER[@]}"; do
  cell="$OUT/isl_${isl}"
  log "rung ISL=$isl"
  gpu_cool
  attempt=0
  while :; do
    thermal_clear
    if timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile "${base_args[@]}" \
        --isl "$isl" --isl-stddev 0 \
        --request-count "${CONTEXT_REQUESTS:-10}" \
        --warmup-request-count "${WARMUP:-1}" \
        --output-artifact-dir "$cell" 2>&1 | tee "$OUT/isl_${isl}.log" >/dev/null; then
      echo "  served"; break
    fi
    # A thermally-killed rung is VOID, not a capacity finding. Conflating the two
    # would publish "engine X cannot serve Nk context" about a rung we ourselves
    # killed - which is exactly what happened on 2026-08-25 (logs.md #29).
    if thermal_aborted && (( attempt < THERMAL_RETRIES )); then
      attempt=$(( attempt + 1 ))
      warn "ISL=$isl killed by thermal watchdog - VOID, not a capacity finding; \
resting then retry $attempt/$THERMAL_RETRIES"
      cat "$THERMAL_FLAG" >> "$OUT/thermal_events.log" 2>/dev/null || true
      thermal_clear; gpu_rest; rm -rf "$cell"
      continue
    fi
    if thermal_aborted; then
      warn "ISL=$isl VOID - thermally killed and retries exhausted (NOT a capacity finding)"
      echo "$isl" >> "$OUT/.void_rungs"
    else
      warn "ISL=$isl FAILED - recorded as a capacity finding, continuing the ladder"
      echo "$isl" >> "$OUT/.failed_rungs"
    fi
    break
  done
done

# ---------------------------------------------------------------------------
# Capability probe. Deliberately AFTER the ladder and reported apart from it:
# its ISL differs per engine by construction, so it is not a paired comparison.
# ---------------------------------------------------------------------------
PROBE_ISL=""; PROBE_RESULT="not_run"
if [[ "${CAPABILITY_PROBE:-0}" == "1" && -n "$POOL" ]]; then
  # MUST cap at the per-request context limit. The pool is TOTAL KV across all
  # concurrent requests; a SINGLE request can never exceed --max-model-len /
  # --context-length. Without the cap this asked vLLM for a 657,403-token request
  # against a 262,144 window and got HTTP 400 on every one (logs.md #26).
  #
  # TEMPLATE_MARGIN: the requested ISL is pre-chat-template. The ladder's top rung
  # (262,016 = ctx - OSL) was rejected 400 for exactly this reason, so the probe
  # leaves room for the template and special tokens rather than repeating it.
  PROBE_ISL=$(python3 -c "
import math
pool_cap = int(math.floor($POOL * ${CAPABILITY_FRACTION:-0.9})) - $OSL
ctx_cap  = ${ENGINE_CTX:-262144} - $OSL - ${TEMPLATE_MARGIN:-512}
print(max(1, min(pool_cap, ctx_cap)))")
  # warmup-request-count must be >0: aiperf rejects 0 with a pydantic
  # "Input should be greater than 0" validation error, which surfaced as
  # "probe rejected_or_failed" and was mistaken for a capacity limit (logs.md #32).
  log "capability probe: ONE request at ISL=$PROBE_ISL (${CAPABILITY_FRACTION:-0.9} x reported pool ${POOL})"
  gpu_cool
  thermal_clear
  if timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile "${base_args[@]}" \
      --isl "$PROBE_ISL" --isl-stddev 0 \
      --request-count 1 --warmup-request-count 1 \
      --output-artifact-dir "$OUT/capability_probe" 2>&1 | tee "$OUT/capability_probe.log" >/dev/null; then
    PROBE_RESULT="served"
  else
    # Same void/failed distinction as the ladder (logs.md #32/#33). A thermally
    # killed probe is NOT evidence the server refused the request - reporting it
    # as "rejected" would be a fabricated capability claim.
    if thermal_aborted; then
      PROBE_RESULT="void_thermal"
      cat "$THERMAL_FLAG" >> "$OUT/thermal_events.log" 2>/dev/null || true
    else
      PROBE_RESULT="rejected_by_server"
    fi
  fi
  echo "  probe result: $PROBE_RESULT"
fi

engine_down
thermal_watchdog_stop
thermal_verdict || warn "thermal/host verdict flagged this run - see the trace before trusting it"
thermal_stop

# ---------------------------------------------------------------------------
# Results. Records REQUESTED vs ACTUAL input tokens per rung, because the
# requested ISL is pre-chat-template: the last rung is the native window minus
# OSL and can cross the window once the template and special tokens land.
# ---------------------------------------------------------------------------
OUT="$OUT" REPO="$REPO" ENGINE="$ENGINE" WORKLOAD="$WORKLOAD" STAMP="$STAMP" \
POOL="${POOL:-}" PROBE_ISL="$PROBE_ISL" PROBE_RESULT="$PROBE_RESULT" \
ENGINE_CTX="${ENGINE_CTX:-262144}" OSL="$OSL" LADDER="${LADDER[*]}" \
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}" \
python3 - "$OUT/context_results.json" <<'PY'
import json, os, pathlib, sys
e = os.environ
out = pathlib.Path(e["OUT"])
failed = set((out / ".failed_rungs").read_text().split()) if (out / ".failed_rungs").exists() else set()
# VOID != FAILED. A rung we killed thermally is not a capacity result. Without
# this the writer marked a voided rung served=true with null metrics, which would
# publish "engine served N tokens" for work we ourselves terminated (logs.md #32).
void = set((out / ".void_rungs").read_text().split()) if (out / ".void_rungs").exists() else set()
ctx = int(e["ENGINE_CTX"]); osl = int(e["OSL"])
rungs = []
for isl in e["LADDER"].split():
    cell = out / f"isl_{isl}"
    row = {"requested_isl": int(isl),
           "served": (isl not in failed) and (isl not in void),
           "void_thermal": isl in void,
           "rejected_by_server": isl in failed}
    ex = next(iter(cell.glob("**/profile_export_aiperf.json")), None)
    if ex:
        d = json.loads(ex.read_text())
        actual = (d.get("input_sequence_length") or {}).get("avg")
        row["actual_input_tokens"] = actual
        row["ttft_ms"] = (d.get("time_to_first_token") or {}).get("avg")
        row["ttft_p95_ms"] = (d.get("time_to_first_token") or {}).get("p95")
        row["errors"] = (d.get("error_request_count") or {}).get("avg", 0)
        if actual is not None and actual + osl > ctx:
            row["WARNING"] = (f"actual prompt {actual:.0f} + OSL {osl} exceeds native window {ctx} "
                              "after chat template - this rung is out of window")
    rungs.append(row)
json.dump({
  "engine": e["ENGINE"], "workload": e["WORKLOAD"], "stamp": e["STAMP"],
  "admission_slots": int(e["MAX_RUNNING_REQUESTS"]),
  "native_context": ctx, "osl": osl,
  "ladder": rungs,
  "capability_probe": {
    "allocator_reported_candidate_limit": int(e["POOL"]) if e["POOL"] else None,
    "candidate_is_proof_of_capacity": False,
    "requested_isl": int(e["PROBE_ISL"]) if e["PROBE_ISL"] else None,
    "result": e["PROBE_RESULT"],
    "note": "per-engine capability finding; ISL differs per engine by construction "
            "so this is NOT a paired comparison and must be reported apart from the ladder",
  },
}, open(sys.argv[1], "w"), indent=2)
print(open(sys.argv[1]).read())
PY

log "done -> ${OUT#"$REPO"/}"
