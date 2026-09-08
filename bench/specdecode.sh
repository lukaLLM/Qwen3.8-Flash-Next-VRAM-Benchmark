#!/usr/bin/env bash
# ============================================================================
# Speculative-decoding track.
#
#   ./bench/specdecode.sh <vllm|sglang> <spec_mtp|spec_dflash|spec_dspark> [--baseline-only]
#
# Runs, per corpus, over the c=1..4 ladder:
#   1. a NO-SPECULATION baseline arm
#   2. a GREEDY CORRECTNESS GATE  <- hard precondition, see below
#   3. the speculative arm
# and reports speedup only against the matching baseline.
#
# FAIRNESS RULE (this replaces "same heads across methods"):
#   ACROSS ENGINES  - hold algorithm, target checkpoint+precision, draft
#                     checkpoint, effective depth, corpus+prompts, sampling,
#                     ISL/OSL, admission cap and concurrency constant.
#   ACROSS METHODS  - run each at its own recommended configuration.
# Comparing a method at its own maximum measures the cap, not the drafter; but
# forcing every method to one width is a DIFFERENT (optional) experiment, not
# the fairness rule. Raw CLI integers need not match when the engines count
# bonus tokens differently - effective behaviour must.
#
# THE GREEDY GATE IS NOT OPTIONAL. "Output looks coherent" is not a gate. For a
# lossless speculative implementation, greedy decoding must reproduce the target
# path EXACTLY. This matters on a GDN hybrid because recurrent state drift
# accumulates silently and looks like a small quality difference rather than a
# bug. A configuration that fails the gate does NOT get its throughput
# published as valid accelerated inference.
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

BASELINE_ONLY=0
ARGS=()
for a in "$@"; do
  case "$a" in --baseline-only) BASELINE_ONLY=1 ;; *) ARGS+=("$a") ;; esac
done

bench_init "${ARGS[0]:?usage: specdecode.sh <vllm|sglang> <spec_*> [--baseline-only]}" \
           "${ARGS[1]:?usage: specdecode.sh <vllm|sglang> <spec_*> [--baseline-only]}"

[[ -n "${SPEC_METHOD:-}" ]] || die "${WORKLOAD}.conf is not a speculative workload (no SPEC_METHOD)"

# Enforced HERE, not in compose. The compose file must not use ${VAR:?...} for
# these: compose interpolates EVERY service on any `up`, so a required-variable
# marker on a spec service makes `docker compose up vllm` fail on an unrelated
# service. Asking for it in the script keeps the failure scoped to the run that
# actually needs the value.
if [[ "$ENGINE" == "vllm" ]]; then
  [[ -n "${VLLM_SPEC_CONFIG:-}" ]] || die "${WORKLOAD}.conf must set VLLM_SPEC_CONFIG"
else
  [[ -n "${SGLANG_SPEC_ALGO:-}" ]] || die "${WORKLOAD}.conf must set SGLANG_SPEC_ALGO"
fi

OUT="$REPO/artifacts/spec_${SPEC_METHOD}_${ENGINE}_${STAMP}"
read -r -a LADDER <<<"${SPEC_LADDER//,/ }"
read -r -a CORPORA <<<"${SPEC_CORPORA//,/ }"

# External draft checkpoints are gated. Downloading and benchmarking an
# unverified community checkpoint - without knowing which target precision it
# was trained against - is how a precision mismatch gets published as a
# speculative-decoding result.
if [[ -n "${SPEC_DRAFT_MODEL:-}" && "${SPEC_DRAFT_VERIFIED:-0}" != "1" ]]; then
  die "${WORKLOAD}.conf names an EXTERNAL draft checkpoint (${SPEC_DRAFT_MODEL}) with
SPEC_DRAFT_VERIFIED=0. Before setting it to 1, confirm: the repo+revision exist;
the model card's TARGET PRECISION matches ${MODEL}; the licence; n_predict; and
that both engines' recipes work against the shipped images. Record the hashes."
fi

log "spec track: $ENGINE / $SPEC_METHOD"
echo "  ladder   : ${LADDER[*]}   admission slots: ${MAX_RUNNING_REQUESTS}"
echo "  corpora  : ${CORPORA[*]}"
echo "  drafter  : $([[ "${SPEC_FIXED_K:-1}" == "1" ]] \
      && echo 'fixed-k (acceptance load-invariant)' \
      || echo 'ADAPTIVE (acceptance is concurrency-dependent - report per c, never pooled)')"

mkdir -p "$OUT"
thermal_start "spec_${SPEC_METHOD}_${ENGINE}"
thermal_watchdog_start "$OUT"

# ---------------------------------------------------------------------------
# Boot helpers. The spec compose services are separate (compose cannot append to
# a command list); the parity check below reads flags back OFF THE RUNNING
# SERVER, which catches drift between the two service definitions where it
# actually matters.
# ---------------------------------------------------------------------------
boot_baseline() {
  settle_gpu
  ( cd "$COMPOSE_DIR" && docker compose down >/dev/null 2>&1 || true )
  log "booting $ENGINE (NO speculation - baseline)"
  ( cd "$COMPOSE_DIR" && \
      MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS" PREFILL_BUDGET="$PREFILL_BUDGET" \
      ENGINE_CTX="$ENGINE_CTX" CUDA_GRAPH_MAX_BS="$CUDA_GRAPH_MAX_BS" \
      docker compose up -d --wait "$ENGINE" )
  preflight
}

boot_spec() {
  settle_gpu
  ( cd "$COMPOSE_DIR" && docker compose down >/dev/null 2>&1 || true )
  log "booting $ENGINE (${SPEC_METHOD})"
  local svc="${ENGINE}_spec"
  # Each knob is its own compose entry. Compose interpolates but does NOT
  # word-split, so a multi-token "extra args" string would arrive as one argv
  # element and argparse would reject it. vLLM's whole payload is a single JSON
  # token, which is why only SGLang needs the split-out variables.
  ( cd "$COMPOSE_DIR" && \
      MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS" PREFILL_BUDGET="$PREFILL_BUDGET" \
      ENGINE_CTX="$ENGINE_CTX" CUDA_GRAPH_MAX_BS="$CUDA_GRAPH_MAX_BS" \
      KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}" MODEL="$MODEL" MODEL_REVISION="$MODEL_REVISION" \
      VLLM_SPEC_CONFIG="${VLLM_SPEC_CONFIG:-}" SGLANG_SPEC_ALGO="${SGLANG_SPEC_ALGO:-}" \
      SGLANG_SPEC_STEPS="${SGLANG_SPEC_STEPS:-3}" \
      SGLANG_SPEC_DRAFT_TOKENS="${SGLANG_SPEC_DRAFT_TOKENS:-4}" \
      SGLANG_SPEC_DRAFT_ARG="${SGLANG_SPEC_DRAFT_ARG:---speculative-eagle-topk=1}" \
      SGLANG_REPLAYSSM="${SGLANG_REPLAYSSM:---enable-linear-replayssm-spec}" \
      docker compose up -d --wait "$svc" )
  preflight
  assert_parity
}

# assert_parity — read the boot args back off the RUNNING container and confirm
# the spec arm booted with the SAME parity flags as the baseline. A spec arm
# that quietly got a different admission cap or context length would produce a
# speedup number that is really a configuration difference.
assert_parity() {
  local args; args=$(docker inspect --format '{{join .Args " "}}' "$CONTAINER" 2>/dev/null || echo "")
  local bad=()
  grep -q -- "$MAX_RUNNING_REQUESTS" <<<"$args" || bad+=("admission cap ${MAX_RUNNING_REQUESTS}")
  grep -q -- "$ENGINE_CTX" <<<"$args"           || bad+=("context ${ENGINE_CTX}")
  grep -q -- "$PREFILL_BUDGET" <<<"$args"       || bad+=("prefill budget ${PREFILL_BUDGET}")
  grep -q -- "${KV_CACHE_DTYPE:-bfloat16}" <<<"$args" || bad+=("kv dtype")
  (( ${#bad[@]} == 0 )) || die "spec arm parity mismatch - missing: ${bad[*]}
booted args: $args"
  echo "  parity vs baseline OK (read back off the running server)"
}

# NOT the repo root: it carries a credential file, and pydantic-settings would
# auto-load it from the CWD and print rejected values into a traceback that lands
# in a published evidence directory. See aiperf_wd() in lib.sh.
cd "$(aiperf_wd)"

spec_common() {   # $1 = corpus
  local corpus="$1"
  # Engine flags from the shared composer, minus --prompt-corpus: this script
  # chooses its own corpus/dataset below. min_tokens is added only where the
  # engine has the field.
  engine_aiperf_args
  ENGINE_ARGS_NOCORPUS=(); local _skip=0
  for _x in ${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"}; do
    if [[ "$_skip" == 1 ]]; then _skip=0; continue; fi
    if [[ "$_x" == "--prompt-corpus" ]]; then _skip=1; continue; fi
    ENGINE_ARGS_NOCORPUS+=("$_x")
  done
  [[ "${ENGINE_MIN_TOKENS:-1}" == "1" ]] && ENGINE_ARGS_NOCORPUS+=(--extra-inputs "min_tokens:$OSL")
  local -a a=(
    --model "$MODEL" --url "$BASE"
    --endpoint-type chat --streaming
    --isl "$ISL" --isl-stddev 0 --osl "$OSL" --osl-stddev 0
    ${ENGINE_ARGS_NOCORPUS[@]+"${ENGINE_ARGS_NOCORPUS[@]}"}
    --gpu-telemetry "$DCGM_URL" "$REPO/bench/dcgm_metrics.csv"
    --slice-duration 5 --goodput "$GOODPUT"
    --random-seed "$SEED" --ui simple
  )
  # gsm8k etc. are DATASETS, not corpora - only sonnet and coding are corpora
  # (enums.py:250-257). Wiring a dataset name into --prompt-corpus would fail.
  case "$corpus" in
    coding|sonnet) a+=(--prompt-corpus "$corpus") ;;
    *)             a+=(--public-dataset "$corpus") ;;
  esac
  printf '%s\n' "${a[@]}"
}

run_arm() {  # $1=arm(baseline|spec) $2=corpus $3=concurrency
  local arm="$1" corpus="$2" c="$3"
  local cell="$OUT/${arm}_${corpus}_c${c}"
  mapfile -t ca < <(spec_common "$corpus")
  gpu_cool
  log "$arm / $corpus / c=$c"
  timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile "${ca[@]}" "${SAMPLING_ARGS[@]}" \
    --concurrency "$c" --request-count "${SPEC_REQUESTS:-100}" \
    --warmup-request-count "${WARMUP:-5}" \
    --output-artifact-dir "$cell" 2>&1 | tee "$OUT/${arm}_${corpus}_c${c}.log" >/dev/null \
    || { warn "$arm/$corpus/c=$c FAILED"; echo "${arm}_${corpus}_c${c}" >> "$OUT/.failed"; return 1; }
}

# ---------------------------------------------------------------------------
# Greedy correctness gate. Runs the SAME prompts at temperature 0 against the
# baseline and the spec arm, then compares outputs.
#
# Comparison basis, in order of strength: generated TOKEN IDS where the server
# exposes them; otherwise EXACT TEXT equality, with retokenisation under the
# pinned tokenizer as a secondary check. Never claim token-id equality from a
# text-only endpoint.
# ---------------------------------------------------------------------------
greedy_capture() {  # $1=tag
  local tag="$1"
  greedy_args
  mapfile -t ca < <(spec_common "${CORPORA[0]}")
  # --export-outputs-json is MANDATORY: without it aiperf writes only metadata and
  # metrics, the gate has no text to compare, and it returns INCONCLUSIVE
  # (logs.md #38). This is the hardest precondition in the track; it must never
  # silently degrade into "no mismatch found".
  timeout --kill-after=60 "$AIPERF_TIMEOUT" "$AIPERF" profile "${ca[@]}" "${SAMPLING_ARGS[@]}" \
    --export-outputs-json \
    --concurrency 1 --request-count "${GREEDY_REQUESTS:-8}" --warmup-request-count 1 \
    --output-artifact-dir "$OUT/greedy_${tag}" 2>&1 | tee "$OUT/greedy_${tag}.log" >/dev/null
  sampling_args   # restore the pinned production sampling
}

# ===========================================================================
sampling_args
# NOTE: no record_provenance here. It reads engine version, image digest and KV
# report OFF THE RUNNING CONTAINER, so it can only run AFTER a boot. Calling it
# before boot_baseline died with "container not found" (logs.md #36). Each arm
# records its own provenance below, which is what we actually want anyway - the
# baseline and spec arms boot DIFFERENT services and must be documented apart.

# --- Baseline arm -----------------------------------------------------------
boot_baseline
record_provenance "$OUT/baseline_provenance"
for corpus in "${CORPORA[@]}"; do
  for c in "${LADDER[@]}"; do run_arm baseline "$corpus" "$c" || true; done
done
[[ "$BASELINE_ONLY" == "1" ]] || greedy_capture baseline

if [[ "$BASELINE_ONLY" == "1" ]]; then
  engine_down; thermal_verdict || true; thermal_stop
  log "baseline-only complete -> ${OUT#"$REPO"/}"; exit 0
fi

# --- Speculative arm --------------------------------------------------------
boot_spec
record_provenance "$OUT/spec_provenance"
greedy_capture spec

# Gate BEFORE any throughput cell is trusted.
log "greedy correctness gate"
GATE_RESULT=$(python3 - "$OUT/greedy_baseline" "$OUT/greedy_spec" <<'PY'
import json, pathlib, sys
def texts(d):
    """Generated text lives in outputs.json (--export-outputs-json), NOT in
    profile_export.jsonl, which carries only metadata and metrics."""
    p = pathlib.Path(d)
    outs = []
    for f in sorted(p.glob("**/outputs.json")):
        try: data = json.loads(f.read_text())
        except Exception: continue
        recs = data if isinstance(data, list) else (
            data.get("records") or data.get("outputs") or data.get("data") or [])
        if isinstance(recs, dict): recs = list(recs.values())
        for r in recs:
            if not isinstance(r, dict): continue
            md = r.get("metadata") or {}
            if md.get("benchmark_phase") == "warmup": continue   # compare profiled only
            t = r.get("response_text") or r.get("text") or r.get("output_text")
            if t: outs.append(t)
    return outs
a, b = texts(sys.argv[1]), texts(sys.argv[2])
if not a or not b:
    # NOT a pass. A gate that cannot read its inputs must fail loudly, or a
    # missing result reads identically to "no mismatch found".
    print(f"INCONCLUSIVE: no generated text (baseline={len(a)} spec={len(b)}). "
          "Is --export-outputs-json set on BOTH captures?")
    sys.exit(2)
n = min(len(a), len(b))
mismatch = [i for i in range(n) if a[i] != b[i]]
if mismatch:
    i = mismatch[0]
    print(f"FAIL: {len(mismatch)}/{n} greedy outputs differ; first at index {i}")
    print(f"  baseline: {a[i][:160]!r}")
    print(f"  spec    : {b[i][:160]!r}")
    sys.exit(1)
print(f"PASS: {n}/{n} greedy outputs identical (exact text)")
PY
) && GATE_OK=1 || GATE_OK=0
echo "$GATE_RESULT" | sed 's/^/  /'
echo "$GATE_RESULT" > "$OUT/greedy_gate.txt"

if [[ "$GATE_OK" != "1" ]]; then
  warn "GREEDY GATE DID NOT PASS. Speculative cells will still run for diagnosis,
but this configuration's throughput MUST NOT be published as valid accelerated
inference until the divergence is explained. This is a correctness finding."
fi

for corpus in "${CORPORA[@]}"; do
  for c in "${LADDER[@]}"; do run_arm spec "$corpus" "$c" || true; done
done

engine_down
thermal_watchdog_stop
thermal_verdict || warn "thermal/host verdict flagged this run"
thermal_stop

# ---------------------------------------------------------------------------
# Results: speedup ONLY against the matching baseline (same engine, corpus and
# concurrency), plus acceptance broken out per concurrency for adaptive drafters.
# ---------------------------------------------------------------------------
OUT="$OUT" REPO="$REPO" ENGINE="$ENGINE" METHOD="$SPEC_METHOD" STAMP="$STAMP" \
LADDER="${LADDER[*]}" CORPORA="${CORPORA[*]}" FIXED_K="${SPEC_FIXED_K:-1}" \
MODEL="$MODEL" MODEL_REVISION="$MODEL_REVISION" DRAFT="${SPEC_DRAFT_MODEL:-in-checkpoint}" \
GATE="$GATE_RESULT" SAMPLING="$SAMPLING_ARGS_STR" \
python3 - "$OUT/spec_results.json" <<'PY'
import json, os, pathlib, sys
e = os.environ; out = pathlib.Path(e["OUT"])
def load(arm, corpus, c):
    d = out / f"{arm}_{corpus}_c{c}"
    f = next((p for p in d.glob("**/profile_export_aiperf.json") if "phases" not in p.parts), None)
    return json.loads(f.read_text()) if f else None
def g(d, k, s="avg"):
    return ((d or {}).get(k) or {}).get(s)
rows = []
for corpus in e["CORPORA"].split():
    for c in e["LADDER"].split():
        b, s = load("baseline", corpus, c), load("spec", corpus, c)
        bt, st = g(b, "output_token_throughput"), g(s, "output_token_throughput")
        rows.append({
            "corpus": corpus, "concurrency": int(c),
            "baseline_output_tok_s": bt, "spec_output_tok_s": st,
            "speedup_vs_matching_no_spec_baseline": (st / bt) if (bt and st) else None,
            "baseline_ttft_ms": g(b, "time_to_first_token"), "spec_ttft_ms": g(s, "time_to_first_token"),
            "baseline_itl_ms": g(b, "inter_token_latency"),  "spec_itl_ms": g(s, "inter_token_latency"),
            "spec_errors": g(s, "error_request_count"),
        })
json.dump({
  "engine": e["ENGINE"], "method": e["METHOD"], "stamp": e["STAMP"],
  "target_checkpoint": e["MODEL"], "target_revision": e["MODEL_REVISION"],
  "target_precision": "FP8", "draft_checkpoint": e["DRAFT"],
  "sampling": e["SAMPLING"],
  "correctness_gate_result": e["GATE"],
  "throughput_publishable": e["GATE"].startswith("PASS"),
  "drafter_kind": "fixed-k" if e["FIXED_K"] == "1" else "adaptive",
  "acceptance_reporting": (
      "load-invariant; may be pooled across the ladder" if e["FIXED_K"] == "1"
      else "CONCURRENCY-DEPENDENT: report per concurrency, never pooled (metrics-reference.md:731-739)"),
  "cells": rows,
  "notes": [
    "speedup is against the SAME engine, corpus and concurrency baseline only",
    "acceptance is content-dependent, hence per-corpus reporting",
    "server-side acceptance counters come from the engine's Prometheus endpoint "
    "(vllm:spec_decode_* / sglang:spec_accept_*); post-process with "
    "`aiperf speed-bench-report --metric accept_length --format both`",
  ],
}, open(sys.argv[1], "w"), indent=2)
print(open(sys.argv[1]).read()[:1200])
PY

log "done -> ${OUT#"$REPO"/}"
