#!/usr/bin/env bash
# Does SGLang's decode CUDA graph change the OUTPUT when it is captured for a
# batch larger than one?
#
# Background. fn_fast measured graph_bs=8 at ~200 tok/s aggregate at c=2 against
# ~115 at c=1, so the batched graph is worth real throughput. Nothing has ever
# checked whether it returns the SAME TOKENS. The one prior "repeated output"
# observation came from ignore_eos runs and is NOT_APPLICABLE (BLOCKERS B-17).
#
# Method. Greedy decoding is deterministic, so output must not depend on how
# many requests happen to share a decode batch. For each graph size we capture
# the same prompts at concurrency 1 (batch 1) and at concurrency 4 (batched),
# then compare every capture against the graph_bs=1 / c=1 reference with
# bench/greedy_compare.py on a TOKEN-ID basis.
#
#   PASS  identical token ids everywhere -> batching is output-neutral
#   FAIL  any divergence -> record the index; do not promote graph_bs>1
#
# Boot-and-capture only: no aiperf, no throughput claim. bench/run.sh fn_fast
# measures speed once correctness is settled.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}
source bench/lib.sh

# REPO is set by bench_init, which runs later; this controller needs a path now.
REPO="${REPO:-$PWD}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO/results/gates/graphbatch_${STAMP}"
mkdir -p "$OUT"
GRAPH_SIZES=(${GRAPH_SIZES:-1 2 8})
CONCS=(${CONCS:-1 4})
SLOTS="${SLOTS:-4}"
CTX="${CTX_OVERRIDE:-16384}"
LIMIT="${LIMIT:-16}"
MAXTOK="${MAXTOK:-256}"

say(){ echo "[graphbatch $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/gate.log"; }
cleanup(){ thermal_watchdog_stop 2>/dev/null||true; thermal_stop 2>/dev/null||true; engine_down 2>/dev/null||true; }

bench_init sglang fn_smoke
trap cleanup EXIT INT TERM
bench_lock
thermal_start "graphbatch"
# Each thermal helper installs its OWN trap, replacing this controller's; re-arm
# after every one or a failure strands the engine's VRAM (BLOCKERS B-15).
trap cleanup EXIT INT TERM
thermal_watchdog_start "$OUT"
trap cleanup EXIT INT TERM

say "graph sizes: ${GRAPH_SIZES[*]}   concurrencies: ${CONCS[*]}   slots=$SLOTS ctx=$CTX"
say "prompts=$LIMIT max_tokens=$MAXTOK  (greedy, no ignore_eos)"

for bs in "${GRAPH_SIZES[@]}"; do
  say "=== booting sglang with cuda-graph-max-bs-decode=$bs ==="
  settle_gpu
  CUDA_GRAPH_MAX_BS="$bs" MAX_RUNNING_REQUESTS="$SLOTS" ENGINE_CTX="$CTX" \
    engine_up 2>&1 | tail -3 | tee -a "$OUT/gate.log"
  CUDA_GRAPH_MAX_BS="$bs" MAX_RUNNING_REQUESTS="$SLOTS" ENGINE_CTX="$CTX" \
    record_provenance "$OUT/prov_bs${bs}" >/dev/null 2>&1 || true
  # Prove the flag reached the server rather than trusting the env map.
  docker inspect --format '{{join .Args " "}}' "$CONTAINER" 2>/dev/null \
    | grep -oE '\--cuda-graph-max-bs-decode [0-9]+' | sed 's/^/    resolved: /' | tee -a "$OUT/gate.log"
  for c in "${CONCS[@]}"; do
    say "capture graph_bs=$bs concurrency=$c"
    uv run bench/greedy_compare.py capture --engine sglang --url "$BASE" \
      --out "$OUT/bs${bs}_c${c}" --limit "$LIMIT" --max-tokens "$MAXTOK" \
      --concurrency "$c" --label "graph_bs=${bs} c=${c}" 2>&1 | tail -2 | tee -a "$OUT/gate.log"
  done
  docker logs "$CONTAINER" > "$OUT/server_bs${bs}.log" 2>&1 || true
  engine_down
done

REF="$OUT/bs${GRAPH_SIZES[0]}_c${CONCS[0]}"
say "=== comparing every capture against $(basename "$REF") ==="
fails=0
for bs in "${GRAPH_SIZES[@]}"; do
  for c in "${CONCS[@]}"; do
    d="$OUT/bs${bs}_c${c}"; [[ "$d" == "$REF" ]] && continue
    printf '  %-18s ' "bs=${bs} c=${c}" | tee -a "$OUT/gate.log"
    # `|| true` is REQUIRED: a FAIL exits 1, and under `set -e` a failing
    # pipeline aborted this loop after its first mismatch, hiding the rest of
    # the matrix (BLOCKERS B-19). A comparison table must print every cell.
    rc=0
    uv run bench/greedy_compare.py compare "$REF" "$d" 2>&1 | tail -1 | tee -a "$OUT/gate.log" || true
    rc=${PIPESTATUS[0]}
    [[ "$rc" == 0 ]] || fails=$((fails+1))
  done
done
# A verdict here is only meaningful beside a SAME-CONFIG control. Without one,
# "N comparisons differ" cannot be distinguished from run-to-run nondeterminism,
# which on this hardware is total (BLOCKERS B-19).
say "raw: $fails of the comparisons are not identical"
say "NOTE: interpret ONLY against a same-config control captured the same way."
say "verdict: $([[ $fails == 0 ]] && echo 'all identical' || echo 'differences present - compare magnitude against the control')"
say "evidence: ${OUT#"$REPO"/}"
