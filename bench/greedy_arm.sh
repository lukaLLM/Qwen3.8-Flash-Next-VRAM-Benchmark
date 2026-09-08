#!/usr/bin/env bash
# Greedy-capture arm: boot ONE engine at the env-supplied config, capture greedy
# outputs with bench/greedy_compare.py at each concurrency in GREEDY_CONC
# (default "1 2"), record provenance, tear down. No aiperf: this exists so a
# LIVE server can be probed for token-level equivalence between two configs or
# two image tags (bench/run.sh always runs a sweep and tears down).
#
#   ./bench/greedy_arm.sh --execute --plan-id FLASHNEXT-R1 <engine> <tag>
#
# env: GREEDY_CONC="1 2"  GREEDY_N=16  GREEDY_MAX_TOKENS=256
#      GREEDY_PROMPTS=lcb_code_2048.jsonl   plus any engine override
#      (CUDA_GRAPH_MAX_BS, LOAD_MODE, LAZY, LLAMA_IMAGE, FREETOKEN_IMAGE ...)
# Evidence: artifacts/greedy_<tag>_<engine>_<stamp>/{c1,c2}/capture.jsonl
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}
source bench/lib.sh

ENGINE_ARG="${1:?usage: greedy_arm.sh <engine> <tag>}"
TAG="${2:?usage: greedy_arm.sh <engine> <tag>}"
bench_init "$ENGINE_ARG" fn_smoke     # engine plumbing only; prompts come from the file

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO/artifacts/greedy_${TAG}_${ENGINE}_${STAMP}"
mkdir -p "$OUT"

cleanup(){ thermal_watchdog_stop 2>/dev/null||true; thermal_stop 2>/dev/null||true; engine_down 2>/dev/null||true; }
trap cleanup EXIT INT TERM

bench_lock
# Library helpers install their own traps and REPLACE ours; re-arm after each
# (run.sh:78-82 does the same; quality.sh once stranded 85 GiB by not doing so).
thermal_start "greedy_${TAG}_${ENGINE}"
trap cleanup EXIT INT TERM
thermal_watchdog_start "$OUT"
trap cleanup EXIT INT TERM
settle_gpu
engine_up
preflight
record_provenance "$OUT"

# A concurrency listed twice (GREEDY_CONC="1 1 2 2") is captured twice on the
# SAME server, into c1 and c1_r2: the run-to-run determinism control without
# which a cross-arm FAIL cannot be read (2026-09-05: every SGLang compare FAILed,
# including c1 vs c2 inside one boot, so the first question is repeatability).
for c in ${GREEDY_CONC:-1 2}; do
  cell="$OUT/c$c"; k=1
  while [ -e "$cell" ]; do k=$((k+1)); cell="$OUT/c${c}_r$k"; done
  log "greedy capture: $ENGINE tag=$TAG c=$c -> $(basename "$cell")"
  ( cd "$(aiperf_wd)" && uv run --project "$REPO" "$REPO/bench/greedy_compare.py" capture \
      --engine "$ENGINE" --url "$BASE" \
      --prompts "$REPO/bench/data/${GREEDY_PROMPTS:-lcb_code_2048.jsonl}" \
      --out "$cell" --limit "${GREEDY_N:-16}" \
      --max-tokens "${GREEDY_MAX_TOKENS:-256}" --concurrency "$c" ) \
    2>&1 | tee -a "$OUT/greedy.log"
done

record_runtime_state "$OUT"
thermal_watchdog_stop
thermal_stop "$OUT"
docker logs "$CONTAINER" > "$OUT/server.log.after" 2>&1 || true
log "done -> $OUT"
