#!/usr/bin/env bash
# Accuracy arm: boot ONE engine at its best-known config, score GSM8K and
# MATH-500 by exact match, tear down. Reuses bench/lib.sh so an accuracy run
# gets the same provenance, thermal watchdog, DCGM trace and memory record as a
# speed run - a score without that evidence is not a result (Auto_Bench.md).
#
#   ./bench/quality.sh --execute --plan-id FLASHNEXT-R1 <engine> [task]
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}
source bench/lib.sh

ENGINE_ARG="${1:?usage: quality.sh <engine> [gsm8k|math500|both]}"
TASK="${2:-both}"
# fn_smoke only supplies engine plumbing (ports, container names, model paths).
# ISL/OSL from it are unused: the prompts come from the dataset file.
bench_init "$ENGINE_ARG" fn_smoke

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO/artifacts/quality_${ENGINE}_${STAMP}"
mkdir -p "$OUT"

cleanup(){ thermal_watchdog_stop 2>/dev/null||true; thermal_stop 2>/dev/null||true; engine_down 2>/dev/null||true; }
trap cleanup EXIT INT TERM

bench_lock
# thermal_start and thermal_watchdog_start each install their OWN generic trap,
# which REPLACES this controller's. run.sh:78-82 re-arms after every such helper
# for exactly that reason; quality.sh did not, and the first failing accuracy run
# stranded 85 GiB of VRAM in a live `flashnext` because engine_down never fired.
thermal_start "quality_${ENGINE}"
trap cleanup EXIT INT TERM
thermal_watchdog_start "$OUT"
trap cleanup EXIT INT TERM
settle_gpu
engine_up
preflight
record_provenance "$OUT"

MODEL_NAME="$(curl -fsS "$BASE/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')"
log "serving model id: $MODEL_NAME"

run_task(){
  local task="$1" data="$2" mt="$3"
  log "$task on $ENGINE (c=${QUALITY_CONCURRENCY:-8}, seed ${SEED_Q:-0})"
  # --project is REQUIRED: aiperf_wd is a clean directory outside the repo (it
  # exists so pydantic-settings cannot auto-load a credential file from CWD), and
  # `uv run` there resolves a DIFFERENT project - a bare env with no openai.
  ( cd "$(aiperf_wd)" && uv run --project "$REPO" "$REPO/bench/quality_run.py" \
      --task "$task" --url "$BASE/v1" --model "$MODEL_NAME" \
      --data "$REPO/bench/data/$data" --out "$OUT/$task" \
      --concurrency "${QUALITY_CONCURRENCY:-8}" --max-tokens "$mt" \
      --seed "${SEED_Q:-0}" ${QUALITY_LIMIT:+--limit $QUALITY_LIMIT} ) \
    2>&1 | tee -a "$OUT/quality.log"
}

[[ "$TASK" == both || "$TASK" == gsm8k   ]] && run_task gsm8k   gsm8k_test.jsonl   1024
[[ "$TASK" == both || "$TASK" == math500 ]] && run_task math500 math500_test.jsonl 3072

record_runtime_state "$OUT"
thermal_watchdog_stop
thermal_stop "$OUT"
docker logs "$CONTAINER" > "$OUT/server.log.after" 2>&1 || true
log "done -> $OUT"
