#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 2 of the overnight run. A SEPARATE script on purpose: bench/overnight.sh
# is already executing, and bash re-reads a running script at a byte offset, so
# editing it mid-flight corrupts it (NOTES.md records that exact accident).
#
# Waits for phase 1, runs the parameter sweeps that phase 1 had no room for, and
# finishes by regenerating the morning report from artifacts.
#
# Same discipline: a failing step is recorded and the queue continues.
# ---------------------------------------------------------------------------
cd /home/luke/Documents/Code/Qwen3.8-Flash-Next-rtx6000pro
LOG=/tmp/overnight.log
say(){ printf '\n===== %s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
clean(){ docker rm -f q38n flashnext freetoken >/dev/null 2>&1; sleep 8; }
run(){ local lf="$1"; shift; "$@" > "$lf" 2>&1
  printf '  rc=%s  %s\n' "$?" "$(grep -aoE 'done ->.*|FAIL.*|GATE FAILED.*|failed to start' "$lf" | tail -1)" | tee -a "$LOG"; }
guard(){ # stop scheduling new work with under 40 min to the 07:04 shutdown
  local left=$(( $(date -d '07:04' +%s) - $(date +%s) ))
  [ "$left" -lt 2400 ] && { say "LESS THAN 40 MIN TO SHUTDOWN - stopping early, generating report"; return 1; }; return 0; }

say "PHASE 2 armed - waiting for phase 1"
while pgrep -f 'bench/overnight.sh' >/dev/null 2>&1; do sleep 60; done
say "PHASE 2 starting"

# UB-01 measured +30.8% prefill at ubatch 2048 on the OLD build. Cheap to confirm
# on this one, and prefill is llama.cpp's weakest axis at long context.
for ub in 1024 2048; do
  guard || break; clean; say "llama.cpp ubatch=$ub"
  run "/tmp/on2_ubatch_${ub}.log" env THINKING=off UBATCH="$ub" \
    ./bench/run.sh --execute --plan-id FLASHNEXT-R1 llamacpp fn_smoke
done

# FreeToken resolved --nvfp4-backend to `triton` by itself; flashinfer is the
# untested alternative and this is its weakest axis (prefill).
for nb in flashinfer triton; do
  guard || break; clean; say "FreeToken nvfp4-backend=$nb"
  run "/tmp/on2_nvfp4_${nb}.log" env THINKING=off FREETOKEN_NVFP4_BACKEND="$nb" \
    ./bench/run.sh --execute --plan-id FLASHNEXT-R1 freetoken fn_smoke
done

# Prefill budget at long context, now that mem-fraction 0.93 freed the pool.
for cp in 8192 16384; do
  guard || break; clean; say "SGLang chunked-prefill=$cp"
  run "/tmp/on2_chunk_${cp}.log" env THINKING=off PREFILL_BUDGET="$cp" \
    ./bench/run.sh --execute --plan-id FLASHNEXT-R1 sglang fn_smoke
done

clean
say "regenerating morning report"
uv run bench/morning_report.py >> "$LOG" 2>&1
say "OVERNIGHT COMPLETE - results/MORNING_REPORT.md"
