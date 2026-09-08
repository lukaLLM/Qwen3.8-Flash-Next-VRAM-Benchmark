#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Overnight queue. Runs unattended; a failing step is RECORDED and the queue
# CONTINUES, because one harness bug must not cost the whole night (five of
# today's "failures" were harness bugs, not engine limits).
#
# Every step: tear down containers first, log to its own file, never abort the
# queue. Ordered so that anything needing a novel code path comes LAST.
# ---------------------------------------------------------------------------
cd /home/luke/Documents/Code/Qwen3.8-Flash-Next-rtx6000pro
LOG=/tmp/overnight.log
say(){ printf '\n===== %s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
clean(){ docker rm -f q38n flashnext freetoken >/dev/null 2>&1; sleep 8; }
run(){ # run <logfile> <cmd...>
  local lf="$1"; shift
  "$@" > "$lf" 2>&1
  local rc=$?
  printf '  rc=%s  %s\n' "$rc" "$(grep -aoE 'done ->.*|FAIL.*|GATE FAILED.*|failed to start' "$lf" | tail -1)" | tee -a "$LOG"
}

say "STEP 1  waiting for the context ladder already in flight"
while pgrep -f 'ladder_chain.sh' >/dev/null 2>&1; do sleep 60; done
say "STEP 1 done"

say "STEP 2  speculation 2x2 at c=1 (isolates engine from drafter)"
for arm in "sglang on" "sglang off" "llamacpp none" "llamacpp ngram-mod"; do
  set -- $arm; e="$1"; sp="$2"; clean
  if [ "$e" = sglang ]; then
    say "  $e spec=$sp"; run "/tmp/on_spec_${e}_${sp}.log" \
      env THINKING=off SPEC="$sp" ./bench/run.sh --execute --plan-id FLASHNEXT-R1 "$e" fn_smoke
  else
    say "  $e spec-type=$sp"; run "/tmp/on_spec_${e}_${sp}.log" \
      env THINKING=off SPEC_TYPE="$sp" ./bench/run.sh --execute --plan-id FLASHNEXT-R1 "$e" fn_smoke
  fi
done

say "STEP 3  real-code ladder, LiveCodeBench inputs at 8K and 32K"
for isl in 8192 32000; do
  for e in sglang freetoken llamacpp; do
    clean; say "  $e CODE_ISL=$isl"
    run "/tmp/on_code_${e}_${isl}.log" \
      env THINKING=off CODE_ISL="$isl" CONCURRENCY=1 ./bench/run.sh --execute --plan-id FLASHNEXT-R1 "$e" fn_code_tune
  done
done

say "STEP 4  SGLang batch-2 decode CUDA graph A/B"
for g in 1 2; do
  clean; say "  cuda-graph-max-bs-decode=$g"
  run "/tmp/on_graph_${g}.log" env THINKING=off CONCURRENCY=1,2 MAX_RUNNING_REQUESTS=2 \
    SGLANG_CUDA_GRAPH_MAX_BS_DECODE="$g" ./bench/run.sh --execute --plan-id FLASHNEXT-R1 sglang fn_fast
done

say "STEP 5  llama.cpp PLE placement: --load-mode dio vs none"
for lm in none dio; do
  clean; say "  load-mode=$lm"
  run "/tmp/on_loadmode_${lm}.log" env THINKING=off LOAD_MODE="$lm" \
    ./bench/run.sh --execute --plan-id FLASHNEXT-R1 llamacpp fn_smoke
done

clean
say "OVERNIGHT QUEUE COMPLETE"
