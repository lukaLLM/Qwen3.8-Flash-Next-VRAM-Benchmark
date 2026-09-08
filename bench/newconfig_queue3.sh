#!/usr/bin/env bash
# Queue 3 (chained after newconfig_queue2.sh): the newest llama.cpp RELEASE,
# b10819 (2026-09-05), built with our own script as a NEW tag and run at the
# resident headline config. Queue2 R3 asks whether the PR #28136 build's speed
# gain is the build, not the placement; this makes the answer installable -
# a release tag a user can actually pull, not an open pull request.
#   build  -> smoke -> greedy at none (gate vs b10666) -> fn_code_tune none 8192/32000
# Rules: existing images untouched; host CUDA untouched (Docker build); one
# engine at a time; build with no container up; failures logged, not retried.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"; LOG="$REPO/results/newconfig_queue3.log"
G="--execute --plan-id FLASHNEXT-R1"
REL_SHA=6a1a922d269908a29cbd4b49c27e6a8e7fd10fae   # tag b10819 (gh api git/ref/tags/b10819)
IMG=llamacpp-b10819:6a1a922d
export THINKING=off

say(){ echo "[nq3 $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say "START $n"; "$@" >>"$LOG" 2>&1; local rc=$?; say "END   $n exit=$rc"; return $rc; }
clean(){ for c in q38n flashnext freetoken q38n-mtp; do docker rm -f "$c" >/dev/null 2>&1; done; sleep 20; }
dcgm(){ curl -fsS --max-time 3 localhost:9401/metrics 2>/dev/null | grep -q '^DCGM_FI_DEV_SM_CLOCK{' \
  || DCGM_NAME=dcgm-bench DCGM_PROF_PORT=9401 DCGM_COUNTERS=bench/dcgm_metrics.csv \
     DCGM_READY_FIELD=DCGM_FI_DEV_SM_CLOCK ./benchmark/dcgm_exporter.sh up >>"$LOG" 2>&1; }
newest(){ ls -dt "$REPO"/artifacts/$1 2>/dev/null | head -1; }
cmp(){ local r; r=$(uv run "$REPO/bench/report_data.py" cmp "$2" "$3" 2>&1 | tail -1); say "COMPARE $1: $r"; }

say "========== QUEUE3 START =========="
clean
run "build $IMG" env QWEN4EXP_BRANCH=master QWEN4EXP_REF=$REL_SHA \
    QWEN4EXP_IMAGE=$IMG QWEN4EXP_IMAGE_FULL=llamacpp-b10819-full:6a1a922d ./scripts/build_llamacpp.sh
if docker image inspect "$IMG" >/dev/null 2>&1; then
  say "revision: $(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' $IMG)"
  say "version:  $(docker run --rm --entrypoint /app/llama-server $IMG --version 2>&1 | head -1)"
  export LLAMA_IMAGE=$IMG
  clean; dcgm; run "smoke b10819" ./bench/run.sh $G llamacpp fn_smoke
  clean; dcgm; run "greedy b10819 LOAD_MODE=none" env LOAD_MODE=none GREEDY_CONC="1" ./bench/greedy_arm.sh $G llamacpp b10819
  B=$(newest 'greedy_b10819_llamacpp_*'); Q=$(newest 'greedy_b10666_llamacpp_*'); P=$(newest 'greedy_pr28136_llamacpp_*')
  [ -n "$B" ] && [ -n "$Q" ] && cmp "b10666 vs b10819" "$Q/c1" "$B/c1"
  [ -n "$B" ] && [ -n "$P" ] && cmp "pr28136 vs b10819" "$P/c1" "$B/c1"
  for isl in 8192 32000; do
    clean; dcgm
    run "resident b10819 isl=$isl" env UBATCH=1024 LOAD_MODE=none LAZY=off CODE_ISL=$isl CONCURRENCY=1 \
        ./bench/run.sh $G llamacpp fn_code_tune
  done
  unset LLAMA_IMAGE
else
  say "SKIP: build failed (ghcr b10666 untouched)"
fi
clean
uv run bench/build_report.py >>"$LOG" 2>&1
say "========== QUEUE3 COMPLETE =========="
