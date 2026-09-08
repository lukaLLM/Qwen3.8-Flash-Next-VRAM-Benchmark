#!/usr/bin/env bash
# New-config test queue — plan: ~/.claude/plans/tests-now-on-this-dynamic-spring.md
# Steps 1, 1b, 2, 3, 4. Written ONCE and never edited while running (bash
# re-reads a running script at a byte offset). Log lives under results/, not
# /tmp, because /tmp is wiped on reboot and the overnight logs were lost that way.
#
# Rules in force: existing images are never modified (new tags only); host CUDA
# is never touched (builds run inside Docker); one engine at a time; builds run
# with no container up; a failed step is logged, its dependents skipped, the
# queue continues; nothing is retried silently.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"; LOG="$REPO/results/newconfig_queue.log"
G="--execute --plan-id FLASHNEXT-R1"
FT_SHA=af71ba43206e124f5ff6419b47ee36c6e9981078
LC_SHA=c6a9e5c9ae6d6a551217f75c9a04b2e8b1aa62dd
export THINKING=off

say(){ echo "[nq $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say "START $n"; "$@" >>"$LOG" 2>&1; local rc=$?; say "END   $n exit=$rc"; return $rc; }
clean(){ for c in q38n flashnext freetoken q38n-mtp; do docker rm -f "$c" >/dev/null 2>&1; done; sleep 20; }
dcgm(){ curl -fsS --max-time 3 localhost:9401/metrics 2>/dev/null | grep -q '^DCGM_FI_DEV_SM_CLOCK{' \
  || DCGM_NAME=dcgm-bench DCGM_PROF_PORT=9401 DCGM_COUNTERS=bench/dcgm_metrics.csv \
     DCGM_READY_FIELD=DCGM_FI_DEV_SM_CLOCK ./benchmark/dcgm_exporter.sh up >>"$LOG" 2>&1; }
newest(){ ls -dt "$REPO"/artifacts/$1 2>/dev/null | head -1; }
cmp(){ # $1=label $2=dirA $3=dirB $4=cell
  local r; r=$(uv run "$REPO/bench/greedy_compare.py" compare "$2/$4" "$3/$4" --require-tokens 2>&1 | head -1)
  say "COMPARE $1 [$4]: $r"; }

say "========== QUEUE START =========="

# ---------------------------------------------------------------- STEP 1
say "=== STEP 1: graph-batch correctness on the CURRENT SGLang image ==="
for bs in 1 2 8; do
  clean; dcgm
  run "greedy sglang graph_bs=$bs" env ENGINE_CTX=16384 MAX_RUNNING_REQUESTS=2 CUDA_GRAPH_MAX_BS=$bs \
      ./bench/greedy_arm.sh $G sglang graphbs$bs
done
A=$(newest 'greedy_graphbs1_sglang_*')
for bs in 2 8; do
  B=$(newest "greedy_graphbs${bs}_sglang_*")
  [ -n "$A" ] && [ -n "$B" ] && for c in c1 c2; do cmp "graph_bs 1 vs $bs" "$A" "$B" "$c"; done
done
clean; dcgm
run "soak sglang graph_bs=8 c=2 20min" env ENGINE_CTX=16384 MAX_RUNNING_REQUESTS=2 CUDA_GRAPH_MAX_BS=8 \
    CONCURRENCY=2 DURATION=1200 ./bench/run.sh $G sglang fn_fast

# ---------------------------------------------------------------- STEP 1b
say "=== STEP 1b: llama.cpp --load-mode sweep on b10666 (viewer question) ==="
for isl in 8192 32000; do
  for lm in none mmap mlock mmap+mlock dio; do
    clean; dcgm
    guard=""
    case "$lm" in mlock|mmap+mlock) ./bench/host_mem_guard.sh q38n 6 "$LOG" & guard=$! ;; esac
    run "loadmode=$lm isl=$isl" env UBATCH=1024 LOAD_MODE="$lm" LAZY=off CODE_ISL=$isl CONCURRENCY=1 \
        ./bench/run.sh $G llamacpp fn_code_tune
    [ -n "$guard" ] && { kill "$guard" 2>/dev/null; wait "$guard" 2>/dev/null; }
  done
done

# ---------------------------------------------------------------- STEP 2
say "=== STEP 2: FreeToken new tag freetoken:af71ba432 (PR #329) ==="
clean
run "build freetoken:af71ba432" docker build --build-arg FREETOKEN_REF=$FT_SHA -t freetoken:af71ba432 docker/
if docker image inspect freetoken:af71ba432 >/dev/null 2>&1; then
  say "image sha inside: $(docker run --rm --entrypoint cat freetoken:af71ba432 /src/freetoken.sha 2>/dev/null)"
  export FREETOKEN_IMAGE=freetoken:af71ba432
  clean; dcgm; run "ft4 gate new tag" ./bench/ft4_gate.sh $G
  clean; dcgm; run "quality freetoken new tag" env ENGINE_CTX=16384 MAX_RUNNING_REQUESTS=4 QUALITY_CONCURRENCY=4 \
      ./bench/quality.sh $G freetoken both
  unset FREETOKEN_IMAGE
else
  say "SKIP step 2 tests: build failed (freetoken:local untouched)"
fi

# ---------------------------------------------------------------- STEP 3
say "=== STEP 3: llama.cpp PR #28136 new tag llamacpp-pr28136:local ==="
clean
run "build llamacpp-pr28136" env QWEN4EXP_PR=28136 QWEN4EXP_REF=$LC_SHA \
    QWEN4EXP_IMAGE=llamacpp-pr28136:local QWEN4EXP_IMAGE_FULL=llamacpp-pr28136-full:local \
    ./scripts/build_llamacpp.sh
if docker image inspect llamacpp-pr28136:local >/dev/null 2>&1; then
  say "on-direct in --help: $(docker run --rm --entrypoint /app/llama-server llamacpp-pr28136:local --help 2>&1 | grep -c 'on-direct') line(s)"
  export LLAMA_IMAGE=llamacpp-pr28136:local
  clean; dcgm; run "smoke pr28136" ./bench/run.sh $G llamacpp fn_smoke
  clean; dcgm; run "greedy pr28136 LOAD_MODE=none" env LOAD_MODE=none GREEDY_CONC=1 ./bench/greedy_arm.sh $G llamacpp pr28136
  unset LLAMA_IMAGE
  clean; dcgm; run "greedy b10666 LOAD_MODE=none"  env LOAD_MODE=none GREEDY_CONC=1 ./bench/greedy_arm.sh $G llamacpp b10666
  P=$(newest 'greedy_pr28136_llamacpp_*'); Q=$(newest 'greedy_b10666_llamacpp_*')
  [ -n "$P" ] && [ -n "$Q" ] && cmp "b10666 vs pr28136 (LAZY=off)" "$Q" "$P" c1
  export LLAMA_IMAGE=llamacpp-pr28136:local
  for isl in 8192 32000; do
    for lz in off on on-direct; do
      clean; dcgm
      run "lazy=$lz isl=$isl (pr28136, mmap)" env UBATCH=1024 LOAD_MODE=mmap LAZY="$lz" CODE_ISL=$isl CONCURRENCY=1 \
          ./bench/run.sh $G llamacpp fn_code_tune
    done
  done
  unset LLAMA_IMAGE
else
  say "SKIP step 3 tests: build failed (ghcr b10666 untouched)"
fi

# ---------------------------------------------------------------- STEP 4
say "=== STEP 4: rescore + regenerate report ==="
clean
uv run bench/rescore.py >>"$LOG" 2>&1
uv run bench/build_report.py >>"$LOG" 2>&1
say "========== QUEUE COMPLETE =========="
