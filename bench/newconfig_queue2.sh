#!/usr/bin/env bash
# Resume queue after the 2026-09-05 15:23 host reset (results/incidents/
# 2026-09-05_host_reset.md). newconfig_queue.sh is NOT edited (a running bash
# re-reads at a byte offset; it is dead, but the rule is the rule) - this is a
# new file. Ordered cheapest-and-safest first; the two arms that were in flight
# when the host reset run LAST under a host-RAM guard.
#
#  R1  determinism controls: same server, same config, captured twice
#      (every cross-arm greedy compare FAILed - unreadable without this)
#  R2  build the PR's merge-base as its own tag -> isolates PR #28136's two
#      commits from the 44 upstream commits + toolchain between b10666 and it
#  R3  the newer build at the RESIDENT headline config (load-mode none, LAZY=off)
#      at 8192/32000 - it decoded 23% faster than b10666 under mmap at 32K
#  R4  FreeToken freetoken:af71ba432: the first attempt failed on MY command
#      (missing -f docker/freetoken.Dockerfile) - build, ft4 gate, accuracy both
#  R5  lazy=on-direct then lazy=on at 32000 on pr28136, mem guard 8 GiB
#  R6  rescore + report
#
# Rules: existing images untouched (new tags only); host CUDA untouched (Docker
# builds); one engine at a time; builds with no container up; a failed step is
# logged, dependents skipped, nothing retried silently.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"; LOG="$REPO/results/newconfig_queue2.log"
G="--execute --plan-id FLASHNEXT-R1"
FT_SHA=af71ba43206e124f5ff6419b47ee36c6e9981078
PR_SHA=c6a9e5c9ae6d6a551217f75c9a04b2e8b1aa62dd
BASE_SHA=67a17c17caa95742186f8b1ecadd1b5abd6d5ebb   # merge-base of PR 28136 (gh api compare)
BASE_IMG=llamacpp-base28136:67a17c17
export THINKING=off

say(){ echo "[nq2 $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say "START $n"; "$@" >>"$LOG" 2>&1; local rc=$?; say "END   $n exit=$rc"; return $rc; }
clean(){ for c in q38n flashnext freetoken q38n-mtp; do docker rm -f "$c" >/dev/null 2>&1; done; sleep 20; }
dcgm(){ curl -fsS --max-time 3 localhost:9401/metrics 2>/dev/null | grep -q '^DCGM_FI_DEV_SM_CLOCK{' \
  || DCGM_NAME=dcgm-bench DCGM_PROF_PORT=9401 DCGM_COUNTERS=bench/dcgm_metrics.csv \
     DCGM_READY_FIELD=DCGM_FI_DEV_SM_CLOCK ./benchmark/dcgm_exporter.sh up >>"$LOG" 2>&1; }
newest(){ ls -dt "$REPO"/artifacts/$1 2>/dev/null | head -1; }
cmp(){ # $1=label $2=cellA $3=cellB
  local r; r=$(uv run "$REPO/bench/report_data.py" cmp "$2" "$3" 2>&1 | tail -1)
  say "COMPARE $1: $r"; }

say "========== QUEUE2 START =========="

# ---------------------------------------------------------------- R1
say "=== R1: determinism controls (same server, captured twice) ==="
clean; dcgm
run "greedy sglang graph_bs=1 x2" env ENGINE_CTX=16384 MAX_RUNNING_REQUESTS=2 CUDA_GRAPH_MAX_BS=1 \
    GREEDY_CONC="1 1 2 2" ./bench/greedy_arm.sh $G sglang graphbs1rep
S=$(newest 'greedy_graphbs1rep_sglang_*'); S0=$(newest 'greedy_graphbs1_sglang_*')
if [ -n "$S" ]; then
  cmp "sglang bs1 c1 vs c1 repeat (same boot)"  "$S/c1" "$S/c1_r2"
  cmp "sglang bs1 c2 vs c2 repeat (same boot)"  "$S/c2" "$S/c2_r2"
  cmp "sglang bs1 c1 vs c2 (same boot)"         "$S/c1" "$S/c2"
  [ -n "$S0" ] && cmp "sglang bs1 c1 vs earlier boot c1" "$S0/c1" "$S/c1"
fi
clean; dcgm
run "greedy llamacpp b10666 x2" env LOAD_MODE=none GREEDY_CONC="1 1" ./bench/greedy_arm.sh $G llamacpp b10666rep
L=$(newest 'greedy_b10666rep_llamacpp_*'); L0=$(newest 'greedy_b10666_llamacpp_*')
if [ -n "$L" ]; then
  cmp "llamacpp b10666 c1 vs c1 repeat (same boot)" "$L/c1" "$L/c1_r2"
  [ -n "$L0" ] && cmp "llamacpp b10666 c1 vs earlier boot c1" "$L0/c1" "$L/c1"
fi

# ---------------------------------------------------------------- R2
say "=== R2: build PR 28136 merge-base $BASE_IMG (control for the build gate) ==="
clean
run "build $BASE_IMG" env QWEN4EXP_BRANCH=master QWEN4EXP_REF=$BASE_SHA \
    QWEN4EXP_IMAGE=$BASE_IMG QWEN4EXP_IMAGE_FULL=llamacpp-base28136-full:67a17c17 \
    ./scripts/build_llamacpp.sh
if docker image inspect "$BASE_IMG" >/dev/null 2>&1; then
  say "base image revision: $(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' $BASE_IMG)"
  say "on-direct in base --help (expect 0): $(docker run --rm --entrypoint /app/llama-server $BASE_IMG --help 2>&1 | grep -c 'on-direct')"
  clean; dcgm
  run "greedy base28136 LOAD_MODE=none" env LLAMA_IMAGE=$BASE_IMG LOAD_MODE=none GREEDY_CONC="1" \
      ./bench/greedy_arm.sh $G llamacpp base28136
  B=$(newest 'greedy_base28136_llamacpp_*'); P=$(newest 'greedy_pr28136_llamacpp_*')
  if [ -n "$B" ]; then
    [ -n "$P" ]  && cmp "base28136 vs pr28136 (PR's own 2 commits)" "$B/c1" "$P/c1"
    [ -n "$L0" ] && cmp "b10666 vs base28136 (upstream drift + toolchain)" "$L0/c1" "$B/c1"
  fi
else
  say "SKIP R2 greedy: base build failed"
fi

# ---------------------------------------------------------------- R3
say "=== R3: pr28136 build at the RESIDENT headline config (none, LAZY=off) ==="
for isl in 8192 32000; do
  clean; dcgm
  run "resident pr28136 isl=$isl" env LLAMA_IMAGE=llamacpp-pr28136:local UBATCH=1024 LOAD_MODE=none LAZY=off \
      CODE_ISL=$isl CONCURRENCY=1 ./bench/run.sh $G llamacpp fn_code_tune
done

# ---------------------------------------------------------------- R4
say "=== R4: FreeToken new tag freetoken:af71ba432 (PR #329) - with -f this time ==="
clean
run "build freetoken:af71ba432" docker build -f docker/freetoken.Dockerfile \
    --build-arg FREETOKEN_REF=$FT_SHA -t freetoken:af71ba432 docker/
if docker image inspect freetoken:af71ba432 >/dev/null 2>&1; then
  say "image sha inside: $(docker run --rm --entrypoint cat freetoken:af71ba432 /src/freetoken.sha 2>/dev/null)"
  export FREETOKEN_IMAGE=freetoken:af71ba432
  clean; dcgm; run "ft4 gate new tag" ./bench/ft4_gate.sh $G
  clean; dcgm; run "quality freetoken new tag" env ENGINE_CTX=16384 MAX_RUNNING_REQUESTS=4 QUALITY_CONCURRENCY=4 \
      ./bench/quality.sh $G freetoken both
  unset FREETOKEN_IMAGE
else
  say "SKIP R4 tests: build failed (freetoken:local untouched)"
fi

# ---------------------------------------------------------------- R5
say "=== R5: remaining lazy arms at 32000 on pr28136 (host reset suspects; mem guard 8 GiB) ==="
for lz in on-direct on; do
  clean; dcgm
  ./bench/host_mem_guard.sh q38n 8 "$LOG" & guard=$!
  run "lazy=$lz isl=32000 (pr28136, mmap) GUARDED" env LLAMA_IMAGE=llamacpp-pr28136:local UBATCH=1024 LOAD_MODE=mmap \
      LAZY="$lz" CODE_ISL=32000 CONCURRENCY=1 ./bench/run.sh $G llamacpp fn_code_tune
  kill "$guard" 2>/dev/null; wait "$guard" 2>/dev/null
done

# ---------------------------------------------------------------- R6
say "=== R6: rescore + regenerate report ==="
clean
uv run bench/rescore.py >>"$LOG" 2>&1
uv run bench/build_report.py >>"$LOG" 2>&1
say "========== QUEUE2 COMPLETE =========="
