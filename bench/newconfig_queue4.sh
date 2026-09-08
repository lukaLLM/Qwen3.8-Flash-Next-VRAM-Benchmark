#!/usr/bin/env bash
# Queue 4: MTP (multi-token prediction) on llama.cpp, built the way Unsloth's
# own guide says to build it.
#
# SOURCE. The guide does not use an upstream pull request: it clones
# danielhanchen/llama.cpp branch qwen4exp/mtp. That is what this builds, pinned
# to its head d1a92352cb (2026-09-05). Nothing MTP-related has merged upstream —
# PRs #27836 and #28097 are both open and both now CONFLICT with master — so no
# release tag contains this, b10819 included.
#
# scripts/build_llamacpp.sh already takes the remote, the source checkout and the
# branch from the environment, so no script edit is needed: QWEN4EXP_UPSTREAM
# points at the fork and QWEN4EXP_SRC at its own clone directory, because the
# existing checkout's "origin" is ggml-org and fetching a fork branch from it
# would fail.
#
# WHAT IS KEPT FROM OUR STUDY, and what is taken from the guide:
#   guide: --spec-type draft-mtp, --spec-draft-n-max 5, the shared Q8_0 sidecar
#   ours:  UD-IQ4_XS (every other llama.cpp number in this study uses it; the
#          guide's UD-Q4_K_XL would change the checkpoint and the comparison)
#
# The claim under test is Unsloth's: 1.3-1.7x faster, ~170 tok/s on this exact
# card against a ~100 tok/s baseline, with no accuracy loss. All three parts are
# measured: speed against this same build with speculation off, greedy token
# identity off-vs-on, and GSM8K.
#
# Rules unchanged: new tag only, existing images untouched, host CUDA untouched
# (Docker build), one engine at a time, build with no container up, failures
# logged and dependents skipped, nothing retried silently.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"; LOG="$REPO/results/newconfig_queue4.log"
G="--execute --plan-id FLASHNEXT-R1"
MTP_REF=d1a92352cbd417fd840b4e765c0b82f5fe3d1d89   # danielhanchen/llama.cpp qwen4exp/mtp
IMG=llamacpp-mtp:d1a92352
NMAX="${NMAX:-5}"                                  # the guide's --spec-draft-n-max
export THINKING=off

say(){ echo "[nq4 $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say "START $n"; "$@" >>"$LOG" 2>&1; local rc=$?; say "END   $n exit=$rc"; return $rc; }
clean(){ for c in q38n flashnext freetoken q38n-mtp; do docker rm -f "$c" >/dev/null 2>&1; done; sleep 20; }
dcgm(){ curl -fsS --max-time 3 localhost:9401/metrics 2>/dev/null | grep -q '^DCGM_FI_DEV_SM_CLOCK{' \
  || DCGM_NAME=dcgm-bench DCGM_PROF_PORT=9401 DCGM_COUNTERS=bench/dcgm_metrics.csv \
     DCGM_READY_FIELD=DCGM_FI_DEV_SM_CLOCK ./benchmark/dcgm_exporter.sh up >>"$LOG" 2>&1; }
newest(){ ls -dt "$REPO"/artifacts/$1 2>/dev/null | head -1; }
cmp(){ local r; r=$(uv run "$REPO/bench/report_data.py" cmp "$2" "$3" 2>&1 | tail -1); say "COMPARE $1: $r"; }

MTP_HOST="$(find "$HOME/.cache/huggingface/hub" -name 'mtp-Qwen3.8*.gguf' | head -1)"
MTP="/hf${MTP_HOST#"$HOME/.cache/huggingface"}"

say "========== QUEUE4 START (MTP, Unsloth qwen4exp/mtp branch) =========="
say "sidecar: ${MTP_HOST:-MISSING} -> $MTP"
[ -n "$MTP_HOST" ] || { say "ABORT: no MTP sidecar GGUF on disk"; exit 1; }

clean
run "build $IMG (danielhanchen/llama.cpp qwen4exp/mtp)" \
    env QWEN4EXP_UPSTREAM=https://github.com/danielhanchen/llama.cpp.git \
        QWEN4EXP_SRC="$HOME/src/llama.cpp-unsloth-mtp" \
        QWEN4EXP_BRANCH=qwen4exp/mtp QWEN4EXP_REF=$MTP_REF \
        QWEN4EXP_IMAGE=$IMG QWEN4EXP_IMAGE_FULL=llamacpp-mtp-full:d1a92352 \
        ./scripts/build_llamacpp.sh
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  say "SKIP: build failed (every existing image untouched)"; say "========== QUEUE4 COMPLETE =========="; exit 1
fi
say "revision: $(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' $IMG)"
say "draft-mtp in --help: $(docker run --rm --entrypoint /app/llama-server $IMG --help 2>&1 | grep -c 'draft-mtp') line(s)"

export LLAMA_IMAGE=$IMG

# ---- MTP-0/MTP-1: does the sidecar load, and does the drafter actually move?
clean; dcgm
run "MTP gate (load + acceptance, n-max $NMAX)" env SPEC_DRAFT_N_MAX=$NMAX ./bench/mtp_gate.sh
GATE=$(ls -dt "$REPO"/results/gates/mtp0_* 2>/dev/null | head -1)
say "gate evidence: $GATE"
if ! grep -q 'MTP-0 PASS' "$GATE/gate.log" 2>/dev/null; then
  say "STOP: the sidecar does not load even on Unsloth's own branch. No speed arm is run."
  grep -iE 'draft|mtp|nextn|error' "$GATE/server.log" 2>/dev/null | head -10 | tee -a "$LOG"
  unset LLAMA_IMAGE; clean; say "========== QUEUE4 COMPLETE =========="; exit 0
fi
say "MTP-0 PASS — the draft head loads"

# ---- Claim 1: "no accuracy degradation". Greedy token identity, off vs on.
clean; dcgm
run "greedy spec=none"      env LOAD_MODE=none SPEC_TYPE=none GREEDY_CONC="1" ./bench/greedy_arm.sh $G llamacpp mtpoff
clean; dcgm
run "greedy spec=draft-mtp" env LOAD_MODE=none SPEC_TYPE=draft-mtp SPEC_DRAFT_MODEL="$MTP" SPEC_DRAFT_N_MAX=$NMAX \
    SPEC_DRAFT_NGL=99 GREEDY_CONC="1" ./bench/greedy_arm.sh $G llamacpp mtpon
OFF=$(newest 'greedy_mtpoff_llamacpp_*'); ON=$(newest 'greedy_mtpon_llamacpp_*')
[ -n "$OFF" ] && [ -n "$ON" ] && cmp "MTP off vs on (greedy)" "$OFF/c1" "$ON/c1"

# ---- Claim 2: "1.3-1.7x, ~170 tok/s". Against THIS build with MTP off.
for isl in 8192 32000; do
  clean; dcgm
  run "mtp off isl=$isl" env UBATCH=1024 LOAD_MODE=none LAZY=off SPEC_TYPE=none CODE_ISL=$isl CONCURRENCY=1 \
      ./bench/run.sh $G llamacpp fn_code_tune
  clean; dcgm
  run "mtp on  isl=$isl" env UBATCH=1024 LOAD_MODE=none LAZY=off SPEC_TYPE=draft-mtp SPEC_DRAFT_MODEL="$MTP" \
      SPEC_DRAFT_N_MAX=$NMAX SPEC_DRAFT_NGL=99 CODE_ISL=$isl CONCURRENCY=1 ./bench/run.sh $G llamacpp fn_code_tune
done

# ---- Claim 3: "no accuracy degradation", measured rather than asserted.
clean; dcgm
run "GSM8K with MTP on" env ENGINE_CTX=16384 MAX_RUNNING_REQUESTS=4 QUALITY_CONCURRENCY=4 LOAD_MODE=none \
    SPEC_TYPE=draft-mtp SPEC_DRAFT_MODEL="$MTP" SPEC_DRAFT_N_MAX=$NMAX SPEC_DRAFT_NGL=99 \
    ./bench/quality.sh $G llamacpp gsm8k
unset LLAMA_IMAGE
clean
uv run bench/rescore.py >>"$LOG" 2>&1
uv run bench/build_report.py >>"$LOG" 2>&1
say "========== QUEUE4 COMPLETE =========="
