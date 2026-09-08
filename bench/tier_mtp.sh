#!/usr/bin/env bash
# TIER x MTP - decode against a VRAM budget, with the draft head off / on the
# GPU / on the CPU, and (separately) against a HOST-RAM budget.
#
# Answers two questions the published ladder cannot:
#
#   1. The first report's ladder ran every tier on this box's 91 GiB of host
#      RAM and never said so. A viewer with 24 GB VRAM and 64 GB RAM is not on
#      that curve. Here the RAM budget is an explicit, VERIFIED cgroup cap.
#   2. At a fixed VRAM budget, is the 2.6 GiB MTP draft head worth the expert
#      layers it displaces? Fixed placement answers "is adding MTP free"; the
#      bounded search answers the question actually asked.
#
# WHAT A RUN CAN END AS - every arm gets one of these, never pass/fail:
#   completes | heavy-paging | oom | guard-stop | bench-error
#
#   ./bench/tier_mtp.sh --execute --plan-id FLASHNEXT-R1 ladder
#   ./bench/tier_mtp.sh --execute --plan-id FLASHNEXT-R1 viewer
#
# Written once; never edited while running (a running bash re-reads at a byte
# offset - B-22 and the 2026-09-05 incident).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}

MODE="${1:-ladder}"
OUT="$REPO/results/tier_mtp"; mkdir -p "$OUT"
LOG="$OUT/tier_mtp.log"
G="--execute --plan-id FLASHNEXT-R1"

# Pinned across EVERY arm, including the MTP-off baseline, so the build is never
# a variable. The harness defaults (draft length 3, thinking on) are both wrong
# for this comparison.
IMG="${TIER_IMAGE:-llamacpp-mtp:d1a92352}"
MTP_HOST="$(find "$HOME/.cache/huggingface/hub" -name 'mtp-Qwen3.8*.gguf' 2>/dev/null | head -1)"
MTP="/hf${MTP_HOST#"$HOME/.cache/huggingface"}"
export THINKING=off
# aiperf's own wrapper is `timeout --kill-after=60 $AIPERF_TIMEOUT` and defaults
# to 3600. tnogpu_cpu spent that full hour to produce nothing (2 of 4 requests,
# ~20 min each). At this workload a healthy arm finishes in ~2 min, so 900 s is
# generous and an arm that exceeds it is itself the finding: too slow to measure.
export AIPERF_TIMEOUT="${AIPERF_TIMEOUT:-900}"

# The published ladder's own per-tier offload (benchmark/tier_full.sh), so the
# new bars sit at the old placements.
ncmoe_for() { case "$1" in 8) echo 48;; 16) echo 45;; 24) echo 42;; 32) echo 36;; 48) echo 23;; *) echo 0;; esac; }

say(){ echo "[tier $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# preflight refuses to run without DCGM, and the exporter does not survive a
# reboot. Bring it up once rather than losing an arm to it.
dcgm(){ curl -fsS --max-time 3 localhost:9401/metrics 2>/dev/null | grep -q '^DCGM_FI_DEV_SM_CLOCK{' \
  || DCGM_NAME=dcgm-bench DCGM_PROF_PORT=9401 DCGM_COUNTERS=bench/dcgm_metrics.csv \
     DCGM_READY_FIELD=DCGM_FI_DEV_SM_CLOCK ./benchmark/dcgm_exporter.sh up >>"$LOG" 2>&1; }

# --- GPU balloon -----------------------------------------------------------
# settle_gpu() in lib.sh waits for VRAM < 2000 MiB with a 300s cap, so a balloon
# started BEFORE it would stall every capped arm for the full 300s (~90 min
# across the ladder). Settle here with the balloon DOWN, allocate, then raise
# the harness's own threshold past what we hold.
BALLOON_PID=""
balloon_up() {
  local leave="$1"
  [[ "$leave" == "96" || "$leave" == "nogpu" ]] && { export SETTLE_VRAM_MIB=2000; return 0; }
  local before; before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  uv run benchmark/vram_cap.py --leave "$leave" > "$OUT/cap_${leave}.log" 2>&1 &
  BALLOON_PID=$!
  for _ in $(seq 1 30); do
    grep -q holding "$OUT/cap_${leave}.log" 2>/dev/null && break
    kill -0 "$BALLOON_PID" 2>/dev/null || { say "FAIL: balloon died during allocation"; return 1; }
    sleep 1
  done
  grep -q holding "$OUT/cap_${leave}.log" 2>/dev/null || { say "FAIL: balloon never reported holding"; return 1; }
  local held; held=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  export SETTLE_VRAM_MIB=$(( held + 2000 ))
  say "  balloon pid $BALLOON_PID holding $(( held - before )) MiB; settle threshold -> ${SETTLE_VRAM_MIB} MiB"
}
balloon_down() {
  [[ -z "$BALLOON_PID" ]] && return 0
  # A "holding" line proves its first second only. If it died mid-run the arm
  # measured a bigger card than it claims, and the number is void.
  if ! kill -0 "$BALLOON_PID" 2>/dev/null; then
    say "  WARNING: balloon $BALLOON_PID was NOT alive at teardown - this arm is VOID"
    BALLOON_PID=""; return 3
  fi
  kill "$BALLOON_PID" 2>/dev/null; wait "$BALLOON_PID" 2>/dev/null
  BALLOON_PID=""; sleep 3
}

# --- cgroup sampler --------------------------------------------------------
# The counters vanish with the container, and they are what separates "slow but
# working" from "thrashing" from "killed".
SAMPLER_PID=""
sampler_start() {
  local dest="$1"
  ( printf 'ts\tcurrent\tanon\tfile\tpgmajfault\tio_read_bytes\tevents_high\tevents_oom\n' > "$dest"
    local id cg
    for _ in $(seq 1 120); do
      id=$(docker inspect -f '{{.Id}}' q38n 2>/dev/null) && [[ -n "$id" ]] && break
      sleep 1
    done
    for c in "/sys/fs/cgroup/system.slice/docker-${id}.scope" "/sys/fs/cgroup/docker/${id}"; do
      [[ -d "$c" ]] && { cg="$c"; break; }
    done
    [[ -n "${cg:-}" ]] || exit 0
    while [[ -d "$cg" ]]; do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%H:%M:%S)" \
        "$(cat "$cg/memory.current" 2>/dev/null || echo 0)" \
        "$(awk '/^anon /{print $2}' "$cg/memory.stat" 2>/dev/null || echo 0)" \
        "$(awk '/^file /{print $2}' "$cg/memory.stat" 2>/dev/null || echo 0)" \
        "$(awk '/^pgmajfault /{print $2}' "$cg/memory.stat" 2>/dev/null || echo 0)" \
        "$(awk '{for(i=1;i<=NF;i++) if($i ~ /^rbytes=/){split($i,a,"="); s+=a[2]} } END{print s+0}' "$cg/io.stat" 2>/dev/null || echo 0)" \
        "$(awk '/^high /{print $2}' "$cg/memory.events" 2>/dev/null || echo 0)" \
        "$(awk '/^oom_kill /{print $2}' "$cg/memory.events" 2>/dev/null || echo 0)" \
        >> "$dest"
      sleep 5
    done ) &
  SAMPLER_PID=$!
}
sampler_stop() { [[ -n "$SAMPLER_PID" ]] && { kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""; }; }

# --- one arm ---------------------------------------------------------------
# $1 label  $2 tier(8|16|24|32|48|96|nogpu)  $3 n_cpu_moe  $4 spec(off|gpu|cpu)
# env: LLAMA_MEM_LIMIT + MEM_CAP_BYTES for a capped arm, LOAD_MODE, LAZY
run_arm() {
  local label="$1" tier="$2" nc="$3" spec="$4"
  local before; before=$(ls -d "$REPO"/artifacts/fn_tier_llamacpp_* 2>/dev/null | wc -l)
  say "ARM $label  tier=$tier n_cpu_moe=$nc spec=$spec load=${LOAD_MODE:-mmap} lazy=${LAZY:-off} cap=${LLAMA_MEM_LIMIT:-none}"

  docker rm -f q38n >/dev/null 2>&1; sleep 5
  dcgm
  # Settle with the balloon DOWN, then allocate.
  local t0=$(date +%s) used
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( used < 2000 )) && break
    (( $(date +%s) - t0 > 120 )) && { say "  GPU did not drain (${used} MiB) - continuing"; break; }
    sleep 5
  done
  balloon_up "$tier" || return 1

  local specenv=()
  case "$spec" in
    off) specenv=(SPEC_TYPE=none) ;;
    gpu) specenv=(SPEC_TYPE=draft-mtp SPEC_DRAFT_MODEL="$MTP" SPEC_DRAFT_N_MAX=5 SPEC_DRAFT_NGL=99) ;;
    cpu) specenv=(SPEC_TYPE=draft-mtp SPEC_DRAFT_MODEL="$MTP" SPEC_DRAFT_N_MAX=5 SPEC_DRAFT_NGL=0) ;;
  esac
  local nglenv=(); [[ "$tier" == "nogpu" ]] && nglenv=(NGL=0)

  sampler_start "$OUT/cgroup_${label}.tsv"
  env LLAMA_IMAGE="$IMG" N_CPU_MOE="$nc" UBATCH=512 \
      LOAD_MODE="${LOAD_MODE:-mmap}" LAZY="${LAZY:-off}" \
      ${LLAMA_MEM_LIMIT:+COMPOSE_OVERRIDE=docker-compose.llamacpp.mem.yaml} \
      ${LLAMA_MEM_LIMIT:+LLAMA_MEM_LIMIT="$LLAMA_MEM_LIMIT"} \
      ${MEM_CAP_BYTES:+MEM_CAP_BYTES="$MEM_CAP_BYTES"} \
      "${specenv[@]}" ${nglenv[@]+"${nglenv[@]}"} \
      ./bench/run.sh $G llamacpp fn_tier >>"$LOG" 2>&1
  local rc=$?
  sampler_stop
  local bal=0; balloon_down || bal=$?

  local after; after=$(ls -d "$REPO"/artifacts/fn_tier_llamacpp_* 2>/dev/null | wc -l)
  local art=""; [[ "$after" -gt "$before" ]] && art=$(ls -dt "$REPO"/artifacts/fn_tier_llamacpp_* | head -1)
  say "  exit=$rc artifact=$(basename "${art:-none}") balloon_ok=$([[ $bal -eq 0 ]] && echo yes || echo NO)"
  echo -e "${label}\t${tier}\t${nc}\t${spec}\t${LOAD_MODE:-mmap}\t${LAZY:-off}\t${LLAMA_MEM_LIMIT:-none}\t${rc}\t$(basename "${art:-none}")\t${bal}" >> "$OUT/arms.tsv"
  return $rc
}

[[ -f "$OUT/arms.tsv" ]] || printf 'label\ttier\tn_cpu_moe\tspec\tload_mode\tlazy\tmem_cap\texit\tartifact\tballoon\n' > "$OUT/arms.tsv"
say "========== MODE=$MODE image=$IMG =========="
[[ -n "$MTP_HOST" ]] || { say "ABORT: no MTP sidecar on disk"; exit 1; }

case "$MODE" in
  smoke)
    # One tier, all three arms, plus a deliberate cap that must be verified.
    run_arm smoke24_off  24 42 off
    run_arm smoke24_cpu  24 42 cpu
    ;;
  ladder)
    # TIERS/SPECS are overridable so a stopped ladder can resume where it left
    # off. CPU drafting is NOT in the default SPECS any more: measured at
    # nogpu/8/16/24 it costs ~4x decode every time (7.2-9.7 against 31-34), and
    # re-proving it at the remaining tiers would cost hours. The arms that
    # established it are kept; ask for it explicitly with SPECS="off gpu cpu".
    for tier in ${TIERS:-nogpu 8 16 24 32 48 96}; do
      nc=$(ncmoe_for "$tier"); [[ "$tier" == "nogpu" ]] && nc=48
      for spec in ${SPECS:-off gpu}; do
        [[ "$tier" == "nogpu" && "$spec" == "gpu" ]] && continue
        run_arm "t${tier}_${spec}" "$tier" "$nc" "$spec"
      done
    done
    ;;
  viewer)
    for cap in 64 56; do
      export LLAMA_MEM_LIMIT="${cap}g" MEM_CAP_BYTES=$(( cap * 1024 * 1024 * 1024 ))
      LOAD_MODE=mmap LAZY=off  run_arm "v${cap}_published" 24 42 off
      LOAD_MODE=mmap LAZY=on   run_arm "v${cap}_lazy"      24 42 off
      LOAD_MODE=mmap LAZY=on   run_arm "v${cap}_lazy_mtp"  24 42 cpu
    done
    unset LLAMA_MEM_LIMIT MEM_CAP_BYTES
    LOAD_MODE=mmap LAZY=on run_arm "v_uncapped_lazy" 24 42 off
    ;;
  *) say "unknown mode: $MODE"; exit 64 ;;
esac
say "========== DONE - classifying =========="
uv run bench/tier_classify.py "$OUT" 2>&1 | tee -a "$LOG"
