#!/usr/bin/env bash
# Queue 7: three small open items, none of them a speed claim.
#
#   A. Lazy-residency gate, corrected image.        ~25 min
#      Answers "you are using your SSD" for the one llama.cpp bar still open
#      (the 2026-09-06 MTP full-window arm ran --lazy-mode auto). The first
#      attempt passed no LLAMA_IMAGE and ran stock b10666, which cannot load
#      the MTP head; it proved nothing. The gate now asserts the image.
#
#   B. bf16 SSM state with decode pinned to Triton.  ~10 min, boot only
#      A CAPACITY arm. --mamba-ssm-dtype bfloat16 halves the recurrent
#      temporal state. On its own it also auto-promotes decode to FlashInfer,
#      which HANGS on this build (queue 5, S1). Naming the decode backend
#      explicitly suppresses the promotion (the engine checks `is None`), so
#      this keeps the working Triton kernels and takes only the smaller state.
#      Read max_total_num_tokens and the mamba admission line, not tok/s.
#
#   C. Pool probe, spec=off rows.                     ~10 min, boot only
#      Those rows measured a script bug for six days: max_mamba_cache_size=1,
#      which the engine refuses because mamba_ratio=5 is a model constant.
#      Fixed on 2026-09-09, never re-run. The spec=on rows are already
#      reproduced and are skipped here.
#
# Order: B and C first (short, SGLang, boot only), then A (llama.cpp, longest).
# One engine at a time; each step tears down before the next.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"

REPO="$PWD"; LOG="$REPO/results/newconfig_queue7.log"
G="--execute --plan-id FLASHNEXT-R1"
say(){ echo "[nq7 $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say "START $n"; "$@" >>"$LOG" 2>&1; local rc=$?; say "END   $n exit=$rc"; return $rc; }
clean(){ for c in q38n flashnext freetoken q38n-mtp q38n-lazy; do docker rm -f "$c" >/dev/null 2>&1; done; sleep 20; }

clean

# ---------------------------------------------------------------------------
# B. bf16 state, Triton decode - boot-only capacity probe
# ---------------------------------------------------------------------------
say "=== B: bf16 SSM state + decode=triton (capacity, boot only) ==="
OUTB="$REPO/results/gates/sglang_bf16state_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTB"
printf 'arm\tssm_dtype\tdecode_backend\tpool_tokens\tmamba_line\tvram_mib\tstatus\n' > "$OUTB/probe.tsv"

bf16_probe() { # $1=arm  $2=ssm selector  $3=decode selector
  local arm="$1" ssm="$2" dec="$3" over pool vram status="booted" mline laline
  over="$(mktemp -t nq7_XXXXXX.yaml)"
  SPEC=on SGLANG_MAMBA_SSM_DTYPE="$ssm" SGLANG_LINEAR_ATTN_DECODE_BACKEND="$dec" \
    python3 bench/sglang_render_compose.py docker/docker-compose.sglang.yaml "$over" \
    || { say "  $arm: render refused"; rm -f "$over"; return 1; }
  cp "$over" "$OUTB/$arm.gen.yaml"
  docker rm -f flashnext >/dev/null 2>&1; sleep 6
  ( cd docker && SGLANG_CONTEXT_LENGTH=262144 SGLANG_MAX_TOTAL_TOKENS=262144 \
      SGLANG_MAX_RUNNING_REQUESTS=1 SGLANG_MAX_MAMBA_CACHE_SIZE=5 \
      SGLANG_MEM_FRACTION_STATIC=0.93 SGLANG_CHUNKED_PREFILL_SIZE=8192 \
      docker compose -f docker-compose.sglang.yaml -f "$over" up -d --wait flashnext >/dev/null 2>&1 )
  local rc=$?
  docker logs flashnext > "$OUTB/$arm.server.log" 2>&1 || true
  if [[ $rc -ne 0 ]]; then
    status="FAILED:$(grep -aoE '(ValueError|RuntimeError|AssertionError|error): [^.]{0,90}' "$OUTB/$arm.server.log" | head -1)"
  else
    pool=$(grep -aoE 'max_total_num_tokens=[0-9]+' "$OUTB/$arm.server.log" | tail -1 | grep -oE '[0-9]+')
    mline=$(grep -aoE 'max_running_requests is capped to [0-9]+ by the mamba state cache[^)]*\)?' "$OUTB/$arm.server.log" | tail -1)
    laline=$(grep -aoE 'Linear attention kernel backend: [^ ]+ [^ ]+ [^ ]+' "$OUTB/$arm.server.log" | tail -1)
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    say "  $arm: pool=${pool:-?} vram=${vram:-?}MiB  ${laline:-no backend line}"
    [[ -n "$mline" ]] && say "  $arm: $mline"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "${ssm:-default}" "${dec:-default}" \
    "${pool:-?}" "${mline:-none}" "${vram:-?}" "$status" >> "$OUTB/probe.tsv"
  docker rm -f flashnext >/dev/null 2>&1; sleep 6; rm -f "$over"
}

bf16_probe control        ""        ""
bf16_probe bf16_triton    bfloat16  triton
say "  -> $OUTB/probe.tsv"
clean

# ---------------------------------------------------------------------------
# C. Pool probe, spec=off rows only (spec=on already reproduced 2026-09-09)
# ---------------------------------------------------------------------------
say "=== C: pool probe spec=off rows (B-30 re-run) ==="
run "C_pool_probe_specoff" env POOL_PROBE_SPEC=off ./bench/sglang_pool_probe.sh \
  || say "WARN pool probe returned nonzero"
clean

# ---------------------------------------------------------------------------
# A. Lazy-residency gate, corrected
# ---------------------------------------------------------------------------
say "=== A: lazy-residency gate (llamacpp-mtp, correct image) ==="
run "A_lazy_residency" ./bench/lazy_residency_gate.sh $G || say "WARN lazy gate returned nonzero"
clean

say "=== DONE ==="
say "B: $OUTB/probe.tsv"
say "C: $(ls -1t results/gates/sglang_pool_probe_*.tsv | head -1)"
say "A: $(ls -1dt results/gates/lazyres_* | head -1)/gate.log"
