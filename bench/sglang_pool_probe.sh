#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# How much KV pool can SGLang actually get on this card, and at what cost?
#
#   ./bench/sglang_pool_probe.sh
#
# BOOT-ONLY. It reads the pool the engine reports about itself and tears down.
# That number is an UPPER-BOUND CANDIDATE, not proof a request that size is
# admitted - context.sh's capability probe is what tests admission. But it is
# cheap, and the pool is what caps context, so it is the right first screen.
#
# The levers, and why these and not others:
#   mem-fraction-static   the direct VRAM budget knob. 0.90 is the validated
#                         value; above it the engine starts refusing to load.
#   speculation           NEXTN costs twice: draft WEIGHTS in VRAM, and FIVE
#                         mamba state slots per request instead of one. Turning
#                         it off should buy pool on both counts - and cost decode.
#
# NOT swept, with reasons:
#   kv-cache-dtype   already fp8_e4m3; this lever is already spent.
#   page-size        64 changes packing granularity, not total capacity.
#   PLE              already streamed from NVMe, so it occupies no VRAM.
#   cuda-graph bs    already the minimum (1); graphs are a small fixed cost.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="results/gates/sglang_pool_probe_$(date -u +%Y%m%dT%H%M%SZ).tsv"
mkdir -p "$(dirname "$OUT")"
printf 'spec\tmem_fraction\tmamba\tpool_tokens\tvram_mib\tstatus\n' > "$OUT"

probe() { # $1=spec on|off  $2=mem-fraction
  local spec="$1" mf="$2" mamba over=""
  [[ "$spec" == off ]] && { mamba=1; over="-f docker-compose.sglang.nospec.yaml"; } || mamba=5
  docker rm -f flashnext >/dev/null 2>&1; sleep 6
  ( cd docker && SGLANG_CONTEXT_LENGTH=262144 SGLANG_MAX_TOTAL_TOKENS=262144 \
      SGLANG_MAX_RUNNING_REQUESTS=1 SGLANG_MAX_MAMBA_CACHE_SIZE=$mamba \
      SGLANG_MEM_FRACTION_STATIC="$mf" SGLANG_CHUNKED_PREFILL_SIZE=8192 \
      docker compose -f docker-compose.sglang.yaml $over up -d --wait flashnext >/dev/null 2>&1 )
  local rc=$? pool="" vram="" status="booted"
  if [[ $rc -ne 0 ]]; then
    status="FAILED:$(docker logs flashnext 2>&1 | grep -aoE 'ValueError: [^.]{0,70}' | head -1)"
  else
    pool=$(docker logs flashnext 2>&1 | grep -aoE 'max_total_num_tokens=[0-9]+' | tail -1 | grep -oE '[0-9]+')
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$spec" "$mf" "$mamba" "${pool:-?}" "${vram:-?}" "$status" | tee -a "$OUT"
  docker rm -f flashnext >/dev/null 2>&1; sleep 6
}

echo "  spec mem   mamba pool      vram   status"
for mf in 0.90 0.93 0.95; do probe on  "$mf"; done
for mf in 0.90 0.93;      do probe off "$mf"; done
echo "  -> $OUT"
