# shellcheck shell=bash
# Shared GPU settle/cooldown helper for the benchmark sweeps.
#
# Why this exists: llama.cpp containers come up in a few seconds, so a sweep
# that stops one arm and immediately starts the next benches a GPU that is
# still hot and still holding the previous arm's VRAM. Two concrete problems:
#
#   1. Thermal throttling. Back-to-back arms run the card at sustained load,
#      so late arms clock lower than early ones and the comparison silently
#      measures ORDER instead of configuration.
#   2. VRAM not yet released. The driver frees a dead container's memory
#      asynchronously. Starting the next arm too early either OOMs it or, when
#      it survives, makes the nvidia-smi reading in the mem CSV wrong because
#      it still includes the previous arm's allocation.
#
# gpu_settle() waits for BOTH conditions, then holds a floor delay so the card
# has actually cooled rather than just reported a low instantaneous temp.
#
# Usage:  source "$(dirname "$0")/gpu_settle.sh"   then call  gpu_settle
# Tune:   COOLDOWN_MIN=60 COOLDOWN_TEMP=45 ./benchmark/dspark_sweep.sh
#         COOLDOWN_MIN=0 ...   disables it (fast, but arms are not comparable)

# floor delay in seconds, applied even if the card is already cool
COOLDOWN_MIN="${COOLDOWN_MIN:-30}"
# wait until the GPU is at or below this temperature (Celsius)
COOLDOWN_TEMP="${COOLDOWN_TEMP:-50}"
# give up waiting on temp/VRAM after this long so a sweep can never hang
COOLDOWN_MAX="${COOLDOWN_MAX:-300}"
# consider VRAM released once usage drops below this (MiB); never 0, since the
# display/compositor and other processes legitimately hold a few hundred MiB
COOLDOWN_VRAM_MIB="${COOLDOWN_VRAM_MIB:-2000}"

gpu_settle() {
  command -v nvidia-smi >/dev/null 2>&1 || { sleep "$COOLDOWN_MIN"; return 0; }
  if [[ "$COOLDOWN_MIN" == 0 && "$COOLDOWN_MAX" == 0 ]]; then
    return 0
  fi

  local t0 used temp waited
  t0=$(date +%s)

  # 1. wait for the previous arm's VRAM to actually be released
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [[ -z "$used" ]] && break
    (( used < COOLDOWN_VRAM_MIB )) && break
    waited=$(( $(date +%s) - t0 ))
    (( waited > COOLDOWN_MAX )) && {
      echo "  cooldown: VRAM still ${used}MiB after ${waited}s, continuing anyway" >&2
      break
    }
    sleep 2
  done

  # 2. wait for the card to cool back down
  while :; do
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [[ -z "$temp" ]] && break
    (( temp <= COOLDOWN_TEMP )) && break
    waited=$(( $(date +%s) - t0 ))
    (( waited > COOLDOWN_MAX )) && {
      echo "  cooldown: still ${temp}C after ${waited}s, continuing anyway" >&2
      break
    }
    sleep 5
  done

  # 3. floor delay - a momentary temp dip is not the same as a settled card
  waited=$(( $(date +%s) - t0 ))
  if (( waited < COOLDOWN_MIN )); then
    sleep $(( COOLDOWN_MIN - waited ))
  fi

  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "  cooldown: $(( $(date +%s) - t0 ))s  ->  ${used}MiB / ${temp}C"
}
