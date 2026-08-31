#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Does Qwen3.8-Flash-Next run with no GPU offload at all?
#
# This is the floor of the "will it run on my machine" question and the only
# tier a viewer without a discrete GPU cares about. It is also the cleanest
# demonstration of the PLE argument: 51.2B of this model's 176.94B parameters
# are a lookup table that does no arithmetic, and lookup tables run on CPUs.
#
# Config, and why:
#   -ngl 0                  nothing on the GPU. Every layer on the CPU backend.
#   --load-mode mmap        87.24 GiB of weights against 91 GB of RAM, most of
#                           it already spoken for. Resident loading cannot fit;
#                           mmap pages are reclaimable and degrade instead.
#   --tensor-read-lazy on   deliberately ON here. It streams tensors >4 GiB from
#                           disk on demand, which is the ONLY way the PLE table
#                           coexists with everything else in 91 GB.
#   --threads 16            physical cores, not the 32 SMT threads.
#   ctx 8192                nobody runs 256K on a CPU. Sized to the use case.
#
# METHODOLOGY NOTE - the usual void rule is suspended for this run.
# Every other test in this study is void if ple_io_monitor.py sees sustained
# major faults, because there they mean accidental swapping. Here streaming from
# disk IS the design, so major faults are expected and are not a fault condition.
# The thermal-throttle void rule still applies. This suspension is deliberate and
# is recorded in FINDINGS so the run cannot be mistaken for a clean-memory one.
#
# Runs after the benchmark queue and the mmap control, via the same lock.
# -----------------------------------------------------------------------------
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=results/cpu_only; mkdir -p "$OUT"

[ -z "${SKIP_LOCK:-}" ] && echo "waiting for other GPU work (blocking on the shared lock) ..."
if [ -z "${SKIP_LOCK:-}" ]; then
  exec 9>/tmp/queue_rest.lock
  flock 9
fi
[ -z "${SKIP_LOCK:-}" ] && echo "lock acquired $(date '+%F %H:%M:%S') - starting CPU-only run"

./scripts/serve.sh --down >/dev/null 2>&1; sleep 3
sync; echo "  RAM before: $(free -g | awk '/^Mem/{print $3"G used, "$7"G available"}')"

NGL=0 THREADS=16 THREADS_BATCH=32 LOAD_MODE=mmap LAZY=on \
LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda13 \
  ./scripts/serve.sh --quant UD-IQ4_XS --ctx 8192 --ngl 0 \
    --load-mode mmap --lazy on --no-wait >/dev/null 2>&1

# 87 GiB streamed from disk on a cold cache: allow 25 minutes to become healthy.
echo -n "  loading (up to 25 min): "
ok=0
for i in $(seq 1 300); do
  if curl -sf -m 5 http://localhost:8000/health >/dev/null 2>&1; then ok=1; break; fi
  # a dead container will never become healthy - fail fast instead of waiting out the timer
  st=$(docker ps -a --filter name=q38n --format '{{.Status}}' | head -1)
  case "$st" in Exited*|Restarting*) echo " container $st"; break ;; esac
  [ $((i % 12)) -eq 0 ] && echo -n "$((i/12))m "
  sleep 5
done
if [ "$ok" != 1 ]; then
  echo "FAILED to become healthy"
  docker logs --tail 40 $(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1) 2>&1 | tail -25
  ./scripts/serve.sh --down >/dev/null 2>&1
  exit 1
fi
echo "healthy after $((SECONDS/60))m$((SECONDS%60))s"
echo "  RAM after load: $(free -g | awk '/^Mem/{print $3"G used, "$7"G available"}')"
echo "  swap: $(free -g | awk '/^Swap/{print $3"G used"}')"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  GPU memory (expect ~0): /'

# Bounded on purpose: CPU decode may be ~1 t/s, so 64 output tokens, one prompt.
echo "  --- speed (256 and 2048 token prompts, 64 output tokens) ---"
timeout 5400 uv run benchmark/speed_bench_v2.py \
  --tag cpu_only --lengths 256,2048 --n-predict 64 --num-prompts 1 \
  2>&1 | tee "$OUT/speed.log" | grep -E "tok \| prefill|error|Error" | sed 's/^/    /'
# PIPESTATUS[0], not $? - $? here is grep's status, not timeout's
[ "${PIPESTATUS[0]}" = 124 ] && echo "    TIMED OUT after 90 min - that is itself the answer"

echo "  --- llama-bench cross-check, -ngl 0 ---"
MODEL=$(./scripts/serve.sh --print --quant UD-IQ4_XS 2>/dev/null | awk '/^model/{print $2}')
timeout 3600 docker run --rm -v "$HOME/.cache/huggingface:/hf" \
  ghcr.io/ggml-org/llama.cpp:full-cuda13 --bench \
  -m "$MODEL" -ngl 0 -t 16 -p 512 -n 64 -r 1 \
  2>&1 | tee "$OUT/llama_bench.log" | grep -E '^\||error' | sed 's/^/    /'
[ "${PIPESTATUS[0]}" = 124 ] && echo "    llama-bench timed out"

echo "  swap after: $(free -g | awk '/^Swap/{print $3"G used"}')"
./scripts/serve.sh --down >/dev/null 2>&1
echo "########## CPU-ONLY DONE $(date '+%H:%M:%S') ##########"
