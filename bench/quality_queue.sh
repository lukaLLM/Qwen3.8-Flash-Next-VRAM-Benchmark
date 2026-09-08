#!/usr/bin/env bash
# Q2: GSM8K + MATH-500 across all three stacks at their best-known configs.
# One engine at a time; each arm boots fresh and tears down.
#
# ENGINE_CTX/MAX_RUNNING_REQUESTS are raised above fn_smoke.conf's values because
# accuracy is concurrency-invariant and 8 slots turn a 45-minute serial arm into
# a few minutes. SGLang divides the pool by slots, so 65536/8 = 8192 per request
# - enough for MATH-500's 3072-token answers plus its prompt.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG=/tmp/claude-1000/q_queue.log
say(){ echo "[queue $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

for eng in sglang llamacpp freetoken; do
  say "=== $eng : gsm8k + math500 ==="
  env THINKING=off ENGINE_CTX=16384 MAX_RUNNING_REQUESTS=4 QUALITY_CONCURRENCY=4 \
      ./bench/quality.sh --execute --plan-id FLASHNEXT-R1 "$eng" both >>"$LOG" 2>&1
  rc=$?
  say "$eng exit=$rc"
  docker rm -f q38n flashnext freetoken >/dev/null 2>&1 || true
  sleep 30
done
say "QUEUE COMPLETE"
