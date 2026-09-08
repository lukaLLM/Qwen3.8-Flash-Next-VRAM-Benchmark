#!/usr/bin/env bash
# ============================================================================
# FT-4 - the FreeToken go/no-go: does the arm LOAD at all, with a live abort.
#
#   ./bench/ft4_gate.sh --execute --plan-id FLASHNEXT-R1 [--ctx 32768]
#
# WHY THIS IS THE RISKIEST GATE SO FAR, and why it is watched rather than
# launched and left:
#
#   fused puts 63.32 GiB of NVFP4 experts on the GPU (of 95.59 GiB) and pins the
#   47.68 GiB PLE table in HOST RAM, on a box with 91 GiB. FT-2 already showed
#   pinned memory is NOT returned promptly between allocations - a nominal 48 GiB
#   probe peaked around 79 GiB and touched swap. If FreeToken stages the PLE load,
#   transient peak can sit well above the table size.
#
#   The 86 GiB cgroup cap is the hard floor under that. This script adds a SOFT
#   one: if host availability falls below ABORT_AVAIL_GIB the container is stopped
#   immediately, because by the time the cgroup killer fires the desktop has
#   already been swapping. This machine has been taken down three times; a watched
#   abort is cheaper than a reboot.
#
# Records: memory.json (peak, restarts, OOM, swap), incidents.md, the boot log,
# and the smoke response - so a failure is evidence, not an anecdote.
# ============================================================================
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/plan_gate.sh"
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

CTX="${CTX:-32768}"
while [ $# -gt 0 ]; do
  case "$1" in --ctx) CTX="$2"; shift 2 ;; *) shift ;; esac
done

ABORT_AVAIL_GIB="${ABORT_AVAIL_GIB:-6}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-1800}"

bench_init freetoken fn_fast
OUT="$REPO/results/gates/ft4_${STAMP}"
mkdir -p "$OUT"

log "FT-4: FreeToken fused load, ctx $CTX"
echo "  cap        ${FREETOKEN_MEM_LIMIT:-86g} cgroup   abort below ${ABORT_AVAIL_GIB} GiB host available"
echo "  evidence   $OUT"

bench_lock
gpu_guard || { incident "$OUT" gate-fail "GPU not idle at start"; exit 1; }

avail0=$(free -g | awk '/^Mem:/{print $7}')
echo "  host available at start: ${avail0} GiB"

( cd "$COMPOSE_DIR" && ENGINE_CTX="$CTX" MAX_RUNNING_REQUESTS=1 CUDA_GRAPH_MAX_BS=1 \
    PREFILL_BUDGET=8192 docker compose -f "$COMPOSE_FILE" up -d "$SERVICE" ) \
  || { incident "$OUT" crash "compose up failed"; exit 1; }

log "watching load (abort if host available < ${ABORT_AVAIL_GIB} GiB)"
printf '  %-8s %-10s %-10s %s\n' "t" "avail_GiB" "swap_GiB" "state" | tee "$OUT/load_watch.log"
t=0; healthy=0
while [ "$t" -lt "$BOOT_TIMEOUT" ]; do
  read -r avail swap <<<"$(free -g | awk '/^Mem:/{a=$7} /^Swap:/{s=$3} END{print a" "s}')"
  state=$(docker inspect -f '{{.State.Status}}/{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo gone)
  printf '  %-8s %-10s %-10s %s\n' "${t}s" "$avail" "$swap" "$state" | tee -a "$OUT/load_watch.log"

  if [ "${avail:-99}" -lt "$ABORT_AVAIL_GIB" ]; then
    incident "$OUT" oom "ABORTED: host available fell to ${avail} GiB (< ${ABORT_AVAIL_GIB}); stopping before the desktop swaps"
    docker stop -t 5 "$CONTAINER" >/dev/null 2>&1
    record_runtime_state "$OUT"; engine_down; exit 2
  fi
  case "$state" in
    */healthy) healthy=1; break ;;
    exited*|gone) incident "$OUT" crash "container is $state during load"; break ;;
  esac
  sleep 10; t=$((t+10))
done

docker logs "$CONTAINER" > "$OUT/server.log" 2>&1 || true

if [ "$healthy" != 1 ]; then
  incident "$OUT" gate-fail "did not become healthy within ${BOOT_TIMEOUT}s"
  record_runtime_state "$OUT"; engine_down
  echo "  FT-4 FAIL - see $OUT/server.log"; exit 1
fi

log "healthy after ${t}s - warm-up request (absorbs first-request JIT)"

# TWO REQUESTS, DELIBERATELY. The first request is where FlashInfer JIT-compiles
# its sampling kernel - that is what killed attempt 3 with
#   flashinfer/sampling.cuh: fatal error: curand.h: No such file or directory
# minutes AFTER a clean load and "API server is ready to serve". So request one
# is a warm-up whose only job is to trigger every lazily-compiled kernel, and it
# gets a long timeout. Request two is the one that counts.
#
# Conflating them would also mean the first measured latency silently included a
# multi-minute nvcc run.
ask() { # $1=max_tokens $2=outfile $3=timeout
  curl -fsS --max-time "$3" "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that reverses a linked list. Explain briefly.\"}],\"max_tokens\":$1,\"temperature\":0}" \
    > "$2" 2>"${2%.json}.err"
}

# READINESS IS NOT THE HEALTHCHECK. /health returns 200 as soon as the HTTP
# server binds - before weights load, before expert banks build. The container
# reported "healthy" after 10s while the engine was still on
# "expert banks: slow path (serial build)", and the warm-up got a flat 503.
#
# /engine/health would be the engine's own signal, but it 404s in this build even
# after "API server is ready to serve", so it cannot be the container probe either.
#
# So: treat docker health as "the process is alive" and establish READINESS the
# only way that cannot lie - keep asking for a real generation until one succeeds.
ready=0; waited=0
while [ "$waited" -lt "${READY_TIMEOUT:-1800}" ]; do
  if ask 8 "$OUT/warmup.json" 300; then ready=1; break; fi
  st=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo gone)
  case "$st" in exited|gone)
    incident "$OUT" crash "container is $st while waiting for readiness"; break ;;
  esac
  sleep 15; waited=$((waited+15))
  [ $((waited % 60)) -eq 0 ] && echo "  still not answering after ${waited}s (engine loading)"
done
if [ "$ready" != 1 ]; then
  incident "$OUT" gate-fail "no successful generation within ${READY_TIMEOUT:-1800}s - see warmup.err and server.log"
  docker logs "$CONTAINER" > "$OUT/server.log" 2>&1 || true
  record_runtime_state "$OUT"; engine_down
  echo "  FT-4 FAIL: warm-up request did not complete"; exit 1
fi
log "warm-up ok - measured smoke request"

t0=$(date +%s%3N)
if ! ask 256 "$OUT/smoke.json" "${SMOKE_TIMEOUT:-900}"; then
  incident "$OUT" gate-fail "smoke request failed after a successful warm-up"
  docker logs "$CONTAINER" > "$OUT/server.log" 2>&1 || true
  record_runtime_state "$OUT"; engine_down
  echo "  FT-4 FAIL: smoke request failed"; exit 1
fi
t1=$(date +%s%3N)
echo "  smoke wall time: $(( (t1-t0) ))ms"

docker logs "$CONTAINER" > "$OUT/server.log" 2>&1 || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$OUT/vram_after_load.txt" 2>&1
record_runtime_state "$OUT"

# The gate PASSES only if the model actually produced work, not merely tokens.
# FINDINGS.md records three runs that reported plausible tok/s while emitting
# nothing but tool-call syntax.
if ! python3 - "$OUT/smoke.json" "$(( t1-t0 ))" <<'PY'
import json,sys,pathlib
p,ms = pathlib.Path(sys.argv[1]), int(sys.argv[2])
try:
    d=json.loads(p.read_text()); m=d["choices"][0]["message"]
    txt=(m.get("content") or "")+(m.get("reasoning_content") or "")
    u=d.get("usage",{}) or {}
    out=u.get("completion_tokens") or 0
    print(f"  usage      : {u.get('prompt_tokens')} in / {out} out")
    print(f"  decode     : {out/(ms/1000):.2f} tok/s wall (single request, not a benchmark)")
    print(f"  has code   : {'def ' in txt}   chars: {len(txt)}")
    print("  ---"); print("  "+txt[:240].replace("\n","\n  "))
    ok = out > 0 and len(txt.strip()) > 0
    print(f"  verdict    : {'PASS' if ok else 'FAIL - tokens but no content'}")
    sys.exit(0 if ok else 1)
except Exception as e:
    print("  smoke unreadable:", e); sys.exit(1)
PY
then
  incident "$OUT" gate-fail "smoke produced no usable output"
  engine_down; echo "  FT-4 FAIL"; exit 1
fi

log "tearing down"; engine_down
echo "  FT-4 evidence -> $OUT"
