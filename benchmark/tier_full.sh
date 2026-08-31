#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Full context ladder per hardware tier, plus llama-bench for external
# comparability.
#
# Context lives in VRAM, so each tier's reachable context is capped by its card,
# and context competes with expert layers for the same memory. The per-tier
# ctx below is the largest that still leaves room to load.
#
#   8 GiB  -> 32768 max   (10.3 GiB of 262144 KV alone exceeds the card)
#   16 GiB -> 131072
#   24 GiB and up -> 262144
#
# llama-bench runs pp512/pp4096/tg128, the format published elsewhere, so these
# numbers can be compared against other people's rather than only our own.
# -----------------------------------------------------------------------------
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source benchmark/gpu_settle.sh

SERVER_IMG="${SERVER_IMG:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
BENCH_IMG="${BENCH_IMG:-ghcr.io/ggml-org/llama.cpp:full-cuda13}"
QUANT=UD-IQ4_XS
OUT="${OUT:-results/tier_full}"; mkdir -p "$OUT"
CSV="$OUT/full.csv"
[ -f "$CSV" ] || echo "cap,ctx,ncmoe,mode,length,prefill_tps,decode_tps,ttft_ms" > "$CSV"

# cap : ctx : starting ncmoe : short lengths : long lengths
TIERS="${TIERS:-8:16384:48:256,2048,8192:12288|16:131072:45:256,2048,8192,32768:65536,122880|24:262144:42:256,2048,8192,32768:65536,131072,245760|32:262144:36:256,2048,8192,32768:65536,131072,245760|48:262144:23:256,2048,8192,32768:131072,245760|96:262144:0:256,2048,8192,32768:65536,131072,245760}"

MODEL_HOST=$(./scripts/serve.sh --print --quant $QUANT 2>/dev/null | awk '/^model/{print $2}' | sed "s|^/hf|$HOME/.cache/huggingface|")

for spec in ${TIERS//|/ }; do
  IFS=: read -r cap ctx nc short long <<< "$spec"
  echo; echo "############ ${cap} GiB  ctx ${ctx} ############"; date '+  %H:%M:%S'
  ./scripts/serve.sh --down >/dev/null 2>&1; pkill -x vram_cap.py 2>/dev/null; sleep 2
  gpu_settle

  CAPPID=""
  if [ "$cap" != "96" ]; then
    ./benchmark/vram_cap.py --leave "$cap" > "$OUT/cap_${cap}.log" 2>&1 & CAPPID=$!
    sleep 6; grep -q holding "$OUT/cap_${cap}.log" || { echo "  cap failed"; continue; }
  fi

  # Choose the load mode from PREDICTED host RAM rather than trying `none` first.
  # `none` puts offloaded experts in ANONYMOUS memory, which cannot be reclaimed:
  # at 16 GiB the ncmoe-45 config wants 83.5 GiB against 91 GiB total, so the box
  # swapped 7 GiB instead of failing. The server stayed healthy and every number
  # after that was measuring swap. mmap pages are reclaimable and degrade instead.
  RAMNEED=$(python3 -c "print(round(27.2 + $nc*60/48,1))")
  if [ -n "${FORCE_MODE:-}" ]; then
    # Control runs need to defeat this heuristic on purpose: the ladder switched
    # from mmap to `none` at exactly the tier where prefill jumped 2.6x, so the
    # two variables have to be separated by forcing the mode the tier would not
    # have picked. Predicted RAM is still printed so a swap risk stays visible.
    MODE=$FORCE_MODE
    echo "  FORCE_MODE=$MODE (heuristic bypassed; predicted host RAM ${RAMNEED} GiB)"
  elif python3 -c "import sys; sys.exit(0 if $RAMNEED > 70 else 1)"; then
    MODE=mmap; echo "  predicted host RAM ${RAMNEED} GiB > 70 - starting on mmap"
  else MODE=none; fi
  N=$nc; started=0
  for _ in $(seq 1 10); do
    LLAMA_IMAGE="$SERVER_IMG" LOAD_MODE=$MODE LAZY=off N_CPU_MOE=$N \
      ./scripts/serve.sh --quant $QUANT --ctx "$ctx" --load-mode $MODE --n-cpu-moe $N --no-wait >/dev/null 2>&1
    for _ in $(seq 1 90); do
      curl -fsS localhost:8000/health >/dev/null 2>&1 && { started=1; break; }
      st=$(docker inspect -f '{{.State.Status}}' q38n_none 2>/dev/null || echo gone)
      case "$st" in exited|dead|gone|restarting) break;; esac
      sleep 5
    done
    [ "$started" = 1 ] && break
    echo "  no start (ncmoe $N mode $MODE) - escalating"
    ./scripts/serve.sh --down >/dev/null 2>&1
    if [ "$MODE" = none ]; then MODE=mmap; elif [ "$N" -lt 48 ]; then N=$((N+3)); [ $N -gt 48 ] && N=48; else break; fi
  done
  [ "$started" = 1 ] || { echo "  TIER FAILED"; [ -n "$CAPPID" ] && kill -TERM $CAPPID 2>/dev/null; continue; }
  echo "  started: ncmoe $N mode $MODE  VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB"

  # Baseline swap at tier start and trip on the DELTA. An absolute threshold
  # fires on swap left over from a previous run and aborts every tier instantly.
  SW0=$(free -g | awk '/^Swap:/{print $3}'); SW0=${SW0:-0}
  ( while sleep 30; do
      sw=$(free -g | awk '/^Swap:/{print $3}'); sw=${sw:-0}
      if [ $((sw - SW0)) -ge 3 ]; then
        echo "  SWAPPING: +$((sw - SW0)) GiB since tier start (${sw} total) - aborting, this tier is void"
        ./scripts/serve.sh --down >/dev/null 2>&1; break
      fi
    done ) & SWPID=$!
  ./benchmark/gpu_telemetry.py watch --seconds 9000 --out "$OUT/gpu_${cap}.json" > "$OUT/gpu_${cap}.log" 2>&1 & G=$!

  # short lengths: 1 warmup + 2 measured. long: 1 + 1, they repeat to <1%.
  uv run benchmark/speed_bench_v2.py --port 8000 --tag "full${cap}s" --n-predict 256 \
      --num-prompts 2 --lengths "$short" 2>&1 | tee "$OUT/short_${cap}.log" | grep -E "tok \| prefill" | sed 's/^/    /'
  [ -n "$long" ] && uv run benchmark/speed_bench_v2.py --port 8000 --tag "full${cap}l" --n-predict 256 \
      --num-prompts 1 --lengths "$long" 2>&1 | tee "$OUT/long_${cap}.log" | grep -E "tok \| prefill" | sed 's/^/    /'

  for f in "$OUT/short_${cap}.log" "$OUT/long_${cap}.log"; do
    [ -f "$f" ] || continue
    grep -E "^\s*[0-9]+ tok \| prefill" "$f" | while read -r line; do
      L=$(awk '{print $1}' <<<"$line")
      PP=$(sed -n 's/.*prefill \([0-9.]*\) t\/s.*/\1/p' <<<"$line")
      TG=$(sed -n 's/.*decode \([0-9.]*\) t\/s.*/\1/p' <<<"$line")
      TT=$(sed -n 's/.*TTFT \([0-9.]*\) ms.*/\1/p' <<<"$line")
      echo "$cap,$ctx,$N,$MODE,$L,$PP,$TG,$TT" >> "$CSV"
    done
  done
  kill -TERM $G $SWPID 2>/dev/null; wait $G 2>/dev/null; tail -6 "$OUT/gpu_${cap}.log" | sed 's/^/    /'
  ./scripts/serve.sh --down >/dev/null 2>&1

  # llama-bench, standard pp512/pp4096/tg128, same placement
  echo "  --- llama-bench ---"
  docker run --rm --gpus all -v "$HOME/.cache/huggingface:/hf:ro" \
    --entrypoint /app/llama-bench "$BENCH_IMG" \
    -m "$(./scripts/serve.sh --print --quant $QUANT 2>/dev/null | awk '/^model/{print $2}')" \
    -ngl 999 -ncmoe "$N" -ot per_layer_token_embd=CPU -fa on -p 512,4096 -n 128 -r 2 \
    2>&1 | grep -E "^\|" | tee "$OUT/llamabench_${cap}.log" | sed 's/^/    /'

  [ -n "$CAPPID" ] && { kill -TERM $CAPPID 2>/dev/null; wait $CAPPID 2>/dev/null; }
done
pkill -x vram_cap.py 2>/dev/null
echo; echo "############ TIER FULL DONE ############"
column -s, -t < "$CSV"
