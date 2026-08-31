#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Start llama-server on a downloaded Qwen3.8-Flash-Next quant.
#
#   ./scripts/serve.sh                                  UD-IQ4_XS, no speculation
#   ./scripts/serve.sh --quant UD-Q4_K_XL               once that one is downloaded too
#   ./scripts/serve.sh --spec ngram-cache --port 8001   the speculative arm
#   ./scripts/serve.sh --ctx 262144 --n-cpu-moe 20
#   ./scripts/serve.sh --ot per_layer_token_embd=CPU   PLE table off the GPU
#   ./scripts/serve.sh --load-mode mmap+mlock          pin weights in RAM
#   ./scripts/serve.sh --down                           stop it
#   ./scripts/serve.sh --print                          show the resolved path, start nothing
#
# All it really does is turn a quant name into the absolute path of shard 1
# inside the HF cache, then hand that to compose as MODEL. That indirection
# exists because the path contains the snapshot commit hash, which changes
# whenever Unsloth re-uploads, and compose cannot glob.
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.yaml"
GGUF_REPO="${GGUF_REPO:-unsloth/Qwen3.8-Flash-Next-GGUF}"
QUANT="${QUANT:-UD-IQ4_XS}"
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
DOWN=0
PRINT=0
WAIT=1

while [ $# -gt 0 ]; do
  case "$1" in
    --quant)      QUANT="$2"; shift 2 ;;
    --spec)       export SPEC_TYPE="$2"; shift 2 ;;
    --ot)         export OT="$2"; shift 2 ;;
    --load-mode)  export LOAD_MODE="$2"; shift 2 ;;
    --ctx)        export CTX="$2"; shift 2 ;;
    --n-cpu-moe)  export N_CPU_MOE="$2"; shift 2 ;;
    --ngl)        export NGL="$2"; shift 2 ;;
    --lazy)       export LAZY="$2"; shift 2 ;;
    --kv-unified) export KV_UNIFIED="$2"; shift 2 ;;
    --port)       export LLAMA_HOST_PORT="$2"; shift 2 ;;
    --name)       export CONTAINER_NAME="$2"; shift 2 ;;
    --down)       DOWN=1; shift ;;
    --print)      PRINT=1; shift ;;
    --no-wait)    WAIT=0; shift ;;
    -h|--help)    awk 'NR>2 && /^# ----/{exit} NR>2{sub(/^#[[:space:]]?/,""); print}' \
                    "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done

# Compose reads .env from the project directory on its own; this is only so the
# token is in the environment for anything else this script shells out to.
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

# Distinct container names per arm, so a baseline and a speculative server can
# coexist. Without this the second `up` renames the first one out from under it.
#
# Sanitised, because --spec-type takes a COMMA-SEPARATED list and docker only
# allows [a-zA-Z0-9_.-] in a name: `--spec ngram-cache,ngram-mod` would
# otherwise be rejected outright with "Invalid container name".
_spec_tag="${SPEC_TYPE:-none}"
export CONTAINER_NAME="${CONTAINER_NAME:-q38n_${_spec_tag//[^a-zA-Z0-9_.-]/_}}"

if [ "$DOWN" = 1 ]; then
  MODEL=unused docker compose -f "$COMPOSE_FILE" down
  exit 0
fi

# --- resolve the quant to shard 1 -------------------------------------------
# Not a plain glob: the cache can hold several snapshots of one repo after an
# upstream re-upload, and only the one refs/main points at is the current
# revision. Fall back to the newest snapshot that has the quant if refs/main is
# missing or points somewhere that does not.
CACHE_DIR="$HF_HOME_DIR/hub/models--${GGUF_REPO//\//--}"
[ -d "$CACHE_DIR" ] || {
  echo "Not downloaded: $GGUF_REPO is not in $HF_HOME_DIR/hub" >&2
  echo "  ./scripts/download_models.sh" >&2
  exit 1
}

SNAP=""
if [ -f "$CACHE_DIR/refs/main" ]; then
  ref="$(cat "$CACHE_DIR/refs/main")"
  [ -d "$CACHE_DIR/snapshots/$ref/$QUANT" ] && SNAP="$CACHE_DIR/snapshots/$ref"
fi
if [ -z "$SNAP" ]; then
  # Pick the snapshot holding the MOST shards of this quant, not the
  # alphabetically last one.
  #
  # The downloader writes a new snapshot directory per repo commit and symlinks
  # only the files that run fetched, so shards of one quant can sit under an
  # older commit while refs/main points at a newer one. Sorting by name then
  # chose a snapshot with a single shard over a complete one and reported
  # "Incomplete: 1 of 3" while every blob was present on disk.
  SNAP="$(for d in "$CACHE_DIR"/snapshots/*/; do
            [ -d "$d/$QUANT" ] || continue
            printf '%s %s\n' "$(find "$d/$QUANT" -name '*.gguf' 2>/dev/null | wc -l)" "${d%/}"
          done | sort -rn | head -1 | cut -d' ' -f2-)"
fi
[ -n "$SNAP" ] || {
  echo "Quant '$QUANT' is not in the cache for $GGUF_REPO." >&2
  echo "Present:" >&2
  find "$CACHE_DIR/snapshots" -mindepth 2 -maxdepth 2 -type d -printf '  %f\n' 2>/dev/null \
    | sort -u >&2
  exit 1
}

SHARD1="$(find "$SNAP/$QUANT" -name '*-00001-of-*.gguf' | sort | head -1)"
[ -n "$SHARD1" ] || { echo "No shard 1 under $SNAP/$QUANT - the download is incomplete." >&2; exit 1; }

# Every shard has to be present. llama.cpp opens shard 1, reads split.count from
# its metadata, and only then goes looking for the rest, so a missing tail shard
# fails minutes in, after the load has already started.
TOTAL="$(basename "$SHARD1" | sed -n 's/.*-00001-of-\([0-9]*\)\.gguf/\1/p')"
HAVE="$(find "$SNAP/$QUANT" -name '*.gguf' | wc -l)"
if [ "$HAVE" -ne "$((10#$TOTAL))" ]; then
  echo "Incomplete: $HAVE of $((10#$TOTAL)) shards for $QUANT. Re-run:" >&2
  echo "  ./scripts/download_models.sh" >&2
  exit 1
fi

# Host path -> container path. The cache is mounted at /hf.
export MODEL="/hf${SHARD1#"$HF_HOME_DIR"}"
# -L to follow symlinks: every file in a snapshot is a symlink into blobs/,
# so without it du sums the link entries and reports 0.0 GiB.
BYTES="$(du -scbL "$SNAP/$QUANT" 2>/dev/null | tail -1 | cut -f1)"

printf 'quant     %s (%d shards, %.1f GiB)\n' "$QUANT" "$((10#$TOTAL))" \
  "$(awk -v b="$BYTES" 'BEGIN{printf "%.1f", b/1073741824}')"
printf 'model     %s\n' "$MODEL"
printf 'spec      %s\n' "${SPEC_TYPE:-none}"
printf 'ctx       %s      n-cpu-moe %s\n' "${CTX:-32768}" "${N_CPU_MOE:-0}"
# Both are load-bearing for Phase 0 and invisible in the container name, so print
# them: the whole point of that config is PLE off the GPU, compute on it.
printf 'ngl       %s\n' "${NGL:-999}"
# provenance: a VRAM or speed number is not reproducible without these
printf 'lazy      %s\n' "${LAZY:-off}"
printf 'kv-unif   %s\n' "${KV_UNIFIED:-true}"
printf 'ot        %s\n' "${OT:-per_layer_token_embd=CPU}"
printf 'load-mode %s\n' "${LOAD_MODE:-mmap}"
printf 'port      %s      container %s\n' "${LLAMA_HOST_PORT:-8000}" "$CONTAINER_NAME"
[ "$PRINT" = 1 ] && exit 0

# The upstream image carries qwen4exp support since build b10658. Compose pulls
# it on first start if it is not present locally.
docker image inspect "${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}" >/dev/null 2>&1 || \
  echo "Pulling ${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13} on first start ..."

docker compose -f "$COMPOSE_FILE" up -d

[ "$WAIT" = 1 ] || exit 0
PORT="${LLAMA_HOST_PORT:-8000}"
echo
echo "Waiting for health on :$PORT. A cold start reads the whole quant off disk"
echo "(~94 GB for IQ4_XS), so the first one is slow. Follow it with:"
echo "  docker logs -f $CONTAINER_NAME"
for _ in $(seq 1 360); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "ready: http://localhost:$PORT"
    exit 0
  fi
  # A container that has died is not going to become healthy; say so rather
  # than spending the next 90 minutes polling a corpse.
  if [ -z "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]; then
    echo "container stopped. Last lines:" >&2
    docker logs --tail 40 "$CONTAINER_NAME" 2>&1 | sed 's/^/  /' >&2
    exit 1
  fi
  sleep 15
done
echo "still not healthy after 90 min - check docker logs $CONTAINER_NAME" >&2
exit 1
