#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Build a llama.cpp CUDA image that can load Qwen3.8-Flash-Next.
#
#   ./scripts/build_llamacpp.sh              build if the image is missing
#   ./scripts/build_llamacpp.sh --rebuild    build even if it is already there
#   ./scripts/build_llamacpp.sh --jobs 16    cap build parallelism (see below)
#   ./scripts/build_llamacpp.sh --ref SHA    build a specific commit
#   ./scripts/build_llamacpp.sh --branch master --ref SHA
#                                            build plain upstream instead of the
#                                            pull ref (PR #27742 is now merged)
#   ./scripts/build_llamacpp.sh --check      preflight only, build nothing
#
# WHY A BUILD AND NOT AN IMAGE
#   qwen4exp arrived in ggml-org/llama.cpp PR #27742, MERGED 2026-08-27 as commit
#   6c84c7d5d. It is therefore in ordinary upstream master now, and `--branch
#   master` is the normal way to build it. The default still fetches
#   pull/27742/head, because that is how every result before 2026-08-27 was built
#   and those need to stay reproducible.
#
#   PULLING IS ALSO AN OPTION NOW, and a reasonable one:
#     docker pull ghcr.io/ggml-org/llama.cpp:server-cuda13
#   That tag is built with CUDA 13.3 too - it is where our CUDA_VERSION came
#   from - and as of b10666 it already contains the qwen4exp merge. Verify any
#   candidate before trusting it:
#     docker run --rm --entrypoint /bin/sh <image> -c 'grep -qa -r qwen4exp /app'
#
#   We keep building for two narrower reasons, neither of which is "the image
#   will not work":
#     - the published tags use generic CUDA arch flags; this builds sm_120a,
#       which is what this card actually is.
#     - a local build can sit on master rather than however far behind the last
#       published tag is, and holding the toolchain fixed is what makes an A/B
#       against an earlier result mean anything.
#
# WHAT IT PRODUCES
#   llamacpp-qwen4exp:local        -> llama-server, for docker/docker-compose.yaml
#   llamacpp-qwen4exp-full:local   -> llama-cli and llama-mtmd-cli, for smoke
#                                     tests and for the multimodal path
#   plus a :<sha> tag on each, so an image can always be traced to a commit.
#   The second build reuses the first one's layers, so it costs a copy.
#
# PINNING
#   A draft PR branch gets force-pushed. QWEN4EXP_REF is the commit this repo
#   was written against; without pinning it, a rebuild silently becomes a
#   different binary and every benchmark before it becomes uncomparable. Bump
#   it deliberately, and say so in the commit message when you do.
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# PR #27742 head as of 2026-08-26.
QWEN4EXP_REF="${QWEN4EXP_REF:-035e22731a7fd70b9854b3a2d64ec68e9b1a45d3}"
QWEN4EXP_PR="${QWEN4EXP_PR:-27742}"
QWEN4EXP_BRANCH="${QWEN4EXP_BRANCH:-}"
QWEN4EXP_SRC="${QWEN4EXP_SRC:-$HOME/src/llama.cpp-qwen4exp}"
QWEN4EXP_UPSTREAM="${QWEN4EXP_UPSTREAM:-https://github.com/ggml-org/llama.cpp.git}"
IMAGE="${QWEN4EXP_IMAGE:-llamacpp-qwen4exp:local}"
IMAGE_FULL="${QWEN4EXP_IMAGE_FULL:-llamacpp-qwen4exp-full:local}"

# 13.3.0 is what upstream CI uses for its server-cuda13 tag and what the DFlash2
# stack on this box already runs on, so the runtime dependency set is known
# good here. Driver 595 (CUDA 13.2) runs a 13.3 userspace via CUDA minor version
# compatibility.
CUDA_VERSION="${CUDA_VERSION:-13.3.0}"
# 120a-real = Blackwell sm_120, which is what the RTX PRO 6000 is. Do NOT use a
# plain "120": ggml's CUDA CMakeLists rewrites ^12[0-9](-real|-virtual)?$ to
# 12Xa, and the "f" suffix variant needs cmake >= 3.31.8 while ubuntu24.04 ships
# 3.28.3. 120a-real passes through untouched and is what upstream uses.
CUDA_ARCH="${CUDA_ARCH:-120a-real}"

FORCE_REBUILD=0
CHECK_ONLY=0
JOBS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --rebuild) FORCE_REBUILD=1; shift ;;
    --check)   CHECK_ONLY=1; shift ;;
    --jobs)    JOBS="$2"; shift 2 ;;
    --ref)     QWEN4EXP_REF="$2"; shift 2 ;;
    # Build plain upstream instead of the pull ref. PR #27742 is merged, so
    # `--branch master --ref <sha>` is now the normal way to build this model.
    --branch)  QWEN4EXP_BRANCH="$2"; shift 2 ;;
    -h|--help) awk 'NR>2 && /^# ----/{exit} NR>2{sub(/^#[[:space:]]?/,""); print}' \
                 "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done

if [ -t 1 ]; then
  B=$'\033[1m'; R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; N=$'\033[0m'
else B=""; R=""; G=""; Y=""; N=""; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '  %s[ ok ]%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s[warn]%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '  %s[fail]%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
step "Preflight"

command -v git >/dev/null || die "git not found"
command -v docker >/dev/null || die "docker not found - install Docker Engine"
docker info >/dev/null 2>&1 \
  || die "cannot talk to the docker daemon (is it running? are you in the docker group?)"
ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '?')"

docker buildx version >/dev/null 2>&1 \
  || die "docker buildx not found - install the docker-buildx-plugin"
ok "buildx $(docker buildx version 2>/dev/null | awk '{print $2}')"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found - install the NVIDIA driver"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
[ "${DRIVER%%.*}" -ge 580 ] 2>/dev/null \
  || die "driver $DRIVER is too old for a CUDA 13 image (need >= 580)"
ok "driver $DRIVER - $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# The GPU passthrough is the most common broken thing on a fresh box and it
# fails at container start with a confusing message, so check it here instead.
docker run --rm --gpus all "docker.io/nvidia/cuda:${CUDA_VERSION}-base-ubuntu24.04" \
  nvidia-smi -L >/dev/null 2>&1 || die "docker cannot see the GPU. Fix with:
      sudo apt install -y nvidia-container-toolkit
      sudo nvidia-ctk runtime configure --runtime=docker
      sudo systemctl restart docker"
ok "nvidia container runtime works"

DOCKER_ROOT="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
avail="$(df -BG --output=avail "$DOCKER_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')"
[ -z "$avail" ] || [ "$avail" -ge 50 ] \
  || warn "only ${avail} GB free on $DOCKER_ROOT; the build wants ~35 GB"
ok "disk checked"

[ "$CHECK_ONLY" -eq 1 ] && { ok "preflight only, nothing built"; exit 0; }

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
if [ -n "${QWEN4EXP_BRANCH:-}" ]; then
  step "Source (upstream ${QWEN4EXP_BRANCH})"
else
  step "Source (PR #${QWEN4EXP_PR})"
fi

have_image() { docker image inspect "$1" >/dev/null 2>&1; }

# Skip the build only if what is already built IS what was asked for. The
# earlier version checked merely that an image existed, so `--ref <other-sha>`
# printed "already present", exited 0, and built nothing - which reads exactly
# like success. A ref request is a request for that commit.
if have_image "$IMAGE" && have_image "$IMAGE_FULL" && [ "$FORCE_REBUILD" -eq 0 ]; then
  rev="$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE" 2>/dev/null || true)"
  want="${QWEN4EXP_REF:0:10}"
  if [ -n "$rev" ] && [ "${rev:0:10}" != "$want" ]; then
    warn "$IMAGE is revision ${rev:0:10}, but ${want} was requested - rebuilding"
  else
    ok "$IMAGE and $IMAGE_FULL already present${rev:+ (revision ${rev})} - use --rebuild to force"
    docker run --rm --entrypoint /app/llama-server "$IMAGE" --version 2>&1 | head -1 || true
    exit 0
  fi
fi

if [ ! -d "$QWEN4EXP_SRC/.git" ]; then
  mkdir -p "$(dirname "$QWEN4EXP_SRC")"
  echo "  cloning $QWEN4EXP_UPSTREAM -> $QWEN4EXP_SRC (blobless, a few hundred MB)"
  git clone --filter=blob:none "$QWEN4EXP_UPSTREAM" "$QWEN4EXP_SRC" || die "clone failed"
fi

# PR #27742 was MERGED on 2026-08-27 as commit 6c84c7d5d, so qwen4exp is now in
# plain upstream master and the pull ref is only needed to reproduce older builds.
#
#   --branch master   build ordinary upstream (what a user can actually get)
#   (default)         fetch pull/27742/head, which is how every result up to
#                     2026-08-27 was built. Kept so those are reproducible.
if [ -n "${QWEN4EXP_BRANCH:-}" ]; then
  git -C "$QWEN4EXP_SRC" fetch --quiet origin "$QWEN4EXP_BRANCH" --force \
    || die "could not fetch origin/${QWEN4EXP_BRANCH}"
else
  # The PR branch lives on a fork we do not control, so fetch the pull ref from
  # upstream rather than adding a remote: pull/N/head always resolves, even after
  # the contributor renames or deletes their branch.
  git -C "$QWEN4EXP_SRC" fetch --quiet origin "pull/${QWEN4EXP_PR}/head:pr-${QWEN4EXP_PR}" --force \
    || die "could not fetch pull/${QWEN4EXP_PR}/head"
fi
git -C "$QWEN4EXP_SRC" checkout --quiet --detach "$QWEN4EXP_REF" || die "cannot check out $QWEN4EXP_REF
      The PR was probably force-pushed past this commit. Check what the head is
      now and pass it explicitly:  --ref \$(git -C $QWEN4EXP_SRC rev-parse pr-${QWEN4EXP_PR})"
SHA_SHORT="$(git -C "$QWEN4EXP_SRC" rev-parse --short=10 HEAD)"
ok "source at $SHA_SHORT"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
step "Build"

BUILDER_ARGS=()
if [ -n "$JOBS" ]; then
  # The Dockerfile hardcodes `cmake --build -j$(nproc)`. On 32 threads that is
  # ~32 concurrent nvcc at roughly 2.5 GB each, which OOMs a 96 GB box. nproc
  # inside the build container honours the cgroup cpuset, so a cpuset-limited
  # docker-container builder is how we cap it without patching upstream.
  # (buildx build has no --cpuset-cpus, and the legacy builder is gone.)
  BLDR="llamacpp-qwen4exp-j${JOBS}"
  docker buildx inspect "$BLDR" >/dev/null 2>&1 || \
    docker buildx create --name "$BLDR" --driver docker-container \
      --driver-opt "cpuset-cpus=0-$((JOBS-1))" >/dev/null \
      || die "could not create the limited buildx builder"
  BUILDER_ARGS=(--builder "$BLDR" --load)
  ok "building with $JOBS cores (builder $BLDR)"
else
  warn "building with all $(nproc) threads; if nvcc OOMs, re-run with --jobs $(( $(nproc) / 2 ))"
fi

build_target() {
  local target="$1" tag="$2" sha_tag="$3"
  echo "  target=$target -> $tag"
  docker buildx build "${BUILDER_ARGS[@]}" \
    --target "$target" \
    -f "$QWEN4EXP_SRC/.devops/cuda.Dockerfile" \
    --build-arg "CUDA_VERSION=$CUDA_VERSION" \
    --build-arg UBUNTU_VERSION=24.04 \
    --build-arg GCC_VERSION=14 \
    --build-arg "CUDA_DOCKER_ARCH=$CUDA_ARCH" \
    --build-arg "APP_VERSION=pr${QWEN4EXP_PR}-${SHA_SHORT}" \
    --build-arg "APP_REVISION=${SHA_SHORT}" \
    -t "$sha_tag" -t "$tag" \
    "$QWEN4EXP_SRC" || die "image build failed (see the output above)"
}

# server first: it is the smaller target and the one compose needs, so a failure
# shows up before paying for the full image. full then reuses its build stage.
build_target server "$IMAGE"      "llamacpp-qwen4exp:${SHA_SHORT}"
build_target full   "$IMAGE_FULL" "llamacpp-qwen4exp-full:${SHA_SHORT}"
ok "built $IMAGE and $IMAGE_FULL (sha tag: ${SHA_SHORT})"

step "Verify"
docker run --rm --entrypoint /app/llama-server "$IMAGE" --version 2>&1 | head -2
# qwen4exp is the whole point of this build; if the arch is not linked in, the
# model will fail to load later with a much less obvious message.
# Search all of /app, not llama-server itself. That file is a ~17 KB launcher
# that dlopens the impl libraries; the arch strings live in libllama.so. Grepping
# the launcher reported a missing architecture on every successful build.
if docker run --rm --entrypoint /bin/sh "$IMAGE" -c \
     'grep -qa -r qwen4exp /app' 2>/dev/null; then
  ok "the architecture is linked in (qwen4exp found in /app)"
else
  warn "no qwen4exp string anywhere in /app - wrong ref, or the arch did not build"
fi

cat <<EOM

Next:
  ./scripts/download_models.sh          fetch UD-IQ4_XS (prompts before any bytes move)
  ./scripts/serve.sh                    start llama-server on it
EOM
