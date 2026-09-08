# FreeToken, containerised for the three-engine study.
#
# FreeToken ships as a pip package with no published image, so we build one.
# That is worth the build rather than running it on the host: it puts all three
# arms behind the SAME cgroup memory guard (mem_limit/memswap_limit) instead of
# giving one engine a different failure mode. This box has been taken down twice
# by host OOM, and the guard must not vary by arm.
#
# Build (this is a long job - the operator runs it):
#   docker build -f docker/freetoken.Dockerfile -t freetoken:local docker/
#
# BASE: -devel. I tried -base plus targeted packages and it cost three build
# cycles, because this arm JIT-COMPILES CUDA AT RUNTIME in three separate places
# and each missing piece only surfaces when that kernel is first needed:
#
#   1. FreeToken's own tvm-ffi kernels   -> nvcc missing, failed in `ft bench bw`
#   2. FreeToken's C++ ple_store ext     -> needed build-essential at pip time
#   3. FlashInfer's sampling kernel      -> curand.h missing, and this one fails
#                                           on the FIRST REQUEST, after a
#                                           successful load and 13 min of uptime
#
# Chasing headers one at a time is the wrong shape of fix: the next JIT kernel
# will want cublas or cusparse and we would find out the same way. -devel carries
# the full toolchain and ends the cycle. The targeted packages below are kept
# because they are cheap and make the requirement explicit.
#
# torch's cu13 wheels carry their own CUDA runtime, so -base looked sufficient.
# It is not: FreeToken JIT-COMPILES its own CUDA kernels at runtime through
# tvm-ffi/ninja (fast_index_copy, the offload gather, ...). Without a compiler
# that fails at USE time, not build time - `ft bench bw` reported
# "/usr/local/cuda/bin/nvcc: not found ... ninja: build stopped" and silently
# dropped the PCIe-gather measurement, and a serve path that needs a JIT kernel
# would have failed the same way forty minutes into a model load.
#
# cuda-nvcc-13-0 + cuda-cudart-dev-13-0 is ~200 MB against the ~5 GB the -devel
# image would add.
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VIRTUAL_ENV=/opt/freetoken \
    PATH=/opt/freetoken/bin:/usr/local/bin:$PATH

ARG CUDA_PKG_VERSION=13-0
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        "cuda-nvcc-${CUDA_PKG_VERSION}" "cuda-cudart-dev-${CUDA_PKG_VERSION}" \
        "libcurand-dev-${CUDA_PKG_VERSION}" \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH

# uv manages the environment here exactly as it does on the host side.
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv

# FreeToken requires Python >=3.10 and publishes cp310-cp313 manylinux wheels.
# 3.12 is pinned because it is the version its accel extras are most widely
# built for, and because a floating interpreter would make the image
# irreproducible.
ARG PYTHON_VERSION=3.12
# PINNED TO A COMMIT, NOT A RELEASE - and that is forced, not preference.
#
# MEASURED 2026-09-02: the released freetoken 0.1.2 (PyPI, uploaded 2026-08-19)
# CANNOT LOAD THIS MODEL. Its registry holds 21 architectures, newest Qwen entry
# `Qwen3_5MoeForConditionalGeneration`, and the checkpoint declares
# `Qwen4ExpForConditionalGeneration`. FT-4 failed with:
#
#   ValueError: Model architecture Qwen4ExpForConditionalGeneration not supported
#
# 0.1.2 predates Qwen3.8-Flash-Next entirely, so docs/models.md listing
# `RadixArk/Qwen3.8-Flash-Next-NVFP4` as supported describes UNRELEASED work.
# Support lives on main in python/freetoken/models/qwen4_exp/.
#
# Every engine in this study is therefore on unreleased code - SGLang on a
# patched fork, llama.cpp on an open PR, FreeToken on main. That is a finding
# about the model's age, not sloppiness, and each one is pinned to a SHA.
ARG FREETOKEN_REF=6eca2d7d2b8576c7ad0ba62853df9f618cba929f
ARG FREETOKEN_REPO=https://github.com/FlashML-org/FreeToken

RUN uv venv --python "${PYTHON_VERSION}" "${VIRTUAL_ENV}"

# freetoken[accel] == freetoken[fi,sgl]:
#   flashinfer-python[cu13] >=0.6,<0.7   attention kernels
#   sglang-kernel ==0.4.5                fused MoE / sampling kernels
#   torch >=2.11,<2.12, triton ==3.6.0
#
# Built from source at the pinned SHA: setup.py compiles a C++/CUDA extension
# (ple_store_ext.cpp, the io_uring PLE store), which is why build-essential
# and nvcc all have to be present above.
# Source tarball by SHA, not git. Two measured failures got us here:
#
#   uv pip install "git+URL@<sha>"  -> "failed to fetch commit ... Git operation
#                                      failed": uv's shallow fetch cannot resolve
#                                      an arbitrary commit, even main's tip.
#   git clone (in-container)        -> "could not read Username for
#                                      'https://github.com'" - git asks to
#                                      authenticate against this PUBLIC repo.
#                                      Confirmed public: anonymous
#                                      info/refs?service=git-upload-pack returns
#                                      a real ref listing and HEAD == our SHA.
#
# codeload serves the tree anonymously, addressed by SHA, so it pins exactly,
# needs no git and no credentials, and cannot drift. The SHA is written into the
# image so provenance can read it back.
ADD "https://codeload.github.com/FlashML-org/FreeToken/tar.gz/${FREETOKEN_REF}" /src/ft.tgz
RUN mkdir -p /src/freetoken \
 && tar xzf /src/ft.tgz -C /src/freetoken --strip-components=1 \
 && rm -f /src/ft.tgz \
 && echo "${FREETOKEN_REF}" > /src/freetoken.sha \
 && test -d /src/freetoken/python/freetoken/models/qwen4_exp \
 && echo "qwen4_exp sources present at ${FREETOKEN_REF}"

RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" "/src/freetoken[accel]"

# Fail the BUILD, not the first benchmark, if the accel extras did not land.
# sgl_kernel dispatches on the live device at import, so it cannot be verified
# here without a GPU - check the import surface that does not need one, and
# leave the device-dependent check to the serve-time smoke request.
RUN "${VIRTUAL_ENV}/bin/python" -c "\
import importlib.metadata as m; \
names=('freetoken','torch','triton','flashinfer-python','sglang-kernel'); \
print('\n'.join(f'{n} {m.version(n)}' for n in names))" \
 && nvcc --version | tail -2 \
 && test -f /usr/local/cuda/include/curand.h \
 && echo "curand.h present (FlashInfer sampling JIT needs it on the FIRST request)" \
 && "${VIRTUAL_ENV}/bin/python" -c "\
from freetoken.models.register import _MODEL_REGISTRY as R; \
a='Qwen4ExpForConditionalGeneration'; \
assert a in R, f'{a} NOT registered - wrong ref? have: {sorted(R)}'; \
print(f'{a} registered OK ({len(R)} architectures)')"

EXPOSE 8000
ENTRYPOINT ["/opt/freetoken/bin/ft"]
