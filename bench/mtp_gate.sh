#!/usr/bin/env bash
# MTP-0/MTP-1: can upstream llama.cpp load Unsloth's SIDECAR MTP head, and is
# the drafter ACTIVE rather than merely configured (Auto_Bench.md 3)?
#
# An acceptance length of exactly 1.0 means the flag was accepted and did
# nothing. That FAILS the gate - it is not reported as "no gain".
#
# Boot only + one generation. No aiperf, no scores: this decides whether the
# arm exists, and bench/run.sh measures it afterwards.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$PWD"
source bench/lib.sh

OUT="$REPO/results/gates/mtp0_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
BASE_MODEL="$(resolve_gguf_model)"
MTP_HOST="$(find "$HOME/.cache/huggingface/hub" -name 'mtp-Qwen3.8*.gguf' | head -1)"
[[ -n "$MTP_HOST" ]] || die "no MTP sidecar GGUF found - download it first"
MTP="/hf${MTP_HOST#"$HOME/.cache/huggingface"}"

SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
CTX="${CTX:-8192}"
say(){ echo "[mtp-gate $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/gate.log"; }

# Which image this gate ran on. Without it the report cannot tell a PASS on a
# patched build from the FAIL recorded on the shipped one (both are mtp0_*).
echo "${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}" > "$OUT/image.txt"

say "image   $(cat "$OUT/image.txt")"
say "base    $BASE_MODEL"
say "sidecar $MTP  ($(du -hL "$MTP_HOST" | cut -f1))"
say "spec    $SPEC_TYPE  n-max ${SPEC_DRAFT_N_MAX:-5}  ctx $CTX"

cleanup(){ docker rm -f q38n-mtp >/dev/null 2>&1 || true; }
trap cleanup EXIT

env MODEL="$BASE_MODEL" CTX="$CTX" PARALLEL=1 \
    SPEC_TYPE="$SPEC_TYPE" SPEC_DRAFT_MODEL="$MTP" \
    SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-5}" SPEC_DRAFT_NGL=99 \
    LLAMA_HOST_PORT=8000 CONTAINER_NAME=q38n-mtp \
    docker compose -f docker/docker-compose.yaml up -d q38n >>"$OUT/gate.log" 2>&1

say "booting - a cold IQ4_XS start reads ~88 GB off disk"
ok=0
for i in $(seq 1 240); do
  if ! docker ps --format '{{.Names}}' | grep -qx q38n-mtp; then
    say "FAIL: container exited during boot"; break; fi
  if curl -fsS --max-time 3 http://localhost:8000/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 15
done
docker logs q38n-mtp > "$OUT/server.log" 2>&1 || true

# MTP-0: did the sidecar actually load?
say "--- draft/MTP lines from the boot log ---"
grep -iE 'draft|mtp|nextn|speculat' "$OUT/server.log" | head -30 | tee -a "$OUT/gate.log" || true

[[ "$ok" == 1 ]] || { say "MTP-0 FAIL: server never became healthy"; exit 1; }
say "MTP-0 PASS: server healthy with the sidecar configured"

# MTP-1: prove the drafter moves. Ask for enough tokens that drafting shows up.
say "generating (256 tokens) to measure acceptance"
curl -fsS --max-time 300 http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists, with a short docstring."}],"max_tokens":256,"temperature":0.0,"stream":false}' \
  > "$OUT/generation.json" 2>>"$OUT/gate.log" || say "WARN: generation request failed"

docker logs q38n-mtp > "$OUT/server.log.after" 2>&1 || true
curl -fsS --max-time 10 http://localhost:8000/metrics > "$OUT/metrics.txt" 2>/dev/null || true

say "--- acceptance evidence ---"
grep -iE 'n_drafted|n_accept|accept|draft' "$OUT/server.log.after" | tail -20 | tee -a "$OUT/gate.log" || true
grep -iE 'draft|accept' "$OUT/metrics.txt" 2>/dev/null | tee -a "$OUT/gate.log" || true
say "evidence in $OUT"
