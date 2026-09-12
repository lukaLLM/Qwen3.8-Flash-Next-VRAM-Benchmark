#!/usr/bin/env bash
# Does `--lazy-mode auto` stream the PLE table off the SSD when `--load-mode
# none` is in force?
#
# WHY THIS IS OPEN. A viewer said "you are using your SSD for engrams for sure".
# For the llama.cpp bars he was looking at, the saved evidence says no:
# --load-mode none, -ot per_layer_token_embd=CPU, RssAnon 27.24 GiB, and
# results/p3-none/io.json shows 202 major faults in 3,600 s at 0.11 MB/s.
#
# But he lands on one bar we cannot yet clear. B-25: the LAZY env var never
# reached any build newer than b10666, so the llama.cpp+MTP full-window arm
# (context_fn_maxctx_llamacpp_20260906T111813Z, the 210 s / 52.6 tok/s bar) ran
# at the build default `--lazy-mode auto`, which streams any tensor over 4 GiB
# from disk. The PLE table is ~25 GiB.
#
# The build's own help text says `--lazy-mode ... (requires mmap)`, which would
# make `auto` INERT under `--load-mode none` - there is no file mapping to
# demand-page. auto_run/TUNING_QUEUE.md 8a flagged exactly this and said
# "measure, do not assume". Help text is documentation, not measurement.
#
# WHAT THIS GATE DOES. Two arms on llamacpp-mtp:d1a92352, both at the headline
# placement (--load-mode none, -ot per_layer_token_embd=CPU, --n-cpu-moe 0),
# differing only in LAZY. A ple_io_monitor watcher runs alongside each, so the
# answer is a fault rate and an NVMe read rate in a JSON file, not an argument.
#
#   LAZY=off    the intended headline configuration
#   LAZY=auto   what the 2026-09-06 bar actually ran
#
# READING IT. If the two arms' major-fault and MB/s numbers agree, `auto` is
# inert under `--load-mode none`, the published bar was never SSD-streaming, and
# B-25 is closed for that artifact. If `auto` shows sustained faults, the bar is
# a disk-streamed number and must be relabelled or re-measured.
#
# Decode, not load, is what matters: the watcher starts AFTER the server is
# healthy, so weight loading is excluded by construction.
#
# ~25 min. Boot is minutes per arm (a cold IQ4_XS start reads ~88 GB).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"
REPO="$PWD"
source bench/lib.sh

OUT="$REPO/results/gates/lazyres_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
say(){ echo "[lazyres $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/gate.log"; }

BASE_MODEL="$(resolve_gguf_model)"
MTP_HOST="$(find "$HOME/.cache/huggingface/hub" -name 'mtp-Qwen3.8*.gguf' | head -1)"
[[ -n "$MTP_HOST" ]] || die "no MTP sidecar GGUF found"
MTP="/hf${MTP_HOST#"$HOME/.cache/huggingface"}"
IMG="${LLAMA_IMAGE:-llamacpp-mtp:d1a92352}"
WATCH_SECS="${WATCH_SECS:-300}"

echo "$IMG" > "$OUT/image.txt"
say "image $IMG   sidecar $(basename "$MTP_HOST")   watch ${WATCH_SECS}s per arm"

cleanup(){ docker rm -f q38n-lazy >/dev/null 2>&1 || true; }
trap cleanup EXIT

arm(){ # $1 = LAZY value
  local lazy="$1" d="$OUT/$1"
  mkdir -p "$d"
  say "=== arm LAZY=$lazy ==="
  cleanup; sleep 10

  env MODEL="$BASE_MODEL" CTX=262144 PARALLEL=1 BATCH=8192 UBATCH=512 \
      OT=per_layer_token_embd=CPU LOAD_MODE=none LAZY="$lazy" N_CPU_MOE=0 \
      SPEC_TYPE=draft-mtp SPEC_DRAFT_MODEL="$MTP" SPEC_DRAFT_N_MAX=5 SPEC_DRAFT_NGL=99 \
      LLAMA_HOST_PORT=8000 CONTAINER_NAME=q38n-lazy \
      LLAMA_IMAGE="$IMG" \
      docker compose -f docker/docker-compose.yaml up -d q38n >>"$d/boot.log" 2>&1

  # The image is NOT optional here and must be asserted, not assumed. The first
  # run of this gate (lazyres_20260909T193851Z) set $IMG but never passed
  # LLAMA_IMAGE to compose, so both arms silently ran the compose DEFAULT -
  # ghcr b10666 - which cannot load the MTP sidecar at all (B-16):
  #   error loading model: check_tensor_dims: tensor 'token_embd.weight' not found
  #   failed to load draft model, '.../mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'
  # Both arms failed in 40s and the gate reported "incomplete" rather than
  # naming the cause. Assert the running container's image so that can only
  # happen once.
  local ran
  ran="$(docker inspect --format '{{.Config.Image}}' q38n-lazy 2>/dev/null)"
  if [[ "$ran" != "$IMG" ]]; then
    say "  FAIL: container is running '$ran', expected '$IMG'"
    return 1
  fi

  local ok=0 i
  for i in $(seq 1 240); do
    docker ps --format '{{.Names}}' | grep -qx q38n-lazy || { say "  container exited during boot"; break; }
    curl -fsS --max-time 3 http://localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 15
  done
  docker logs q38n-lazy > "$d/server.log" 2>&1 || true
  [[ "$ok" == 1 ]] || { say "  FAIL: never healthy"; return 1; }

  # Prove the flag was honoured rather than assuming it (B-25 is exactly the
  # failure of assuming). The MTP build reads LLAMA_ARG_LAZY_MODE only.
  grep -aiE 'lazy' "$d/server.log" | head -5 | tee -a "$OUT/gate.log" >/dev/null || true

  # Resident-set evidence: with a resident PLE table this should be ~27 GiB.
  local pid
  pid="$(docker inspect --format '{{.State.Pid}}' q38n-lazy 2>/dev/null)"
  grep -E 'RssAnon|RssFile' "/proc/$pid/status" 2>/dev/null | tee "$d/rss.txt" || true

  say "  healthy; watching ${WATCH_SECS}s of DECODE (load already finished)"
  python3 benchmark/ple_io_monitor.py watch \
      --seconds "$WATCH_SECS" --interval 2 --pid "$pid" \
      --out "$d/io.json" >>"$d/watch.log" 2>&1 &
  local wpid=$!

  # Drive real decode for the whole window so the watcher sees token-time I/O,
  # not an idle server.
  local end=$((SECONDS + WATCH_SECS - 10))
  while [[ $SECONDS -lt $end ]]; do
    curl -fsS --max-time 120 http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Explain how a B-tree stays balanced on insert."}],"max_tokens":256,"temperature":0.7,"stream":false}' \
      >>"$d/gen.jsonl" 2>/dev/null || true
  done
  wait $wpid || true
  docker logs q38n-lazy > "$d/server.log.after" 2>&1 || true
  say "  io.json: $(python3 -c "
import json;d=json.load(open('$d/io.json'))
print(' '.join(f'{k}={d[k]}' for k in ('seconds','major_faults','major_faults_per_s','nvme_read_mb','nvme_read_mbs','verdict') if k in d))
" 2>/dev/null || echo unreadable)"
  cleanup; sleep 10
}

arm off  || say "arm off FAILED"
arm auto || say "arm auto FAILED"

say "=== comparison ==="
python3 - "$OUT" <<'PY' | tee -a "$OUT/gate.log"
import json, pathlib, sys
o = pathlib.Path(sys.argv[1])
rows = {}
for a in ("off", "auto"):
    p = o / a / "io.json"
    if p.exists():
        rows[a] = json.loads(p.read_text())
if len(rows) < 2:
    print("incomplete - cannot conclude"); raise SystemExit
for a, d in rows.items():
    print(f"  LAZY={a:<5} faults={d.get('major_faults')} "
          f"faults/s={d.get('major_faults_per_s')} MB/s={d.get('nvme_read_mbs')} "
          f"verdict={d.get('verdict')}")
# RESIDENCY is read from RssAnon, not inferred from I/O. The first corrected
# run (lazyres_20260912T121739Z) showed why: auto had ZERO major faults and
# 0.24 MB/s NVMe - and RssAnon of 0.56 GB against 28.7 GB for off. The table
# was NOT resident; its on-demand reads were simply served from a warm page
# cache (62 GB cached, GGUF just read by the previous arm). I/O alone would
# have called that "inert". It is not inert; it is deferred and cache-hit.
def rss_anon_gb(a):
    try:
        for line in (o / a / "rss.txt").read_text().splitlines():
            if line.startswith("RssAnon"):
                return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return None
r_off, r_auto = rss_anon_gb("off"), rss_anon_gb("auto")
f_off = rows["off"].get("major_faults_per_s") or 0
f_auto = rows["auto"].get("major_faults_per_s") or 0
print(f"\n  RssAnon  off={r_off if r_off is None else f'{r_off:.1f} GB'}  "
      f"auto={r_auto if r_auto is None else f'{r_auto:.1f} GB'}")
if r_off and r_auto and r_auto < 0.5 * r_off:
    print("\n  CONCLUSION: `auto` DEFERS the PLE table under --load-mode none - "
          "it is NOT resident (RssAnon above).")
    if f_auto > 50:
        print("  In this run the deferred reads HIT THE SSD (sustained major faults). "
              "The 2026-09-06 MTP bar is a disk-streamed number.")
    else:
        print("  In this run the deferred reads were served from PAGE CACHE (no "
              "faults, no NVMe). No SSD traffic was measured - but that is a "
              "warm-cache result. Under a cold cache or RAM pressure the same "
              "configuration WOULD stream from disk. The published MTP bar was "
              "not resident either; it was not SSD-streaming only because the "
              "GGUF was warm from the arms before it.")
else:
    print("\n  CONCLUSION: `auto` and `off` are both resident - `auto` is inert "
          "under --load-mode none for this build.")
PY
say "evidence in $OUT"
