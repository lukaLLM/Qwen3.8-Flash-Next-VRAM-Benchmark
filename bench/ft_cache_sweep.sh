#!/usr/bin/env bash
# FREETOKEN LIVE CACHE SWEEP: resize the KV pool WITHOUT restarting the server,
# and measure what each size costs in speed.
#
# WHY THIS EXISTS. Every other sweep in this study pays a full model load per
# arm. FreeToken exposes POST /v1/cache/rebuild ("live pool resizing without a
# restart", `ft ctl cache --moe N --kv N --mamba N --swa N`), so a pool sweep
# that would be N boots becomes ONE boot and N rebuilds. On an arm whose load is
# the most expensive in this study, that is the difference between an overnight
# queue and a coffee break.
#
# WHAT IS MEASURED, per pool size:
#   1. rebuild wall time      - what a live resize actually costs
#   2. the geometry the engine reports afterwards (it rounds to page units, so
#      the requested size is NOT necessarily the size you get - we record both)
#   3. prefill and decode on a fixed prompt, same prompt every time
#
# THE POINT IS THE SHAPE. Decode should be flat in KV size until the pool is too
# small to hold the working set, then fall off a cliff. Where that cliff sits is
# the useful number: it is the smallest pool that still serves this workload,
# and everything above it is VRAM you could have spent elsewhere.
#
# HONEST LIMIT: one server, one boot, sizes visited in one order. A rebuild is
# not a fresh allocator, so a size measured after a large one may fragment
# differently than the same size measured from cold. Findings here are leads for
# a booted confirmation, not replacements for one.
#
#   ./bench/ft_cache_sweep.sh --execute --plan-id FLASHNEXT-R1
#
# env: KV_SIZES  (default "262144 131072 65536 32768 16384") KV pool in TOKENS
#      SWEEP_ISL (default 8192)  SWEEP_OSL (default 256)  REPS (default 2)
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}
source bench/lib.sh

bench_init freetoken fn_smoke
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO/artifacts/ftcache_${STAMP}"
mkdir -p "$OUT"
KV_SIZES="${KV_SIZES:-262144 131072 65536 32768 16384}"
ISL="${SWEEP_ISL:-8192}"; OSL="${SWEEP_OSL:-256}"; REPS="${REPS:-2}"

cleanup(){ thermal_watchdog_stop 2>/dev/null||true; thermal_stop 2>/dev/null||true; engine_down 2>/dev/null||true; }
trap cleanup EXIT INT TERM
bench_lock
thermal_start "ftcache"; trap cleanup EXIT INT TERM
thermal_watchdog_start "$OUT"; trap cleanup EXIT INT TERM
settle_gpu
engine_up
preflight
record_provenance "$OUT"

curl -fsS --max-time 30 "$BASE/v1/cache/status" > "$OUT/geometry_boot.json" 2>/dev/null || true
log "boot geometry -> $OUT/geometry_boot.json"

# One prompt, built once, reused for every size: the pool is the only variable.
PROMPT_FILE="$OUT/prompt.json"
uv run python - "$ISL" "$OSL" "$PROMPT_FILE" <<'PY'
import json, sys, pathlib
isl, osl, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
# A REAL prompt from the same corpus the code arms use, not synthetic filler:
# repeated filler would be served largely from the prefix cache and would
# measure the cache rather than the pool (CORPUS-0, and the 31.8-67.2% repeated
# 8-gram corpus this study already disqualified once).
src = pathlib.Path("bench/data/lcb_code_8192.jsonl")
items = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
body = items[0].get("question") or items[0].get("prompt") or items[0].get("text")
words = body.split()
# Trim or repeat-with-offset to land near the requested length. Truncation keeps
# it a real document; padding, if needed, draws from LATER items rather than
# repeating this one.
approx = words[: int(isl * 0.75)]
i = 1
while len(approx) < int(isl * 0.75) and i < len(items):
    nxt = items[i].get("question") or items[i].get("prompt") or items[i].get("text") or ""
    approx += nxt.split(); i += 1
approx = approx[: int(isl * 0.75)]
pathlib.Path(out).write_text(json.dumps({
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": " ".join(approx)}],
    "max_tokens": osl, "temperature": 0.7, "top_p": 0.8,
    "chat_template_kwargs": {"enable_thinking": False}, "stream": False}))
print(f"  prompt ~{len(approx)} words -> {out}")
PY

echo "[" > "$OUT/sweep.json.tmp"; first=1
for kv in $KV_SIZES; do
  log "=== KV pool -> $kv tokens ==="
  r0=$(date +%s.%N)
  rc=0
  # Use the SHIPPED CLI, not a hand-built POST. /v1/cache/rebuild takes
  # num_pages, not tokens: the client reads the live geometry and converts with
  # the pool's own page size (control_cli._rebuild_body). Re-implementing that
  # here would silently request the wrong pool size the day the page size
  # changes. `ft ctl` runs inside the container, against its own server.
  docker exec "$CONTAINER" /opt/freetoken/bin/ft ctl --json --base-url "http://127.0.0.1:${FT_INNER_PORT:-8000}" \
    cache rebuild --kv "$kv" \
    > "$OUT/rebuild_${kv}.json" 2>>"$OUT/sweep.log" || rc=$?
  r1=$(date +%s.%N)
  rebuild_s=$(awk -v a="$r0" -v b="$r1" 'BEGIN{printf "%.2f", b-a}')
  status="$(uv run python -c "
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print('error'); raise SystemExit
print(d.get('status') or d.get('result') or ('ok' if d.get('geometry') else '?'))" "$OUT/rebuild_${kv}.json" 2>/dev/null || echo error)"
  log "  rebuild status=$status in ${rebuild_s}s (curl rc=$rc)"
  curl -fsS --max-time 30 "$BASE/v1/cache/status" > "$OUT/geometry_${kv}.json" 2>/dev/null || true
  uv run python -c "
import json,sys
try: g=json.load(open(sys.argv[1])).get('geometry') or {}
except Exception: raise SystemExit
pages, ps = g.get('num_pages') or 0, g.get('page_size') or 0
print(f\"  geometry now: kv {pages} pages x {ps} = {pages*ps} tokens, moe={g.get('moe_cache_size')}, mamba={g.get('num_mamba_slots')}\")" "$OUT/geometry_${kv}.json" 2>/dev/null | tee -a "$OUT/sweep.log" || true

  # Measure only if the resize actually happened. A failed rebuild leaves the
  # PREVIOUS pool in place, so measuring anyway would silently record the old
  # size's speed under the new size's label.
  ttft=""; dec=""; ok=0
  if [ "$status" = "ok" ] || [ "$status" = "completed" ] || [ "$status" = "success" ]; then
    for rep in $(seq 1 "$REPS"); do
      m=$(uv run python - "$BASE" "$PROMPT_FILE" "$kv-$rep" <<'PY'
import json, sys, time, urllib.request
base, pf, tag = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.load(open(pf))
# UNIQUE PREFIX PER REQUEST. Sending the identical prompt twice measures the
# prefix cache, not the pool: the first pass of this sweep read 53.3 s then
# 4.5 s for the same prompt at the same pool size. A marker at token 0 defeats
# the cache on every engine without needing an engine-specific flush endpoint -
# the same rule the context ladder uses.
body["messages"][0]["content"] = f"[run {tag}] " + body["messages"][0]["content"]
t0 = time.time()
req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r: j = json.load(r)
el = time.time() - t0
u = j.get("usage") or {}
comp = u.get("completion_tokens") or 0
print(f"{el:.3f} {comp} {u.get('prompt_tokens') or 0}")
PY
) || m=""
      [ -n "$m" ] && { ok=1; log "  rep $rep: $m (elapsed_s completion_tokens prompt_tokens)"; echo "$kv $rep $m" >> "$OUT/reps.txt"; }
    done
  else
    log "  SKIP measurement: rebuild did not report success"
  fi
  [ "$first" = 1 ] || echo "," >> "$OUT/sweep.json.tmp"; first=0
  printf '{"kv_tokens_requested":%s,"rebuild_s":%s,"rebuild_status":"%s","measured":%d}' \
    "$kv" "$rebuild_s" "$status" "$ok" >> "$OUT/sweep.json.tmp"
done
echo "]" >> "$OUT/sweep.json.tmp"; mv "$OUT/sweep.json.tmp" "$OUT/sweep.json"

record_runtime_state "$OUT" 2>/dev/null || true
docker logs "$CONTAINER" > "$OUT/server.log.after" 2>&1 || true
thermal_watchdog_stop; thermal_stop "$OUT"
log "done -> $OUT"
