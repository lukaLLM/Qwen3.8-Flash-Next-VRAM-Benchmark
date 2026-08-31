#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Slot-reuse correctness AT LONG CONTEXT.
#
# The queue's step 5 probes slot reuse at 32768 tokens. That is the wrong
# regime: the upstream report describes recall errors at LONG context after a
# slot has been used and cleared, so a clean result at 32K says very little
# about the numbers this study actually publishes - 131072 and 245760.
#
# Every long-context and multi-turn figure in FINDINGS depends on this coming
# back clean, so it is run where the claims live.
#
#   --parallel 1 deliberately. Slot reuse still happens with one slot (it is
#   used, cleared, used again), and four slots at 245760 tokens each would need
#   ~1M of context, which does not fit. The single-slot case is also the one
#   most users actually hit.
#
#   ctx 262144, --load-mode none, uncapped card: the fastest configuration
#   available, because each 245760-token probe costs ~4 min of prefill and each
#   cell needs three of them (fresh, interferer, reused).
#
#   --max-tokens 4096: a thinking model can spend the whole budget reasoning and
#   never reach the answer. That would hit fresh and reused equally, so the
#   paired comparison survives, but a run where everything fails proves nothing.
#
# GUARD: if the 32K run from the queue scored zero on fresh probes, the harness
# is broken rather than the engine, and this exits instead of burning 2.5 hours.
# -----------------------------------------------------------------------------
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=results/slot_reuse; mkdir -p "$OUT"

[ -z "${SKIP_LOCK:-}" ] && echo "waiting for other GPU work (blocking on the shared lock) ..."
if [ -z "${SKIP_LOCK:-}" ]; then
  exec 9>/tmp/queue_rest.lock
  flock 9
fi
[ -z "${SKIP_LOCK:-}" ] && echo "lock acquired $(date '+%F %H:%M:%S')"

# --- guard on the short run ---------------------------------------------------
python3 - <<'PY' || { echo "ABORTING long-context slot-reuse run."; exit 1; }
import json, pathlib, sys
p = pathlib.Path("results/slot_reuse/p1.json")
if not p.exists():
    print("  no 32K result yet (results/slot_reuse/p1.json missing) - "
          "cannot confirm the harness works. Refusing to spend 2.5h.")
    sys.exit(1)
d = json.load(p.open())
print(f"  32K baseline: fresh {d['fresh']}/{d['n']}, reused {d['reused']}/{d['n']}")
if d["fresh"] == 0:
    print("  fresh probes scored ZERO at 32K. That is a broken harness or a "
          "model that ignores context, not a slot bug. Fix before scaling up.")
    sys.exit(1)
print("  harness demonstrably works at 32K - proceeding to long context.")
PY

source benchmark/gpu_settle.sh
source benchmark/dump_compose.sh
./scripts/serve.sh --down >/dev/null 2>&1; sleep 2
gpu_settle

# Do NOT discard stderr here. The first attempt at this run died on
# "unknown option: --lazy" and reported only "server failed to start", which
# cost the whole stage. serve.sh now accepts --lazy, and its output is kept.
# Record the resolved compose - same variables as the serve call below.
LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda13 CTX=262144 \
LOAD_MODE=none LAZY=off PARALLEL=1 \
  dump_compose "$OUT" || echo "  WARN: compose.txt not recorded"

LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda13 \
LOAD_MODE=none LAZY=off PARALLEL=1 \
  ./scripts/serve.sh --quant UD-IQ4_XS --ctx 262144 --load-mode none --lazy off \
  2>&1 | tail -5 | sed 's/^/    /'
if ! curl -fsS -m 10 http://localhost:8000/health >/dev/null 2>&1; then
  echo "  server did not come up healthy - aborting"
  docker logs --tail 30 "$(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1)" 2>&1 | tail -20 | sed 's/^/    /'
  exit 1
fi

uv run benchmark/slot_reuse.py --selftest 2>&1 | tail -1 | sed 's/^/  /'

for TOK in ${TOKENS_LIST:-131072 245760}; do
  echo "  ===== $TOK tokens, depths 10/50/90, 3 tasks, 1 repeat ====="
  date '+  start %H:%M:%S'
  uv run benchmark/slot_reuse.py --tokens $TOK --depths 10,50,90 --repeats 1 \
    --max-tokens 4096 --ctx 262144 --out "$OUT/long_${TOK}.json" 2>&1 | sed 's/^/    /'
  date '+  end   %H:%M:%S'
done

./scripts/serve.sh --down >/dev/null 2>&1
echo
echo "########## SLOT REUSE ACROSS CONTEXT ##########"
python3 - <<'PY'
import json, pathlib
rows = []
for name, f in (("32768 (p1)", "p1.json"), ("32768 (p4)", "p4.json"),
                ("131072", "long_131072.json"), ("245760", "long_245760.json")):
    p = pathlib.Path("results/slot_reuse") / f
    if p.exists():
        d = json.load(p.open()); rows.append((name, d["fresh"], d["reused"], d["n"]))
if not rows:
    print("  no results"); raise SystemExit
print(f"  {'context':>12} {'fresh':>9} {'reused':>9}")
gap = False
for n, fr, re_, tot in rows:
    print(f"  {n:>12} {fr:>4}/{tot:<4} {re_:>4}/{tot:<4}" + ("   <-- GAP" if re_ < fr else ""))
    if re_ < fr: gap = True
print()
want = {"32768 (p1)", "32768 (p4)", "131072", "245760"}
missing = sorted(want - {r[0] for r in rows})
if missing:
    print(f"  INCOMPLETE - no data for: {', '.join(missing)}")
if gap:
    print("  -> RECALL DEGRADES AFTER SLOT REUSE. Every long-context and multi-turn")
    print("     number in FINDINGS needs an asterisk, and this is a publishable bug.")
elif missing:
    print("  -> No gap in the contexts that DID run, but coverage is incomplete.")
    print("     The asterisk stays on until the missing contexts are measured -")
    print("     a clean result at 131072 says nothing about 245760.")
else:
    print("  -> No fresh/reused gap at any context. The asterisk comes off the")
    print("     long-context numbers - the graders were shown able to fail.")
PY
echo "########## DONE $(date '+%H:%M:%S') ##########"
