#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Is the 48 GiB prefill cliff VRAM, or is it the load mode?
#
# The ladder found prefill jumping 292 -> 747 t/s between the 32 and 48 GiB
# tiers, but 8-32 GiB ran mmap and 48 GiB ran `--load-mode none`, because the
# load-mode heuristic keys off predicted host RAM and crossed its threshold at
# that exact tier. Two variables moved together.
#
# WHY THE EARLIER ARGUMENT FOR "IT MUST BE VRAM" DOES NOT WORK.
# It cited T2's +5% decode for load mode. T2 ran at `--n-cpu-moe 0`, where load
# mode only decides where the PLE table lives. At `--n-cpu-moe 23` it decides
# whether 23 expert layers sit in reclaimable page cache or anonymous resident
# memory. Different mechanism, different scale; the 5% does not transfer.
#
# WHY COLD AND WARM.
# A cold mmap arm measures the disk as much as the load mode. The asymmetry
# matters:
#   - if mmap comes out FAST, the VRAM conclusion is safe either way, and a cold
#     cache only makes it stronger;
#   - if mmap comes out SLOW, cold-vs-warm is the whole question, and one arm
#     cannot tell load mode from page-cache misses.
# So both are run. The cold->warm delta is itself a reportable number.
#
# Tensor placement is identical across all three arms: --n-cpu-moe 23,
# ctx 262144, same quant, same -ub. Only the load path differs.
# -----------------------------------------------------------------------------
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=results/mmap_control; mkdir -p "$OUT"

[ -z "${SKIP_LOCK:-}" ] && echo "waiting for other GPU work (blocking on the shared lock) ..."
if [ -z "${SKIP_LOCK:-}" ]; then
  exec 9>/tmp/queue_rest.lock
  flock 9
fi
[ -z "${SKIP_LOCK:-}" ] && echo "lock acquired $(date '+%F %H:%M:%S')"

source benchmark/gpu_settle.sh
./scripts/serve.sh --down >/dev/null 2>&1; sleep 2

# Pass 1 - cold. Drop what we can of the page cache first so "cold" means it.
# Requires no privileges for the posix_fadvise path; if the sysctl is not
# writable we say so rather than silently measuring a warm cache.
if sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
  echo "  page cache dropped - pass 1 is genuinely cold"
  COLD_STATE=cold
else
  echo "  NOTE: could not drop page cache (needs sudo). Pass 1 is 'as-found',"
  echo "        not guaranteed cold. The cold/warm delta is a lower bound."
  COLD_STATE=as-found
fi
echo "  buff/cache before pass 1: $(free -g | awk '/^Mem/{print $6}')G"

echo "  ===== pass 1: mmap, $COLD_STATE ====="
OUT=results/mmap_control_p1 TIERS="48:262144:23:256,2048,8192,32768:65536,131072" \
FORCE_MODE=mmap ./benchmark/tier_full.sh 2>&1 | tee "$OUT/pass1.log" \
  | grep -E "tok \| prefill|started:|verdict|FAIL" | sed 's/^/    /'

echo "  buff/cache after pass 1: $(free -g | awk '/^Mem/{print $6}')G"
echo "  ===== pass 2: mmap, warmed by pass 1 ====="
OUT=results/mmap_control_p2 TIERS="48:262144:23:256,2048,8192,32768:65536,131072" \
FORCE_MODE=mmap ./benchmark/tier_full.sh 2>&1 | tee "$OUT/pass2.log" \
  | grep -E "tok \| prefill|started:|verdict|FAIL" | sed 's/^/    /'

echo
echo "########## resident none  vs  mmap cold  vs  mmap warm ##########"
echo "########## 48 GiB, --n-cpu-moe 23, ctx 262144, identical placement ##########"
python3 - <<'PY'
import csv, pathlib
def load(path):
    p = pathlib.Path(path); out = {}
    if not p.exists(): return out
    for r in csv.DictReader(p.open()):
        if r["cap"] == "48":
            out[int(r["length"])] = float(r["prefill_tps"])
    return out
none = load("results/tier_full/full.csv")
cold = load("results/mmap_control_p1/full.csv")
warm = load("results/mmap_control_p2/full.csv")
if not cold and not warm:
    print("  no control rows - check the pass logs"); raise SystemExit
ks = sorted(set(none) & (set(cold) | set(warm)))
print(f"  {'len':>7} {'none':>9} {'cold':>9} {'warm':>9}   {'warm/none':>9}")
for L in ks:
    c = cold.get(L); w = warm.get(L); n = none[L]
    print(f"  {L:>7} {n:>9.1f} {(f'{c:.1f}' if c else '-'):>9} "
          f"{(f'{w:.1f}' if w else '-'):>9}   {(f'{w/n:.2f}x' if w else '-'):>9}")
sh = [L for L in ks if L in warm and L <= 8192]
if sh:
    r = sum(warm[L]/none[L] for L in sh)/len(sh)
    print(f"\n  warm mmap / resident none, short prompts: {r:.2f}x")
    if r > 0.85:
        print("  -> load mode is NOT the cliff. Prefill really does escape the")
        print("     bus between 32 and 48 GiB. Headline stands.")
    elif r < 0.60:
        print("  -> LOAD MODE explains most of the jump. The 'prefill escapes the")
        print("     bus' claim is NOT supported and must be rewritten.")
    else:
        print("  -> both contribute. Neither single-cause story is honest;")
        print("     report the split.")
    if sh and all(L in cold for L in sh):
        cr = sum(warm[L]/cold[L] for L in sh)/len(sh)
        print(f"  warm/cold = {cr:.2f}x  (page-cache effect on the mmap path alone)")
PY
./scripts/serve.sh --down >/dev/null 2>&1
echo "########## CONTROL DONE $(date '+%H:%M:%S') ##########"
