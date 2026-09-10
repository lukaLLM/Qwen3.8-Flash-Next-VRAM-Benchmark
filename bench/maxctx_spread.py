#!/usr/bin/env python3
"""Full-window arms with their SPREAD, not a point estimate.

Exists because of B-31: a byte-identical re-run of the published SGLang
full-window arm moved decode 126.91 -> 154.03 (+21.4%) while TTFT and prefill
reproduced within 2.5%. Decode there is NEXTN speculative decoding, whose rate
depends on acceptance on the generated text, and fn_maxctx.conf ran
CONTEXT_REQUESTS=1 - so both published figures are one sample each.

So this prints min/median/max per arm and never a bare mean. A single-request
cell is shown with n=1 and no spread, which is the honest rendering of what the
published bars are.

    python3 bench/maxctx_spread.py [glob]        default: *maxctx*sglang*
"""
import json, pathlib, statistics, sys

A = pathlib.Path(__file__).resolve().parent.parent / "artifacts"
pat = sys.argv[1] if len(sys.argv) > 1 else "*maxctx*sglang*"

def arm_label(pf):
    knobs = [f"{k.replace('sglang_','')}={v}" for k, v in pf.items()
             if k.startswith("sglang_")
             and v not in ("engine-default", "off", "none", "not-applicable", None)]
    return ", ".join(knobs) or "control"

rows = {}
for d in sorted(A.glob(pat)):
    pj = d / "provenance.json"
    if not pj.exists():
        continue
    pf = json.load(open(pj)).get("parity_flags", {})
    for p in d.rglob("profile_export_aiperf.json"):
        if "phases" in p.parts or p.parent.name != "isl_261504":
            continue
        j = json.loads(p.read_text())
        g = lambda k: (j.get(k) or {}).get("avg")
        isl, t = g("input_sequence_length"), g("time_to_first_token")
        if not (isl and t):
            continue
        n = int((g("request_count") or 1))
        # Acceptance, if context.sh captured it (added 2026-09-10). Under NEXTN
        # this is what sets the decode rate, so it is the first thing to look at
        # when two identical configs disagree on decode (B-31).
        acc = None
        try:
            cr = json.loads((d / "context_results.json").read_text())
            for r in cr.get("ladder", []):
                a = (r.get("acceptance") or {})
                for k, v in a.items():
                    if "accept" in k:
                        acc = v
                        break
        except Exception:
            pass
        rows.setdefault(arm_label(pf), []).append(
            (d.name[-16:], n, t / 1000, isl / (t / 1000),
             g("output_token_throughput_per_user"), acc))

if not rows:
    sys.exit("no full-window cells found")

def fmt(vals):
    if len(vals) == 1:
        return f"{vals[0]:8.1f}  (n=1, no spread)"
    lo, hi, med = min(vals), max(vals), statistics.median(vals)
    return f"{med:8.1f}  [{lo:.1f} - {hi:.1f}]  spread {100*(hi/lo-1):5.1f}%"

for arm, cells in rows.items():
    print(f"\n=== {arm}   ({len(cells)} boot{'s' if len(cells)!=1 else ''}) ===")
    for stamp, n, ttft, pre, dec, acc in cells:
        a = f"  accept={acc:.3f}" if acc is not None else "  accept=n/a"
        print(f"    {stamp}  reqs={n}  ttft={ttft:6.1f}s  prefill={pre:7.0f}  decode={dec:7.2f}{a}")
    if len(cells) > 1:
        print(f"    TTFT s   {fmt([c[2] for c in cells])}")
        print(f"    prefill  {fmt([c[3] for c in cells])}")
        print(f"    decode   {fmt([c[4] for c in cells])}")

ctl = rows.get("control")
oth = {k: v for k, v in rows.items() if k != "control"}
if ctl and oth and len(ctl) > 1:
    cp = statistics.median([c[3] for c in ctl])
    cd = statistics.median([c[4] for c in ctl])
    cs = max(c[3] for c in ctl) / min(c[3] for c in ctl) - 1
    print(f"\n=== vs control (median prefill {cp:.0f}, decode {cd:.2f}) ===")
    for arm, cells in oth.items():
        ap = statistics.median([c[3] for c in cells])
        ad = statistics.median([c[4] for c in cells])
        dp = 100 * (ap / cp - 1)
        verdict = "INSIDE the control's own spread" if abs(dp) <= 100 * cs else "outside control spread"
        print(f"    {arm}\n      prefill {dp:+6.1f}%   decode {100*(ad/cd-1):+6.1f}%   -> {verdict}")
