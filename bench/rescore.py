#!/usr/bin/env python3
"""Recompute accuracy from saved responses, without re-running any model.

Every item's raw response text is kept, so a scorer bug is a re-score rather
than a re-run. Rewrites summary.json in place and prints the before/after.
"""
import json, math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from quality_run import score_gsm8k, score_math, extract_gsm8k

def wilson(k, n):
    if not n: return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (round(c-h, 6), round(c+h, 6))

for res in sorted(pathlib.Path("artifacts").glob("quality_*/*/results.jsonl")):
    task = res.parent.name
    rows = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]
    scored = [r for r in rows if r.get("error") is None]
    n_ok = 0
    for r in scored:
        t = r.get("response") or ""
        ok = score_math(t, r["gold"]) if task == "math500" else score_gsm8k(t, r["gold"])
        r["correct"] = ok
        if task == "gsm8k": r["extracted"] = extract_gsm8k(t)
        n_ok += ok
    sf = res.parent / "summary.json"
    if not sf.is_file(): continue
    j = json.loads(sf.read_text()); before = j.get("accuracy")
    acc = n_ok / len(scored) if scored else 0.0
    j.update(accuracy=round(acc, 6), n_correct=n_ok, n_scored=len(scored),
             ci95_wilson=list(wilson(n_ok, len(scored))), rescored=True)
    sf.write_text(json.dumps(j, indent=2))
    res.write_text("\n".join(json.dumps(r) for r in rows))
    arm = res.parent.parent.name.split("_")[1]
    print(f"  {arm:10} {task:8} {before:.4f} -> {acc:.4f}  ({n_ok}/{len(scored)})")
