# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27"]
# ///
"""Does recall break after a server slot is reused?

An upstream report on the merged qwen4exp code describes exact long-context
recall on a fresh server, but occasional digit errors after a slot has been used
and cleared. It was reported on Vulkan, so it is not proof of a CUDA bug here -
but it is a SILENT WRONG ANSWER mode, and until it is ruled out every
long-context and multi-turn number in FINDINGS.md carries an asterisk.

The design is a before-and-after comparison, not a pass rate:

  before  a probe before a DIFFERENT large request
  after   the same probe after that large request occupied and released a slot

The server does not restart before each "before" probe. Thus, only the first
probe can be fresh. The JSON keeps the historical `fresh` and `reused` keys for
compatibility. Do not describe the old `fresh` key as a clean server state.

THREE GRADERS, because one friendly needle proves little:

  code           a unique 8-char token at a given depth -> exact match
  contradiction  two conflicting values, early and late -> must return the LAST
  date           several plausible dates, one tied to a unique event

The contradiction and date tasks carry deliberate distractors. A grader that
cannot fail proves nothing, so --selftest feeds deliberately wrong answers and
asserts they are marked wrong.

  ./benchmark/slot_reuse.py --selftest
  ./benchmark/slot_reuse.py --tokens 32768 --depths 10,50,90 --repeats 3
"""
from __future__ import annotations

import argparse, json, random, re, sys, time
from pathlib import Path
import httpx

FILLER = [
    "sensor {i} reported nominal drift of {a}.{b} units at cycle {c}",
    "batch {i} completed with {a} records merged and {b} deferred",
    "node {i} rotated its lease after {a} seconds, quorum {b}",
    "cache {i} evicted {a} entries, hit ratio {b}.{c} percent",
]

def filler_line(i: int, rng: random.Random) -> str:
    return FILLER[i % len(FILLER)].format(
        i=i, a=rng.randint(10, 9999), b=rng.randint(10, 99), c=rng.randint(100, 999))

def count_tokens(c: httpx.Client, url: str, text: str) -> int:
    r = c.post(f"{url}/tokenize", json={"content": text}, timeout=120)
    r.raise_for_status()
    return len(r.json().get("tokens", []))

def build(c: httpx.Client, url: str, target: int, budget: int,
          seed: int = 11) -> tuple[list[str], int]:
    """Filler lines whose joined length is close to `target` but never over
    `budget`.

    The first version of this estimated tokens-per-line from 200 sample lines
    and added 50 lines of slack. The filler templates vary in length, so a few
    percent of estimation error on a 245,760-token target consumed the entire
    12,288-token headroom under n_ctx and the server answered 400. Estimation is
    not good enough at this scale, so the assembled text is measured and the
    line count corrected until it actually fits."""
    want = min(target, budget)
    n = int(want / (count_tokens(c, url, "\n".join(
        filler_line(i, random.Random(seed)) for i in range(200))) / 200))
    lines, tok = [], 0
    for _ in range(5):
        rng = random.Random(seed)
        lines = [filler_line(i, rng) for i in range(n)]
        tok = count_tokens(c, url, "\n".join(lines))
        if 0.97 * want <= tok <= budget:
            return lines, tok
        n = max(1, int(n * want / max(tok, 1)))
        if tok > budget:            # overshoot: undershoot deliberately next time
            n = int(n * 0.98)
    return lines, tok

def make_probe(kind: str, lines: list[str], depth_pct: int, rng: random.Random) -> tuple[str, str, str]:
    """(prompt, question, expected)"""
    body = list(lines)
    at = max(1, int(len(body) * depth_pct / 100))
    if kind == "code":
        exp = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
        body.insert(at, f"*** the access code for vault seven is {exp} ***")
        q = "What is the access code for vault seven? Reply with the code only."
    elif kind == "contradiction":
        old, exp = rng.randint(100, 499), rng.randint(500, 999)
        body.insert(max(1, at // 3), f"*** the pressure limit is {old} kPa ***")
        body.insert(at, f"*** correction: the pressure limit is now {exp} kPa ***")
        q = ("The pressure limit is stated twice and later corrected. "
             "What is the CORRECTED limit in kPa? Reply with the number only.")
        exp = str(exp)
    else:  # date
        exp = f"{rng.randint(2019,2024)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        for _ in range(3):  # distractors, other events on other dates
            d = f"{rng.randint(2019,2024)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
            body.insert(rng.randint(1, len(body) - 1), f"*** the audit was filed on {d} ***")
        body.insert(at, f"*** the kepler migration completed on {exp} ***")
        q = "On what date did the kepler migration complete? Reply YYYY-MM-DD only."
    return "\n".join(body), q, exp

def grade(kind: str, answer: str, expected: str) -> bool:
    a = (answer or "").strip()
    if kind == "code":
        return expected.upper() in a.upper()
    if kind == "contradiction":
        nums = re.findall(r"\d+", a)
        return bool(nums) and expected in nums[:3]
    m = re.search(r"\d{4}-\d{2}-\d{2}", a)
    return bool(m) and m.group(0) == expected

def ask(c: httpx.Client, url: str, prompt: str, q: str, max_tokens: int = 2048) -> str:
    r = c.post(f"{url}/v1/chat/completions", json={
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": f"{prompt}\n\n{q}"}],
        "max_tokens": max_tokens, "temperature": 0, "top_k": 1, "cache_prompt": False,
    }, timeout=3600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""

def selftest() -> int:
    ok = True
    for kind, good, bad in (("code", "ABCD2345", "ZZZZ9999"),
                            ("contradiction", "The limit is 742 kPa", "The limit is 123 kPa"),
                            ("date", "It was 2021-03-04", "It was 1999-01-01")):
        exp = {"code": "ABCD2345", "contradiction": "742", "date": "2021-03-04"}[kind]
        if not grade(kind, good, exp): print(f"  {kind}: FAILED to accept a correct answer"); ok = False
        if grade(kind, bad, exp):     print(f"  {kind}: accepted a WRONG answer"); ok = False
        if grade(kind, "", exp):      print(f"  {kind}: accepted an EMPTY answer"); ok = False
    print("  graders can both pass and fail correctly" if ok else "  GRADER SELFTEST FAILED")
    return 0 if ok else 1

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--depths", default="10,50,90")
    ap.add_argument("--repeats", type=int, default=3)
    # A thinking model can spend the whole budget reasoning and never emit the
    # answer. That failure hits fresh and reused equally so the paired result
    # stays valid, but if it hits everything the run proves nothing - so the
    # budget is raised for long-context runs rather than left at the default.
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--ctx", type=int, default=262144,
                    help="server n_ctx; the prompt is sized to fit under it")
    ap.add_argument("--out", default="results/slot_reuse/slot_reuse.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    rng = random.Random(5)
    rows = []
    with httpx.Client() as c:
        # leave room for the generation and the chat template on top of the
        # prompt; overrunning n_ctx is a 400, not a graceful truncation
        budget = a.ctx - a.max_tokens - 512
        lines, built = build(c, a.url, a.tokens, budget)
        print(f"  target {a.tokens} tok, built {built} tok, "
              f"budget {budget} (ctx {a.ctx} - gen {a.max_tokens} - 512)")
        if built > budget:
            print("  FAILED to fit the prompt under n_ctx - aborting rather than "
                  "sending a request that will 400")
            return 1
        # the interferer: a different large request that occupies and frees the slot
        inter = "\n".join(lines[: int(len(lines) * 0.8)]) + "\n\nSummarise in one word."
        for kind in ("code", "contradiction", "date"):
            for depth in [int(x) for x in a.depths.split(",")]:
                prompt, q, exp = make_probe(kind, lines, depth, rng)
                for rep in range(a.repeats):
                    fresh = grade(kind, ask(c, a.url, prompt, q, a.max_tokens), exp)
                    ask(c, a.url, inter, "Summarise in one word.", 64)  # dirty the slot
                    reused = grade(kind, ask(c, a.url, prompt, q, a.max_tokens), exp)
                    rows.append({"task": kind, "depth": depth, "rep": rep,
                                 "fresh": fresh, "reused": reused})
                    print(f"  {kind:<14} depth {depth:>2}%  rep {rep}  "
                          f"fresh={'PASS' if fresh else 'FAIL'}  reused={'PASS' if reused else 'FAIL'}")
    f = sum(r["fresh"] for r in rows); u = sum(r["reused"] for r in rows); n = len(rows)
    print(f"\n  before interferer  {f}/{n}   after interferer {u}/{n}")
    print("  -> recall decreased after the direct interferer" if u < f else
          "  -> no before/after gap; this does not prove a clean-slot baseline" if n else "")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"rows": rows, "fresh": f, "reused": u, "n": n}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
