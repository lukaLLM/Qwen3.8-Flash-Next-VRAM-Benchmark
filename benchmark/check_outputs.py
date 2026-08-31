# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Did a benchmark run actually produce useful work, or just tokens?

Every speed number in this study assumes the model was doing the workload. That
assumption broke once already and nothing caught it: with thinking disabled, the
model answered coding prompts with `tool_call` blocks and stopped, because there
is no tool executor to answer them. Turns collapsed to ~94 tokens each and the
run measured how fast the model emits tool-call syntax. The numbers looked
perfectly reasonable - 158 tok/s - and were meaningless.

So speed is not enough. This reads the generated text and answers: is this run
worth reporting?

Flags, and why each one matters:

  tool_call        the model asked to use tools instead of writing code. With no
                   executor the conversation cannot progress.
  too_short        a coding turn producing under ~150 tokens is a stub.
  uniform_length   near-identical lengths across turns means the model is
                   emitting the same boilerplate every turn, not solving tasks.
  no_code          a coding workload with no fenced code block produced no code.
  repetition       one line repeated many times is a degenerate loop.
  empty            no visible content at all.

  ./benchmark/check_outputs.py <bench_results_dir> [--expect-code] [--json out]
"""
from __future__ import annotations

import argparse, json, re, statistics as st, sys
from pathlib import Path


def load_turns(d: Path) -> list[dict]:
    """Prefer the per-turn response files; fall back to the transcript."""
    turns = []
    rd = d / "responses"
    if rd.exists():
        for f in sorted(rd.glob("*.md")):
            body = f.read_text()
            body = body.split("\n\n", 1)[1] if "\n\n" in body else body
            turns.append({"id": f.stem, "text": body})
    if not turns:
        t = d / "transcript.json"
        if t.exists():
            msgs = json.loads(t.read_text()).get("messages", [])
            for i, m in enumerate(msgs):
                if m.get("role") == "assistant":
                    turns.append({"id": f"m{i}", "text": m.get("content") or ""})
    stats = d / "stats.json"
    if stats.exists():
        rows = {r.get("index"): r for r in json.loads(stats.read_text()).get("rows", [])}
        for i, t in enumerate(turns, start=1):
            r = rows.get(i) or {}
            t["completion_tokens"] = r.get("completion_tokens")
            t["error"] = r.get("error")
    return turns


def repetition_ratio(text: str) -> float:
    """Fraction of lines that are duplicates. 0 = all unique."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 6:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def analyse(turns: list[dict], expect_code: bool, turns_run: int = 0) -> dict:
    per, flags = [], set()
    for t in turns:
        txt = t["text"]
        has_tool = bool(re.search(r"<tool_call>|<function=|\"tool_calls\"", txt))
        has_code = bool(re.search(r"```", txt))
        rep = repetition_ratio(txt)
        per.append({"id": t["id"], "chars": len(txt),
                    "tokens": t.get("completion_tokens"),
                    "tool_call": has_tool, "code_block": has_code,
                    "repetition": round(rep, 2), "error": t.get("error")})
        if has_tool:
            flags.add("tool_call")
        if rep > 0.5:
            flags.add("repetition")
        if not txt.strip():
            flags.add("empty")

    # A verdict based on a fraction of the turns is much weaker than one based on
    # all of them, and saying only "USABLE" hides that. Turns go missing two ways:
    # truncated at the output cap, or finished with reasoning but no visible
    # content - the harness saves neither.
    if turns_run and len(per) < turns_run:
        flags.add("partial_coverage")

    toks = [p["tokens"] for p in per if p["tokens"]]
    if toks:
        med = st.median(toks)
        if med < 150:
            flags.add("too_short")
        if len(toks) > 2:
            mean = st.mean(toks)
            # near-identical lengths every turn = boilerplate, not problem solving
            if mean and st.pstdev(toks) / mean < 0.08:
                flags.add("uniform_length")
    if expect_code and per and not any(p["code_block"] for p in per):
        flags.add("no_code")

    return {"turns": per, "flags": sorted(flags),
            "turns_run": turns_run, "turns_seen": len(per),
            "median_tokens": st.median(toks) if toks else None,
            "total_tokens": sum(toks) if toks else 0,
            "tool_call_turns": sum(1 for p in per if p["tool_call"]),
            "code_turns": sum(1 for p in per if p["code_block"]),
            "verdict": "DEGENERATE" if flags else "USABLE"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--expect-code", action="store_true",
                    help="the workload is a coding task, so a run with no code block is degenerate")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    d = Path(a.dir)
    turns = load_turns(d)
    if not turns:
        print(f"  no generated text found under {d}")
        return 2
    # how many turns the harness actually ran, from stats.json
    turns_run = 0
    st = d / "stats.json"
    if st.exists():
        turns_run = len([r for r in json.loads(st.read_text()).get("rows", []) if not r.get("error")])
    res = analyse(turns, a.expect_code, turns_run)

    print(f"  {'turn':<6}{'tokens':>8}{'chars':>8}{'tool':>6}{'code':>6}{'rep':>6}")
    for p in res["turns"]:
        print(f"  {p['id']:<6}{str(p['tokens'] or '-'):>8}{p['chars']:>8}"
              f"{'YES' if p['tool_call'] else '-':>6}{'yes' if p['code_block'] else '-':>6}"
              f"{p['repetition']:>6}")
    print(f"\n  median {res['median_tokens']} tok/turn | total {res['total_tokens']} | "
          f"{res['tool_call_turns']} turns used tool calls | {res['code_turns']} contained code")
    if res["turns_run"]:
        print(f"  coverage: {res['turns_seen']} of {res['turns_run']} turns had saved text")

    if res["verdict"] == "USABLE":
        print("  VERDICT: USABLE - the model did the workload.")
    else:
        print(f"  VERDICT: DEGENERATE - {', '.join(res['flags'])}")
        why = {
            "tool_call": "asked for tools instead of writing code; nothing answers them",
            "too_short": "turns are stubs, not solutions",
            "uniform_length": "same length every turn - boilerplate, not problem solving",
            "no_code": "a coding workload that produced no code block",
            "partial_coverage": (f"only {res['turns_seen']} of {res['turns_run']} turns "
                                 "had saved text - the rest were truncated at the cap or "
                                 "produced no visible content, so this verdict rests on a "
                                 "fraction of the run"),
            "repetition": "a line repeats many times - degenerate loop",
            "empty": "no visible content",
        }
        for f in res["flags"]:
            print(f"    - {f}: {why.get(f,'')}")
        # partial_coverage alone does NOT invalidate a rate measurement: decode
        # tok/s comes from completion_tokens and predicted_ms in stats.json,
        # which exist for every turn whether or not its text was saved. Say so,
        # rather than implying the speed number is wrong.
        if res["flags"] == ["partial_coverage"]:
            print("  The RATE is still valid - it is computed from per-turn timings,")
            print("  not from the saved text. What is weak here is the CONTENT")
            print("  verification: it rests on a fraction of the turns.")
        else:
            print("  Speed numbers from this run describe the wrong activity.")

    # a short excerpt so a human can eyeball it
    first = turns[0]["text"].strip().replace("\n", " ")[:200]
    print(f"\n  first turn excerpt: {first!r}")

    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
    return 0 if res["verdict"] == "USABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
