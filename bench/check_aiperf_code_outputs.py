#!/usr/bin/env python3
"""Reject real-code speed runs whose saved responses did not do code work."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def repetition_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    args = ap.parse_args()

    # A SPEED run pins output length with ignore_eos, which forbids the model to
    # stop. Forced past its natural end it emits filler - empty turn markers, or
    # the answer restarted - and this checker then scores that filler as a
    # repetition loop. That is the harness's artifact, not the model's output, so
    # judging such a run is meaningless: report NOT_APPLICABLE rather than a
    # verdict. (2026-09-04: this scored 5 responses as degenerate and nearly put
    # an engine-quality claim in a published report.)
    forced = False
    for probe in ("run.log", "provenance.json"):
        f = args.artifact / probe
        if f.is_file() and "ignore_eos:true" in f.read_text(errors="ignore"):
            forced = True
            break

    report: dict = {"artifact": str(args.artifact), "cells": {}, "flags": [],
                    "forced_generation": forced}
    total = code = tools = empty = repetitive = 0
    # A CONCURRENCY SWEEP writes concurrency_<n>/outputs.json; a SINGLE-cell run
    # writes outputs.json at the top level. Globbing only the former silently
    # scored every single-concurrency run as 0 responses and stamped it
    # DEGENERATE - which is how three 32K real-code runs (71 KB of saved output
    # each) were reported as degenerate on 2026-09-04 without ever being read.
    # A checker that cannot find the output must say so, not return "no code".
    files = sorted(args.artifact.glob("concurrency_*/outputs.json"))
    if not files and (args.artifact / "outputs.json").is_file():
        files = [args.artifact / "outputs.json"]
    if not files:
        report["flags"].append("missing_outputs_json")

    for path in files:
        rows = json.loads(path.read_text()).get("data", [])
        cell = {"responses": len(rows), "code_responses": 0, "tool_calls": 0,
                "empty": 0, "repetitive": 0}
        for row in rows:
            text = row.get("response_text") or ""
            has_code = bool(re.search(r"```(?:python)?|\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[:(]", text))
            has_tool = bool(re.search(r"<tool_call>|<function=|\"tool_calls\"", text))
            is_empty = not text.strip()
            is_repetitive = repetition_ratio(text) > 0.5
            total += 1
            code += has_code
            tools += has_tool
            empty += is_empty
            repetitive += is_repetitive
            cell["code_responses"] += has_code
            cell["tool_calls"] += has_tool
            cell["empty"] += is_empty
            cell["repetitive"] += is_repetitive
        report["cells"][path.parent.name] = cell

    report.update({"responses": total, "code_responses": code, "tool_calls": tools,
                   "empty": empty, "repetitive": repetitive,
                   "code_fraction": (code / total if total else 0.0)})
    if empty:
        report["flags"].append("empty_response")
    if tools:
        report["flags"].append("unhandled_tool_call")
    if repetitive:
        report["flags"].append("repetition_loop")
    if total and code / total < 0.5:
        report["flags"].append("less_than_half_contain_code")
    if forced:
        # The run cannot be judged: ignore_eos forced generation past the model's
        # own stop, so every "repetition loop" here may be filler the harness
        # demanded. Counts are kept for inspection; the verdict is withheld.
        report["flags"].append("forced_generation_ignore_eos")
        report["verdict"] = "NOT_APPLICABLE"
    else:
        report["verdict"] = "USABLE" if not report["flags"] else "DEGENERATE"

    out = args.artifact / "output_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] in ("USABLE", "NOT_APPLICABLE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
