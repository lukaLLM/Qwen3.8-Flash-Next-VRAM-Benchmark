#!/usr/bin/env python3
# Run with: uv run bench/corpus_audit.py   (uses the project env, see pyproject.toml)
"""CORPUS-0: how much does an aiperf prompt repeat itself at long context?

WHY THIS GATE EXISTS

A speculative drafter's hit rate rises with repeated text. So a repetitive prompt
inflates acceptance, which inflates decode tok/s, and the engine gets the credit.
That is not hypothetical here: this repo's previous harness built long prompts by
CYCLING a 101-problem corpus of ~50,140 tokens, so a 131K prompt was 2.6x repeated
text and a 262K prompt 5.2x. The SGLang arm then measured 186.5 tok/s decode at
121K against 146.3 at 8.8K - faster on a longer prompt, which is not a thing that
happens for real.

aiperf should not have that problem. `sample_tokens_from_corpus` takes a CONTIGUOUS
slice at a random offset and only wraps past the end of the corpus, so a prompt
shorter than the corpus repeats nothing. But that is an argument, not a measurement,
and the two corpora are very different:

  sonnet  shakespeare.txt, ~1.3M tokens of natural prose. A 262K prompt is a fifth
          of it and cannot wrap.
  coding  a TEMPLATE-GENERATED pool of only ~500K tokens (coding_content.py:684).
          A 262K prompt eats over half the pool, and the pool is built by filling
          the same structural templates repeatedly - so it can carry heavy internal
          repetition WITHOUT ever wrapping.

This script measures it instead of arguing about it.

WHAT IS MEASURED

  repeated-8gram coverage - the fraction of token positions covered by an 8-gram
  that already appeared EARLIER IN THE SAME PROMPT. This is close to what an n-gram
  drafter actually exploits, and it is what an MTP head benefits from indirectly.

  distinct-8gram ratio    - unique 8-grams / total. A blunter view of the same thing.
  longest repeated span   - the worst single case, in tokens.

THE GATE: a corpus is usable at an ISL if coverage stays under 5%. Above that, a
speculative-decoding number measured on it is about the corpus, not the engine.

    uv run bench/corpus_audit.py
    uv run bench/corpus_audit.py --lengths 8192,262144 --repeats 1
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER = (Path.home() / ".cache/huggingface/hub"
                     / "models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
                     / "snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594")
GATE_PCT = 5.0
N = 8


def ngram_stats(toks: list[int], n: int = N) -> dict:
    """Repetition of `toks` against itself. One pass, one set."""
    total = len(toks)
    if total < n:
        return {"tokens": total, "repeated_8gram_pct": 0.0,
                "distinct_8gram_ratio": 1.0, "longest_repeat_tokens": 0}
    seen: set[tuple[int, ...]] = set()
    covered = bytearray(total)          # 1 per position inside a repeated gram
    longest = run = 0
    for i in range(total - n + 1):
        g = tuple(toks[i:i + n])
        if g in seen:
            for j in range(i, i + n):
                covered[j] = 1
            run += 1
            longest = max(longest, run + n - 1)
        else:
            seen.add(g)
            run = 0
    return {
        "tokens": total,
        "repeated_8gram_pct": round(100.0 * sum(covered) / total, 3),
        "distinct_8gram_ratio": round(len(seen) / (total - n + 1), 4),
        "longest_repeat_tokens": longest,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--lengths", default="8192,32768,131072,262144")
    ap.add_argument("--corpora", default="sonnet,coding")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42,
                    help="must match the workload conf's SEED (fn_*.conf use 42)")
    ap.add_argument("--out", default=str(REPO / "results/corpus_audit/corpus_audit.json"))
    a = ap.parse_args()

    # Do NOT set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE. They are presence-checked,
    # so even "=0" counts as offline, and offline mode routes aiperf through
    # snapshot_download(local_files_only=True), which rejects a local path as an
    # invalid repo id. Same trap as bench/lib.sh - see the note there. The
    # tokenizer is a local directory, so no hub call happens anyway; the short
    # timeouts just make a stray one fail fast instead of hanging.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from aiperf.common import random_generator as rng
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config.dataset.content import PromptConfig
    from aiperf.dataset.generator.coding_content import CodingContentGenerator
    from aiperf.dataset.generator.prompt import PromptGenerator

    # Seed the generator exactly as a real run does, so this audit describes the
    # prompts the benchmark will actually send, not a different random draw.
    rng.init(a.seed)
    tok = Tokenizer.from_pretrained(a.tokenizer)
    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    corpora = [c.strip() for c in a.corpora.split(",") if c.strip()]

    print(f"  tokenizer {Path(a.tokenizer).parent.parent.name}")
    print(f"  gate      repeated-8gram coverage < {GATE_PCT}%\n")
    print(f"  {'corpus':8} {'target':>8} {'actual':>8} {'rep8%':>8} {'distinct':>9} {'longest':>8}  verdict")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8}  -------")

    rows = []
    for corpus in corpora:
        if corpus == "coding":
            gen = CodingContentGenerator(config=PromptConfig(), tokenizer=tok)
        else:
            gen = PromptGenerator(prompts=PromptConfig(), prefix_prompts=None, tokenizer=tok)
        for L in lengths:
            for r in range(a.repeats):
                text = gen.generate_prompt(L)
                st = ngram_stats(tok.encode(text))
                ok = st["repeated_8gram_pct"] < GATE_PCT
                st |= {"corpus": corpus, "target_tokens": L, "rep": r, "pass": ok}
                rows.append(st)
                if r == 0:
                    print(f"  {corpus:8} {L:>8} {st['tokens']:>8} "
                          f"{st['repeated_8gram_pct']:>7.2f}% {st['distinct_8gram_ratio']:>9.4f} "
                          f"{st['longest_repeat_tokens']:>8}  {'PASS' if ok else 'FAIL'}")

    # Worst case per corpus decides usability - a corpus is only as good as its
    # worst rung, because the long rungs are exactly where speculation is judged.
    print("\n  worst case per corpus:")
    verdict = {}
    for c in corpora:
        w = max((r for r in rows if r["corpus"] == c), key=lambda r: r["repeated_8gram_pct"])
        verdict[c] = {"worst_pct": w["repeated_8gram_pct"], "at_tokens": w["target_tokens"],
                      "usable": w["repeated_8gram_pct"] < GATE_PCT}
        print(f"    {c:8} {w['repeated_8gram_pct']:>6.2f}% at {w['target_tokens']} "
              f"-> {'USABLE' if verdict[c]['usable'] else 'NOT USABLE for speculative arms'}")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gate": "CORPUS-0", "seed": a.seed, "utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": a.tokenizer, "ngram_n": N, "gate_pct": GATE_PCT,
        "verdict": verdict, "rows": rows,
        "calculation": "repeated_8gram_pct = positions covered by an 8-gram seen "
                       "earlier in the SAME prompt / total tokens",
    }, indent=2))
    print(f"\n  wrote {out}")
    return 0 if all(v["usable"] for v in verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
