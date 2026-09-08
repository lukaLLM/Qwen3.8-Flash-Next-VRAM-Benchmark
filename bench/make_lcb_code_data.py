#!/usr/bin/env python3
"""Build deterministic, length-controlled real-code prompts from local LCB data.

The source is the checked-in LiveCodeBench v5 first-100 input set already used
by benchmark/speed_bench_v2.py. This script only pads with other complete LCB
problems; it does not use AIPerf's synthetic ``coding`` corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "benchmark/data/lcb_release_v5_first100.inputs.json"
TOKENIZER = (
    Path.home()
    / ".cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/"
    "snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594"
)
SUFFIX = (
    "\n\nInstruction: write a complete, correct Python solution for the LAST "
    "problem above. Return executable code and explain the approach in comments."
)


def load_problems() -> list[str]:
    raw = json.loads(SOURCE.read_text())
    out: list[str] = []
    for item in raw.get("data", []):
        for payload in item.get("payloads", []):
            parts = [m["content"] for m in payload.get("messages", []) if m.get("content")]
            if parts:
                out.append("\n\n".join(parts))
    if not out:
        raise SystemExit(f"no prompts found in {SOURCE}")
    return out


def make_prompt(tokenizer, encoded: list[list[int]], target: int, sample: int) -> str:
    prefix = tokenizer.encode(
        f"[LCB real-code sample={sample} target={target} seed=42]\n",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    final = encoded[(sample * 7) % len(encoded)]
    budget = target - len(prefix) - len(suffix)
    if budget <= len(final):
        body = final[-budget:]
    else:
        pad: list[int] = []
        idx = sample * 7 + 1
        while len(pad) < budget - len(final):
            pad.extend(encoded[idx % len(encoded)])
            idx += 1
        body = pad[-(budget - len(final)) :] + final
    ids = prefix + body + suffix
    if len(ids) != target:
        raise AssertionError((len(ids), target))
    return tokenizer.decode(ids, skip_special_tokens=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="2048,8192,32000")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--output-dir", type=Path, default=REPO / "bench/data")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    problems = load_problems()
    encoded = [tokenizer.encode(p, add_special_tokens=False) for p in problems]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for target in sorted({int(x) for x in args.lengths.split(",") if x.strip()}):
        path = args.output_dir / f"lcb_code_{target}.jsonl"
        with path.open("w") as f:
            for sample in range(args.samples):
                text = make_prompt(tokenizer, encoded, target, sample)
                # Output length and engine-specific fields belong to the
                # workload/controller, not the source dataset. Keeping the
                # JSONL prompt-only lets every A/B use the same immutable text.
                row = {"text": text}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
