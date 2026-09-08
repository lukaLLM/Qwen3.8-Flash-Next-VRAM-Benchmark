#!/usr/bin/env python3
"""Exact-match accuracy against a local OpenAI-compatible server.

No LLM judge anywhere: GSM8K is scored by the final number, MATH-500 by
math_verify (HF's symbolic checker, the same one the MATH-500 card recommends).
aiperf is deliberately NOT used - its --accuracy-benchmark reports a false 0%
pass@1 (W9 in PLAN.md).

Every item's prompt, raw response, extracted answer and verdict is written to
results.jsonl, so a score can be recomputed from disk without re-running the
model (Auto_Bench.md 11).
"""
from __future__ import annotations
import argparse, asyncio, json, pathlib, re, time, sys

NUM = re.compile(r"-?\d+(?:\.\d+)?")

def extract_gsm8k(text: str) -> str | None:
    """Last number in the response. The standard GSM8K convention: the model
    states its reasoning then the answer, so the final number is the answer."""
    m = NUM.findall(text.replace(",", ""))
    # Return the token VERBATIM. An earlier version called .rstrip(".0") here to
    # normalise "18.0" -> "18"; rstrip takes a CHARACTER SET, so it also ate
    # trailing zeros: 70000 -> 7, 540 -> 54, 20 -> 2. That scored 451 correct
    # answers as wrong and put GSM8K at 61% against a published 97.27%.
    # No normalisation is needed - score_gsm8k compares with float().
    return m[-1] if m else None

def score_gsm8k(resp: str, gold: str) -> bool:
    got = extract_gsm8k(resp)
    if got is None: return False
    try: return abs(float(got) - float(gold)) < 1e-6
    except ValueError: return got == gold

def score_math(resp: str, gold: str) -> bool:
    from math_verify import parse, verify
    try:
        return bool(verify(parse(f"${gold}$"), parse(resp)))
    except Exception:
        return False

async def one(client, model, item, task, sampler, max_tokens, sem, retries=2):
    prompt = item["question"] + (
        "\n\nSolve the problem. Put your final answer in \\boxed{}."
        if task == "math500" else
        "\n\nSolve the problem. End your reply with the final numeric answer.")
    async with sem:
        for attempt in range(retries + 1):
            try:
                t0 = time.time()
                r = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, **sampler)
                text = r.choices[0].message.content or ""
                ok = score_math(text, item["gold"]) if task == "math500" else score_gsm8k(text, item["gold"])
                return {"id": item["id"], "gold": item["gold"], "response": text,
                        "extracted": None if task == "math500" else extract_gsm8k(text),
                        "correct": ok, "finish_reason": r.choices[0].finish_reason,
                        "completion_tokens": getattr(r.usage, "completion_tokens", None),
                        "latency_s": round(time.time() - t0, 2), "error": None}
            except Exception as e:
                if attempt == retries:
                    # An errored item is recorded as an ERROR, never as a wrong
                    # answer: silently scoring failures as incorrect would make a
                    # flaky server look like a dumb model.
                    return {"id": item["id"], "gold": item["gold"], "response": "",
                            "extracted": None, "correct": None, "error": repr(e)[:300]}
                await asyncio.sleep(2 * (attempt + 1))

async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gsm8k", "math500"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.80)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    a = ap.parse_args()

    from openai import AsyncOpenAI
    items = [json.loads(l) for l in a.data.read_text().splitlines() if l.strip()]
    if a.limit: items = items[:a.limit]
    sampler = {"temperature": a.temperature, "top_p": a.top_p, "seed": a.seed,
               "presence_penalty": a.presence_penalty,
               "extra_body": {"top_k": a.top_k, "chat_template_kwargs": {"enable_thinking": False}}}
    client = AsyncOpenAI(base_url=a.url, api_key="local", timeout=900, max_retries=0)
    sem = asyncio.Semaphore(a.concurrency)

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = await asyncio.gather(*(one(client, a.model, it, a.task, sampler, a.max_tokens, sem) for it in items))
    (a.out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    n_err = sum(1 for r in rows if r["correct"] is None)
    scored = [r for r in rows if r["correct"] is not None]
    n_ok = sum(1 for r in scored if r["correct"])
    trunc = sum(1 for r in scored if r.get("finish_reason") == "length")
    acc = n_ok / len(scored) if scored else 0.0
    # Wilson interval: the normal approximation is wrong near the ceiling, and
    # GSM8K sits at ~0.97 on this model.
    import math
    z, n = 1.96, len(scored) or 1
    d = 1 + z*z/n; c = (acc + z*z/(2*n)) / d
    h = z*math.sqrt(acc*(1-acc)/n + z*z/(4*n*n)) / d
    summary = {"task": a.task, "model": a.model, "url": a.url, "n_items": len(items),
               "n_scored": len(scored), "n_errors": n_err, "n_correct": n_ok,
               "accuracy": round(acc, 6), "ci95_wilson": [round(c-h, 6), round(c+h, 6)],
               "truncated": trunc, "seed": a.seed, "concurrency": a.concurrency,
               "max_tokens": a.max_tokens, "sampler": sampler, "wall_s": round(time.time()-t0, 1)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if n_err: print(f"WARNING: {n_err} items errored and are EXCLUDED from the score", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
