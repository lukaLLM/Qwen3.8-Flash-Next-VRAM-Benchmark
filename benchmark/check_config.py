# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27"]
# ///
"""Does the running server match the model card for the mode we are testing?

Run this against a live server BEFORE any measured request. Three separate
faults in this study would have been caught here instead of after the fact:

  - The preserved-reasoning test ran THINKING mode on the NON-THINKING
    temperature and top-p, which is why its result is only exploratory.
  - We never set --reasoning-format or --reasoning-preserve, while the server
    logged "chat template supports preserving reasoning" on every single start.
    Turns that were all reasoning came back with empty content, were saved
    nowhere, and silently dropped out of the conversation.
  - Output caps of 24,576 and 49,152 were chosen by guesswork against a model
    card that recommends 262,144 for reasoning and 131,072 for the final
    answer.

Values are from the official card:
https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8

  ./benchmark/check_config.py --mode thinking
  ./benchmark/check_config.py --mode non-thinking --max-tokens 49152 --strict
"""
from __future__ import annotations

import argparse, json, sys
import httpx

CARD = {
    "thinking":     {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                     "min_p": 0.0, "presence_penalty": 0.0, "repeat_penalty": 1.0},
    "non-thinking": {"temperature": 0.7, "top_p": 0.80, "top_k": 20,
                     "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0},
}
NATIVE_CTX = 262144
REC_MAX_REASONING = 262144   # card: reasoning content
REC_MAX_ANSWER = 131072      # card: final response


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", choices=sorted(CARD), required=True)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="the per-request output cap this run will use")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on any mismatch, not just on hard faults")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    try:
        props = httpx.get(f"{a.url}/props", timeout=15).json()
    except Exception as e:
        print(f"  cannot read {a.url}/props: {e}")
        return 2

    gen = props.get("default_generation_settings") or {}
    params = gen.get("params") or gen
    caps = props.get("chat_template_caps") or {}
    want = CARD[a.mode]

    hard, soft = [], []
    print(f"  model card mode: {a.mode}")
    print(f"  {'param':<18}{'server':>10}{'card':>10}   status")
    for k, v in want.items():
        got = params.get(k)
        if got is None:
            soft.append(f"{k} not reported by /props")
            status = "not reported"
        elif abs(float(got) - float(v)) < 1e-6:
            status = "ok"
        else:
            status = "MISMATCH"
            hard.append(f"{k}: server {got}, card {v}")
        print(f"  {k:<18}{str(got):>10}{str(v):>10}   {status}")

    # engine-level settings the card and the server both care about
    n_ctx = gen.get("n_ctx") or params.get("n_ctx")
    print(f"\n  n_ctx {n_ctx} (native {NATIVE_CTX})")
    if n_ctx and n_ctx > NATIVE_CTX:
        soft.append(f"n_ctx {n_ctx} exceeds native {NATIVE_CTX}; that needs YaRN, "
                    "and static YaRN degrades short-prompt quality")

    if caps.get("supports_preserve_reasoning"):
        print("  chat template supports preserving reasoning")
        soft.append("template supports --reasoning-preserve; set it explicitly "
                    "so history behaviour is a decision and not a default")

    if a.max_tokens:
        cap_ref = REC_MAX_REASONING if a.mode == "thinking" else REC_MAX_ANSWER
        print(f"  output cap {a.max_tokens} (card suggests up to {cap_ref} for this mode)")
        if a.max_tokens < cap_ref // 4:
            soft.append(f"output cap {a.max_tokens} is far below the card's {cap_ref}; "
                        "turns may be truncated, and this harness DISCARDS the text "
                        "of a truncated turn")

    # a quantised KV cache silently enables Hadamard attention rotation, which the
    # qwen4exp sparse-attention path does not support
    for key in ("type_k", "type_v", "cache_type_k", "cache_type_v"):
        val = str(params.get(key) or gen.get(key) or "")
        if val and "f16" not in val.lower() and val.lower() not in ("none", "0", ""):
            hard.append(f"{key}={val}: quantised KV needs LLAMA_ATTN_ROT_DISABLE=1 "
                        "on qwen4exp, or the sparse-attention path asserts")

    print()
    for h in hard:
        print(f"  FAULT: {h}")
    for s in soft:
        print(f"  NOTE:  {s}")
    verdict = "FAIL" if hard else ("WARN" if soft else "PASS")
    print(f"  CONFIG CHECK: {verdict}")

    if a.json:
        import pathlib
        pathlib.Path(a.json).write_text(json.dumps(
            {"mode": a.mode, "verdict": verdict, "faults": hard, "notes": soft,
             "server_params": {k: params.get(k) for k in want},
             "card": want, "n_ctx": n_ctx}, indent=2))
    if hard:
        return 1
    return 1 if (a.strict and soft) else 0


if __name__ == "__main__":
    raise SystemExit(main())
