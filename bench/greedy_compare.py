#!/usr/bin/env python3
"""Greedy equivalence gate: do two configs of the SAME engine emit the same tokens?

Why this exists
---------------
auto_run/TUNING_QUEUE.md item 5 requires "exact token-ID equality under greedy
decoding" before any rebuilt image is promoted, and bench/specdecode.sh:200-203
already states the policy:

    Comparison basis, in order of strength: generated TOKEN IDS where the server
    exposes them; otherwise EXACT TEXT equality [...] Never claim token-id
    equality from a text-only endpoint.

specdecode.sh implements only the text half, inline, for its own two arms. This
is that gate as a standalone tool, with the token-ID half implemented.

Basis per engine (verified against each server's source, 2026-09-05):
  llamacpp   POST /completion  {"return_tokens": true}      -> token ids
  sglang     POST /generate    {"return_logprob": true}     -> token ids
  freetoken  POST /v1/chat/completions                      -> TEXT ONLY; its
             OpenAI route hard-refuses logprobs ("logprobs is not supported")
             and returns "logprobs": None unconditionally.

Both arms of a comparison must be the same engine. Comparing across engines is
meaningless here: they run different checkpoints and different tokenisers, so a
divergence would say nothing about the change under test.

NO ignore_eos. The model must be allowed to stop; forcing generation past its
natural end produces filler that is not a property of the config under test
(results/BLOCKERS.md B-17).
"""
from __future__ import annotations
import argparse, json, pathlib, sys, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

# The model is ChatML. Building the turn explicitly keeps the prompt string
# byte-identical across arms without depending on each server's own templating.
def chatml(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

def post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def gen_llamacpp(base: str, prompt: str, n: int) -> tuple[str, list[int] | None]:
    j = post(f"{base}/completion", {
        "prompt": prompt, "n_predict": n,
        "temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 0,
        "cache_prompt": False,      # a warm prefix must not change the result
        "return_tokens": True,
    })
    return j.get("content", ""), (j.get("tokens") or None)

def gen_sglang(base: str, prompt: str, n: int) -> tuple[str, list[int] | None]:
    j = post(f"{base}/generate", {
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "top_k": 1, "top_p": 1.0,
                            "max_new_tokens": n},
        "return_logprob": True,
    })
    meta = j.get("meta_info") or {}
    # output_token_logprobs rows are [logprob, token_id, token_text]
    rows = meta.get("output_token_logprobs") or []
    ids = [r[1] for r in rows if isinstance(r, (list, tuple)) and len(r) > 1]
    return j.get("text", ""), (ids or None)

def gen_freetoken(base: str, prompt_user: str, n: int) -> tuple[str, list[int] | None]:
    j = post(f"{base}/v1/chat/completions", {
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": prompt_user}],
        "max_tokens": n, "temperature": 0.0, "top_p": 1.0, "seed": 0,
    })
    return (j["choices"][0]["message"].get("content") or ""), None

GEN = {"llamacpp": gen_llamacpp, "sglang": gen_sglang, "freetoken": gen_freetoken}

def cmd_capture(a) -> int:
    items = [json.loads(l) for l in a.prompts.read_text().splitlines() if l.strip()][:a.limit]
    out = a.out; out.mkdir(parents=True, exist_ok=True)

    def one(idx_item):
        i, it = idx_item
        user = it.get("question") or it.get("prompt") or it.get("text") or ""
        if isinstance(user, list):                       # chat-shaped dataset row
            user = next((m.get("content", "") for m in user if m.get("role") == "user"), "")
        pid = it.get("id", f"item-{i}")
        try:
            if a.engine == "freetoken":
                text, ids = gen_freetoken(a.url, user, a.max_tokens)
            else:
                text, ids = GEN[a.engine](a.url, chatml(user), a.max_tokens)
            err = None
        except Exception as e:                            # recorded, never scored
            text, ids, err = "", None, repr(e)[:300]
        return {"id": pid, "text": text, "token_ids": ids, "error": err,
                "n_tokens": len(ids) if ids else None}

    # CONCURRENCY IS THE INSTRUMENT, not a speed-up. At --concurrency 1 the
    # server decodes at batch 1; above it, requests overlap and the decode batch
    # grows, which is the only way to exercise a CUDA graph captured for batch>1.
    # Greedy output must not depend on how many requests happen to share a batch;
    # comparing a c=1 capture against a c=N capture is exactly that test.
    if a.concurrency > 1:
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            rows = list(ex.map(one, enumerate(items)))
    else:
        rows = [one(x) for x in enumerate(items)]
    for i, r in enumerate(rows):
        print(f"  [{i+1}/{len(rows)}] {r['id']}  {len(r['text'])} chars  "
              f"{'ids='+str(len(r['token_ids'])) if r['token_ids'] else 'TEXT ONLY'}"
              f"{'  ERR' if r['error'] else ''}")
    n_ids = sum(r["token_ids"] is not None for r in rows)
    basis = "token_ids" if n_ids == len(rows) and rows else "text"
    (out / "capture.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (out / "capture_meta.json").write_text(json.dumps(
        {"engine": a.engine, "url": a.url, "basis": basis, "n": len(rows),
         "max_tokens": a.max_tokens, "prompts": str(a.prompts), "label": a.label,
         "concurrency": a.concurrency,
         "n_errors": sum(r["error"] is not None for r in rows),
         "sampling": "greedy temperature=0 top_k=1 top_p=1 seed=0, no ignore_eos"}, indent=2))
    print(f"  basis={basis}  errors={sum(r['error'] is not None for r in rows)}  -> {out}")
    return 0

def cmd_compare(a) -> int:
    def load(d: pathlib.Path):
        m = json.loads((d / "capture_meta.json").read_text())
        rows = {r["id"]: r for r in
                (json.loads(l) for l in (d / "capture.jsonl").read_text().splitlines() if l.strip())}
        return m, rows
    try:
        ma, ra = load(a.a); mb, rb = load(a.b)
    except Exception as e:
        print(f"INCONCLUSIVE: cannot read captures ({e})"); return 2
    if ma["engine"] != mb["engine"]:
        print(f"INCONCLUSIVE: different engines ({ma['engine']} vs {mb['engine']}); "
              "a cross-engine divergence says nothing about the change under test")
        return 2
    if ma["basis"] != mb["basis"]:
        print(f"INCONCLUSIVE: basis mismatch ({ma['basis']} vs {mb['basis']}); "
              "refusing to claim token-id equality against a text-only capture")
        return 2
    basis = ma["basis"]
    common = sorted(set(ra) & set(rb))
    usable = [i for i in common if not ra[i]["error"] and not rb[i]["error"]]
    if not usable:
        # A gate that cannot read its inputs must fail loudly, or a missing
        # result reads identically to "no mismatch found".
        print(f"INCONCLUSIVE: no comparable items (common={len(common)}, "
              f"errors A={ma['n_errors']} B={mb['n_errors']})")
        return 2
    key = "token_ids" if basis == "token_ids" else "text"
    diffs = []
    for i in usable:
        x, y = ra[i][key], rb[i][key]
        if x != y:
            if basis == "token_ids":
                n = min(len(x or []), len(y or []))
                at = next((k for k in range(n) if x[k] != y[k]), n)
            else:
                n = min(len(x), len(y))
                at = next((k for k in range(n) if x[k] != y[k]), n)
            diffs.append((i, at, len(x or []), len(y or [])))
    print(f"  basis={basis}  engine={ma['engine']}  compared={len(usable)}"
          f"  A={ma.get('label') or a.a.name}(c={ma.get('concurrency','?')})"
          f"  B={mb.get('label') or b_name(a)}(c={mb.get('concurrency','?')})")
    if diffs:
        print(f"FAIL: {len(diffs)}/{len(usable)} differ under greedy decoding")
        for i, at, la, lb in diffs[:5]:
            print(f"    {i}: first divergence at index {at} (lenA={la} lenB={lb})")
        return 1
    print(f"PASS: {len(usable)}/{len(usable)} identical ({basis})")
    return 0

def b_name(a): return a.b.name

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("capture")
    c.add_argument("--engine", required=True, choices=sorted(GEN))
    c.add_argument("--url", required=True, help="e.g. http://localhost:8001")
    c.add_argument("--prompts", type=pathlib.Path,
                   default=pathlib.Path("bench/data/lcb_code_2048.jsonl"))
    c.add_argument("--out", type=pathlib.Path, required=True)
    c.add_argument("--limit", type=int, default=16)
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--label", default="")
    c.add_argument("--concurrency", type=int, default=1,
                   help="requests in flight; >1 forces the server to batch decode")
    c.set_defaults(fn=cmd_capture)
    d = s.add_parser("compare")
    d.add_argument("a", type=pathlib.Path); d.add_argument("b", type=pathlib.Path)
    d.set_defaults(fn=cmd_compare)
    a = p.parse_args()
    return a.fn(a)

if __name__ == "__main__":
    raise SystemExit(main())
