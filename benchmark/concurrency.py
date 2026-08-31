# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27"]
# ///
"""Does throughput scale with concurrent requests?

Motivated by a measurement, not a hunch. DCGM profiling during single-stream
decode showed the tensor pipe at 0.64% and the memory interface at 14.6%, with
the card drawing 218 W of a 600 W budget and nothing throttling. Neither compute
nor bandwidth is close to saturated, so single-stream decode is latency bound:
each token needs a little work across 10 active experts at a 2560 hidden dim,
and every token waits on the one before it.

If that reading is right, concurrent requests should be close to free - they add
parallel work against expert weights that are already being read. Aggregate
throughput should scale nearly linearly until something actually saturates.

If it does NOT scale, the latency-bound story is wrong and something else is
serialising, which is a more interesting result than confirmation.

The server must be started with --parallel >= the highest concurrency tested,
or requests queue instead of running together and the test measures nothing.

  uv run benchmark/concurrency.py --levels 1,2,4,8,16 --out results/conc/conc.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def one(client: httpx.Client, url: str, prompt: str, n_predict: int) -> dict:
    t0 = time.perf_counter()
    r = client.post(f"{url}/completion", json={
        "prompt": prompt, "n_predict": n_predict, "temperature": 0,
        "cache_prompt": False,
        "ignore_eos": True,   # fixed token count; this model stops early otherwise
    }, timeout=1800.0)
    r.raise_for_status()
    b = r.json()
    t = b.get("timings", {})
    return {"wall": time.perf_counter() - t0,
            "predicted_n": t.get("predicted_n", 0),
            "tg_tps": t.get("predicted_per_second", 0.0),
            "prompt_n": t.get("prompt_n", 0),
            "pp_tps": t.get("prompt_per_second", 0.0)}


def props(url: str) -> tuple[int | None, int | None]:
    """(total_slots, per-slot context).

    Per-slot context is the one that matters and the one that is easy to get
    wrong: --parallel N divides the context N ways, so a server started with
    `--ctx 32768 --parallel 16` gives each request 2048 tokens, and a 3600-token
    prompt comes back as a bare 400. Ask the server instead of assuming.
    """
    try:
        d = httpx.get(f"{url}/props", timeout=10).json()
        n = d.get("total_slots")
        ctx = (d.get("default_generation_settings") or {}).get("n_ctx") or d.get("n_ctx")
        return n, ctx
    except Exception:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--levels", default="1,2,4,8,16")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--prompt-words", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    have, slot_ctx = props(a.url)
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    print(f"  server slots: {have}   per-slot ctx: {slot_ctx}   levels: {levels}")
    if have is not None and max(levels) > have:
        print(f"  WARNING: max level {max(levels)} exceeds {have} slots - the excess will")
        print(f"           queue rather than run concurrently. Restart with --parallel {max(levels)}.")

    words = a.prompt_words
    if slot_ctx:
        # ~9 tokens per repetition of the filler sentence, and leave room for the
        # generation plus a margin, or the request is rejected outright.
        budget = max(64, (slot_ctx - a.n_predict - 256) // 9)
        if words > budget:
            print(f"  prompt trimmed {words} -> {budget} reps to fit {slot_ctx} tokens/slot")
            words = budget
    base = "The quick brown fox jumps over the lazy dog. " * words
    rows = []
    with httpx.Client() as c:
        # warm the server once so the first level is not paying for a cold path
        one(c, a.url, base + "\n\nwarmup.", 16)
        for n in levels:
            # unique prefix per request so nothing is served from a shared cache
            prompts = [f"[req {i}] " + base + "\n\nContinue." for i in range(n)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as ex:
                res = list(ex.map(lambda p: one(c, a.url, p, a.n_predict), prompts))
            wall = time.perf_counter() - t0
            tot = sum(r["predicted_n"] for r in res)
            agg = tot / wall
            per = statistics.fmean(r["tg_tps"] for r in res)
            rows.append({"concurrency": n, "aggregate_tps": round(agg, 2),
                         "per_request_tps": round(per, 2), "wall_s": round(wall, 2),
                         "tokens": tot})
            print(f"  c={n:<3} aggregate {agg:8.2f} tok/s   per-request {per:7.2f} tok/s"
                  f"   wall {wall:6.1f}s")

    if rows:
        b = rows[0]["aggregate_tps"]
        print("\n  scaling against c=1:")
        for r in rows:
            eff = r["aggregate_tps"] / b / r["concurrency"] * 100
            print(f"    c={r['concurrency']:<3} {r['aggregate_tps']/b:5.2f}x   "
                  f"efficiency {eff:5.1f}%")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
