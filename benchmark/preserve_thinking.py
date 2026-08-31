# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27"]
# ///
"""Does preserving thinking blocks across turns help or hurt?

Qwen3.8-Flash-Next keeps the `<think>` block of every historical assistant turn
by default ("preserved thinking"). Qwen argue it improves decision consistency
in agent scenarios and "improves KV cache utilization". Both halves are
testable, and the second one is not obvious.

The template implements it directly:

    {%- if preserve_thinking is undefined or preserve_thinking is true
           or loop.index0 > ns.last_query_index %}
        '<think>' + reasoning_content + '</think>' + content
    {%- else %}
        content

so undefined means ON, and llama.cpp can flip it with
`--chat-template-kwargs '{"preserve_thinking": false}'`.

Verified 2026-08-30 against a live server: the template contains the
`preserve_thinking` branch AND `/props` reports
`chat_template_caps.supports_preserve_reasoning: true`, so the engine's own
`--reasoning-preserve` / `--no-reasoning-preserve` flag is ALSO available. An
earlier version of this comment claimed the marker was absent; that was wrong.
This harness keeps using the chat-template kwarg because it drives the template
variable directly and is visible in the request we send.

Do not confuse the two engine flags:
  --reasoning-preserve  keeps historical reasoning in the PROMPT (what this
                        test measures)
  --reasoning-format    controls whether <think> is parsed OUT of
                        message.content into message.reasoning_content. Left at
                        its default here. That is the one responsible for a
                        reasoning-only turn arriving with empty content.

WHY THIS IS WORTH MEASURING, AND WHAT THE TENSION IS

  preserve=true   history is append-only. The prompt prefix never changes, so
                  the server's KV cache keeps hitting and each turn only
                  prefills the newly added tokens. But the prompt grows by the
                  whole reasoning trace every turn - measured at ~27,000 tokens
                  per turn on this model - so context fills fast.

  preserve=false  prompts are far shorter. But dropping earlier thinking
                  REWRITES the middle of the history, so the cached prefix is
                  invalidated and the server may re-prefill from the divergence
                  point every single turn.

So it is fewer tokens against cacheable tokens, and which wins is an empirical
question about this model's prefill speed and reasoning verbosity. With ~27k
reasoning tokens per turn and 262144 of context, preserve=true plausibly
exhausts the context in under ten turns - which would make the "better for
agents" claim expensive rather than wrong.

WHAT IT MEASURES

  per turn:  prompt tokens sent, tokens actually prefilled (cache_prompt=true
             means `prompt_n` counts only what was NOT served from cache - that
             is the cache-effectiveness signal), prefill ms, decode tokens,
             reasoning tokens, wall time
  overall:   turns completed before context ran out, cumulative prefill work,
             total wall

  ./benchmark/preserve_thinking.py --turns 8 --out results/preserve/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


# Recommended thinking-mode sampler from the Qwen3.8-Flash-Next-FP8 model
# card. The old version of this test used the non-thinking temperature and
# top-p values while thinking stayed enabled. Results from that old version
# are not valid evidence for the recommended thinking configuration.
THINKING_SAMPLER = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}

# The same escalating coding session used everywhere else in this repo, so the
# workload is comparable. Trimmed to the build phase.
TURNS = [
    "Build a Gradio chat app in a single Python file using gr.Blocks with gr.Chatbot (type=\"messages\") that connects to a llama.cpp server at http://localhost:8000/v1. Use the openai Python client. Stream tokens to the UI. Keep multi-turn history and send the full conversation each request.",
    "Add persistent chat history: save each conversation to disk as JSON, with a sidebar to load past conversations and a New Chat button.",
    "Add a settings panel with sliders for temperature, top_p, top_k, min_p and repeat_penalty, applied to every request.",
    "Add speed statistics: show decode tokens/sec per message and a running session average, reset on New Chat.",
    "Add a Stop button that cancels an in-flight stream and a Regenerate button that reruns the last user message.",
    "Add a docstring to every function and full type hints to every signature.",
    "Refactor: extract all llama.cpp server communication into a separate module with a clean interface.",
    "Add an Export button that saves the current conversation to a markdown file including the speed stats.",
]


def turn(client: httpx.Client, url: str, messages: list, preserve: bool, max_tokens: int) -> dict:
    body = {
        "model": "qwen3.8-flash-next",
        "messages": messages,
        "max_tokens": max_tokens,
        **THINKING_SAMPLER,
        "cache_prompt": True,          # so prompt_n reveals what was NOT cached
        "chat_template_kwargs": {"preserve_thinking": preserve},
        "timings_per_token": False,
    }
    t0 = time.perf_counter()
    r = client.post(f"{url}/v1/chat/completions", json=body, timeout=3600.0)
    wall = time.perf_counter() - t0
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}", "wall": wall}
    d = r.json()
    m = d["choices"][0]["message"]
    t = d.get("timings", {}) or {}
    u = d.get("usage", {}) or {}
    return {
        "wall": wall,
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning_content") or "",
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "prefilled_n": t.get("prompt_n"),      # tokens actually processed
        "prefill_ms": t.get("prompt_ms"),
        "decode_tps": t.get("predicted_per_second"),
        "finish": d["choices"][0].get("finish_reason"),
    }


def run_arm(url: str, preserve: bool, n_turns: int, max_tokens: int) -> dict:
    msgs, rows = [], []
    with httpx.Client() as c:
        for i, prompt in enumerate(TURNS[:n_turns], 1):
            msgs.append({"role": "user", "content": prompt})
            r = turn(c, url, msgs, preserve, max_tokens)
            if "error" in r:
                print(f"    turn {i}: STOPPED - {r['error'][:120]}")
                rows.append({"turn": i, "error": r["error"]})
                break
            # echo the assistant turn back INCLUDING its reasoning, so the
            # template has something to preserve or drop. Sending only content
            # would make both arms identical and measure nothing.
            msgs.append({"role": "assistant", "content": r["content"],
                         "reasoning_content": r["reasoning"]})
            row = {"turn": i,
                   "prompt_tokens": r["prompt_tokens"],
                   "prefilled_n": r["prefilled_n"],
                   "cached_n": (r["prompt_tokens"] - r["prefilled_n"])
                                if (r["prompt_tokens"] and r["prefilled_n"] is not None) else None,
                   "prefill_ms": r["prefill_ms"],
                   "completion_tokens": r["completion_tokens"],
                   "reasoning_chars": len(r["reasoning"]),
                   "decode_tps": r["decode_tps"],
                   "wall": round(r["wall"], 1),
                   "finish": r["finish"]}
            rows.append(row)
            print(f"    turn {i}: prompt {row['prompt_tokens']:>7} "
                  f"prefilled {str(row['prefilled_n']):>7} "
                  f"cached {str(row['cached_n']):>7}  "
                  f"gen {row['completion_tokens']:>6}  "
                  f"{row['decode_tps'] or 0:.1f} t/s  {row['wall']:.0f}s")
    ok = [r for r in rows if "error" not in r]
    return {"preserve_thinking": preserve, "turns_completed": len(ok), "rows": rows,
            "total_wall_s": round(sum(r["wall"] for r in ok), 1),
            "total_prefilled": sum(r["prefilled_n"] or 0 for r in ok),
            "total_cached": sum(r["cached_n"] or 0 for r in ok),
            "final_prompt_tokens": ok[-1]["prompt_tokens"] if ok else 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--out", default="results/preserve")
    # Arm selection exists so the order can be counterbalanced across sessions.
    # The original run did both arms in one invocation with preserve=on always
    # first, which is why its result is exploratory. This test is ALSO about KV
    # cache behaviour, so each arm needs a freshly started server - running the
    # second arm against a server warmed by the first contaminates exactly the
    # thing being measured.
    ap.add_argument("--arm", choices=("on","off","both"), default="both",
                    help="run a single arm so an external controller can "
                         "alternate the order and restart the server between arms")
    a = ap.parse_args()

    out = {"sampler": THINKING_SAMPLER,
           "method_note": "One run per arm. Repeat and counterbalance before publication."}
    arms = {"on":(True,), "off":(False,), "both":(True,False)}[a.arm]
    for preserve in arms:
        print(f"\n  === preserve_thinking = {preserve} ===")
        out["on" if preserve else "off"] = run_arm(a.url, preserve, a.turns, a.max_tokens)

    # Write the artifact BEFORE any summary work. An earlier version unpacked
    # out["on"], out["off"] first, so a single-arm run raised KeyError and the
    # completed arm's data was never written to disk at all - 25 minutes of GPU
    # time surviving only as terminal text.
    p = Path(a.out); p.mkdir(parents=True, exist_ok=True)
    (p / "preserve_thinking.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {p/'preserve_thinking.json'}")

    print("\n  === summary ===")
    if "on" not in out or "off" not in out:
        print(f"  single arm '{a.arm}' - an external controller pools the arms")
        return 0
    on, off = out["on"], out["off"]
    print(f"  {'':<22}{'preserve=on':>14}{'preserve=off':>14}")
    for k, lab in (("turns_completed","turns completed"), ("final_prompt_tokens","final prompt tok"),
                   ("total_prefilled","tokens prefilled"), ("total_cached","tokens from cache"),
                   ("total_wall_s","total wall s")):
        print(f"  {lab:<22}{on[k]:>14,}{off[k]:>14,}")
    if on["total_prefilled"] and off["total_prefilled"]:
        d = off["total_prefilled"] / on["total_prefilled"]
        print(f"\n  preserve=off prefilled {d:.2f}x the tokens of preserve=on")
        print("  (>1 means dropping thinking INVALIDATED the cache and cost more work,")
        print("   <1 means the shorter prompts won despite losing cache hits)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
