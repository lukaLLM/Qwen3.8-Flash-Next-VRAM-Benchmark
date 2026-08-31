# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1.90,<3"]
# ///
"""Headless speed bench for the llama.cpp speculative-decoding stacks.

The workload: a fixed 18-prompt suite sent as ONE cumulative conversation,
in two phases:
  build (1-9)    the model creates a Gradio chat app and extends the SAME
                 code base feature by feature (history, settings, stats...)
  maint (10-18)  realistic vibe-coding follow-ups on the finished app -
                 full re-emission, docstring/type-hint pass, renames, a
                 vague bug report, a refactor, pytest tests, a small
                 feature, review-and-fix, README
The model keeps re-emitting and modifying its own prior code - exactly the
iterative-coding scenario where n-gram drafters should help (or not). The
maint phase is where the advantage should peak; aggregates are reported per
phase and overall.

Measured per turn, straight from the llama.cpp `timings` block plus a
client-side clock: TTFT, decode t/s, prompt t/s, generated tokens, draft
tokens, draft accepted, accept %.

Workflow:
  1. start ONE docker service (baseline / dflash / dflash+ngram)
  2. uv run benchmark/bench_ngram.py   (uv resolves the inline deps above
     into an isolated env - the repo venv is not touched)
     for the baseline service use:  uv run benchmark/bench_ngram.py --baseline
     (targets port 8000; accept shows "-" since there is no spec decoding)
  3. swap the docker service, run again
  4. compare the table printed at the end (or bench_results/*/stats.json)

Each run saves into its own folder benchmark/bench_results/<ts>_<alias>[_<tag>]/ :
  stats.json        per-turn rows + aggregates + settings + server info
  transcript.json   the full cumulative conversation
  responses/NN.md   the model's evolving app code, one file per turn
"""

# Copied from deepseek-v4-flash-dspark-rtx6000pro@d790b4f, benchmark/.
# Kept deliberately close to the original so results stay comparable across
# the three studies. Adaptations for this repo:
#   - default URL 8001 -> 8000 (this repo runs one service on 8000)
#   - TURN PROMPTS ARE UNCHANGED, deliberately - they are what makes these numbers
#   - comparable against the Qwen3.6 and DeepSeek V4 Flash studies

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from openai import OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR / "bench_results"

FALLBACK_N_CTX = 262144  # 256k - the real value is probed from the server
CTX_CRIT_PCT = 0.95

# keys copied out of the llama.cpp `timings` block (see benchmark/leaderboard.py)
TIMING_KEYS = (
    "predicted_n",
    "predicted_ms",
    "predicted_per_second",
    "prompt_n",
    "prompt_ms",
    "prompt_per_second",
    "draft_n",
    "draft_n_accepted",
)

# ----------------------------------------------------------------------------
# The prompt suite - kept verbatim. The model does the app-building, not us.
# ----------------------------------------------------------------------------

BENCH_PROMPTS = [
    """Build a Gradio chat app in a single Python file using gr.Blocks with gr.Chatbot (type="messages") that connects to a llama.cpp server at http://localhost:8001/v1. Use the openai Python client with base_url="http://localhost:8001/v1" and a dummy api_key. Stream tokens to the UI. Keep multi-turn history in the message list and send the full conversation each request. Create a separate venv and manage packages using uv.""",
    """Add persistent chat history: save each conversation to disk as JSON, a sidebar or dropdown to load past sessions, a New Chat button, and a Delete button. Store title, timestamp, and messages per session.""",
    """Add a settings panel with sliders for temperature, top_p, top_k, min_p, repeat_penalty, and max_tokens, plus a system prompt textbox, all passed through to the llama.cpp request. Put the llama.cpp-specific params that are not in the standard OpenAI schema into extra_body.""",
    """On startup, call the server's /v1/models and /health (or /props) endpoints to show the loaded model, context size, and connection status in the UI. If the server is down, show a clear error instead of crashing.""",
    """After each response, read the timings object from the llama.cpp reply and display tokens/sec, prompt eval time, and draft acceptance rate under the message. Capture the final stream chunk so timings are available.""",
    """Add speed statistics: for each message show predicted_per_second (decode t/s) and prompt_per_second. Maintain a running session average of decode t/s across all assistant turns, plus min, max, and the total tokens generated and total time. Show these aggregates in a stats panel that updates after every turn, and reset them on New Chat. Compute the average as total predicted tokens divided by total predicted time, not a mean of per-turn rates, so it is not skewed by short messages.""",
    """Add a Stop button that cancels an in-flight stream, a Regenerate button that reruns the last user turn, and an Edit option on the last user message.""",
    """Show a running token count for the conversation and warn when it approaches the model's context window, using prompt_tokens returned by the server.""",
    """Style the whole UI as a dark hacker terminal: near-black background, a monospace font like JetBrains Mono or Fira Code everywhere, green or amber text for output, a subtle terminal-style prompt caret on the input, minimal chrome, and matching dark styling for the chatbot bubbles, sidebar, sliders, and stats panel. Apply it with a custom gr.themes theme plus custom CSS so it stays consistent across all components.""",
    # --- maint phase: realistic follow-up work on the finished app ----------
    """Show me the complete, final app.py with everything we have built so far in a single code block. No explanations, just the full file.""",
    """Add a docstring to every function and full type hints to every function signature in app.py. Do not change any behavior. Show the complete updated file.""",
    """Rename things for clarity: the function that sends the chat request should be called send_chat_request, the input textbox variable should be message_input, and all stats-panel related variables should get a stats_ prefix. Update every usage consistently and show the complete updated file.""",
    """Bug report from a user: when the llama.cpp server goes down while the app is already open, clicking Send crashes the app with a traceback instead of showing the connection warning. Find the cause in your code, fix it, and show the complete updated file.""",
    """Refactor: extract all llama.cpp server communication (the health check, the model probe, the streaming request and the timings parsing) into a LlamaClient class in the same file, and make the UI code use it. Behavior must stay identical. Show the complete updated file.""",
    """Write pytest unit tests for the session persistence functions (save, load, list, delete) and for the stats aggregation logic. Use tmp_path for the filesystem and do not hit the network. Put everything in one test_app.py and show the full file.""",
    """Add an Export button that saves the current conversation to a markdown file, with the speed stats table appended at the end. Show the complete updated file.""",
    """Do a code review of the whole app: list the top 5 issues with a severity for each, then apply all 5 fixes and show the complete updated file.""",
    """Write a README.md for the app: features, setup with uv, how to run it, configuration options, and a short troubleshooting section.""",
]

BENCH_LABELS = [
    "build app",
    "chat history",
    "settings panel",
    "startup probes",
    "timings display",
    "speed stats",
    "stop/regen/edit",
    "token counter",
    "terminal theme",
    "full re-emit",
    "docstrings+types",
    "renames",
    "bugfix report",
    "client refactor",
    "pytest tests",
    "export feature",
    "review+fix",
    "write readme",
]

# build = greenfield feature-by-feature; maint = work on the existing file
BENCH_PHASES = ["build"] * 9 + ["maint"] * 9

# ----------------------------------------------------------------------------
# Server probing
# ----------------------------------------------------------------------------


@dataclass
class ServerInfo:
    ok: bool = False
    base_url: str = ""
    model_id: str | None = None
    n_ctx: int | None = None
    build: str | None = None
    error: str | None = None


def probe_server(base_url: str) -> ServerInfo:
    """Check /health, then read model id + context size from /v1/models and /props."""
    info = ServerInfo(base_url=base_url)
    try:
        r = httpx.get(f"{base_url}/health", timeout=3.0)
        if r.status_code != 200:
            info.error = f"/health returned HTTP {r.status_code}"
            return info
    except Exception as e:
        info.error = f"{type(e).__name__}: {e}"
        return info
    info.ok = True
    try:
        data = (httpx.get(f"{base_url}/v1/models", timeout=3.0).json().get("data")) or []
        if data:
            info.model_id = data[0].get("id")
            meta = data[0].get("meta") or {}
            if isinstance(meta.get("n_ctx"), int):
                info.n_ctx = meta["n_ctx"]
    except Exception:
        pass
    try:
        props = httpx.get(f"{base_url}/props", timeout=3.0).json()
        dgs = props.get("default_generation_settings") or {}
        if isinstance(dgs.get("n_ctx"), int):
            info.n_ctx = dgs["n_ctx"]  # authoritative over /v1/models meta
        info.build = props.get("build_info")
        if not info.model_id:
            info.model_id = props.get("model_alias")
    except Exception:
        pass
    if info.n_ctx is None:
        info.n_ctx = FALLBACK_N_CTX
    return info


# ----------------------------------------------------------------------------
# One streamed turn
# ----------------------------------------------------------------------------


def stream_turn(client: OpenAI, model_id: str, api_messages: list[dict],
                args: argparse.Namespace, progress_prefix: str) -> dict:
    """Send one request, stream it, return
    {text, timings, usage, ttft, error}. Prints live progress on one line."""
    text = ""
    timings: dict | None = None
    usage: dict | None = None
    ttft: float | None = None
    error: str | None = None
    stream = None
    last_print = 0.0
    t0 = time.perf_counter()
    try:
        # only send sampler params the user explicitly set - anything left
        # unset falls back to the server/model defaults
        sampler_kwargs = {}
        if args.temperature is not None:
            sampler_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            sampler_kwargs["top_p"] = args.top_p
        if args.presence_penalty is not None:
            sampler_kwargs["presence_penalty"] = args.presence_penalty
        if args.max_tokens is not None:
            sampler_kwargs["max_tokens"] = args.max_tokens
        extra_body = {
            # llama.cpp extension - ship the timings block in stream chunks
            "timings_per_token": True,
            # keep the server prefix cache warm across turns
            "cache_prompt": True,
        }
        # llama.cpp sampler params the openai SDK has no kwargs for
        # Qwen's non-thinking mode. The recommended sampler differs per mode, so
        # the mode and the sampler must be set together or the numbers describe
        # a configuration nobody should run.
        if args.no_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if args.top_k is not None:
            extra_body["top_k"] = args.top_k
        if args.min_p is not None:
            extra_body["min_p"] = args.min_p
        if args.repeat_penalty is not None:
            extra_body["repeat_penalty"] = args.repeat_penalty
        stream = client.chat.completions.create(
            model=model_id,
            messages=api_messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
            **sampler_kwargs,
        )
        for chunk in stream:
            t = (chunk.model_extra or {}).get("timings")
            if not isinstance(t, dict):
                t = getattr(chunk, "timings", None)
            if isinstance(t, dict):
                timings = t  # cumulative - the last chunk holds the totals
            u = getattr(chunk, "usage", None)
            if u is not None:
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", None),
                    "completion_tokens": getattr(u, "completion_tokens", None),
                    "total_tokens": getattr(u, "total_tokens", None),
                }
            if chunk.choices:  # the usage-only final chunk has empty choices
                delta = chunk.choices[0].delta
                if delta is not None and delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text += delta.content
                    now = time.perf_counter()
                    if now - last_print >= 0.25:
                        last_print = now
                        n = (timings or {}).get("predicted_n") or 0
                        tps = (timings or {}).get("predicted_per_second")
                        speed = f" {tps:7.1f} t/s" if tps else ""
                        print(f"\r{progress_prefix} {n:6d} tok{speed}   ",
                              end="", flush=True)
    except KeyboardInterrupt:
        raise  # handled by main - save what we have
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
    return {"text": text, "timings": timings, "usage": usage, "ttft": ttft,
            "wall_s": time.perf_counter() - t0, "error": error}


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------


def make_row(index: int, label: str, result: dict) -> dict:
    t = result.get("timings") or {}
    u = result.get("usage") or {}
    row = {"index": index, "label": label, "error": result.get("error")}
    row.update({k: t.get(k) for k in TIMING_KEYS})
    row["ttft"] = result.get("ttft")
    row["wall_s"] = result.get("wall_s")
    row["prompt_tokens"] = u.get("prompt_tokens")
    row["completion_tokens"] = u.get("completion_tokens")
    return row


def accept_str(row: dict) -> str:
    n = row.get("draft_n") or 0
    acc = row.get("draft_n_accepted") or 0
    if n <= 0:
        return "-"  # baseline server / no speculative decoding
    return f"{100.0 * acc / n:.0f}% ({acc}/{n})"


def fmt(x, spec=".1f", dash="-") -> str:
    return format(x, spec) if isinstance(x, (int, float)) else dash


def early_late_tps(valid: list[dict]) -> tuple[float | None, float | None]:
    """Decode t/s (sum/sum) over the first 3 vs last 3 valid turns - shows how
    speed scales with context depth. None/None when fewer than 6 turns."""
    if len(valid) < 6:
        return None, None

    def tps(chunk):
        n = sum(r["predicted_n"] for r in chunk)
        ms = sum(r["predicted_ms"] for r in chunk)
        return 1000.0 * n / ms if ms else None

    return tps(valid[:3]), tps(valid[-3:])


def aggregate(rows: list[dict]) -> dict | None:
    """Overall decode t/s = total predicted tokens / total predicted time -
    NOT a mean of per-turn rates - so short turns do not skew it."""
    valid = [r for r in rows if r and not r.get("error")
             and r.get("predicted_n") and r.get("predicted_ms")]
    if not valid:
        return None
    tot_n = sum(r["predicted_n"] for r in valid)
    tot_ms = sum(r["predicted_ms"] for r in valid)
    rates = [r["predicted_per_second"] for r in valid if r.get("predicted_per_second")]
    ttfts = [r["ttft"] for r in valid if r.get("ttft") is not None]
    walls = [r["wall_s"] for r in valid if r.get("wall_s")]
    total_wall_s = sum(walls) if walls else None
    prompt_n = sum(r.get("prompt_n") or 0 for r in valid)
    prompt_ms = sum(r.get("prompt_ms") or 0 for r in valid)
    draft_n = sum(r.get("draft_n") or 0 for r in valid)
    draft_acc = sum(r.get("draft_n_accepted") or 0 for r in valid)
    early_tps, late_tps = early_late_tps(valid)
    return {
        "turns": len(valid),
        "total_tokens": tot_n,
        "total_time_s": tot_ms / 1000.0,
        "decode_tps": 1000.0 * tot_n / tot_ms if tot_ms else None,
        "min_tps": min(rates) if rates else None,
        "max_tps": max(rates) if rates else None,
        "avg_ttft_s": sum(ttfts) / len(ttfts) if ttfts else None,
        "avg_tokens_per_turn": tot_n / len(valid),
        "prompt_tps": 1000.0 * prompt_n / prompt_ms if prompt_ms else None,
        "draft_n": draft_n,
        "draft_n_accepted": draft_acc,
        "accept_pct": 100.0 * draft_acc / draft_n if draft_n else None,
        # end-to-end throughput - includes prefill wait and network, the
        # "felt" speed of the whole session
        "total_wall_s": total_wall_s,
        "e2e_tps": tot_n / total_wall_s if total_wall_s else None,
        # speculation economics - how much of the output came from drafts
        # for free, and how much drafting was wasted
        "spec_share_pct": 100.0 * draft_acc / tot_n if draft_n and tot_n else None,
        "draft_rejected": (draft_n - draft_acc) if draft_n else None,
        "draft_overhead": draft_n / tot_n if draft_n and tot_n else None,
        # context-depth scaling
        "early_tps": early_tps,
        "late_tps": late_tps,
    }


def agg_accept_str(agg: dict) -> str:
    if agg.get("accept_pct") is None:
        return "-"
    return f"{agg['accept_pct']:.0f}% ({agg['draft_n_accepted']}/{agg['draft_n']})"


def print_run_table(rows: list[dict]) -> None:
    head = f"{'#':>2}  {'prompt':<16} {'tokens':>7} {'decode t/s':>11} {'prompt t/s':>11} {'TTFT':>7} {'ctx used':>9}  accept"
    print(head)
    print("-" * len(head))
    prev_phase = None
    for r in rows:
        phase = r.get("phase")
        if prev_phase is not None and phase != prev_phase:
            print(f"--- {phase} phase " + "-" * (len(head) - len(str(phase)) - 11))
        prev_phase = phase
        if r.get("error") and not r.get("predicted_n"):
            print(f"{r['index']:>2}  {r['label']:<16} ERROR: {r['error']}")
            continue
        used = (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
        ttft = f"{r['ttft']:.2f}s" if r.get("ttft") is not None else "-"
        print(f"{r['index']:>2}  {r['label']:<16} "
              f"{r.get('predicted_n') or '-':>7} "
              f"{fmt(r.get('predicted_per_second')):>11} "
              f"{fmt(r.get('prompt_per_second')):>11} "
              f"{ttft:>7} "
              f"{used:>9,}  "
              f"{accept_str(r)}")


def print_aggregates(agg: dict | None) -> None:
    if not agg:
        print("no completed turns - nothing to aggregate")
        return
    print(f"overall decode : {fmt(agg['decode_tps'])} t/s  "
          f"(= {agg['total_tokens']:,} tok / {agg['total_time_s']:.1f}s, "
          f"min {fmt(agg['min_tps'])} / max {fmt(agg['max_tps'])} per turn)")
    print(f"prompt (prefill): {fmt(agg['prompt_tps'])} t/s")
    print(f"avg TTFT       : {fmt(agg['avg_ttft_s'], '.2f')}s   "
          f"avg tokens/turn: {fmt(agg['avg_tokens_per_turn'], '.0f')}")
    print(f"draft accept   : {agg_accept_str(agg)}")
    if agg.get("e2e_tps"):
        print(f"end-to-end     : {fmt(agg['e2e_tps'])} t/s  "
              f"(incl. prefill; total wall {fmt(agg.get('total_wall_s'))}s)")
    if agg.get("spec_share_pct") is not None:
        print(f"speculation    : {agg['spec_share_pct']:.0f}% of output tokens came from drafts · "
              f"{agg.get('draft_rejected') or 0:,} drafted+rejected "
              f"(overhead {fmt(agg.get('draft_overhead'), '.2f')} draft tok per output tok)")
    if agg.get("early_tps") and agg.get("late_tps"):
        print(f"ctx scaling    : {fmt(agg['early_tps'])} t/s (first 3 turns) -> "
              f"{fmt(agg['late_tps'])} t/s (last 3 turns)")


# ----------------------------------------------------------------------------
# Run persistence + cross-run comparison
# ----------------------------------------------------------------------------


def write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def list_runs() -> list[str]:
    if not BENCH_DIR.is_dir():
        return []
    return sorted((p.name for p in BENCH_DIR.iterdir() if (p / "stats.json").is_file()))


def load_run(run_id: str) -> dict | None:
    try:
        return json.loads((BENCH_DIR / run_id / "stats.json").read_text())
    except Exception:
        return None


def find_ref_run(runs: list[str], ref_arg: str | None) -> str | None:
    """Pick the speedup reference: --ref if given, else the newest run that
    reported no draft stats (= a baseline run without speculative decoding)."""
    if ref_arg:
        if ref_arg in runs:
            return ref_arg
        print(f"warning: --ref {ref_arg} not found under bench_results/")
        return None
    for rid in sorted(runs, reverse=True):
        d = load_run(rid)
        agg = (d or {}).get("aggregates") or {}
        if agg.get("decode_tps") and agg.get("accept_pct") is None:
            return rid
    return None


def print_comparison(ref_arg: str | None = None) -> None:
    runs = list_runs()
    if not runs:
        return
    ref_id = find_ref_run(runs, ref_arg)
    ref_tps = None
    if ref_id:
        ref_tps = ((load_run(ref_id) or {}).get("aggregates") or {}).get("decode_tps")
    print()
    print("=== all runs (bench_results/) " + "=" * 50)
    if ref_id and ref_tps:
        print(f"speedup reference: {ref_id} ({fmt(ref_tps)} t/s)")
    else:
        print("speedup reference: none found (run --baseline once, or pass --ref RUN_ID)")
    head = (f"{'run':<44} {'model':<24} {'decode t/s':>11} {'speedup':>8} "
            f"{'maint t/s':>10} {'m.accept':>8} {'accept':>7} {'e2e t/s':>9} "
            f"{'tokens':>8}  complete")
    print(head)
    print("-" * len(head))
    for rid in runs:
        d = load_run(rid)
        if not d:
            continue
        agg = d.get("aggregates") or {}
        maint = (d.get("aggregates_by_phase") or {}).get("maint") or {}
        model = (d.get("server") or {}).get("model") or "?"
        tps = agg.get("decode_tps")
        speedup = f"{tps / ref_tps:.2f}x" if tps and ref_tps else "-"
        acc = f"{agg['accept_pct']:.0f}%" if agg.get("accept_pct") is not None else "-"
        m_acc = f"{maint['accept_pct']:.0f}%" if maint.get("accept_pct") is not None else "-"
        toks = f"{agg['total_tokens']:,}" if agg.get("total_tokens") else "-"
        print(f"{rid:<44} {model:<24} {fmt(tps):>11} {speedup:>8} "
              f"{fmt(maint.get('decode_tps')):>10} {m_acc:>8} {acc:>7} "
              f"{fmt(agg.get('e2e_tps')):>9} {toks:>8}  "
              f"{'no' if d.get('aborted') else 'yes'}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=None,
                   help="server base url (default http://localhost:8000 or "
                        "$LLAMA_BENCH_URL; takes precedence over --baseline)")
    p.add_argument("--baseline", action="store_true",
                   help="bench the baseline server instead - http://localhost:"
                        "$LLAMA_BASELINE_HOST_PORT (default 8000, the "
                        "llamacpp_baseline service, no speculative decoding); "
                        "gives the reference decode t/s for the comparison table")
    p.add_argument("--tag", default="",
                   help="suffix for the run folder name, e.g. an ablation label "
                        "(useful when LLAMA_SPEC_TYPE variants share one alias)")
    p.add_argument("--ref", default=None,
                   help="run id used as the speedup reference in the comparison "
                        "table (default: newest run without draft stats, i.e. "
                        "the latest baseline run)")
    p.add_argument("--temperature", type=float, default=None,
                   help="optional override; unset = server/model default")
    p.add_argument("--top-p", type=float, default=None,
                   help="optional override; unset = server/model default")
    p.add_argument("--top-k", type=int, default=None,
                   help="optional override; unset = server/model default")
    p.add_argument("--min-p", type=float, default=None,
                   help="optional override; unset = server/model default")
    p.add_argument("--no-thinking", action="store_true",
                   help="send chat_template_kwargs={'enable_thinking': False}. "
                        "Pair this with the model card's NON-thinking sampler "
                        "(temp 0.7, top-p 0.80, top-k 20, presence-penalty 1.5); "
                        "the thinking sampler is a different set.")
    p.add_argument("--presence-penalty", type=float, default=None,
                   help="optional override; the non-thinking mode recommends 1.5")
    p.add_argument("--repeat-penalty", type=float, default=None,
                   help="optional override; unset = server/model default")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="optional per-turn cap; unset = server/model default")
    p.add_argument("--turns", type=int, default=None,
                   help="stop after the first N turns (default: all 18). "
                        "--turns 9 = build phase only, which excludes the "
                        "re-emit turns that inflate draft acceptance")
    p.add_argument("--system", default="", help="optional system prompt")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.url:
        base_url = args.url.rstrip("/")
    elif args.baseline:
        port = os.environ.get("LLAMA_BASELINE_HOST_PORT", "8000")
        base_url = f"http://localhost:{port}"
    else:
        base_url = os.environ.get("LLAMA_BENCH_URL", "http://localhost:8000").rstrip("/")

    info = probe_server(base_url)
    if not info.ok or not info.model_id:
        print(f"server unreachable at {base_url}: {info.error or 'no model id'}")
        print("start one service first, e.g.:")
        svc = "llamacpp_baseline" if args.baseline else "llamacpp_dflash_ngram"
        print(f"  docker compose -f docker/docker-compose.yaml up -d {svc}")
        return 1
    n_ctx = info.n_ctx or FALLBACK_N_CTX
    print(f"server : {base_url}  ({info.build or 'build unknown'})")
    print(f"model  : {info.model_id}   ctx {n_ctx:,}")
    overrides = {k: v for k, v in {
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "min_p": args.min_p, "repeat_penalty": args.repeat_penalty,
        "max_tokens": args.max_tokens,
    }.items() if v is not None}
    if overrides:
        print("params : " + "  ".join(f"{k} {v}" for k, v in overrides.items())
              + "  (everything else = server/model defaults)")
    else:
        print("params : server/model defaults (no sampler overrides sent)")

    run_id = time.strftime("%Y%m%d-%H%M%S") + "_" + sanitize_name(info.model_id)
    if args.tag:
        run_id += "_" + sanitize_name(args.tag)
    run_dir = BENCH_DIR / run_id
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    print(f"run    : bench_results/{run_id}/")
    print()

    client = OpenAI(base_url=f"{base_url}/v1", api_key="sk-local",
                    timeout=httpx.Timeout(1200.0, connect=5.0))

    settings = {
        "overrides": overrides,
        "system_prompt": args.system,
        "note": "params not in overrides were left to server/model defaults",
    }
    sys_msgs = [{"role": "system", "content": args.system}] if args.system.strip() else []
    conversation: list[dict] = []
    rows: list[dict] = []
    aborted = True  # flipped to False only when every prompt completed

    def persist() -> None:
        write_json_atomic(run_dir / "stats.json", {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "server": {"base_url": base_url, "model": info.model_id,
                       "build": info.build, "n_ctx": n_ctx},
            "settings": settings,
            "aborted": aborted,
            "rows": rows,
            "aggregates": aggregate(rows),
            "aggregates_by_phase": {
                ph: aggregate([r for r in rows if r.get("phase") == ph])
                for ph in ("build", "maint")
            },
        })
        write_json_atomic(run_dir / "transcript.json", {"messages": conversation})

    # --turns trims the run to the first N prompts. Turns 1-9 are the build
    # phase; turn 10 is "show me the complete final file", which is near
    # perfectly draftable (99% accept measured) and inflates any speculation
    # method, so --turns 9 is the fair setting for spec-decoding A/Bs.
    prompts = BENCH_PROMPTS[:args.turns] if args.turns else BENCH_PROMPTS
    total = len(prompts)
    try:
        for i, prompt in enumerate(prompts, 1):
            label = BENCH_LABELS[i - 1]
            # context guard - stop cleanly instead of overflowing the window
            last = rows[-1] if rows else None
            used = ((last or {}).get("prompt_tokens") or 0) + \
                   ((last or {}).get("completion_tokens") or 0)
            if used >= CTX_CRIT_PCT * n_ctx:
                print(f"stopping - context nearly full ({used:,}/{n_ctx:,})")
                break

            prefix = f"[{i}/{total}] {label:<16}"
            result = stream_turn(client, info.model_id,
                                 sys_msgs + conversation + [{"role": "user", "content": prompt}],
                                 args, prefix)
            row = make_row(i, label, result)
            row["phase"] = BENCH_PHASES[i - 1]
            rows.append(row)
            if result["text"]:
                conversation.append({"role": "user", "content": prompt})
                conversation.append({"role": "assistant", "content": result["text"]})
                (run_dir / "responses" / f"{i:02d}.md").write_text(
                    f"# [{i}] {label}\n\n{result['text']}")
            persist()

            ttft = f"TTFT {row['ttft']:.2f}s" if row.get("ttft") is not None else "TTFT -"
            print(f"\r{prefix} {row.get('predicted_n') or 0:6d} tok "
                  f"{fmt(row.get('predicted_per_second')):>7} t/s  {ttft}  "
                  f"accept {accept_str(row)}" + " " * 10)
            if result["error"]:
                print(f"aborting run - turn {i} failed: {result['error']}")
                break
        else:
            aborted = False
    except KeyboardInterrupt:
        print("\ninterrupted - saving partial run")
    finally:
        persist()

    print()
    print(f"=== run {'complete' if not aborted else 'ABORTED'}: {run_id} " + "=" * 20)
    print_run_table(rows)
    print()
    print_aggregates(aggregate(rows))
    for ph in ("build", "maint"):
        agg_ph = aggregate([r for r in rows if r.get("phase") == ph])
        if agg_ph:
            print(f"  {ph:<5} phase   : {fmt(agg_ph['decode_tps'])} t/s   "
                  f"draft accept {agg_accept_str(agg_ph)}")
    print_comparison(args.ref)
    return 0 if not aborted else 2


if __name__ == "__main__":
    sys.exit(main())
