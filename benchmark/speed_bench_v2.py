# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27"]
# ///
"""Speed bench v2: prefill (prompt processing) + decode vs prompt length.

Answers the question the aiperf speed sweep cannot: what is the raw
prompt-processing (prefill) throughput of each stack, and how does it scale
with prompt length? DFlash / MTP draft heads must ingest the prompt too, so
their prefill cost differs from the baseline - this script measures exactly
that, per server config, straight from the llama.cpp `timings` block:

  prompt_n / prompt_ms / prompt_per_second        prefill (server-side TTFT)
  predicted_n / predicted_ms / predicted_per_second   decode at that depth
  draft_n / draft_n_accepted                      speculation acceptance

Prompt content: real LiveCodeBench problems (benchmark/data/
lcb_release_v5_first100.inputs.json) concatenated to the exact target token
count, always ending with one COMPLETE problem plus an instruction to solve
it. The decode reading therefore measures the model writing a fresh Python
solution - real generation - not greedy continuation of visible prompt text,
which drafters copy perfectly and which inflates acceptance to 100%. The
final problem rotates per request (same rotation for every config, so A/B
stays fair) and each prompt carries a unique prefix + cache_prompt:false so
every request is a full cold prefill.

llama-bench is NOT usable here: it cannot load a speculative draft model,
so it would show baseline prefill only.

Workflow (same one-server-at-a-time flow as bench_ngram.py):
  1. start ONE docker service (baseline / dflash / mtp / dflash+ngram)
  2. uv run benchmark/speed_bench_v2.py        # auto-detects :8000/:8001
  3. swap the docker service, run again
  4. uv run benchmark/speed_bench_v2_summary.py   # cross-config tables

Each run saves artifacts/<tag>/speed_v2/run_<ts>.json (per-request rows +
per-length aggregates) and appends aggregate rows to
benchmark/results_speed_v2.csv - the summary script reads the CSV.
"""

# Copied from deepseek-v4-flash-dspark-rtx6000pro@d790b4f, benchmark/.
# Kept deliberately close to the original so results stay comparable across
# the three studies. Adaptations for this repo:
#   - default port/URL detection unchanged (127.0.0.1, autodetect)
#   - ALIAS_TO_TAG: added qwen3.8-flash-next -> iq4xs
#   - DEFAULT_LENGTHS: swapped to this repo's Phase 1 ladder (256..262144)

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ARTIFACTS_DIR = REPO_DIR / "artifacts"
RESULTS_CSV = SCRIPT_DIR / "results_speed_v2.csv"
LCB_INPUTS = SCRIPT_DIR / "data" / "lcb_release_v5_first100.inputs.json"

CANDIDATE_PORTS = (8000, 8001)
FALLBACK_N_CTX = 262144
CTX_MARGIN = 64  # slack tokens kept free below n_ctx

# Phase 1 ladder from PLAN.md. 262144 is this model's native context; the
# top two rungs need a load check before they are worth queueing.
DEFAULT_LENGTHS = (256, 2048, 8192, 32768, 131072, 262144)

# the model gets this after the final (complete) problem; decode then
# measures fresh solution-writing instead of prompt-text continuation
TASK_SUFFIX = (
    "\n\nInstruction: write a complete, correct Python solution for the "
    "LAST problem shown above. Explain your approach in code comments as "
    "you go."
)

TIMING_KEYS = (
    "prompt_n",
    "prompt_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_second",
    "draft_n",
    "draft_n_accepted",
)

# map server aliases (docker-compose --alias) to the short artifact tags the
# repo already uses; unknown aliases fall back to a sanitized alias
ALIAS_TO_TAG = {
    "qwen3.8-flash-next": "iq4xs",
    "qwen3.6-27B": "base",
    "qwen36-dflash15": "dflash",
    "qwen3.6-27b-mtp8": "mtp",
    "qwen36-dflash-ngram": "dflash_ngram",
}


@dataclass
class ServerInfo:
    base_url: str = ""
    model_id: str | None = None
    n_ctx: int | None = None
    build: str | None = None


def detect_server(host: str, port: int | None, url: str | None) -> str:
    if url:
        return url.rstrip("/")
    if port is not None:
        return f"http://{host}:{port}"
    for candidate in CANDIDATE_PORTS:
        base = f"http://{host}:{candidate}"
        try:
            if httpx.get(f"{base}/health", timeout=3.0).status_code == 200:
                return base
        except Exception:
            continue
    raise SystemExit(f"No healthy server found on {host} ports {CANDIDATE_PORTS}.")


def probe_server(base_url: str) -> ServerInfo:
    info = ServerInfo(base_url=base_url)
    try:
        data = (httpx.get(f"{base_url}/v1/models", timeout=10.0).json().get("data")) or []
        if data:
            info.model_id = data[0].get("id")
            meta = data[0].get("meta") or {}
            if isinstance(meta.get("n_ctx"), int):
                info.n_ctx = meta["n_ctx"]
    except Exception:
        pass
    try:
        props = httpx.get(f"{base_url}/props", timeout=10.0).json()
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


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Prompt construction: LCB problems, exact token counts via server /tokenize
# ----------------------------------------------------------------------------


def tokenize(client: httpx.Client, base_url: str, text: str) -> list[int]:
    r = client.post(
        f"{base_url}/tokenize",
        json={"content": text, "add_special": False},
        timeout=300.0,
    )
    r.raise_for_status()
    return r.json()["tokens"]


def load_problem_texts() -> list[str]:
    data = json.loads(LCB_INPUTS.read_text())
    problems: list[str] = []
    for item in data.get("data", []):
        for payload in item.get("payloads", []):
            parts = [m["content"] for m in payload.get("messages", []) if m.get("content")]
            if parts:
                problems.append("\n\n".join(parts))
    if not problems:
        raise SystemExit(f"No problem content found in {LCB_INPUTS}")
    return problems


def tokenize_problems(client: httpx.Client, base_url: str) -> list[list[int]]:
    texts = load_problem_texts()
    print(f"tokenizing {len(texts)} LCB problems ...", end="", flush=True)
    toks = [tokenize(client, base_url, t) for t in texts]
    print(f" done ({sum(len(t) for t in toks)} tokens total)")
    return toks


def make_prompt_ids(client: httpx.Client, base_url: str,
                    problems: list[list[int]], suffix_ids: list[int],
                    final_idx: int, length: int) -> list[int]:
    """Exactly `length` ids: [unique prefix][padding problems][final problem,
    complete][task suffix]. The unique prefix defeats prefix caching even if
    a build ignored cache_prompt:false; the final problem is kept whole so
    'solve the LAST problem' is a real task."""
    prefix = tokenize(client, base_url, f"[{uuid.uuid4().hex[:8]}] ")
    budget = length - len(prefix) - len(suffix_ids)
    if budget <= 0:
        raise SystemExit(f"length {length} too small for prefix+suffix")

    final = problems[final_idx % len(problems)]
    if len(final) >= budget:
        # tiny prompt: keep the tail of the problem (statement end survives)
        return prefix + final[len(final) - budget:] + suffix_ids

    pad_needed = budget - len(final)
    padding: list[int] = []
    i = final_idx + 1
    while len(padding) < pad_needed:
        padding.extend(problems[i % len(problems)])
        i += 1
    padding = padding[len(padding) - pad_needed:]  # front-truncate to fit
    ids = prefix + padding + final + suffix_ids
    assert len(ids) == length
    return ids


# ----------------------------------------------------------------------------
# One measured request
# ----------------------------------------------------------------------------


def run_request(client: httpx.Client, base_url: str, prompt_ids: list[int],
                n_predict: int) -> dict:
    payload = {
        "prompt": prompt_ids,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,  # llama.cpp extension: force a cold prefill
        # Qwen3.8-Flash-Next emits EOS after ~1 token on a raw completion prompt,
        # so without this every decode sample is discarded as an early stop and
        # the whole decode column comes back empty. Forcing the full n_predict is
        # also what llama-bench's tg128 does: for a THROUGHPUT measurement we want
        # a fixed token count, not however many the model felt like emitting.
        # It does mean decode here is measured on continuation tokens rather than
        # a natural answer, which is the standard trade and worth stating.
        "ignore_eos": True,
    }
    start = time.perf_counter()
    r = client.post(f"{base_url}/completion", json=payload, timeout=1800.0)
    r.raise_for_status()
    wall_s = time.perf_counter() - start
    body = r.json()
    timings = body.get("timings") or {}
    row = {k: timings.get(k) for k in TIMING_KEYS}
    row["wall_s"] = round(wall_s, 3)
    row["text_head"] = (body.get("content") or "")[:200]
    return row


def decode_valid(row: dict, n_predict: int) -> bool:
    """Early-stopped generations give unusable decode readings (in the worst
    case predicted_ms ~ 0 and llama.cpp reports absurd t/s)."""
    need = min(n_predict, max(64, n_predict // 4))
    return (row.get("predicted_n") or 0) >= need


def check_row(row: dict, length: int) -> None:
    prompt_n = row.get("prompt_n")
    if prompt_n is None:
        print("    WARN: no timings block in response (old server build?)")
        return
    if prompt_n < 0.98 * length:
        print(f"    WARN: prompt_n={prompt_n} << requested {length} "
              "(prefix cache hit? results invalid for this row)")
    elif prompt_n > 1.05 * length:
        print(f"    WARN: prompt_n={prompt_n} >> requested {length}")


# ----------------------------------------------------------------------------
# Aggregation + persistence
# ----------------------------------------------------------------------------


def med(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def aggregate(length: int, rows: list[dict]) -> dict:
    def col(key, subset):
        return [r.get(key) for r in subset]

    valid = [r for r in rows if r.get("decode_valid")]
    draft_n = [r.get("draft_n") for r in valid if r.get("draft_n")]
    accepted = [r.get("draft_n_accepted") for r in valid if r.get("draft_n")]
    accept_pct = (
        round(100.0 * sum(accepted) / sum(draft_n), 1) if draft_n and sum(draft_n) else None
    )
    pp = [v for v in col("prompt_per_second", rows) if v is not None]
    itl = [
        r["predicted_ms"] / r["predicted_n"]
        for r in valid
        if r.get("predicted_ms") and r.get("predicted_n")
    ]
    return {
        "length": length,
        "reps": len(rows),
        "pp_tps_med": med(col("prompt_per_second", rows)),
        "pp_tps_min": round(min(pp), 2) if pp else None,
        "pp_tps_max": round(max(pp), 2) if pp else None,
        "ttft_ms_med": med(col("prompt_ms", rows)),
        "tg_tps_med": med(col("predicted_per_second", valid)),
        "itl_ms_med": round(statistics.median(itl), 3) if itl else None,
        "accept_pct": accept_pct,
        "wall_s_med": med(col("wall_s", rows)),
    }


CSV_COLUMNS = [
    "timestamp", "tag", "alias", "build", "source", "n_predict",
    "length", "reps", "pp_tps_med", "pp_tps_min", "pp_tps_max",
    "ttft_ms_med", "tg_tps_med", "itl_ms_med", "accept_pct", "wall_s_med",
]


def append_csv(rows: list[dict]) -> None:
    new_file = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def print_agg_line(agg: dict) -> None:
    def fmt(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "-"

    print(
        f"  {agg['length']:>7} tok | prefill {fmt(agg['pp_tps_med'])} t/s "
        f"(min {fmt(agg['pp_tps_min'])}, max {fmt(agg['pp_tps_max'])}) | "
        f"TTFT {fmt(agg['ttft_ms_med'])} ms | decode {fmt(agg['tg_tps_med'])} t/s | "
        f"accept {fmt(agg['accept_pct'], '%')}"
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--tag", default=None,
                        help="Artifact tag (default: derived from server alias).")
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)),
                        help="Comma-separated prompt token lengths.")
    parser.add_argument("--n-predict", type=int, default=256,
                        help="Tokens to generate per request (decode + acceptance reading).")
    parser.add_argument("--no-ledger", action="store_true",
                        help="Do not append to the shared results_speed_v2.csv. "
                             "Correction runs must not contaminate the historical "
                             "ledger, whose rows were produced under superseded "
                             "methods; the per-run JSON artifact is the record.")
    parser.add_argument("--num-prompts", type=int, default=3,
                        help="Measured prompts per length (+1 discarded warmup).")
    args = parser.parse_args()

    lengths = sorted({int(x) for x in args.lengths.split(",") if x.strip()})

    base_url = detect_server(args.host, args.port, args.url)
    info = probe_server(base_url)
    alias = info.model_id or "unknown"
    tag = args.tag or ALIAS_TO_TAG.get(alias, sanitize_name(alias))
    # spec-decoding expectation comes from the known alias, not the tag,
    # so custom --tag values do not trigger false warnings
    known = ALIAS_TO_TAG.get(alias)
    is_spec = known is not None and known != "base"

    print(f"server={base_url} alias={alias} tag={tag} n_ctx={info.n_ctx} "
          f"build={info.build}")
    print(f"lengths={lengths} n_predict={args.n_predict} "
          f"num_prompts={args.num_prompts} source=lcb")

    client = httpx.Client()
    problems = tokenize_problems(client, base_url)
    suffix_ids = tokenize(client, base_url, TASK_SUFFIX)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows: list[dict] = []
    aggs: list[dict] = []
    draft_seen = False
    req_i = 0  # global request counter, rotates the final problem

    for length in lengths:
        if length + args.n_predict + CTX_MARGIN > (info.n_ctx or FALLBACK_N_CTX):
            print(f"  {length:>7} tok | SKIP (n_ctx={info.n_ctx} too small)")
            continue
        rows: list[dict] = []
        for rep in range(args.num_prompts + 1):  # rep 0 is a discarded warmup
            is_warmup = rep == 0
            # stride 7 spreads final problems across the dataset
            ids = make_prompt_ids(client, base_url, problems, suffix_ids,
                                  final_idx=req_i * 7, length=length)
            req_i += 1
            label = "warmup" if is_warmup else f"prompt {rep}/{args.num_prompts}"
            print(f"  {length:>7} tok | {label} ...", end="", flush=True)
            try:
                row = run_request(client, base_url, ids, args.n_predict)
            except Exception as e:
                print(f" ERROR: {type(e).__name__}: {e}")
                continue
            row["decode_valid"] = decode_valid(row, args.n_predict)
            pp = row.get("prompt_per_second")
            tg = row.get("predicted_per_second")
            print(f" prefill {round(pp, 1) if pp else '-'} t/s | "
                  f"decode {round(tg, 1) if tg and row['decode_valid'] else '-'} t/s"
                  f"{' (discarded)' if is_warmup else ''}")
            check_row(row, length)
            if not row["decode_valid"]:
                print(f"    WARN: only {row.get('predicted_n') or 0}/{args.n_predict} "
                      "tokens generated (early stop) - decode stats excluded")
            if row.get("draft_n"):
                draft_seen = True
            if not is_warmup:
                row.update({"length": length, "rep": rep})
                rows.append(row)
                all_rows.append(row)
        if rows:
            agg = aggregate(length, rows)
            aggs.append(agg)
            print_agg_line(agg)

    if is_spec and not draft_seen:
        print("WARN: no draft_n in any response - is the intended spec-decoding "
              "server actually the one running?")
    if known == "base" and draft_seen:
        print("WARN: draft_n present but this alias is the baseline - wrong server?")

    if not aggs:
        print("No results collected.")
        return 1

    run_dir = ARTIFACTS_DIR / tag / "speed_v2"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = run_dir / f"run_{timestamp}.json"
    write_json_atomic(run_json, {
        "server": {"base_url": base_url, "alias": alias, "n_ctx": info.n_ctx,
                   "build": info.build},
        "settings": {"tag": tag, "lengths": lengths, "n_predict": args.n_predict,
                     "num_prompts": args.num_prompts, "source": "lcb",
                     "timestamp": timestamp},
        "rows": all_rows,
        "per_length": aggs,
    })

    csv_rows = [
        {"timestamp": timestamp, "tag": tag, "alias": alias, "build": info.build,
         "source": "lcb", "n_predict": args.n_predict, **agg}
        for agg in aggs
    ]
    if not args.no_ledger:
        append_csv(csv_rows)

    print(f"\nsaved {run_json}")
    if args.no_ledger:
        print(f"--no-ledger: {len(csv_rows)} rows NOT appended to {RESULTS_CSV}")
    else:
        print(f"appended {len(csv_rows)} rows to {RESULTS_CSV}")
    print("next: run the other configs, then "
          "`uv run benchmark/speed_bench_v2_summary.py`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
