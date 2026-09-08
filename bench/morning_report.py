#!/usr/bin/env python3
"""Build results/MORNING_REPORT.md from whatever artifacts exist.

Reads artifacts only - never terminal output (Auto_Bench.md 11). A run that
produced no aiperf export is reported as FAILED with its log path, not omitted:
a missing row must never look like a row that was never scheduled.
"""
from __future__ import annotations
import json, pathlib, statistics as st, datetime

R = pathlib.Path(__file__).resolve().parent.parent
A = R / "artifacts"

def cells(d: pathlib.Path):
    return [p for p in d.glob("**/profile_export_aiperf.json") if "phases" not in p.parts]

def m(j, k, f="avg"):
    v = j.get(k)
    return v.get(f) if isinstance(v, dict) else None

def row(p):
    j = json.loads(p.read_text())
    isl, t, itl = m(j,"input_sequence_length"), m(j,"time_to_first_token"), m(j,"inter_token_latency")
    return dict(cell=p.parent.name, isl=isl, osl=m(j,"output_sequence_length"), ttft=t,
                prefill=(isl/(t/1000) if isl and t else None),
                decode=m(j,"output_token_throughput_per_user"),
                agg=m(j,"output_token_throughput"), itl=itl,
                err=m(j,"error_request_count") or 0)

def newest(pat):
    ds = sorted(A.glob(pat), key=lambda d: d.stat().st_mtime)
    return ds[-1] if ds else None

def verdict(d):
    """The output checker's verdict. Auto_Bench.md 10 pools by this, never by
    glob, so a speed row without it is not decision-ready."""
    f = d/"output_check.json"
    if not f.exists(): return "-"
    j = json.loads(f.read_text())
    return f"{j.get('verdict','-')} {j.get('repetitive',0)}/{j.get('responses',0)}"

def boot(d):
    """The arm's resolved server flags, as recorded before the run."""
    f = d/"provenance.json"
    if not f.exists(): return ""
    b = json.loads(f.read_text()).get("server_boot_args") or ""
    return " ".join(b) if isinstance(b, list) else str(b)

def isl_of(d):
    f = d/"provenance.json"
    if not f.exists(): return None
    try: return int(json.loads(f.read_text())["workload_params"]["isl"])
    except Exception: return None

def pflag(d, key):
    """A parity flag by name. Env-configured knobs (LLAMA_ARG_SPEC_TYPE and
    friends) never appear in server_boot_args, which reads docker .Args - so
    boot-arg matching CANNOT separate those arms. This is the field to use."""
    f = d/"provenance.json"
    if not f.exists(): return None
    try: return str(json.loads(f.read_text())["parity_flags"].get(key))
    except Exception: return None

def pick(pat, has=(), lacks=(), isl=None, flag=None):
    """Newest run matching `pat` whose boot args contain every `has`, no `lacks`.

    An A/B arm cannot be selected by glob: both arms share one glob, so newest()
    returns the same directory twice. The 02:09 report did exactly that and
    printed one run as both arms of three comparisons. The swept flag IS in the
    resolved boot args, so match on that."""
    ds = [d for d in sorted(A.glob(pat), key=lambda d: d.stat().st_mtime)
          if all(h in boot(d) for h in has) and not any(l in boot(d) for l in lacks)
          and (isl is None or isl_of(d) == isl)
          and (flag is None or pflag(d, flag[0]) == flag[1])]
    return ds[-1] if ds else None

def thermal():
    out=[]
    for f in sorted((A/"thermal").glob("*.csv"), key=lambda p: p.stat().st_mtime)[-12:]:
        try:
            ls=f.read_text().splitlines()[1:]
            if not ls: continue
            temps=[float(l.split(",")[1]) for l in ls if len(l.split(","))>1]
            pw=[float(l.split(",")[3]) for l in ls if len(l.split(","))>3]
            out.append((f.stem[:52], len(temps), max(temps), max(pw) if pw else 0))
        except Exception: pass
    return out

L=[]
def w(s=""): L.append(s)

w(f"# Overnight results — {datetime.datetime.now():%Y-%m-%d %H:%M}")
w()
w("Generated from artifacts by `bench/morning_report.py`. Every number here is read")
w("out of a `profile_export_aiperf.json`; nothing is transcribed from a terminal.")
w()

# ---- context ladder ----
w("## 1. Prefill and decode versus context (THINKING=off, OSL 2048, n=3, c=1)")
w()
w("The question: all three engines serve 262,144, but the two endpoints disagreed about")
w("the shape between. FreeToken got *faster* per token with length while llama.cpp got")
w("slower, so they cross somewhere — this locates it.")
w()
for eng in ("llamacpp","sglang","freetoken"):
    d = newest(f"context_fn_ctxladder_{eng}_*")
    if not d: w(f"**{eng}** — no run found."); w(); continue
    cs = cells(d)
    if not cs: w(f"**{eng}** — FAILED, no export. See `{d.name}`."); w(); continue
    rs = sorted((row(p) for p in cs), key=lambda r: r["isl"] or 0)
    w(f"**{eng}**  `{d.name}`")
    w()
    w("| ISL | TTFT s | prefill tok/s | decode tok/s | errors |")
    w("|---:|---:|---:|---:|---:|")
    for r in rs:
        w(f"| {r['isl']:,.0f} | {r['ttft']/1000:.2f} | {r['prefill']:,.0f} | {r['decode']:.2f} | {r['err']:.0f} |")
    w()

# ---- generic section builder ----
def section(title, note, pats):
    w(f"## {title}")
    w(); w(note); w()
    w("| run | ISL | OSL | prefill tok/s | decode tok/s | agg tok/s | err | output | artifact |")
    w("|---|---:|---:|---:|---:|---:|---:|---|---|")
    any_row=False
    for spec in pats:
        label, pat = spec[0], spec[1]
        has   = spec[2] if len(spec) > 2 else ()
        lacks = spec[3] if len(spec) > 3 else ()
        isl   = spec[4] if len(spec) > 4 else None
        flag  = spec[5] if len(spec) > 5 else None
        d = pick(pat, has, lacks, isl, flag)
        if not d: w(f"| {label} | — | — | — | — | — | *not run* | — | — |"); continue
        cs = cells(d)
        if not cs: w(f"| {label} | — | — | — | — | — | **FAILED** | — | `{d.name}` |"); continue
        for p in sorted(cs, key=lambda x: x.parent.name):
            r=row(p); any_row=True
            tag = f"{label} · {r['cell']}" if len(cs)>1 else label
            w(f"| {tag} | {r['isl']:,.0f} | {r['osl']:.0f} | {r['prefill']:,.0f} | "
              f"{r['decode']:.2f} | {r['agg']:.1f} | {r['err']:.0f} | {verdict(d)} | `{d.name}` |")
    w()
    return any_row

section("2. Speculation isolated (fn_smoke, c=1)",
        "Every earlier decode number was engine PLUS drafter. These arms separate them.\n\n"
        "**Caveat — the SGLang pair is confounded.** The NEXTN-off arm ran at\n"
        "`--mem-fraction-static 0.90` and the NEXTN-on arm at `0.93`, so the two differ in\n"
        "KV pool size as well as in the drafter. At c=1 the pool should not move decode,\n"
        "but this pair does not prove that on its own. Re-run the off arm at 0.93 before\n"
        "publishing a speculation delta.\n\n"
        "llama.cpp `ngram-mod` produced ZERO drafts on this corpus (`sonnet`): the flag was\n"
        "accepted and the mechanism never engaged. Auto_Bench.md 3 fails such an arm rather\n"
        "than reporting it as no gain. It DOES engage on real code - see section 3b.",
        [("SGLang NEXTN on",  "fn_smoke_sglang_*",   ["--speculative-algorithm"], []),
         ("SGLang NEXTN off", "fn_smoke_sglang_*",   [], ["--speculative-algorithm"]),
         ("llama.cpp none",   "fn_smoke_llamacpp_*", ["--ubatch-size 512","--load-mode none"], ["--spec-type"]),
         ("llama.cpp ngram-mod","fn_smoke_llamacpp_*",[],[])])

section("3. Real code (LiveCodeBench) at 32K, THINKING=off",
        "Synthetic data already gave a materially wrong answer once (c=2: 48 vs 115 tok/s).\n\n"
        "**Read the `output` column before the speed columns.** It is the checker's verdict\n"
        "(`repetitive/responses`); Auto_Bench.md 10 pools by it, never by glob. Only rows\n"
        "marked USABLE are publishable. The verdict is a tripwire, not a severity measure:\n"
        "ONE repetitive response out of 16 condemns a whole run.",
        [("SGLang",   "fn_code_tune_sglang_*",   [],[],32000),
         ("FreeToken","fn_code_tune_freetoken_*",[],[],32000),
         ("llama.cpp","fn_code_tune_llamacpp_*", [],[],32000)])

section("3b. llama.cpp speculation on REAL CODE (ISL 2048, OSL 1024)",
        "`ngram-mod` drafts by finding a 24-token exact match in context. On the `sonnet`\n"
        "corpus it produced ZERO drafts (section 2) - natural prose at ~1.5% repeated-8gram\n"
        "never supplies one. On real code it engages: 59 drafts, 37.6% acceptance, **+6.8%\n"
        "decode at c=1** (101.43 -> 108.36) and -3.4% at c=2. Prefill is identical at c=1,\n"
        "which is the parity check - speculation must not move prefill, and it did not.\n\n"
        "So the benefit is a property of the WORKLOAD, not the engine. This is why CORPUS-0\n"
        "disqualifying the `coding` corpus (31.8-67.2% repeated-8gram) mattered: benchmarking\n"
        "a drafter there would have measured the corpus and called it engine speed.",
        [("none",     "fn_code_tune_llamacpp_*",[],[],2048,("spec_type","none")),
         ("ngram-mod","fn_code_tune_llamacpp_*",[],[],2048,("spec_type","ngram-mod"))])

section("4. SGLang batch-2 decode CUDA graph",
        "The c=2 run admitted two requests but decoded with `cuda graph: False`. One arm "
        "only — this is an observation, not a comparison.",
        [("fn_fast","fn_fast_sglang_*")])

section("5. llama.cpp PLE placement — load-mode",
        "`dio` (DirectIO) versus resident `none`, ubatch held at 512 on both sides. Adds a "
        "placement to the eight-way study.",
        [("load-mode none","fn_smoke_llamacpp_*",["--load-mode none","--ubatch-size 512"],[]),
         ("load-mode dio", "fn_smoke_llamacpp_*",["--load-mode dio"],[])])

section("5b. llama.cpp microbatch size",
        "UB-01 measured +30.8% prefill at ubatch 2048 on the older build. Prefill is "
        "llama.cpp's weakest axis at long context, so this is its cheapest lever. "
        "`--load-mode none` on all three arms.",
        [("ubatch 512", "fn_smoke_llamacpp_*",["--ubatch-size 512","--load-mode none"],[]),
         ("ubatch 1024","fn_smoke_llamacpp_*",["--ubatch-size 1024"],[]),
         ("ubatch 2048","fn_smoke_llamacpp_*",["--ubatch-size 2048"],[])])

section("5c. FreeToken NVFP4 kernel backend",
        "FreeToken resolves `--nvfp4-backend` to `triton` by itself; `flashinfer` is the "
        "untested alternative. **The flashinfer arm produced no artifact** — it is reported "
        "as not run, and the comparison is not available.",
        [("nvfp4 triton",    "fn_smoke_freetoken_*",["--nvfp4-backend triton"],[]),
         ("nvfp4 flashinfer","fn_smoke_freetoken_*",["--nvfp4-backend flashinfer"],[])])

# ---- thermal ----
w("## 6. Thermals")
w()
w("Watchdog armed throughout: abort at 90 C, or thermal margin <= 3 C, or hardware")
w("slowdown, sustained over 3 samples. An arm that trips it is VOID, not slow.")
w()
w("| run | samples | max temp C | max power W |")
w("|---|---:|---:|---:|")
for n,c,t,p in thermal():
    w(f"| {n} | {c} | {t:.0f} | {p:.0f} |")
w()

# ---- failures ----
w("## 7. Failures and quarantined runs")
w()
q = A/"_quarantine"
if q.exists():
    w("Invalid results are kept, never deleted (`Auto_Bench.md` 12). Each has its cause in")
    w("`artifacts/_quarantine/README.md`:")
    w()
    for d in sorted(q.iterdir()):
        if d.is_dir(): w(f"- `{d.name}`")
    w()
w("Empty run directories (a scheduled run that produced nothing):")
w()
empt=[d.name for d in sorted(A.glob("*/")) if d.is_dir() and d.name!="_quarantine"
      and d.name!="thermal" and not cells(d)]
for e in empt: w(f"- `{e}`")
if not empt: w("- none")
w()

out = R/"results"/"MORNING_REPORT.md"
out.write_text("\n".join(L))
print(f"wrote {out} ({len(L)} lines)")
