#!/usr/bin/env python3
"""PNG figures for the engine benchmark report, drawn from artifacts only.

Colour: three hues, not four. llama.cpp with and without the draft head is ONE
engine with a feature switched on, so it is one hue in two treatments (solid /
hatched) rather than a fourth category. That is both truer to the data and the
only way the palette clears the all-pairs gate - four distinct hues fail the
normal-vision floor (validate_palette.js, dataviz skill).

Hexes are the reference palette's validated dark steps:
  blue #3987e5  orange #d95926  aqua #199e70   on surface #1B2220
All-pairs: CVD dE 9.4, normal-vision dE 20.9, all >= 3:1 on the surface.
"""
from __future__ import annotations
import pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import report_data as D

R = pathlib.Path(__file__).resolve().parent.parent
OUT = R / "assets"
OUT.mkdir(exist_ok=True)

BG, PANEL, INK, MUTED, GRID = "#1B2220", "#222B29", "#E7EEEC", "#8FA09D", "#2E3A38"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
C = {"llama.cpp": BLUE, "llama.cpp + MTP": BLUE, "SGLang": ORANGE, "FreeToken": AQUA,
     # Same engine, one feature switched on - so the same hue in the hatched
     # treatment, exactly as llama.cpp + MTP relates to llama.cpp. Still three
     # hues, so the all-pairs palette gate is unaffected.
     "SGLang + FlashInfer GDN": ORANGE}
HATCH = {"llama.cpp + MTP": "///", "SGLang + FlashInfer GDN": "///"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "grid.color": GRID, "grid.linewidth": 0.7,
})

def frame(ax, title, sub, foot=None):
    """Title/subtitle/footer as FIGURE text with reserved margins. Drawing them
    through set_title plus a transAxes line collided (title over subtitle, axis
    label over footer) - rendering and looking is step 7 of the method, and it
    caught exactly that."""
    fig = ax.figure
    fig.text(0.012, 0.955, title, color=INK, fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.885, sub, color=MUTED, fontsize=10.5, va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    if foot:
        fig.text(0.012, 0.022, foot, color=MUTED, fontsize=8.5, family="DejaVu Sans Mono", va="bottom")

def save(fig, name, bottom=0.17):
    # No bbox_inches="tight": it re-crops around the figure text and undoes the
    # reserved margins, which is how the footer ended up under the axis label.
    fig.subplots_adjust(left=0.11, right=0.975, top=0.80, bottom=bottom)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  assets/{name}")

# ---------------------------------------------------------------- 1. ladders
KEYS = {"llama.cpp": "llamacpp", "llama.cpp + MTP": "llamacpp+mtp",
        "SGLang": "sglang", "FreeToken": "freetoken"}
STYLE = {"llama.cpp": dict(ls="-", marker="o"), "llama.cpp + MTP": dict(ls="--", marker="s"),
         "SGLang": dict(ls="-", marker="o"), "FreeToken": dict(ls="-", marker="o")}

def _ladder_line(idx, ylab, title, sub, foot, name, ymax=None):
    """Line chart over context. A line shows the SHAPE - which is the whole
    finding here - where grouped bars only show six unrelated comparisons."""
    lad = D.ladder()
    xs = list(lad)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for sname in ("SGLang", "FreeToken", "llama.cpp + MTP", "llama.cpp"):
        k = KEYS[sname]
        pts = [(x, lad[x][k][idx]) for x in xs if k in lad[x] and lad[x][k][idx]]
        if not pts: continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C[sname], linewidth=2.2,
                markersize=7, markeredgecolor=BG, markeredgewidth=1.2, label=sname, zorder=3,
                **STYLE[sname])
        # End labels collide when two series finish near the same value (the MTP
        # line used to end mid-chart while rungs were still running, right on top of
        # FreeToken). Nudge the dashed series up and out of the way.
        x, y = pts[-1]
        off = (6, 13) if sname == "llama.cpp + MTP" else (9, 0)
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=off,
                    color=C[sname], fontsize=10, fontweight="bold",
                    va="center", ha="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs); ax.set_xticklabels([f"{x}K" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel("prompt length"); ax.set_ylabel(ylab)
    ax.set_ylim(0, ymax or None)
    ax.set_xlim(xs[0] * 0.85, xs[-1] * 1.5)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    frame(ax, title, sub, foot)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              labelcolor=INK, fontsize=10)
    save(fig, name, bottom=0.24)

def chart_ladder():
    _ladder_line(1, "decode tok/s",
                 "Decode collapses with context - unless something drafts for you",
                 "tokens per second, one request at a time, 2,048-token output",
                 "dashed = the same llama.cpp build with the draft head on  ·  thinking off  ·  natural prose",
                 "eb_decode_ladder.png", ymax=215)
    _ladder_line(0, "prefill tok/s",
                 "Prefill is where the engines are furthest apart",
                 "tokens per second of prompt ingestion, same runs",
                 "SGLang holds ~8,000 tok/s across the whole range; llama.cpp decays; FreeToken improves with length",
                 "eb_prefill_ladder.png")

# ---------------------------------------------------------------- 2. MTP gain
def chart_mtp():
    m = D.mtp(); fc = D.full_context()
    rows = [(f"{int(a['isl'])//1000}K", a["off"]["decode"], a["on"]["decode"]) for a in m["arms"]]
    if fc.get("llamacpp") and fc.get("llamacpp+mtp"):
        rows.append(("262K", fc["llamacpp"]["decode"], fc["llamacpp+mtp"]["decode"]))
    if not rows: return
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ys = range(len(rows))
    for i, (lab, off, on) in enumerate(rows):
        ax.barh(i - 0.19, off, height=0.34, color=BLUE, alpha=0.45, edgecolor=BG, zorder=3)
        ax.barh(i + 0.19, on, height=0.34, color=BLUE, hatch="///", edgecolor=BG, linewidth=1.2, zorder=3)
        ax.text(off + 2, i - 0.19, f"{off:.0f}", va="center", color=MUTED, fontsize=9)
        ax.text(on + 2, i + 0.19, f"{on:.0f}", va="center", color=INK, fontsize=9, fontweight="bold")
        xmax = max(r[2] for r in rows) * 1.28
        ax.text(xmax * 0.93, i, f"{on/off:.2f}x", va="center", ha="right",
                color=ORANGE, fontsize=12, fontweight="bold")
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("decode tok/s"); ax.set_xlim(0, max(r[2] for r in rows) * 1.28)
    ax.invert_yaxis(); ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    frame(ax, "The draft head pays more the longer the prompt",
          "llama.cpp, same build, draft head off (solid) against on (hatched)",
          "danielhanchen/llama.cpp qwen4exp/mtp @ d1a92352  ·  --spec-type draft-mtp --spec-draft-n-max 5  ·  47-80% of drafted tokens accepted")
    save(fig, "eb_mtp_gain.png")

# ---------------------------------------------------------------- 3. energy
def chart_energy():
    en = D.energy_per_request()
    if not en: return
    order = sorted(en.items(), key=lambda kv: kv[1]["kj"])
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for i, (name, v) in enumerate(order):
        col = C.get(name, BLUE)
        ax.barh(i, v["kj"], height=0.55, color=col, hatch=HATCH.get(name),
                edgecolor=BG, linewidth=1.2, zorder=3)
        ax.text(v["kj"] + 1.5, i, f"{v['kj']:,.0f} kJ", va="center", color=INK,
                fontsize=10, fontweight="bold")
        ax.text(v["kj"] + 16, i, f"{v['seconds']:,.0f} s x {v['watts']:.0f} W",
                va="center", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([o[0] for o in order])
    ax.set_xlabel("energy for one full-window request (kJ)")
    ax.set_xlim(0, max(v["kj"] for _, v in order) * 1.42)
    ax.invert_yaxis(); ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    frame(ax, "Same watts, wildly different bills",
          "one 262,144-token prompt: median power under load x time on the card",
          "every engine draws 358-489 W - the difference is how long it holds them  ·  no thermal throttling in any arm")
    save(fig, "eb_energy.png")

# ---------------------------------------------------------------- 4. boot
def chart_boot():
    bt = D.boot_times()
    if not bt: return
    order = sorted(bt.items(), key=lambda kv: kv[1]["total_med"])
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    for i, (name, b) in enumerate(order):
        col = C.get(name, BLUE)
        boot, gen = b["boot_med"], b["gen_med"] or 0
        ax.barh(i, boot, height=0.5, color=col, edgecolor=BG, linewidth=1.2, zorder=3)
        ax.barh(i, gen, left=boot, height=0.5, color=col, alpha=0.42,
                edgecolor=BG, linewidth=1.2, zorder=3)
        ax.text(boot + gen + 2.5, i, f"{b['total_med']:.0f} s", va="center",
                color=INK, fontsize=10.5, fontweight="bold")
        if b.get("http_med") is not None:
            ax.plot([b["http_med"]], [i], marker="|", markersize=16, color=INK, zorder=5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([o[0] for o in order])
    ax.set_xlabel("seconds from cold container")
    ax.set_xlim(0, max(b["total_med"] for _, b in order) * 1.22)
    ax.invert_yaxis(); ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=MUTED, label="until it serves"),
                       Patch(facecolor=MUTED, alpha=0.42, label="first generation"),
                       plt.Line2D([], [], marker="|", markersize=12, color=INK,
                                  linestyle="none", label="/health returns 200")],
              frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              labelcolor=INK, fontsize=9.5)
    frame(ax, "The fastest engine to run is the slowest to start",
          "cold container to an answer in hand, median of repeated boots",
          "FreeToken answers /health 78 s before it can serve - polling it would rank this table backwards")
    save(fig, "eb_boot.png", bottom=0.26)

# ---------------------------------------------------------------- 5. full window
def chart_full():
    fc = D.full_context()
    if not fc: return
    order = [e for e in ("sglang+fi", "sglang", "freetoken", "llamacpp+mtp", "llamacpp") if e in fc]
    lab = {e: D.LABEL[e] for e in order}
    # Height tracks the bar count: the figure was sized for four and a fifth or
    # sixth arm would crowd the labels.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6 + 0.55 * max(0, len(order) - 4)))
    for ax, key, xlabel, fmt in ((a1, "ttft", "time to first token (s)", lambda v: f"{v/1000:,.0f} s"),
                                 (a2, "decode", "decode tok/s", lambda v: f"{v:,.1f}")):
        vals = [(lab[e], fc[e][key] / (1000 if key == "ttft" else 1), C[lab[e]], HATCH.get(lab[e])) for e in order]
        for i, (nm, v, col, h) in enumerate(vals):
            ax.barh(i, v, height=0.58, color=col, hatch=h, edgecolor=BG, linewidth=1.2, zorder=3)
            ax.text(v * 1.03, i, fmt(fc[order[i]][key]), va="center", color=INK,
                    fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(vals))); ax.set_yticklabels([v[0] for v in vals], fontsize=10)
        ax.set_xlabel(xlabel); ax.invert_yaxis()
        ax.set_xlim(0, max(v[1] for v in vals) * 1.3)
        ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    # The count is drawn from the data, not typed: this headline said "All four"
    # while the chart was being extended, which is the exact class of drift
    # Auto_Bench.md 11 forbids (nothing typed by hand).
    n_word = {1: "One", 2: "Both", 3: "All three", 4: "All four",
              5: "All five", 6: "All six"}.get(len(order), f"All {len(order)}")
    verb = "serves" if len(order) == 1 else "serve"
    fig.text(0.012, 0.955, f"{n_word} {verb} the full window. Only the waiting differs.",
             color=INK, fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.885, "one 262,144-token prompt, 128-token answer", color=MUTED, fontsize=10.5, va="top")
    fig.text(0.012, 0.022, "left: how long before the first token. right: how fast it writes after that.",
             color=MUTED, fontsize=8.5, family="DejaVu Sans Mono", va="bottom")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.78, bottom=0.17, wspace=0.42)
    fig.savefig(OUT / "eb_full_window.png", dpi=150); plt.close(fig)
    print("  assets/eb_full_window.png")

# ---------------------------------------------------------------- 6. accuracy
def chart_accuracy():
    acc = D.accuracy()
    if not acc: return
    def nm(k):
        e, _, tag = k.partition("@")
        base = D.LABEL.get(e, e)
        return base + (" + MTP" if "mtp" in tag else "")
    rows = []
    for k, v in acc.items():
        for task in ("gsm8k", "math500"):
            if task in v: rows.append((nm(k), task, v[task]))
    tasks = [("gsm8k", "GSM8K"), ("math500", "MATH-500")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (tk, tlab) in zip(axes, tasks):
        rs = [r for r in rows if r[1] == tk]
        rs.sort(key=lambda r: -r[2]["accuracy"])
        for i, (name, _, j) in enumerate(rs):
            lo, hi = j["ci95_wilson"]; a = j["accuracy"]
            col = C.get(name.replace(" + MTP", "") if "MTP" in name else name, BLUE)
            ax.errorbar(100 * a, i, xerr=[[100 * (a - lo)], [100 * (hi - a)]], fmt="o",
                        color=col, markersize=9, capsize=5, linewidth=2, zorder=3,
                        markeredgecolor=BG, markeredgewidth=1.2)
            ax.text(100 * hi + 0.25, i, f"{100*a:.2f}%", va="center", color=INK, fontsize=9.5)
        ax.set_yticks(range(len(rs))); ax.set_yticklabels([r[0] for r in rs], fontsize=9.5)
        # Room for the value label; without it the right-hand panel clipped "93.00%".
        los = [100 * r[2]["ci95_wilson"][0] for r in rs]
        his = [100 * r[2]["ci95_wilson"][1] for r in rs]
        span = max(his) - min(los)
        ax.set_xlim(min(los) - span * 0.08, max(his) + span * 0.42)
        ax.set_xlabel(f"{tlab}  (95% interval)"); ax.invert_yaxis()
        ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
    fig.text(0.012, 0.955, "Every interval overlaps every other one", color=INK,
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.885, "exact-match accuracy with 95% confidence intervals", color=MUTED,
             fontsize=10.5, va="top")
    fig.text(0.012, 0.022, "the speed table spans 5.8x; this one spans half a point  ·  errored runs excluded, never scored wrong",
             color=MUTED, fontsize=8.5, family="DejaVu Sans Mono", va="bottom")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.78, bottom=0.19, wspace=0.55)
    fig.savefig(OUT / "eb_accuracy.png", dpi=150); plt.close(fig)
    print("  assets/eb_accuracy.png")

# ---------------------------------------------------------------- 7. load modes
def chart_loadmode():
    arms = D.load_modes()
    if not arms: return
    modes = ["none", "mmap", "mlock", "mmap+mlock", "dio"]
    isls = ["8192", "32000"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    w = 0.38
    for j, isl in enumerate(isls):
        vals = [next((a["prefill"] for a in arms if str(a["isl_label"]) == isl and a["load_mode"] == m), 0)
                for m in modes]
        pos = [i - 0.19 + j * w for i in range(len(modes))]
        ax.bar(pos, vals, width=w * 0.92, color=(BLUE if j == 0 else AQUA), edgecolor=BG,
               linewidth=1.2, label=f"{int(isl)//1000}K prompt", zorder=3)
        for x, v in zip(pos, vals):
            if v: ax.text(x, v + 25, f"{v:,.0f}", ha="center", color=INK, fontsize=8.5)
    ax.set_xticks(range(len(modes))); ax.set_xticklabels([f"--load-mode\n{m}" for m in modes], fontsize=9)
    ax.set_ylabel("prefill tok/s"); ax.set_ylim(0, 2500)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    frame(ax, "Five ways to place 47.68 GiB. None of them matters here.",
          "prefill throughput, same image, same placement, real coding prompts",
          "every mode within 4% at 8K and 0.5% at 32K  ·  no arm ran out of memory  ·  an earlier 1.87x came from a different placement")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              labelcolor=INK, fontsize=10)
    save(fig, "eb_loadmode.png", bottom=0.26)

# ---------------------------------------------------------------- 8. thermals
def chart_thermal():
    th = D.thermals()
    if not th: return
    names = list(th)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for i, n_ in enumerate(names):
        t = th[n_]
        col = C.get(n_, BLUE)
        ax.barh(i, t["temp_med"], height=0.5, color=col, hatch=HATCH.get(n_),
                edgecolor=BG, linewidth=1.2, zorder=3)
        ax.plot([t["temp_max"]], [i], marker="D", markersize=8, color=INK, zorder=5)
        ax.text(t["temp_max"] + 1.5, i, f"peak {t['temp_max']:.0f}", va="center",
                color=MUTED, fontsize=9)
    lim = min(t["temp_max"] + t["margin_min"] for t in th.values())
    ax.axvline(lim, color=ORANGE, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(lim - 1.5, len(names) - 0.35, "thermal limit", color=ORANGE, fontsize=9.5,
            ha="right", fontweight="bold")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("GPU temperature (C)"); ax.set_xlim(0, lim + 8); ax.invert_yaxis()
    ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    frame(ax, "The card is never the constraint",
          "median temperature under load, with peak marked, at the full 262,144-token window",
          "hardware thermal slowdown never engaged in any arm  ·  closest approach to the limit was 12 C of headroom")
    save(fig, "eb_thermal.png", bottom=0.19)

# ---------------------------------------------------------------- 9. KV pool wall
def chart_ftcache():
    fc = D.ft_cache()
    if not fc["sizes"]: return
    rows = sorted(fc["sizes"], key=lambda r: r["kv_tokens_requested"])
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    xs = [r["kv_tokens_requested"] for r in rows]
    served = [(r["kv_tokens_requested"], r["steady"]) for r in rows if r["reps"]]
    dropped = [r["kv_tokens_requested"] for r in rows if not r["reps"]]
    if served:
        ax.plot([p[0] for p in served], [p[1] for p in served], color=AQUA, linewidth=2.4,
                marker="o", markersize=9, markeredgecolor=BG, markeredgewidth=1.2, zorder=3,
                label="request served")
        for x, y in served:
            ax.annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 12),
                        ha="center", color=INK, fontsize=9.5, fontweight="bold")
    for x in dropped:
        ax.plot([x], [0.6], marker="x", markersize=13, markeredgewidth=3, color=ORANGE, zorder=4)
    if dropped:
        ax.annotate("request REFUSED\n(pool smaller than the prompt)",
                    (max(dropped), 0.6), textcoords="offset points", xytext=(14, 26),
                    color=ORANGE, fontsize=10, fontweight="bold")
    prompt_len = next((r["in_tokens"] for r in rows if r.get("in_tokens")), None)
    if prompt_len:
        ax.axvline(prompt_len, color=MUTED, linestyle=":", linewidth=1.6, zorder=2)
        ax.text(prompt_len * 1.05, 10.4, f"the prompt itself\n{prompt_len:,} tokens",
                color=MUTED, fontsize=9.5)
    ax.set_xscale("log", base=2); ax.set_xticks(xs)
    ax.set_xticklabels([f"{x//1024}K" for x in xs]); ax.minorticks_off()
    ax.set_xlabel("KV pool size (tokens)"); ax.set_ylabel("request time (s)")
    ax.set_ylim(0, 12); ax.set_xlim(min(xs) * 0.7, max(xs) * 1.6)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    frame(ax, "A bigger KV cache buys nothing until it buys everything",
          "one request, resized live on a running server, ~1 s per resize",
          "FreeToken: ft ctl cache --kv N  ·  unique prefix per request so the prefix cache cannot answer it")
    save(fig, "eb_ftcache.png", bottom=0.19)

# ---------------------------------------------------- 10. MTP vs VRAM budget
def chart_tier_mtp():
    """The draft head is a big-VRAM feature. One hue in two treatments again:
    it is one engine with a feature switched on, and the x axis is the amount of
    the model that had to be pushed off the card."""
    rows = D.tier_mtp()
    if not rows: return
    rows = [r for r in rows if r.get("off", {}).get("decode")]
    lab = [("no GPU" if r["tier"] == "nogpu" else f"{r['tier']} GiB") for r in rows]
    off = [r["off"]["decode"] for r in rows]
    gpu = [(r.get("gpu") or {}).get("decode") for r in rows]

    fig, ax = plt.subplots(figsize=(11, 4.9))
    x = range(len(rows)); w = 0.38
    ax.bar([i - 0.19 for i in x], off, width=w * 0.92, color=BLUE, edgecolor=BG,
           linewidth=1.2, label="draft head off", zorder=3)
    ax.bar([i + 0.19 for i in x], [g or 0 for g in gpu], width=w * 0.92, color=BLUE,
           edgecolor=BG, linewidth=1.2, hatch="///", alpha=0.75,
           label="draft head on (MTP)", zorder=3)
    for i, v in zip(x, off):
        ax.text(i - 0.19, v + 2.5, f"{v:.0f}", ha="center", color=INK, fontsize=8.5)
    for i, (g, o, r) in enumerate(zip(gpu, off, rows)):
        if g:
            ax.text(i + 0.19, g + 2.5, f"{g:.0f}", ha="center", color=INK, fontsize=8.5)
            ratio = g / o
            ax.text(i + 0.19, g + 11, f"{ratio:.2f}x",
                    ha="center", fontsize=8.5, fontweight="bold",
                    color=(AQUA if ratio > 1 else ORANGE))
        else:
            # "we did not run it" is not the same claim as "it ran and failed".
            # The 8 GiB arm exited non-zero but left an EMPTY artifact - the
            # failure capture could not read an already-removed container (fixed
            # afterwards), so its cause is unrecorded and must not be captioned
            # as an out-of-memory.
            v = (r.get("gpu") or {}).get("verdict")
            note = "not run" if not v else ("no result" if v == "bench-error" else v)
            ax.text(i + 0.19, 4, note, ha="center", va="bottom", rotation=90,
                    fontsize=8.5, color=MUTED, fontweight="bold")
    ax.set_xticks(list(x))
    # Short second line: the full phrase collided between ticks at this width.
    ax.set_xticklabels([f"{l}\n{r['n_cpu_moe']} on CPU" for l, r in zip(lab, rows)], fontsize=9)

    ax.set_ylabel("decode tok/s"); ax.set_ylim(0, 185)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    frame(ax, "Multi-token prediction is a big-VRAM feature",
          "decode tok/s at a 2,048-token prompt, single stream, VRAM held down with a balloon allocator",
          "the draft head pays only when NO experts are offloaded  ·  at 24 GiB it costs 3.4x  ·  "
          "capping VRAM reproduces capacity, not a 4090's bandwidth")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              labelcolor=INK, fontsize=10)
    save(fig, "eb_tier_mtp.png", bottom=0.28)

if __name__ == "__main__":
    print("charts ->")
    chart_ladder(); chart_mtp(); chart_energy(); chart_boot()
    chart_full(); chart_accuracy(); chart_loadmode(); chart_thermal(); chart_ftcache()
    chart_tier_mtp()
