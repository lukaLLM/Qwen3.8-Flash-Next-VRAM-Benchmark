#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Render the README charts from the shipped evidence directories.

Nothing here is typed by hand: every number is read from the same artifacts the
report cites, and each chart prints the values it drew so they can be checked
against the README tables.

  ladder_decode.png     - decode tok/s per VRAM tier at a 2,048-token prompt
  ladder_vs_context.png - decode tok/s against prompt length, one line per tier
  ple_placement.png     - the PLE lookup table on CPU against on GPU
  concurrency.png       - aggregate tok/s against concurrent clients, KV layouts

Each chart is written twice: <name>.png on a light surface and <name>_dark.png
for dark GitHub, selected in the README with a <picture> element.

Regenerate after a rerun:  uv run benchmark/plot_results.py
"""

from __future__ import annotations

import csv
import json
import re
from math import log2
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"
RESULTS = REPO / "results"

# Palette shared with the DFlash and DSpark write-ups: blue carries the story,
# green the second series, gray the reference. Identity never rides on color
# alone - every chart names its ticks or lines and labels its values directly.
LIGHT = dict(
    blue="#2a78d6", green="#1f9d63", gray="#898781",
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
    muted="#898781", grid="#e1e0d9",
    ramp_lo=(0.78, 0.86, 0.95), ramp_hi=(0.05, 0.24, 0.44),
)
# Dark surface matches GitHub's dark theme. The tier ramp brightens with
# capacity here (the largest tier must be the most visible on a dark ground),
# where the light ramp deepens instead.
DARK = dict(
    blue="#4d96e8", green="#2fbf7f", gray="#8b949e",
    surface="#0d1117", ink="#e6edf3", ink2="#9aa4b2",
    muted="#8b949e", grid="#2d333b",
    ramp_lo=(0.24, 0.31, 0.43), ramp_hi=(0.74, 0.87, 1.00),
)

P = LIGHT          # active palette, set per pass in main
SUFFIX = ""        # "" or "_dark"
VERBOSE = True     # print drawn values on the light pass only

TIERS = [8, 16, 24, 32, 48, 96]
TIER_NAME = {8: "3060 / 4060", 16: "4060 Ti", 24: "3090 / 4090",
             32: "5090", 48: "2× 3090", 96: "PRO 6000"}


def _lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def tier_color(c):
    i = TIERS.index(c)
    return _lerp(P["ramp_lo"], P["ramp_hi"], i / (len(TIERS) - 1))


def style_axes(ax, y_grid=True):
    ax.set_facecolor(P["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(P["grid"])
    if y_grid:
        ax.grid(axis="y", color=P["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(colors=P["muted"], labelcolor=P["muted"], length=0,
                   labelsize=10.5)


def bar_round(ax, x, h, w, color):
    """A bar with rounded top corners, drawn as a path in data coordinates."""
    ry = min(h * 0.5, 0.035 * ax.get_ylim()[1])
    rx = w * 0.30
    x0, x1 = x - w / 2, x + w / 2
    verts = [(x0, 0), (x0, h - ry), (x0, h), (x0 + rx, h), (x1 - rx, h),
             (x1, h), (x1, h - ry), (x1, 0), (x0, 0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
             MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0))


def titles(ax, title, sub):
    ax.set_title(title, color=P["ink"], fontsize=14, fontweight="bold",
                 loc="left", pad=24)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9.5, color=P["ink2"])


def xlabel(ax, text):
    ax.set_xlabel(text, color=P["muted"], fontsize=10.5, labelpad=8)


def ylabel(ax, text):
    ax.set_ylabel(text, color=P["muted"], fontsize=10.5, labelpad=10)


def say(*a):
    if VERBOSE:
        print(*a)


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"{name}{SUFFIX}.png"
    fig.savefig(out, facecolor=P["surface"], bbox_inches="tight",
                pad_inches=0.25)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


# ---------- data ----------

def ladder_rows():
    rows = list(csv.DictReader((RESULTS / "tier_full" / "full.csv").open()))
    for r in rows:
        for k in ("cap", "length"):
            r[k] = int(r[k])
        for k in ("prefill_tps", "decode_tps"):
            r[k] = float(r[k])
    return rows


def cpu_decode_2k():
    t = (RESULTS / "cpu_only" / "speed.log").read_text()
    m = re.search(r"2048 tok \| prefill [\d.]+ t/s \(min.*?decode ([\d.]+)", t)
    return float(m.group(1))


# ---------- charts ----------

def chart_ladder_decode():
    rows = ladder_rows()
    at2k = {r["cap"]: r["decode_tps"] for r in rows if r["length"] == 2048}
    cpu = cpu_decode_2k()
    vals = [("no GPU", cpu, P["gray"])] + [
        (f"{c} GiB", at2k[c], tier_color(c)) for c in TIERS]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=200)
    fig.patch.set_facecolor(P["surface"])
    ax.set_ylim(0, max(v for _, v, _ in vals) * 1.18)
    style_axes(ax)
    for i, (label, v, color) in enumerate(vals):
        bar_round(ax, i, v, 0.62, color)
        ax.text(i, v + ax.get_ylim()[1] * 0.02, f"{v:.1f}", ha="center",
                fontsize=12, color=P["ink"], fontweight="bold")
    ax.set_xticks(range(len(vals)))
    names = ["CPU only"] + [TIER_NAME[c] for c in TIERS]
    ax.set_xticklabels([f"{l}\n{n}" for (l, _, _), n in zip(vals, names)])
    ylabel(ax, "decode tok/s")
    titles(ax, "One 125B model, from no GPU to 96 GiB",
           "decode at a 2,048-token prompt - VRAM software-capped, expert layers move to system RAM until the model fits")
    say("  ladder 2K decode:", {l: round(v, 1) for l, v, _ in vals})
    save(fig, "ladder_decode")


# Canonical prompt lengths for the context chart. Points sit at equal steps so
# every part of the sweep gets the same width - a log axis crams 128K -> 245K
# into a sliver while 256 -> 2K gets three times the room, which reads as if
# the axis ran backwards. Off-grid lengths (12,288 and 122,880) interpolate
# between their neighbors.
CTX_GRID = [256, 2048, 8192, 32768, 65536, 131072, 245760]
CTX_LABEL = ["256", "2K", "8K", "32K", "64K", "128K", "245K"]


def ctx_pos(x):
    if x <= CTX_GRID[0]:
        return 0.0
    for i in range(len(CTX_GRID) - 1):
        a, b = CTX_GRID[i], CTX_GRID[i + 1]
        if x <= b:
            return i + (log2(x) - log2(a)) / (log2(b) - log2(a))
    return float(len(CTX_GRID) - 1)


def chart_ladder_vs_context():
    rows = ladder_rows()
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=200)
    fig.patch.set_facecolor(P["surface"])
    style_axes(ax)
    for c in TIERS:
        pts = sorted((r["length"], r["decode_tps"]) for r in rows if r["cap"] == c)
        xs = [ctx_pos(x) for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, color=tier_color(c), linewidth=2.2,
                solid_capstyle="round", zorder=3)
        ax.plot(xs, ys, "o", color=tier_color(c), markersize=6.5,
                markeredgecolor=P["surface"], markeredgewidth=1.5, zorder=4,
                label=f"{c} GiB")
    # The lines converge on the right - the chart's point - so end labels would
    # pile up there. A legend, largest tier first, names them instead.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left", frameon=False,
              fontsize=10, labelcolor=P["ink2"], ncols=2, columnspacing=1.2,
              handletextpad=0.4)
    ax.set_xticks(range(len(CTX_GRID)))
    ax.set_xticklabels(CTX_LABEL)
    ax.set_xlim(-0.25, len(CTX_GRID) - 0.75)
    ax.set_ylim(0, 120)
    xlabel(ax, "prompt tokens")
    ylabel(ax, "decode tok/s")
    d = {(r["cap"], r["length"]): r["decode_tps"] for r in rows}
    lead2k = d[(96, 2048)] / d[(24, 2048)]
    lead245 = d[(96, 245760)] / d[(24, 245760)]
    ax.text(0.985, 0.955,
            f"96 GiB leads 24 GiB by {lead2k:.1f}× at 2K,\nonly {lead245:.2f}× at 245K",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            color=P["ink2"])
    titles(ax, "The tiers converge as the prompt grows",
           "every tier pays for long context, and the fastest tier pays most")
    say(f"  convergence: {lead2k:.2f}x at 2K -> {lead245:.2f}x at 245K")
    save(fig, "ladder_vs_context")


def chart_ple_placement():
    s = json.loads((RESULTS / "corrections" / "20260830T145932Z_PLE-01" /
                    "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), dpi=200)
    fig.patch.set_facecolor(P["surface"])
    panels = [
        (axes[0], "Decode tok/s", s["cpu_decode_med"], s["cuda0_decode_med"],
         f"{s['decode_ratio']:.1f}× slower\non the GPU"),
        (axes[1], "Prefill tok/s", s["cpu_prefill_med"], s["cuda0_prefill_med"],
         f"{s['prefill_ratio']:.1f}× slower\non the GPU"),
    ]
    for ax, name, ram, gpu, note in panels:
        ax.set_ylim(0, ram * 1.22)
        ax.set_xlim(-0.65, 1.65)
        style_axes(ax)
        bar_round(ax, 0, ram, 0.55, P["blue"])
        bar_round(ax, 1, gpu, 0.55, P["gray"])
        for i, v in ((0, ram), (1, gpu)):
            txt = f"{v:,.2f}" if v < 10 else f"{v:,.1f}"
            ax.text(i, v + ax.get_ylim()[1] * 0.025, txt, ha="center",
                    fontsize=12, color=P["ink"], fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["table in\nsystem RAM", "table on\nthe GPU"])
        ax.set_title(name, color=P["ink"], fontsize=11.5, fontweight="bold",
                     loc="left", pad=10)
        ax.text(0.97, 0.72, note, transform=ax.transAxes, ha="right",
                fontsize=10.5, color=P["ink2"])
    fig.suptitle("The faster memory loses: the 27.2 GiB lookup table belongs in RAM",
                 color=P["ink"], fontsize=13.5, fontweight="bold", x=0.02,
                 ha="left")
    fig.text(0.02, 0.90, "same model, same context - the only change is "
             "-ot per_layer_token_embd=CPU or =CUDA0", fontsize=9.5,
             color=P["ink2"])
    fig.subplots_adjust(top=0.78, wspace=0.28)
    say(f"  PLE decode {s['cpu_decode_med']} vs {s['cuda0_decode_med']}"
        f" ({s['decode_ratio']}x), prefill {s['cpu_prefill_med']} vs"
        f" {s['cuda0_prefill_med']} ({s['prefill_ratio']}x)")
    save(fig, "ple_placement")


def chart_concurrency():
    s = json.loads((RESULTS / "corrections" / "20260830T193552Z_CONC-01" /
                    "summary.json").read_text())["aggregate"]
    series = [("unified KV", "unified", P["blue"]),
              ("non-unified KV", "non-unified", P["green"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=200)
    fig.patch.set_facecolor(P["surface"])
    style_axes(ax)
    ax.set_xscale("log", base=2)
    peak = {}
    for label, key, color in series:
        levels = sorted(int(k) for k in s[key])
        means = [sum(s[key][str(l)]) / len(s[key][str(l)]) for l in levels]
        for l in levels:
            ax.plot([l] * len(s[key][str(l)]), s[key][str(l)], "o",
                    color=color, markersize=4, alpha=0.35, zorder=2)
        ax.plot(levels, means, color=color, linewidth=2.2, zorder=3,
                solid_capstyle="round")
        ax.plot(levels, means, "o", color=color, markersize=7,
                markeredgecolor=P["surface"], markeredgewidth=1.5, zorder=4)
        ax.annotate(label, (levels[-1], means[-1]), textcoords="offset points",
                    xytext=(10, -3), fontsize=10, color=color, fontweight="bold")
        peak[key] = means[-1]
        say(f"  {key}: c=1 {means[0]:.1f} -> c=16 {means[-1]:.1f}"
            f" ({means[-1] / means[0]:.2f}x), reps={len(s[key]['1'])}")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlim(0.85, 27)
    xlabel(ax, "concurrent clients")
    ylabel(ax, "aggregate decode tok/s")
    ax.text(0.03, 0.95, "unified peaks near 4 clients, then declines;\n"
            "non-unified keeps climbing - "
            f"{peak['non-unified'] / peak['unified']:.2f}× more\n"
            "aggregate at 16 clients", transform=ax.transAxes, va="top",
            fontsize=10.5, color=P["ink2"])
    titles(ax, "One request does not saturate the GPU",
           "16 slots, six passes per point (small dots) - both KV layouts, means joined")
    save(fig, "concurrency")


if __name__ == "__main__":
    for P, SUFFIX, VERBOSE in ((LIGHT, "", True), (DARK, "_dark", False)):
        chart_ladder_decode()
        chart_ladder_vs_context()
        chart_ple_placement()
        chart_concurrency()
