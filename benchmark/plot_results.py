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

Regenerate after a rerun:  uv run benchmark/plot_results.py
"""

from __future__ import annotations

import csv
import json
import re
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
BLUE = "#2a78d6"
GREEN = "#1f9d63"
GRAY = "#898781"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

# Ordinal ramp for the VRAM tiers, pale to deep as capacity grows.
def _lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))

_LO = (0.78, 0.86, 0.95)
_HI = (0.05, 0.24, 0.44)
TIERS = [8, 16, 24, 32, 48, 96]
TIER_COLOR = {c: _lerp(_LO, _HI, i / (len(TIERS) - 1)) for i, c in enumerate(TIERS)}
TIER_NAME = {8: "3060 / 4060", 16: "4060 Ti", 24: "3090 / 4090",
             32: "5090", 48: "2× 3090", 96: "PRO 6000"}

DPI = 200


def style_axes(ax, y_grid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if y_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, length=0, labelsize=9)


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
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=22)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9, color=INK_2)


def save(fig, name):
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"{name}.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
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
    vals = [("no GPU", cpu, GRAY)] + [
        (f"{c} GiB", at2k[c], TIER_COLOR[c]) for c in TIERS]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_ylim(0, max(v for _, v, _ in vals) * 1.18)
    style_axes(ax)
    for i, (label, v, color) in enumerate(vals):
        bar_round(ax, i, v, 0.62, color)
        ax.text(i, v + ax.get_ylim()[1] * 0.02, f"{v:.1f}", ha="center",
                fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(range(len(vals)))
    names = ["CPU only"] + [TIER_NAME[c] for c in TIERS]
    ax.set_xticklabels([f"{l}\n{n}" for (l, _, _), n in zip(vals, names)])
    ax.set_ylabel("decode tok/s", color=MUTED, fontsize=9)
    titles(ax, "One 125B model, from no GPU to 96 GiB",
           "decode at a 2,048-token prompt - VRAM software-capped, expert layers move to system RAM until the model fits")
    print("  ladder 2K decode:", {l: round(v, 1) for l, v, _ in vals})
    save(fig, "ladder_decode")


def chart_ladder_vs_context():
    rows = ladder_rows()
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.set_xscale("log", base=2)
    for c in TIERS:
        pts = sorted((r["length"], r["decode_tps"]) for r in rows if r["cap"] == c)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=TIER_COLOR[c], linewidth=2,
                solid_capstyle="round", zorder=3)
        ax.plot(xs, ys, "o", color=TIER_COLOR[c], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
                label=f"{c} GiB")
    # The lines converge on the right - the chart's point - so end labels would
    # pile up there. A legend, largest tier first, names them instead.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left", frameon=False,
              fontsize=9, labelcolor=INK_2, ncols=2, columnspacing=1.2,
              handletextpad=0.4)
    ticks = [256, 2048, 8192, 32768, 65536, 131072, 245760]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["256", "2K", "8K", "32K", "64K", "128K", "245K"])
    ax.set_xlim(220, 300000)
    ax.set_xlabel("prompt tokens", color=MUTED, fontsize=9)
    ax.set_ylabel("decode tok/s", color=MUTED, fontsize=9)
    d = {(r["cap"], r["length"]): r["decode_tps"] for r in rows}
    lead2k = d[(96, 2048)] / d[(24, 2048)]
    lead245 = d[(96, 245760)] / d[(24, 245760)]
    ax.text(0.985, 0.955,
            f"96 GiB leads 24 GiB by {lead2k:.1f}× at 2K,\nonly {lead245:.2f}× at 245K",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=INK_2)
    titles(ax, "The tiers converge as the prompt grows",
           "every tier pays for long context, and the fastest tier pays most")
    print(f"  convergence: {lead2k:.2f}x at 2K -> {lead245:.2f}x at 245K")
    save(fig, "ladder_vs_context")


def chart_ple_placement():
    s = json.loads((RESULTS / "corrections" / "20260830T145932Z_PLE-01" /
                    "summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
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
        bar_round(ax, 0, ram, 0.55, BLUE)
        bar_round(ax, 1, gpu, 0.55, GRAY)
        for i, v in ((0, ram), (1, gpu)):
            txt = f"{v:,.2f}" if v < 10 else f"{v:,.1f}"
            ax.text(i, v + ax.get_ylim()[1] * 0.025, txt, ha="center",
                    fontsize=11, color=INK, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["table in\nsystem RAM", "table on\nthe GPU"])
        ax.set_title(name, color=INK, fontsize=11, fontweight="bold",
                     loc="left", pad=10)
        ax.text(0.97, 0.72, note, transform=ax.transAxes, ha="right",
                fontsize=10, color=INK_2)
    fig.suptitle("The faster memory loses: the 27.2 GiB lookup table belongs in RAM",
                 color=INK, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.90, "same model, same context - the only change is "
             "-ot per_layer_token_embd=CPU or =CUDA0", fontsize=9, color=INK_2)
    fig.subplots_adjust(top=0.78, wspace=0.28)
    print(f"  PLE decode {s['cpu_decode_med']} vs {s['cuda0_decode_med']}"
          f" ({s['decode_ratio']}x), prefill {s['cpu_prefill_med']} vs"
          f" {s['cuda0_prefill_med']} ({s['prefill_ratio']}x)")
    save(fig, "ple_placement")


def chart_concurrency():
    s = json.loads((RESULTS / "corrections" / "20260830T193552Z_CONC-01" /
                    "summary.json").read_text())["aggregate"]
    series = [("unified KV", "unified", BLUE), ("non-unified KV", "non-unified", GREEN)]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.set_xscale("log", base=2)
    gain = {}
    for label, key, color in series:
        levels = sorted(int(k) for k in s[key])
        means = [sum(s[key][str(l)]) / len(s[key][str(l)]) for l in levels]
        for l in levels:
            ax.plot([l] * len(s[key][str(l)]), s[key][str(l)], "o",
                    color=color, markersize=4, alpha=0.30, zorder=2)
        ax.plot(levels, means, color=color, linewidth=2, zorder=3,
                solid_capstyle="round")
        ax.plot(levels, means, "o", color=color, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.annotate(label, (levels[-1], means[-1]), textcoords="offset points",
                    xytext=(10, -3), fontsize=9.5, color=color, fontweight="bold")
        gain[key] = means[-1]
        print(f"  {key}: c=1 {means[0]:.1f} -> c=16 {means[-1]:.1f}"
              f" ({means[-1] / means[0]:.2f}x), reps={len(s[key]['1'])}")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlim(0.85, 27)
    ax.set_xlabel("concurrent clients", color=MUTED, fontsize=9)
    ax.set_ylabel("aggregate decode tok/s", color=MUTED, fontsize=9)
    ax.text(0.03, 0.95, "unified peaks near 4 clients, then declines;\n"
            "non-unified keeps climbing - "
            f"{gain['non-unified'] / gain['unified']:.2f}× more\n"
            "aggregate at 16 clients", transform=ax.transAxes, va="top",
            fontsize=10, color=INK_2)
    titles(ax, "One request does not saturate the GPU",
           "16 slots, six passes per point (small dots) - both KV layouts, means joined")
    save(fig, "concurrency")


if __name__ == "__main__":
    chart_ladder_decode()
    chart_ladder_vs_context()
    chart_ple_placement()
    chart_concurrency()
