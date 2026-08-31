#!/usr/bin/env python3
"""Render the tier-ladder charts to static SVG for the README.

GitHub does not run JavaScript in markdown, so the charts have to be plain SVG
with literal colours - no <style> block, no CSS variables, no script. Each chart
is emitted twice, light and dark, and the README picks between them with
<picture media="(prefers-color-scheme: dark)">.

Colours are the validated ordinal blue ramp (steps 250/400/500/600/700 light,
100/200/300/400/500 dark). Six tiers do not fit: adjacent ordinal steps need
dL >= 0.06 and blue's usable range only holds five, so 16 GiB is omitted from the
line charts - it tracks 24 GiB within 4%.

  ./benchmark/make_charts.py          # results/tier_full/full.csv -> assets/*.svg
"""
from __future__ import annotations
import csv, pathlib, math

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC  = ROOT / "results/tier_full/full.csv"
OUT  = ROOT / "assets"

CPU_ONLY_DECODE = 8.34          # results/cpu_only/speed.log, 2048-token prompt
BAR_LEN = 2048

THEME = {
    "light": dict(bg="#ffffff", ink="#14161d", ink2="#565a66", ink3="#878b97",
                  grid="#eceef2", hair="#e5e6ea", flag="#b0492c", nogpu="#9c9992",
                  ramp=["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark":  dict(bg="#1c1e24", ink="#f3f4f7", ink2="#b3b7c2", ink3="#82868f",
                  grid="#282b33", hair="#2e313a", flag="#e08a6a", nogpu="#7e7b74",
                  ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf"]),
}
LINE_TIERS = [8, 24, 32, 48, 96]          # 16 omitted: ramp holds five steps
BAR_TIERS  = [8, 16, 24, 32, 48, 96]
LABELS = {8: "3060 / 4060", 16: "4060 Ti", 24: "3090 / 4090",
          32: "5090", 48: "2x 3090", 96: "PRO 6000"}
XT = [256, 2048, 8192, 32768, 65536, 131072, 245760]
XL = ["256", "2K", "8K", "32K", "64K", "128K", "245K"]


def load():
    rows = list(csv.DictReader(SRC.open()))
    d = {}
    for r in rows:
        d.setdefault(int(r["cap"]), []).append(
            (int(r["length"]), float(r["prefill_tps"]), float(r["decode_tps"])))
    for k in d:
        d[k].sort()
    return d


def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;")


def text(x, y, s, fill, size=11, anchor="start", weight="400", spacing=None):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{ls}>{esc(s)}</text>')


def head(w, h, t, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" font-family="IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace" '
            f'role="img" aria-label="{esc(title)}">'
            f'<rect width="{w}" height="{h}" fill="{t["bg"]}"/>')


def bars(data, mode):
    t = THEME[mode]
    W, H = 940, 470
    L, R, TOP, B = 78, 916, 96, 388
    mx = 120
    Y = lambda v: B - (v / mx) * (B - TOP)
    o = [head(W, H, t, "Decode speed by VRAM tier, CPU-only through 96 GiB")]
    o.append(text(20, 34, "Adding any GPU matters more than adding VRAM", t["ink"], 19, weight="600"))
    o.append(text(20, 56, "decode tok/s at a 2,048-token prompt", t["ink2"], 12.5))
    for g in range(0, 121, 20):
        o.append(f'<line x1="{L}" x2="{R}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(text(L - 11, Y(g) + 4, g, t["ink3"], 11, "end"))
    o.append(text(20, TOP - 30, "DECODE TOK/S", t["ink3"], 10, spacing="0.06em"))

    series = [("no GPU", "CPU only", CPU_ONLY_DECODE, t["nogpu"])]
    for i, cap in enumerate(BAR_TIERS):
        v = next(d for l, p, d in data[cap] if l == BAR_LEN)
        series.append((f"{cap} GiB", LABELS[cap], v, t["ramp"][min(4, max(0, i - 1))]))

    step = (R - L) / len(series)
    bw = min(74, step * 0.6)
    cx = lambda i: L + step * i + step / 2
    for i, (k, sub, v, col) in enumerate(series):
        x, y = cx(i) - bw / 2, Y(v)
        o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(2,B-y):.1f}" rx="4" fill="{col}"/>')
        o.append(text(cx(i), y - 10, f"{v:.1f}", t["ink"], 12.5, "middle", "600"))
        o.append(text(cx(i), B + 22, k, t["ink"], 12, "middle", "500"))
        o.append(text(cx(i), B + 38, sub, t["ink3"], 10.5, "middle"))
    # the two cliffs and the plateau between them
    o.append(f'<path d="M{cx(0)+26:.1f} {Y(CPU_ONLY_DECODE)-16:.1f} L{cx(1)-26:.1f} {Y(series[1][2])-16:.1f}" '
             f'stroke="{t["flag"]}" stroke-width="1.5" stroke-dasharray="3 3" fill="none"/>')
    o.append(text(cx(0) + 8, Y(series[1][2]) - 26, "4.3x", t["flag"], 12, weight="600"))
    by = Y(62)
    o.append(f'<path d="M{cx(1)-bw/2:.1f} {by:.1f} L{cx(1)-bw/2:.1f} {by-9:.1f} '
             f'L{cx(5)+bw/2:.1f} {by-9:.1f} L{cx(5)+bw/2:.1f} {by:.1f}" '
             f'stroke="{t["ink3"]}" stroke-width="1" fill="none"/>')
    o.append(text((cx(1) + cx(5)) / 2, by - 17, "6x the VRAM, +45% decode", t["ink2"], 11.5, "middle"))
    o.append(f'<path d="M{cx(5)+26:.1f} {Y(series[5][2])-16:.1f} L{cx(6)-26:.1f} {Y(series[6][2])-16:.1f}" '
             f'stroke="{t["flag"]}" stroke-width="1.5" stroke-dasharray="3 3" fill="none"/>')
    o.append(text(cx(6) - 6, Y(series[6][2]) - 26, "2.1x", t["flag"], 12, "end", "600"))
    o.append(f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{t["hair"]}" stroke-width="1"/>')
    o.append(text(20, H - 16, "llama.cpp b10666  ·  UD-IQ4_XS  ·  -ot per_layer_token_embd=CPU  ·  -ub 512", t["ink3"], 10))
    return "".join(o) + "</svg>"


def linechart(data, mode, idx, maxy, ticks, title, sub, unit):
    t = THEME[mode]
    W, H = 940, 470
    L, R, TOP, B = 82, 810, 92, 386
    x0, x1 = math.log(256), math.log(245760)
    X = lambda v: L + (math.log(v) - x0) / (x1 - x0) * (R - L)
    Y = lambda v: B - (v / maxy) * (B - TOP)
    o = [head(W, H, t, title)]
    o.append(text(20, 34, title, t["ink"], 19, weight="600"))
    o.append(text(20, 56, sub, t["ink2"], 12.5))
    for g in ticks:
        o.append(f'<line x1="{L}" x2="{R+96}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(text(L - 11, Y(g) + 4, g, t["ink3"], 11, "end"))
    for v, lab in zip(XT, XL):
        o.append(text(X(v), B + 22, lab, t["ink3"], 11, "middle"))
    o.append(text(20, TOP - 30, unit, t["ink3"], 10, spacing="0.06em"))
    o.append(text((L + R) / 2, B + 44, "PROMPT TOKENS", t["ink3"], 10, "middle", spacing="0.06em"))
    o.append(f'<line x1="{L}" x2="{R+96}" y1="{B}" y2="{B}" stroke="{t["hair"]}" stroke-width="1"/>')
    for i, cap in enumerate(LINE_TIERS):
        col = t["ramp"][i]
        pts = [(l, r[idx]) for l, *r in ((x[0], x[1], x[2]) for x in data[cap])]
        d = " ".join(("L" if j else "M") + f"{X(l):.1f} {Y(v):.1f}" for j, (l, v) in enumerate(pts))
        o.append(f'<path d="{d}" stroke="{col}" stroke-width="2" fill="none" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for l, v in pts:
            o.append(f'<circle cx="{X(l):.1f}" cy="{Y(v):.1f}" r="4.5" fill="{col}" '
                     f'stroke="{t["bg"]}" stroke-width="2"/>')
        ll, lv = pts[-1]
        o.append(text(X(ll) + 11, Y(lv) + 4, f"{cap} GiB", col, 12, weight="500"))
    o.append(text(20, H - 16, "lines stop where the tier runs out of context  ·  16 GiB omitted, tracks 24 GiB within 4%", t["ink3"], 10))
    return "".join(o) + "</svg>"


def main():
    data = load()
    OUT.mkdir(exist_ok=True)
    charts = {
        "ladder_decode": lambda m: bars(data, m),
        "ladder_vs_context": lambda m: linechart(
            data, m, 1, 120, [0, 20, 40, 60, 80, 100, 120],
            "Decode: the tiers converge as the prompt grows",
            "every tier pays for long context, and the fastest pays most", "DECODE TOK/S"),
        "ladder_prefill": lambda m: linechart(
            data, m, 0, 2000, [0, 500, 1000, 1500, 2000],
            "Prefill: this is what VRAM actually buys",
            "prompt processing, same runs and the same lengths", "PREFILL TOK/S"),
    }
    for name, fn in charts.items():
        for mode in ("light", "dark"):
            p = OUT / f"{name}{'' if mode == 'light' else '_dark'}.svg"
            p.write_text(fn(mode))
            print(f"  {p.relative_to(ROOT)}  {p.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
