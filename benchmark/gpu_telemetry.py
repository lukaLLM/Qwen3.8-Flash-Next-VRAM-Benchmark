#!/usr/bin/env python3
"""GPU telemetry sampler and thermal watchdog for benchmark arms.

Two jobs:

1. **Say what the GPU was actually doing.** tok/s tells you how fast a run was,
   not why. `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` against `DCGM_FI_PROF_DRAM_ACTIVE`
   separates compute-bound from memory-bound, which is the interesting question
   for a MoE with a 2560 hidden dim: during decode it is reading a lot of expert
   weights and doing very little maths per byte, so the expectation is low
   tensor activity and high DRAM activity. If that is what we see, it explains
   the decode numbers and predicts what would and would not help.

2. **Catch a run that was throttled.** `gpu_settle.sh` cools the card *between*
   arms; nothing was watching *during* one. A run that thermally throttles is
   not a slow configuration, it is an invalid measurement, and DCGM exposes the
   throttle counters directly rather than making us infer it from clocks.

Sources, in order of preference:
  - a dcgm-exporter with profiling counters (see dcgm_exporter.sh, port 9402)
  - any dcgm-exporter (port 9401 here) - no tensor/DRAM ratios
  - nvidia-smi - always available, no profiling metrics

Usage
-----
  ./benchmark/gpu_telemetry.py watch --seconds 300 --out results/tag/gpu.json
  ./benchmark/gpu_telemetry.py sample          # one reading, for a sanity check
"""
from __future__ import annotations

import argparse
import signal
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORTS = (9402, 9401)          # profiling exporter first, then the default one
WANT = (
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "DCGM_FI_PROF_DRAM_ACTIVE",
    "DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_SM_OCCUPANCY",
    "DCGM_FI_PROF_PCIE_TX_BYTES", "DCGM_FI_PROF_PCIE_RX_BYTES",
    "DCGM_FI_DEV_FB_USED", "DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_MEMORY_TEMP",
    "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_SM_CLOCK",
    "DCGM_FI_DEV_THERMAL_VIOLATION", "DCGM_FI_DEV_POWER_VIOLATION",
    "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_MEM_COPY_UTIL",
)
LINE = re.compile(r"^(DCGM_FI_[A-Z0-9_]+)\{[^}]*\}\s+([0-9.eE+-]+)")


def dcgm_endpoint() -> str | None:
    for p in PORTS:
        try:
            with urllib.request.urlopen(f"http://localhost:{p}/metrics", timeout=3) as r:
                body = r.read().decode("utf-8", "replace")
            if "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE" in body or "DCGM_FI_DEV_GPU_TEMP" in body:
                return f"http://localhost:{p}/metrics"
        except Exception:
            continue
    return None


def scrape(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        body = r.read().decode("utf-8", "replace")
    out = {}
    for line in body.splitlines():
        m = LINE.match(line)
        if m and m.group(1) in WANT:
            out.setdefault(m.group(1), float(m.group(2)))   # first GPU only
    return out


def smi() -> dict:
    q = ("memory.used,temperature.gpu,power.draw,clocks.sm,"
         "utilization.gpu,utilization.memory")
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    v = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    f = lambda i: float(v[i]) if v[i].replace(".", "").isdigit() else 0.0
    return {"DCGM_FI_DEV_FB_USED": f(0), "DCGM_FI_DEV_GPU_TEMP": f(1),
            "DCGM_FI_DEV_POWER_USAGE": f(2), "DCGM_FI_DEV_SM_CLOCK": f(3),
            "DCGM_FI_DEV_GPU_UTIL": f(4), "DCGM_FI_DEV_MEM_COPY_UTIL": f(5)}


def read(url: str | None) -> dict:
    if url:
        try:
            s = scrape(url)
            if s:
                return s
        except Exception:
            pass
    return smi()


def summarise(rows: list[dict], key: str):
    vals = [r[key] for r in rows if key in r]
    if not vals:
        return None
    return {"mean": round(statistics.fmean(vals), 4),
            "max": round(max(vals), 4), "min": round(min(vals), 4)}


def cmd_sample(args) -> int:
    url = dcgm_endpoint()
    print(f"  source: {url or 'nvidia-smi (no dcgm-exporter reachable)'}")
    for k, v in sorted(read(url).items()):
        print(f"    {k:<34} {v}")
    return 0


def cmd_watch(args) -> int:
    url = dcgm_endpoint()
    prof = bool(url) and "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE" in read(url)
    print(f"  source: {url or 'nvidia-smi'}"
          f"{'  (profiling metrics available)' if prof else '  (no tensor/DRAM ratios)'}")
    if not prof:
        print("  hint: ./benchmark/dcgm_exporter.sh up   for tensor-pipe and DRAM activity")

    rows, t0 = [], time.time()
    first = read(url)
    thermal0 = first.get("DCGM_FI_DEV_THERMAL_VIOLATION")
    power0 = first.get("DCGM_FI_DEV_POWER_VIOLATION")
    hot = []
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(args.interval)
            r = read(url)
            r["_t"] = round(time.time() - t0, 1)
            rows.append(r)
            t = r.get("DCGM_FI_DEV_GPU_TEMP", 0)
            if t >= args.max_temp:
                hot.append((r["_t"], t))
                if len(hot) == 1:
                    print(f"  WATCHDOG: {t:.0f}C at t={r['_t']}s exceeds {args.max_temp:.0f}C")
    except KeyboardInterrupt:
        pass

    if not rows:
        print("  no samples"); return 1

    clock = summarise(rows, "DCGM_FI_DEV_SM_CLOCK")
    clock_drop = bool(clock and clock["max"] > 0 and clock["mean"] < 0.75 * clock["max"])
    last = rows[-1]
    thermal_us = (last.get("DCGM_FI_DEV_THERMAL_VIOLATION", 0) - thermal0) if thermal0 is not None else None
    power_us = (last.get("DCGM_FI_DEV_POWER_VIOLATION", 0) - power0) if power0 is not None else None

    res = {
        "seconds": round(time.time() - t0, 1), "samples": len(rows),
        "source": url or "nvidia-smi", "profiling_metrics": prof,
        "tensor_active": summarise(rows, "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"),
        "dram_active": summarise(rows, "DCGM_FI_PROF_DRAM_ACTIVE"),
        "sm_active": summarise(rows, "DCGM_FI_PROF_SM_ACTIVE"),
        "sm_occupancy": summarise(rows, "DCGM_FI_PROF_SM_OCCUPANCY"),
        "pcie_rx_bytes": summarise(rows, "DCGM_FI_PROF_PCIE_RX_BYTES"),
        "pcie_tx_bytes": summarise(rows, "DCGM_FI_PROF_PCIE_TX_BYTES"),
        "fb_used_mib": summarise(rows, "DCGM_FI_DEV_FB_USED"),
        "gpu_temp_c": summarise(rows, "DCGM_FI_DEV_GPU_TEMP"),
        "memory_temp_c": summarise(rows, "DCGM_FI_DEV_MEMORY_TEMP"),
        "power_w": summarise(rows, "DCGM_FI_DEV_POWER_USAGE"),
        "sm_clock_mhz": summarise(rows, "DCGM_FI_DEV_SM_CLOCK"),
        "gpu_util_pct": summarise(rows, "DCGM_FI_DEV_GPU_UTIL"),
        "thermal_violation_us": thermal_us,
        "power_violation_us": power_us,
        "samples_over_max_temp": len(hot),
        "max_temp_threshold_c": args.max_temp,
        "sm_clock_drop": clock_drop,
        "power_violation_trusted": False,
    }
    # (clock stats are computed before res is built - referencing them inside the
    # dict while defining them afterwards raised UnboundLocalError and threw away
    # a whole arm's telemetry.)
    # POWER_VIOLATION is not trustworthy on this driver: its absolute value read
    # 1.69e12 us on a box with 21 hours of uptime, i.e. 20x longer than the
    # machine had been running. It flagged an arm as throttled that peaked at
    # 65 C, drew 302 W of a 600 W limit and held 2778 of 2835 MHz. Report it,
    # do not act on it.
    #
    # Thermal violation and measured temperature are the reliable signals, plus
    # a real clock drop as a cross-check.
    throttled = bool(thermal_us) or len(hot) > 0 or clock_drop
    res["verdict"] = ("THROTTLED - this arm is not comparable with an unthrottled one"
                      if throttled else "ok")

    def show(label, d, unit="", scale=1.0):
        if d:
            print(f"  {label:<20} mean {d['mean']*scale:8.2f}{unit}   max {d['max']*scale:8.2f}{unit}")
    show("tensor pipe", res["tensor_active"], "%", 100)
    show("DRAM active", res["dram_active"], "%", 100)
    show("SM active", res["sm_active"], "%", 100)
    show("VRAM", res["fb_used_mib"], " MiB")
    show("GPU temp", res["gpu_temp_c"], " C")
    show("power", res["power_w"], " W")
    show("SM clock", res["sm_clock_mhz"], " MHz")
    if thermal_us is not None:
        print(f"  throttle (thermal)   {thermal_us:.0f} us      (power) {power_us:.0f} us")
    if res["tensor_active"] and res["dram_active"] and res["dram_active"]["mean"] > 0:
        t, dm = res["tensor_active"]["mean"], res["dram_active"]["mean"]
        ratio = dm / max(t, 1e-9)
        res["dram_over_tensor"] = round(ratio, 1)
        # A ratio alone is misleading. 92x sounds decisive but meant 0.003% tensor
        # against 0.28% DRAM - both idle, nothing saturated, and the real answer
        # is that the GPU was waiting, not that memory was the constraint.
        # PCIe has to be part of this verdict. With experts offloaded, tensor and
        # GPU-DRAM activity are both near zero while the bus carries 14.5 GB/s of
        # expert weights - calling that "latency bound" is true of the GPU and
        # blind to where the work is actually going.
        rx = (res.get("pcie_rx_bytes") or {}).get("mean", 0) / 1e9
        if rx > 5 and dm < 0.30 and t < 0.30:
            label = f"HOST-PATH BOUND - {rx:.1f} GB/s over PCIe, GPU idle"
            res["bound_by"] = "pcie"
        elif dm < 0.30 and t < 0.30:
            label, res["bound_by"] = "LATENCY BOUND - neither is saturated", "latency"
        elif ratio > 2:
            label, res["bound_by"] = "memory bound", "memory"
        elif ratio < 0.5:
            label, res["bound_by"] = "compute bound", "compute"
        else:
            label, res["bound_by"] = "mixed", "mixed"
        print(f"  DRAM/tensor ratio    {ratio:.1f}x  ({label})")
        print(f"    tensor {t*100:.2f}%  DRAM {dm*100:.2f}%  - a ratio without these is meaningless")
    print(f"  verdict              {res['verdict']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        if args.samples:
            res["rows"] = rows
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"  wrote {args.out}")
    return 2 if throttled else 0


class _Stop(KeyboardInterrupt):
    """Raised on SIGTERM so a watch loop shuts down the same way Ctrl-C does.

    The wrapper starts these as background jobs from a non-interactive shell,
    and POSIX says such jobs ignore SIGINT - so `kill -INT` was silently doing
    nothing and the wrapper hung in `wait` for 50 minutes. SIGTERM is not
    ignorable that way, so that is what we listen for.
    """


def _on_term(signum, frame):
    raise _Stop()


def main() -> int:
    signal.signal(signal.SIGTERM, _on_term)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.set_defaults(func=cmd_sample)
    w = sub.add_parser("watch")
    w.add_argument("--seconds", type=float, default=300)
    w.add_argument("--interval", type=float, default=2.0)
    w.add_argument("--max-temp", type=float, default=84.0,
                   help="watchdog threshold in C (default 84)")
    w.add_argument("--out", default=None)
    w.add_argument("--samples", action="store_true", help="include per-sample rows")
    w.set_defaults(func=cmd_watch)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
