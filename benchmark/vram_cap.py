#!/usr/bin/env python3
"""Hold N GiB of VRAM so one big card can stand in for a smaller one.

Allocates and keeps a block of device memory, then waits. Everything else on the
GPU sees only what is left, so an RTX PRO 6000 can be made to look - to an
allocator - like a 5090 or a 3090.

Uses the CUDA driver API through ctypes, so it needs only libcuda.so (which ships
with the driver) and no CUDA toolkit, no torch, no compile step.

  ./benchmark/vram_cap.py --leave 32       leave 32 GiB free, hold the rest
  ./benchmark/vram_cap.py --hold 64        hold exactly 64 GiB
  ./benchmark/vram_cap.py --leave 24 &     background; kill it to release

WHAT THIS DOES AND DOES NOT SIMULATE - state this wherever the numbers appear:

  capacity      yes. The allocator genuinely cannot use the held memory.
  bandwidth     NO. This card keeps its own memory bandwidth. A 3090 has
                936 GB/s against roughly 1.6 TB/s here, so a 24 GiB cap
                OVERSTATES a real 3090 on whatever stays resident.
  compute       NO. Blackwell SMs, not Ampere or Ada.
  topology      NO. Two 24 GB cards are not one 48 GiB pool: they add PCIe
                transfers and per-card pools. A single capped pool is an
                optimistic UPPER BOUND for any multi-GPU config.

  The one genuinely good proxy is the 5090: same Blackwell generation, similar
  memory bandwidth, so a 32 GiB cap should predict it closely. Everything else
  is a memory-envelope experiment, and ratios between strategies transfer far
  better than absolute tok/s.
"""
from __future__ import annotations

import argparse
import ctypes
import signal
import sys
import time

GiB = 1024 ** 3


def _chk(rc: int, what: str, lib) -> None:
    if rc != 0:
        name = ctypes.c_char_p()
        try:
            lib.cuGetErrorName(rc, ctypes.byref(name))
            detail = name.value.decode() if name.value else str(rc)
        except Exception:
            detail = str(rc)
        raise RuntimeError(f"{what} failed: {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hold", type=float, help="GiB to occupy")
    g.add_argument("--leave", type=float, help="GiB to leave free for everything else")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--headroom", type=float, default=0.5,
                    help="GiB left unallocated so the context itself fits (default 0.5)")
    a = ap.parse_args()

    lib = ctypes.CDLL("libcuda.so.1")
    _chk(lib.cuInit(0), "cuInit", lib)
    dev = ctypes.c_int()
    _chk(lib.cuDeviceGet(ctypes.byref(dev), a.device), "cuDeviceGet", lib)
    ctx = ctypes.c_void_p()
    _chk(lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate", lib)

    free_b, total_b = ctypes.c_size_t(), ctypes.c_size_t()
    _chk(lib.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b)), "cuMemGetInfo", lib)
    free_g, total_g = free_b.value / GiB, total_b.value / GiB
    print(f"  device {a.device}: {total_g:.1f} GiB total, {free_g:.1f} GiB free")

    hold = a.hold if a.hold is not None else max(0.0, free_g - a.leave)
    hold = min(hold, free_g - a.headroom)
    if hold <= 0:
        print(f"  nothing to hold (free {free_g:.1f} GiB, asked to leave {a.leave})")
        return 1

    ptr = ctypes.c_void_p()
    _chk(lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(int(hold * GiB))), "cuMemAlloc", lib)
    _chk(lib.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b)), "cuMemGetInfo", lib)
    now_free = free_b.value / GiB
    print(f"  holding {hold:.1f} GiB -> {now_free:.1f} GiB visible to everything else")
    print(f"  simulating a ~{now_free:.0f} GiB card. Ctrl-C or SIGTERM to release.")
    sys.stdout.flush()

    stop = {"v": False}
    def bye(signum, frame):
        stop["v"] = True
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    try:
        while not stop["v"]:
            time.sleep(1)
    finally:
        lib.cuMemFree_v2(ptr)
        print("  released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
