#!/usr/bin/env python3
"""FT-2: can this process pin N GiB of host memory?

WHY THIS GATE EXISTS

FreeToken keeps Qwen3.8-Flash-Next's 47.68 GiB PLE table PINNED in host RAM. This
box's hard RLIMIT_MEMLOCK is 11.49 GiB (`ulimit -Hl` = 12053420 KiB), so a HOST
process would die at ~11.5 GiB no matter how much RAM were free, and raising it
needs a root change plus a relogin.

Running the arm in a container is supposed to make that moot: docker's
`--ulimit memlock=-1` (compose: `ulimits: memlock: -1`) lifts the limit outright.
This proves it rather than assuming it, and it does so in five minutes instead of
forty minutes into a model load.

WHY IT STEPS UP INSTEAD OF JUMPING TO 48

Pinned memory is UNRECLAIMABLE. The kernel cannot page it out or drop it under
pressure, so an over-large pin on a 91 GiB box is exactly the shape of failure
that has taken this machine down before. Each step allocates, reports, and frees
before the next one is attempted, and the caller is expected to watch host memory
between steps.

    python3 scripts/pin_probe.py --steps 4,16
    python3 scripts/pin_probe.py --steps 48
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys

GIB = 1024 ** 3


def limits() -> dict:
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    inf = resource.RLIM_INFINITY
    return {
        "memlock_soft": "unlimited" if soft == inf else soft,
        "memlock_hard": "unlimited" if hard == inf else hard,
        "memlock_hard_gib": None if hard == inf else round(hard / GIB, 2),
        "unlimited": hard == inf,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", default="4,16",
                    help="GiB sizes to attempt, in order (default: 4,16)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lim = limits()
    print(f"  RLIMIT_MEMLOCK hard : "
          f"{'unlimited' if lim['unlimited'] else str(lim['memlock_hard_gib']) + ' GiB'}")
    if not lim["unlimited"]:
        print("  NOTE: not unlimited - expect failure above that figure. In a container "
              "this means --ulimit memlock=-1 did not apply.")

    import torch
    print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  FAIL: no CUDA device - pinned host memory needs a CUDA context")
        return 2

    results = []
    for s in [int(x) for x in a.steps.split(",") if x.strip()]:
        n = s * GIB
        print(f"  pinning {s:>3} GiB ...", end=" ", flush=True)
        buf = None
        try:
            buf = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            # Touch a page per 4 KiB so the pages are really resident, not just
            # reserved. A pin that is never touched can look like a success.
            buf[::4096] = 1
            print("OK")
            results.append({"gib": s, "ok": True, "error": None})
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {str(e)[:110]}")
            results.append({"gib": s, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
            break
        finally:
            del buf                      # free before the next, larger, step
            torch.cuda.empty_cache()

    ok = [r["gib"] for r in results if r["ok"]]
    print(f"  largest successful pin: {max(ok) if ok else 0} GiB")
    payload = {"gate": "FT-2", "limits": lim, "torch": torch.__version__,
               "results": results, "largest_gib": max(ok) if ok else 0,
               "ple_requirement_gib": 47.68}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
