#!/usr/bin/env python3
"""PLE / weight paging monitor for Qwen3.8-Flash-Next.

This model keeps a ~27 GiB PLE (per-layer n-gram embedding) table that we
deliberately place outside VRAM with `-ot per_layer_token_embd=CPU`. That table
is then an mmap'd file region, which means the kernel is free to evict it. If it
does, llama.cpp silently re-reads it from NVMe on every token and the benchmark
measures the SSD instead of the model.

The failure is invisible in tok/s alone - the number just comes out lower and
still looks plausible. Nothing else in this suite catches it, so every timed run
should have `watch` running alongside it.

Subcommands
-----------
  residency   exact page-cache residency of the weight blobs, via mincore(2)
  warm        sequentially read the blobs so the PLE table is cache-resident
  watch       sample majflt + NVMe reads for a window, emit JSON and a verdict

Cold vs warm protocol (Phase 2)
-------------------------------
  cold:  start the server, let it load, then
           sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
         GPU-resident weights are untouched by this; the mmap'd PLE pages are
         the ones that get evicted, which is exactly the variable under test.
  warm:  ./ple_io_monitor.py warm     (then confirm with `residency`)

Examples
--------
  ./ple_io_monitor.py residency
  ./ple_io_monitor.py warm
  ./ple_io_monitor.py watch --seconds 120 --out ../results/phase1/io.json
"""
from __future__ import annotations

import argparse
import signal
import ctypes
import glob
import json
import mmap
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE")
SECTOR = 512  # /proc/diskstats reports sectors in 512-byte units regardless of device
DEFAULT_REPO = "models--unsloth--Qwen3.8-Flash-Next-GGUF"

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


class _PyBuffer(ctypes.Structure):
    _fields_ = [("buf", ctypes.c_void_p), ("obj", ctypes.py_object),
                ("len", ctypes.c_ssize_t), ("itemsize", ctypes.c_ssize_t),
                ("readonly", ctypes.c_int), ("ndim", ctypes.c_int),
                ("format", ctypes.c_char_p),
                ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
                ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
                ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
                ("internal", ctypes.c_void_p)]


def _addr_of(mm) -> int:
    """Base address of a read-only mmap.

    ctypes.c_char.from_buffer() refuses a read-only buffer, so go through the
    buffer protocol directly. PyBUF_SIMPLE (0) accepts read-only.
    """
    view = _PyBuffer()
    if ctypes.pythonapi.PyObject_GetBuffer(ctypes.py_object(mm), ctypes.byref(view), 0) != 0:
        raise RuntimeError("PyObject_GetBuffer failed")
    try:
        return view.buf
    finally:
        ctypes.pythonapi.PyBuffer_Release(ctypes.byref(view))


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))


def weight_files(quant: str | None) -> list[tuple[str, Path]]:
    """(display name, real path) for the .gguf files being served.

    Snapshot entries are symlinks into blobs/, so the readable name and the file
    we actually have to mmap live at different paths. Report the shard name;
    operate on the blob.
    """
    root = hf_home() / "hub" / DEFAULT_REPO / "snapshots"
    pat = f"*{quant}*" if quant else "*"
    out, seen = [], set()
    for p in glob.glob(str(root / "*" / pat / "*.gguf")):
        link = Path(p)
        rp = link.resolve()
        if rp.exists() and rp not in seen:
            seen.add(rp)
            out.append((link.name, rp))
    return sorted(out)


def server_pid(explicit: int | None) -> int | None:
    if explicit:
        return explicit
    # Container processes are visible in the host PID namespace, so a plain
    # pgrep finds the server. Match on the process NAME (-x), never -f: -f
    # matches whole command lines, so any shell that merely mentions
    # "llama-server" - including the one running this script - matches itself.
    try:
        r = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True)
        for tok in r.stdout.split():
            pid = int(tok)
            try:
                if Path(f"/proc/{pid}/comm").read_text().strip() == "llama-server":
                    return pid
            except OSError:
                continue
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# counters
# --------------------------------------------------------------------------
def majflt(pid: int) -> int | None:
    """Major faults = pages that had to come from disk. Field 12 of /proc/PID/stat.

    comm can contain spaces and parens, so split on the LAST ')' rather than
    naively on whitespace.
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    rest = raw[raw.rindex(")") + 2:].split()
    return int(rest[9])  # stat field 12 == index 9 after pid and comm


def disk_read_sectors(dev_re: str) -> int:
    total = 0
    rx = re.compile(rf"\s\d+\s+\d+\s+({dev_re})\s")
    for line in Path("/proc/diskstats").read_text().splitlines():
        if rx.search(line):
            total += int(line.split()[5])
    return total


# --------------------------------------------------------------------------
# residency
# --------------------------------------------------------------------------
def residency(name: str, path: Path) -> dict:
    """Exact page-cache residency via mincore(2), chunked so the vector stays small."""
    size = path.stat().st_size
    resident = 0
    chunk = 1 << 30  # 1 GiB at a time
    with open(path, "rb") as f:
        off = 0
        while off < size:
            n = min(chunk, size - off)
            mm = mmap.mmap(f.fileno(), n, mmap.MAP_SHARED, mmap.PROT_READ, offset=off)
            try:
                addr = _addr_of(mm)
                pages = (n + PAGE - 1) // PAGE
                vec = (ctypes.c_ubyte * pages)()
                if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(n), vec) != 0:
                    raise OSError(ctypes.get_errno(), "mincore failed")
                resident += sum(1 for b in vec if b & 1)
                del vec
            finally:
                mm.close()
            off += n
    return {
        "file": name,
        "size_gb": round(size / 1e9, 2),
        "cached_gb": round(resident * PAGE / 1e9, 2),
        "cached_pct": round(100 * resident * PAGE / size, 1) if size else 0.0,
    }


def cmd_residency(args) -> int:
    files = weight_files(args.quant)
    if not files:
        print("no .gguf found in the HF cache for that quant", file=sys.stderr)
        return 1
    rows = [residency(n, f) for n, f in files]
    tot, cac = sum(r["size_gb"] for r in rows), sum(r["cached_gb"] for r in rows)
    for r in rows:
        print(f"  {r['file'][:58]:<58} {r['cached_gb']:7.2f} / {r['size_gb']:7.2f} GB  {r['cached_pct']:5.1f}%")
    print(f"  {'TOTAL':<58} {cac:7.2f} / {tot:7.2f} GB  {100*cac/tot if tot else 0:5.1f}%")
    if args.json:
        Path(args.json).write_text(json.dumps({"files": rows, "total_gb": tot, "cached_gb": cac}, indent=2))
    return 0


def cmd_warm(args) -> int:
    files = weight_files(args.quant)
    if not files:
        print("no .gguf found", file=sys.stderr)
        return 1
    t0 = time.time()
    for name, f in files:
        n = 0
        with open(f, "rb", buffering=0) as fh:
            while True:
                b = fh.read(64 << 20)
                if not b:
                    break
                n += len(b)
        print(f"  read {name[:58]:<58} {n/1e9:6.1f} GB")
    dt = time.time() - t0
    total = sum(f.stat().st_size for _, f in files)
    print(f"  warmed {total/1e9:.1f} GB in {dt:.0f}s ({total/dt/1e6:.0f} MB/s)")
    print("  confirm with: ./ple_io_monitor.py residency")
    return 0


# --------------------------------------------------------------------------
# watch
# --------------------------------------------------------------------------
def cmd_watch(args) -> int:
    pid = server_pid(args.pid)
    if pid is None:
        print("no llama-server process found; pass --pid", file=sys.stderr)
        return 1

    f0, s0, t0 = majflt(pid), disk_read_sectors(args.device), time.time()
    print(f"  watching pid {pid}, device /{args.device}/, {args.seconds}s "
          f"(Ctrl-C to stop early)")
    samples = []
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(args.interval)
            now = time.time()
            fn, sn = majflt(pid), disk_read_sectors(args.device)
            if fn is None:
                print("  server exited during the window", file=sys.stderr)
                break
            samples.append({
                "t": round(now - t0, 1),
                "majflt": fn - f0,
                "read_mb": round((sn - s0) * SECTOR / 1e6, 1),
            })
            f0, s0 = fn, sn
    except KeyboardInterrupt:
        print("  stopped early")

    elapsed = time.time() - t0
    total_mb = sum(s["read_mb"] for s in samples)
    total_flt = sum(s["majflt"] for s in samples)
    rate = total_mb / elapsed if elapsed else 0.0

    # A run is void, not slow, if the kernel was pulling weights back off disk.
    #
    # Throughput alone is too lenient. A Phase 3 arm logged 458,120 major faults
    # over ~1000 s - the PLE table being re-read from NVMe continuously - and
    # this called it "ok" because 2 MB/s is far under a 50 MB/s threshold meant
    # for catastrophic thrashing. Sustained faulting at ANY rate means the table
    # is not staying resident, so the fault rate is the better signal.
    flt_rate = total_flt / elapsed if elapsed else 0.0
    void = rate >= args.void_mbs or flt_rate >= args.void_faults
    result = {
        "pid": pid,
        "seconds": round(elapsed, 1),
        "nvme_read_mb": round(total_mb, 1),
        "nvme_read_mbs": round(rate, 2),
        "major_faults": total_flt,
        "major_faults_per_s": round(flt_rate, 1),
        "void_threshold_mbs": args.void_mbs,
        "void_threshold_faults_per_s": args.void_faults,
        "verdict": "VOID - weights were being paged from disk" if void else "ok",
        "samples": samples if args.samples else None,
    }
    print(f"  NVMe read : {total_mb:9.1f} MB   ({rate:.2f} MB/s)")
    print(f"  majflt    : {total_flt:9d}   ({flt_rate:.0f}/s)")
    print(f"  verdict   : {result['verdict']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"  wrote {args.out}")
    return 2 if void else 0


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
    ap.add_argument("--quant", default="UD-IQ4_XS", help="quant dir to look at (default: UD-IQ4_XS)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("residency", help="page-cache residency of the weight blobs")
    r.add_argument("--json", default=None)
    r.set_defaults(func=cmd_residency)

    w = sub.add_parser("warm", help="sequentially read the blobs into page cache")
    w.set_defaults(func=cmd_warm)

    t = sub.add_parser("watch", help="sample majflt + NVMe reads during a run")
    t.add_argument("--seconds", type=float, default=120)
    t.add_argument("--interval", type=float, default=2.0)
    t.add_argument("--pid", type=int, default=None)
    t.add_argument("--device", default="nvme0n1", help="regex matched against /proc/diskstats")
    t.add_argument("--void-mbs", type=float, default=50.0,
                   help="sustained MB/s above which the run is marked void (default 50)")
    t.add_argument("--void-faults", type=float, default=50.0,
                   help="major faults/s above which the run is marked void (default 50). "
                        "Catches slow continuous paging that the MB/s threshold misses.")
    t.add_argument("--out", default=None)
    t.add_argument("--samples", action="store_true", help="include per-sample detail in the JSON")
    t.set_defaults(func=cmd_watch)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
