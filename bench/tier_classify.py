#!/usr/bin/env python3
"""Classify each tier arm, from its own evidence, into one verdict.

Never pass/fail. A configuration that will not start is a CAPACITY FINDING for
that placement, and a configuration that finishes only by paging from NVMe is
not the same result as one that fits - reporting both as "works" is how a
"64 GB is fine" claim gets made without being true.

  completes      finished, no cap pressure worth noting
  heavy-paging   finished, but the cgroup was reclaiming hard (high events, or
                 major faults / device reads far above the resident arms)
  oom            the kernel killed it - this placement does not fit this budget
  guard-stop     host_mem_guard stopped the container to protect the desktop;
                 NOT an OOM, and not evidence about the cap
  bench-error    it started and the benchmark failed for some other reason
"""
from __future__ import annotations
import csv, json, pathlib, sys

R = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else R / "results" / "tier_mtp")

def cells(d):
    return [p for p in d.glob("**/profile_export_aiperf.json") if "phases" not in p.parts]

def classify(art: pathlib.Path, cg: pathlib.Path | None):
    ev = {}
    mem = art / "memory.json"
    fail = art / "failure.json"
    if mem.is_file():
        try: ev = json.loads(mem.read_text())
        except Exception: ev = {}
    fj = {}
    if fail.is_file():
        try: fj = json.loads(fail.read_text())
        except Exception: fj = {}

    oom = str(ev.get("oom_killed", fj.get("oom_killed", ""))).lower() == "true"
    events = ev.get("memory_events") or fj.get("memory_events") or {}
    oom_kill = int(events.get("oom_kill", 0) or 0)
    high = int(events.get("high", 0) or 0)
    done = bool(cells(art))

    # Sampler evidence: major faults and device reads while it ran.
    majf = reads = 0
    if cg and cg.is_file():
        rows = list(csv.DictReader(cg.open(), delimiter="\t"))
        if rows:
            try:
                majf = int(rows[-1]["pgmajfault"]) - int(rows[0]["pgmajfault"])
                reads = int(rows[-1]["io_read_bytes"]) - int(rows[0]["io_read_bytes"])
            except Exception:
                pass

    if oom or oom_kill:
        v = "oom"
    elif not done:
        # A guard stop leaves the container exited with OOMKilled=false and the
        # guard's own line in the queue log; anything else is a benchmark fault.
        v = "guard-stop" if fj.get("container_state") == "exited" and not oom else "bench-error"
    elif high > 0 or majf > 50_000:
        v = "heavy-paging"
    else:
        v = "completes"
    return dict(verdict=v, oom_kill=oom_kill, high=high, majfault=majf,
                read_gib=round(reads / 2**30, 2), peak_gib=round((ev.get("memory_peak_bytes") or 0) / 2**30, 1),
                cap=ev.get("cap_readback", "not-verified"))

def main():
    tsv = OUT / "arms.tsv"
    if not tsv.is_file():
        print(f"no {tsv}"); return 1
    rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
    # arms.tsv accumulates across invocations (a re-run after fixing a preflight
    # failure appends rather than replaces). Report the LAST row per label so a
    # fixed arm supersedes the broken attempt instead of sitting beside it.
    latest = {}
    for r in rows: latest[r["label"]] = r
    rows = list(latest.values())
    print(f"{'arm':22} {'tier':6} {'spec':4} {'cap':6} {'verdict':13} {'peak':>7} {'reads':>8} {'majflt':>8}  decode")
    out = []
    for r in rows:
        art = R / "artifacts" / r["artifact"]
        if r["artifact"] in ("none", "") or not art.is_dir():
            print(f"{r['label']:22} {r['tier']:6} {r['spec']:4} {r['mem_cap']:6} {'no-artifact':13}")
            continue
        cg = OUT / f"cgroup_{r['label']}.tsv"
        c = classify(art, cg)
        dec = ""
        cs = cells(art)
        if cs:
            j = json.loads(cs[0].read_text())
            v = (j.get("output_token_throughput_per_user") or {}).get("avg")
            if v: dec = f"{v:.1f} tok/s"
        if r.get("balloon") not in ("0", "", None):
            c["verdict"] += " (VOID: balloon)"
        print(f"{r['label']:22} {r['tier']:6} {r['spec']:4} {r['mem_cap']:6} {c['verdict']:13} "
              f"{c['peak_gib']:>6.1f}G {c['read_gib']:>7.2f}G {c['majfault']:>8}  {dec}")
        out.append(dict(r, **c, decode=dec))
    (OUT / "verdicts.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {OUT/'verdicts.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
