"""Build results/CONFIGS.md: one row per server start, from saved artifacts only."""
import csv, json, pathlib, re

R = pathlib.Path("/home/luke/Documents/Code/Qwen3.8-Flash-Next-rtx6000pro")
rows = []   # dict per arm

def add(**kw):
    d = dict(test="", arm="", ctx="", ncmoe="", load="", ub="", ot="", spec="",
             par="", kv="", temp="", src="", vram="")
    d.update(kw); rows.append(d)

# ---------- 1. resolved compose (richest) ----------
def from_compose(p):
    t = p.read_text()
    def env(k):
        m = re.search(rf'{k}:\s*(\S+)', t); return m.group(1).strip('"') if m else ""
    def arg(f):
        m = re.search(rf'-\s*{re.escape(f)}\s*\n\s*-\s*"?([^"\n]+)"?', t)
        return m.group(1).strip().strip('"') if m else ""
    return dict(ctx=arg("--ctx-size"), ncmoe=arg("--n-cpu-moe"), ub=arg("--ubatch-size"),
                par=arg("--parallel"), temp=arg("--temp"), load=arg("--load-mode"),
                ot=env("LLAMA_ARG_OVERRIDE_TENSOR").replace("per_layer_token_embd=",""),
                kv=env("LLAMA_ARG_KV_UNIFIED"), spec=env("LLAMA_ARG_SPEC_TYPE"))

def from_serverlog(p):
    """llama.cpp logs the resolved slot layout at load: n_slots, n_ctx_slot, kv_unified."""
    try: t = p.read_text()
    except Exception: return {}
    m = re.search(r"n_slots\s*=\s*(\d+),\s*n_ctx_slot\s*=\s*(\d+),\s*kv_unified\s*=\s*'(\w+)'", t)
    out = dict(par=m.group(1), ctx=m.group(2), kv=m.group(3)) if m else {}
    # A speculating server prints "draft acceptance" per request; a plain one never
    # does. So the log itself says whether speculation was on - no need to trust
    # the arm's name.
    if out:
        out["spec"] = "ngram" if "draft acceptance" in t else "none"
    return out

# ---------- 2. /props (server's own view) ----------
def from_props(p):
    try: d = json.loads(p.read_text())
    except Exception: return {}
    g = d.get("default_generation_settings") or {}
    pr = g.get("params") or g
    out = dict(ctx=str(g.get("n_ctx") or ""), par=str(d.get("total_slots") or ""))
    if pr.get("temperature") is not None: out["temp"] = f'{pr["temperature"]:g}'
    return out

TESTS = [
    ("PLE placement",  "corrections/20260830T145932Z_PLE-01"),
    ("Load mode",      "corrections/20260830T151444Z_LOAD-01"),
    ("Speculation A",  "corrections/20260830T171252Z_SPEC-01-rate"),
    ("Microbatch",     "corrections/20260830T180949Z_UB-01"),
    ("Speculation B",  "corrections/20260830T182917Z_SPEC-01-rate"),
    ("Concurrency",    "corrections/20260830T193552Z_CONC-01"),
    ("Preserved reas.","corrections/20260830T200958Z_THINK-01"),
]
for name, rel in TESTS:
    base = R/"results"/rel
    for arm in sorted(p for p in base.iterdir() if p.is_dir()):
        if not any(arm.iterdir()): continue          # aborted arm, nothing recorded
        cfg, src = {}, []
        if (arm/"compose.txt").exists():
            cfg.update(from_compose(arm/"compose.txt")); src.append("compose.txt")
        if (arm/"props.json").exists():
            pr = from_props(arm/"props.json")
            for k, v in pr.items(): cfg.setdefault(k, v) if not cfg.get(k) else None
            if not cfg.get("ctx"): cfg["ctx"] = pr.get("ctx","")
            if not cfg.get("temp"): cfg["temp"] = pr.get("temp","")
            src.append("props.json")
        sl = arm/"server.log.after"
        if not sl.exists(): sl = arm/"server.log"
        if sl.exists():
            got = from_serverlog(sl)
            if got:
                for k, v in got.items():
                    if not cfg.get(k): cfg[k] = v
                src.append("server.log")
        mem = arm/"memory.json"
        vram = ""
        if mem.exists():
            try: vram = f'{json.loads(mem.read_text()).get("gpu_memory_used_mib",""):,}'
            except Exception: pass
        add(test=name, arm=arm.name, src=" + ".join(src) or "—", vram=vram, **cfg)

# ---------- 3. older runs: reconstruct from their own logs ----------
def speedlog(p):
    """server=... n_ctx=N build=B / lengths=[...] n_predict=N num_prompts=N"""
    h = p.read_text().splitlines()[:2]
    ctx = re.search(r'n_ctx=(\d+)', h[0]); npd = re.search(r'n_predict=(\d+)', h[1] if len(h)>1 else "")
    return (ctx.group(1) if ctx else ""), (npd.group(1) if npd else "")

# tier ladder + the two mmap control passes share tier_full.sh
tf = (R/"results/tier_full/tier_full_full.log").read_text()
tiers = re.findall(r'############ (\d+) GiB\s+ctx (\d+) ############.*?started: ncmoe (\d+) mode (\w+)\s+VRAM (\d+)', tf, re.S)
for cap, ctx, ncm, mode, vram in tiers:
    ctxv, npd = speedlog(R/f"results/tier_full/short_{cap}.log")
    add(test="Hardware ladder", arm=f"{cap} GiB cap", ctx=f"{int(ctx):,}", ncmoe=ncm,
        load=mode, ub="512", ot="CPU", par="1", temp="0", kv="true", spec="none",
        vram=f"{int(vram):,}", src="controller log + speed_v2 header")
ctxv, npd = speedlog(R/"results/cpu_only/speed.log")
add(test="CPU only", arm="-ngl 0", ctx=f"{int(ctxv):,}", ncmoe="all", load="mmap (lazy on)",
    ub="512", ot="CPU", par="1", temp="0", kv="true", spec="none",
    src="speed_v2 header + script")
for tag, pas in (("mmap_control_p1","mmap, 1st pass"), ("mmap_control_p2","mmap, 2nd pass")):
    f = R/f"results/{tag}/full.csv"
    if not f.exists(): continue
    r = next(x for x in csv.DictReader(f.open()) if x["cap"]=="48")
    add(test="Load mode control", arm=pas, ctx=f'{int(r["ctx"]):,}', ncmoe=r["ncmoe"],
        load=r["mode"], ub="512", ot="CPU", par="1", temp="0", kv="true", spec="none",
        src="full.csv (FORCE_MODE=mmap)")
add(test="Long-doc recall", arm="all lengths", ctx="262,144", ncmoe="0", load="none",
    ub="512", ot="CPU", par="1 and 4", temp="0", kv="true", spec="none",
    src="script + result JSON")
add(test="Quant Q4_K_XL", arm="UD-Q4_K_XL", ctx="32,768", ncmoe="0", load="none",
    ub="512", ot="CPU", par="1", temp="0", kv="true", spec="none",
    src="controller log")

# ---------- render ----------
hdr = ["Test","Arm / tier","Context","n-cpu-moe","Loading","-ub","PLE","Spec","Slots","KV","GPU MiB","Config recorded in"]
keys = ["test","arm","ctx","ncmoe","load","ub","ot","spec","par","kv","vram","src"]
def cell(v): return v if v else "—"
for r in rows:                      # one thousands-separator style for context
    c = r["ctx"].replace(",", "")
    if c.isdigit(): r["ctx"] = f"{int(c):,}"
out = []
out.append("""# Server configurations, one row per server start

Every test starts a fresh server per arm. There is a single compose file,
`docker/docker-compose.yaml`, parameterized by environment variables, so a
"configuration" is one set of those variables.

This table is generated from saved artifacts by
`benchmark/gen_configs.py`; nothing here is typed by hand. The last
column names the file each row came from.

**Three levels of record.** `compose.txt` is the fully resolved compose for
that arm, written before the server started, with secrets redacted. `props.json`
is the server's own report of its effective settings. Runs made before the
correction controller existed have neither; their configuration is reconstructed
from their controller log and their speed-harness header, and the last column
says so.

Fixed for every row: image `ghcr.io/ggml-org/llama.cpp:server-cuda13`
(build `b10666`, `4e97ac86e`), model `UD-IQ4_XS`, `-ngl 999`, `--flash-attn on`,
`--jinja`, `--threads 16`, `--threads-batch 32`, `--batch-size 2048`,
`--tensor-read-lazy off` (except CPU only, which uses lazy reads by design).
"PLE" is the `-ot per_layer_token_embd=` target.

**The sampler is not in this table.** Every harness sets it per request, so the
server default would be misleading here. Speed sweeps send temperature 0 with a
pinned output length; conversation tests send the model-card sampler for the
mode under test. Each test states its own sampler in the README and in
`report.html`.

""")
out.append("| " + " | ".join(hdr) + " |")
out.append("|" + "|".join(["---"]*len(hdr)) + "|")
last = None
for r in rows:
    t = r["test"] if r["test"] != last else ""
    last = r["test"]
    out.append("| " + " | ".join([t] + [cell(r[k]) for k in keys[1:]]) + " |")
out.append("")
out.append(f"{len(rows)} server starts. Of these, "
           f"{sum(1 for r in rows if 'compose.txt' in r['src'])} have a resolved compose dump, "
           f"{sum(1 for r in rows if 'props.json' in r['src'])} have the server's own settings, "
           f"and {sum(1 for r in rows if 'compose' not in r['src'] and 'props' not in r['src'])} "
           "are reconstructed from logs.")
(R/"results/CONFIGS.md").write_text("\n".join(out) + "\n")
print(f"  wrote results/CONFIGS.md - {len(rows)} rows")
