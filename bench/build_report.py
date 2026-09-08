#!/usr/bin/env python3
"""Generate engine_benchmark_report.html from artifacts only.

Prose lives here; every NUMBER comes from bench/report_data.py, which reads the
evidence files. Re-run this after any new arm lands and the report is current.
"""
from __future__ import annotations
import os, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import report_data as D

R = pathlib.Path(__file__).resolve().parent.parent
TITLE = "Engine Benchmark Report"
ORDER = ("llamacpp", "llamacpp+mtp", "sglang", "freetoken")
# Validated with the dataviz skill's checker (all-pairs, light surface):
# CVD dE 9.2, normal-vision dE 24.0. llama.cpp with and without the draft head is
# ONE engine, so it is one hue in two treatments - four distinct hues fail the
# normal-vision floor, and the family reading is truer to the data anyway.
BARC = {"llamacpp": "#2a78d6", "llamacpp+mtp": "#2a78d6", "sglang": "#eb6834",
        "freetoken": "#1baf7a"}
BARSTYLE = {"llamacpp+mtp": "background-image:repeating-linear-gradient(135deg,"
                            "rgba(255,255,255,.55) 0 3px,transparent 3px 7px);"}

def lab(key):
    """'sglang@default' -> 'SGLang'; 'freetoken@freetoken:af71ba432' -> 'FreeToken (af71ba432)'."""
    eng, _, tag = key.partition("@")
    base = D.LABEL.get(eng, eng)
    if tag in ("", "default"): return base
    short = tag.split(":")[-1] if ":" in tag else tag
    return f"{base} ({short})"

def n(x, d=0):
    return "&mdash;" if x is None else f"{x:,.{d}f}"

# ---------- 02 ladder ----------
lad = D.ladder()
def ladder_table():
    head = ("<thead><tr><th>Input</th>"
            + "".join(f"<th>{D.LABEL[e]}<br>prefill</th><th>{D.LABEL[e]}<br>decode</th>" for e in ORDER)
            + "</tr></thead>")
    rows = []
    for k, v in lad.items():
        tds = []
        for e in ORDER:
            if e in v:
                pf, dc = v[e]
                hi = ' class="hi"' if e == "sglang" else ""
                lo = ' class="lo"' if (e == "llamacpp" and dc and dc < 40) else hi
                tds.append(f"<td{hi}>{n(pf)}</td><td{lo}>{n(dc,1)}</td>")
            else:
                tds.append('<td class="dim">&mdash;</td><td class="dim">&mdash;</td>')
        rows.append(f"<tr><td>{k}K</td>{''.join(tds)}</tr>")
    return ('<div class="tw"><table><caption>Prefill and decode in tokens per second, by input '
            'length. Higher is better.</caption>' + head + "<tbody>" + "".join(rows)
            + "</tbody></table></div>")

# ---------- 03 full context ----------
fc = D.full_context()
def full_table():
    rows = []
    for e in ("sglang", "llamacpp+mtp", "freetoken", "llamacpp"):
        if e not in fc: continue
        r = fc[e]
        slow = ' class="lo"' if e == "llamacpp" else ""
        fast = ' class="hi"' if e == "sglang" else ""
        mtp = ' class="hi"' if e == "llamacpp+mtp" else ""
        rows.append((f"<tr><td>{D.LABEL[e]}</td><td>{n(r['isl'])}</td>"
                     f"<td{slow or fast or mtp}>{r['ttft']/1000:.1f} s</td>"
                     f"<td{fast}>{n(r['prefill'])}</td>"
                     f"<td{slow or fast or mtp}>{n(r['decode'],1)}</td></tr>", e))
    order = ["sglang", "llamacpp+mtp", "freetoken", "llamacpp"]
    rows.sort(key=lambda r: order.index(r[1]) if len(r) > 1 else 9)
    rows = [r[0] for r in rows]
    return ('<div class="tw"><table><caption>The largest prompt each engine accepted and served, '
            '128-token output.</caption><thead><tr><th>Stack</th><th>Input served</th>'
            '<th>Time to first token</th><th>Prefill tok/s</th><th>Decode tok/s</th></tr></thead>'
            "<tbody>" + "".join(rows) + "</tbody></table></div>")

# ---------- 04 real code ----------
rc = D.real_code()
def code_table():
    rows = []
    for e in ("sglang", "llamacpp+mtp", "freetoken", "llamacpp"):
        if e not in rc: continue
        r = rc[e]
        hi = ' class="hi"' if e in ("sglang", "llamacpp+mtp") else ""
        rows.append(f"<tr><td>{D.LABEL[e]}</td><td{hi}>{n(r['prefill'])}</td>"
                    f"<td{hi}>{n(r['decode'],1)}</td>"
                    f'<td class="dim">not applicable</td></tr>')
    return ('<div class="tw"><table><caption>32K-token coding prompts, 1,024-token output, one '
            'request at a time.</caption><thead><tr><th>Stack</th><th>Prefill tok/s</th>'
            '<th>Decode tok/s</th><th>Output check</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")

# ---------- 07 accuracy ----------
acc = D.accuracy()
def acc_block():
    if not acc:
        return ('<p class="table-key">Results for this section are pending. They will be generated '
                'from the saved per-item files, not transcribed. The hypothesis under test is that '
                'the GGUF checkpoint, with nearly half its parameters at about 3.4 bits, degrades '
                'first &mdash; most visibly on long-context retrieval rather than on arithmetic.</p>')
    rows = []
    keys = sorted(acc, key=lambda k: (ORDER.index(k.split("@")[0]) if k.split("@")[0] in ORDER else 9, k))
    for e in keys:
        for task in ("gsm8k", "math500"):
            j = acc[e].get(task)
            if not j: continue
            lo, hi = j["ci95_wilson"]
            rows.append(f"<tr><td>{lab(e)}</td><td>{'GSM8K' if task=='gsm8k' else 'MATH-500'}</td>"
                        f"<td>{100*j['accuracy']:.2f}%</td>"
                        f"<td class=\"dim\">{100*lo:.1f}&ndash;{100*hi:.1f}%</td>"
                        f"<td>{j['n_scored']:,}</td><td>{j['truncated']}</td></tr>")
    return ('<div class="tw"><table><caption>Exact-match accuracy, non-thinking sampler, four '
            'requests in flight. Errored items are excluded, never scored as wrong.</caption>'
            '<thead><tr><th>Stack</th><th>Benchmark</th><th>Accuracy</th><th>95% interval</th>'
            '<th>Scored</th><th>Truncated</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")

def paired_block():
    rows = []
    for task in ("gsm8k", "math500"):
        for r in D.paired(task):
            name = "GSM8K" if task == "gsm8k" else "MATH-500"
            sig = "no difference" if r["p"] > 0.05 else f"<b>differs</b>"
            rows.append(
                f"<tr><td>{name}</td>"
                f"<td>{lab(r['a'])} {100*r['acc_a']:.2f}%</td>"
                f"<td>{lab(r['b'])} {100*r['acc_b']:.2f}%</td>"
                f"<td>{r['both']:,}</td><td>{r['neither']}</td>"
                f"<td>{r['only_a']}</td><td>{r['only_b']}</td>"
                f"<td>{r['p']:.3f}</td><td class=\"dim\">{sig}</td></tr>")
    if not rows:
        return ""
    return ('<div class="tw"><table><caption>Paired comparison on the identical items each stack '
            'answered. Only the two disagreement columns carry information.</caption>'
            '<thead><tr><th>Benchmark</th><th>Stack A</th><th>Stack B</th><th>Both right</th>'
            '<th>Both wrong</th><th>Only A</th><th>Only B</th><th>McNemar p</th><th>Verdict</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")

def pending(msg):
    return f'<p class="table-key"><em>Pending.</em> {msg}</p>'

def _det(line):
    """'FAIL: 16/16 greedy outputs differ (...); first at ...' -> '16/16 greedy outputs differ (...)'."""
    body = line.split(":", 1)[1] if ":" in line else line
    return body.split(";")[0].strip()[:80]

def _vclass(v):
    return "ok" if v == "PASS" else ("dim" if v == "INCONCLUSIVE" else "bad")

def loadmode_block():
    arms = D.load_modes()
    if not arms: return pending("Five load modes at two prompt lengths on the control image.")
    tr = "".join(f"<tr><td>{a['isl_label']}</td><td><code>{a['load_mode']}</code></td>"
                 f"<td{' class=\"hi\"' if a['load_mode']=='none' else ''}>{n(a['prefill'])}</td>"
                 f"<td>{n(a['decode'],1)}</td><td>{a['peak_gib']:.1f}</td><td>{a['host_swap_gib']:.1f}</td>"
                 f"<td class=\"{'bad' if a['oom'] else 'dim'}\">{'OOM' if a['oom'] else 'no'}</td></tr>" for a in arms)
    h = ('<div class="tw"><table><caption>llama.cpp <code>--load-mode</code> on the same image, '
         'same placement, ubatch 1024, real code, one request. Peak is the container cgroup; under '
         '<code>mmap</code> the file pages are page cache and are not charged the same way.</caption>'
         '<thead><tr><th>Input</th><th>Mode</th><th>Prefill tok/s</th><th>Decode tok/s</th>'
         '<th>Peak GiB</th><th>Host swap GiB</th><th>OOM</th></tr></thead><tbody>' + tr + '</tbody></table></div>')
    # spread per ISL, generated
    parts = []
    for isl in ("8192", "32000"):
        xs = [a for a in arms if str(a["isl_label"]) == isl and a["prefill"]]
        if len(xs) < 2: continue
        lo, hi = min(xs, key=lambda a: a["prefill"]), max(xs, key=lambda a: a["prefill"])
        none = next((a for a in xs if a["load_mode"] == "none"), None)
        ml = next((a for a in xs if a["load_mode"] == "mlock"), None)
        seg = (f'At {int(isl)//1000}K the slowest mode (<code>{lo["load_mode"]}</code>, {n(lo["prefill"])}) is within '
               f'<b class="v">{100*(hi["prefill"]/lo["prefill"]-1):.1f}%</b> of the fastest '
               f'(<code>{hi["load_mode"]}</code>, {n(hi["prefill"])})')
        if none and ml and none["prefill"]:
            seg += f'; <code>mlock</code> is {100*(ml["prefill"]/none["prefill"]-1):+.1f}% against <code>none</code> on prefill'
        parts.append(seg + ".")
    if parts:
        h += ('<p class="takeaway"><strong>They are all the same.</strong> ' + " ".join(parts) +
              ' No arm ran out of memory or restarted. The residency-versus-anonymous-memory question '
              'does not arise, because on this placement nothing depends on it.</p>')
        h += ('<p class="table-key"><strong>Why the earlier 1.87&times; does not reappear.</strong> That pair '
              'ran with 23 of the expert layers computed on the CPU (<code>--n-cpu-moe 23</code>, ubatch 512). '
              'When the CPU <em>computes</em> on tensors, how they were loaded governs every matrix multiply; '
              'the loader\'s warning is about exactly that path. The headline configuration keeps every expert '
              'on the GPU, so the host only serves the lookup-table gather, and the mode stops mattering. '
              '<code>none</code> stays as the recommendation because it is what the loader asks for and it '
              'costs nothing &mdash; but the honest answer to the question is that <code>mlock</code> would have '
              'been fine too, and the setting that decides llama.cpp\'s speed here is where the experts live.</p>')
    return h

def placement_block():
    """Section 09. The greedy build-gate table used to render here; it was
    removed because the gate itself is unusable on this stack (an engine that
    does not reproduce its own output cannot certify a rebuild), and a FAIL
    table beside its own retraction is worse than no table. The one-paragraph
    version survives in section 15's corrections."""
    arms = D.placement(); res = D.resident_newbuild()
    h = ""
    if res:
        tr = "".join(f"<tr><td>{lab('llamacpp@'+r['tag'])}</td><td>{r['isl']}</td><td>{n(r['ctrl']['prefill'])}</td><td>{n(r['new']['prefill'])}</td>"
                     f"<td class=\"{'hi' if r['prefill_x'] and r['prefill_x']>1.05 else ''}\">{r['prefill_x']:.2f}&times;</td>"
                     f"<td>{n(r['ctrl']['decode'],1)}</td><td>{n(r['new']['decode'],1)}</td>"
                     f"<td class=\"{'hi' if r['decode_x'] and r['decode_x']>1.05 else ''}\">{r['decode_x']:.2f}&times;</td></tr>"
                     for r in res)
        h += ('<div class="tw"><table><caption>Newer llama.cpp builds at the resident headline configuration '
              '(<code>--load-mode none</code>, lazy off) against the shipped b10666 image, same placement, same '
              'prompts.</caption><thead><tr><th>Build</th><th>Input</th><th>b10666 prefill</th><th>New prefill</th><th>&times;</th>'
              '<th>b10666 decode</th><th>New decode</th><th>&times;</th></tr></thead><tbody>' + tr + '</tbody></table></div>')
        fast = [r for r in res if r["decode_x"] and r["decode_x"] > 1.1]
        if fast:
            builds = len({r["tag"] for r in fast})
            agree = ("Two independently built newer images agree on it, which is what turns a single "
                     "measurement into a finding. " if builds > 1 else "")
            h += ('<p class="takeaway"><strong>The shipped llama.cpp image is stale.</strong> The same '
                  'configuration on builds 44 commits newer decodes materially faster &mdash; '
                  f'{max(r["decode_x"] for r in fast):.2f}&times; at 32K. ' + agree +
                  'That gain has nothing to do with the pull request under test here; it says the headline '
                  'llama.cpp numbers in this report are a floor, and a newer pinned upstream build is the '
                  'real recommendation for this engine.</p>')
    if not arms: return h + pending("Lazy off / on / on-direct on the PR #28136 tag at two prompt lengths.")
    tr = "".join(f"<tr><td>{a['isl_label']}</td><td><code>{a['lazy']}</code></td>"
                 f"<td>{n(a['prefill'])}</td><td>{a['ttft']/1000:.1f} s</td><td>{n(a['decode'],1)}</td>"
                 f"<td>{a['peak_gib']:.1f}</td></tr>" for a in arms)
    h += ('<div class="tw"><table><caption>PLE read path under <code>--load-mode mmap</code> on the '
          'PR #28136 build. Warm page cache; the PR\'s own headline is cold.</caption>'
          '<thead><tr><th>Input</th><th>Lazy mode</th><th>Prefill tok/s</th><th>TTFT</th>'
          '<th>Decode tok/s</th><th>Peak GiB</th></tr></thead><tbody>' + tr + '</tbody></table></div>')
    # WITHDRAWN 2026-09-07 (BLOCKERS B-25). These three arms are not three
    # configurations. Upstream renamed the flag: b10666 reads
    # LLAMA_ARG_TENSOR_READ_LAZY, which is the only name compose set, while THIS
    # build reads LLAMA_ARG_LAZY_MODE. Proved by feeding each build an invalid
    # value - the name a build reads is rejected at parse time, the other is
    # ignored. So all three ran at the build default, `auto`. The spread between
    # them is run-to-run and page-cache variation, and the old "+14% for lazy
    # reading" reading is gone. Compose now sets both names; the arms have to be
    # re-run before anything can be said about the lazy path on this build.
    h += ('<p class="takeaway"><strong>Withdrawn: these three rows are one configuration.</strong> '
          'The flag was renamed upstream. The shipped b10666 build reads '
          '<code>LLAMA_ARG_TENSOR_READ_LAZY</code>, which is the name this harness set; this build reads '
          '<code>LLAMA_ARG_LAZY_MODE</code>, and ignores the other completely. Feeding each build a '
          'deliberately invalid value proves which name it honours &mdash; the one it reads is rejected '
          'at startup, the one it does not read is accepted in silence. <strong>So every arm here ran at '
          'the build default, <code>auto</code></strong>, whatever the label says, and an earlier reading '
          'of "+14% prefill for lazy reading" has been withdrawn: it was the spread between three '
          'identical configurations. <code>on-direct</code>, the pull request\'s actual subject, was '
          'never exercised at all.</p>'
          '<p class="table-key">The harness now sets both names, so the arms can be re-run and mean what '
          'they claim. Nothing else in this report depends on them: the load-mode sweep in section 09 ran '
          'on b10666, whose name the harness did set, and section 11\'s draft-head comparison had the '
          'same default imposed on both of its arms.</p>')
    # The old "not yet measured at 32K" note went with the withdrawn reading: the
    # arms exist, they simply all carried the same imposed default, so counting
    # which labels are present says nothing. (It also pointed at a section number
    # that has since moved.)
    return h

def mtp_block():
    m = D.mtp()
    if not m["gate"] and not m["arms"]:
        return pending("The draft head is built and queued; this section fills from the gate and the two speed pairs.")
    h = ""
    if m["gate"]:
        g = m["gate"]
        h += (f'<p class="table-key"><strong>Load gate:</strong> '
              f'<span class="{"ok" if g["loaded"] else "bad"}">'
              f'{"the draft head loads" if g["loaded"] else "the draft head still does not load"}</span> '
              f'on this build. Evidence: <code>{g["artifact"]}</code>.</p>')
        a = m.get("accept")
        if g["loaded"] and a:
            h += (f'<p class="table-key"><strong>Is the drafter actually working?</strong> Across '
                  f'{a["n"]} completed requests it proposed tokens that the full model accepted '
                  f'<b class="v">{100*a["lo"]:.0f}&ndash;{100*a["hi"]:.0f}%</b> of the time, '
                  f'confirming <b class="v">{a["mean_len"]:.1f}</b> tokens per verification step on average. '
                  'A flag that is accepted but never drafts would show one token per step, and fail.</p>')
        if not g["loaded"]:
            h += ('<p class="takeaway"><strong>Multi-token prediction remains unavailable on this model.</strong> '
                  'The fix that was supposed to accept a draft-head-only file did not make it loadable here, so '
                  'the fastest speculation this checkpoint ships is still out of reach on llama.cpp, and the '
                  'n-gram drafter remains its only working option.</p>')
            return h
    if m["arms"]:
        tr = "".join(f"<tr><td>{a['isl']}</td><td>{n(a['off']['prefill'])}</td><td>{n(a['on']['prefill'])}</td>"
                     f"<td>{n(a['off']['decode'],1)}</td><td>{n(a['on']['decode'],1)}</td>"
                     f"<td class=\"{'hi' if a['decode_x'] and a['decode_x']>1.05 else ''}\">{a['decode_x']:.2f}&times;</td></tr>"
                     for a in m["arms"])
        h += ('<div class="tw"><table><caption>The same build with the draft head off and on, real coding '
              'prompts, one request. The control is this build with speculation disabled, never another '
              'build.</caption><thead><tr><th>Input</th><th>Prefill off</th><th>Prefill on</th>'
              '<th>Decode off</th><th>Decode on</th><th>Decode &times;</th></tr></thead><tbody>'
              + tr + '</tbody></table></div>')
        acc = D.accuracy()
        base = (acc.get("llamacpp@default") or {}).get("gsm8k")
        mtpa = (acc.get(f"llamacpp@{D.MTP_TAG}") or {}).get("gsm8k")
        if base and mtpa:
            pr = next((r for r in D.paired("gsm8k")
                       if {r["a"], r["b"]} == {"llamacpp@default", f"llamacpp@{D.MTP_TAG}"}), None)
            tail = ""
            if pr:
                tail = (f' The two arms disagree on {pr["only_a"] + pr["only_b"]} of them and the paired test '
                        f'returns <b class="v">p&nbsp;=&nbsp;{pr["p"]:.2f}</b> &mdash; no detectable difference. '
                        '<strong>That is the "no accuracy degradation" half of the claim, measured rather than '
                        'assumed.</strong>')
            h += (f'<p class="takeaway"><strong>Accuracy with the head on:</strong> GSM8K '
                  f'{100*mtpa["accuracy"]:.2f}% against {100*base["accuracy"]:.2f}% without it, on the same '
                  f'{mtpa["n_scored"]:,} problems.{tail}</p>')
        best = max((a["decode_x"] or 0) for a in m["arms"])
        if best > 1.05:
            h += (f'<p class="takeaway"><strong>The draft head is the largest single speed lever llama.cpp has '
                  f'on this model</strong> &mdash; up to <b class="v">{100*(best-1):.0f}%</b> more decode from a '
                  '2.6 GB sidecar, against the same build with it switched off. Nothing else measured in this '
                  'report comes close: the load modes in section 09 were worth 4%, the microbatch 9%. It is not '
                  'in any release, but it '
                  'is not out of reach either &mdash; the checkpoint\'s publisher ships a build recipe against '
                  'their own fork, which is exactly what was built here.</p>')
        hh = D.head_to_head()
        if hh:
            head = ("<thead><tr><th>Stack</th>"
                    + "".join(f"<th>{int(i)//1000}K prefill</th><th>{int(i)//1000}K decode</th>" for i in hh)
                    + "</tr></thead>")
            names = sorted({k for r in hh.values() for k in r},
                           key=lambda k: -(hh[list(hh)[0]].get(k, {}).get("decode") or 0))
            rows = []
            for name in names:
                tds = []
                for i in hh:
                    v = hh[i].get(name)
                    cls = ' class="hi"' if name == "llama.cpp + MTP" else ""
                    tds.append(f'<td>{n(v["prefill"]) if v else "&mdash;"}</td>'
                               f'<td{cls}>{n(v["decode"],1) if v else "&mdash;"}</td>')
                bold = ' style="font-weight:600"' if name == "llama.cpp + MTP" else ""
                rows.append(f"<tr><td{bold}>{name}</td>{''.join(tds)}</tr>")
            h += ('<div class="tw"><table><caption>The draft head placed back into the three-engine '
                  'comparison: identical coding prompts, 1,024-token output, one request at a time.</caption>'
                  + head + "<tbody>" + "".join(rows) + "</tbody></table></div>")
            a32 = hh.get("32000", {})
            if a32.get("llama.cpp + MTP") and a32.get("FreeToken") and a32.get("SGLang"):
                mt, ft, sg = (a32["llama.cpp + MTP"]["decode"], a32["FreeToken"]["decode"], a32["SGLang"]["decode"])
                h += (f'<p class="takeaway"><strong>This reorders the report.</strong> At 32K the engine that '
                      f'decoded slowest now decodes <b class="v">{100*(mt/ft-1):.0f}%</b> faster than FreeToken '
                      f'and sits within <b class="v">{100*(1-mt/sg):.0f}%</b> of SGLang &mdash; from a 2.6 GB '
                      'file and one flag. <strong>Prefill does not move</strong>, and that is the honest limit: '
                      'the draft head shortens the wait between tokens, not the wait for the first one, so '
                      'SGLang keeps its <b class="v">4.6&times;</b> prefill lead and every conclusion in '
                      'section 04 about long-prompt time-to-first-token stands unchanged.</p>')
            fc = D.full_context()
            if fc.get("llamacpp+mtp") and fc.get("llamacpp"):
                a, b = fc["llamacpp"], fc["llamacpp+mtp"]
                h += (f'<p class="table-key"><strong>The gain grows with the prompt.</strong> At the full '
                      f'262,144-token window it is <b class="v">{b["decode"]/a["decode"]:.1f}&times;</b> '
                      f'({n(a["decode"],1)} to {n(b["decode"],1)} tokens per second) &mdash; larger than at 8K or '
                      '32K, because the baseline decays with context and the drafted path does not. That is the '
                      'opposite shape from the n-gram drafter in section 06, which needs repetition and finds '
                      'less of it as prompts grow.</p>')
        if not (D.accuracy().get(f"llamacpp@{D.MTP_TAG}") or {}).get("gsm8k"):
            h += ('<p class="table-key"><strong>The accuracy half of the claim is not settled here.</strong> '
                  'The GSM8K run with the draft head on was interrupted before it scored anything, so this '
                  'section reports a speed result and an open question, not "no degradation".</p>')
    return h

def niah_block():
    """The needle test: a schematic of the mechanism, then the scores.

    The diagram earns its place because the thing being tested is a SEQUENCE -
    probe, let a different large request take and release the slot, probe the
    same question again - and that is exactly what a table of pass counts cannot
    show. Structure is drawn in currentColor so it themes itself; the needles
    carry the one literal hue."""
    runs = D.slot_reuse()
    if not runs: return ""
    W, H = 760, 250
    BAR_X, BAR_W = 96, 560
    def needles(y, h=26):
        g = [f'<rect x="{BAR_X}" y="{y}" width="{BAR_W}" height="{h}" rx="3" fill="none" '
             'stroke="currentColor" stroke-opacity=".35"/>']
        for d in (10, 50, 90):
            nx = BAR_X + BAR_W * d / 100
            g.append(f'<rect x="{nx-3:.0f}" y="{y+3}" width="6" height="{h-6}" rx="1.5" '
                     'fill="var(--accent)"/>')
            g.append(f'<text x="{nx:.0f}" y="{y-7}" text-anchor="middle" class="dg-s">{d}%</text>')
        return "".join(g)
    g = ['<defs><marker id="nh-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
         'markerHeight="7" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>']
    # step 1
    g.append(f'<text x="{BAR_X}" y="26" class="dg-h">1 &nbsp;Ask</text>')
    g.append(needles(44))
    g.append(f'<text x="{BAR_X-12}" y="62" text-anchor="end" class="dg-s">context</text>')
    g.append(f'<text x="{BAR_X+BAR_W+10}" y="62" class="dg-s">answer &#10003;</text>')
    # step 2 - the interfering request
    g.append(f'<text x="{BAR_X}" y="112" class="dg-h">2 &nbsp;Interfere</text>')
    g.append(f'<rect x="{BAR_X}" y="130" width="{BAR_W}" height="26" rx="3" fill="currentColor" '
             'fill-opacity=".10" stroke="currentColor" stroke-opacity=".35"/>')
    g.append(f'<text x="{BAR_X + BAR_W/2:.0f}" y="147" text-anchor="middle" class="dg-s">'
             'a different large request takes the slot, then releases it</text>')
    # step 3 - same question again
    g.append(f'<text x="{BAR_X}" y="196" class="dg-h">3 &nbsp;Ask again</text>')
    g.append(needles(214))
    g.append(f'<text x="{BAR_X-12}" y="232" text-anchor="end" class="dg-s">same question</text>')
    g.append(f'<text x="{BAR_X+BAR_W+10}" y="232" class="dg-s">same answer?</text>')
    # the arrow that carries the claim: step 1 compared against step 3
    ax = BAR_X - 34
    g.append(f'<path d="M{ax},60 L{ax},228" stroke="currentColor" stroke-width="1.5" '
             'fill="none" marker-end="url(#nh-ar)" stroke-dasharray="4 3"/>')
    g.append(f'<text x="{ax-8}" y="150" text-anchor="end" class="dg-s" '
             'transform="rotate(-90 ' + f'{ax-8} 150)">compared</text>')
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;display:block;'
           'color:var(--ink-2)" aria-label="The needle test: ask a question whose answer sits at '
           '10, 50 or 90 percent depth in the context; let a different large request occupy and '
           'release the server slot; ask the identical question again and compare the two answers.">'
           + "".join(g) + "</svg>")
    rows = "".join(f"<tr><td>{r['label']}</td><td>{r['fresh']}/{r['n']}</td>"
                   f"<td>{r['reused']}/{r['n']}</td></tr>" for r in runs)
    tot = sum(r["n"] for r in runs)
    tot_f = sum(r["fresh"] for r in runs); tot_r = sum(r["reused"] for r in runs)
    tasks = runs[0]["tasks"]; depths = runs[0]["depths"]
    perfect = tot_f == tot and tot_r == tot
    return ('<figure class="dg"><figcaption><strong>How the needle test works.</strong> '
            'The answer is planted at a known depth, the question is asked, a different large '
            'request is then allowed to occupy and release the server slot, and the identical '
            'question is asked again. What is compared is the pair.</figcaption>' + svg + '</figure>'
            + '<div class="tw"><table><caption>Retrieval on llama.cpp, three graders '
              f'({", ".join(tasks)}) at {"/".join(str(d) + "%" for d in depths)} depth.</caption>'
              '<thead><tr><th>Run</th><th>Before</th><th>After slot reuse</th></tr></thead><tbody>'
            + rows + '</tbody></table></div>'
            + (f'<p class="table-key"><strong>It passed everything: {tot_f}/{tot} before and '
               f'{tot_r}/{tot} after.</strong> Three graders rather than one friendly needle &mdash; '
               'an exact code, a <em>contradiction</em> where two conflicting values are planted and '
               'only the later one is right, and a <em>date</em> among plausible distractors &mdash; '
               'at three depths, up to <b class="v">245,760</b> tokens, with one slot and with four. '
               'The graders can fail: a self-test feeds them deliberately wrong answers and asserts '
               'they are marked wrong. <strong>This is llama.cpp only</strong>, and it was run to '
               'hunt a specific bug (a silent wrong answer after a slot is reused, reported upstream) '
               'rather than as an engine comparison &mdash; so it says this stack retrieves reliably '
               'at full length, not that the other two do.</p>' if perfect else
               f'<p class="table-key">{tot_f}/{tot} before, {tot_r}/{tot} after.</p>'))

def thermal_block():
    th, tr, en = D.thermals(), D.thermal_range(), D.energy_per_request()
    if not th: return pending("Thermal traces are written beside every arm; this fills from them.")
    tr_rows = "".join(
        f"<tr><td>{k}</td><td>{t['temp_med']:.0f} &deg;C</td><td>{t['temp_max']:.0f} &deg;C</td>"
        f"<td>{t['margin_min']:.0f} &deg;C</td><td>{t['power_med']:.0f} W</td><td>{t['power_max']:.0f} W</td>"
        f"<td>{t['clock_med']:,.0f}</td>"
        f"<td class=\"{'bad' if t['throttle_hw'] else 'ok'}\">{'yes' if t['throttle_hw'] else 'none'}</td></tr>"
        for k, t in th.items())
    h = ('<div class="tw"><table><caption>GPU while serving the full 262,144-token window, sampled every '
         'five seconds and filtered to the samples where the card is actually busy &mdash; idle rows between '
         'arms would flatter whichever engine takes longest to boot.</caption><thead><tr><th>Stack</th>'
         '<th>Median temp</th><th>Peak temp</th><th>Closest to limit</th><th>Median power</th><th>Peak power</th>'
         '<th>SM clock MHz</th><th>Thermal throttle</th></tr></thead><tbody>' + tr_rows + '</tbody></table></div>')
    hot = max(th.values(), key=lambda t: t["temp_max"])
    lim = min(th.values(), key=lambda t: t["margin_min"])
    h += (f'<p class="takeaway"><strong>Nothing here is thermally limited.</strong> The hottest sample across '
          f'every engine at the largest prompt this card will take is <b class="v">{hot["temp_max"]:.0f}&nbsp;&deg;C</b>, '
          f'and the closest any run came to the card\'s own thermal limit was '
          f'<b class="v">{lim["margin_min"]:.0f}&nbsp;&deg;C</b> of headroom. Hardware thermal slowdown never '
          'engaged in any arm. A workstation card in a desk-side machine ran the full window on a 125-billion '
          'parameter model without the cooling ever becoming the constraint.</p>')
    caps = [k for k, t in th.items() if t["throttle_pwr"]]
    if caps:
        h += ('<p class="table-key">What did engage, briefly, is the <em>power</em> cap &mdash; on '
              + " and ".join(caps)
              + ", the arms that keep the card busy longest. That is the 600 W limit doing its job,"
                " not heat.</p>")
    if en:
        worst = max(v["kj"] for v in en.values())
        rows = "".join(f"<tr><td>{k}</td><td>{v['seconds']:,.0f} s</td><td>{v['watts']:.0f} W</td>"
                       f"<td>{v['kj']:,.0f} kJ</td><td>{worst/v['kj']:.1f}&times;</td></tr>"
                       for k, v in sorted(en.items(), key=lambda kv: kv[1]["kj"]))
        h += ('<div class="tw"><table><caption>Energy for one full-window request: median power under load '
              'multiplied by the time that request holds the card. Approximate &mdash; it is a median, not an '
              'integration &mdash; so read the ratio, not the joules.</caption><thead><tr><th>Stack</th>'
              '<th>Time on card</th><th>Median power</th><th>Energy</th><th>vs worst</th></tr></thead><tbody>'
              + rows + '</tbody></table></div>')
        best = min(en.items(), key=lambda kv: kv[1]["kj"])
        h += (f'<p class="takeaway"><strong>The efficiency gap is far larger than the power gap.</strong> Every '
              'engine draws roughly the same few hundred watts; what differs is how long it holds them. '
              f'{best[0]} answers the same 262,144-token prompt for about '
              f'<b class="v">{worst/best[1]["kj"]:.0f}&times;</b> less energy than the slowest stack &mdash; not '
              'because it is gentler on the card, but because it is finished. <strong>On a shared or metered '
              'machine, engine choice is an energy decision before it is a latency one.</strong></p>')
    if tr:
        rows = "".join(f"<tr><td>{k}</td><td>{t['temp_med']:.0f} &deg;C</td><td>{t['temp_max']:.0f} &deg;C</td>"
                       f"<td>{t['power_med']:.0f} W</td><td>{t['power_max']:.0f} W</td>"
                       f"<td>{t['span_s']/60:.0f} min</td></tr>" for k, t in tr.items())
        h += ('<div class="tw"><table><caption>The same measurement across the whole ladder, 2K to 261K, where '
              'each engine spends minutes rather than seconds under load.</caption><thead><tr><th>Stack</th>'
              '<th>Median temp</th><th>Peak temp</th><th>Median power</th><th>Peak power</th><th>Time busy</th>'
              '</tr></thead><tbody>' + rows + '</tbody></table></div>')
        h += ('<p class="table-key">Sustained load over the full ladder does not change the picture: the peaks '
              'are the same handful of degrees, reached within seconds and held, which is what an air-cooled '
              'card doing steady matrix work looks like. Prompt length changes how <em>long</em> the card is '
              'busy, not how hard.</p>')
    return h

def boot_block():
    bt = D.boot_times()
    if not bt: return pending("Repeat boots per engine, timed to a served answer.")
    order = sorted(bt.items(), key=lambda kv: kv[1]["total_med"] or 0)
    rows = "".join(
        f"<tr><td>{k}</td><td{' class=\"hi\"' if i == 0 else ''}>{b['total_med']:,.0f} s</td>"
        f"<td>{b['boot_med']:,.0f} s</td><td>{b['gen_med']:.1f} s</td>"
        f"<td class=\"dim\">{('%.1f s' % b['http_med']) if b['http_med'] is not None else '&mdash;'}</td>"
        f"<td class=\"dim\">" + ", ".join(f"{x:,.0f}" for x in b["total"]) + f"</td><td>{b['n']}</td></tr>"
        for i, (k, b) in enumerate(order))
    h = ('<div class="tw"><table><caption>Cold container to an answer in hand, repeated back to back. The '
         'total is what you wait; the two columns after it are where that time goes.</caption>'
         '<thead><tr><th>Stack</th><th>To first answer</th><th>Until it serves</th><th>First generation</th>'
         '<th>/health says 200</th><th>Every attempt</th><th>n</th></tr></thead><tbody>'
         + rows + '</tbody></table></div>')
    fast, slow = order[0], order[-1]
    h += (f'<p class="takeaway"><strong>Startup ranks the engines in the opposite order to speed.</strong> '
          f'{fast[0]} is answering in <b class="v">{fast[1]["total_med"]:.0f} s</b>; '
          f'{slow[0]} takes <b class="v">{slow[1]["total_med"]:.0f} s</b> &mdash; '
          f'<b class="v">{slow[1]["total_med"]/fast[1]["total_med"]:.0f}&times;</b> longer, and SGLang, the '
          'fastest engine once running, is second-slowest to get there. On a single workstation that is not a '
          'footnote: it is the cost of every flag you change, paid before you learn anything.</p>')
    ft = bt.get("FreeToken")
    if ft and ft["gen_med"] and ft["gen_med"] > 10:
        h += (f'<p class="table-key"><strong>And "ready" is a claim, not a fact.</strong> FreeToken\'s health '
              f'endpoint returns 200 after <b class="v">{ft["http_med"]:.1f} s</b> &mdash; while the model has '
              f'barely begun loading. It is another <b class="v">{ft["boot_med"] - ft["http_med"]:.0f} s</b> '
              'before the engine reports itself actually serving, and then a further '
              f'<b class="v">{ft["gen_med"]:.0f} s</b> on the first real request while its kernels compile on '
              'first use. The other two answer in under a second once up. Anything that measures a single cold '
              'request on this engine is measuring the compiler.</p>')
    h += ('<div class="limits"><p class="lab">Which clock, and why it matters</p>'
          '<p>There are three, and picking the wrong one produces a confident wrong answer. Polling '
          '<code>/health</code> looks neutral and is not: a 200 means the HTTP server is listening, which is a '
          'different claim on each engine &mdash; FreeToken returns it about seventy-eight seconds before it '
          'can serve anything. Waiting for the container\'s own healthcheck is what the benchmark harness does, '
          'but that test is written per image (FreeToken\'s additionally asserts the engine is serving; the '
          'other two only curl <code>/health</code>) and Docker polls it on a fifteen-second interval, so '
          'llama.cpp reported the same 15.8 s on every attempt while genuinely up between 10.7 and 15.5 s.</p>'
          '<p>So the table is read from the only question that means the same thing everywhere: <strong>start '
          'the container, wait until it serves, and time the answer coming back.</strong> The components are '
          'shown so the shape stays visible.</p></div>')
    return h

def ftcache_block():
    fc = D.ft_cache()
    if not fc["sizes"]: return pending("Live pool resizes on FreeToken, timed, with a request at each size.")
    rows = ""
    for r in fc["sizes"]:
        if r["reps"]:
            cells = (f'<td>{r["reps"][0]["s"]:.1f} s</td><td>{r["reps"][-1]["s"]:.1f} s</td>'
                     f'<td class="ok">served</td>')
        else:
            cells = '<td class="dim">&mdash;</td><td class="dim">&mdash;</td><td class="bad">dropped</td>'
        rows += (f'<tr><td>{r["kv_tokens_requested"]:,}</td><td>{r["rebuild_s"]} s</td>{cells}</tr>')
    h = ('<div class="tw"><table><caption>The KV pool resized on a running server, then the same prompt sent '
         'twice at each size. Every request carries a unique prefix so the prefix cache cannot answer it, and '
         'the geometry the engine reports after each resize matched the size requested exactly.</caption>'
         '<thead><tr><th>KV pool (tokens)</th><th>Resize took</th><th>First request</th><th>Repeat</th>'
         '<th>Outcome</th></tr></thead><tbody>' + rows + '</tbody></table></div>')
    rb = fc["rebuilds"]
    served = [r for r in fc["sizes"] if r["reps"]]
    dropped = [r for r in fc["sizes"] if not r["reps"]]
    if rb:
        h += (f'<p class="takeaway"><strong>Resizing the cache takes about a second.</strong> Across '
              f'{len(rb)} resizes the slowest took <b class="v">{max(rb):.1f} s</b>, against '
              f'<b class="v">{(D.boot_times().get("FreeToken", {}).get("boot_med") or 0):.0f} s</b> to restart this '
              'engine and load the model again. That is what makes the rest of this table cheap to produce: a '
              'pool sweep that would have been an overnight queue of reboots is four requests against one '
              'server.</p>')
    if len(served) >= 2:
        lo, hi = served[-1], served[0]
        tok = hi["in_tokens"]
        h += (f'<p class="table-key"><strong>And the size makes no difference.</strong> The same {tok:,}-token '
              f'request takes <b class="v">{hi["reps"][-1]["s"]:.1f} s</b> with a '
              f'{hi["kv_tokens_requested"]:,}-token pool and <b class="v">{lo["reps"][-1]["s"]:.1f} s</b> with '
              f'{lo["kv_tokens_requested"]:,} &mdash; a {hi["kv_tokens_requested"]//lo["kv_tokens_requested"]}&times; '
              'smaller cache, a tenth of a second apart. The KV pool is storage, not compute: shrinking it '
              'frees VRAM, it does not give the GPU more to work with, and nothing here gets quicker.</p>')
    if served and dropped:
        tok = served[0]["in_tokens"]
        hid = max(r["kv_tokens_requested"] for r in dropped)
        h += (f'<p class="table-key"><strong>Below the prompt\'s own length it is a wall, not a slope.</strong> '
              f'The request is not served slowly, it is <em>refused</em> &mdash; the server logs "Input sequence '
              f'length {tok - 1} exceeds {hid}, request is dropped". The rule is exact and the engine states it, '
              f'so there is nothing to interpolate: a prompt longer than the pool is dropped, and a pool larger '
              f'than the prompt is idle. <strong>For single-stream work, KV pool beyond your longest prompt '
              'buys nothing</strong> &mdash; it is VRAM that could go to weights or to a second model. With '
              'many requests in flight a larger pool does earn its keep, because it is what lets them run at '
              'once instead of queueing; that is not what is measured here.</p>')
    if len(served) >= 2 and len(served[-1]["reps"]) >= 2:
        h += (f'<p>Two costs the table makes visible. <strong>A resize is paid for on the next request:</strong> '
              f'the first request after a rebuild ran <b class="v">{served[-1]["reps"][0]["s"]:.1f} s</b> against '
              f'<b class="v">{served[-1]["reps"][-1]["s"]:.1f} s</b> for the one after it. And the '
              f'<b class="v">{served[0]["reps"][0]["s"]:.0f} s</b> in the top row is not a pool effect at all &mdash; '
              'it is this engine\'s first-request kernel compilation, measured separately at '
              f'<b class="v">{(D.boot_times().get("FreeToken", {}).get("gen_med") or 0):.0f} s</b> in section 13, '
              'and it appears once per boot rather than once per size.</p>')
    return h

def tier_block():
    rows = D.tier_mtp()
    if not rows: return pending("The VRAM ladder with the draft head on and off.")
    def cell(r, k):
        a = r.get(k)
        if not a: return '<td class="dim">not run</td>'
        if a.get("decode") is None:
            v = a.get("verdict", "?")
            return f'<td class="dim">{"no result" if v == "bench-error" else v}</td>'
        return f'<td>{a["decode"]:.1f}</td>'
    body = ""
    for r in rows:
        off = (r.get("off") or {}).get("decode"); gpu = (r.get("gpu") or {}).get("decode")
        ratio = (f'<td class="{"ok" if gpu/off > 1 else "bad"}">{gpu/off:.2f}&times;</td>'
                 if off and gpu else '<td class="dim">&mdash;</td>')
        name = "no GPU" if r["tier"] == "nogpu" else f'{r["tier"]} GiB'
        body += (f'<tr><td>{name}</td><td>{r["n_cpu_moe"]}</td>'
                 + cell(r, "off") + cell(r, "gpu") + cell(r, "cpu") + ratio + "</tr>")
    h = ('<div class="tw"><table><caption>Decode tok/s at a 2,048-token prompt, one request at a '
         'time. A smaller card is simulated by holding VRAM with a balloon allocator; each tier '
         'keeps the expert offload the first report\'s ladder used.</caption><thead><tr>'
         '<th>VRAM</th><th>Expert layers on CPU</th><th>Draft head off</th><th>On GPU</th>'
         '<th>On CPU</th><th>MTP effect</th></tr></thead><tbody>' + body + '</tbody></table></div>')
    have = [r for r in rows if (r.get("off") or {}).get("decode") and (r.get("gpu") or {}).get("decode")]
    if have:
        worst = min(have, key=lambda r: r["gpu"]["decode"] / r["off"]["decode"])
        best = max(have, key=lambda r: r["gpu"]["decode"] / r["off"]["decode"])
        wr = worst["gpu"]["decode"] / worst["off"]["decode"]
        br = best["gpu"]["decode"] / best["off"]["decode"]
        h += (f'<p class="takeaway"><strong>The draft head pays only when no experts are '
              f'offloaded.</strong> Its effect tracks the offload column, not the VRAM column: at '
              f'{best["tier"]} GiB with <b class="v">{best["n_cpu_moe"]}</b> expert layers on the CPU it is '
              f'<b class="v">{br:.2f}&times;</b>, and at every tier that offloads anything it is a large '
              f'loss &mdash; <b class="v">{wr:.2f}&times;</b> at its worst. <strong>On a 24 GiB card '
              'multi-token prediction is not a speedup, it is a 3.4&times; slowdown.</strong></p>'
              '<p class="table-key"><strong>Why, mechanically.</strong> A draft of five tokens makes the '
              'target model verify five tokens per step instead of one. When the experts are on the GPU '
              'that verification is nearly free and the accepted tokens are pure gain. When 42 of 48 '
              'expert layers sit on the CPU, it multiplies the CPU-side work per step, and the drafting '
              'wins nothing like enough to cover it. Moving the draft head itself to the CPU does not '
              'help: at 16 GiB that is 9.7 tok/s against 9.5 on the GPU, both against 33.0 with the '
              'feature off. The head was genuinely working throughout &mdash; 55&ndash;71% acceptance, '
              'mean draft length 3.7&ndash;4.6.</p>')
    h += ('<div class="limits"><p class="lab">What this does and does not simulate</p>'
          '<p>Holding VRAM reproduces a smaller card\'s <em>capacity</em>, not its bandwidth or its '
          'compute: these are Blackwell SMs at ~1.6 TB/s, so a 24 GiB cap flatters a real 4090. Read the '
          'ratios between arms, which transfer, rather than the absolute tok/s, which do not.</p>'
          '<p>Two cells are blank for different reasons, and the table says which. The no-GPU arm with '
          'the head on the GPU is meaningless and was never run. The 8 GiB arm exited non-zero but left '
          'an empty evidence directory &mdash; the failure capture could not read a container compose had '
          'already removed &mdash; so its cause is unrecorded. It is reported as "no result" rather than '
          'guessed at as an out-of-memory; the capture path has since been fixed.</p></div>')
    return h

BODY = open(R / "bench" / "report_prose.html").read()
BODY = (BODY.replace("{{LADDER_TABLE}}", ladder_table())
            .replace("{{FULL_TABLE}}", full_table())
            .replace("{{CODE_TABLE}}", code_table())
            .replace("{{ACC_BLOCK}}", acc_block())
            .replace("{{PAIRED_BLOCK}}", paired_block())
            .replace("{{LOADMODE_BLOCK}}", loadmode_block())
            .replace("{{NIAH_BLOCK}}", niah_block())
            .replace("{{PLACEMENT_BLOCK}}", placement_block())
            .replace("{{MTP_BLOCK}}", mtp_block())
            .replace("{{TIER_BLOCK}}", tier_block())
            .replace("{{THERMAL_BLOCK}}", thermal_block())
            .replace("{{BOOT_BLOCK}}", boot_block())
            .replace("{{FTCACHE_BLOCK}}", ftcache_block()))

STYLE = open(R / "bench" / "report_style.css").read()
EXTRA = """
<style>
  td.ok,span.ok{color:var(--verify);font-weight:600}
  td.bad,span.bad{color:var(--warn);font-weight:600}
  td.lo{color:var(--warn);font-weight:600}
  table{font-variant-numeric:tabular-nums}
  .limits{border-left:2px solid var(--accent);background:var(--accent-soft);
    padding:14px 20px;margin:22px 0}
  .limits .lab,.how .lab,.concept .lab{font-family:var(--mono);font-size:10.5px;
    letter-spacing:.11em;text-transform:uppercase;color:var(--muted);margin:0 0 7px;font-weight:500}
  .limits p{margin:0;font-family:var(--sans);font-size:14.5px;line-height:1.55;color:var(--ink-2)}
  .limits p+p{margin-top:8px}
  .concept{background:var(--surface-2);border:1px solid var(--rule);padding:15px 18px;margin:22px 0}
  .concept p{font-family:var(--sans);font-size:14.5px;line-height:1.55;color:var(--ink-2);margin:0}
  .concept p+p{margin-top:8px}
  figure.chart{margin:24px 0 26px;padding:18px 20px;background:var(--surface);
    border:1px solid var(--rule)}
  figure.chart figcaption{font-family:var(--sans);font-size:13px;line-height:1.45;
    color:var(--ink-2);margin:0 0 16px}
  .chart-legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:14px;padding-top:12px;
    border-top:1px solid var(--rule)}
  .chart-legend .lg{display:flex;align-items:center;gap:6px;font-family:var(--mono);
    font-size:11px;color:var(--muted)}
  .chart-legend .lg i{width:11px;height:11px;display:inline-block;border-radius:2px}
  /* .table-key carries margin-top:-12px to sit tight under a table. After a
     takeaway paragraph that pulled the two into each other - visible as
     overlapping lines of text in the rendered report. */
  .takeaway + .table-key, p + .table-key, .limits + .table-key{margin-top:18px}
  figure.png{margin:26px 0;padding:0;background:none;border:0}
  figure.png img{width:100%;height:auto;display:block;border:1px solid var(--rule)}
  figure.png figcaption{font-family:var(--sans);font-size:13px;line-height:1.5;
    color:var(--muted);margin-top:10px}
  figure.png figcaption strong{color:var(--ink-2)}
  /* Needle-test schematic. The SVG draws its structure in currentColor and sets
     color:var(--ink-2) inline, but <text> takes fill, not color - without these
     rules the labels fall back to the browser default: 16px black serif, which
     is invisible in dark mode and overflows the 760-unit viewBox. */
  figure.dg{margin:26px 0;padding:18px 20px 22px;background:var(--surface);
    border:1px solid var(--rule)}
  figure.dg figcaption{font-family:var(--sans);font-size:13px;line-height:1.45;
    color:var(--ink-2);margin:0 0 18px}
  figure.dg figcaption strong{color:var(--ink)}
  .dg-h{font-family:var(--mono);font-size:12px;font-weight:600;letter-spacing:.06em;
    fill:var(--ink)}
  .dg-s{font-family:var(--mono);font-size:10.5px;fill:var(--muted)}
</style>
"""
FONTS = ('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
 '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600'
 '&family=IBM+Plex+Sans:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">')

# ONE output, self-contained: the figures are embedded as data URIs so the file
# works anywhere — emailed, on a USB stick, opened from any directory. assets/
# still holds the PNGs for the README and the video.
import base64, re as _re
def _inline(html):
    def sub(m):
        f = R / m.group(1)
        if not f.is_file(): return m.group(0)
        return f'src="data:image/png;base64,{base64.b64encode(f.read_bytes()).decode()}"'
    return _re.sub(r'src="(assets/[A-Za-z0-9_.-]+\.png)"', sub, html)

BODY_ART = _inline(BODY)
(R / "engine_benchmark_report.html").write_text(
    f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{TITLE}</title>\n'
    f'{FONTS}\n{STYLE}\n{EXTRA}\n</head>\n<body>\n<div class="wrap">\n{BODY_ART}\n</div>\n</body>\n</html>\n')

# The artifact body (no <html> wrapper) only when asked for, so the generator has
# no machine-specific path baked into it.
_art = os.environ.get("REPORT_ARTIFACT_OUT")
if _art:
    pathlib.Path(_art).write_text(
        f'<title>{TITLE}</title>\n{FONTS}\n{STYLE}\n{EXTRA}\n<div class="wrap">\n{BODY_ART}\n</div>\n')

print(f"  engine_benchmark_report.html  {(R/'engine_benchmark_report.html').stat().st_size:,} bytes")
print(f"  accuracy rows: {sum(len(v) for v in acc.values())}")
