# The evidence behind the engine benchmark report

Every number in [`engine_benchmark_report.html`](../engine_benchmark_report.html) is read from
these directories by `bench/report_data.py`. Nothing in the report is typed in by hand — run
`uv run bench/build_report.py` and the numbers come back from here. If a run is missing, the
sentence that cited it disappears rather than going stale.

Each directory holds `provenance.json` — the *resolved* server flags, image digest, model
revision and workload parameters, so what actually ran is recorded rather than what was intended
— alongside `profile_export_aiperf.json` (the client-side measurement, taken by one AIPerf
process for all engines so the metrics are computed identically) and whatever else the arm
produced: `summary.json` and per-item responses for accuracy, `boots.json` for startup,
`sweep.json` for the cache resize, `capture.jsonl` for the determinism control.
`thermal/` holds one GPU trace per run, sampled every five seconds — including traces for the
void runs listed below, since a run that was interrupted still recorded what the card was doing.

## Context ladder — section 03

prefill and decode against prompt length, 2K to 254K

- `context_fn_ctxladder_freetoken_20260903T210949Z` — 4.0M
- `context_fn_ctxladder_freetoken_20260903T214035Z` — 9.5M
- `context_fn_ctxladder_freetoken_20260906T174446Z` — 3.9M
- `context_fn_ctxladder_llamacpp_20260903T212404Z` — 2.8M
- `context_fn_ctxladder_llamacpp_20260903T220644Z` — 12M
- `context_fn_ctxladder_llamacpp_20260906T145514Z` — 8.2M · llamacpp-mtp:d1a92352, spec=draft-mtp, load-mode none
- `context_fn_ctxladder_llamacpp_20260906T175538Z` — 6.1M · load-mode none
- `context_fn_ctxladder_llamacpp_20260906T182130Z` — 6.2M · llamacpp-mtp:d1a92352, spec=draft-mtp, load-mode none
- `context_fn_ctxladder_sglang_20260903T205922Z` — 11M
- `context_fn_ctxladder_sglang_20260903T212624Z` — 20M
- `context_fn_ctxladder_sglang_20260906T173725Z` — 8.1M

## Full window — section 04

the largest prompt each engine accepted and served

- `context_fn_maxctx_freetoken_20260903T162420Z` — 1.5M
- `context_fn_maxctx_llamacpp_20260903T173832Z` — 2.2M
- `context_fn_maxctx_llamacpp_20260906T111813Z` — 2.2M · llamacpp-mtp:d1a92352, spec=draft-mtp, load-mode none
- `context_fn_maxctx_sglang_20260903T204814Z` — 6.2M

## Real code — sections 05, 07, 09, 10, 11

LiveCodeBench prompts: the load-mode sweep, the lazy-PLE placement arms, and the draft head on and off

- `fn_code_tune_freetoken_20260903T225554Z` — 2.2M
- `fn_code_tune_freetoken_20260903T231437Z` — 3.5M
- `fn_code_tune_llamacpp_20260905T111107Z` — 2.6M · load-mode none
- `fn_code_tune_llamacpp_20260905T111656Z` — 2.6M · load-mode mmap
- `fn_code_tune_llamacpp_20260905T112254Z` — 2.6M · load-mode mlock
- `fn_code_tune_llamacpp_20260905T112840Z` — 2.6M · load-mode mmap+mlock
- `fn_code_tune_llamacpp_20260905T113438Z` — 2.6M · load-mode dio
- `fn_code_tune_llamacpp_20260905T114020Z` — 4.4M · load-mode none
- `fn_code_tune_llamacpp_20260905T115053Z` — 4.4M · load-mode mmap
- `fn_code_tune_llamacpp_20260905T120142Z` — 4.4M · load-mode mlock
- `fn_code_tune_llamacpp_20260905T121218Z` — 4.4M · load-mode mmap+mlock
- `fn_code_tune_llamacpp_20260905T122300Z` — 4.4M · load-mode dio
- `fn_code_tune_llamacpp_20260905T125130Z` — 2.6M · llamacpp-pr28136:local, load-mode mmap
- `fn_code_tune_llamacpp_20260905T125706Z` — 2.5M · llamacpp-pr28136:local, load-mode mmap, lazy=on
- `fn_code_tune_llamacpp_20260905T130224Z` — 2.5M · llamacpp-pr28136:local, load-mode mmap, lazy=on-direct
- `fn_code_tune_llamacpp_20260905T130741Z` — 4.3M · llamacpp-pr28136:local, load-mode mmap
- `fn_code_tune_llamacpp_20260905T220259Z` — 2.6M · llamacpp-mtp:d1a92352, load-mode none
- `fn_code_tune_llamacpp_20260905T220846Z` — 2.6M · llamacpp-mtp:d1a92352, spec=draft-mtp, load-mode none
- `fn_code_tune_llamacpp_20260905T222024Z` — 4.2M · llamacpp-mtp:d1a92352, load-mode none
- `fn_code_tune_llamacpp_20260905T222937Z` — 4.3M · llamacpp-mtp:d1a92352, spec=draft-mtp, load-mode none
- `fn_code_tune_llamacpp_20260906T155752Z` — 4.3M · llamacpp-pr28136:local, load-mode mmap, lazy=on-direct
- `fn_code_tune_llamacpp_20260906T160724Z` — 4.3M · llamacpp-pr28136:local, load-mode mmap, lazy=on
- `fn_code_tune_llamacpp_20260906T161624Z` — 2.5M · llamacpp-pr28136:local, load-mode none
- `fn_code_tune_llamacpp_20260906T162141Z` — 4.2M · llamacpp-pr28136:local, load-mode none
- `fn_code_tune_sglang_20260903T225048Z` — 5.4M
- `fn_code_tune_sglang_20260903T230845Z` — 7.7M

## Accuracy — section 08

GSM8K and MATH-500, every prompt, response, extracted answer and verdict

- `quality_freetoken_20260904T193052Z` — 4.9M
- `quality_llamacpp_20260904T180528Z` — 5.3M
- `quality_llamacpp_20260906T135854Z` — 3.1M · llamacpp-mtp:d1a92352, spec=draft-mtp, load-mode none
- `quality_sglang_20260904T164329Z` — 4.3M

## Soak — section 07

twenty minutes at graph batch 8, two concurrent requests

- `fn_fast_sglang_20260905T104808Z` — 50M

## Determinism control — section 15

the same server, same configuration, captured twice

- `greedy_mtprep_llamacpp_20260905T221716Z` — 140K · llamacpp-mtp:d1a92352, load-mode none
- `greedy_specoffrep_sglang_20260906T144959Z` — 212K

## Boot time — section 13

cold container to a served answer, repeated

- `boot_freetoken_20260906T134213Z` — 124K
- `boot_llamacpp_20260906T132928Z` — 64K · load-mode none
- `boot_sglang_20260906T133125Z` — 196K

## Live cache resize — section 14

the KV pool resized on a running server

- `ftcache_20260906T120411Z` — 140K

## VRAM tier x MTP — section 15

`fn_tier_llamacpp_20260907T*` (22 dirs) plus `results/tier_mtp/` — `verdicts.json`
(one row per arm with its verdict), `arms.tsv`, `cap_<tier>.log` (the balloon
allocator's own record of what it held) and `cgroup_<arm>.tsv` (the container's
memory and disk counters sampled every 5 s while the arm ran). A smaller card is
simulated by holding VRAM; each tier keeps the expert offload the first report's
ladder used. Arms: draft head off / on the GPU / on the CPU, and the viewer's
24 GiB + 64 GB / 56 GB RAM caps with the PLE resident or read from disk.
Read by `report_data.tier_mtp()` via `verdicts.json`; the cap is proven by
`memory.json.cap_readback` — what Docker and the live cgroup actually reported — not
by the value the run asked for.

## What is deliberately not here

Kept on the machine, left out of the repo. Nothing was deleted: a run that produced a bad number
is evidence about the harness, and deleting it would hide the correction.

### Void runs

Every one is kept, never deleted, and never pooled into a result.

| run | why it is void |
|---|---|
| `quality_llamacpp_20260904T163822Z` | killed mid-run: 1,143 errored items beside an apparent 60.8%. Counting those errors as wrong answers would have reported ~8% and looked like catastrophic quantisation damage instead of an interrupted run. |
| `fn_code_tune_llamacpp_20260905T131640Z` | the host reset at profiling 8 of 16; no measurement was written. |
| `quality_llamacpp_20260905T223747Z` | killed by a ten-minute command timeout before scoring anything. Superseded by `quality_llamacpp_20260906T135854Z`. |
| `context_fn_ctxladder_llamacpp_20260906T110020Z` | a `timeout` wrapper was absorbed by the script's own signal trap; the client hung with an idle GPU. Superseded by `…20260906T145514Z`. |
| `fn_tier_llamacpp_20260907T154236Z`, `…154328Z` | first smoke attempts, failed preflight (DCGM exporter down); re-run as `…154504Z`, `…154728Z`. |
| `fn_tier_llamacpp_20260907T160057Z` | **no-GPU + CPU draft** — hit the 3600 s client timeout at 2 of 4 requests (~20 min each). Kept as the "too slow to measure" evidence; carried in `verdicts.json` as `bench-error`. |
| `fn_tier_llamacpp_20260907T170403Z` | **8 GiB + GPU draft** — exited non-zero and left an empty directory: the failure capture could not read a container compose had already removed. Cause unrecorded; captioned "no result", never "OOM". Capture path fixed afterwards. |
| `fn_tier_llamacpp_20260907T172722Z` | t24_off from the first pass, superseded by `…173126Z` after the run was stopped to patch the driver. |
| `fn_tier_llamacpp_20260907T172937Z`, `…193112Z` | arms in flight when the run was stopped by hand; partial, not pooled. |
| `ftcache_20260906T115423Z`, `…115655Z`, `…115941Z` | the first sweeps sent an identical prompt at every pool size and were answered from the prefix cache (53 s, then 4.5 s). Superseded by `…120411Z`. |

The out-of-window top rung is a different case: those directories are here and the
reader drops just the one bad cell. The ladder's 261,120-token rung asked for 261,120 input plus a
2,048-token answer against a 262,144-token window; SGLang and llama.cpp refused it, FreeToken
served it anyway. It was re-derived at 259,584 and re-run on all four arms.

### Superseded, quarantined and empty

Earlier boots, earlier sweeps and the whole 2026-09-03 screening pass are kept locally but not
shipped. `_quarantine/` holds runs rejected when they were made. A handful of directories are
empty shells from arms that failed before writing anything.

Inside the directories that *are* here, everything the harness wrote is included — including the
raw per-request exports and server logs. They are large, and they are the point: the claim is
that the numbers came from these runs, and that is checkable only if the runs are here.
