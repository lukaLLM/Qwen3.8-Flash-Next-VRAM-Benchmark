# Benchmark harnesses

Methodology — how arms are ordered, what voids a run, what each run saves — is
in the main [README](../README.md) and in [`report.html`](../report.html).
This file is an index.

Every controller prints its plan and exits unless given `--execute` and
`--plan-id CORRECTION-R1`. Each writes one evidence directory per run under
`results/`.

## Controllers

| Script | What it measures |
|---|---|
| `tier_full.sh` | The hardware ladder: VRAM caps × prompt lengths, plus llama-bench per tier |
| `cpu_only.sh` | The model with no GPU at all (`-ngl 0`) |
| `mmap_control_v2.sh` | Loading mode at identical tensor placement: resident, mmap cold, mmap warm |
| `correction_run_v2.sh` | Placement (`--only PLE-01`), microbatch (`--only UB-01`), loading mode (`--only LOAD-01`) |
| `spec01_rate.sh` | n-gram speculation as a decode rate, four counterbalanced pairs |
| `spec01_smoke.sh` | Whether greedy decoding is output-identical between speculation arms |
| `conc01.sh` | Unified against non-unified KV, 1–16 concurrent, three sweeps per start |
| `think01.sh` | Preserved reasoning on and off, fresh server per arm |
| `slot_reuse_long.sh` | Long-document recall before and after the server serves other work |

## Measurement harnesses

| Script | Role |
|---|---|
| `speed_bench_v2.py` | Prefill and decode at given prompt lengths. Writes per-request rows |
| `bench_ngram.py` | The multi-turn coding conversation. Saves every turn's text |
| `preserve_thinking.py` | One preserved-reasoning arm; records prefilled and cached tokens per turn |
| `slot_reuse.py` | Plants facts in a long document and grades recall. `--selftest` proves the checkers can fail |
| `concurrency.py` | Launches N requests together, reports aggregate and per-request rates |

## Guards and monitors

| Script | Role |
|---|---|
| `gpu_settle.sh` | Waits for VRAM release and for the card to cool, then holds a floor delay |
| `gpu_telemetry.py` | Per-arm temperature, power, clocks, throttle flags, DCGM counters. Emits a verdict |
| `ple_io_monitor.py` | Major page-fault and disk-read rate. Voids an arm that swaps |
| `check_config.py` | Compares the live server against the model card for the mode under test |
| `check_outputs.py` | Reads the generated text and flags tool calls, stubs, missing code, partial coverage |
| `vram_cap.py` | Holds VRAM so the card presents a smaller pool |
| `dcgm_exporter.sh` | Starts a DCGM exporter with profiling counters (`dcgm-counters.csv`) |
| `dump_compose.sh` | Writes the resolved, redacted compose file into an arm's evidence directory |

`data/` holds the fixed prompt source for the speed sweep.

`plot_results.py` renders the README charts in [`../assets/`](../assets/)
from the shipped evidence directories.

`gen_configs.py` regenerates [`../results/CONFIGS.md`](../results/CONFIGS.md),
the per-server-start configuration table, from the saved artifacts.
