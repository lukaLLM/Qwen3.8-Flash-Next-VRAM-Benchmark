# What we actually tried, per engine

A map of every knob each compose file exposes, what we swept it through, and what
came out. Written for slide-making: the "tested" column is what a viewer should
believe we measured, and the blanks are honest gaps rather than omissions.

**101 runs carry full provenance** — 57 llama.cpp, 27 SGLang, 17 FreeToken —
each recording the server's resolved flags rather than the intent. Counts below
are runs, not requests.

---

## llama.cpp — `docker/docker-compose.yaml`

| Knob | Default | Values tested | Outcome | Section |
|---|---|---|---|---|
| `LOAD_MODE` | `mmap` | **none, mmap, mlock, mmap+mlock, dio** | All five within 4% at 8K and 0.5% at 32K. `mlock` matches `none`. The first report's 1.87× does not reappear because that pair computed 23 expert layers on the CPU. | 09 |
| `LAZY` | `off` | **off, on, on-direct** | +14% prefill at 8K; **nothing at 32K** (1.3% spread). A short-prompt effect, not a general win. | 10 |
| `LLAMA_IMAGE` | ghcr b10666 | **b10666, `llamacpp-pr28136:local`, `llamacpp-mtp:d1a92352`** | Both newer builds decode **~1.20× faster at 32K** than the shipped image, agreeing with each other. The shipped image is stale. | 10 |
| `SPEC_TYPE` | `none` | **none, ngram-mod, draft-mtp** | MTP: 1.63× at 8K, 1.69× at 32K, **2.6× at the full window**, no accuracy cost. n-gram: +6.8% on code, **zero** on prose. | 06, 11 |
| `SPEC_DRAFT_N_MAX` | 3 | **3, 5** | Used 5 for every MTP arm (the guide's value); 47–80% of drafted tokens accepted, 4.1 confirmed per step. Not swept as its own variable. | 11 |
| `CTX` | 32768 | **7,168 → 262,144** (8 values) | Full window served by all four arms. | 04 |
| `PARALLEL` | 1 | **1, 2, 4, 8** | Slot count paired with concurrency. | 07 |
| `UBATCH` | 512 | **512, 1024, 2048** | 1024 is the sweet spot (+8.8%); 2048 no better. *Established in the first report; carried forward here.* | — |
| `OT` / `N_CPU_MOE` | PLE→CPU / 0 | held fixed | The headline placement. Varying it is the first report's subject. | — |
| `KV_UNIFIED` | true | **not swept** | Non-unified KV is the open recommendation for c≥4 — measured in the first report, not re-run here. | — |
| `THREADS` / `THREADS_BATCH` | 16 / 32 | not swept | Left at default throughout. | — |
| `BATCH`, `NGL`, `MIN_P` | 2048 / 999 / 0.0 | not swept | Held constant so the sweeps above are attributable. | — |

## SGLang — `docker/docker-compose.sglang.yaml`

| Knob | Default | Values tested | Outcome | Section |
|---|---|---|---|---|
| `SGLANG_MEM_FRACTION_STATIC` | 0.93 | **0.90, 0.93** | 0.90→0.93 grows the KV pool 166,720 → **262,144 tokens for +1.3 GB** — the whole window, speculation kept. The largest free win here. | 07 |
| `SGLANG_CUDA_GRAPH_MAX_BS_DECODE` | 1 | **1, 2, 8** | Batch 8 sustains **201.6 tok/s** aggregate at two requests over a 20-minute soak, zero errors, zero restarts. Output-identity could not be gated (see 15). | 07 |
| `SGLANG_MAX_RUNNING_REQUESTS` | 1 | **1, 2, 4** | Admission slots; 4 slots at 16,384 context for the accuracy runs. | 08 |
| `SGLANG_CONTEXT_LENGTH` | 131072 | **6,144 → 262,144** (8 values) | Full window at 0.93 mem-fraction. | 04 |
| KV cache dtype | `fp8_e4m3` | **fp8_e4m3, bfloat16** | 8-bit KV is the headline config. Not isolated as its own variable — confounded with engine (see 15). | 15 |
| NEXTN speculation | on | **on, off** | On by default in every headline number. The on/off pair is **confounded** — the arms differ in KV pool fraction too — so +58.9% is not published as a speculation delta. | 06 |
| `SGLANG_CHUNKED_PREFILL_SIZE` | 8192 | **2048, 8192** | Prefill budget, paired with workload. | — |
| `SGLANG_PLE_QUEUE_DEPTH` | 512 | not swept | io_uring queue depth for the NVMe PLE path; left at default. | — |
| `SGLANG_MAX_MAMBA_CACHE_SIZE` | 5 | not swept | The mamba state cache is what actually caps concurrency on this model. | — |

## FreeToken — `docker/docker-compose.freetoken.yaml`

| Knob | Default | Values tested | Outcome | Section |
|---|---|---|---|---|
| KV pool (live, `ft ctl cache --kv`) | — | **4K, 8K, 16K, 64K, 128K tokens** | Resize costs **~1 s** against 82 s to restart. Speed is flat from 64K down to 16K; below the prompt's own length the request is **refused**, not slowed. | 14 |
| `FREETOKEN_IMAGE` | `freetoken:local` | `local` tested; `af71ba432` **built but not measured** | The newer build exists; its accuracy re-run is the main open item. | 15 |
| `MAX_RUNNING_REQUESTS` | 1 | **1, 2, 4, 8** | Slots; 4 for the accuracy runs. | 08 |
| `ENGINE_CTX` | 262144 | **16,384 → 262,144** (7 values) | Full window served. | 04 |
| `FREETOKEN_PLE_BACKEND` | `disk` | not swept | Disk with O_DIRECT throughout. | 02 |
| `FREETOKEN_MOE_BACKEND` | `offload` | **offload** only | `fused` was tried and refused for NVFP4 early on (a fixed blocker), so offload is the only working value. | — |
| `FREETOKEN_NVFP4_BACKEND` | `auto` | **not swept** | `flashinfer` vs `triton` vs auto is an open item — the one row in section 07 with no data. | 07 |
| `FREETOKEN_EXPERT_LOAD` | `serial` | not swept | Left at default. | — |
| `FREETOKEN_MEMORY_RATIO` | 0.90 | **0.80, 0.90** | Paired with pool sizing. | — |

---

## Coverage at a glance, for a slide

- **llama.cpp is the most explored:** 5 load modes × 3 lazy modes × 3 builds ×
  3 speculation types, plus the context ladder. 57 runs.
- **SGLang's big lever was memory fraction** — three hundredths bought the whole
  262,144-token window. 27 runs.
- **FreeToken's big lever was the live cache resize** — the only engine here that
  can be re-tuned without a restart. 17 runs.
- **Deliberately not swept:** thread counts, batch size, NGL, PLE queue depth,
  expert-load order. Held constant so the sweeps above are attributable to one
  variable at a time.
- **Known gaps, stated in the report:** FreeToken's NVFP4 kernel backend, the
  accuracy re-run on the newer FreeToken build, KV-dtype isolated from engine,
  and cold-page-cache prefill for the lazy PLE path.
