# Server configurations, one row per server start

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


| Test | Arm / tier | Context | n-cpu-moe | Loading | -ub | PLE | Spec | Slots | KV | GPU MiB | Config recorded in |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PLE placement | A1 | 32,768 | 0 | none | 512 | CPU | none | 1 | true | 63,407 | compose.txt + props.json + server.log |
|  | A2 | 32,768 | 0 | none | 512 | CPU | none | 1 | true | 63,407 | compose.txt + props.json + server.log |
|  | B1 | 32,768 | 0 | none | 512 | CUDA0 | none | 1 | true | 90,927 | compose.txt + props.json + server.log |
|  | B2 | 32,768 | 0 | none | 512 | CUDA0 | none | 1 | true | 90,927 | compose.txt + props.json + server.log |
| Load mode | M1 | 262,144 | 23 | mmap | 512 | CPU | none | 1 | true | 45,527 | compose.txt + props.json + server.log |
|  | M2 | 262,144 | 23 | mmap | 512 | CPU | none | 1 | true | 45,527 | compose.txt + props.json + server.log |
|  | N1 | 262,144 | 23 | none | 512 | CPU | none | 1 | true | 45,581 | compose.txt + props.json + server.log |
|  | N2 | 262,144 | 23 | none | 512 | CPU | none | 1 | true | 45,581 | compose.txt + props.json + server.log |
| Speculation A | p1_ngram-mod | 262,144 | — | — | — | — | ngram | 1 | true | — | props.json + server.log |
|  | p1_none | 262,144 | — | — | — | — | none | 1 | true | — | props.json + server.log |
|  | p2_ngram-mod | 262,144 | — | — | — | — | ngram | 1 | true | — | props.json + server.log |
|  | p2_none | 262,144 | — | — | — | — | none | 1 | true | — | props.json + server.log |
| Microbatch | ub1024_a | 32,768 | 0 | none | 1024 | CPU | none | 1 | true | 63,759 | compose.txt + props.json + server.log |
|  | ub1024_b | 32,768 | 0 | none | 1024 | CPU | none | 1 | true | 63,759 | compose.txt + props.json + server.log |
|  | ub2048_a | 32,768 | 0 | none | 2048 | CPU | none | 1 | true | 64,465 | compose.txt + props.json + server.log |
|  | ub2048_b | 32,768 | 0 | none | 2048 | CPU | none | 1 | true | 64,465 | compose.txt + props.json + server.log |
|  | ub256_a | 32,768 | 0 | none | 256 | CPU | none | 1 | true | 63,231 | compose.txt + props.json + server.log |
|  | ub256_b | 32,768 | 0 | none | 256 | CPU | none | 1 | true | 63,231 | compose.txt + props.json + server.log |
|  | ub512_a | 32,768 | 0 | none | 512 | CPU | none | 1 | true | 63,407 | compose.txt + props.json + server.log |
|  | ub512_b | 32,768 | 0 | none | 512 | CPU | none | 1 | true | 63,407 | compose.txt + props.json + server.log |
| Speculation B | p1_ngram-mod | 262,144 | — | — | — | — | ngram | 1 | true | — | props.json + server.log |
|  | p1_none | 262,144 | — | — | — | — | none | 1 | true | — | props.json + server.log |
|  | p2_ngram-mod | 262,144 | — | — | — | — | ngram | 1 | true | — | props.json + server.log |
|  | p2_none | 262,144 | — | — | — | — | none | 1 | true | — | props.json + server.log |
| Concurrency | p1_kvfalse | 8,192 | — | — | — | — | none | 16 | false | — | props.json + server.log |
|  | p1_kvtrue | 131,072 | — | — | — | — | none | 16 | true | — | props.json + server.log |
|  | p2_kvfalse | 8,192 | — | — | — | — | none | 16 | false | — | props.json + server.log |
|  | p2_kvtrue | 131,072 | — | — | — | — | none | 16 | true | — | props.json + server.log |
| Preserved reas. | redo_on | 262,144 | — | — | — | — | none | 1 | true | — | props.json + server.log |
|  | s1_off | 262,144 | — | — | — | — | none | 1 | true | — | props.json + server.log |
|  | s1_on | 262,144 | — | — | — | — | none | 1 | true | — | props.json + server.log |
| Hardware ladder | 8 GiB cap | 16,384 | 48 | mmap | 512 | CPU | none | 1 | true | 96,641 | controller log + speed_v2 header |
|  | 16 GiB cap | 131,072 | 45 | mmap | 512 | CPU | none | 1 | true | 96,535 | controller log + speed_v2 header |
|  | 24 GiB cap | 262,144 | 42 | mmap | 512 | CPU | none | 1 | true | 96,171 | controller log + speed_v2 header |
|  | 32 GiB cap | 262,144 | 36 | mmap | 512 | CPU | none | 1 | true | 94,805 | controller log + speed_v2 header |
|  | 48 GiB cap | 262,144 | 23 | none | 512 | CPU | none | 1 | true | 93,665 | controller log + speed_v2 header |
|  | 96 GiB cap | 262,144 | 0 | none | 512 | CPU | none | 1 | true | 72,255 | controller log + speed_v2 header |
| CPU only | -ngl 0 | 8,192 | all | mmap (lazy on) | 512 | CPU | none | 1 | true | — | speed_v2 header + script |
| Load mode control | mmap, 1st pass | 262,144 | 23 | mmap | 512 | CPU | none | 1 | true | — | full.csv (FORCE_MODE=mmap) |
|  | mmap, 2nd pass | 262,144 | 23 | mmap | 512 | CPU | none | 1 | true | — | full.csv (FORCE_MODE=mmap) |
| Long-doc recall | all lengths | 262,144 | 0 | none | 512 | CPU | none | 1 and 4 | true | — | script + result JSON |
| Quant Q4_K_XL | UD-Q4_K_XL | 32,768 | 0 | none | 512 | CPU | none | 1 | true | — | controller log |

42 server starts. Of these, 16 have a resolved compose dump, 31 have the server's own settings, and 11 are reconstructed from logs.
