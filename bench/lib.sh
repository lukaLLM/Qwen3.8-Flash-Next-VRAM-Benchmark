#!/usr/bin/env bash
# ============================================================================
# Shared benchmark library. SOURCE this, do not execute it.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
#   bench_init vllm w1_chat
#
# Extracted from run.sh so discover.sh / validate.sh / specdecode.sh cannot
# drift from it. Every fairness step lives here exactly once: if a caller skips
# settle/preflight/provenance, that is now visible as a missing call rather
# than as a subtly different copy of the same 20 lines.
#
# WORKING DIRECTORY MATTERS: aiperf must run from the repo root, which contains
# no credential file. Both aiperf and its mock server use pydantic-settings,
# which auto-loads one from the CWD, rejects unknown keys, and prints the value
# in the resulting traceback. Compose and its credentials live in docker/.
# See FINDINGS.md F6.
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Thermal and telemetry policy. HOUSE POLICY, not vendor specification.
#
# GPU_TEMP_ABORT is a safety floor we chose, NOT NVIDIA's throttle point. This
# card reports T.Limit MARGINS rather than absolute trip points: at 29C idle
# nvidia-smi reports temperature.gpu.tlimit=64, putting the real thermal limit
# near 93C, with Slowdown/Shutdown quoted as -2/-5 offsets from the
# max-operating reference. The widely repeated "Blackwell throttles at 88-90C"
# figure UNDERSTATES this card. See FINDINGS.md F4a.
#
# We still abort at 87C, because the point is to refuse to start a measured run
# on a hot card - not to predict the hardware's own limit.
# ---------------------------------------------------------------------------
: "${GPU_TEMP_ABORT:=90}"       # abort ceiling; see the watchdog notes below
: "${SETTLE_TEMP:=50}"          # full settle target (engine down, VRAM freed)
: "${SETTLE_VRAM_MIB:=2000}"
: "${SETTLE_FLOOR:=30}"
: "${SETTLE_CAP:=300}"
: "${COOLDOWN_TEMP_INTER:=60}"  # between cells with the server still booted
: "${COOLDOWN_FLOOR:=45}"
: "${COOLDOWN_CAP:=240}"
: "${THERMAL_INTERVAL:=5}"      # sampler cadence, seconds
: "${HOST_CPU_SATURATION_PCT:=90}"  # per-core ceiling before a cell is suspect

# HARD CEILING on any single aiperf invocation. On 2026-08-25 a c=64 cell hung
# after logging "_finalize_and_process_results completed": the process stayed
# alive, the engine went idle (0 requests, GPU 0%/5.6W) and NOTHING advanced for
# 4h18m of unattended time. A cell that should take ~10 minutes must never be
# able to consume a night.
#
# Generous on purpose - this is a hang detector, not a performance budget. Any
# cell exceeding it is broken, not slow.
: "${AIPERF_TIMEOUT:=3600}"

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARN: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# bench_init <engine> <workload>
#
# Resolves paths, sources the workload conf, and sets the engine-specific
# control-plane facts. Sets globals the other functions rely on.
# ---------------------------------------------------------------------------
bench_init() {
  ENGINE="${1:?usage: bench_init <vllm|sglang> <workload>}"
  WORKLOAD="${2:?usage: bench_init <vllm|sglang> <workload>}"

  case "$ENGINE" in
    vllm|sglang|llamacpp|freetoken) ;;
    *) die "engine must be vllm|sglang|llamacpp|freetoken, got '$ENGINE'" ;;
  esac

  BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO="$(cd "$BENCH_DIR/.." && pwd)"
  COMPOSE_DIR="$REPO/docker"
  # Workload configs are .conf on purpose: this repo treats the dotfile form as
  # secret-bearing, and both the permission rules and pydantic-settings treat
  # the two differently. Never rename these.
  WORKLOAD_FILE="$BENCH_DIR/workloads/${WORKLOAD}.conf"
  [[ -f "$WORKLOAD_FILE" ]] || die "no such workload: $WORKLOAD_FILE"

  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

  # ---------------------------------------------------------------------------
  # Per-engine facts. Checkpoint identity lives HERE, not in each workload conf,
  # so it cannot drift from the compose files - preflight asserts the served
  # model matches MODEL. MODEL is the SERVED NAME (what /v1/models reports),
  # which on this study is an alias and NOT an HF repo id; the repo and its
  # pinned revision are carried separately for provenance.
  #
  # Fields, and why each one is per-engine rather than global:
  #   COMPOSE_FILE/SERVICE  this repo keeps one compose file per arm, not one
  #                         file with N services
  #   RESET_PATH            EMPTY means the engine has no cache-flush endpoint.
  #                         llama.cpp has only POST /slots/:id (server.cpp:286)
  #                         and FreeToken documents none at all, so those two
  #                         rely on --cache-bust instead. See engine_aiperf_args.
  #   SERVER_METRICS        FreeToken documents no Prometheus endpoint; the
  #                         post-run gate would otherwise fail that arm on every
  #                         single run for a property of the engine.
  #   EXTRA_INPUTS          llama.cpp's schema has ignore_eos
  #                         (server-schema.cpp:471) but NO min_tokens, so sending
  #                         it risks a 400. ignore_eos alone still forces
  #                         generation to max_tokens, so exact OSL survives.
  # ---------------------------------------------------------------------------
  ENGINE_EXTRA_INPUTS=()
  case "$ENGINE" in
    llamacpp)
      : "${MODEL:=qwen3.8-flash-next}"          # docker-compose.yaml --alias
      : "${MODEL_REPO:=unsloth/Qwen3.8-Flash-Next-GGUF}"
      : "${MODEL_REVISION:=824f539b2710e5a9e47af4952cf6578cf5ee8932}"
      : "${MODEL_QUANT:=UD-IQ4_XS}"
      COMPOSE_FILE="docker-compose.yaml"; SERVICE="q38n"   # `server` is the NETWORK name, not the service
      CONTAINER="${CONTAINER_NAME:-q38n}"
      : "${ENGINE_HOST_PORT:=8000}"
      RESET_PATH=""                              # no flush endpoint; cache-bust instead
      SERVER_METRICS=1; METRICS_PATH=/metrics    # compose passes --metrics
      ENGINE_EXTRA_INPUTS=(ignore_eos:true cache_prompt:false)
      ENGINE_MIN_TOKENS=0                        # NOT in the schema - would 400
      ;;
    sglang)
      : "${MODEL:=Qwen3.8-Flash-Next}"           # --served-model-name
      : "${MODEL_REPO:=RadixArk/Qwen3.8-Flash-Next-NVFP4}"
      : "${MODEL_REVISION:=7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
      : "${MODEL_QUANT:=NVFP4}"
      COMPOSE_FILE="docker-compose.sglang.yaml"; SERVICE="flashnext"
      # SPEC=off removes the four NEXTN flags. Without NEXTN a request needs 1
      # mamba state slot instead of 5, so admission is no longer pinned by the
      # state cache - the mamba sizing below accounts for that.
      #
      # This USED to swap in a hand-maintained docker-compose.sglang.nospec.yaml,
      # and that file had drifted: it defaulted --mem-fraction-static to 0.90
      # against the base's 0.93, and hard-coded --cuda-graph-max-bs-decode 1
      # where the base reads the env var. So every SPEC=off arm was a
      # TWO-VARIABLE comparison and silently ignored CUDA_GRAPH_MAX_BS. The
      # override is now RENDERED from the base by engine_env_map (see
      # bench/sglang_render_compose.py), so those two values are inherited by
      # construction and the drift is unrepresentable rather than policed.
      COMPOSE_OVERRIDE=""
      CONTAINER="${SGLANG_CONTAINER_NAME:-flashnext}"
      : "${ENGINE_HOST_PORT:=8001}"
      RESET_PATH="/flush_cache"
      SERVER_METRICS=1; METRICS_PATH=/metrics    # compose passes --enable-metrics
      ENGINE_EXTRA_INPUTS=(ignore_eos:true)
      ENGINE_MIN_TOKENS=1
      ;;
    freetoken)
      : "${MODEL:=Qwen3.8-Flash-Next}"
      : "${MODEL_REPO:=RadixArk/Qwen3.8-Flash-Next-NVFP4}"
      : "${MODEL_REVISION:=7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
      : "${MODEL_QUANT:=NVFP4}"
      # Containerised like the other two arms. FreeToken ships as a pip package
      # with no image, so docker/freetoken.Dockerfile builds one - and that is
      # worth the build: it puts all three engines behind the SAME cgroup
      # memory guard (mem_limit/memswap_limit) instead of giving one of them a
      # different failure mode. On a 91 GiB box whose 47.68 GiB PLE table is
      # pinned in host RAM, the guard is the only thing between an overrun and
      # a dead machine, and it must not vary by arm.
      COMPOSE_FILE="docker-compose.freetoken.yaml"; SERVICE="freetoken"
      CONTAINER="${FREETOKEN_CONTAINER_NAME:-freetoken}"
      : "${ENGINE_HOST_PORT:=8002}"
      RESET_PATH=""                              # none documented
      # W4, TWICE CORRECTED. The docs list no metrics endpoint. I then found
      # /engine/metrics registered in the SOURCE and "corrected" W4 to say the arm
      # does serve metrics. Measured at runtime, that was also wrong:
      #
      #   /health           200
      #   /v1/models        200
      #   /engine/health    404
      #   /engine/metrics   404
      #
      # Those routes exist in the package but are not mounted by this serve path.
      # So the arm genuinely has no reachable Prometheus endpoint, and the honest
      # setting is 0 - recorded as a property of the engine, not an exemption
      # granted to make a gate pass. preflight enumerates the candidates anyway so
      # the record is measured rather than asserted.
      SERVER_METRICS=0; METRICS_PATH=/engine/metrics
      ENGINE_EXTRA_INPUTS=(ignore_eos:true)
      ENGINE_MIN_TOKENS="${ENGINE_MIN_TOKENS:-0}"  # unconfirmed; FT-1 flips this to 1
      ;;
    vllm)
      # Kept from the upstream harness. No vLLM arm in this study.
      : "${MODEL:=Qwen/Qwen3.8-27B-FP8}"
      : "${MODEL_REVISION:=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a}"
      COMPOSE_FILE="docker-compose.yaml"; SERVICE="vllm"
      CONTAINER="EngineBench_vLLM"
      : "${ENGINE_HOST_PORT:=8010}"
      RESET_PATH="/reset_prefix_cache"           # needs VLLM_SERVER_DEV_MODE=1
      SERVER_METRICS=1; METRICS_PATH=/metrics
      ENGINE_EXTRA_INPUTS=(ignore_eos:true)
      ENGINE_MIN_TOKENS=1
      ;;
  esac

  # A workload conf uses bare assignments, so `source` silently overwrites any
  # value the caller exported. The chunked-prefill sweep of 2026-09-04 ran two
  # IDENTICAL arms for this reason: fn_smoke.conf pins PREFILL_BUDGET=2048, so
  # `env PREFILL_BUDGET=16384 ./bench/run.sh ...` measured the control twice and
  # nothing in the run reported that the override had been discarded. The ubatch
  # sweep beside it worked only because UBATCH is not conf-declared.
  #
  # Snapshot the caller's overrides, source, then put them back. Only NON-EMPTY
  # overrides are restored: Auto_Bench.md 3 records an empty override silently
  # re-selecting the control configuration, and that must not be honoured here.
  # ISL_LADDER is here so a SINGLE rung of a ladder workload can be re-run
  # without editing the conf: the corpus, OSL, engine context and sampler stay
  # exactly as the published ladder ran them, and only the rung changes. Added
  # 2026-09-05 to measure one high rung inside a bounded run time.
  local _sweepable=(ISL ISL_LADDER OSL CONCURRENCY DURATION WARMUP SEED GOODPUT PROMPT_CORPUS
                    ENGINE_CTX PREFILL_BUDGET MAX_RUNNING_REQUESTS CUDA_GRAPH_MAX_BS
                    UBATCH LOAD_MODE LAZY THINKING SPEC SPEC_TYPE
                    SPEC_DRAFT_MODEL SPEC_DRAFT_N_MAX SPEC_DRAFT_NGL
                    # CONTEXT_REQUESTS was missing until 2026-09-10 and queue 6
                    # lost its whole repeat design to it: exported 3, conf said
                    # 1, conf won, and `workload_overrides` recorded None so the
                    # evidence did not show the request had been dropped.
                    CONTEXT_REQUESTS THERMAL_RETRIES
                    SGLANG_MEM_FRACTION_STATIC
                    SGLANG_LINEAR_ATTN_DECODE_BACKEND SGLANG_LINEAR_ATTN_PREFILL_BACKEND
                    SGLANG_MAMBA_SSM_DTYPE SGLANG_SPEC_DRAFT_QUANT
                    SGLANG_SLEEP_ON_IDLE SGLANG_DISABLE_FI_AUTOTUNE
                    SGLANG_PYTORCH_ALLOC_CONF SGLANG_MAMBA_CONV_DTYPE_VAL
                    SGLANG_NUMA_BIND_V2_VAL SGLANG_OMP_NUM_THREADS
                    SGLANG_JIT_CACHE_HOST_DIR)
  local _ov=() _n _val
  for _n in "${_sweepable[@]}"; do
    [[ -n "${!_n:-}" ]] && _ov+=("${_n}=${!_n}")
  done

  # shellcheck disable=SC1090
  source "$WORKLOAD_FILE"

  WORKLOAD_OVERRIDES=""
  for _val in "${_ov[@]+"${_ov[@]}"}"; do
    _n="${_val%%=*}"
    if [[ "${!_n:-}" != "${_val#*=}" ]]; then
      echo "[bench] env override kept over ${WORKLOAD}.conf: ${_val} (conf said ${!_n:-unset})" >&2
      WORKLOAD_OVERRIDES+="${_val} "
    fi
    printf -v "$_n" '%s' "${_val#*=}"
    export "$_n"
  done
  export WORKLOAD_OVERRIDES="${WORKLOAD_OVERRIDES% }"

  PORT="${ENGINE_HOST_PORT:-8010}"
  BASE="http://localhost:${PORT}"
  DCGM_URL="${DCGM_URL:-localhost:9401}"
  # The environment is uv-managed: pyproject.toml + uv.lock pin aiperf to the
  # source checkout (the PyPI 0.12.0 release lacks reset_kv_cache despite
  # reporting the same version). `uv sync` reproduces it exactly.
  AIPERF="$REPO/.venv/bin/aiperf"
  [[ -x "$AIPERF" ]] || die "aiperf not found - run: uv sync --project $REPO"

  # Run aiperf OFFLINE. It resolves the tokenizer through the HF Hub API at
  # startup; on 2026-08-26 that call hung for 4 minutes on flaky connectivity and
  # timed out `profile_configure`, killing a c=8 cell and aborting the whole
  # validation ladder (logs.md #40). The checkpoint is fully cached (29 GB, 74
  # files at the pinned revision), so no network access is needed or wanted:
  # a benchmark must not depend on the internet mid-run, and a silent re-download
  # would also mean the run no longer used the revision we pinned.
  # DO NOT SET HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE HERE. MEASURED 2026-09-02:
  # they break tokenizer loading outright, and they do it in a way that reads as
  # a missing cache rather than a bad setting.
  #
  #   HF_HUB_OFFLINE is PRESENCE-checked, not value-checked - `=0` is still
  #   "offline". Offline mode sends aiperf down Tokenizer._from_pretrained_local,
  #   which resolves the name through snapshot_download(local_files_only=True).
  #   That rejects a local PATH outright (it validates the string as a repo id)
  #   and it also rejects our repo ID, because this cache was built by
  #   scripts/download_models.sh with aria2c rather than by huggingface_hub.
  #   Every combination fails:
  #     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   path FAIL   repo id FAIL
  #     HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=1   path FAIL   repo id FAIL
  #     both UNSET                                path OK     repo id OK
  #
  # The original reason for forcing offline was real - a hub call hung for four
  # minutes mid-run and killed a c=8 cell - so keep the protection, just not by
  # a mechanism that also disables the tokenizer. Short timeouts make a stray
  # hub call fail in seconds instead of hanging, and TOKENIZER is a local path,
  # which needs no hub call at all.
  export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-5}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-10}"
  export HF_HUB_DISABLE_TELEMETRY=1

  # MODEL is a served ALIAS here, not an HF repo id, so aiperf cannot resolve a
  # tokenizer from it - and with HF_HUB_OFFLINE=1 the attempt fails hard rather
  # than silently. Point every engine at the SAME local tokenizer instead.
  #
  # That is a fairness property, not a convenience: prompts are built client
  # side, so one tokenizer means all three engines receive byte-identical text
  # at identical token targets. Three tokenizers would make "32K context" three
  # different lengths. The audit asks for exactly this
  # ("--tokenizer SAME_HF_TOKENIZER_PATH").
  : "${TOKENIZER:=$HOME/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
  [[ -d "$TOKENIZER" ]] || die "tokenizer dir not found: $TOKENIZER"
}

# ---------------------------------------------------------------------------
# engine_aiperf_args - the flags that differ BY ENGINE, in one place so run.sh,
# context.sh, discover.sh and validate.sh cannot describe the same engine
# differently. Populates the global array ENGINE_ARGS.
#
# Cold prefill is enforced two ways, and the belt-and-braces is deliberate:
#
#   1. --cache-bust system_prefix   works on EVERY engine, needs no endpoint.
#      Injects a per-trajectory SHA-256 marker at token 0, so no two
#      trajectories share a prefix. This is the only mechanism llama.cpp and
#      FreeToken have, since neither exposes a flush hook.
#   2. --reset-kv-cache             only where the engine serves one.
#
# Auto_Bench.md 8 is explicit that cold, warm and resident are three states that
# must never be averaged. Without (1), a repeated length reads tens of times
# faster and that is the easiest way to publish a wrong number from this stack.
# ---------------------------------------------------------------------------
engine_aiperf_args() {
  ENGINE_ARGS=(
    --tokenizer "$TOKENIZER"
    --cache-bust "${CACHE_BUST:-system_prefix}"
  )
  if [[ -n "${INPUT_FILE:-}" ]]; then
    [[ -f "$INPUT_FILE" ]] || die "custom dataset not found: $INPUT_FILE (run bench/make_lcb_code_data.py)"
    ENGINE_ARGS+=(--input-file "$INPUT_FILE" --custom-dataset-type "${CUSTOM_DATASET_TYPE:-single_turn}")
  else
    ENGINE_ARGS+=(--prompt-corpus "${PROMPT_CORPUS:-sonnet}")
  fi
  # min_tokens is OSL-dependent, so workload_args adds it - but only where the
  # engine actually has the field.
  local ei
  for ei in ${ENGINE_EXTRA_INPUTS[@]+"${ENGINE_EXTRA_INPUTS[@]}"}; do
    ENGINE_ARGS+=(--extra-inputs "$ei")
  done
  if [[ -n "${RESET_PATH:-}" ]]; then
    ENGINE_ARGS+=(--reset-kv-cache --reset-kv-cache-path "$RESET_PATH")
  fi
  local dataset_desc="corpus=${PROMPT_CORPUS:-sonnet}"
  [[ -n "${INPUT_FILE:-}" ]] && dataset_desc="dataset=$(basename "$INPUT_FILE") type=${CUSTOM_DATASET_TYPE:-single_turn}"
  ENGINE_ARGS_STR="tokenizer=$(basename "$TOKENIZER") ${dataset_desc} \
cache_bust=${CACHE_BUST:-system_prefix} reset=${RESET_PATH:-none} \
extra=${ENGINE_EXTRA_INPUTS[*]:-none}"
}

# ---------------------------------------------------------------------------
# GPU state helpers
# ---------------------------------------------------------------------------
_gpu_read() {  # -> "used_mib temp_c"
  nvidia-smi --query-gpu=memory.used,temperature.gpu \
    --format=csv,noheader,nounits | head -1 | tr -d ','
}

# gpu_guard — hard abort ceiling. Call before STARTING any measured arm.
# settle_gpu()/gpu_cool() wait for a target; this refuses outright. Without it
# a settle that times out silently proceeds on a hot card.
gpu_guard() {
  local used temp; read -r used temp <<<"$(_gpu_read)"
  if (( temp >= GPU_TEMP_ABORT )); then
    die "GPU at ${temp}C >= abort ceiling ${GPU_TEMP_ABORT}C - refusing to start. \
Cool the card and retry (house policy, see FINDINGS F4a)."
  fi
  echo "  gpu_guard OK: ${temp}C < ${GPU_TEMP_ABORT}C, ${used} MiB"
}

# settle_gpu — full settle, for use with the engine DOWN. Waits for VRAM to be
# released as well as temperature, because a run started on a still-draining
# GPU measures the previous run's thermal and allocator tail.
settle_gpu() {
  local start floor cap used temp el
  start=$(date +%s); floor="$SETTLE_FLOOR"; cap="$SETTLE_CAP"
  log "settling GPU (VRAM < ${SETTLE_VRAM_MIB} MiB, temp <= ${SETTLE_TEMP}C, floor ${floor}s)"
  while :; do
    read -r used temp <<<"$(_gpu_read)"
    el=$(( $(date +%s) - start ))
    if (( el >= floor )) && (( used < SETTLE_VRAM_MIB )) && (( temp <= SETTLE_TEMP )); then
      echo "settled at ${el}s: ${used} MiB, ${temp}C"; break
    fi
    if (( el >= cap )); then
      warn "settle cap ${cap}s reached at ${used} MiB / ${temp}C - continuing"; break
    fi
    sleep 5
  done
  # Settling is best-effort; the abort ceiling is not.
  gpu_guard
}

# gpu_cool — between-cell cooldown with the server STILL BOOTED. Deliberately
# has no VRAM condition: the model is resident, so VRAM will never drop and a
# settle_gpu() here would always hit its cap and just waste 300s.
gpu_cool() {
  local start el used temp
  start=$(date +%s)
  log "inter-cell cooldown (temp <= ${COOLDOWN_TEMP_INTER}C, floor ${COOLDOWN_FLOOR}s)"
  while :; do
    read -r used temp <<<"$(_gpu_read)"
    el=$(( $(date +%s) - start ))
    if (( el >= COOLDOWN_FLOOR )) && (( temp <= COOLDOWN_TEMP_INTER )); then
      echo "cooled at ${el}s: ${temp}C"; break
    fi
    if (( el >= COOLDOWN_CAP )); then
      warn "cooldown cap ${COOLDOWN_CAP}s reached at ${temp}C - continuing"; break
    fi
    sleep 5
  done
  gpu_guard
}

# ---------------------------------------------------------------------------
# Background telemetry sampler: GPU + HOST, one CSV per script run.
#
# Redundant with, but far cheaper and more skimmable than, the per-cell DCGM
# telemetry inside each aiperf artifact - and it covers two things that
# telemetry path CANNOT provide on this hardware:
#
#   1. Throttle ATTRIBUTION. Every DCGM_FI_PROF_* field is unavailable on this
#      card (FINDINGS F4a), so clocks_event_reasons.* from nvidia-smi is how we
#      tell a power-capped card from a thermally-throttled one. A card sitting
#      at 600W losing clock to its power cap is NOT thermal throttling.
#   2. HOST load. DCGM is a GPU instrument. At high concurrency the aiperf
#      client does its own tokenization, dispatch and stream parsing; a client
#      that saturates a core inflates measured TTFT/ITL in a way that is
#      indistinguishable from a slow engine. cpu_pct_per_core_max matters more
#      than the average - one pinned core vanishes into a 32-core mean.
# ---------------------------------------------------------------------------
thermal_start() {
  local tag="${1:-run}"
  THERMAL_CSV="$REPO/artifacts/thermal/${tag}_${STAMP}.csv"
  mkdir -p "$(dirname "$THERMAL_CSV")"
  {
    echo "timestamp,temp_c,tlimit_margin_c,power_w,power_limit_w,sm_clock_mhz,mem_clock_mhz,gpu_util_pct,mem_util_pct,mem_used_mib,clocks_event_reasons,hw_thermal_slowdown,sw_power_cap,cpu_pct_total,cpu_pct_per_core_max,load_1m,mem_used_host_mib,mem_available_mib"
  } > "$THERMAL_CSV"

  (
    # Per-core deltas need a previous sample; seed it before the loop.
    mapfile -t prev < <(grep '^cpu[0-9]' /proc/stat)
    while :; do
      gpu=$(nvidia-smi --query-gpu=temperature.gpu,temperature.gpu.tlimit,power.draw,enforced.power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,clocks_event_reasons.active,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap \
             --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
      mapfile -t cur < <(grep '^cpu[0-9]' /proc/stat)
      max_core=0; tot_busy=0; tot_all=0
      for i in "${!cur[@]}"; do
        read -ra c <<<"${cur[$i]}"; read -ra p <<<"${prev[$i]:-${cur[$i]}}"
        idle=$(( (c[4]+c[5]) - (p[4]+p[5]) ))
        all=0; for f in {1..8}; do all=$(( all + c[f] - p[f] )); done
        (( all <= 0 )) && continue
        busy=$(( all - idle )); pct=$(( 100 * busy / all ))
        (( pct > max_core )) && max_core=$pct
        tot_busy=$(( tot_busy + busy )); tot_all=$(( tot_all + all ))
      done
      prev=("${cur[@]}")
      cpu_tot=0; (( tot_all > 0 )) && cpu_tot=$(( 100 * tot_busy / tot_all ))
      load1=$(awk '{print $1}' /proc/loadavg)
      read -r mtot mavail < <(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{print t, a}' /proc/meminfo)
      printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gpu" "$cpu_tot" "$max_core" "$load1" \
        "$(( (mtot-mavail)/1024 ))" "$(( mavail/1024 ))" >> "$THERMAL_CSV"
      sleep "$THERMAL_INTERVAL"
    done
  ) 9>&- &
  THERMAL_PID=$!
  # Never leave a sampler behind, even on a failed or interrupted run.
  trap 'thermal_stop' EXIT INT TERM
  echo "  thermal+host sampler -> ${THERMAL_CSV#"$REPO"/} (pid $THERMAL_PID, ${THERMAL_INTERVAL}s)"
}

thermal_stop() {
  [[ -n "${THERMAL_PID:-}" ]] || return 0
  kill "$THERMAL_PID" 2>/dev/null || true
  wait "$THERMAL_PID" 2>/dev/null || true
  THERMAL_PID=""
}

# ---------------------------------------------------------------------------
# ACTIVE thermal watchdog. This is the DFlash discipline: do not merely RECORD
# an overheat after the fact, STOP the work and let the card rest.
#
# The existing controls are all boundary checks - gpu_guard refuses to START an
# arm, gpu_cool waits BETWEEN cells, thermal_verdict judges AFTERWARDS. None of
# them can act during a long cell. A 3-hour unattended run needs something that
# intervenes mid-cell.
#
# On breach it kills the in-flight aiperf and drops a flag file. The run loops
# treat a thermally-killed cell as INVALID (never partial-credit, never resumed),
# rest the card with gpu_rest, and retry the cell once from scratch.
# ---------------------------------------------------------------------------
# GPU_TEMP_ABORT was raised 87 -> 90 on 2026-08-25. Evidence: across every run so
# far the card has NEVER reported hw_thermal_slowdown, it sits pinned at its 600W
# power cap, and its own reported limit is ~93C. 87 destroyed a valid 64k prefill
# rung. 90 keeps a 3C margin and matches the DFlash reference baseline, which
# treats a 90C peak as normal healthy operation.
#
# Degrees remaining to the CARD'S OWN limit before we intervene. Hardware-relative,
# so it stays correct if the card's limit differs from our assumption.
: "${GPU_MARGIN_MIN:=3}"
# Consecutive breaching samples required. At 10s interval, 3 = 30s sustained.
: "${GPU_BREACH_SAMPLES:=3}"
: "${GPU_TEMP_RESUME:=60}"      # rest until at or below this before retrying
: "${WATCHDOG_INTERVAL:=10}"
: "${THERMAL_RETRIES:=1}"       # retries per cell after a thermal kill

thermal_watchdog_start() {
  THERMAL_FLAG="${1:?thermal_watchdog_start <outdir>}/.thermal_abort"
  rm -f "$THERMAL_FLAG"
  (
    set +e            # a watchdog must never die from a transient non-zero
    breach=0
    while :; do
      # Three signals per tick, not one:
      #   temp     - absolute reading
      #   margin   - degrees remaining to the CARD'S OWN limit (hardware truth,
      #              not our guess). This card reports T.Limit margins.
      #   slowdown - whether the hardware is ACTUALLY throttling right now.
      read -r temp margin slowdown <<<"$(nvidia-smi \
        --query-gpu=temperature.gpu,temperature.gpu.tlimit,clocks_event_reasons.hw_thermal_slowdown \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ',')"
      [[ -z "$temp" ]] && { sleep "$WATCHDOG_INTERVAL"; continue; }

      hot=0
      # (a) hardware says it is thermally slowing down - believe it immediately
      [[ "$slowdown" == "Active" ]] && hot=1
      # (b) too close to the card's own limit
      [[ -n "$margin" ]] && (( margin <= GPU_MARGIN_MIN )) && hot=1
      # (c) absolute ceiling as a backstop
      (( temp >= GPU_TEMP_ABORT )) && hot=1

      if (( hot )); then breach=$(( breach + 1 )); else breach=0; fi

      # SUSTAINED, not instantaneous. A single spiky sample is not "too hot" -
      # thermal damage is about sustained heat. Requiring N consecutive breaches
      # means a transient touch of the ceiling is ignored, while a genuine climb
      # is caught within N*WATCHDOG_INTERVAL seconds.
      if (( breach >= GPU_BREACH_SAMPLES )); then
        printf '%s temp=%sC margin=%sC slowdown=%s - %d consecutive breaches, killing aiperf\n' \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$temp" "$margin" "$slowdown" "$breach" >> "$THERMAL_FLAG"
        pgrep -f '^aiperf ' | while read -r ap; do kill "$ap" 2>/dev/null; done || true
        sleep 5
        pgrep -f '^aiperf ' | while read -r ap; do kill -9 "$ap" 2>/dev/null; done || true
        breach=0
        sleep 30
      fi
      sleep "$WATCHDOG_INTERVAL"
    done
  ) 9>&- &
  WATCHDOG_PID=$!
  trap 'thermal_watchdog_stop' EXIT INT TERM
  echo "  thermal watchdog armed (abort ${GPU_TEMP_ABORT}C OR margin<=${GPU_MARGIN_MIN}C OR hw-slowdown, \
sustained ${GPU_BREACH_SAMPLES}x${WATCHDOG_INTERVAL}s, pid $WATCHDOG_PID)"
}

thermal_watchdog_stop() {
  [[ -n "${WATCHDOG_PID:-}" ]] || return 0
  kill "$WATCHDOG_PID" 2>/dev/null || true
  wait "$WATCHDOG_PID" 2>/dev/null || true
  WATCHDOG_PID=""
}

# thermal_aborted — did the watchdog fire since the last clear?
thermal_aborted() { [[ -f "${THERMAL_FLAG:-/nonexistent}" ]]; }
thermal_clear()   { rm -f "${THERMAL_FLAG:-/nonexistent}"; }

# gpu_rest — deep cooldown after a thermal kill. Unlike gpu_cool this is not a
# short inter-cell breather: it waits for a genuinely cool card and is willing to
# wait a long time, because the alternative is thrashing the same hot cell.
gpu_rest() {
  local start el used temp
  start=$(date +%s)
  log "THERMAL REST: waiting for GPU <= ${GPU_TEMP_RESUME}C before retrying"
  while :; do
    read -r used temp <<<"$(_gpu_read)"
    el=$(( $(date +%s) - start ))
    if (( temp <= GPU_TEMP_RESUME )); then
      echo "  rested ${el}s -> ${temp}C"; break
    fi
    if (( el >= ${GPU_REST_CAP:-1800} )); then
      warn "rest cap reached at ${temp}C after ${el}s - continuing anyway"; break
    fi
    sleep 15
  done
}

# thermal_verdict — read back the trace and report whether the run is valid.
# A threshold crossing INVALIDATES the cell: stop, keep the artifact, cool, and
# rerun the whole cell. Never resume a partially-hot cell.
thermal_verdict() {
  [[ -f "${THERMAL_CSV:-/nonexistent}" ]] || { warn "no thermal trace to check"; return 0; }
  python3 - "$THERMAL_CSV" "$GPU_TEMP_ABORT" "$HOST_CPU_SATURATION_PCT" <<'PY'
import csv, sys
path, abort_c, cpu_max = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = list(csv.DictReader(open(path)))
if not rows:
    print("  thermal: no samples"); sys.exit(0)
def col(n):
    out = []
    for r in rows:
        try: out.append(float(r[n]))
        except (TypeError, ValueError): pass
    return out
t, p, core = col("temp_c"), col("power_w"), col("cpu_pct_per_core_max")
throttled = [r for r in rows if r.get("hw_thermal_slowdown", "").strip() == "Active"]
capped    = [r for r in rows if r.get("sw_power_cap", "").strip() == "Active"]
print(f"  thermal: {len(rows)} samples, peak {max(t):.0f}C mean {sum(t)/len(t):.1f}C, "
      f"peak {max(p):.0f}W")
print(f"  throttle: hw_thermal Active in {len(throttled)}/{len(rows)} samples, "
      f"sw_power_cap in {len(capped)}/{len(rows)}")
bad = []
if max(t) >= abort_c:
    bad.append(f"peak {max(t):.0f}C >= abort ceiling {abort_c:.0f}C - CELL INVALID, rerun it")
if core and max(core) >= cpu_max:
    bad.append(f"client CPU peaked at {max(core):.0f}% on a core (>= {cpu_max:.0f}%) - "
               "latency numbers are suspect, rerun with the client constrained")
for b in bad: print(f"  INVALID: {b}")
sys.exit(1 if bad else 0)
PY
}

# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------
# BENCH_SKIP_BOOT=1 points the harness at an ALREADY-RUNNING server instead of
# booting compose. This exists for the mock-server verification path, which
# exercises discover.sh/validate.sh logic (SLA mapping, boundary confirmation,
# percentage math, failure paths) at zero GPU cost against an assertable
# capacity knee. It must NEVER be set for a real measured run - provenance would
# describe an engine this script did not start.
engine_env_map() {
  # THE COMPOSE FILES SHARE NO VARIABLE VOCABULARY, so the workload's neutral
  # knobs have to be translated per engine. RULES.md 4 is the warning here:
  # --max-num-batched-tokens vs --chunked-prefill-size "looked equivalent, they
  # were not, and it silently gave one engine 2x the prefill budget". Treat these
  # as ANALOGOUS, and say so in the write-up.
  ENV_MAP=()
  case "$ENGINE" in
    llamacpp)
      # MODEL is REQUIRED by that compose (${MODEL:?...}) and is a host path the
      # container sees under /hf.
      ENV_MAP=(
        "MODEL=$(resolve_gguf_model)"
        "CTX=${ENGINE_CTX:-32768}"
        "PARALLEL=${MAX_RUNNING_REQUESTS:-1}"
        "BATCH=${PREFILL_BUDGET:-2048}"
        "UBATCH=${UBATCH:-512}"
        # The published-baseline placement, so this arm reproduces FINDINGS.md
        # rather than inventing a new configuration.
        "OT=${OT:-per_layer_token_embd=CPU}"
        "LOAD_MODE=${LOAD_MODE:-none}"
        "LAZY=${LAZY:-off}"
        "N_CPU_MOE=${N_CPU_MOE:-0}"
        "SPEC_TYPE=${SPEC_TYPE:-none}"
        # Sidecar draft head for SPEC_TYPE=draft-mtp. Empty on every other arm,
        # which llama.cpp reads as "no draft model"; the boot assertion in
        # spec_active_assert() is what proves that, rather than trusting it.
        "SPEC_DRAFT_MODEL=${SPEC_DRAFT_MODEL:-}"
        "SPEC_DRAFT_N_MAX=${SPEC_DRAFT_N_MAX:-3}"
        "SPEC_DRAFT_NGL=${SPEC_DRAFT_NGL:-99}"
        "LLAMA_HOST_PORT=${ENGINE_HOST_PORT}"
        "CONTAINER_NAME=${CONTAINER}"
      )
      ;;
    sglang)
      # PER-REQUEST CONTEXT ENTITLEMENT, matched to llama.cpp. RULES.md 4 and
      # Auto_Bench.md: concurrency arms must give each engine the SAME per-slot
      # context, or the ladder compares two different experiments.
      #
      # llama.cpp splits its context across slots: ENGINE_CTX / PARALLEL.
      # SGLang sizes ADMISSION from the pool: it will not admit more requests
      # than max_total_num_tokens / context_len. Setting both to ENGINE_CTX -
      # which is what this map used to do - means the pool holds exactly ONE
      # full context, so SGLang silently resolved max_running_requests to 1:
      #
      #   server_args:  max_running_requests=8       (what we asked)
      #   resolved:     max_running_requests=1       (what it used)
      #   #running-req: 1, 715 samples, never higher
      #
      # It then served the c=2/4/8 cells SERIALLY and the extra load showed up
      # as queue time: per-user throughput flat at ~128-132 tok/s while TTFT
      # climbed 280ms -> 14.9s. Nothing errored and OSL was exact, so the
      # post-run gate passed a run that measured queueing, not concurrency.
      #
      # So: pool = ENGINE_CTX, per-request context = ENGINE_CTX / slots. That is
      # the same entitlement llama.cpp gives each of its slots.
      ENV_MAP=(
        "SGLANG_CONTEXT_LENGTH=$(( ${ENGINE_CTX:-131072} / ${MAX_RUNNING_REQUESTS:-1} ))"
        "SGLANG_MAX_TOTAL_TOKENS=${ENGINE_CTX:-131072}"
        "SGLANG_MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-1}"
        # THE MAMBA STATE CACHE IS WHAT CAPS CONCURRENCY ON THIS MODEL, not KV.
        # Qwen3.8-Flash-Next is hybrid (QSA + linear/mamba state), and SGLang's
        # NEXTN path resolves to EAGLE, which needs FIVE state slots per request
        # (1 + speculative_num_draft_tokens 4). The engine says so outright:
        #
        #   max_running_requests is capped to 1 by the mamba state cache
        #   (max_mamba_cache_size=5, 5 state slots per request)
        #
        # So the validated single-stream value of 5 silently forces admission 1
        # no matter what --max-running-requests asks for. Scale it with the
        # ladder: slots x 5. Speculation and concurrency compete for the SAME
        # resource here, which is itself a finding worth reporting rather than
        # configuring around.
        # 5 slots per request is `mamba_ratio`, a MODEL constant, not a speculation
        # cost. I originally scaled this to 1 when speculation was off; the engine
        # refused to boot:
        #   RuntimeError: Hybrid (mamba/linear-attention) state cache is too small
        #   to serve any requests. max_mamba_cache_size=1, mamba_ratio=5
        # So concurrency costs 5 state slots per request whether NEXTN is on or off,
        # and my earlier claim that "speculation and concurrency compete for the same
        # resource" was wrong about the mechanism - the state cost is architectural.
        "SGLANG_MAX_MAMBA_CACHE_SIZE=${SGLANG_MAX_MAMBA_CACHE_SIZE:-$(( ${MAX_RUNNING_REQUESTS:-1} * 5 ))}"
        "SGLANG_CHUNKED_PREFILL_SIZE=${PREFILL_BUDGET:-8192}"
        "SGLANG_CUDA_GRAPH_MAX_BS_DECODE=${CUDA_GRAPH_MAX_BS:-1}"
        "SGLANG_HOST_PORT=${ENGINE_HOST_PORT}"

        # MEM FRACTION WAS NEVER IN THIS MAP, and record_provenance defaulted
        # its parity field to 0.90 while the compose defaults to 0.93. Every
        # spec-on artifact therefore RECORDS 0.90 and RAN 0.93 - checked against
        # quality_sglang_20260904T160940Z, whose server_boot_args says 0.93.
        # Naming it here, at the compose's own default, makes the two agree by
        # construction instead of by coincidence.
        "SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.93}"

        # OPTIONAL LAUNCH KNOBS. EMPTY IS THE CONTROL: an empty value emits NO
        # argument at all (bench/sglang_render_compose.py), so an arm with none
        # of these set reproduces the argv every existing artifact was measured
        # with - verified byte-for-byte against the base command list.
        #
        # They are listed here EVEN WHEN EMPTY on purpose. `env "${ENV_MAP[@]}"`
        # only ADDS to the inherited environment; it does not clear it. Without
        # these lines a stray `export SGLANG_MAMBA_SSM_DTYPE=...` in the
        # operator's shell would reach the renderer and switch a knob on with
        # nothing in the evidence saying so.
        #
        # --gdn-mtp-cache-mode is deliberately NOT here: bench/sglang_flagcap.sh
        # proved this image's argparse does not have it (it is a source-level
        # feature of jpezzulli/sglang-rtxpro6000). Offering it would only
        # produce boot failures.
        #
        # COUPLED PAIR, not two knobs: on SM100+ the engine raises ValueError
        # for --linear-attn-decode-backend flashinfer unless the ssm dtype is
        # bfloat16, AND setting the ssm dtype to bfloat16 alone auto-promotes
        # decode (and verify) to flashinfer. See
        # results/gates/sglang_flagcap_*/FINDING.md.
        "SGLANG_LINEAR_ATTN_DECODE_BACKEND=${SGLANG_LINEAR_ATTN_DECODE_BACKEND:-}"
        "SGLANG_LINEAR_ATTN_PREFILL_BACKEND=${SGLANG_LINEAR_ATTN_PREFILL_BACKEND:-}"
        # NOT a free knob: it sets the dtype of the mamba TEMPORAL state, so it
        # changes numerics and the per-slot state footprint that caps admission.
        # Compare max_total_num_tokens, not only tok/s.
        "SGLANG_MAMBA_SSM_DTYPE=${SGLANG_MAMBA_SSM_DTYPE:-}"
        # Unset, the draft head INHERITS --quantization (modelopt_fp4).
        "SGLANG_SPEC_DRAFT_QUANT=${SGLANG_SPEC_DRAFT_QUANT:-}"
        # Empty/0 = off. Never enable inside a published ladder: this harness
        # idles the server 45-240s between cells and settle_gpu asserts on
        # freed VRAM, so a sleeping server puts wake latency into TTFT.
        "SGLANG_SLEEP_ON_IDLE=${SGLANG_SLEEP_ON_IDLE:-}"
        "SGLANG_DISABLE_FI_AUTOTUNE=${SGLANG_DISABLE_FI_AUTOTUNE:-}"

        # CONTAINER-SIDE ENVIRONMENT. Prefixed selectors, so setting one does
        # not also change the harness's own process environment. These can never
        # appear in server_boot_args (docker .Args is argv only) - the same
        # blind spot that made LAZY unattributable in B-25 - so each one gets a
        # named parity field in record_provenance.
        "SGLANG_PYTORCH_ALLOC_CONF=${SGLANG_PYTORCH_ALLOC_CONF:-}"
        "SGLANG_MAMBA_CONV_DTYPE_VAL=${SGLANG_MAMBA_CONV_DTYPE_VAL:-}"
        "SGLANG_NUMA_BIND_V2_VAL=${SGLANG_NUMA_BIND_V2_VAL:-}"
        "SGLANG_OMP_NUM_THREADS=${SGLANG_OMP_NUM_THREADS:-}"
        # One switch for the persistent-JIT group. OFF by default: a warm JIT
        # cache changes BOOT TIME, which bench/boot_time.sh publishes.
        "SGLANG_JIT_CACHE_HOST_DIR=${SGLANG_JIT_CACHE_HOST_DIR:-}"
      )

      # Render the run-scoped override from the base compose. Done HERE because
      # engine_env_map runs at the top of both engine_up and engine_down, so
      # teardown cannot end up using a different file than boot did.
      #
      # The generated file is EVIDENCE, not a temp file: record_provenance
      # copies it into the artifact. That finally gives the SGLang arm the
      # resolved-launch-config record Auto_Bench.md 5.1 requires and that only
      # the llama.cpp arm has had.
      SGLANG_GEN_COMPOSE="$REPO/artifacts/_compose/sglang_${STAMP}.gen.yaml"
      env "${ENV_MAP[@]}" SPEC="${SPEC:-on}" \
        python3 "$BENCH_DIR/sglang_render_compose.py" \
          "$COMPOSE_DIR/$COMPOSE_FILE" "$SGLANG_GEN_COMPOSE" \
        || die "sglang compose render failed - refusing to boot an unspecified server"
      # Absolute path, and ALWAYS second: engine_up cd's to $COMPOSE_DIR and the
      # project directory comes from the FIRST -f, which is what makes
      # `security_opt: seccomp=./sglang/seccomp-io_uring.json` resolve.
      COMPOSE_OVERRIDE="$SGLANG_GEN_COMPOSE"
      ;;
    freetoken)
      ENV_MAP=(
        "ENGINE_CTX=${ENGINE_CTX:-32768}"
        "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-1}"
        "PREFILL_BUDGET=${PREFILL_BUDGET:-8192}"
        "CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-1}"
        "FREETOKEN_HOST_PORT=${ENGINE_HOST_PORT}"
      )
      ;;
  esac
}

# This repo keeps ONE COMPOSE FILE PER ARM rather than one file with N services
# (the three engines share no flags at all), so the file and the service name
# are both per-engine. All three arms are containers, which means all three sit
# behind the same cgroup memory guard - on a 91 GiB box that must not vary by
# arm.
# engine_down MUST pass the same env as engine_up. MEASURED: the llama.cpp
# compose declares `${MODEL:?...}`, so a bare `docker compose down` dies with
#   error while interpolating services.q38n.command.[]:
#   required variable MODEL is missing a value
# and - because the failure was swallowed by `|| true` - the container was left
# RUNNING while the script reported "tearing down". It then held 62 GiB of VRAM
# into the next arm, which is exactly how one engine silently corrupts another.
engine_down() {
  [[ "${BENCH_SKIP_BOOT:-0}" == "1" ]] && { echo "  (skip-boot: leaving server up)"; return 0; }
  [[ -n "${COMPOSE_FILE:-}" ]] || return 0
  engine_env_map
  ( cd "$COMPOSE_DIR" && env "${ENV_MAP[@]}" \
      docker compose -f "$COMPOSE_FILE" ${COMPOSE_OVERRIDE:+-f "$COMPOSE_OVERRIDE"} \
      down >/dev/null 2>&1 ) || true
  # Belt and braces: if interpolation still fails for any reason, the container
  # must not survive into the next arm.
  [[ -n "${CONTAINER:-}" ]] && docker rm -f "$CONTAINER" >/dev/null 2>&1
  return 0
}

engine_up() {
  if [[ "${BENCH_SKIP_BOOT:-0}" == "1" ]]; then
    warn "BENCH_SKIP_BOOT=1 - using the already-running server at $BASE. NOT a valid measured run."
    return 0
  fi
  log "booting $ENGINE"
  engine_down
  engine_env_map

  # Never discard the launcher's stderr (Auto_Bench.md 4): a hidden "unknown
  # option" once burned a whole stage that reported only "server failed to
  # start". --wait already fails on an unhealthy container, but the reason for
  # the failure only exists in the output.
  ( cd "$COMPOSE_DIR" && env "${ENV_MAP[@]}" \
      docker compose -f "$COMPOSE_FILE" ${COMPOSE_OVERRIDE:+-f "$COMPOSE_OVERRIDE"} \
      up -d --wait "$SERVICE" ) \
    || { warn "$ENGINE failed to become healthy - last 40 lines:"
         # Into the artifact as well as stderr: a boot failure is the evidence
         # for a capacity finding, and compose may remove the container before
         # anything else can read it (measured 2026-09-07, t8_gpu, which left an
         # EMPTY artifact dir because capture_failure found nothing to inspect).
         [[ -n "${OUT:-}" && -d "${OUT:-}" ]] && \
           docker logs --tail 5000 "$CONTAINER" > "$OUT/server.log.failure" 2>&1 || true
         # `docker logs`, NOT `docker compose logs`: compose would re-interpolate
         # the file without ENV_MAP and report a missing MODEL that IS set,
         # burying the real error under a fake one.
         docker logs --tail 40 "$CONTAINER" 2>&1 | sed 's/^/    /' >&2 \
           || ( cd "$COMPOSE_DIR" && env "${ENV_MAP[@]}" \
                docker compose -f "$COMPOSE_FILE" logs --tail 40 "$SERVICE" 2>&1 ) >&2
         die "$ENGINE failed to start"; }
}

# ---------------------------------------------------------------------------
# resolve_gguf_model - absolute path of shard 1 for the llama.cpp arm, as the
# CONTAINER sees it. Echoes the path; dies if the download is incomplete.
#
# Same resolution scripts/serve.sh does, and for the same reason: refs/main can
# point at a commit whose snapshot does not contain the quant we want (this repo
# holds UD-IQ4_XS under 824f539b while refs/main points at 83cadfda), so
# "newest commit" is the wrong rule. Prefer refs/main only IF it has the quant,
# else take the snapshot holding the most shards of it.
# ---------------------------------------------------------------------------
resolve_gguf_model() {
  local repo="${GGUF_REPO:-unsloth/Qwen3.8-Flash-Next-GGUF}" quant="${MODEL_QUANT:-UD-IQ4_XS}"
  local home="${HF_HOME:-$HOME/.cache/huggingface}"
  local cache="$home/hub/models--${repo//\//--}" snap=""
  [[ -d "$cache" ]] || die "not downloaded: $repo"
  if [[ -f "$cache/refs/main" ]]; then
    local ref; ref="$(cat "$cache/refs/main")"
    [[ -d "$cache/snapshots/$ref/$quant" ]] && snap="$cache/snapshots/$ref"
  fi
  if [[ -z "$snap" ]]; then
    snap="$(find "$cache/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
      | while read -r d; do echo "$(ls "$d/$quant"/*.gguf 2>/dev/null | wc -l) $d"; done \
      | sort -rn | head -1 | cut -d" " -f2-)"
  fi
  [[ -n "$snap" && -d "$snap/$quant" ]] || die "quant '$quant' not in the cache for $repo"
  local shard1; shard1="$(find "$snap/$quant" -name '*-00001-of-*.gguf' | sort | head -1)"
  [[ -n "$shard1" ]] || die "no shard 1 under $snap/$quant - download incomplete"
  # every shard named by the -of-NNNNN suffix must be present
  local total; total="$(sed -E 's/.*-of-0*([0-9]+)\.gguf/\1/' <<<"$shard1")"
  local have; have="$(ls "$snap/$quant"/*-of-*.gguf 2>/dev/null | wc -l)"
  [[ "$have" -ge "$total" ]] || die "incomplete: $have of $total shards for $quant"
  printf '/hf%s' "${shard1#"$home"}"
}

# ---------------------------------------------------------------------------
# aiperf_wd - a CLEAN working directory to run aiperf from. Echoes the path.
#
# MEASURED 2026-09-02: aiperf and its mock server use pydantic-settings, which
# auto-loads a credential file from the CURRENT WORKING DIRECTORY, rejects
# unknown keys, and prints the rejected value in the resulting traceback. Proof,
# with no need to open the file:
#
#   aiperf-mock-server --help   from the repo root   -> ValidationError
#   aiperf-mock-server --help   from a clean dir     -> works
#
# The upstream harness solved this by keeping ITS repo root free of credentials
# (its own file lives in docker/). THIS repo root is not free of one, and it is
# not ours to move - scripts/serve.sh and download_models.sh source it for a
# token. So rather than requiring a clean repo root, run aiperf from a clean
# directory. Every path aiperf is given is absolute, so nothing else cares.
#
# This is a leak-prevention control as much as a bug fix: the failure mode is a
# secret printed into a traceback, and evidence directories get published.
# ---------------------------------------------------------------------------
aiperf_wd() {
  local wd="${AIPERF_WD:-${TMPDIR:-/tmp}/aiperf-wd-$$}"
  mkdir -p "$wd"
  printf '%s' "$wd"
}

# ---------------------------------------------------------------------------
# incident <outdir> <kind> <text> - the run's own incident log.
#
# The audit lists "crashes, workarounds and required patches" as a deliverable,
# and FINDINGS.md already carries a "What would invalidate a run" section. This
# makes that a recorded artifact instead of something reconstructed afterwards
# from memory and scrollback.
#
# One line per crash, workaround, patch, retry or manual intervention, with the
# UTC timestamp and the evidence that showed it. Nothing gets fixed silently:
# an arm that needed a workaround to produce a number is not the same result as
# one that did not, and the reader is entitled to know which they are looking at.
#
# kind is free text, but keep it to a small vocabulary so the logs stay greppable:
#   crash | oom | throttle | retry | workaround | patch | gate-fail | note
# ---------------------------------------------------------------------------
incident() {
  local out="${1:?incident <outdir> <kind> <text>}" kind="${2:?}" ; shift 2
  local text="$*"
  mkdir -p "$out"
  local f="$out/incidents.md"
  if [[ ! -f "$f" ]]; then
    {
      echo "# Incidents - ${WORKLOAD:-?} / ${ENGINE:-?} / ${STAMP:-?}"
      echo
      echo "One line per crash, workaround, patch, retry or manual intervention."
      echo "An empty file means the run needed none - that is itself a result."
      echo
      echo "| utc | kind | detail |"
      echo "|---|---|---|"
    } > "$f"
  fi
  printf '| %s | %s | %s |\n' "$(date -u +%FT%TZ)" "$kind" "${text//|/\\|}" >> "$f"
  warn "incident [$kind] $text"
}

# ---------------------------------------------------------------------------
# record_runtime_state <outdir> - restarts, OOM kills and peak memory.
#
# Auto_Bench.md 5/7 and the operator's standing instruction: watch restarts and
# failed calls, not just throughput. Three failures this captures, all of which
# have actually happened on this box:
#
#   1. A RESTART LOOP. NOTES.md records `restart: unless-stopped` turning an OOM
#      into a restart loop, which cost a whole tier-ladder run and read as a
#      hang rather than a failure. Every service here is `restart: "no"`, so a
#      non-zero RestartCount means something restarted it anyway - void the arm.
#   2. A CONTAINED OOM. `OOMKilled=true` with the host alive is a CAPACITY
#      FINDING, recorded and reported, never retried until it passes.
#   3. A GLOBAL OOM. The host killer reports OOMKilled=FALSE on the container,
#      so cgroup memory.events is the only place the truth survives. The SGLang
#      arm peaked at 83.69 of 86 GiB; there is very little room.
#
# Works for both shapes: a docker container, or the systemd scope FreeToken
# runs under, so memory.json has ONE schema across all three arms.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# verify_mem_cap - prove the host-RAM cap was APPLIED, not merely requested.
#
# memory.json's cap_bytes is just an echo of MEM_CAP_BYTES; it says what we
# wanted, not what the kernel enforced. A typo in a compose override, or an
# override file that was never passed, produces an uncapped run that looks
# capped in its own evidence - which is exactly the failure mode this whole
# experiment must not have. So read it back from BOTH sides and refuse to
# continue on a mismatch: docker's own record of the limit, and the live cgroup.
#
# Echoes "intended docker_mem docker_swap cgroup_max cgroup_swap_max"; returns
# non-zero if docker or the cgroup disagrees with the intent.
# ---------------------------------------------------------------------------
verify_mem_cap() {
  local want="${MEM_CAP_BYTES:-}"
  [[ -z "$want" ]] && { echo "  mem cap          not requested"; return 0; }
  docker inspect "$CONTAINER" >/dev/null 2>&1 || { warn "verify_mem_cap: no container"; return 1; }
  local dmem dswap id cg cmax cswap
  dmem=$(docker inspect -f '{{.HostConfig.Memory}}' "$CONTAINER" 2>/dev/null || echo 0)
  dswap=$(docker inspect -f '{{.HostConfig.MemorySwap}}' "$CONTAINER" 2>/dev/null || echo 0)
  id=$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null || true)
  for c in "/sys/fs/cgroup/system.slice/docker-${id}.scope" "/sys/fs/cgroup/docker/${id}"; do
    [[ -d "$c" ]] && { cg="$c"; break; }
  done
  cmax=$(cat "$cg/memory.max" 2>/dev/null || echo unknown)
  cswap=$(cat "$cg/memory.swap.max" 2>/dev/null || echo unknown)
  echo "  mem cap          want=$want docker=$dmem/$dswap cgroup=$cmax swap=$cswap"
  MEM_CAP_READBACK="$want $dmem $dswap $cmax $cswap"
  if [[ "$dmem" != "$want" ]]; then
    warn "MEM CAP NOT APPLIED: asked $want, docker reports $dmem - the compose override did not take"
    return 1
  fi
  # memswap_limit == mem_limit is the idiom the sglang/freetoken arms use: equal
  # values mean the cgroup may not swap at all, so pressure surfaces as an OOM
  # instead of silently becoming disk (FINDINGS.md 7).
  if [[ "$dswap" != "$want" ]]; then
    warn "swap allowance is $dswap, not $want - this cgroup CAN swap, and a slow run may be measuring disk"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# capture_failure - evidence BEFORE teardown.
#
# bench/run.sh boots, then records provenance, and only writes memory.json
# after aiperf succeeds; its EXIT trap then calls engine_down. So a failed boot
# or a non-zero benchmark used to destroy the container before anything had
# looked at it - and a deliberate OOM test whose evidence is deleted on the way
# out is not a test. Called from the cleanup path while the container still
# exists. Safe to call twice; safe to call when there is nothing to capture.
# ---------------------------------------------------------------------------
capture_failure() {
  local out="${1:-${OUT:-}}"
  [[ -z "$out" || ! -d "$out" ]] && return 0
  [[ -f "$out/failure.json" ]] && return 0
  [[ -n "${CONTAINER:-}" ]] || return 0
  if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    # The container is ALREADY GONE - which is exactly what a failed
    # `compose up --wait` leaves behind. Silence here produced an empty artifact
    # directory for the one case this function exists to record, so write what
    # is still knowable instead of returning.
    cat > "$out/failure.json" <<JSON
{
  "container_state": "absent",
  "exit_code": null,
  "oom_killed": null,
  "note": "Container was gone before capture. Typical of a failed boot: compose removes it after --wait fails. See server.log.failure (written by engine_up) for the loader output; a CUDA OOM at load appears there, not in cgroup counters, because the cgroup died with the container.",
  "mem_cap_readback": "${MEM_CAP_READBACK:-not-verified}",
  "attempted_mem_cap": "${MEM_CAP_BYTES:-unset}",
  "host_available_bytes": $(free -b | awk '/^Mem:/{print $7}'),
  "host_swap_used_bytes": $(free -b | awk '/^Swap:/{print $3}'),
  "gpu_used_mib": $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo null)
}
JSON
    printf '%s\n' "${ENV_MAP[@]+"${ENV_MAP[@]}"}" > "$out/attempted_env.txt" 2>/dev/null || true
    warn "failure evidence -> $out/failure.json (container absent)"
    return 0
  fi

  local st ex oom id cg cmax cswap peak events
  st=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo unknown)
  ex=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER" 2>/dev/null || echo null)
  oom=$(docker inspect -f '{{.State.OOMKilled}}' "$CONTAINER" 2>/dev/null || echo null)
  id=$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null || true)
  for c in "/sys/fs/cgroup/system.slice/docker-${id}.scope" "/sys/fs/cgroup/docker/${id}"; do
    [[ -d "$c" ]] && { cg="$c"; break; }
  done
  cmax=$(cat "$cg/memory.max" 2>/dev/null || echo null)
  cswap=$(cat "$cg/memory.swap.max" 2>/dev/null || echo null)
  peak=$(cat "$cg/memory.peak" 2>/dev/null || echo null)
  events="{}"
  [[ -r "$cg/memory.events" ]] && events=$(awk 'BEGIN{printf "{"} {printf "%s\"%s\": %s", (NR>1?", ":""), $1, $2} END{printf "}"}' "$cg/memory.events")

  # 5000, not 200. A failing server's LAST 200 lines are its shutdown cascade;
  # the cause is further up. On 2026-09-09 the S1 arm hung for 600s and died on
  # SGLang's own warmup timeout - the 200-line capture contained the teardown
  # traceback and not one line of its own, so the backend resolution that would
  # have named the loaded kernel was gone with the container. Auto_Bench.md 4:
  # never discard the launcher's stderr.
  docker logs --tail 5000 "$CONTAINER" > "$out/server.log.failure" 2>&1 || true
  printf '%s\n' "${ENV_MAP[@]+"${ENV_MAP[@]}"}" > "$out/attempted_env.txt" 2>/dev/null || true

  cat > "$out/failure.json" <<JSON
{
  "container_state": "$st",
  "exit_code": $ex,
  "oom_killed": $oom,
  "cgroup_memory_max": "$cmax",
  "cgroup_memory_swap_max": "$cswap",
  "cgroup_memory_peak": "$peak",
  "memory_events": $events,
  "mem_cap_readback": "${MEM_CAP_READBACK:-not-verified}",
  "host_available_bytes": $(free -b | awk '/^Mem:/{print $7}'),
  "host_swap_used_bytes": $(free -b | awk '/^Swap:/{print $3}'),
  "note": "Captured before engine_down. oom_killed=true is a capacity finding for THIS placement, not proof that no placement fits. A guard stop (host_mem_guard.sh) is NOT an OOM - it appears here as container_state=exited with oom_killed=false."
}
JSON
  warn "failure evidence -> $out/failure.json (state=$st exit=$ex oom=$oom)"
}

record_runtime_state() {
  local out="${1:?record_runtime_state <outdir>}"
  local cg="" restarts="null" oomkilled="null" status="null"

  if [[ -n "${CONTAINER:-}" ]] && docker inspect "$CONTAINER" >/dev/null 2>&1; then
    restarts=$(docker inspect -f '{{.RestartCount}}' "$CONTAINER" 2>/dev/null || echo null)
    oomkilled=$(docker inspect -f '{{.State.OOMKilled}}' "$CONTAINER" 2>/dev/null || echo null)
    status=\"$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo unknown)\"
    local id; id=$(docker inspect -f '{{.Id}}' "$CONTAINER" 2>/dev/null || true)
    for c in "/sys/fs/cgroup/system.slice/docker-${id}.scope" \
             "/sys/fs/cgroup/docker/${id}"; do
      [[ -d "$c" ]] && { cg="$c"; break; }
    done
  elif [[ -n "${FREETOKEN_SCOPE:-}" ]]; then
    status=\"scope\"
    cg="/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/${FREETOKEN_SCOPE}.scope"
  fi

  local peak="null" events="{}" swap_peak="null"
  if [[ -n "$cg" && -d "$cg" ]]; then
    [[ -r "$cg/memory.peak" ]] && peak=$(cat "$cg/memory.peak" 2>/dev/null || echo null)
    [[ -r "$cg/memory.swap.peak" ]] && swap_peak=$(cat "$cg/memory.swap.peak" 2>/dev/null || echo null)
    if [[ -r "$cg/memory.events" ]]; then
      events=$(awk 'BEGIN{printf "{"} {printf "%s\"%s\": %s", (NR>1?", ":""), $1, $2} END{printf "}"}' "$cg/memory.events")
    fi
  fi

  # Host-side swap movement. FINDINGS.md 7 records a tier where the kernel
  # swapped 7 GiB INSTEAD of failing and the server stayed healthy - every
  # number would have measured swap while looking perfectly normal.
  local swap_used; swap_used=$(free -b | awk '/^Swap:/{print $3}')

  cat > "$out/memory.json" <<JSON
{
  "engine": "$ENGINE",
  "container": "${CONTAINER:-null}",
  "scope": "${FREETOKEN_SCOPE:-null}",
  "state": $status,
  "restart_count": $restarts,
  "oom_killed": $oomkilled,
  "cgroup": "${cg:-null}",
  "memory_peak_bytes": $peak,
  "memory_swap_peak_bytes": $swap_peak,
  "memory_events": $events,
  "host_swap_used_bytes": $swap_used,
  "cap_bytes": ${MEM_CAP_BYTES:-null},
  "cap_readback": "${MEM_CAP_READBACK:-not-verified}",
  "note_cap": "cap_bytes is what was ASKED for; cap_readback is want/docker_mem/docker_swap/cgroup_max/cgroup_swap_max as verify_mem_cap read them back. Trust the readback.",
  "note": "restart_count>0 voids the arm (every service is restart:no). oom_killed=true is a capacity finding. memory.events.oom_kill>0 with oom_killed=false means the HOST killer fired."
}
JSON
  echo "  runtime state    restarts=$restarts oom=$oomkilled peak=$peak -> $out/memory.json"

  if [[ "$restarts" != "null" && "$restarts" != "0" ]]; then
    incident "$out" crash "RestartCount=$restarts - arm is VOID (services are restart:no)"
  fi
  if [[ "$oomkilled" == "true" ]]; then
    incident "$out" oom "OOMKilled=true, cgroup peak $peak of cap ${MEM_CAP_BYTES:-unset} - capacity finding, do NOT retry blindly"
  fi
  local okill; okill=$(sed -n 's/.*oom_kill \([0-9]*\).*/\1/p' <<<"$events" 2>/dev/null || true)
  if [[ -n "$okill" && "$okill" != "0" && "$oomkilled" != "true" ]]; then
    incident "$out" oom "memory.events oom_kill=$okill with OOMKilled=false - the HOST killer fired, not the container cap"
  fi
}

# ---------------------------------------------------------------------------
# Preflight. Every assert here has failed for real at least once; a silent
# version of any of them produces plausible-looking but wrong numbers.
# ---------------------------------------------------------------------------
preflight() {
  log "preflight"
  local served health
  served=$(curl -fsS "${BASE}/v1/models" \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')
  [[ "$served" == "$MODEL" ]] || die "server has '$served', expected '$MODEL'"
  echo "  model            $served"

  # ------------------------------------------------------------------------
  # SGLANG LAUNCH-KNOB ASSERTS (Auto_Bench.md 3: assert the RESOLVED
  # configuration, and prove the feature is ACTIVE, not merely configured).
  #
  # The failure this exists to prevent: the renderer emits nothing for an empty
  # selector, so a typo'd selector name would boot the CONTROL while the queue
  # script, the artifact name and the report all say "arm". That is a wrong
  # number with correct-looking provenance, which is worse than a crash.
  # ------------------------------------------------------------------------
  if [[ "$ENGINE" == "sglang" ]]; then
    local argv resolved
    argv="$(docker inspect --format '{{join .Args " "}}' "$CONTAINER" 2>/dev/null)" \
      || die "cannot read argv for $CONTAINER"
    resolved="$(docker logs "$CONTAINER" 2>&1 | grep -a 'server_args=ServerArgs(' | tail -1)"

    # 0. CAPABILITY. Refuse a flag this image's argparse does not have, using
    #    the newest bench/sglang_flagcap.sh manifest. Absent manifest is not a
    #    pass: run the probe, it costs 30s and no GPU.
    local capfile
    capfile="$(ls -1dt "$REPO"/results/gates/sglang_flagcap_*/flags.tsv 2>/dev/null | head -1)"

    local _sel _name _flag _val _argname _rest
    for _sel in \
      "SGLANG_LINEAR_ATTN_DECODE_BACKEND:--linear-attn-decode-backend:linear_attn_decode_backend" \
      "SGLANG_LINEAR_ATTN_PREFILL_BACKEND:--linear-attn-prefill-backend:linear_attn_prefill_backend" \
      "SGLANG_MAMBA_SSM_DTYPE:--mamba-ssm-dtype:mamba_ssm_dtype" \
      "SGLANG_SPEC_DRAFT_QUANT:--speculative-draft-model-quantization:speculative_draft_model_quantization"
    do
      # triple is  SELECTOR:--flag:serverargs_attr_name
      _name="${_sel%%:*}"; _rest="${_sel#*:}"
      _flag="${_rest%%:*}"; _argname="${_rest#*:}"
      _val="${!_name:-}"
      if [[ -n "$_val" ]]; then
        if [[ -n "$capfile" ]] && ! awk -v f="$_flag" '$1==f && $2 ~ /^yes/' "$capfile" | grep -q .; then
          die "$_flag is not in this image's argparse (see $capfile) - do not guess, run bench/sglang_flagcap.sh"
        fi
        # 1. POSITIVE: the flag and its value must be in the argv that booted.
        grep -q -- "$_flag $_val" <<<"$argv" \
          || die "requested $_flag $_val is NOT in the booted argv - the render was a no-op and this arm is measuring the control"
        # 3. RESOLVED, not merely requested. A flag can be accepted at boot and
        #    then overridden by the engine's own _handle_* resolution.
        if [[ -n "$resolved" ]]; then
          grep -q "${_argname}='${_val}'" <<<"$resolved" \
            || warn "$_flag requested $_val but ServerArgs resolved it differently - see server_args.txt"
        fi
        echo "  sglang knob      $_flag $_val"
      else
        # 2. NEGATIVE: an unset knob must be ABSENT. `env "${ENV_MAP[@]}"` only
        #    ADDS to the inherited environment, so a stray export in the
        #    operator's shell would otherwise switch a knob on invisibly.
        grep -q -- "$_flag " <<<"$argv" \
          && die "$_flag is in the booted argv but no selector requested it - the environment is contaminated"
      fi
    done

    # 4. ACTIVE, not merely configured. The linear-attn backends are the whole
    #    point of this campaign, and 'configured but inert' is exactly what the
    #    MTP-1 gate was written to catch. The engine prints its resolution as
    #      Linear attention kernel backend: decode=X, prefill=Y, verify=Z
    if [[ -n "${SGLANG_LINEAR_ATTN_DECODE_BACKEND:-}${SGLANG_MAMBA_SSM_DTYPE:-}" ]]; then
      local laline
      laline="$(docker logs "$CONTAINER" 2>&1 | grep -a 'Linear attention kernel backend:' | tail -1)"
      [[ -n "$laline" ]] || die "no 'Linear attention kernel backend:' line - cannot prove which kernel is live"
      echo "  ${laline#*] }"
      # bfloat16 ssm dtype auto-promotes decode AND verify to flashinfer - but
      # ONLY when no decode backend was named. An explicit backend suppresses
      # the promotion (the engine checks `linear_attn_decode_backend is None`),
      # which is exactly how the bf16-state capacity arm keeps decode on the
      # Triton kernels that work. So: assert the promotion when it is expected,
      # assert the named backend when one was named.
      if [[ -n "${SGLANG_LINEAR_ATTN_DECODE_BACKEND:-}" ]]; then
        grep -q "decode=${SGLANG_LINEAR_ATTN_DECODE_BACKEND}" <<<"$laline" \
          || die "requested decode=${SGLANG_LINEAR_ATTN_DECODE_BACKEND} but engine resolved: $laline"
      elif [[ "${SGLANG_MAMBA_SSM_DTYPE:-}" == "bfloat16" ]]; then
        grep -q 'decode=flashinfer' <<<"$laline" \
          || die "mamba-ssm-dtype=bfloat16 did not promote decode to flashinfer - got: $laline"
      fi
    fi

    # 5. MEM FRACTION. Assert the argv value equals what provenance will record.
    #    Not hypothetical: every spec-on artifact before 2026-09-09 recorded
    #    0.90 while the server ran 0.93.
    grep -q -- "--mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.93}" <<<"$argv" \
      || die "argv mem-fraction disagrees with SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.93} - provenance would be a fabrication"
  fi

  # FreeToken's HTTP front end is live before its model worker is ready. On
  # 2026-09-03 /v1/models and /health both returned 200 while every completion
  # still returned 503 "model is still loading". The /health PAYLOAD, not its
  # status code, is the readiness proof: maintenance changes to "serving" only
  # after graph capture finishes and the API logs "ready to serve".
  if [[ "$ENGINE" == "freetoken" ]]; then
    health=$(curl -fsS "${BASE}/health") || die "${BASE}/health not reachable"
    HEALTH="$health" MODEL="$MODEL" python3 - <<'PY'
import json, os
d = json.loads(os.environ["HEALTH"])
if d.get("status") != "ok" or d.get("maintenance") != "serving":
    raise SystemExit(
        "FAIL: FreeToken front end is live but model is not ready: "
        f"status={d.get('status')!r} maintenance={d.get('maintenance')!r}"
    )
if d.get("model") != os.environ["MODEL"]:
    raise SystemExit(
        f"FAIL: FreeToken health reports model {d.get('model')!r}, "
        f"expected {os.environ['MODEL']!r}"
    )
PY
    echo "  model readiness  /health status=ok maintenance=serving"
  fi

  # W4 runtime route census. Documentation first led us to assume there was no
  # metrics route; seeing /engine/metrics registered in source then led to the
  # opposite assumption. Both were guesses about runtime wiring. Enumerate the
  # live routes, print their status, and carry the exact census into provenance.
  # A future build that mounts one therefore changes the evidence automatically.
  RUNTIME_ROUTE_STATUS=""
  if [[ "$ENGINE" == "freetoken" ]]; then
    local route code
    for route in /health /v1/models /metrics /engine/health /engine/status /engine/metrics /engine/stats; do
      code=$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' "${BASE}${route}" 2>/dev/null || true)
      [[ -n "$code" ]] || code="000"
      printf '  route            %-18s %s\n' "$route" "$code"
      RUNTIME_ROUTE_STATUS+="${route}=${code};"
    done
  fi

  # SGLang without --enable-metrics produces NO server_metrics_export.* and only
  # logs a warning, so assert the endpoint is really Prometheus-shaped.
  #
  # Captured to a variable, NOT piped into `grep -q`. Under `set -o pipefail`,
  # `curl ... | grep -q PATTERN` is a RACE: grep exits the moment it matches,
  # closing the pipe, and curl dies writing to it (exit 23 "Failure writing
  # output to destination"). pipefail then reports the pipeline as failed even
  # though the endpoint was fine. Bigger /metrics payloads lose this race more
  # often, so it passes in testing and fails on a real engine.
  #
  # SERVER_METRICS=0 means the ENGINE has no Prometheus endpoint, which is a
  # property of the engine and not a fault in the run. FreeToken documents only
  # /v1/chat/completions, /v1/responses, /v1/models and /v1/messages. Gating on
  # it would fail that arm on every single run forever. Record the absence as a
  # finding and carry on - but still probe, because "undocumented" is not
  # "absent" and a working endpoint would be worth having.
  if [[ "${SERVER_METRICS:-1}" == "1" ]]; then
    local metrics
    metrics=$(curl -fsS "${BASE}${METRICS_PATH:-/metrics}") \
      || die "${BASE}${METRICS_PATH:-/metrics} not reachable"
    grep -q '^# HELP' <<<"$metrics" \
      || die "${BASE}${METRICS_PATH:-/metrics} is not Prometheus format (SGLang needs --enable-metrics)"
    echo "  metrics          OK at ${METRICS_PATH:-/metrics}"
  else
    if curl -fsS "${BASE}/metrics" 2>/dev/null | grep -q '^# HELP'; then
      echo "  /metrics         PRESENT but undeclared - set SERVER_METRICS=1 for $ENGINE"
    else
      echo "  /metrics         none (expected for $ENGINE; server-side metrics unavailable)"
    fi
  fi

  # aiperf treats a failed KV reset as fatal mid-sweep. Better to find out now.
  # An EMPTY RESET_PATH means the engine has no flush endpoint at all
  # (llama.cpp offers only POST /slots/:id; FreeToken documents none), so those
  # arms get their cold prefill from --cache-bust instead. Asserting a hook that
  # cannot exist would fail the arm for the wrong reason.
  if [[ -n "${RESET_PATH:-}" ]]; then
    local code
    code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "${BASE}${RESET_PATH}")
    [[ "$code" == "2"* ]] || die "POST ${RESET_PATH} returned $code"
    echo "  KV reset hook    ${RESET_PATH} -> $code"
  else
    echo "  KV reset hook    none for $ENGINE - cold prefill via --cache-bust ${CACHE_BUST:-system_prefix}"
  fi

  # DCGM field assertions. A row in dcgm_metrics.csv is NOT proof of a column:
  # an unknown field name makes the exporter exit 1, but a known-yet-unsupported
  # one (every DCGM_FI_PROF_* on this card) exports NOTHING SILENTLY. Assert the
  # fields we actually rely on, not just that the endpoint answers. F4a.
  local dcgm; dcgm=$(curl -fsS "http://${DCGM_URL}/metrics") \
    || die "DCGM at ${DCGM_URL} not reachable"
  local missing=()
  for f in DCGM_FI_DEV_SM_CLOCK DCGM_FI_DEV_MEM_CLOCK DCGM_FI_DEV_GPU_TEMP \
           DCGM_FI_DEV_POWER_USAGE DCGM_FI_DEV_THERMAL_VIOLATION DCGM_FI_DEV_POWER_VIOLATION; do
    grep -q "^${f}{" <<<"$dcgm" || missing+=("$f")
  done
  (( ${#missing[@]} == 0 )) || die "DCGM missing fields: ${missing[*]} (clocks/throttle unverifiable)"
  echo "  DCGM fields      OK (${DCGM_URL})"

  # LEAK GUARD. pydantic-settings auto-loads a credential file FROM THE CURRENT
  # WORKING DIRECTORY, rejects unknown keys, and prints the rejected value in the
  # traceback - and evidence directories get published.
  #
  # Upstream enforced this by requiring a clean repo root, because its own
  # credential file lives in docker/. THIS repo keeps one at the root and it is
  # not ours to move: scripts/serve.sh and scripts/download_models.sh source it.
  # So the guard now asserts the thing that actually matters - that the directory
  # aiperf will RUN from is clean - rather than the proxy for it.
  #
  # Verified separately: `aiperf-mock-server --help` raises a ValidationError from
  # the repo root and works from a clean directory, which is what aiperf_wd() gives
  # every controller.
  local _wd; _wd="$(aiperf_wd)"
  local _leaky=()
  for _f in "$_wd"/.env "$_wd"/.env.*; do [[ -e "$_f" ]] && _leaky+=("$_f"); done
  (( ${#_leaky[@]} == 0 )) \
    || die "aiperf working dir $_wd contains ${_leaky[*]} - pydantic-settings would load and leak it"
  echo "  aiperf cwd       clean ($_wd)"
  if [[ -f "$REPO/.env" ]]; then
    echo "  note             a credential file exists at the repo root; aiperf does not run there"
  fi

  gpu_guard
}

# ---------------------------------------------------------------------------
# Provenance. Captured AFTER the engine is up, so versions are the real ones
# and not what we hoped the image contained.
# ---------------------------------------------------------------------------
record_provenance() {
  local out="${1:?record_provenance <outdir>}"
  log "recording provenance"
  mkdir -p "$out"
  local engine_ver image_digest kv_report
  if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    # No container: only legitimate under BENCH_SKIP_BOOT (mock verification).
    # Marked loudly so a mock artifact can never be mistaken for a measured one.
    [[ "${BENCH_SKIP_BOOT:-0}" == "1" ]] \
      || die "container $CONTAINER not found and BENCH_SKIP_BOOT is not set - engine did not boot"
    engine_ver="MOCK-NOT-A-REAL-ENGINE"; image_digest="MOCK"; kv_report="MOCK"
  else
    if [[ "$ENGINE" == "vllm" ]]; then
      # `grep -oE '[0-9.]+'` on "V1 LLM engine (v0.27.1)" emits TWO lines - the
      # "1" from "V1" and then "0.27.1" - so the old form recorded a two-line
      # engine_version. Strip the prefix first and take a single match.
      engine_ver=$(docker logs "$CONTAINER" 2>&1 \
        | grep -oE 'V1 LLM engine \(v[0-9.]+\)' | head -1 \
        | sed -E 's/.*\(v([0-9.]+)\).*/\1/' || echo unknown)
    elif [[ "$ENGINE" == "sglang" ]]; then
      engine_ver=$(docker exec "$CONTAINER" python3 -c 'import sglang;print(sglang.__version__)' 2>/dev/null || echo unknown)
    elif [[ "$ENGINE" == "freetoken" ]]; then
      engine_ver=$(curl -fsS "${BASE}/health" 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version", "unknown"))' \
        || echo unknown)
    else
      engine_ver=$(docker exec "$CONTAINER" /app/llama-server --version 2>/dev/null \
        | head -1 || echo unknown)
    fi
    image_digest=$(docker image inspect "$(docker inspect --format '{{.Config.Image}}' "$CONTAINER")" \
      --format '{{index .RepoDigests 0}}' 2>/dev/null || echo unknown)
    # KV capacity as the engine itself reports it.
    #
    # THE `|| true` IS LOad-BEARING. These patterns are vLLM and SGLang strings;
    # llama.cpp prints NONE of them, so grep exits 1, `pipefail` propagates it,
    # and `set -e` killed the script here - silently, with no message, right
    # after logging "recording provenance". Three llama.cpp runs died at exactly
    # this line and left empty evidence directories, and the orphaned thermal
    # sampler (a `( ) &` subshell, which inherits the parent's argv in ps) made
    # it look like a live run that was hanging.
    #
    # kv_report is OPTIONAL METADATA. Its absence is not a run failure, so it
    # must never be able to abort the run.
    kv_report=$(docker logs "$CONTAINER" 2>&1 \
      | grep -oE 'GPU KV cache size: [0-9,]+ tokens|max_total_num_tokens=[0-9]+|Mamba Cache is allocated[^)]*|KV Cache is allocated[^)]*|n_ctx_slot = [0-9]+|n_slots = [0-9]+|kv_unified = .[a-z]+.' \
      | head -4 | paste -sd'; ' - || true)
    [[ -n "$kv_report" ]] || kv_report="none reported by $ENGINE"
  fi

  # These are the values the selected engine actually receives. The old
  # recorder populated every arm from vLLM defaults, so SGLang artifacts said
  # mem_fraction=0.8 / KV=bfloat16 / graph=2 even though the recovered boot
  # argv proved 0.90 / fp8_e4m3 / graph=1. A plausible but false parity summary
  # is worse than an explicitly non-applicable field.
  local provenance_mem_fraction provenance_kv_dtype provenance_graph_max
  case "$ENGINE" in
    llamacpp)
      provenance_mem_fraction="not-applicable"
      provenance_kv_dtype="${KV_CACHE_DTYPE:-engine-default}"
      provenance_graph_max="engine-managed"
      ;;
    sglang)
      # 0.93, NOT 0.90. This default said 0.90 while the compose has defaulted
      # to 0.93 since 2026-09-03 and nothing ever set the variable, so every
      # spec-on artifact RECORDED 0.90 and RAN 0.93 (verified against
      # quality_sglang_20260904T160940Z: parity says 0.90, server_boot_args says
      # 0.93). engine_env_map now sets the variable explicitly, so this fallback
      # should never be reached - it is kept aligned so that if it ever is, it
      # cannot fabricate a value again.
      provenance_mem_fraction="${SGLANG_MEM_FRACTION_STATIC:-0.93}"
      provenance_kv_dtype="fp8_e4m3"
      provenance_graph_max="${CUDA_GRAPH_MAX_BS:-1}"
      ;;
    freetoken)
      provenance_mem_fraction="${FREETOKEN_MEMORY_RATIO:-0.90}"
      provenance_kv_dtype="bfloat16"
      provenance_graph_max="${CUDA_GRAPH_MAX_BS:-1}"
      ;;
    vllm)
      provenance_mem_fraction="${VLLM_MEM_UTIL:-0.8}"
      provenance_kv_dtype="${KV_CACHE_DTYPE:-bfloat16}"
      provenance_graph_max="${CUDA_GRAPH_MAX_BS:-64}"
      ;;
  esac

  # SGLang launch knobs. Empty selector == flag not passed == "engine-default";
  # anything else is the literal value that reached the renderer. Non-sglang
  # arms get "not-applicable" - this file's own lesson is that a plausible
  # default is worse than an explicit absence.
  local sgl_la_decode sgl_la_prefill sgl_ssm_dtype sgl_spec_quant sgl_sleep_idle
  local sgl_no_autotune sgl_alloc_conf sgl_conv_dtype sgl_numa sgl_omp sgl_jit_dir
  if [[ "$ENGINE" == "sglang" ]]; then
    sgl_la_decode="${SGLANG_LINEAR_ATTN_DECODE_BACKEND:-engine-default}"
    sgl_la_prefill="${SGLANG_LINEAR_ATTN_PREFILL_BACKEND:-engine-default}"
    sgl_ssm_dtype="${SGLANG_MAMBA_SSM_DTYPE:-engine-default}"
    sgl_spec_quant="${SGLANG_SPEC_DRAFT_QUANT:-engine-default}"
    sgl_sleep_idle="${SGLANG_SLEEP_ON_IDLE:-off}"
    sgl_no_autotune="${SGLANG_DISABLE_FI_AUTOTUNE:-off}"
    sgl_alloc_conf="${SGLANG_PYTORCH_ALLOC_CONF:-engine-default}"
    sgl_conv_dtype="${SGLANG_MAMBA_CONV_DTYPE_VAL:-engine-default}"
    sgl_numa="${SGLANG_NUMA_BIND_V2_VAL:-engine-default}"
    sgl_omp="${SGLANG_OMP_NUM_THREADS:-engine-default}"
    sgl_jit_dir="${SGLANG_JIT_CACHE_HOST_DIR:-none}"
  else
    sgl_la_decode="not-applicable";  sgl_la_prefill="not-applicable"
    sgl_ssm_dtype="not-applicable";  sgl_spec_quant="not-applicable"
    sgl_sleep_idle="not-applicable"; sgl_no_autotune="not-applicable"
    sgl_alloc_conf="not-applicable"; sgl_conv_dtype="not-applicable"
    sgl_numa="not-applicable";       sgl_omp="not-applicable"
    sgl_jit_dir="not-applicable"
  fi

  # The resolved launch configuration, as evidence (Auto_Bench.md 5.1). The
  # SGLang arm has never had one: llama.cpp gets benchmark/dump_compose.sh and
  # this arm got nothing but argv.
  if [[ "$ENGINE" == "sglang" ]]; then
    # SGLang's own resolved ServerArgs dump is ONE ~6KB line, which is why the
    # server_backend_lines grep excludes it. It goes to a sibling file so the
    # resolved value of every knob is auditable without bloating provenance.
    docker logs "$CONTAINER" 2>&1 | grep -a 'server_args=ServerArgs(' | tail -1 \
      > "$out/server_args.txt" 2>/dev/null || true
    [[ -n "${SGLANG_GEN_COMPOSE:-}" && -f "${SGLANG_GEN_COMPOSE:-}" ]] && \
      cp "$SGLANG_GEN_COMPOSE" "$out/docker-compose.sglang.gen.yaml" || true
  fi

  local input_sha256="none"
  [[ -n "${INPUT_FILE:-}" && -f "$INPUT_FILE" ]] && input_sha256="$(sha256sum "$INPUT_FILE" | cut -d' ' -f1)"

  MODEL="$MODEL" MODEL_REPO="${MODEL_REPO:-unset}" MODEL_QUANT="${MODEL_QUANT:-unset}" \
  MODEL_REVISION="${MODEL_REVISION:-unset}" ENGINE_ARGS_STR="${ENGINE_ARGS_STR:-unset}" \
  STAMP="$STAMP" WORKLOAD="$WORKLOAD" ENGINE="$ENGINE" REPO="$REPO" \
  ENGINE_VER="$engine_ver" IMAGE_DIGEST="$image_digest" KV_REPORT="$kv_report" \
  ENGINE_CTX="${ENGINE_CTX:-262144}" PREFILL_BUDGET="${PREFILL_BUDGET:-8192}" \
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}" CUDA_GRAPH_MAX_BS="$provenance_graph_max" \
  KV_CACHE_DTYPE="$provenance_kv_dtype" MEM_FRACTION="$provenance_mem_fraction" \
  SGL_LA_DECODE="$sgl_la_decode" SGL_LA_PREFILL="$sgl_la_prefill" \
  SGL_SSM_DTYPE="$sgl_ssm_dtype" SGL_SPEC_QUANT="$sgl_spec_quant" \
  SGL_SLEEP_IDLE="$sgl_sleep_idle" SGL_NO_AUTOTUNE="$sgl_no_autotune" \
  SGL_ALLOC_CONF="$sgl_alloc_conf" SGL_CONV_DTYPE="$sgl_conv_dtype" \
  SGL_NUMA="$sgl_numa" SGL_OMP="$sgl_omp" SGL_JIT_DIR="$sgl_jit_dir" \
  ISL="${ISL:-}" OSL="${OSL:-}" CONCURRENCY="${CONCURRENCY:-}" DURATION="${DURATION:-}" \
  CONTEXT_REQUESTS="${CONTEXT_REQUESTS:-unset}" \
  WARMUP="${WARMUP:-}" SEED="${SEED:-}" GOODPUT="${GOODPUT:-}" \
  UBATCH="${UBATCH:-unset}" LOAD_MODE="${LOAD_MODE:-unset}" LAZY="${LAZY:-unset}" \
  N_CPU_MOE="${N_CPU_MOE:-unset}" NGL="${NGL:-unset}" SPEC_DRAFT_NGL="${SPEC_DRAFT_NGL:-unset}" \
  MEM_CAP="${MEM_CAP_BYTES:-unset}" \
  IMAGE_TAG="${LLAMA_IMAGE:-${SGLANG_IMAGE:-${FREETOKEN_IMAGE:-default}}}" \
  SPEC="${SPEC:-unset}" SPEC_TYPE="${SPEC_TYPE:-unset}" \
  WORKLOAD_OVERRIDES="${WORKLOAD_OVERRIDES:-}" \
  SAMPLING="${SAMPLING_ARGS_STR:-unset}" THERMAL_CSV="${THERMAL_CSV:-none}" CONTAINER="$CONTAINER" \
  RUNTIME_ROUTE_STATUS="${RUNTIME_ROUTE_STATUS:-not-probed}" \
  INPUT_FILE="${INPUT_FILE:-none}" INPUT_SHA256="$input_sha256" \
  CUSTOM_DATASET_TYPE="${CUSTOM_DATASET_TYPE:-none}" \
  python3 - "$out/provenance.json" <<'PY'
import json, os, subprocess, sys
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True).strip()
    except Exception: return "unknown"
e = os.environ
json.dump({
  "stamp": e["STAMP"], "workload": e["WORKLOAD"], "engine": e["ENGINE"],
  "engine_version": e["ENGINE_VER"], "image_digest": e["IMAGE_DIGEST"],
  "model": e["MODEL"], "model_revision": e["MODEL_REVISION"],
  "kv_report_from_engine_log": e["KV_REPORT"],
  "parity_flags": {
    "context": e["ENGINE_CTX"], "mem_fraction": e["MEM_FRACTION"],
    "prefill_budget": e["PREFILL_BUDGET"], "admission_slots": e["MAX_RUNNING_REQUESTS"],
    "cudagraph_max_bs": e["CUDA_GRAPH_MAX_BS"], "prefix_cache": "disabled",
    # PINNED, not inherited: `auto` resolves to the MODEL dtype, not the weight
    # format, and is therefore checkpoint-dependent. See FINDINGS F11.
    "kv_cache_dtype": e["KV_CACHE_DTYPE"],
    # Swept knobs. These were readable ONLY from server_boot_args until
    # 2026-09-04, when an A/B report could not attribute its own arms and the
    # boot args had to be diffed by hand to recover which run was which.
    # RULES.md 2 requires every parity flag in a named field.
    # How many MEASURED requests the cell asked for. A full-window cell at 1 is
    # a single sample of a speculative-decode rate, whose spread across
    # identical boots is 31.6% (B-31) - so this number is needed to know whether
    # a decode figure can be read at all.
    "context_requests": e["CONTEXT_REQUESTS"],
    "ubatch": e["UBATCH"], "load_mode": e["LOAD_MODE"],
    # LAZY reaches the server as LLAMA_ARG_TENSOR_READ_LAZY, an env var, which
    # server_boot_args (docker .Args) can NEVER show - the same blind spot that
    # made the spec_type arms unattributable (B-15). Named field or nothing.
    "lazy": e["LAZY"],
    # Which image tag the arm ran on, so two tags of one engine (freetoken:local
    # vs freetoken:af71ba432, ghcr b10666 vs llamacpp-pr28136) are separable in
    # the report. "default" = the compose default for that engine.
    "image_tag": e["IMAGE_TAG"],
    "speculation": e["SPEC"], "spec_type": e["SPEC_TYPE"],
    # WHERE THE WEIGHTS ARE. Until 2026-09-07 an arm's offload level was
    # invisible in provenance: the VRAM-tier ladder chose --n-cpu-moe per tier
    # and nothing recorded it, so a tier could not be audited from its own
    # evidence. --n-cpu-moe and --n-gpu-layers decide what sits in host RAM,
    # which is the whole subject of the tier work.
    "n_cpu_moe": e["N_CPU_MOE"], "ngl": e["NGL"],
    # Where the MTP draft head sits: 0 = CPU (no VRAM cost), 99 = GPU.
    "spec_draft_ngl": e["SPEC_DRAFT_NGL"],
    # SGLANG LAUNCH KNOBS, one named field each (RULES.md 2). "engine-default"
    # means the flag was NOT passed and the engine resolved it itself - which is
    # not the same as knowing what it resolved to, so read these together with
    # server_backend_lines and server_args.txt. On a non-sglang arm they are
    # "not-applicable" rather than a plausible-looking false value.
    #
    # The container-env ones (alloc conf, conv dtype, numa, omp) can NEVER
    # appear in server_boot_args, because docker .Args holds argv only. That is
    # the same blind spot that let LAZY go unattributed for three campaigns
    # (B-25), so they are named here or they are invisible.
    "sglang_linear_attn_decode_backend": e["SGL_LA_DECODE"],
    "sglang_linear_attn_prefill_backend": e["SGL_LA_PREFILL"],
    "sglang_mamba_ssm_dtype": e["SGL_SSM_DTYPE"],
    "sglang_spec_draft_quantization": e["SGL_SPEC_QUANT"],
    "sglang_sleep_on_idle": e["SGL_SLEEP_IDLE"],
    "sglang_disable_flashinfer_autotune": e["SGL_NO_AUTOTUNE"],
    "sglang_pytorch_cuda_alloc_conf": e["SGL_ALLOC_CONF"],
    "sglang_mamba_conv_dtype": e["SGL_CONV_DTYPE"],
    "sglang_numa_bind_v2": e["SGL_NUMA"],
    "sglang_omp_num_threads": e["SGL_OMP"],
    "sglang_jit_cache_host_dir": e["SGL_JIT_DIR"],
    # The intended host-RAM cap. What was ACTUALLY applied is read back from
    # docker and the live cgroup into memory.json - never trust this field alone.
    "mem_cap_bytes_intended": e["MEM_CAP"],
  },
  # What the caller changed relative to the workload conf - i.e. the axis under
  # test. Empty means this run IS the control.
  "swept_overrides": e["WORKLOAD_OVERRIDES"],
  "workload_params": {
    "isl": e["ISL"], "osl": e["OSL"], "concurrency": e["CONCURRENCY"],
    "duration_s": e["DURATION"], "warmup": e["WARMUP"], "seed": e["SEED"],
    "goodput_slo": e["GOODPUT"], "sampling": e["SAMPLING"],
    "input_file": e["INPUT_FILE"], "input_sha256": e["INPUT_SHA256"],
    "custom_dataset_type": e["CUSTOM_DATASET_TYPE"],
  },
  "thermal_trace": e["THERMAL_CSV"],
  # The FULL argv the server actually booted with, read back off the container.
  # Without this, unpinned defaults (notably the ATTENTION BACKEND, which each
  # engine picks from its own candidate set) are unrecoverable once the
  # container is removed - and an attention-backend difference shows up exactly
  # as a prefill/TTFT difference, which is where our engine gap lives.
  "server_boot_args": sh(f"docker inspect --format '{{{{join .Args \" \"}}}}' {e['CONTAINER']}"),
  # Targeted: exclude SGLang's server_args dump, which is a single ~6KB line that
  # matches any 'backend' grep and bloated every artifact when first added.
  "server_backend_lines": sh(
      f"docker logs {e['CONTAINER']} 2>&1 | grep -vE 'server_args=' "
      # The lines that PROVE a linear-attn backend is live do not all contain
      # the word "backend" - 'Using FlashInfer GDN kernels' and 'Using CuTe DSL
      # GDN prefill' are the runtime evidence for the flashinfer GDN path, and
      # the old pattern could not see either. head raised 6 -> 12 to fit them.
      "| grep -iE 'attention backend|Use .* backend|Using .*[Bb]ackend|kernel backend"
      "|GDN kernel|GDN prefill|linear attention|linear-attn|RecoverSSM' "
      "| head -12 | cut -c1-200"),
  "runtime_route_probe": {
      k: v for k, v in (
          item.split("=", 1) for item in e.get("RUNTIME_ROUTE_STATUS", "").split(";")
          if "=" in item
      )
  },
  "aiperf_version": sh(f"{e['REPO']}/.venv/bin/aiperf --version"),
  "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader"),
  "gpu_at_start": sh("nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader"),
  # This card reports MARGIN, not an absolute trip point - see FINDINGS F4a.
  "gpu_thermal_margin_at_start_c": sh("nvidia-smi --query-gpu=temperature.gpu.tlimit --format=csv,noheader"),
  "gpu_power_limit_w": sh("nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader"),
  "cuda": sh("nvidia-smi | grep -oE 'CUDA Version: [0-9.]+'"),
  "host_kernel": sh("uname -r"),
  "host_cores": sh("nproc"),
  # RULES.md 2: a number whose provenance we cannot reconstruct gets deleted.
  #
  # "aiperf 0.12.0" is AMBIGUOUS and that ambiguity is load-bearing. The PyPI
  # release 0.12.0 does NOT contain reset_kv_cache; the git checkout that also
  # calls itself 0.12.0 does. Same version string, different code, different
  # behaviour. So record the resolved MODULE PATH and, when it is a checkout,
  # its commit - the version string alone is not provenance here.
  "aiperf_module": sh(f"uv run --project {e['REPO']} python -c "
                      "'import aiperf,os;print(os.path.dirname(aiperf.__file__))'"),
  "source_commits": {
    "repo": sh(f"git -C {e['REPO']} rev-parse HEAD"),
    "aiperf": sh(f"uv run --project {e['REPO']} python -c "
                 "'import aiperf,os;print(os.path.dirname(aiperf.__file__))' "
                 "| xargs -r dirname | xargs -r dirname "
                 "| xargs -r -I{} git -C {} rev-parse HEAD"),
  },
  "model_repo": e.get("MODEL_REPO", ""),
  "model_quant": e.get("MODEL_QUANT", ""),
  "engine_args": e.get("ENGINE_ARGS_STR", ""),
}, open(sys.argv[1], "w"), indent=2)
print(open(sys.argv[1]).read())
PY
}

# ---------------------------------------------------------------------------
# Sampling parity. NOT optional and NOT a default we inherit.
#
# run.sh historically passed no sampling parameters at all, leaving each server
# on its own defaults - an unpinned parity variable under rules 4 and 6. These
# values are READ FROM the pinned checkpoint's generation_config.json
# (temperature 1.0 / top_p 0.95 / top_k 20, identical in the BF16 and FP8
# snapshots), not invented here. top_k is a vendor extension, not a standard
# OpenAI Chat Completions field: confirm both servers honour it.
# ---------------------------------------------------------------------------
# These populate the GLOBAL array SAMPLING_ARGS - they do not print. An earlier
# version printed the flags and was called as $(sampling_args), which ran it in
# a subshell, so SAMPLING_ARGS_STR never reached the caller and provenance
# silently recorded "unset". Use: sampling_args; aiperf ... "${SAMPLING_ARGS[@]}"
sampling_args() {
  # THINKING=off runs the model's NON-THINKING mode, which is two changes, not one:
  # the chat template must stop emitting reasoning AND the sampler must switch to
  # the card's instruct values. Doing only one of them measures a mismatched config.
  #
  # Qwen3.8-Flash-Next model card:
  #   thinking      temp 1.0  top_p 0.95  top_k 20  presence 0.0
  #   non-thinking  temp 0.7  top_p 0.80  top_k 20  presence 1.5
  #
  # Why it matters for these tables: with thinking ON, completion_tokens include
  # reasoning tokens, so "decode tok/s" counts tokens the user never sees and OSL
  # is not comparable across engines whose templates differ. Thinking OFF makes
  # every engine emit only answer tokens.
  local temp top_p top_k presence
  if [[ "${THINKING:-on}" == "off" ]]; then
    temp="${SAMPLING_TEMPERATURE:-0.7}"; top_p="${SAMPLING_TOP_P:-0.80}"
    top_k="${SAMPLING_TOP_K:-20}";       presence="${SAMPLING_PRESENCE:-1.5}"
  else
    temp="${SAMPLING_TEMPERATURE:-1.0}"; top_p="${SAMPLING_TOP_P:-0.95}"
    top_k="${SAMPLING_TOP_K:-20}";       presence="${SAMPLING_PRESENCE:-0.0}"
  fi
  SAMPLING_ARGS=(
    --extra-inputs "temperature:${temp}"
    --extra-inputs "top_p:${top_p}"
    --extra-inputs "top_k:${top_k}"
    --extra-inputs "presence_penalty:${presence}"
  )
  if [[ "${THINKING:-on}" == "off" ]]; then
    # aiperf accepts a JSON string for --extra-inputs, which is the only way to
    # send a nested object. This is the cross-engine switch: all three serve an
    # OpenAI chat endpoint and pass chat_template_kwargs to the template.
    SAMPLING_ARGS+=(--extra-inputs '{"chat_template_kwargs": {"enable_thinking": false}}')
  fi
  SAMPLING_ARGS_STR="thinking=${THINKING:-on} temperature=${temp},top_p=${top_p},top_k=${top_k},presence=${presence}"
}

# ---------------------------------------------------------------------------
# workload_args — dataset SHAPE flags, built in one place so discover.sh and
# validate.sh cannot describe the same workload differently.
#
# Multi-turn is opt-in: a conf sets TURNS_MEAN>1 and gets the conversation
# flags. Flag spellings verified against aiperf's CLI reference
# (cli-options.md:611-644) rather than guessed - the aliases are
# --conversation-turn-mean/--session-turns-mean etc., and the DELAYS ARE
# MILLISECONDS.
#
# What multi-turn actually measures here (plan.md §D): concurrency limits
# concurrent CONVERSATIONS, and the inter-turn delay is a CLIENT-SIDE wait with
# no request in flight - so it is growing prompt history and bursty arrivals,
# NOT a server-side slot held across think time.
# ---------------------------------------------------------------------------
workload_args() {
  # Engine-specific flags (tokenizer, corpus, cache-bust, reset hook, the
  # engine's own extra-inputs) come from ONE place so this function and
  # engine_aiperf_args cannot describe the same engine differently.
  engine_aiperf_args
  if [[ -n "${INPUT_FILE:-}" ]]; then
    WORKLOAD_ARGS=(
      --osl "$OSL" --osl-stddev "${OSL_STDDEV:-0}"
      ${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"}
    )
  else
    WORKLOAD_ARGS=(
      --isl "$ISL" --isl-stddev "${ISL_STDDEV:-0}"
      --osl "$OSL" --osl-stddev "${OSL_STDDEV:-0}"
      ${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"}
    )
  fi
  # ignore_eos + min_tokens force exactly OSL output tokens per REQUEST (so per
  # turn in multi-turn). This model thinks by default and is far more verbose
  # than typical; without this a throughput number measures the model's
  # verbosity, not the engine.
  #
  # llama.cpp has ignore_eos (server-schema.cpp:471) but NO min_tokens, and
  # sending an unknown field risks a 400. ignore_eos alone still runs generation
  # to max_tokens, so exact OSL survives - the post-run gate proves it by failing
  # any cell whose measured OSL drifts more than 2% from the request.
  if [[ "${ENGINE_MIN_TOKENS:-1}" == "1" ]]; then
    WORKLOAD_ARGS+=(--extra-inputs "min_tokens:$OSL")
  fi
  if [[ -n "${TURNS_MEAN:-}" ]] && (( TURNS_MEAN > 1 )); then
    WORKLOAD_ARGS+=(
      --conversation-num "${CONVERSATION_NUM:-200}"
      --conversation-turn-mean "$TURNS_MEAN"
      --conversation-turn-stddev "${TURNS_STDDEV:-0}"
      --conversation-turn-delay-mean "${TURN_DELAY_MS:-0}"
      --conversation-turn-delay-stddev "${TURN_DELAY_STDDEV_MS:-0}"
    )
    WORKLOAD_SHAPE="multi-turn: ${TURNS_MEAN}+-${TURNS_STDDEV:-0} turns, \
${TURN_DELAY_MS:-0}+-${TURN_DELAY_STDDEV_MS:-0}ms client-side delay, \
${CONVERSATION_NUM:-200} conversations"
  else
    WORKLOAD_SHAPE="single-turn"
  fi
}

# greedy_args — for the speculative correctness gate ONLY. Never for throughput.
greedy_args() {
  SAMPLING_ARGS=(--extra-inputs "temperature:0" --extra-inputs "top_p:1")
  SAMPLING_ARGS_STR="temperature=0,top_p=1 (greedy)"
}

# ---------------------------------------------------------------------------
# Post-run gate. The aiperf-era successor to check.py. A run that trips any of
# these is not a result.
# ---------------------------------------------------------------------------
post_gate() {
  local out="${1:?post_gate <outdir>}" want_osl="${2:?post_gate <outdir> <osl>}"
  log "post-run gate"
  python3 - "$out" "$want_osl" "${SERVER_METRICS:-1}" <<'PY'
import json, sys, pathlib
out, want_osl = pathlib.Path(sys.argv[1]), float(sys.argv[2])
# An engine with no Prometheus endpoint (FreeToken) cannot produce
# server_metrics_export.json. Gating on it would fail that arm on every run for
# a property of the engine rather than a fault in the measurement.
want_server_metrics = (len(sys.argv) < 4 or sys.argv[3] == "1")
fail = []
# RUN-LEVEL exports only. A recursive glob also matches
# profile_runs/run_*/phases/{warmup,profiling}/profile_export_aiperf.json, and
# those per-phase files legitimately carry no telemetry_data and no sibling
# server_metrics_export.json - gating them produces guaranteed false failures.
# The run-level rollup is the artifact that actually has both.
cells = sorted(p for p in out.glob("**/profile_export_aiperf.json")
               if "phases" not in p.parts)
if not cells:
    fail.append("no cells produced")
for c in cells:
    d = json.loads(c.read_text()); name = c.parent.name
    err = (d.get("error_request_count") or {}).get("avg", 0) or 0
    if err: fail.append(f"{name}: {err} errored requests")
    osl = (d.get("output_sequence_length") or {}).get("avg")
    if osl is not None and abs(osl - want_osl) > 0.02 * want_osl:
        fail.append(f"{name}: OSL {osl} != requested {want_osl} (ignore_eos/min_tokens not honored?)")
    if want_server_metrics and not (c.parent / "server_metrics_export.json").exists():
        fail.append(f"{name}: no server_metrics_export.json (engine /metrics not scraped)")

    # Auto_Bench.md 5/7: record WHY, not just that. A 400 from an unsupported
    # sampling field, a 500 under load and a timeout are three different
    # failures that error_request_count alone flattens into one number.
    codes = {}
    for k, v in (d.get("error_response_codes") or d.get("response_codes") or {}).items():
        codes[str(k)] = v
    if codes:
        print(f"  {name}: response codes {codes}")
    tel = d.get("telemetry_data") or {}
    if not (tel.get("summary") or {}).get("endpoints_successful"):
        fail.append(f"{name}: no GPU telemetry")
    else:
        keys = set()
        for ep in (tel.get("endpoints") or {}).values():
            for g in (ep.get("gpus") or {}).values():
                keys |= set((g.get("metrics") or {}).keys())
        if "nvidia_sm_clock" not in keys:
            fail.append(f"{name}: no sm_clock - DCGM not used, clocks unverifiable")
print(f"  {len(cells)} cells checked")
if fail:
    print("\nGATE FAILED:"); [print("  -", f) for f in fail]; sys.exit(1)
print("  all checks passed")
PY
}
