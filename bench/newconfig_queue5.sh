#!/usr/bin/env bash
# Queue 5: the viewer's SGLang feedback, tested one knob at a time.
#
# WHAT PROMPTED IT. A viewer said the SGLang arm is "missing a ton of speed
# optimizations", pointing at jpezzulli/sglang-rtxpro6000. Reading our own
# evidence, he is right about the mechanism and we can name it exactly:
#
#   Linear attention kernel backend: decode=triton, prefill=triton, verify=triton
#
# That line is in EVERY SGLang artifact this study has published. Qwen3.8-Flash-
# Next is a hybrid GDN model, so the linear-attention path is what most of its
# layers run, and it has been on Triton throughout. His launcher passes
# --linear-attn-{decode,prefill}-backend flashinfer; ours passes neither.
#
# WHY OURS FELL BACK. Read from the image's own server_args.py (Auto_Bench 3 -
# read the engine source, not the flag list). _handle_linear_attn_backend
# auto-promotes decode to FlashInfer only when mamba_ssm_dtype == "bfloat16".
# configs/mamba_utils.py defaults that to float32 and our compose never set it,
# so the promotion never fired. Full trace:
#   results/gates/sglang_flagcap_20260909T180800Z/FINDING.md
#
# THE PAIR IS NOT SEPARABLE. Same function raises
#   ValueError: --linear-attn-decode-backend flashinfer on SM100+ requires
#               --mamba-ssm-dtype bfloat16
# and setting the ssm dtype to bfloat16 ALONE auto-promotes decode and verify.
# So S1 below is ONE two-flag arm and is labelled as such. Anyone reading this
# table must not split it into two rows.
#
# NOT TESTED HERE, and why:
#   --gdn-mtp-cache-mode none   bench/sglang_flagcap.sh proved this image's
#                               argparse does not have it. It is a source-level
#                               feature of his fork (worth +10.6% KV pool on his
#                               box) and needs the fork build, not a flag.
#   --sleep-on-idle             this harness idles the server 45-240s between
#                               cells and settle_gpu asserts on freed VRAM, so
#                               it would put wake latency into TTFT. Its own
#                               experiment, never inside a ladder.
#   --mem-fraction-static > 0.93  the pool is capped by --context-length at the
#                               full window, not by memory (FULL_CONTEXT.md).
#                               Already spent; not a full-window lever.
#
# ORDER. Boot-only first (cheap, settles capacity and boot failures for free),
# then fn_smoke per arm, then the full window only for arms that earned it.
# Nothing here reruns the published bars: those are re-measured as controls in
# the same session so the comparison is within-session (Auto_Bench 4/8).
#
# EXPECTED DURATION. Stage A ~15 min, Stage B ~35 min, Stage C ~40 min.
#
# STAGE C IS GATED: it runs only for arms Stage B promoted. The rule is +3%
# decode over the SAME-SESSION control, written into Stage C rather than left
# to judgement after the fact.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"

REPO="$PWD"; LOG="$REPO/results/newconfig_queue5.log"
G="--execute --plan-id FLASHNEXT-R1"
export THINKING=off          # the published full-window bars are THINKING=off

say(){ echo "[nq5 $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say "START $n"; "$@" >>"$LOG" 2>&1; local rc=$?; say "END   $n exit=$rc"; return $rc; }
clean(){ for c in q38n flashnext freetoken q38n-mtp; do docker rm -f "$c" >/dev/null 2>&1; done; sleep 20; }
dcgm(){ curl -fsS --max-time 3 localhost:9401/metrics 2>/dev/null | grep -q '^DCGM_FI_DEV_SM_CLOCK{' \
  || DCGM_NAME=dcgm-bench DCGM_PROF_PORT=9401 DCGM_COUNTERS=bench/dcgm_metrics.csv \
     DCGM_READY_FIELD=DCGM_FI_DEV_SM_CLOCK ./benchmark/dcgm_exporter.sh up >>"$LOG" 2>&1; }
newest(){ ls -dt "$REPO"/artifacts/$1 2>/dev/null | head -1; }

# Decode tok/s out of an artifact, so a promotion decision is made from evidence
# rather than from the terminal (Auto_Bench 1).
decode_of(){ python3 - "$1" <<'PY' 2>/dev/null
import json,pathlib,sys
c=[p for p in pathlib.Path(sys.argv[1]).rglob("profile_export_aiperf.json") if "phases" not in p.parts]
if not c: raise SystemExit(1)
j=json.loads(sorted(c)[0].read_text())
print(f"{(j.get('output_token_throughput_per_user') or {}).get('avg', 0):.2f}")
PY
}

# The arms. NAME|selector assignments (space separated, empty = control).
ARMS=(
  "S0_control|"
  "S1_flashinfer_gdn|SGLANG_MAMBA_SSM_DTYPE=bfloat16 SGLANG_LINEAR_ATTN_DECODE_BACKEND=flashinfer SGLANG_LINEAR_ATTN_PREFILL_BACKEND=flashinfer"
  "S2_prefill_only|SGLANG_LINEAR_ATTN_PREFILL_BACKEND=flashinfer"
  "S3_draft_unquant|SGLANG_SPEC_DRAFT_QUANT=unquant"
  "S4_env_group|SGLANG_PYTORCH_ALLOC_CONF=expandable_segments:True SGLANG_NUMA_BIND_V2_VAL=false SGLANG_OMP_NUM_THREADS=4"
  "S5_prefill4096|PREFILL_BUDGET=4096"
)

dcgm; clean
say "flag capability manifest: $(ls -1dt "$REPO"/results/gates/sglang_flagcap_*/flags.tsv 2>/dev/null | head -1)"

# ---------------------------------------------------------------------------
# STAGE A - boot only. Capacity and boot failures settled without aiperf.
# ---------------------------------------------------------------------------
say "=== STAGE A: boot-only pool probe ==="
run "A_pool_probe" ./bench/sglang_pool_probe.sh || say "WARN pool probe returned nonzero"
clean

# ---------------------------------------------------------------------------
# STAGE B - fn_smoke per arm. One variable per arm, control re-run FIRST so the
# comparison is within-session.
# ---------------------------------------------------------------------------
say "=== STAGE B: fn_smoke per arm ==="
declare -A SMOKE_DECODE SMOKE_DIR
for entry in "${ARMS[@]}"; do
  name="${entry%%|*}"; envs="${entry#*|}"
  say "--- $name  [${envs:-control}] ---"
  if run "B_$name" env $envs ./bench/run.sh $G sglang fn_smoke; then
    d="$(newest "fn_smoke_sglang_*")"
    SMOKE_DIR[$name]="$d"; SMOKE_DECODE[$name]="$(decode_of "$d" || echo 0)"
    say "  $name decode=${SMOKE_DECODE[$name]} evidence=$(basename "${d:-none}")"
  else
    SMOKE_DECODE[$name]=0
    say "  $name FAILED - not promoted"
  fi
  clean
done

say "=== STAGE B summary ==="
for entry in "${ARMS[@]}"; do
  name="${entry%%|*}"; say "  $name  decode=${SMOKE_DECODE[$name]:-0}"
done

# ---------------------------------------------------------------------------
# STAGE C - the full window, for promoted arms only.
#
# PROMOTION RULE, written down rather than judged: at least +3% decode over the
# S0 control measured in THIS session. Below that, a single fn_smoke cell cannot
# distinguish a win from run-to-run spread, and spending 10 minutes of full
# window on it would only add a noisy row to the chart.
# ---------------------------------------------------------------------------
ctl="${SMOKE_DECODE[S0_control]:-0}"
say "=== STAGE C: full window (control decode=$ctl) ==="
[[ "$(echo "$ctl > 0" | bc -l)" == 1 ]] || { say "FATAL: control arm produced no decode number - stopping"; exit 1; }

# The control bar is always re-measured at the full window, promoted or not:
# every other bar in the chart needs a same-session reference.
run "C_S0_control" ./bench/context.sh $G sglang fn_maxctx || say "WARN control maxctx failed"
clean

for entry in "${ARMS[@]}"; do
  name="${entry%%|*}"; envs="${entry#*|}"
  [[ "$name" == "S0_control" ]] && continue
  d="${SMOKE_DECODE[$name]:-0}"
  if [[ "$(echo "$d >= $ctl * 1.03" | bc -l)" == 1 ]]; then
    say "--- PROMOTED $name ($d vs $ctl) -> full window ---"
    run "C_$name" env $envs ./bench/context.sh $G sglang fn_maxctx || say "WARN $name maxctx failed"
    clean
  else
    say "--- not promoted: $name ($d vs control $ctl, needs >= 3%) ---"
  fi
done

say "=== DONE. Evidence: artifacts/fn_smoke_sglang_* and artifacts/context_fn_maxctx_sglang_* ==="
say "Next: uv run --with matplotlib bench/make_charts.py && uv run bench/build_report.py"
