#!/usr/bin/env bash
# ============================================================================
# Unattended pipeline driver. Chains discovery + validation for both engines.
#
#   nohup ./bench/run_all.sh > artifacts/run_all.log 2>&1 &
#   disown
#
# WHY THIS EXISTS: nothing here should depend on an interactive session being
# alive. Each stage is a normal detached process writing its own artifacts, so
# a locked laptop, a dropped SSH connection or a closed editor cannot stall the
# pipeline between stages.
#
# ONE ENGINE AT A TIME (rule 6). Both compose services bind the same host port,
# so an overlap fails on the port bind rather than quietly sharing VRAM. Stages
# are strictly sequential for that reason - do not parallelise them.
#
# FAILURE POLICY: a failed stage is RECORDED and the pipeline CONTINUES to the
# next independent stage. A discovery failure does skip its own validation,
# because validate.sh needs a confirmed ceiling and would otherwise fail
# noisily for a reason already recorded upstream.
#
# Resumable: stages whose output already exists are skipped, so re-running
# after an interruption picks up where it stopped rather than redoing hours of
# GPU time. Use FORCE=1 to re-run everything.
# ============================================================================
set -uo pipefail   # deliberately NOT -e: a failed stage must not kill the run

# --- auto_run/Auto_Bench.md 1: nothing runs by accident --------------------
# This driver is the most dangerous one to start by accident - it chains hours
# of GPU work across both engines - so it gets the same gate as the rest.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/plan_gate.sh"
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STATE="$REPO/artifacts/run_all_${STAMP}"
mkdir -p "$STATE"
SUMMARY="$STATE/summary.txt"

: "${WORKLOADS:=w1_chat}"          # space-separated; add w2_multiturn when ready
: "${ENGINES:=vllm sglang}"
: "${FORCE:=0}"
: "${DO_VALIDATE:=1}"

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$SUMMARY"; }

stage() {   # stage <label> <marker-file-that-means-done> <command...>
  local label="$1" marker="$2"; shift 2
  if [[ "$FORCE" != "1" && -s "$marker" ]]; then
    say "SKIP  $label (already done: ${marker#"$REPO"/})"
    return 0
  fi
  say "START $label"
  local t0=$SECONDS
  if "$@" >"$STATE/${label}.log" 2>&1; then
    say "OK    $label  ($(( (SECONDS-t0)/60 ))m)"
    return 0
  else
    say "FAIL  $label  ($(( (SECONDS-t0)/60 ))m) - see ${STATE#"$REPO"/}/${label}.log"
    return 1
  fi
}

say "pipeline start: engines='$ENGINES' workloads='$WORKLOADS' validate=$DO_VALIDATE"
say "state dir: ${STATE#"$REPO"/}"

for wl in $WORKLOADS; do
  for eng in $ENGINES; do
    disc_marker="$REPO/bench/workloads/.discovered/${wl}_${eng}.json"

    if stage "discover_${wl}_${eng}" "$disc_marker" \
         "$HERE/discover.sh" "$eng" "$wl"; then
      ceiling=$(python3 -c "
import json;print(json.load(open('$disc_marker'))['confirmed_sla_concurrency_ceiling'])" 2>/dev/null || echo '?')
      say "      -> confirmed ceiling: $ceiling"

      if [[ "$DO_VALIDATE" == "1" ]]; then
        # No stable pre-known marker for validation output, so it is keyed on a
        # per-stage sentinel rather than an artifact path.
        vmark="$STATE/.validated_${wl}_${eng}"
        if stage "validate_${wl}_${eng}" "$vmark" \
             "$HERE/validate.sh" "$eng" "$wl"; then
          date > "$vmark"
        fi
      fi
    else
      say "      -> skipping validation for ${wl}/${eng}: no confirmed ceiling"
    fi
  done
done

# Always leave the GPU cold and the port free, whatever happened above.
( cd "$REPO/docker" && docker compose down >/dev/null 2>&1 || true )

say "pipeline complete"
say "GPU: $(nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader)"
echo
echo "===== SUMMARY ====="
cat "$SUMMARY"
