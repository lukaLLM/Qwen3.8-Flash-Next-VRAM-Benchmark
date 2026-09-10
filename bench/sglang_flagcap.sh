#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Which of the sibling fork's launch flags does OUR image's argparse actually
# have?
#
#   ./bench/sglang_flagcap.sh
#
# WHY THIS EXISTS. The flags we want to test come from
# jpezzulli/sglang-rtxpro6000 (branch pennyroyal-main-sm120-final). The image we
# actually serve with, sglang-flashnext-sm120:local, is a DIFFERENT overlay -
# yepapa-nest/qwen38-flashnext-rtx6000 on base SGLang d91c3682. The two trees
# share an ancestor and nothing else: the fork's server_args.py is a rewritten
# annotated-dataclass file. So "he uses --linear-attn-decode-backend" is not
# evidence that our binary accepts it.
#
# Asking argparse costs ~30 s and no GPU. Discovering a missing flag from a
# four-minute boot failure costs a slot, and discovering it from a run that
# booted anyway costs a wrong number.
#
# NO GPU, no --gpus, no model load, no lock. Safe to run under a live benchmark.
#
# Three probes, because each answers a question the others cannot:
#   1. structured   build the parser, dump every action -> existence AND the
#                   default AND the choices AND whether it takes a value. The
#                   defaults are what let the ENV_MAP comments be written
#                   truthfully instead of guessed.
#   2. --help text  the human-readable record, and the fallback if the
#                   ServerArgs API differs in this build.
#   3. env grep     an env var this build never READS is a silent no-op. That
#                   is the B-25 failure class - a setting that is accepted,
#                   ignored, and reported as active - so it is checked here
#                   rather than trusted.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${SGLANG_IMAGE:-sglang-flashnext-sm120:local}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/gates/sglang_flagcap_${STAMP}"
mkdir -p "$OUT"

# The candidates. Sourced from the sibling's configs/pennyroyal/serve-flash-next*.sh
# and from docker/sglang/serve-flash-next.sh, which was vendored from
# pennyroyal-v2.1.0 and is the closest thing in this repo to his launcher.
CANDIDATES=(
  # --- the ones the plan wants to A/B ---
  --linear-attn-decode-backend
  --linear-attn-prefill-backend
  --gdn-mtp-cache-mode
  --mamba-ssm-dtype
  --speculative-draft-model-quantization
  --sleep-on-idle
  --chunked-prefill-size
  --mem-fraction-static
  # --- in his launcher, not in our compose; probed so the gap is documented ---
  --speculative-token-map
  --ple-offload-embedding
  --disable-flashinfer-autotune
  --weight-loader-drop-cache-after-load
  --enable-hierarchical-cache
  --hicache-size
  --json-model-override-args
  --watchdog-timeout
  --enable-request-time-stats-logging
  --mamba-track-interval
  --mamba-radix-cache-strategy
  # --- expected ABSENT/unused; probed to close the question rather than assert it ---
  --enable-torch-compile
  --torch-compile-max-bs
  --attention-backend
  --cuda-graph-max-bs
)

# Env vars the sibling's launcher exports. Checked for reachability, not existence.
ENV_CANDIDATES=(
  SGLANG_MAMBA_CONV_DTYPE
  SGLANG_NUMA_BIND_V2
  SGLANG_JIT_CACHE_DIR
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN
  SGLANG_ENABLE_OVERLAP_PLAN_STREAM
  SGLANG_QWEN4_PLE_NVME_BACKEND
)

docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "no such image: $IMAGE" >&2; exit 1; }

# Image identity FIRST. A flag table without it cannot be attributed to a build,
# and this repo has already been bitten by two images sharing a gate name.
{
  echo "image        $IMAGE"
  docker image inspect "$IMAGE" --format 'id           {{.Id}}'
  docker image inspect "$IMAGE" \
    --format 'build_commit {{index .Config.Labels "SGLANG_BUILD_COMMIT"}}'
  docker image inspect "$IMAGE" \
    --format 'image_tag    {{index .Config.Labels "SGLANG_IMAGE_TAG"}}'
} > "$OUT/image.txt" 2>&1
cat "$OUT/image.txt"

# --user so the probe cannot leave root-owned evidence: results/gates/ already
# carries two root-owned files from earlier `docker run` probes.
DRUN=(docker run --rm --user "$(id -u):$(id -g)" --entrypoint python "$IMAGE")

echo "[1/3] dumping argparse actions"
"${DRUN[@]}" -c '
import argparse, json, sys
try:
    from sglang.srt.server_args import ServerArgs
except Exception as e:
    print(json.dumps({"error": repr(e)})); sys.exit(0)
p = argparse.ArgumentParser()
ServerArgs.add_cli_args(p)
print(json.dumps([{
    "opts": a.option_strings,
    "default": repr(a.default),
    "choices": [str(c) for c in a.choices] if a.choices else None,
    "nargs": a.nargs,
    "action": type(a).__name__,
} for a in p._actions]))
' > "$OUT/flags.json" 2>"$OUT/flags.json.err"

echo "[2/3] capturing --help"
"${DRUN[@]}" -m sglang.launch_server --help > "$OUT/help.txt" 2>&1

echo "[3/3] checking env-var reachability in the installed tree"
: > "$OUT/env_vars.tsv"
printf 'env_var\tread_by_build\tfiles\n' >> "$OUT/env_vars.tsv"
for v in "${ENV_CANDIDATES[@]}"; do
  hits=$(docker run --rm --user "$(id -u):$(id -g)" --entrypoint bash "$IMAGE" -c \
    "grep -rl '$v' /sgl-workspace/sglang/python/sglang 2>/dev/null | head -3 | tr '\n' ',' ")
  [[ -n "$hits" ]] && read_by=yes || read_by=NO
  printf '%s\t%s\t%s\n' "$v" "$read_by" "${hits:-none}" >> "$OUT/env_vars.tsv"
done

# Build the table. Prefer the structured dump; fall back to --help text.
python3 - "$OUT" "${CANDIDATES[@]}" <<'PY' | tee "$OUT/flags.tsv"
import json, pathlib, sys
out = pathlib.Path(sys.argv[1]); cands = sys.argv[2:]
actions, err = [], None
try:
    data = json.loads((out / "flags.json").read_text())
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
    else:
        actions = data
except Exception as e:
    err = repr(e)

help_txt = (out / "help.txt").read_text(errors="replace") if (out / "help.txt").exists() else ""
by_opt = {o: a for a in actions for o in a["opts"]}

print("flag\tpresent\tkind\tdefault\tchoices")
missing = []
for f in cands:
    a = by_opt.get(f)
    if a:
        # store_true has nargs 0; anything else consumes a value.
        kind = "bool" if a["action"] == "_StoreTrueAction" or a["nargs"] == 0 else "value"
        ch = ",".join(a["choices"]) if a["choices"] else "-"
        print(f"{f}\tyes\t{kind}\t{a['default']}\t{ch}")
    elif err and f in help_txt:
        # Structured probe unavailable, but the flag is in the help text.
        print(f"{f}\tyes(help)\t?\t?\t?")
    else:
        print(f"{f}\tNO\t-\t-\t-")
        missing.append(f)

if err:
    print(f"\n# structured probe unavailable, fell back to --help: {err}", file=sys.stderr)
print(f"\n# SUPPORTED {len(cands) - len(missing)}/{len(cands)}", file=sys.stderr)
if missing:
    print("# MISSING " + " ".join(missing), file=sys.stderr)
PY

echo
echo "  env-var reachability:"
column -t -s$'\t' "$OUT/env_vars.tsv" | sed 's/^/    /'
echo
echo "  -> $OUT"
