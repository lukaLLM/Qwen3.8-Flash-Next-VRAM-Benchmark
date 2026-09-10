#!/usr/bin/env python3
"""Render a run-scoped compose override for the SGLang arm.

    bench/sglang_render_compose.py <base-compose> <out-override>

WHY A RENDERER AND NOT A SIMPLER MECHANISM.

The base compose holds SGLang's argv as a literal YAML sequence. Four obvious
ways to make a flag optional all fail:

  - A second `-f` fragment that "adds" one argument. Compose REPLACES `command`
    on merge, it never appends. The fragment would drop every other flag.
  - `${VAR:+--flag}` inside the sequence. A sequence entry always produces
    exactly ONE element; there is no way to produce zero. An empty element is an
    empty argv token and argparse rejects it.
  - `command: ["bash","-c","... ${EXTRA}"]`. Works, but destroys the per-line
    commentary in the base file (which carries the measured pool table next to
    --mem-fraction-static) and adds shell word-splitting as a silent-corruption
    path.
  - An entrypoint wrapper script. Breaks provenance: record_provenance() reads
    `docker inspect --format '{{join .Args " "}}'`, and a wrapper collapses that
    to the wrapper path. `server_boot_args` is the most-cited evidence field in
    results/BLOCKERS.md; replacing it with a log grep trades a reliable
    mechanism for a fragile one.

So: parse the base, mutate the argv list, emit an override that restates
`command` in full. `.Args` stays a real argv array, provenance keeps working,
and an unset knob emits NOTHING AT ALL - which is the property the whole design
turns on. Auto_Bench.md 3 names the exact bug this prevents: "${VAR:-default}
taking the default on empty as well as unset - an empty override once silently
re-selected the control configuration."

The base file is read with yaml.safe_load and NOT interpolated: `${SGLANG_CTX
:-131072}` stays a literal string here and is interpolated by compose at `up`
time, exactly as before.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import sys

import yaml

SERVICE = "flashnext"

# (env selector, flag, kind). Kind "value" consumes the next argv element;
# "bool" is a store_true and takes none.
#
# Every flag here was proved present in THIS image's argparse by
# bench/sglang_flagcap.sh. `--gdn-mtp-cache-mode` is deliberately absent from
# the list: the probe reported it MISSING (it is a source-level feature of
# jpezzulli/sglang-rtxpro6000), so offering it would only produce boot failures.
KNOBS = [
    ("SGLANG_LINEAR_ATTN_DECODE_BACKEND", "--linear-attn-decode-backend", "value"),
    ("SGLANG_LINEAR_ATTN_PREFILL_BACKEND", "--linear-attn-prefill-backend", "value"),
    ("SGLANG_MAMBA_SSM_DTYPE", "--mamba-ssm-dtype", "value"),
    ("SGLANG_SPEC_DRAFT_QUANT", "--speculative-draft-model-quantization", "value"),
    ("SGLANG_SLEEP_ON_IDLE", "--sleep-on-idle", "bool"),
    ("SGLANG_DISABLE_FI_AUTOTUNE", "--disable-flashinfer-autotune", "bool"),
]

# Removed together for SPEC=off. Each takes a value, so the flag AND the
# following element go.
SPEC_FLAGS = [
    "--speculative-algorithm",
    "--speculative-num-steps",
    "--speculative-eagle-topk",
    "--speculative-num-draft-tokens",
]

# Container-side environment. These can never appear in `server_boot_args`
# (docker .Args holds argv only), which is the same blind spot that made LAZY
# unattributable in B-25 - so record_provenance() names each one separately.
ENV_PASSTHROUGH = [
    ("SGLANG_PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"),
    ("SGLANG_MAMBA_CONV_DTYPE_VAL", "SGLANG_MAMBA_CONV_DTYPE"),
    ("SGLANG_NUMA_BIND_V2_VAL", "SGLANG_NUMA_BIND_V2"),
    ("SGLANG_OMP_NUM_THREADS", "OMP_NUM_THREADS"),
]

# One switch for the persistent-JIT group. OFF by default: a warm JIT cache
# changes BOOT TIME, which bench/boot_time.sh publishes as a headline number.
JIT_ENV = {
    "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
    "TRITON_CACHE_DIR": "/cache/triton",
    "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer",
    "SGLANG_JIT_CACHE_DIR": "/cache/sglang/jit",
}


def sel(name: str) -> str:
    """A selector is set only if it is non-empty. Empty == unset == control."""
    return (os.environ.get(name) or "").strip()


def die(msg: str) -> None:
    sys.stderr.write(f"sglang_render_compose: {msg}\n")
    sys.exit(2)


def drop_flag(cmd: list[str], flag: str, takes_value: bool) -> list[str]:
    out, i = [], 0
    while i < len(cmd):
        if cmd[i] == flag:
            i += 2 if takes_value else 1
            continue
        out.append(cmd[i])
        i += 1
    return out


def set_flag(cmd: list[str], flag: str, value: str | None) -> list[str]:
    """Replace in place if present, else append.

    Never a blind append: if a future base compose gains one of these flags,
    appending would produce a duplicate and argparse would silently take the
    last one, so the artifact would name a value the server did not use.
    """
    if flag in cmd:
        i = cmd.index(flag)
        if value is None:
            return cmd
        cmd = list(cmd)
        cmd[i + 1] = value
        return cmd
    cmd = list(cmd)
    cmd.append(flag)
    if value is not None:
        cmd.append(value)
    return cmd


def main() -> None:
    if len(sys.argv) != 3:
        die("usage: sglang_render_compose.py <base-compose> <out-override>")
    base_path, out_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

    doc = yaml.safe_load(base_path.read_text())
    try:
        cmd = list(doc["services"][SERVICE]["command"])
    except (KeyError, TypeError):
        die(f"no services.{SERVICE}.command in {base_path}")
    cmd = [str(c) for c in cmd]

    spec = (os.environ.get("SPEC") or "on").strip()
    if spec == "off":
        for f in SPEC_FLAGS:
            cmd = drop_flag(cmd, f, takes_value=True)

    requested: dict[str, str] = {}
    for env_name, flag, kind in KNOBS:
        v = sel(env_name)
        if not v:
            continue
        if kind == "bool":
            if v in ("0", "false", "off", "no"):
                continue
            cmd = set_flag(cmd, flag, None)
            requested[flag] = "true"
        else:
            cmd = set_flag(cmd, flag, v)
            requested[flag] = v

    # --- refuse illegal combinations here, before any GPU time -------------
    #
    # These mirror _handle_linear_attn_backend in the image's server_args.py.
    # Reproducing them costs nothing and turns a four-minute boot failure into
    # an immediate message naming the fix.
    decode = requested.get("--linear-attn-decode-backend")
    ssm = requested.get("--mamba-ssm-dtype")
    if decode == "flashinfer" and ssm != "bfloat16":
        die(
            "--linear-attn-decode-backend flashinfer on SM100+ requires "
            "--mamba-ssm-dtype bfloat16 (the engine raises ValueError). "
            "Set SGLANG_MAMBA_SSM_DTYPE=bfloat16; this is one two-flag arm, "
            "not two independent knobs."
        )
    if decode == "flashkda":
        die("--linear-attn-decode-backend flashkda is prefill-only; the engine refuses it")
    if spec == "off" and requested.get("--speculative-draft-model-quantization"):
        die("--speculative-draft-model-quantization is meaningless with SPEC=off")

    svc: dict = {"command": cmd}

    env = {}
    for env_name, container_name in ENV_PASSTHROUGH:
        v = sel(env_name)
        if v:
            env[container_name] = v
    jit_dir = sel("SGLANG_JIT_CACHE_HOST_DIR")
    if jit_dir:
        env.update(JIT_ENV)
        svc["volumes"] = [{"type": "bind", "source": jit_dir, "target": "/cache"}]
    if env:
        svc["environment"] = env

    body = yaml.safe_dump({"services": {SERVICE: svc}}, sort_keys=False, width=10**6)
    header = (
        "# GENERATED by bench/sglang_render_compose.py - do not edit.\n"
        f"# base           {base_path.name}\n"
        f"# base_sha256    {hashlib.sha256(base_path.read_bytes()).hexdigest()}\n"
        f"# spec           {spec}\n"
        "# knobs          "
        + (", ".join(f"{k}={v}" for k, v in sorted(requested.items())) or "(none - control)")
        + "\n"
        "# container_env  "
        + (", ".join(f"{k}={v}" for k, v in sorted(env.items())) or "(none)")
        + "\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body)


if __name__ == "__main__":
    main()
