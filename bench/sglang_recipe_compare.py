#!/usr/bin/env python3
"""Compare the saved NVMe winner with the September SGLang cookbook recipes.

Dry run prints the plan. Execute with --execute --plan-id SGLANG-RECIPE-SEP12.
Prerequisites: both Docker images and pinned HF snapshots already downloaded.
One GPU, one fresh server per arm, one active request, 4096 output tokens,
three measured requests plus one excluded warmup at each context length.
Expect 1–2 hours including JIT/loading; evidence goes to artifacts/recipe_compare_*.
This is a synthetic context/decode comparison, not an accuracy benchmark.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess as sp
import threading
import time

import httpx
import yaml

REPO = Path(__file__).resolve().parents[1]
HF = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
RDX = "models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594"
NVDA = "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47"
# lmsysorg/sglang:dev-qwen38-next-local as pulled 2026-09-12. The digest first
# recorded here (9d278f74...) matched nothing on the Hub for this tag; this is
# the platform image's RepoDigest. The build-commit label assert below
# (4ccff141...) is the check that ties it to the cookbook's verified source.
IMAGE = "lmsysorg/sglang@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6"
NAME = "sglang-recipe-compare"
BASE = "http://127.0.0.1:8001"
ARMS = ["baseline", "radix_recipe16", "radix_context1", "nvidia_context1"]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def command(argv, **kwargs):
    return sp.check_output(argv, text=True, timeout=kwargs.pop("timeout", 60), **kwargs)


def memory():
    return {k: int(v.split()[0]) for k, v in
            (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}


def swapouts():
    return int(re.search(r"^pswpout (\d+)$", Path("/proc/vmstat").read_text(), re.M)[1])


def config(arm):
    if arm == "baseline":
        svc = yaml.safe_load((REPO / "docker/best.sglang.yaml").read_text())["services"]["flashnext"]
        return svc["image"], svc["command"], svc["environment"], True
    slots = 16 if arm == "radix_recipe16" else 1
    model = NVDA if arm.startswith("nvidia") else RDX
    argv = ["sglang", "serve", "--model-path", "/hf/hub/" + model,
            "--served-model-name", "Qwen3.8-Flash-Next", "--tp", "1"]
    if model == RDX:
        argv += ["--quantization", "modelopt_fp4"]
    for flag, value in {
        "fp4-gemm-backend": "flashinfer_cutlass", "moe-runner-backend": "flashinfer_cutlass",
        "page-size": 64, "mamba-track-interval": 64, "chunked-prefill-size": 4096,
        "context-length": 262144, "speculative-algorithm": "NEXTN",
        "speculative-num-steps": 3, "speculative-eagle-topk": 1,
        "speculative-num-draft-tokens": 4, "mamba-radix-cache-strategy": "extra_buffer_lazy",
        # Cookbook parity at 16 slots (48). At 1 slot the linear scale gives 3,
        # and NEXTN on this model needs 5 mamba states per request (mamba_ratio=5,
        # a model constant - results/FULL_CONTEXT.md). With 3 the first request
        # can never acquire its states: on the older image that is a boot-time
        # RuntimeError, on dev-qwen38-next-local it is a silent deadlock that
        # surfaces as the 600 s warmup timeout (radix_context1, 2026-09-12).
        "max-running-requests": slots, "max-mamba-cache-size": 48 if slots == 16 else 5 * slots,
        "mamba-ssm-dtype": "bfloat16", "reasoning-parser": "qwen3",
        "mem-fraction-static": .96, "host": "0.0.0.0", "port": 8000,
    }.items():
        argv += ["--" + flag, str(value)]
    # Avoid capturing unused admission sizes in the single-stream adaptation.
    if slots == 1:
        argv += ["--cuda-graph-max-bs-decode", "1"]
    argv += ["--ple-offload-embedding", "--enable-metrics"]
    return IMAGE, argv, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                         "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "1"}, False


def snapshot(out):
    p = sp.run(["docker", "logs", NAME], capture_output=True, text=True, timeout=30)
    (out / "server.log.after").write_text(p.stdout + p.stderr)
    p = sp.run(["docker", "inspect", NAME, "--format", "{{json .State}}"],
               capture_output=True, text=True, timeout=30)
    (out / "container_state.json").write_text(p.stdout)
    p = sp.run(["docker", "exec", NAME, "sh", "-c",
                "cat /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.swap.current"],
               capture_output=True, text=True, timeout=30)
    (out / "cgroup.txt").write_text(p.stdout + p.stderr)
    try:
        r = httpx.get(BASE + "/metrics", timeout=10)
        r.raise_for_status()
        (out / "metrics.txt").write_text(r.text)
    except httpx.HTTPError:
        pass


def monitor(out, stop, invalid):
    fields = "temperature.gpu,memory.used,power.draw,clocks.sm,clocks_event_reasons.hw_thermal_slowdown"
    initial_swap = swapouts()
    with (out / "telemetry.jsonl").open("a") as f:
        while not stop.is_set():
            try:
                gpu = command(["nvidia-smi", "--query-gpu=" + fields,
                               "--format=csv,noheader,nounits"]).strip().split(", ")
                mem = memory()
                row = dict(time=dt.datetime.now(dt.timezone.utc).isoformat(), gpu=gpu,
                           available_kib=mem["MemAvailable"], swapout_pages=swapouts() - initial_swap)
                f.write(json.dumps(row) + "\n"); f.flush()
                reason = None
                if float(gpu[0]) >= 90 or gpu[-1].strip() == "Active":
                    reason = "thermal_limit"
                if mem["MemAvailable"] < 6 * 1024**2:
                    reason = "host_memory_below_6GiB"
                # Global pswpout includes unrelated processes (notably a paused
                # checkpoint downloader). The container has memory.swap.max=0;
                # retain global activity for attribution, not a false failure.
                if reason:
                    invalid.append(reason)
                    save(out / "invalid.json", invalid)
                    sp.run(["docker", "stop", "-t", "5", NAME], capture_output=True, timeout=20)
                    return
            except Exception as exc:
                invalid.append("telemetry_failed: " + str(exc))
                save(out / "invalid.json", invalid)
                sp.run(["docker", "stop", "-t", "5", NAME], capture_output=True, timeout=20)
                return
            stop.wait(5)


def settle():
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        gpu = command(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu",
                       "--format=csv,noheader,nounits"]).strip().split(",")
        if float(gpu[0]) < 2000 and float(gpu[1]) < 45:
            return
        time.sleep(5)
    raise RuntimeError("GPU did not become idle and cool")


def assert_live(argv, info):
    actual = info.get("server_args", info if "model_path" in info else None)
    if not isinstance(actual, dict):
        raise RuntimeError("server_info has no server_args; cannot verify resolved options")
    aliases = {"tp": "tp_size", "fp4-gemm-backend": "fp4_gemm_runner_backend"}
    skip = {"model-path", "served-model-name", "host", "port", "enable-metrics"}
    for i, flag in enumerate(argv):
        if not flag.startswith("--") or flag[2:] in skip:
            continue
        key = aliases.get(flag[2:], flag[2:].replace("-", "_"))
        expected = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
        got = actual.get(key)
        if expected == "auto" and got is not None:
            continue
        if str(got).lower() != str(expected).lower():
            # NEXTN may resolve internally to EAGLE; record the actual algorithm.
            if key == "speculative_algorithm" and expected == "NEXTN" and got == "EAGLE":
                continue
            raise RuntimeError(f"resolved {key}={got!r}, requested {expected!r}")


def smoke(out):
    payload = {"model": "Qwen3.8-Flash-Next", "messages": [{"role": "user", "content":
        "Write a Python function triangular(n) returning the sum of integers 1 through n. "
        "Include an assert for n=10. Keep the answer short."}], "max_tokens": 256,
        "temperature": .7, "top_p": .8, "top_k": 20, "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": False}}
    r = httpx.post(BASE + "/v1/chat/completions", json=payload, timeout=180)
    r.raise_for_status(); data = r.json(); save(out / "smoke.json", data)
    content = data["choices"][0]["message"].get("content") or ""
    (out / "smoke.md").write_text(content)
    if "def triangular" not in content or "assert" not in content:
        raise RuntimeError("smoke output is missing the requested function/assert")


def run_arm(arm, root, args):
    out = root / arm; out.mkdir()
    image, argv, env, nvme = config(arm)
    snapshot_path = NVDA if arm.startswith("nvidia") else RDX
    if not (HF / "hub" / snapshot_path / "config.json").exists():
        raise RuntimeError("missing checkpoint " + snapshot_path)
    index = json.loads((HF / "hub" / snapshot_path / "model.safetensors.index.json").read_text())
    missing = [f for f in set(index["weight_map"].values()) if not (HF / "hub" / snapshot_path / f).is_file()]
    if missing:
        raise RuntimeError("checkpoint download incomplete: " + ", ".join(missing))
    image_data = json.loads(command(["docker", "image", "inspect", image]))[0]
    image_provenance = {k: image_data.get(k) for k in ["Id", "RepoDigests", "Created"]}
    labels = image_data.get("Config", {}).get("Labels") or {}
    image_provenance["source_revision"] = labels.get("org.opencontainers.image.revision")
    save(out / "image.json", image_provenance)
    if arm != "baseline" and labels.get("ai.sglang.build.commit") != "4ccff141dbe992794f9da6c3aa23535b4f72000d":
        raise RuntimeError("recipe image does not identify the cookbook's verified source commit")
    launch = ["docker", "run", "-d", "--name", NAME, "--label", "benchmark=sglang-recipe-compare",
              "--gpus", "all", "--ipc", "host", "--shm-size", "32g", "--memory", "86g",
              "--memory-swap", "86g", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
              "-p", "127.0.0.1:8001:8000", "-v", str(HF) + ":/hf:ro"]
    if nvme:
        launch += ["--security-opt", "seccomp=" + str(REPO / "docker/sglang/seccomp-io_uring.json")]
    for key, val in env.items():
        launch += ["-e", key + "=" + str(val)]
    launch += ["--entrypoint", argv[0], image_data["Id"], *argv[1:]]
    save(out / "launch.json", launch); save(out / "environment.json", env)
    settle()
    if not nvme and memory()["MemAvailable"] < 64 * 1024**2:
        raise RuntimeError("pinned-host recipe requires 64 GiB available before loading")
    if shutil.disk_usage(REPO).free < 40 * 1024**3:
        raise RuntimeError("less than 40 GiB free disk")
    stop = threading.Event(); invalid = []
    created = False
    watcher = None
    try:
        (out / "container_id.txt").write_text(command(launch, timeout=120)); created = True
        limits = command(["docker", "inspect", NAME, "--format", "{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}"])
        if limits.split() != [str(86 * 1024**3)] * 2:
            raise RuntimeError("container memory/no-swap limits did not apply")
        (out / "memory_limits.txt").write_text(limits)
        watcher = threading.Thread(target=monitor, args=(out, stop, invalid), daemon=True); watcher.start()
        deadline = time.monotonic() + 1800
        with httpx.Client(timeout=5) as client:
            while time.monotonic() < deadline:
                if invalid:
                    raise RuntimeError(str(invalid))
                if command(["docker", "inspect", NAME, "--format", "{{.State.Running}}"]).strip() != "true":
                    raise RuntimeError("server exited during boot")
                try:
                    if client.get(BASE + "/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(5)
            else:
                raise RuntimeError("server health deadline exceeded")
        info = httpx.get(BASE + "/get_server_info", timeout=30).json()
        save(out / "server_info.json", info)
        assert_live(argv, info)
        smoke(out)
        snapshot(out)
        boot = (out / "server.log.after").read_text()
        pools = re.findall(r"max_total_num_tokens=(\d+)", boot)
        if not pools:
            raise RuntimeError("cannot determine actual KV pool")
        pool = int(pools[-1]); save(out / "pool.json", {"tokens": pool})
        print(f"{arm}: healthy, KV pool {pool}", flush=True)
        # A fixed synthetic output budget permits greedy; use the same sampler,
        # corpus, local tokenizer and random seed for every arm.
        for isl in args.lengths:
            cell = out / f"isl_{isl}"; cell.mkdir()
            if isl + args.osl + 512 > min(pool, 262144):
                save(cell / "status.json", {"status": "skipped_pool_limit", "pool": pool})
                continue
            if invalid:
                raise RuntimeError(str(invalid))
            ap = [str(REPO / ".venv/bin/aiperf"), "profile", "--model", "Qwen3.8-Flash-Next",
                  "--url", BASE, "--endpoint-type", "chat", "--streaming",
                  "--tokenizer", str(HF / "hub" / RDX), "--prompt-corpus", "sonnet",
                  "--cache-bust", "system_prefix", "--reset-kv-cache", "--reset-kv-cache-path", "/flush_cache",
                  "--isl", str(isl), "--isl-stddev", "0", "--osl", str(args.osl), "--osl-stddev", "0",
                  "--extra-inputs", "ignore_eos:true", "--extra-inputs", f"min_tokens:{args.osl}",
                  "--extra-inputs", "temperature:0", "--extra-inputs", "top_p:1",
                  "--extra-inputs", '{"chat_template_kwargs":{"enable_thinking":false}}',
                  "--concurrency", "1", "--request-count", str(args.repeats), "--warmup-request-count", "1",
                  "--random-seed", "42", "--export-outputs-json", "--output-artifact-dir", str(cell), "--ui", "simple"]
            save(cell / "command.json", ap)
            (cell / "vmstat.before").write_text(Path("/proc/vmstat").read_text())
            print(f"{arm}: ISL={isl} OSL={args.osl}", flush=True)
            with (cell / "client.log").open("w") as log:
                p = sp.Popen(ap, cwd="/tmp", stdout=log, stderr=sp.STDOUT, start_new_session=True)
                try:
                    p.wait(timeout=1800)
                finally:
                    if p.poll() is None:
                        os.killpg(p.pid, signal.SIGTERM)
                        try:
                            p.wait(timeout=15)
                        except sp.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL)
                            p.wait()
            snapshot(cell)
            (cell / "vmstat.after").write_text(Path("/proc/vmstat").read_text())
            save(cell / "status.json", {"status": "completed" if p.returncode == 0 else "client_failed", "exit": p.returncode})
            if p.returncode != 0:
                raise RuntimeError(f"client failed at {isl}; see client.log")
            data = json.loads((cell / "profile_export_aiperf.json").read_text())
            osl = data.get("output_sequence_length") or {}
            # Tolerate ONE token of tokenizer disagreement between aiperf and the
            # server: r3's baseline measured 4095/4096/4096 against a plan of
            # 4096 and was rejected, throwing away a valid 3-request cell. The
            # study's own post_gate accepts the same slack (FreeToken 127/128).
            # Anything beyond one token is still a real length failure.
            if (osl.get("count") != args.repeats
                    or osl.get("min", 0) < args.osl - 1 or osl.get("max", 0) > args.osl):
                save(cell / "status.json", {"status": "invalid_output_length_or_count", "observed": osl})
                raise RuntimeError("measured output lengths/count differ from plan")
            outputs = json.loads((cell / "outputs.json").read_text()).get("data", [])
            save(cell / "output_capture_check.json", {
                "saved_responses": len(outputs),
                "empty_responses": sum(not (r.get("response_text") or "").strip() for r in outputs),
                "quality_verdict": "not_applicable_forced_ignore_eos",
            })
            if len(outputs) < args.repeats:
                raise RuntimeError("missing saved output responses")
    finally:
        if created:
            try:
                snapshot(out)
            finally:
                stop.set()
                if watcher:
                    watcher.join(timeout=30)
                sp.run(["docker", "rm", "-f", NAME], capture_output=True, timeout=60)
                time.sleep(30)


def summarize(root):
    rows = []
    for arm in ARMS:
        for cell in sorted((root / arm).glob("isl_*"), key=lambda p: int(p.name[4:])):
            status = json.loads((cell / "status.json").read_text()) if (cell / "status.json").exists() else {}
            row = {"arm": arm, "isl": int(cell.name[4:]), **status}
            path = cell / "profile_export_aiperf.json"
            if path.exists():
                data = json.loads(path.read_text())
                for key in ["input_sequence_length", "output_sequence_length", "time_to_first_token",
                            "output_token_throughput_per_user", "error_request_count"]:
                    row[key] = data.get(key)
            if (root / arm / "invalid.json").exists():
                row["invalid"] = json.loads((root / arm / "invalid.json").read_text())
            rows.append(row)
    save(root / "summary.json", rows)
    lines = ["# SGLang recipe comparison", "", "One active request; fixed synthetic output length; greedy; cold prefixes.",
             "Warmups excluded. Prefill below is average input tokens / average TTFT, including API overhead.",
             "", "| Arm | Input target | Status | TTFT median (s) | Effective prefill (tok/s) | Decode median (tok/s) | Decode min–max |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        ttft = row.get("time_to_first_token") or {}
        decode = row.get("output_token_throughput_per_user") or {}
        isl = row.get("input_sequence_length") or {}
        clean = row.get("status") == "completed" and not row.get("invalid")
        fmt = lambda x: f"{x:.2f}" if x is not None and clean else "—"
        prefill = isl.get("avg", 0) * 1000 / ttft["avg"] if ttft.get("avg") else None
        median = ttft["p50"] / 1000 if "p50" in ttft else None
        status = "VOID " + str(row["invalid"]) if row.get("invalid") else row.get("status", "incomplete")
        lines.append(f"| {row['arm']} | {row['isl']} | {status} | {fmt(median)} | {fmt(prefill)} | {fmt(decode.get('p50'))} | {fmt(decode.get('min'))}–{fmt(decode.get('max'))} |")
    for path in root.glob("*/failure.json"):
        lines += ["", f"{path.parent.name}: {json.loads(path.read_text())['error']}"]
    lines += ["", "Skipped pool limits are admission estimates, not failed requests. Saved forced-length text is not an accuracy score.",
              "GPU power, temperature, clocks, host availability and swap are in each arm's telemetry.jsonl; DCGM profiling is not collected.",
              "", "Recompute with: `.venv/bin/python -c 'from pathlib import Path; from bench.sglang_recipe_compare import summarize; summarize(Path(\"" + str(root) + "\"))'`", ""]
    (root / "SUMMARY.md").write_text("\n".join(lines))


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--plan-id")
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    ap.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768, 65536, 131072, 253952])
    ap.add_argument("--osl", type=int, default=4096)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", action="store_true", help="add previously unrun arms to --out without overwriting evidence")
    args = ap.parse_args()
    if not args.execute:
        print(__doc__)
        for arm in args.arms:
            image, argv, env, nvme = config(arm)
            print(json.dumps(dict(arm=arm, image=image, command=argv, environment=env), indent=2))
        return
    if args.plan_id != "SGLANG-RECIPE-SEP12":
        ap.error("--plan-id SGLANG-RECIPE-SEP12 required")
    with open("/tmp/queue_rest.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if command(["docker", "ps", "--format", "{{.Names}}"]).strip():
            raise RuntimeError("running containers found; resolve before benchmarking")
        root = (args.out or REPO / "artifacts" / ("recipe_compare_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
        if args.resume:
            if not args.out:
                ap.error("--resume requires --out")
            old = json.loads((root / "plan.json").read_text())
            if any(old[k] != getattr(args, k) for k in ["lengths", "osl", "repeats"]):
                ap.error("resumed arms must use the original workload")
            if any((root / arm).exists() for arm in args.arms):
                ap.error("an arm already has evidence; use a fresh output directory to repeat it")
        else:
            root.mkdir(parents=True, exist_ok=False)
            save(root / "plan.json", {**vars(args), "out": str(root), "note": "c=1 synthetic, greedy, fixed output; not a quality benchmark"})
        save(root / ("invocation_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"),
             {**vars(args), "out": str(root)})
        if not (root / "hardware.txt").exists():
            (root / "hardware.txt").write_text(command(["nvidia-smi"]) + "\n" + command(["free", "-h"]))
        print(root, flush=True)
        for arm in args.arms:
            try:
                run_arm(arm, root, args)
            except Exception as exc:
                print(f"{arm}: FAILED: {exc}", flush=True)
                (root / arm).mkdir(exist_ok=True)
                save(root / arm / "failure.json", {"error": str(exc)})
            summarize(root)


if __name__ == "__main__":
    main()
