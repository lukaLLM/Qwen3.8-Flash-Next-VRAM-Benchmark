#!/usr/bin/env bash
# BOOT-TIME: how long does this engine take to become ready, N times over?
#
# WHY THIS IS A RESULT, not plumbing. Every other number here is steady-state
# throughput, which assumes the server is already up. On one workstation the
# server is NOT already up: you start it when you sit down, and you restart it
# every time you change a flag. That cost differs enormously by engine and by
# placement, and nothing in this study had measured it.
#
# WHAT IS MEASURED. Wall time from `docker compose up -d` to the container
# reporting HEALTHY - the same gate bench/lib.sh's engine_up waits on, so this
# is exactly the wait every other arm pays before it can start. Then one real
# generation, because "healthy" is a promise and FINDINGS.md 7 records a tier
# that passed health and died on the first request.
#
# THE PAGE CACHE IS THE VARIABLE, and it cannot be controlled here: dropping it
# needs root (benchmark/mmap_control_v2.sh:45 tries and says so). Run 1 after a
# reboot is cold; later runs read from RAM. That is WHY this repeats N times and
# reports EVERY attempt rather than an average - the spread between attempt 1
# and attempt N is the cache effect, and it is the interesting part.
#
#   ./bench/boot_time.sh --execute --plan-id FLASHNEXT-R1 <engine> [n=3]
#
# env: BOOT_N (default 3), plus any engine override (LLAMA_IMAGE, LOAD_MODE,
#      SPEC_TYPE, FREETOKEN_IMAGE, ENGINE_CTX ...) - boot cost depends on them,
#      so they are recorded in provenance beside the timings.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source bench/plan_gate.sh
plan_gate FLASHNEXT-R1 "$@"
set -- ${GATE_ARGV[@]+"${GATE_ARGV[@]}"}
source bench/lib.sh

ENGINE_ARG="${1:?usage: boot_time.sh <engine> [n]}"
N="${2:-${BOOT_N:-3}}"
bench_init "$ENGINE_ARG" fn_smoke

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO/artifacts/boot_${ENGINE}_${STAMP}"
mkdir -p "$OUT"

cleanup(){ thermal_watchdog_stop 2>/dev/null||true; thermal_stop 2>/dev/null||true; engine_down 2>/dev/null||true; }
trap cleanup EXIT INT TERM
bench_lock
thermal_start "boot_${ENGINE}"; trap cleanup EXIT INT TERM
thermal_watchdog_start "$OUT"; trap cleanup EXIT INT TERM

log "boot timing: $ENGINE x $N"
echo "[" > "$OUT/boots.json.tmp"
for i in $(seq 1 "$N"); do
  engine_down
  settle_gpu
  # Host page-cache state at the START of this attempt. It cannot be reset
  # without root, but it CAN be recorded, and it is what explains the spread.
  cached_mib="$(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)"
  avail_mib="$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)"

  # TWO clocks, because they answer different questions and the coarser one was
  # the only one measured at first:
  #
  #   compose_healthy_s - what `docker compose up -d --wait` returns on, i.e.
  #     the container's OWN healthcheck. Every service here polls every 15s, so
  #     this is QUANTISED TO 15s and cannot resolve a boot faster than that. It
  #     is also not the same test per engine: llama.cpp and SGLang curl /health,
  #     FreeToken asserts status==ok AND maintenance=='serving', which is
  #     strictly more than the other two prove.
  #
  #   health_ok_s - this script polling the SAME endpoint every 0.25s from the
  #     host. Same question for every engine, resolution-independent, and it is
  #     the number the comparison should be read from.
  probe="$OUT/probe_$i.txt"
  ( while :; do
      if curl -fsS --max-time 2 "$BASE/health" >/dev/null 2>&1; then date +%s.%N > "$probe"; break; fi
      sleep 0.25
    done ) & probe_pid=$!

  t0=$(date +%s.%N)
  engine_up
  t1=$(date +%s.%N)
  wait "$probe_pid" 2>/dev/null || true
  hok="$(cat "$probe" 2>/dev/null || echo "$t1")"
  health=$(awk -v a="$t0" -v b="$hok" 'BEGIN{printf "%.2f", b-a}')

  # Healthy is not the same as serving. Time one real 16-token generation too.
  g0=$(date +%s.%N)
  gen_ok=1
  curl -fsS --max-time 300 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}' \
    > "$OUT/gen_$i.json" 2>>"$OUT/boot.log" || gen_ok=0
  g1=$(date +%s.%N)

  boot=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  log "  health endpoint answered at ${health}s; compose called it healthy at ${boot}s"
  first=$(awk -v a="$g0" -v b="$g1" 'BEGIN{printf "%.2f", b-a}')
  log "attempt $i/$N: healthy in ${boot}s, first generation ${first}s (cached ${cached_mib} MiB, ok=$gen_ok)"
  [ "$i" -gt 1 ] && echo "," >> "$OUT/boots.json.tmp"
  printf '{"attempt":%d,"boot_s":%s,"health_ok_s":%s,"first_gen_s":%s,"gen_ok":%d,"host_cached_mib":%s,"host_available_mib":%s}' \
    "$i" "$boot" "$health" "$first" "$gen_ok" "$cached_mib" "$avail_mib" >> "$OUT/boots.json.tmp"
  docker logs "$CONTAINER" > "$OUT/server_$i.log" 2>&1 || true
done
echo "]" >> "$OUT/boots.json.tmp"
mv "$OUT/boots.json.tmp" "$OUT/boots.json"

record_provenance "$OUT"
record_runtime_state "$OUT" 2>/dev/null || true
thermal_watchdog_stop; thermal_stop "$OUT"
log "done -> $OUT"
uv run python - "$OUT/boots.json" <<'PY'
import json,sys,statistics as st
r=json.load(open(sys.argv[1]))
b=[x["boot_s"] for x in r]; h=[x.get("health_ok_s", x["boot_s"]) for x in r]
print(f"  health endpoint : {', '.join(f'{x:.1f}s' for x in h)}   median {st.median(h):.1f}s")
print(f"  compose healthy : {', '.join(f'{x:.1f}s' for x in b)}   median {st.median(b):.1f}s")
print(f"  the gap is the 15s healthcheck interval, not the engine")
PY
