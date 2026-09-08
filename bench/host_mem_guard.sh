#!/usr/bin/env bash
# Host-RAM guard for arms with NO cgroup memory cap (the llama.cpp compose has
# none). Polls `free`; if host available drops below LIM GiB while CONTAINER is
# running, stops the container and exits 3. Exits 0 once the container has
# come and gone, or if it never appears within 15 min. Same rule ft4_gate.sh
# applies to FreeToken loads; here it protects the --load-mode mlock arms,
# which pin pages and could otherwise push the host into swap or OOM.
#   ./bench/host_mem_guard.sh <container> [lim_gib=6] [logfile]
C="${1:?container}"; LIM="${2:-6}"; LOG="${3:-/dev/null}"
start=$(date +%s); seen=0
while :; do
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$C"; then
    seen=1
    a=$(free -g | awk '/^Mem:/{print $7}')
    if [ "${a:-99}" -lt "$LIM" ]; then
      echo "[mem-guard $(date -u +%T)] host available ${a} GiB < ${LIM} GiB -> stopping $C" | tee -a "$LOG" >&2
      docker stop -t 5 "$C" >/dev/null 2>&1 || true
      exit 3
    fi
  elif [ "$seen" = 1 ] || [ $(( $(date +%s) - start )) -gt 900 ]; then
    exit 0
  fi
  sleep 5
done
