#!/usr/bin/env bash
# Writes the fully resolved compose file for one arm, so that every server start
# has a primary record of the environment that produced it.
#
# Without this, an arm's configuration can only be reconstructed afterwards from
# the server's own /props and its log. That is enough for the fields the server
# reports, but it loses the rest: n-cpu-moe, loading mode, ubatch size and the
# tensor override target.
#
#   source benchmark/dump_compose.sh
#   VAR=... VAR=... dump_compose <evidence-dir>
#
# Give it the same variables you give to serve.sh. It resolves MODEL by itself
# if you do not set it.
#
# The compose file holds HF_TOKEN. Redaction is therefore not cosmetic, and the
# function refuses to write a file that still contains a secret.

redact() {
  sed -E 's/^([[:space:]]*[A-Za-z_]*(TOKEN|SECRET|PASSWORD|APIKEY|API_KEY)[A-Za-z_]*:).*/\1 <redacted>/I'
}

dump_compose() {
  local d="${1:-}"
  if [ -z "$d" ]; then
    echo "      dump_compose: no evidence directory given" >&2
    return 2
  fi
  mkdir -p "$d"

  local model="${MODEL:-}"
  if [ -z "$model" ]; then
    model=$(./scripts/serve.sh --print 2>/dev/null | awk '/^model/{print $2}')
  fi

  if ! MODEL="$model" docker compose -f docker/docker-compose.yaml config 2>&1 \
       | redact > "$d/compose.txt"; then
    echo "      dump_compose: compose config failed, see $d/compose.txt" >&2
    return 1
  fi

  # A token that survived the filter must never reach an evidence directory,
  # because these directories are published. Destroy the file instead.
  if grep -qE '(hf_[A-Za-z0-9]{20,}|[A-Za-z_]*(TOKEN|SECRET|PASSWORD|API_?KEY)[A-Za-z_]*:[[:space:]]*[^<[:space:]])' \
       "$d/compose.txt"; then
    rm -f "$d/compose.txt"
    echo "      dump_compose: a secret survived redaction - compose.txt deleted" >&2
    return 3
  fi

  return 0
}
