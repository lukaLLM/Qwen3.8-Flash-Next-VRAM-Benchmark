#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Model downloader. Put models in the two lists below and run it.
#
#   ./scripts/download_models.sh            check for updates, ask, download in tmux
#   ./scripts/download_models.sh --check     report what is missing/updated, exit
#   ./scripts/download_models.sh --fg        run here instead of in tmux (debugging)
#   ./scripts/download_models.sh --yes       skip the prompt, fetch everything stale
#   ./scripts/download_models.sh --conn 4    connections per file (default from
#                                            the physical NIC's link speed)
#
# STOPPING IS SAFE. Selections are fetched one at a time, files within a
# selection one at a time, and every byte is written straight into the HF cache
# and resumed with aria2c -c. Kill it whenever you like and re-run: finished
# files are skipped, the file that was in flight picks up at its current offset.
#
# STOPPING IT, QUICKLY
#
#   pkill -x aria2c                     from any shell, attached or not
#   tmux kill-session -t hf-download    stops the whole queue, not just this file
#   Ctrl-C                              only if you are attached to the window
#
# Signal the process; do not try to fake a keystroke. `tmux send-keys C-c` does
# not reliably reach aria2c, and Ctrl-C is useless when you are not attached.
#
# Use `pkill -x`, never `pkill -f`. -x matches the process NAME. -f matches the
# whole command line, so every shell whose command line merely mentions aria2c
# matches too - including the one you are typing the pkill into, which then
# kills itself and leaves the download running. `pgrep -f` misreports the same
# way: it finds itself and tells you aria2c is still up after it has exited.
#
# Nothing is lost at any point. aria2c writes the segment map on SIGINT and
# SIGTERM and flushes it periodically anyway, so even SIGKILL costs only the
# last few seconds of transfer. Re-run the script to continue.
#
# TMUX, THE WHOLE OF IT
#
#   ./scripts/download_models.sh    starts session 'hf-download' and attaches you
#   Ctrl-B then D                   detach; the download keeps running
#   tmux attach -t hf-download      reattach later
#   tmux ls                         what is running
#   ./scripts/download_models.sh    run it again and it reattaches to the
#                                   existing session instead of starting a
#                                   second one, so this is also "show me"
#
# Detaching is not stopping. The session survives a dropped ssh connection, a
# locked or sleeping laptop, and logging out entirely - this box has lingering
# enabled (`loginctl enable-linger`), so systemd keeps user processes alive with
# no login session at all. Killing the tmux server is what stops it.
#
# Run inside an existing tmux window and the script skips the handoff and runs
# inline in that window, which is fine: it is still tmux, so it still survives.
#
# It does not use `hf download`, on purpose. huggingface_hub downloads into a
# process-unique temp file and passes no resume offset to http_get, so every
# interruption restarts every unfinished file from zero - on the xet path and on
# plain HTTP alike. That is deliberate upstream (see huggingface_hub#4228, which
# traded resume away to avoid cache corruption on filesystems with broken
# flock), and it is not something a wrapper can fix. Measured here 2026-08-27:
# ten hours of transfer left 46 GB on disk that the next run would have ignored.
#
# Re-running is safe and cheap. One API call per repo asks the Hub for its
# current commit and the size of every file in it, and each spec is called
# complete only when every file it wants is on disk at that size. So it catches
# a new revision, a half-finished transfer and a corrupt file, without moving
# any bytes to find out. Statuses:
#
#   current   nothing to do (a note is added if the repo moved to a new commit
#             but the files are byte-identical, which costs only a re-link)
#   MISSING   not in the cache at all
#   PARTIAL   some files unfinished or the wrong size
#   NOMATCH   the pattern matches no file in the repo (check for a typo)
#   ERROR     the Hub could not be reached, or the repo needs a token
#
# Gotchas worth knowing:
#   - A partial file's SIZE MEANS NOTHING while it is being fetched. aria2 writes
#     N segments at different offsets, so `ls -l` shows nearly the full size on a
#     file that is 30% done. Measure allocated blocks instead:
#       stat -c '%b*%B' f | bc      (or trust aria2c's own percentage)
#   - blobs/<sha>.incomplete.aria2 is the segment map. DO NOT DELETE IT. Without
#     it aria2c treats the apparent size as a contiguous prefix, resumes past
#     every hole, and produces a right-length wrong-contents file. To discard a
#     partial, delete the .incomplete AND its .aria2 together.
#   - A slow connection is not a failed one. --timeout only fires when a
#     connection stops responding; one that degrades to 13 KiB/s is never
#     retried. That is what --lowest-speed-limit (MIN_SPEED, default 200K) is
#     for. Seen here: connections opened during a bad stretch stayed at 20 KiB/s
#     for hours while a fresh connection to the same URL got 10.4 MB/s. If it
#     ever looks stuck at CN:1, stop and restart - fresh sockets fix it.
#   - A file that already has a snapshot symlink is treated as done, so one
#     present at the wrong size is unlinked before downloading. That is the only
#     case where this script deletes anything, and it says so when it does.
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ============================ EDIT THIS ======================================
# GGUF repos: "repo:PATTERN" pulls only *PATTERN*.gguf. PATTERN is usually a
# quant name, but any filename fragment works.
#
# Qwen3.8-Flash-Next ships one directory per quant, each split into 3-4 shards.
# The quant name alone is a safe pattern - it appears in both the directory and
# every shard filename, and no two quant names are prefixes of each other.
#
# Sizes are total on-disk, from Unsloth's own table. This box has 96 GB VRAM +
# 91 GB RAM = ~187 GB, so every quant below fits in memory; the choice is about
# how much of it has to live in system RAM. Q4_K_XL is the quality reference,
# IQ4_XS is the one most likely to leave room for a long context.
#
#   UD-Q4_K_XL  111.3 GB   mean KLD 0.045   93.5% top-1
#   UD-IQ4_XS    93.7 GB   mean KLD 0.079   91.1% top-1   <- the one we run
#   UD-Q3_K_XL   90.0 GB   mean KLD 0.100   90.4% top-1
#   UD-IQ3_XXS   82.0 GB   mean KLD 0.157   87.6% top-1
#   UD-Q2_K_XL   78.9 GB   mean KLD 0.213   85.2% top-1
#   UD-IQ1_M     74.5 GB   mean KLD 0.302   82.4% top-1
#   UD-IQ1_S     72.5 GB   mean KLD 0.375   80.2% top-1
#
# Two quants, 205 GB together, meant to be fetched in one overnight run.
#
#   UD-IQ4_XS   the one we serve. 87.2 GiB on disk against ~92 GiB of usable
#               VRAM, so it is the largest quant that could plausibly sit on
#               the GPU with room left for context.
#   UD-Q4_K_XL  the quality reference for Phase 4. At 103.7 GiB it can never be
#               GPU-resident, so it only ever measures the offloaded regime.
#
# Both are split into shards where shard 1 is a ~10 MB index and the weights
# live in the shards after it.
#
# Nothing here moves bytes on its own: the script prompts, and the answer can be
# a single number. Interrupting costs nothing either way - a finished shard is
# symlinked into the snapshot, and an in-flight one resumes at its offset.
GGUF_MODELS=(
  "unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS"    #  93.7 GB
  "unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL"   # 111.3 GB
  # "unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q3_K_XL" #  90.0 GB, the one quant with
  #   real VRAM headroom (83.8 GiB) if Phase 2 ever needs a fully GPU-resident arm
  # "unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q2_K_XL" #  78.9 GB, the low end
)

# Full repos: everything in them.
#
# The BF16 safetensors are 355 GB and are only worth pulling if we end up
# converting our own GGUF - the one way to get the 4B MTP module into a GGUF,
# since the upstream converter drops it. Left commented so nobody starts a
# 355 GB transfer by answering "all" at the prompt.
HF_MODELS=(

  # The BF16 safetensors are 355 GB and are only worth pulling if we end up
  # converting our own GGUF - see the note above.
  # "Qwen/Qwen3.8-Flash-Next"
)
# =============================================================================

SESSION="hf-download"
LOCKFILE="/tmp/hf-download.lock"
LOGDIR="${XDG_CACHE_HOME:-$HOME/.cache}/hf-download-logs"
# No stall watchdog any more. It existed to kill and restart a transfer that had
# gone quiet, which was only ever safe under the assumption that a restart
# resumes. It does not, so the watchdog was a way to lose hours of transfer to a
# few quiet minutes. aria2c retries forever in place instead, keeping the bytes.

MODE=auto            # auto | fg | check
ASSUME_YES=0
CONN="${CONN:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --check)  MODE=check; shift ;;
    --fg)     MODE=fg; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    # --jobs is the old name from when this ran several repos at once. It is
    # kept so old invocations do not error out, but it means connections now.
    --conn|--jobs) CONN="$2"; shift 2 ;;
    # Print the header block, however long it grows: skip the opening rule and
    # stop at the closing one.
    -h|--help)
      awk 'NR>2 && /^# ----/{exit} NR>2{sub(/^#[[:space:]]?/,""); print}' \
        "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done

# --- flat spec list; index into this is what the selection prompt returns ----
SPECS=()
for s in "${GGUF_MODELS[@]}"; do SPECS+=("gguf|$s"); done
for s in "${HF_MODELS[@]}";   do SPECS+=("full|$s"); done
if [ ${#SPECS[@]} -eq 0 ]; then
  echo "No models listed. Add some to GGUF_MODELS / HF_MODELS in $0."
  exit 0
fi

# --- environment: token, PATH, dependency guards -----------------------------
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ] \
   && grep -qE '^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=' .env; then
  set -a; . ./.env; set +a
fi

# Prefer this repo's venv: all we need from it is huggingface_hub, for the Hub
# metadata (file list, sizes, sha256, commit). The bytes come over aria2c, so
# neither the `hf` CLI nor hf_xet is on the critical path any more.
[ -x .venv/bin/python3 ] && export PATH="$REPO_ROOT/.venv/bin:$PATH"

HF_PY="$(command -v python3)"
if ! "$HF_PY" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "Missing dependency: huggingface_hub is not importable from $HF_PY."
  echo "  uv venv .venv && uv add huggingface-hub"
  exit 127
fi
if ! command -v aria2c >/dev/null 2>&1; then
  echo "Missing dependency: aria2c not found. It is what makes a stopped"
  echo "download resumable - see the note at the top of this script."
  echo "  sudo apt install aria2"
  exit 127
fi

unset HF_HUB_ENABLE_HF_TRANSFER || true

# --- how many connections per file -------------------------------------------
link_mbits() {
  local best=0 s
  for i in /sys/class/net/*/; do
    [ "$(basename "$i")" = lo ] && continue
    # Physical NICs only, i.e. the ones backed by a real device. Every veth that
    # docker and k3s create reports speed=10000, so without this test a box with
    # a 1 Gb/s NIC behind eighteen veth pairs - which is exactly this box - reads
    # as a 10 Gb/s link and gets tuned for bandwidth it does not have.
    [ -e "$i/device" ] || continue
    [ "$(cat "$i/operstate" 2>/dev/null)" = up ] || continue
    s="$(cat "$i/speed" 2>/dev/null || echo 0)"
    [[ "$s" =~ ^[0-9]+$ ]] && [ "$s" -gt "$best" ] && best="$s"
  done
  echo "$best"
}
# Connections aria2 opens per file. Files are fetched one at a time, so this is
# the only concurrency there is, and it is per-file rather than across files on
# purpose: a file only becomes permanent when it finishes, so spreading the link
# across several files just means several partial files and nothing banked.
if [ -z "$CONN" ]; then
  mb="$(link_mbits)"
  if   [ "$mb" -ge 10000 ]; then CONN=16
  elif [ "$mb" -ge 1000  ]; then CONN=8
  else                           CONN=4
  fi
  echo "  link: ${mb:-unknown} Mb/s -> $CONN connections per file"
fi
[ "$CONN" -lt 1 ] && CONN=1
[ "$CONN" -gt 16 ] && CONN=16

# --- what is missing, what changed ------------------------------------------
# Local commit sha lives in the cache at refs/main; the Hub's is one API call.
# For a GGUF spec, having the right sha is not enough - the pattern must
# actually be present, or adding a new quant to the list would read as "current".
#
# Fields are separated by \x1f, not tab: tab is an IFS *whitespace* character, so
# `read` silently collapses runs of them and an empty pattern field would shift
# every column after it.
SEP=$'\x1f'
# A spec is complete when every file it asks for is present as a blob named by
# its sha256, at the size the Hub reports. Deliberately NOT decided by comparing
# refs/main against the Hub's commit: HF blobs are content-addressed, so a repo
# that gets a new commit for a README edit - which this one does often - leaves
# every GGUF byte-identical and already on disk. The old sha comparison reported
# UPDATED for a fully downloaded 87 GB quant and offered to fetch it again.
#
# A genuinely changed file has a different sha256, so it simply shows up as a
# missing blob. Content addressing makes "updated" and "missing" the same
# question, which is why UPDATED no longer exists as a status.
#
# Counting .incomplete blobs would not work either: blobs are shared across
# every spec in a repo, so an unfinished quant nobody asked for would mark all
# of them dirty forever.
#
# One more thing the blob name has to account for: not every file in a repo is an
# LFS file. Anything small enough - config.json, chat_template.jinja, the 192
# layer-*.complete.json sidecars in the NVFP4 repo - is stored in git proper and
# has no sha256 at all. The cache names those blobs by their **git blob sha1**
# instead, which the API returns as blob_id. Confirmed against this box's cache:
# LFS blobs are 64 hex characters, plain ones 40.
#
# Every GGUF spec globs *PATTERN*.gguf, so until the first full-repo spec landed
# this branch was never taken and keying on lfs.sha256 alone was harmless. For a
# safetensors repo it is not: half the files have no sha256, so they never count
# as present, the spec reports PARTIAL forever, and - worse - the manifest in
# run_job drops them, leaving a checkpoint with weights and no config.json.
status_tsv() {
  printf '%s\n' "${SPECS[@]}" | "$HF_PY" -c '
import os, sys, fnmatch
from huggingface_hub import HfApi
api = HfApi()
S = "\x1f"

def digest(s):
    """(blob name, kind) - sha256 for LFS files, git blob sha1 for the rest."""
    lfs = s.lfs
    sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if sha:
        return sha, "sha256"
    return (getattr(s, "blob_id", None) or ""), "sha1"
home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
info_cache = {}
for i, line in enumerate(sys.stdin.read().splitlines(), 1):
    kind, spec = line.split("|", 1)
    if kind == "gguf" and ":" in spec:
        repo, pat = spec.split(":", 1)
    else:
        repo, pat = spec, ""
    d = os.path.join(home, "hub", "models--" + repo.replace("/", "--"))
    ref = os.path.join(d, "refs", "main")
    local = open(ref).read().strip() if os.path.isfile(ref) else ""
    try:
        if repo not in info_cache:
            info_cache[repo] = api.model_info(repo, files_metadata=True)
        info = info_cache[repo]
    except Exception as e:
        print(S.join([str(i), repo, pat, "ERROR", "0", type(e).__name__])); continue
    remote = info.sha or ""
    glob_pat = f"*{pat}*.gguf" if pat else "*"
    want = [s for s in info.siblings if fnmatch.fnmatch(s.rfilename, glob_pat)]
    present = ok = size = 0
    for s in want:
        sha, _kind = digest(s)
        p = os.path.join(d, "blobs", sha) if sha else ""
        try:
            n = os.path.getsize(p)
        except OSError:
            continue                         # not there at all
        present += 1
        if s.size is None or n == s.size:
            ok += 1; size += n
    if not want:
        st, note = "NOMATCH", "no file matches this pattern"
    elif present == 0:
        st, note = "MISSING", ""
    elif ok < len(want):
        st, note = "PARTIAL", f"{len(want) - ok} of {len(want)} files unfinished"
    else:
        st = "current"
        note = "" if local == remote else "new commit, same files - re-run to relink"
    print(S.join([str(i), repo, pat, st, str(size), note]))
'
}

human() { awk -v b="$1" 'BEGIN{if(b<1)print "-";else if(b<1073741824)printf "%.0f MB",b/1048576;else printf "%.1f GB",b/1073741824}'; }

echo "Checking ${#SPECS[@]} repos against the Hub ..."
TSV="$(status_tsv)"

printf '\n  %-3s %-58s %-8s %9s\n' "#" "repo" "status" "local"
printf '  %s\n' "$(printf '%.0s-' {1..81})"
FETCH=()
while IFS="$SEP" read -r idx repo pat st size err; do
  [ -z "$idx" ] && continue
  label="$repo"; [ -n "$pat" ] && label="$repo:$pat"
  # Truncate from the left: the tail of a long GGUF label is the part that
  # distinguishes it, the org prefix is not.
  [ "${#label}" -gt 58 ] && label=".."${label: -56}
  printf '  %-3s %-58s %-8s %9s %s\n' "$idx" "$label" "$st" "$(human "$size")" "$err"
  # ERROR and NOMATCH are not fetchable: one could not be checked, the other
  # asks for a file the repo does not have. Downloading either is a no-op.
  case "$st" in current|ERROR|NOMATCH) ;; *) FETCH+=("$idx") ;; esac
done <<< "$TSV"
echo

n_miss=$(grep -c "${SEP}MISSING${SEP}" <<< "$TSV" || true)
n_par=$(grep -c "${SEP}PARTIAL${SEP}" <<< "$TSV" || true)
n_cur=$(grep -c "${SEP}current${SEP}" <<< "$TSV" || true)
echo "  ${#FETCH[@]} to fetch ($n_miss missing, $n_par partial, $n_cur current)."

[ "$MODE" = check ] && exit 0
if [ ${#FETCH[@]} -eq 0 ]; then
  echo "  Nothing to do."
  exit 0
fi

# --- choose ------------------------------------------------------------------
SELECT="${HF_DL_SELECT:-}"
if [ -z "$SELECT" ]; then
  if [ "$ASSUME_YES" = 1 ]; then
    SELECT="$(IFS=,; echo "${FETCH[*]}")"
  elif [ ${#FETCH[@]} -eq 1 ]; then
    # Accept the index too, not just y. The prompt names an entry by number, so
    # typing that number is the obvious answer and used to silently do nothing.
    read -rp "  Download ${FETCH[0]}? [y/N] " a
    case "$a" in
      [Yy]*|a|A|all|"${FETCH[0]}") ;;
      *) echo "  Nothing downloaded."; exit 0 ;;
    esac
    SELECT="${FETCH[0]}"
  else
    read -rp "  Download which?  [a]ll / [n]one / numbers e.g. 1,3 : " a
    case "$a" in
      a|A|all|"") SELECT="$(IFS=,; echo "${FETCH[*]}")" ;;
      n|N|none)   echo "  Nothing downloaded."; exit 0 ;;
      *)          SELECT="$(tr -d ' ' <<< "$a")" ;;
    esac
  fi
fi
[ -z "$SELECT" ] && { echo "  Nothing selected."; exit 0; }

# --- hand off to tmux --------------------------------------------------------
if [ "$MODE" = auto ] && [ -z "${TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo
    echo "tmux is not installed, and it is what keeps a long download alive when"
    echo "the SSH session drops. Install it, then run this again:"
    echo "  sudo apt install tmux"
    echo
    echo "Or run in the foreground (dies with the terminal):"
    echo "  ./scripts/download_models.sh --fg"
    exit 127
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "  A download is already running. Attaching - Ctrl-B D to detach."
    exec tmux attach -t "$SESSION"
  fi
  echo "  Starting in tmux session '$SESSION'. Ctrl-B D to detach."
  # tmux starts the child from the server's environment, not this shell's, so
  # every knob has to be written into the command line the way HF_DL_SELECT is.
  # Without this, `STALL_SECS=1800 ./scripts/download_models.sh` would run with
  # the default 300 inside tmux and say nothing about it.
  ENV_PREFIX="HF_DL_SELECT='$SELECT' CONN='$CONN' MIN_SPEED='${MIN_SPEED:-200K}'"
  # ionice/nice so a download does not starve benchmarks of disk I/O.
  tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
    "$ENV_PREFIX ionice -c3 nice -n10 '$REPO_ROOT/scripts/download_models.sh' --fg --yes; \
     echo; echo 'Done. Press any key to close.'; read -rn1"
  exec tmux attach -t "$SESSION"
fi

# --- single instance ---------------------------------------------------------
# Nothing else stops two runs colliding now that systemd is gone, and two hf
# processes on one repo cache is exactly what the stall watchdog cannot reason
# about: they share a directory, so they share a progress measurement.
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "  Another download already holds $LOCKFILE."
  echo "  Attach to it with:  tmux attach -t $SESSION"
  if command -v fuser >/dev/null 2>&1; then
    echo "  Held by:"
    fuser -v "$LOCKFILE" 2>&1 | sed 's/^/    /'
  fi
  exit 1
fi

# --- build the queue ---------------------------------------------------------
# One entry per selection, in list order, run one at a time. An earlier version
# merged every selection of a repo into a single hf invocation with repeated
# --include flags, to keep two hf processes off one cache directory. That is a
# real problem but this is the wrong trade: hf_xet does not resume, so bytes
# only become permanent when a file finishes and is symlinked into the snapshot.
# Fetching everything at once means every file is partial until the very end,
# and one interruption discards all of it. Measured on 2026-08-26: two quants
# merged into one call left 46 GB on disk after ten hours with not one shard
# finished, so an interruption at hour ten would have kept nothing.
#
# Serially, each finished file is banked and a restart skips it.
declare -a QUEUE=()
declare -a CHOSEN=()
for idx in ${SELECT//,/ }; do
  line="$(awk -F"$SEP" -v i="$idx" '$1==i{print; exit}' <<< "$TSV")"
  [ -z "$line" ] && { echo "  no such entry: $idx" >&2; continue; }
  repo="$(cut -d"$SEP" -f2 <<< "$line")"; pat="$(cut -d"$SEP" -f3 <<< "$line")"
  CHOSEN+=("$repo|$pat")
  QUEUE+=("$repo|$pat")
done
[ ${#QUEUE[@]} -eq 0 ] && { echo "  Nothing selected."; exit 0; }

# --- drop files whose size disagrees with the Hub ----------------------------
# A file that already has a snapshot symlink is treated as done, so a truncated
# or corrupt one is never repaired by re-running - it would just report PARTIAL
# forever. Unlink exactly those files (and their blobs) so they are fetched
# again. An interrupted download does not land here: it leaves a .incomplete
# blob and no symlink, and aria2c -c continues that file from its current size.
printf '%s\n' "${CHOSEN[@]}" | "$HF_PY" -c '
import os, sys, fnmatch
from huggingface_hub import HfApi
api, home = HfApi(), os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
info_cache = {}
for line in sys.stdin.read().splitlines():
    repo, pat = line.split("|", 1)
    d = os.path.join(home, "hub", "models--" + repo.replace("/", "--"))
    ref = os.path.join(d, "refs", "main")
    if not os.path.isfile(ref): continue
    local = open(ref).read().strip()
    try:
        if repo not in info_cache:
            info_cache[repo] = api.model_info(repo, files_metadata=True)
        info = info_cache[repo]
    except Exception:
        continue
    if (info.sha or "") != local: continue      # a new revision, different dir
    for s in info.siblings:
        if not fnmatch.fnmatch(s.rfilename, f"*{pat}*.gguf" if pat else "*"): continue
        if s.size is None: continue
        p = os.path.join(d, "snapshots", local, s.rfilename)
        try:
            if os.path.getsize(p) == s.size: continue
        except OSError:
            continue
        blob = os.path.realpath(p)
        for victim in (p, blob):
            try: os.unlink(victim)
            except OSError: pass
        print(f"  discarding corrupt {repo}/{s.rfilename} - will re-download")
'

mkdir -p "$LOGDIR"
echo
echo "  ${#QUEUE[@]} selections, one at a time, $CONN connections per file"
[ -n "${HF_TOKEN:-}" ] && echo "  HF_TOKEN found." || echo "  No HF_TOKEN - public repos only."
echo "  logs: $LOGDIR"
echo

# --- the transfer engine -----------------------------------------------------
# This does NOT use `hf download`, and that is the whole point.
#
# huggingface_hub downloads into a process-unique temp file and never resumes:
#
#   tmp_path = incomplete_path.with_name(f"{stem}.{uuid4().hex[:8]}.incomplete")
#   with tmp_path.open("wb") as f: ...            # fresh file, truncating
#   http_get(url, f, ...)                          # no resume offset
#
# (file_download.py, huggingface/huggingface_hub#4228 - deliberate, to avoid
# cache corruption on filesystems where flock silently succeeds for everyone.)
# So every interruption restarts every unfinished file from zero, on the xet
# path AND on plain HTTP. Measured here on 2026-08-27: a night of transfers left
# 46 GB on disk that the next run would not have touched.
#
# aria2c resumes (-c), and the Hub serves ranges: a HEAD at an arbitrary offset
# returns 206 with a Content-Range. So we drive the transfer ourselves and write
# the HF cache layout by hand. It is not much code and it is the difference
# between "stop it whenever you like" and "never stop it".
#
# Integrity is not taken on trust: an LFS blob's filename in the cache IS the
# sha256 of its contents, so every finished file is hashed and compared with the
# name it is about to be stored under.
#
# Layout written (identical to what hf would have produced):
#   blobs/<sha256>                      finished, immutable
#   blobs/<sha256>.incomplete           in flight, resumable, ours alone
#   blobs/<sha256>.incomplete.aria2     aria2's segment map - DO NOT DELETE
#   snapshots/<commit>/<path>           relative symlink into blobs/
#   refs/main                           the commit
#
# The .aria2 control file is the one piece of state that must survive a stop.
# With -x N the file is written as N segments at different offsets, so a partial
# is full of holes and its apparent size means nothing. The control file records
# which ranges are real. Delete it and aria2c falls back to treating the
# apparent size as a contiguous prefix, resumes past every hole, and produces a
# file that is the right length and wrong contents - caught by the sha256 check
# at the end, but only after re-downloading the whole thing.
fetch_file() {
  # blob name, byte size, path within the repo, repo, commit, digest kind.
  # The blob name IS the digest, so which algorithm to verify with follows from
  # how the file is stored: sha256 for LFS, git blob sha1 for everything else.
  local sha="$1" size="$2" rpath="$3" repo="$4" commit="$5" kind="${6:-sha256}"
  local dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${repo//\//--}"
  local blob="$dir/blobs/$sha" part="$dir/blobs/$sha.incomplete"
  local link="$dir/snapshots/$commit/$rpath"
  local url="https://huggingface.co/$repo/resolve/$commit/$rpath"
  local name; name="$(basename "$rpath")"

  # Relative symlink, so the cache stays movable. Depth is however deep the file
  # sits inside the snapshot, plus the two levels for snapshots/<commit>.
  local depth rel=""
  depth=$(( $(tr -cd '/' <<< "$rpath" | wc -c) + 2 ))
  for _ in $(seq 1 "$depth"); do rel="../$rel"; done

  mkdir -p "$dir/blobs" "$(dirname "$link")" "$dir/refs"

  if [ -f "$blob" ] && [ "$(stat -c %s "$blob")" = "$size" ]; then
    [ -L "$link" ] || ln -sfn "${rel}blobs/$sha" "$link"
    printf '  %-46s already complete\n' "${name:0:46}"
    return 0
  fi

  # Allocated blocks, NOT stat %s. aria2 -x N writes N segments at different
  # offsets, so a barely-started file already reports its full apparent size
  # while most of it is holes. %s would have claimed 39.9 of 43.8 GB on a file
  # that actually held 14.3.
  local have=0
  [ -f "$part" ] && have=$(( $(stat -c %b "$part") * $(stat -c %B "$part") ))
  printf '  %-46s %.1f/%.1f GB on disk, resuming\n' "${name:0:46}" \
    "$(awk -v b="$have" 'BEGIN{printf "%.1f", b/1e9}')" \
    "$(awk -v b="$size" 'BEGIN{printf "%.1f", b/1e9}')"

  # --max-tries=0 is infinite: a bad line should cost time, never progress.
  # Everything already on disk is kept, so a retry is free.
  #
  # --lowest-speed-limit is the one that matters, and leaving it at aria2's
  # default of 0 (disabled) cost hours here. --timeout only fires on a
  # connection that stops responding; a connection that merely goes slow is
  # never dropped and never retried. Observed 2026-08-27: connections opened
  # during a bad stretch degraded to 13 KiB/s and stayed there for hours - CN
  # fell from 8 to 1 and the survivor never recovered - while a freshly opened
  # connection to the same URL got 10.4 MB/s. With a floor, that connection is
  # closed and reopened instead of limping.
  #
  # Set it well below a healthy rate but far above a stuck one. A genuinely slow
  # line just retries and keeps its bytes; nothing is lost either way.
  local auth=()
  [ -n "${HF_TOKEN:-}" ] && auth=(--header "Authorization: Bearer $HF_TOKEN")
  aria2c -c -x "$CONN" -s "$CONN" -k 10M \
    --max-tries=0 --retry-wait=15 --timeout=60 --connect-timeout=30 \
    --lowest-speed-limit="${MIN_SPEED:-200K}" \
    --max-file-not-found=5 --file-allocation=none \
    --allow-overwrite=true --auto-file-renaming=false \
    --summary-interval=30 --console-log-level=warn \
    "${auth[@]}" -d "$dir/blobs" -o "$sha.incomplete" "$url" 9>&- || return 1

  local got; got="$(stat -c %s "$part" 2>/dev/null || echo 0)"
  if [ "$got" != "$size" ]; then
    echo "  $name: got $got bytes, expected $size - leaving it in place to resume" >&2
    return 1
  fi
  printf '  %-46s verifying %s ...\n' "${name:0:46}" "$kind"
  # A git blob sha1 is sha1 over "blob <bytes>\0" followed by the contents, which
  # is why it is not just sha1sum of the file. Cheap either way: every non-LFS
  # file here is a few KB.
  local actual
  if [ "$kind" = sha1 ]; then
    actual="$( { printf 'blob %s\0' "$size"; cat "$part"; } | sha1sum | cut -d' ' -f1)"
  else
    actual="$(sha256sum "$part" | cut -d' ' -f1)"
  fi
  if [ "$actual" != "$sha" ]; then
    echo "  $name: $kind MISMATCH (got $actual, want $sha) - discarding" >&2
    rm -f "$part"
    return 1
  fi
  mv -f "$part" "$blob"
  ln -sfn "${rel}blobs/$sha" "$link"
  echo "$commit" > "$dir/refs/main"
  printf '  %-46s done, banked\n' "${name:0:46}"
}

run_job() (
  repo="$1"; pat="$2"
  label="$repo"; [ -n "$pat" ] && label="$repo:$pat"
  # The log name carries the pattern too. With one log per repo, two selections
  # from the same repo would truncate each other's log on the way past.
  log="$LOGDIR/${repo//\//--}${pat:+--$pat}.log"
  : > "$log"

  # sha256 / size / path for exactly the files this selection asks for, newest
  # commit. Ordered smallest first so the quick wins are banked before the box
  # commits hours to a 46 GB shard.
  manifest="$("$HF_PY" -c '
import sys, fnmatch
from huggingface_hub import HfApi
repo, pat = sys.argv[1], sys.argv[2]
info = HfApi().model_info(repo, files_metadata=True)
glob = f"*{pat}*.gguf" if pat else "*"
rows = []
for s in info.siblings:
    if not fnmatch.fnmatch(s.rfilename, glob): continue
    lfs = s.lfs
    sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    kind = "sha256"
    if not sha:
        # Not an LFS file: the cache names it by its git blob sha1. Skipping
        # these is what would leave a safetensors repo without its config.json.
        sha, kind = (getattr(s, "blob_id", None) or ""), "sha1"
    if not sha or not s.size: continue
    rows.append((s.size, sha, kind, s.rfilename))
for size, sha, kind, path in sorted(rows):
    print(f"{sha}\t{size}\t{kind}\t{path}")
print(f"COMMIT\t{info.sha}\t-\t-")
' "$repo" "$pat" 2>>"$log")" || { echo "  [$label] could not read the file list from the Hub - see $log"; exit 1; }

  commit="$(awk -F'\t' '$1=="COMMIT"{print $2}' <<< "$manifest")"
  [ -n "$commit" ] || { echo "  [$label] no commit returned"; exit 1; }

  rc=0
  while IFS=$'\t' read -r sha size kind rpath; do
    [ "$sha" = COMMIT ] && continue
    [ -z "$sha" ] && continue
    fetch_file "$sha" "$size" "$rpath" "$repo" "$commit" "$kind" || rc=1
  done <<< "$manifest"

  [ "$rc" = 0 ] && { echo "  [$label] complete."; exit 0; }
  echo "  [$label] finished with errors - re-run to continue from where it stopped"
  exit 1
)

# --- run the queue, one at a time --------------------------------------------
# run_job is a ( ) subshell, so calling it in the foreground is the whole of the
# sequencing: its `exit` scopes to itself and a failure cannot take the queue
# down with it. A failed entry is recorded and the next one still runs.
FAILED=()
n=0
for entry in "${QUEUE[@]}"; do
  n=$(( n + 1 ))
  repo="${entry%%|*}"; pat="${entry#*|}"
  label="$repo"; [ -n "$pat" ] && label="$repo:$pat"
  echo
  echo "  ===== [$n/${#QUEUE[@]}] $label ====="
  run_job "$repo" "$pat" || FAILED+=("$label")
done

echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED (${#FAILED[@]}): ${FAILED[*]}"
  echo "Logs in $LOGDIR. Re-run to continue: finished files are skipped and the"
  echo "one that was in flight resumes at its current offset."
  exit 1
fi
echo "All downloads complete."
echo "Confirm with: $0 --check   (it verifies every file against the Hub's size)"
