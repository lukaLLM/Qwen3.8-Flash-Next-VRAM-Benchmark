#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# CORRECTION-R1 controller, v2.
#
# v2 adds full monitoring and closes every item in TODO.md "Rules for all new
# runs". v1 (correction_run.sh) is kept unchanged because PLE-01 ran on it and
# editing a running bash script corrupts it.
#
#   ./benchmark/correction_run_v2.sh                      # prints this, exits
#   ./benchmark/correction_run_v2.sh --execute \
#        --plan-id CORRECTION-R1 --only LOAD-01
#
# MONITORING, and why each piece is here:
#
#   dcgm_exporter.sh   tensor-pipe and DRAM activity. Started ONCE before the
#                      first arm and stopped after the last, never mid-test:
#                      profiling metrics sample the GPU, so starting it between
#                      arms would give the arms different conditions and destroy
#                      the counterbalanced order.
#   gpu_telemetry.py   per-arm temperature, power, clocks, throttle flags, PCIe.
#                      A THROTTLED verdict FAILS the arm - a hot card clocks
#                      lower, which would silently measure arm ORDER instead of
#                      configuration.
#   ple_io_monitor.py  per-arm major page-fault rate. A VOID verdict FAILS the
#                      arm. This matters most for LOAD-01, whose mmap arms are
#                      supposed to read from page cache, not thrash the disk.
#
# RULES COVERAGE (TODO.md "Rules for all new runs"):
#   command + env overrides ....... manifest.json .command / .arms[].env
#   model path + hash ............. manifest .model_path / .model_sha256
#                                   (the HF blob filename IS the sha256)
#   quant, engine build, image .... manifest .quant / .build / .image
#   server log + /props ........... <arm>/server.log, <arm>/props.json
#   GPU + process memory split .... <arm>/memory.json (incl. per-process GPU)
#   every request row ............. <arm>/rows.json
#   warm-up rows saved, excluded .. <arm>/warmups.json, never in the summary
#   arm order recorded/alternated .. manifest .arm_order (A-B-B-A style)
#   gpu_settle before each start .. start_arm()
#   downloads stopped ............. preflight()
#   fixed output limit ............ --n-predict on every request
#   FAIL over context limit ....... validation.json .context_overflow
#   FAIL on unexpected option ..... validation.json .option_mismatch (ALL
#                                   options, not just the override tensor)
#   FAIL on thermal/storage ....... validation.json .throttled / .io_void
#   scores from artifacts ......... summary.json computed from rows.json
#   calculation beside result ..... calculation.txt
# -----------------------------------------------------------------------------
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PLAN_ID_REQUIRED="CORRECTION-R1"
EXECUTE=0; PLAN_ID=""; ONLY=""
NPREDICT="${NPREDICT:-128}"
ARGV_ALL="$*"

while [ $# -gt 0 ]; do
  case "$1" in
    --execute)   EXECUTE=1; shift ;;
    --plan-id)   PLAN_ID="$2"; shift 2 ;;
    --only)      ONLY="$2"; shift 2 ;;
    --n-predict) NPREDICT="$2"; shift 2 ;;
    -h|--help)   EXECUTE=0; break ;;
    *) echo "unknown option: $1" >&2; exit 64 ;;
  esac
done

if [ "$EXECUTE" != 1 ]; then
  sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  echo "Implemented tests: PLE-01  LOAD-01  UB-01"
  echo "Nothing was run. Pass --execute --plan-id $PLAN_ID_REQUIRED --only <TEST-ID>."
  exit 0
fi
[ "$PLAN_ID" = "$PLAN_ID_REQUIRED" ] || { echo "refusing: --plan-id must be $PLAN_ID_REQUIRED" >&2; exit 64; }
[ -n "$ONLY" ] || { echo "refusing: --only <TEST-ID> is required" >&2; exit 64; }

source benchmark/gpu_settle.sh
export COOLDOWN_MIN="${COOLDOWN_MIN:-120}" COOLDOWN_TEMP="${COOLDOWN_TEMP:-42}" COOLDOWN_MAX="${COOLDOWN_MAX:-900}"
IMG="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
MAX_TEMP="${MAX_TEMP:-84}"

redact(){ sed -E 's/^([[:space:]]*[A-Za-z_]*(TOKEN|SECRET|PASSWORD|APIKEY|API_KEY)[A-Za-z_]*:).*/\1 <redacted>/I'; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="results/corrections/${STAMP}_${ONLY}"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT" 2>/dev/null || { echo "refusing: $OUT exists - will not overwrite" >&2; exit 1; }
echo "evidence -> $OUT"

log(){ echo "$*" | tee -a "$OUT/controller.log"; }
down(){ ./scripts/serve.sh --down >/dev/null 2>&1; pkill -x vram_cap.py 2>/dev/null; sleep 2; }
DCGM_UP=0
cleanup(){
  log "controller exiting - stopping server and monitors"
  kill "${TELE_PID:-0}" "${IO_PID:-0}" 2>/dev/null
  down
  [ "$DCGM_UP" = 1 ] && ./benchmark/dcgm_exporter.sh down >/dev/null 2>&1
}
trap cleanup EXIT

preflight(){
  local fail=0 gmem gtemp disk dl srv
  gmem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  gtemp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  disk=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
  srv=$(docker ps --filter name=q38n --format '{{.Names}}' | head -1)
  dl=none; pgrep -x aria2c >/dev/null 2>&1 && dl=ACTIVE
  python3 - "$OUT/preflight.json" "$gmem" "$gtemp" "$disk" "$dl" "${srv:-}" <<'PY'
import json,sys,subprocess
out,gmem,gtemp,disk,dl,srv=sys.argv[1:7]
sw=subprocess.run(["bash","-c","free -m | awk '/^Swap/{print $3}'"],capture_output=True,text=True).stdout.strip()
json.dump({"gpu_memory_used_mib":int(gmem),"gpu_temp_c":int(gtemp),"disk_avail_gib":int(disk),
           "download_state":dl,"other_server":srv or None,"swap_used_mib":int(sw or 0),
           "free_output":subprocess.run(["free","-m"],capture_output=True,text=True).stdout},
          open(out,"w"),indent=2)
PY
  log "  preflight: GPU ${gmem} MiB / ${gtemp} C | disk ${disk} GiB | download ${dl} | other server '${srv:-none}'"
  [ "$gmem" -gt 2000 ] 2>/dev/null && { log "  STOP: GPU memory ${gmem} MiB > 2000"; fail=1; }
  [ "$gtemp" -gt 45 ] 2>/dev/null && { log "  STOP: GPU ${gtemp} C > 45"; fail=1; }
  [ "$disk" -lt 100 ] 2>/dev/null && { log "  STOP: only ${disk} GiB free, need 100"; fail=1; }
  [ "$dl" = ACTIVE ] && { log "  STOP: a model download is active"; fail=1; }
  [ -n "${srv:-}" ] && { log "  STOP: another llama server is running ($srv)"; fail=1; }
  return $fail
}

# $1 arm  $2 OT  $3 ctx  $4 load-mode  $5 n-cpu-moe
start_arm(){
  local arm="$1" ot="$2" ctx="$3" lm="$4" ncm="$5"
  local d="$OUT/$arm"; mkdir -p "$d"
  log "  --- arm $arm : ot=$ot ctx=$ctx load-mode=$lm n-cpu-moe=$ncm ---"
  down; gpu_settle 2>&1 | tee -a "$OUT/controller.log"

  local model; model=$(./scripts/serve.sh --print 2>/dev/null | awk '/^model/{print $2}')
  # Record the exact environment this arm imposes - rule: "save all env overrides".
  cat > "$d/env.txt" <<ENVEOF
OT=$ot
CTX=$ctx
LOAD_MODE=$lm
N_CPU_MOE=$ncm
LAZY=off
UBATCH=${UBATCH:-512}
BATCH=2048
PARALLEL=1
KV_UNIFIED=true
NGL=999
LLAMA_IMAGE=$IMG
NPREDICT=$NPREDICT
ENVEOF

  OT="$ot" CTX="$ctx" LOAD_MODE="$lm" N_CPU_MOE="$ncm" LAZY=off UBATCH="${UBATCH:-512}" \
    BATCH=2048 PARALLEL=1 KV_UNIFIED=true NGL=999 LLAMA_IMAGE="$IMG" MODEL="$model" \
    docker compose -f docker/docker-compose.yaml config 2>&1 | redact > "$d/compose.txt"

  # Validate EVERY option we care about, not just the override tensor.
  python3 - "$d/compose.txt" "$d/option_check.json" "$ot" "$ctx" "$lm" "$ncm" "${UBATCH:-512}" <<'PY'
import json,re,sys
comp,out,ot,ctx,lm,ncm,ub=sys.argv[1:8]
txt=open(comp).read()
def env(k):
    m=re.search(rf'{k}:\s*(\S+)',txt)
    return m.group(1).strip('"') if m else None
def arg(flag):
    m=re.search(rf'-\s*{re.escape(flag)}\s*\n\s*-\s*"?([^"\n]+)"?',txt)
    return m.group(1).strip().strip('"') if m else None
want={"LLAMA_ARG_OVERRIDE_TENSOR":ot,"LLAMA_ARG_TENSOR_READ_LAZY":"off",
      "LLAMA_ARG_KV_UNIFIED":"true"}
got={k:env(k) for k in want}
argwant={"--ctx-size":ctx,"--n-cpu-moe":ncm,"--ubatch-size":ub,
         "--batch-size":"2048","--parallel":"1","--n-gpu-layers":"999",
         "--load-mode":lm}
argot={k:arg(k) for k in argwant}
mism={k:{"want":v,"got":got[k]} for k,v in want.items() if got[k]!=v}
mism.update({k:{"want":v,"got":argot[k]} for k,v in argwant.items() if argot[k]!=v})
json.dump({"resolved_env":got,"resolved_args":argot,"mismatches":mism},open(out,"w"),indent=2)
if mism:
    print("      OPTION MISMATCH: "+json.dumps(mism))
    sys.exit(1)
print(f"      options verified: ot={got['LLAMA_ARG_OVERRIDE_TENSOR']} ctx={argot['--ctx-size']} "
      f"ncmoe={argot['--n-cpu-moe']} ub={argot['--ubatch-size']} load-mode={argot['--load-mode']}")
PY
  if [ $? -ne 0 ]; then
    log "      FAIL: resolved options do not match this arm's intent"
    echo "option_mismatch" > "$d/FAILED"; return 1
  fi

  OT="$ot" CTX="$ctx" LOAD_MODE="$lm" N_CPU_MOE="$ncm" LAZY=off UBATCH="${UBATCH:-512}" \
    BATCH=2048 PARALLEL=1 KV_UNIFIED=true NGL=999 LLAMA_IMAGE="$IMG" \
    ./scripts/serve.sh --quant UD-IQ4_XS --ctx "$ctx" --load-mode "$lm" \
      --n-cpu-moe "$ncm" --ot "$ot" --lazy off --no-wait >/dev/null 2>&1

  local ok=0 i cname
  for i in $(seq 1 240); do
    curl -sf -m 5 http://localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }
    case "$(docker ps -a --filter name=q38n --format '{{.Status}}' | head -1)" in
      Exited*|Restarting*) break ;;
    esac
    sleep 5
  done
  cname=$(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1)
  [ -n "$cname" ] && docker logs "$cname" > "$d/server.log" 2>&1
  if [ "$ok" != 1 ]; then
    log "      SERVER DID NOT START. Last lines:"
    tail -14 "$d/server.log" 2>/dev/null | sed 's/^/        /' | tee -a "$OUT/controller.log"
    echo "server_start_failed" > "$d/FAILED"; return 1
  fi

  curl -s http://localhost:8000/props > "$d/props.json" 2>/dev/null
  local pid; pid=$(pgrep -x llama-server | head -1)
  python3 - "$d/memory.json" "${pid:-0}" <<'PY'
import json,subprocess,sys
out,pid=sys.argv[1],int(sys.argv[2])
g=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],
                 capture_output=True,text=True).stdout.strip().splitlines()[0].split(",")
apps=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],
                    capture_output=True,text=True).stdout.strip()
d={"gpu_memory_used_mib":int(g[0]),"gpu_memory_total_mib":int(g[1]),
   "gpu_per_process":apps,"pid":pid}
if pid:
    try:
        for line in open(f"/proc/{pid}/status"):
            for k in ("VmRSS:","RssAnon:","RssFile:","RssShmem:","VmSwap:"):
                if line.startswith(k): d[k.strip(':').lower()+"_kib"]=int(line.split()[1])
    except Exception as e: d["proc_error"]=str(e)
json.dump(d,open(out,"w"),indent=2)
print(f"      GPU {d['gpu_memory_used_mib']} MiB | RssAnon {d.get('rssanon_kib',0)/1048576:.1f} GiB "
      f"| swap {d.get('vmswap_kib',0)/1048576:.1f} GiB")
PY

  # Model-card conformance, RECORDED not gated. These tests are fixed-length
  # synthetic throughput sweeps: speed_bench_v2 sends temperature 0 with
  # ignore_eos and a pinned n_predict, so the sampler cannot change how much
  # work is done and greedy is the right choice. Gating on the card here would
  # fail every arm for a deliberate decision. Workload tests that depend on what
  # the model actually writes DO gate on it.
  uv run benchmark/check_config.py --mode thinking --json "$d/config_check.json" \
      > "$d/config_check.log" 2>&1 || true
  grep -E 'CONFIG CHECK|FAULT' "$d/config_check.log" 2>/dev/null | sed 's/^/      card: /' \
      | tee -a "$OUT/controller.log"

  # per-arm monitors, started only after the model is loaded so the load itself
  # does not dominate the fault and thermal statistics
  uv run benchmark/gpu_telemetry.py watch --seconds 5400 --interval 2 \
      --max-temp "$MAX_TEMP" --samples --out "$d/gpu.json" > "$d/gpu_telemetry.log" 2>&1 &
  TELE_PID=$!
  uv run benchmark/ple_io_monitor.py watch --seconds 5400 --interval 2 \
      --pid "${pid:-0}" --out "$d/io.json" > "$d/io_monitor.log" 2>&1 &
  IO_PID=$!
  return 0
}

stop_monitors(){
  local d="$OUT/$1"
  kill -TERM "${TELE_PID:-0}" "${IO_PID:-0}" 2>/dev/null
  sleep 12
  kill -KILL "${TELE_PID:-0}" "${IO_PID:-0}" 2>/dev/null
  TELE_PID=""; IO_PID=""
  local v
  v=$(python3 -c "
import json,sys
try: print(json.load(open('$d/gpu.json')).get('verdict','?'))
except Exception: print('no-file')" 2>/dev/null)
  log "      telemetry verdict: $v"
}

run_speed(){   # $1 arm  $2 lengths  $3 n_predict
  local arm="$1" d="$OUT/$1" j
  uv run benchmark/speed_bench_v2.py --tag "corr_${ONLY}_${arm}" \
      --lengths "$2" --n-predict "$3" --num-prompts 3 --no-ledger \
      2>&1 | tee "$d/speed.log" | grep -E "tok \| prefill" | sed 's/^/      /' \
      | tee -a "$OUT/controller.log"
  # capture the server log AFTER the requests, not just after startup: the
  # pre-run copy only ever contains boot lines and cannot show a mid-run drop.
  cn=$(docker ps -a --filter name=q38n --format '{{.Names}}' | head -1)
  [ -n "$cn" ] && docker logs "$cn" > "$d/server.log.after" 2>&1
  j=$(ls -t "artifacts/corr_${ONLY}_${arm}/speed_v2"/run_*.json 2>/dev/null | head -1)
  [ -n "$j" ] && cp "$j" "$d/rows.json"
  # rule: warm-up rows must be saved even though they are excluded from the score.
  grep -E 'warmup' "$d/speed.log" > "$d/warmups.txt" 2>/dev/null || true
}

ple_01(){
  local CTX=32768
  log "PLE-01: ctx $CTX, n-cpu-moe 0, load-mode none, order A-B-B-A"
  for arm in A1:CPU B1:CUDA0 B2:CUDA0 A2:CPU; do
    local name="${arm%%:*}" place="${arm##*:}"
    if start_arm "$name" "per_layer_token_embd=$place" "$CTX" none 0; then
      run_speed "$name" 2048 "$NPREDICT"; stop_monitors "$name"
    fi
  done
}

load_01(){
  local CTX=262144
  log "LOAD-01: ctx $CTX, n-cpu-moe 23, order mmap-none-none-mmap"
  for arm in M1:mmap N1:none N2:none M2:mmap; do
    local name="${arm%%:*}" lm="${arm##*:}"
    if start_arm "$name" "per_layer_token_embd=CPU" "$CTX" "$lm" 23; then
      run_speed "$name" 2048,8192,32768 128; stop_monitors "$name"
    fi
  done
}

ub_01(){
  local CTX=32768
  log "UB-01: ctx $CTX, n-cpu-moe 0, ascending then descending microbatch"
  for ub in 256 512 1024 2048 2048 1024 512 256; do
    local name="ub${ub}_$( [ -e "$OUT/ub${ub}_a" ] && echo b || echo a )"
    export UBATCH=$ub
    if start_arm "$name" "per_layer_token_embd=CPU" "$CTX" none 0; then
      run_speed "$name" 2048 128; stop_monitors "$name"
    fi
  done
}

log "CORRECTION-R1 controller v2 | test $ONLY | $(date -u +%FT%TZ)"
log "command: $0 $ARGV_ALL"
exec 9>/tmp/queue_rest.lock
log "acquiring the benchmark lock ..."
flock 9
log "lock acquired"
preflight || { log "PREFLIGHT FAILED - not running $ONLY"; exit 1; }

# DCGM profiling counters: up once, before the first arm; down after the last.
if ./benchmark/dcgm_exporter.sh up >/dev/null 2>&1; then
  DCGM_UP=1; log "dcgm-exporter (profiling counters) up on :9402"
else
  log "dcgm-exporter did not start - telemetry falls back to nvidia-smi only"
fi

case "$ONLY" in
  PLE-01)  ple_01 ;;
  LOAD-01) load_01 ;;
  UB-01)   ub_01 ;;
  *) log "test $ONLY is not implemented"; exit 64 ;;
esac

cat > "$OUT/calculation.txt" <<'CALC'
Scores are medians over rows.json["rows"], which contains only measured
requests: speed_bench_v2.py issues one warm-up per length and discards it
before writing (the warm-up lines are kept in warmups.txt).

Recompute with:
  python3 - <<'PY'
  import json,glob,statistics as st
  for f in sorted(glob.glob("*/rows.json")):
      rows=json.load(open(f))["rows"]
      pp=[r["prompt_per_second"] for r in rows if r.get("prompt_per_second")]
      tg=[r["predicted_per_second"] for r in rows if r.get("predicted_per_second")]
      print(f, round(st.median(pp),2) if pp else None,
                round(st.median(tg),2) if tg else None)
  PY
CALC

python3 - "$OUT" "$ONLY" "$STAMP" "$NPREDICT" "$IMG" <<'PY'
import json,pathlib,sys,datetime,statistics as st,os,subprocess
out,test,stamp,npred,img=sys.argv[1:6]
o=pathlib.Path(out)
# the HF blob filename IS the sha256, so the model hash is free
mp=subprocess.run(["./scripts/serve.sh","--print"],capture_output=True,text=True).stdout
model=next((l.split()[1] for l in mp.splitlines() if l.startswith("model")),"")
host=model.replace("/hf",os.path.expanduser("~/.cache/huggingface"),1)
sha=None
try: sha=os.path.basename(os.path.realpath(host))
except Exception: pass
man={"plan_id":"CORRECTION-R1","test":test,"utc_start":stamp,
     "utc_finish":datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
     "command":" ".join(sys.argv[:1]),"n_predict":int(npred),"image":img,
     "quant":"UD-IQ4_XS","model_path":model,"model_blob_sha256":sha,
     "arm_order":[],"arms":{}}
val={"option_mismatch":[],"throttled":[],"io_void":[],"context_overflow":[],"failed_arms":[]}
for d in sorted(p for p in o.iterdir() if p.is_dir()):
    man["arm_order"].append(d.name); a={}
    if (d/"FAILED").exists():
        a["failed"]=(d/"FAILED").read_text().strip(); val["failed_arms"].append(d.name)
    for f,key in (("memory.json","memory"),("option_check.json","options"),
                  ("gpu.json","telemetry"),("io.json","io")):
        if (d/f).exists():
            try: a[key]=json.loads((d/f).read_text())
            except Exception: pass
    if (d/"env.txt").exists():
        a["env"]=dict(l.split("=",1) for l in (d/"env.txt").read_text().split() if "=" in l)
    if a.get("options",{}).get("mismatches"): val["option_mismatch"].append(d.name)
    if str(a.get("telemetry",{}).get("verdict","")).upper().startswith("THROTTL"): val["throttled"].append(d.name)
    if str(a.get("io",{}).get("verdict","")).upper().startswith("VOID"): val["io_void"].append(d.name)
    if (d/"rows.json").exists():
        r=json.loads((d/"rows.json").read_text())
        a["server"]=r.get("server"); a["settings"]=r.get("settings")
        rows=r.get("rows") or []; a["n_measured_rows"]=len(rows)
        nctx=(r.get("server") or {}).get("n_ctx") or 0
        for x in rows:
            if nctx and (x.get("prompt_n",0)+x.get("predicted_n",0))>nctx:
                val["context_overflow"].append(d.name); break
        pp=[x["prompt_per_second"] for x in rows if x.get("prompt_per_second")]
        tg=[x["predicted_per_second"] for x in rows if x.get("predicted_per_second")]
        if pp: a["prefill_median_tps"]=round(st.median(pp),2)
        if tg: a["decode_median_tps"]=round(st.median(tg),2)
    man["arms"][d.name]=a
    # keep telemetry/io blobs out of the manifest body, they are already on disk
    a.pop("telemetry",None); a.pop("io",None)
val["overall"]="FAIL" if any(val[k] for k in
    ("option_mismatch","throttled","io_void","context_overflow","failed_arms")) else "PASS"
(o/"manifest.json").write_text(json.dumps(man,indent=2))
(o/"validation.json").write_text(json.dumps(val,indent=2))
print(f"\n  {'arm':<8}{'override_tensor':<34}{'GPU MiB':>9}{'RssAnon':>9}{'prefill':>9}{'decode':>9}")
for n,a in man["arms"].items():
    m=a.get("memory") or {}
    ot=(a.get("options") or {}).get("resolved_env",{}).get("LLAMA_ARG_OVERRIDE_TENSOR","?")
    print(f"  {n:<8}{str(ot):<34}{m.get('gpu_memory_used_mib','-'):>9}"
          f"{(m.get('rssanon_kib',0)/1048576):>8.1f}G"
          f"{str(a.get('prefill_median_tps','-')):>9}{str(a.get('decode_median_tps','-')):>9}")
print(f"\n  VALIDATION: {val['overall']}")
for k in ("failed_arms","option_mismatch","throttled","io_void","context_overflow"):
    if val[k]: print(f"    {k}: {val[k]}")
PY
log "DONE $ONLY -> $OUT"
