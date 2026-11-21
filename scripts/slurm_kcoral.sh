#!/usr/bin/env bash
# Keep an authenticated kcoral benchmark server running as a 1-GPU step of a (preemptible) Slurm job.
#
#   scripts/slurm_kcoral.sh JOB_OR_ARRAY_ID URL_FILE
#   python evolution/remote/kcoral_remote.py <workload> vN --remote @URL_FILE      # clients
#   evolve init --task <task> --remote @URL_FILE
#
# SECURITY. kcoral runs arbitrary code for any client and has no client authentication, so the server
# itself only ever binds 127.0.0.1. In front of it, on the node's network address, runs
# evolution/remote/kcoral_auth_proxy.py, which forwards only requests carrying a per-launch bearer token.
# The token is generated for each launch, written with mode 0600 to URL_FILE.token, passed to the step in
# its environment (never on a command line), and sent by kcoral_remote on every request.
# Residual risk: processes of other users on the same node could reach the loopback port; our
# allocations have been exclusive, but Slurm marks them OverSubscribe=OK. kcoral's bubblewrap sandbox
# needs `bwrap`, which these nodes lack (user namespaces themselves work), so workers run unsandboxed.
#
# Each round resolves the job's RUNNING task (an array's running element has its own job id; srun needs it),
# cancels this URL file's previous server step if one survived (it is named after the URL file), and
# launches the server step. srun runs with a parent-death signal, so a hard-killed wrapper takes its step
# with it. When the step ends (preemption, cancel), the URL and token files are removed and the loop waits
# for a running task again (every RECHECK_SECONDS while none runs). No `--overlap`: the step waits for its GPU
# like any other. Clients re-read URL_FILE and its token and retry (evolution/remote/kcoral_remote.py).
#
# The loop is light (it sleeps and waits on srun) and may run on a login node. Stop it with Ctrl+C/SIGTERM.
#
# Env:
#   KCORAL_VENV        environment with the `benchmark` and `server` groups (default: <repo>/.venv)
#   KCORAL_CPUS / KCORAL_MEM   step resources (default 8 / 64G)
#   CUDA_HOME          CUDA toolkit whose bin/ (ncu, compute-sanitizer) the server's requests get on PATH
#                      (default /usr/local/cuda)
#   RECHECK_SECONDS    wait between checks while no task runs (default 150)
#   KCORAL_SERVER_ARGS extra `kcoral server` arguments
set -uo pipefail

if [ $# -ne 2 ]; then
  sed -n '2,6p' "$0" >&2
  exit 2
fi
JOB=$1
URL_FILE=$(realpath -m "$2")
TOKEN_FILE="$URL_FILE.token"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${KCORAL_VENV:-$ROOT/.venv}
CUDA_BIN=${CUDA_HOME:-/usr/local/cuda}/bin  # ncu / compute-sanitizer for the tool CLIs
CPUS=${KCORAL_CPUS:-8}
MEM=${KCORAL_MEM:-64G}
RECHECK=${RECHECK_SECONDS:-150}
EXTRA=${KCORAL_SERVER_ARGS:-}
PROXY="$ROOT/evolution/remote/kcoral_auth_proxy.py"
# One server per URL file: the step name identifies it, so a new wrapper can find a surviving predecessor.
STEP_NAME="kcoral-$(printf %s "$URL_FILE" | md5sum | cut -c1-12)"
mkdir -p "$(dirname "$URL_FILE")"

log() { echo "[slurm_kcoral $(date '+%F %T')] $*" >&2; }

STEP_PID=
cleanup() {
  [ -n "$STEP_PID" ] && kill "$STEP_PID" 2>/dev/null
  rm -f "$URL_FILE" "$TOKEN_FILE"
  log "stopped"
  exit 0
}
trap cleanup INT TERM

running_job() { squeue -j "$JOB" -t RUNNING -h -o %A 2>/dev/null | head -1; }

cancel_predecessors() {
  # Only steps of this user with this URL file's step name.
  squeue -s -u "$USER" -h -o "%i %j" 2>/dev/null | awk -v n="$STEP_NAME" '$2 == n {print $1}' |
    while read -r step; do
      log "cancelling surviving server step $step"
      scancel "$step"
    done
}

while true; do
  job=$(running_job)
  if [ -z "$job" ]; then
    rm -f "$URL_FILE" "$TOKEN_FILE"
    log "no RUNNING task of $JOB; rechecking in ${RECHECK}s"
    sleep "$RECHECK" & wait $!
    continue
  fi
  cancel_predecessors
  port=$((20000 + RANDOM % 40000))
  inner=$((20000 + RANDOM % 40000))
  token=$(head -c 48 /dev/urandom | base64 | tr -d '+/=\n')
  (umask 077 && printf '%s\n' "$token" > "$TOKEN_FILE.tmp" && mv "$TOKEN_FILE.tmp" "$TOKEN_FILE")
  log "starting kcoral server on job $job (proxy port $port, loopback port $inner), step $STEP_NAME"
  # The step runs kcoral on loopback and the token proxy on the network, publishes the proxy URL, and ends
  # (taking both down) as soon as either exits.
  KCORAL_PROXY_TOKEN="$token" setpriv --pdeathsig TERM -- \
    srun --jobid="$job" -J "$STEP_NAME" -N1 -n1 --gres=gpu:1 --cpus-per-task="$CPUS" --mem="$MEM" \
    bash -c "
      # Requests inherit this environment: the server venv goes first on PATH, so python in --cmd
      # scripts and tool targets is the provisioned interpreter, not the bare system one; CUDA's bin
      # follows, so the tool CLIs find ncu and compute-sanitizer.
      export VIRTUAL_ENV='$VENV' PATH='$VENV/bin':'$CUDA_BIN':\$PATH
      env -u KCORAL_PROXY_TOKEN '$VENV/bin/kcoral' server --host 127.0.0.1 --port $inner --gpus 0 --sandbox none \
        --max-timeout-seconds 3600 --log-dir '' $EXTRA &
      server=\$!
      '$VENV/bin/python' '$PROXY' --listen 0.0.0.0:$port --upstream 127.0.0.1:$inner &
      proxy=\$!
      trap 'kill \$server \$proxy 2>/dev/null' EXIT TERM INT
      echo \"http://\$(hostname):$port\" > '$URL_FILE.tmp' && mv '$URL_FILE.tmp' '$URL_FILE'
      wait -n \$server \$proxy
    " &
  STEP_PID=$!
  wait "$STEP_PID"
  rc=$?
  STEP_PID=
  rm -f "$URL_FILE" "$TOKEN_FILE"
  log "server step on job $job ended (rc=$rc)"
  sleep 10 & wait $!
done
