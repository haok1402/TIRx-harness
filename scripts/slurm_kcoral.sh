#!/usr/bin/env bash
# Keep a kcoral benchmark server running as a 1-GPU step of a (preemptible) Slurm job, and publish its URL.
#
#   scripts/slurm_kcoral.sh JOB_OR_ARRAY_ID URL_FILE
#   python evolution/remote/kcoral_remote.py <workload> vN --remote @URL_FILE      # clients
#   evolve init --task <task> --remote @URL_FILE
#
# Each round resolves the job's RUNNING task (an array's running element gets its own job id; srun needs that
# id, not ARRAY_TASK), launches `kcoral server` as a step on it, and writes http://<node>:<port> to URL_FILE
# (atomically, on shared storage). When the step ends (the task is preempted, or the step is cancelled), the
# URL file is removed and the loop waits for a running task again: every RECHECK_SECONDS while none runs. The
# server takes no `--overlap`: like any other step it waits for its GPU. Clients re-read URL_FILE and retry
# (evolution/remote/kcoral_remote.py), so a scoring request in flight is re-sent to the relaunched server.
#
# The step is named kcoral-server-<this script's pid>, so `squeue -s` shows it and only it can be cancelled.
# The loop itself is light (it sleeps and waits on srun) and may run on a login node; the server runs only in
# the step. Stop it with Ctrl+C / SIGTERM: the current step is cancelled and the URL file removed.
#
# Env:
#   KCORAL_VENV       environment with the `server` group (default: <repo>/.venv)
#   KCORAL_PORT       fixed port (default: a random port in 20000-59999 per launch)
#   KCORAL_CPUS / KCORAL_MEM   step resources (default 8 / 64G)
#   RECHECK_SECONDS   wait between checks while no task runs (default 150)
#   KCORAL_SERVER_ARGS extra `kcoral server` arguments
set -uo pipefail

if [ $# -ne 2 ]; then
  sed -n '2,8p' "$0" >&2
  exit 2
fi
JOB=$1
URL_FILE=$(realpath -m "$2")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV=${KCORAL_VENV:-$ROOT/.venv}
CPUS=${KCORAL_CPUS:-8}
MEM=${KCORAL_MEM:-64G}
RECHECK=${RECHECK_SECONDS:-150}
EXTRA=${KCORAL_SERVER_ARGS:-}
mkdir -p "$(dirname "$URL_FILE")"

log() { echo "[slurm_kcoral $(date '+%F %T')] $*" >&2; }

STEP_PID=
cleanup() {
  [ -n "$STEP_PID" ] && kill "$STEP_PID" 2>/dev/null
  rm -f "$URL_FILE"
  log "stopped"
  exit 0
}
trap cleanup INT TERM

running_job() { squeue -j "$JOB" -t RUNNING -h -o %A 2>/dev/null | head -1; }

while true; do
  job=$(running_job)
  if [ -z "$job" ]; then
    rm -f "$URL_FILE"
    log "no RUNNING task of $JOB; rechecking in ${RECHECK}s"
    sleep "$RECHECK" & wait $!
    continue
  fi
  port=${KCORAL_PORT:-$((20000 + RANDOM % 40000))}
  log "starting kcoral server on job $job, port $port"
  # The step publishes its own URL (it alone knows the node), then becomes the server.
  srun --jobid="$job" -J "kcoral-server-$$" -N1 -n1 --gres=gpu:1 --cpus-per-task="$CPUS" --mem="$MEM" \
    bash -c "set -e; echo \"http://\$(hostname):$port\" > '$URL_FILE.tmp'; mv '$URL_FILE.tmp' '$URL_FILE'
             exec '$VENV/bin/kcoral' server --host 0.0.0.0 --port $port --gpus 0 --sandbox none \
               --max-timeout-seconds 3600 --log-dir '' $EXTRA" &
  STEP_PID=$!
  wait "$STEP_PID"
  rc=$?
  STEP_PID=
  rm -f "$URL_FILE"
  log "server step on job $job ended (rc=$rc)"
  # Back off briefly before relaunching (a port clash or a crash loop must not spin).
  sleep 10 & wait $!
done
