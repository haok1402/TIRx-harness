#!/bin/bash
# Usage: run.sh <log> <cmd...>   -- runs a command as a 1-GPU Slurm step in the running task of
# array 2394207 (PithTrain), falling back to 3006977 (PithSim).
LOG=$1; shift
# ARRAY=3006977 picks PithSim first (e.g. when PithTrain's GPUs are all busy).
FIRST=${ARRAY:-2394207}; SECOND=$([ "$FIRST" = 2394207 ] && echo 3006977 || echo 2394207)
JOB=$(squeue -j $FIRST -t RUNNING -h -o %A | head -1)
[ -z "$JOB" ] && JOB=$(squeue -j $SECOND -t RUNNING -h -o %A | head -1)
[ -z "$JOB" ] && { echo "no running task"; exit 3; }
srun --jobid=$JOB -N1 -n1 --gres=gpu:1 --cpus-per-task=${CPUS:-8} --mem=64G bash -lc "export PATH=/usr/local/cuda/bin:\$PATH CUDA_HOME=/usr/local/cuda PYTHONPATH=\$HOME/kernel-evolve/sm90-probe/plugins:\$HOME/kernel-evolve/TIRx-kernels:\$PYTHONPATH; cd \$HOME/kernel-evolve/sm90-probe && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader && $*" > "$LOG" 2>&1
