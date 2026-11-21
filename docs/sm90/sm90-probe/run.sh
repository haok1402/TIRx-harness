#!/bin/bash
# Usage: run.sh <log> <cmd...>   -- runs a command as a 1-GPU Slurm step in the running task of
# array 2394207 (PithTrain), falling back to 3006977 (PithSim).
LOG=$1; shift
JOB=$(squeue -j 2394207 -t RUNNING -h -o %A | head -1)
[ -z "$JOB" ] && JOB=$(squeue -j 3006977 -t RUNNING -h -o %A | head -1)
[ -z "$JOB" ] && { echo "no running task"; exit 3; }
srun --jobid=$JOB -N1 -n1 --gres=gpu:1 --cpus-per-task=${CPUS:-8} --mem=64G bash -lc "export PATH=/usr/local/cuda/bin:\$PATH CUDA_HOME=/usr/local/cuda PYTHONPATH=\$HOME/kernel-evolve/sm90-probe/plugins:\$HOME/kernel-evolve/TIRx-kernels:\$PYTHONPATH; cd \$HOME/kernel-evolve/sm90-probe && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader && $*" > "$LOG" 2>&1
