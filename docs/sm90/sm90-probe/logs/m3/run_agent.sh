#!/bin/bash
# M3: the evolution agent, in a CPU-only Slurm step (no GPU visible); GPU work only via kcoral.
cd /mnt/weka/shrd/ci/haok/TIRx-harness/workspace/m3-stack/kda_flow_runs/m3-rmsnorm-h200/worktree
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""   # no CUDA device even if one were visible: GPU work only via kcoral
export PATH=$HOME/agents/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
echo "agent step: host=$(hostname) CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}' gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ') start=$(date +%T)"
timeout 7200 claude -p "$(cat /mnt/weka/home/hao.kang/kernel-evolve/m3/goal.txt)" --model claude-opus-5-5 \
  --dangerously-skip-permissions --output-format stream-json --verbose \
  > /mnt/weka/home/hao.kang/kernel-evolve/m3/agent.jsonl 2> /mnt/weka/home/hao.kang/kernel-evolve/m3/agent.stderr
echo "agent exit $? at $(date +%T)"
