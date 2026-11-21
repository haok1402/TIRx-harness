#!/bin/bash
# M3, resumed session: same agent, same worktree, CPU-only step; GPU work only via kcoral.
cd /mnt/weka/shrd/ci/haok/TIRx-harness/workspace/m3-stack/kda_flow_runs/m3-rmsnorm-h200/worktree
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
export PATH=$HOME/agents/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
echo "resume step: host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ') start=$(date +%T) limit=5719s"
timeout 5719 claude -p "$(cat /mnt/weka/home/hao.kang/kernel-evolve/m3/goal_resume.txt)" --resume be762bb2-8bb6-40f7-94bb-59285adfba10 \
  --model claude-opus-5-5 --dangerously-skip-permissions --output-format stream-json --verbose \
  > /mnt/weka/home/hao.kang/kernel-evolve/m3/agent_resume.jsonl 2> /mnt/weka/home/hao.kang/kernel-evolve/m3/agent_resume.stderr
echo "agent exit $? at $(date +%T)"
