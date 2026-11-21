#!/bin/bash
# C3 validation for tasks whose seed needs tvm-tirx haok/sm90-stack-fixes (B3): the b3env venv (wheel
# installed, FlashInfer @ f2e04400, ninja on PATH), the h-c3 worktree and a TIRx-kernels worktree.
#   c3_run_b3.sh <workload> <version|baseline> <kernels-wt>
cd $HOME/kernel-evolve/wt/h-c3
export PYTHONPATH=$PWD:$3
export PATH=/mnt/weka/shrd/ci/haok/TIRx-kernels/b3env/bin:/usr/local/cuda/bin:$PATH
python -c "import tvm_tirx, tirx_kernels; print('tvm_tirx', tvm_tirx.__file__, 'tirx_kernels', tirx_kernels.__file__)"
python evolution/benchmark/adapter.py $1 $2
