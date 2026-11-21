#!/bin/bash
# A3 registry check under the b3env venv (tvm-tirx wheel built from haok/sm90-stack-fixes 801280ac4), with a
# given tvm-tirx worktree's Python over the wheel's libraries. Usage: run_a3reg_b3env.sh <tvm-tirx-wt> <out.jsonl>
B3E=/mnt/weka/shrd/ci/haok/TIRx-kernels/b3env
SP=$(echo $B3E/lib/python3*/site-packages)
export TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib
TK=${TK:-$HOME/kernel-evolve/TIRx-kernels}
export PYTHONPATH=$1/python:$TK PATH=$B3E/bin:/usr/local/cuda/bin:$PATH
cd $TK && $B3E/bin/python $HOME/kernel-evolve/sm90-probe/a3_registry.py $2
