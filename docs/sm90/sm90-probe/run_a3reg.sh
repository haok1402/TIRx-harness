#!/bin/bash
# A3 registry check with a given tvm-tirx worktree's Python. Usage: run_a3reg.sh <tvm-tirx-wt> <out.jsonl> [max_configs]
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib
TK=${TK:-$HOME/kernel-evolve/TIRx-kernels}
export PYTHONPATH=$1/python:$TK PATH=/usr/local/cuda/bin:$PATH
cd $TK && $HOME/kernel-evolve/tvm-tirx/.venv/bin/python $HOME/kernel-evolve/sm90-probe/a3_registry.py $2 ${3:-}
