#!/bin/bash
# CPU-only pytest of a tvm-tirx worktree's codegen + tile suites. Usage: run_a3suite.sh <wt> <log>
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib
export PYTHONPATH=$1/python CUDA_VISIBLE_DEVICES= PATH=/usr/local/cuda/bin:$PATH
cd $1 && $HOME/kernel-evolve/tvm-tirx/.venv/bin/python -m pytest -q -p no:cacheprovider -n 8 -rfE \
  tests/python/tirx/codegen tests/python/tirx/tile > $2 2>&1
