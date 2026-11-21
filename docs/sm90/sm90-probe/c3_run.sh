#!/bin/bash
# C3 validation: run a packaged H200 task (baseline self-check and/or a candidate) from the h-c3 worktree,
# with the C1 venv, the dev tvm-tirx stack (A10 remarks) and a TIRx-kernels worktree for seeds.
#   c3_run.sh <workload> <version|baseline> <kernels-wt>
cd $HOME/kernel-evolve/wt/h-c3
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib
export PYTHONPATH=$PWD:$HOME/kernel-evolve/wt/sm90-stack/python:$3
export PATH=/usr/local/cuda/bin:$PATH
$HOME/kernel-evolve/wt/h-c1/.venv/bin/python -c "import tvm_tirx, tirx_kernels; print('tvm_tirx', tvm_tirx.__file__, 'tirx_kernels', tirx_kernels.__file__)"
$HOME/kernel-evolve/wt/h-c1/.venv/bin/python evolution/benchmark/adapter.py $1 $2
