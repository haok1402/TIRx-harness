#!/bin/bash
# Real-NVRTC compile on upstream main of every row A3 rejected. Usage: run_a3confirm.sh <in.jsonl> <out.jsonl>
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib
export PYTHONPATH=$HOME/kernel-evolve/wt/tirx-main/python:$HOME/kernel-evolve/TIRx-kernels PATH=/usr/local/cuda/bin:$PATH
unset TVM_TIRX_COMPILE_FORCE_FALLBACK
cd $HOME/kernel-evolve/TIRx-kernels && $HOME/kernel-evolve/tvm-tirx/.venv/bin/python $HOME/kernel-evolve/sm90-probe/a3_confirm.py $1 $2
