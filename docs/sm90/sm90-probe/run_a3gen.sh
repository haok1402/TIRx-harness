#!/bin/bash
# A3: run the PTX floor generator from the a3 worktree (main build's libraries + NVRTC from its venv).
# Usage: run_a3gen.sh <out.py> [gen_floors args...]
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib PYTHONPATH=$HOME/kernel-evolve/wt/a3/python
export PATH=/usr/local/cuda/bin:$PATH
OUT=$1; shift
cd $HOME/kernel-evolve/wt/a3 && $HOME/kernel-evolve/tvm-tirx/.venv/bin/python -m tvm_tirx.backend.cuda.ptx.gen_floors --out $OUT "$@"
