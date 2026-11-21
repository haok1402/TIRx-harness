#!/bin/bash
# Like run_b.sh, but with the pinned-reference venv (flashinfer @ f2e04400 from git, cutlass-dsl 4.8.0.dev0);
# Run a B-track script (torch cu134, flashinfer), the tvm-tirx dev stack's Python
# (wt/sm90-stack, C++ from the main build) and a TIRx-kernels worktree.  Usage: run_b.sh <kernels-wt> <script> [args]
KWT=$1; shift
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib
export PYTHONPATH=$HOME/kernel-evolve/wt/sm90-stack/python:$KWT
export PATH=/mnt/weka/shrd/ci/haok/TIRx-kernels/refenv/bin:/usr/local/cuda/bin:$PATH
python -c "import tvm_tirx, tirx_kernels, torch; print('tvm_tirx', tvm_tirx.__file__); print('tirx_kernels', tirx_kernels.__file__); print('torch', torch.__version__)"
python "$@"
