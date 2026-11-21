#!/bin/bash
# B3 environment: the pinned-reference venv plus tvm-tirx built from haok/sm90-stack-fixes (ISSUE-001..003
# merged; wheel installed in the venv, no PYTHONPATH override), and a TIRx-kernels worktree.
# Usage: run_b3.sh <kernels-wt> <script|-m mod> [args]
KWT=$1; shift
export PYTHONPATH=$KWT
export PATH=/mnt/weka/shrd/ci/haok/TIRx-kernels/b3env/bin:/usr/local/cuda/bin:$PATH
python -c "import tvm_tirx, tirx_kernels, torch; print('tvm_tirx', tvm_tirx.__file__, tvm_tirx.__version__ if hasattr(tvm_tirx, '__version__') else ''); print('tirx_kernels', tirx_kernels.__file__)"
python "$@"
