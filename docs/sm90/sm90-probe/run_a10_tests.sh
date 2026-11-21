#!/bin/bash
# A10 tests + controls on this GPU node, against a worktree's Python. Usage: run_a10_tests.sh <worktree> <tag>
WT=$1; TAG=$2
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib PYTHONPATH=$WT/python:$HOME/kernel-evolve/sm90-probe/plugins
PY=$HOME/kernel-evolve/tvm-tirx/.venv/bin/python; OUT=$HOME/kernel-evolve/sm90-probe/logs/a10; mkdir -p $OUT
cd $WT; export PATH=$HOME/kernel-evolve/tvm-tirx/.venv/bin:$PATH
$PY -m pytest -p no:cacheprovider -p report_tvm_tirx -v ${TESTS:-tests/python/backend/cuda/test_perf_remarks.py tests/python/tirx/test_cuda_backend_config.py} > $OUT/tests_$TAG.log 2>&1
tail -3 $OUT/tests_$TAG.log
