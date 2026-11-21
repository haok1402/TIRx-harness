#!/bin/bash
cd $HOME/kernel-evolve/sm90-probe; W=$HOME/kernel-evolve/wt; L=logs/a10
./run_a10_tests.sh $W/nvrtc-perf-remarks a10
TESTS=tests/python/tirx/test_cuda_backend_config.py ./run_a10_tests.sh $W/upstream-main upstream
PROBE_CU_OUT=$PWD/$L/sass_inputs ./run_a10.sh $W/nvrtc-perf-remarks a10 > /dev/null
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
( export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib PYTHONPATH=$W/nvrtc-perf-remarks/python
  $HOME/kernel-evolve/tvm-tirx/.venv/bin/python a10_controls.py $L/sass_inputs > $L/controls_a10.jsonl 2>&1 )
SASS_OUT=$PWD/$L/sass_inputs ./sass_check.sh > $L/sass_check.log 2>&1
( export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib PYTHONPATH=$W/a2-a1-a10/python
  $HOME/kernel-evolve/tvm-tirx/.venv/bin/python -W always probe_gemm.py > $L/gemm_a2a1a10.log 2>&1 )
./run_a2.sh $W/a2-a1-a10 a2a1a10
echo DONE
