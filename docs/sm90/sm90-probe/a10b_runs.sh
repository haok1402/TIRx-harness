#!/bin/bash
cd $HOME/kernel-evolve/sm90-probe; W=$HOME/kernel-evolve/wt
./run_a10_tests.sh $W/nvrtc-perf-remarks a10v2
TESTS='tests/python/backend/cuda/test_wgmma_helpers.py tests/python/backend/cuda/test_perf_remarks.py' ./run_a10_tests.sh $W/sm90-wgmma-helpers a7v2
SASS_OUT=$PWD/logs/a10/sass_inputs ./sass_check.sh > logs/a10/sass_check_v2.log 2>&1
./run_a2.sh $W/a2-a1-a10 a2a1a10v2
