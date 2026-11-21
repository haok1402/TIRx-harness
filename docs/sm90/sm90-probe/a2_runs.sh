#!/bin/bash
# All A2 H200 evidence in one step.
cd $HOME/kernel-evolve/sm90-probe
./run_a2.sh $HOME/kernel-evolve/wt/sm90-test-gate gate                  # A2 alone
./run_a2.sh $HOME/kernel-evolve/wt/a2-plus-a1 gate_plus_a1               # A2 + A1 (merge order)
[ -f logs/a2/upstream.xml ] || ./run_a2.sh $HOME/kernel-evolve/wt/upstream-main upstream
# B200 compile emulation of the tests A2 keeps at the 10.0 floor because they fail on every arch
B=tests/python/tirx/tile/cuda
NPROC=4 TESTS="$B/test_cp_async_bulk.py::test_dsmem $B/test_pattern_transfers.py::test_dsmem_remote_addresses \
  $B/test_ldmatrix_stmatrix.py::test_ldstmatrix $B/test_ldmatrix_stmatrix.py::test_ldstmatrix_swizzle \
  $B/test_mma_sync.py $B/test_explicit_patterns.py::test_global_gpu \
  $B/test_cp_async_bulk_tensor.py::test_descriptor_static_boundary_matrix \
  $B/test_add_sub_mul_div.py::test_binary_op_local_subcta_trivial" \
  ./run_a2.sh $HOME/kernel-evolve/wt/sm90-test-gate emul_sm100_buggroup -p emulate_sm100_target
./a2_neutrality.sh sm100 > logs/a2/neutrality_sm100.log 2>&1
