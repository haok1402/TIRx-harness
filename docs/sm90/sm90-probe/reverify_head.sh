#!/bin/bash
# Re-run the phase-1 evidence against tvm-tirx HEAD (haok/sm90-plan = upstream 6569e2a18 + noop_barrier fix)
# and TIRx-kernels upstream 16f05fc. Writes logs/plan/*.
PY=$HOME/kernel-evolve/tvm-tirx/.venv/bin/python
cd $HOME/kernel-evolve/sm90-probe
export PROBE_OUT=$PWD/out_head
$PY probe_wgmma.py        > logs/plan/wgmma.log 2>&1
$PY probe_gemm.py         > logs/plan/gemm.log 2>&1
$PY probe_cluster_pdl.py  > logs/plan/cluster_pdl.log 2>&1
$PY probe_txl.py          > logs/plan/txl.log 2>&1
cd $HOME/kernel-evolve/tvm-tirx
$PY -m pytest -q -p no:cacheprovider -p sm90_unskip -n 8 tests/python/tirx/codegen/test_hopper.py tests/python/tirx/codegen/test_dsmem.py \
   tests/python/tirx/tile/cuda/test_cp_async_bulk.py tests/python/tirx/tile/cuda/test_cp_async_bulk_tensor.py \
   --junitxml=$HOME/kernel-evolve/sm90-probe/logs/plan/upstream_hopper.xml > $HOME/kernel-evolve/sm90-probe/logs/plan/upstream_hopper.log 2>&1
# upstream gate as-is (no plugin): how many of these run on H200?
$PY -m pytest -q -p no:cacheprovider -n 8 tests/python/tirx/codegen/test_hopper.py tests/python/tirx/tile/cuda/test_cp_async_bulk_tensor.py -rs \
   > $HOME/kernel-evolve/sm90-probe/logs/plan/upstream_gate_asis.log 2>&1
cd $HOME/kernel-evolve/sm90-probe/upstream_sm90_head
$PY -m pytest -q -p no:cacheprovider -p sm90_unskip -n 8 codegen tile --junitxml=$HOME/kernel-evolve/sm90-probe/logs/plan/upstream_broad.xml \
   > $HOME/kernel-evolve/sm90-probe/logs/plan/upstream_broad.log 2>&1
echo DONE
