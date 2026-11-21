#!/bin/bash
# A1 evidence: run the WGMMA fence tests and probe_wgmma against a tvm-tirx worktree's Python
# (compiled libraries from the main build; A1 changes Python only). Usage: run_a1.sh <worktree> <tag>
WT=$1; TAG=$2
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build      # bypass the main tree's editable finder
export TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib
export PYTHONPATH=$WT/python:$HOME/kernel-evolve/sm90-probe/plugins
PY=$HOME/kernel-evolve/tvm-tirx/.venv/bin/python
OUT=$HOME/kernel-evolve/sm90-probe/logs/a1
cd $WT
echo "worktree $(git log --oneline -1)"; $PY -c "import tvm_tirx; print('tvm_tirx from', tvm_tirx.__file__)"
# -p sm90_unskip: tvm-tirx's codegen conftest skips every GPU test below sm_100 (fixed separately, A2)
$PY -m pytest -p no:cacheprovider -p sm90_unskip -v \
  "tests/python/tirx/codegen/test_hopper.py::test_wgmma_noop_barrier_fences_register_by_reference" \
  "tests/python/tirx/codegen/test_hopper.py::test_wgmma_ss_nt" 2>&1 | tail -25
PROBE_OUT=$OUT/out_$TAG $PY $HOME/kernel-evolve/sm90-probe/probe_wgmma.py 2>&1 | grep -E '^\{|SUMMARY|Error' | cut -c1-200
