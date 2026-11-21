#!/bin/bash
# A2 evidence: tvm-tirx codegen + tile/cuda suites on this GPU, NO plugin, from a worktree's Python
# (compiled libraries from the main build; A2 changes tests and testing helpers only).
# Usage: run_a2.sh <worktree> <tag> [pytest args...]
WT=$1; TAG=$2; shift 2
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib PYTHONPATH=$WT/python:$HOME/kernel-evolve/sm90-probe/plugins
PY=$HOME/kernel-evolve/tvm-tirx/.venv/bin/python; OUT=$HOME/kernel-evolve/sm90-probe/logs/a2
cd $WT; echo "worktree $(git log --oneline -1) dirty=$(git status --short | grep -v PR_DESC | wc -l)"
$PY -c "import tvm_tirx; print('tvm_tirx from', tvm_tirx.__file__)"
$PY -m pytest -p no:cacheprovider -p report_tvm_tirx -q -n ${NPROC:-8} -rfE ${TESTS:-tests/python/tirx/codegen tests/python/tirx/tile/cuda} \
   --junitxml=$OUT/$TAG.xml "$@" > $OUT/$TAG.log 2>&1
tail -1 $OUT/$TAG.log
