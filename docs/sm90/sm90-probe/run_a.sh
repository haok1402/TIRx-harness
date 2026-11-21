#!/bin/bash
# Track-A evidence (A3/A4/A5): tvm-tirx tests from a worktree's Python on this GPU (compiled libraries from
# the main build; these items change Python only). Usage: run_a.sh <worktree> <logdir> <tag> <pytest args...>
WT=$1; OUT=$HOME/kernel-evolve/sm90-probe/logs/$2; TAG=$3; shift 3
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib PYTHONPATH=$WT/python:$HOME/kernel-evolve/sm90-probe/plugins
PY=$HOME/kernel-evolve/tvm-tirx/.venv/bin/python; mkdir -p $OUT
cd $WT; echo "worktree $(git log --oneline -1) dirty=$(git status --short | grep -v PR_DESC | wc -l)"
$PY -m pytest -p no:cacheprovider -q -rfEX "$@" > $OUT/$TAG.log 2>&1; echo "pytest rc=$?"
