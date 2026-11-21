#!/bin/bash
# A10 evidence against a tvm-tirx worktree's Python (C++ from the main build; A10 is Python-only).
# Usage: run_a10.sh <worktree> <tag>
WT=$1; TAG=$2
SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages)
export SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib PYTHONPATH=$WT/python:$HOME/kernel-evolve/sm90-probe/plugins
PY=$HOME/kernel-evolve/tvm-tirx/.venv/bin/python; OUT=$HOME/kernel-evolve/sm90-probe/logs/a10; mkdir -p $OUT
cd $WT; echo "worktree $(git log --oneline -1) dirty=$(git status --short | grep -v PR_DESC | wc -l)"
$PY -c "import tvm_tirx; print('tvm_tirx from', tvm_tirx.__file__)"
$PY $HOME/kernel-evolve/sm90-probe/probe_remarks.py > $OUT/remarks_$TAG.jsonl 2>&1
cut -c1-400 $OUT/remarks_$TAG.jsonl
