#!/bin/bash
# B200 / CPU-only neutrality of A2's collection gate: compare which tests are collected and statically
# skipped, upstream main vs A2. Mode "sm100": env and Target emulate a B200 (emulate_sm100_target).
# Mode "nogpu": run in a step with no visible GPU.  Usage: a2_neutrality.sh sm100|nogpu
MODE=$1; SP=$(echo $HOME/kernel-evolve/tvm-tirx/.venv/lib/python3*/site-packages); OUT=$HOME/kernel-evolve/sm90-probe/logs/a2
EXTRA=""; [ "$MODE" = sm100 ] && EXTRA="-p emulate_sm100_target"
for T in upstream-main sm90-test-gate; do
  WT=$HOME/kernel-evolve/wt/$T
  ( cd $WT && SKBUILD_EDITABLE_SKIP=$HOME/kernel-evolve/tvm-tirx/build TVM_TIRX_LIBRARY_PATH=$SP/tvm_tirx/lib \
    PYTHONPATH=$WT/python:$HOME/kernel-evolve/sm90-probe/plugins DUMP_COLLECTION=$OUT/collect_${MODE}_$T.tsv \
    $HOME/kernel-evolve/tvm-tirx/.venv/bin/python -m pytest -p no:cacheprovider -p report_tvm_tirx -p dump_collection $EXTRA \
      --collect-only -q tests/python/tirx/codegen tests/python/tirx/tile/cuda 2>&1 | grep -E '^tvm_tirx:|tests collected|error' | head -5 )
done
python3 - "$OUT/collect_${MODE}_upstream-main.tsv" "$OUT/collect_${MODE}_sm90-test-gate.tsv" <<'PY'
import sys
a = dict(l.split("\t", 1) for l in open(sys.argv[1])); b = dict(l.split("\t", 1) for l in open(sys.argv[2]))
print(f"upstream {len(a)} items ({sum(v.startswith('skip') for v in a.values())} static skip); "
      f"A2 {len(b)} items ({sum(v.startswith('skip') for v in b.values())} static skip)")
print("only upstream:", sorted(set(a) - set(b))[:10]); print("only A2:", sorted(set(b) - set(a))[:10])
diff = [k for k in set(a) & set(b) if a[k].split("\t")[0] != b[k].split("\t")[0]]
print(f"same nodeid, different run/skip status: {len(diff)}", sorted(diff)[:10])
PY
