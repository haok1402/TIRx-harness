#!/bin/bash
# ISSUE-003 A/B: warpgroup index source and k-loop unroll, B1 and B2 (same node, sequential runs).
cd $HOME/kernel-evolve/sm90-probe
for v in "tx 0" "builtin 0" "tx 8" "builtin 8" "builtin 4"; do set -- $v
  echo "== B1 wg=$1 unroll=$2"; B1_WG=$1 B1_UNROLL=$2 b/run_b.sh $HOME/kernel-evolve/wt/k-sm90-gemm b/b1_gemm.py gated bf16 2>&1 | grep -E 'gated_geomean|"error"' | cut -c1-300
done
for v in tx shfl builtin; do
  echo "== B2 wg=$v"; B2_WG=$v b/run_b.sh $HOME/kernel-evolve/wt/k-sm90-fp8-gemm b/b2_fp8.py gated auto 2>&1 | grep -E 'gated_geomean|"error"' | cut -c1-300
done
