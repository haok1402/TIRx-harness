#!/bin/bash
cd $HOME/kernel-evolve/sm90-probe
./run_a2.sh $HOME/kernel-evolve/wt/sm90-test-gate gate_v2r
./run_a2.sh $HOME/kernel-evolve/wt/a2-plus-a1 gate_plus_a1_v2r
./a2_neutrality.sh sm100 > logs/a2/neutrality_sm100_v2r.log 2>&1
