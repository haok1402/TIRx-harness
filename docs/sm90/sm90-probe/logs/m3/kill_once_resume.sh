#!/bin/bash
# M3: cancel the kcoral server step once, mid-run: >= 30 min after the agent start and while a request is in
# flight (from /health), or at 80 min regardless. Cancels only this run's step (named after its URL file).
F=$HOME/kernel-evolve/m3/kcoral.url
START=$(date +%s)
while true; do
  now=$(date +%s); el=$((now - START))
  inflight=$(curl -s -m 5 -H "Authorization: Bearer $(cat $F.token 2>/dev/null)" "$(cat $F 2>/dev/null)/health" 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['load']['requests_in_progress'])" 2>/dev/null || echo 0)
  if { [ $el -ge 180 ] && [ "${inflight:-0}" -ge 1 ]; } || [ $el -ge 2400 ]; then
    J=$(squeue -j 3006977 -t RUNNING -h -o %A | head -1)
    S=$(squeue -s -j $J -h -o '%i %j' | awk '$2 == "kcoral-656b88f872b0" {print $1}')
    echo "$(date +%T) elapsed=${el}s in_flight=${inflight}: scancel $S"
    scancel $S
    exit 0
  fi
  sleep 20
done
