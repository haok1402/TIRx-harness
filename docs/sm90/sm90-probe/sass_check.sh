#!/bin/bash
# Compile every dumped probe .cu for sm_90a with nvcc; keep PTX + cubin + SASS and the ptxas -v report, then:
#  - perf : ptxas performance remarks, i.e. lines with a (C75xx) code (WGMMA serialization etc.)
#  - spill: spill stores reported by ptxas -v
#  - chain: structural WGMMA chain check (sass_chain_check.py --ptx): gsb0 only on the last GMMA of a
#           chain, and no SASS chain shorter than its PTX commit group ("serialized"). "unchained"
#           (one GMMA per group) is informational: deliberate per-MMA commits look the same.
#           perf= (ptxas C75xx) is the authoritative serialization signal.
# Kernels themselves compile through NVRTC; TIRx (A10) reports the same (C75xx) lines from NVRTC.
HERE=$(cd "$(dirname "$0")" && pwd)
cd "${SASS_OUT:-$HERE/out}"; mkdir -p sass
for f in ${1:-*.cu}; do
  b=${f%.cu}
  nvcc -std=c++17 -arch=sm_90a -cubin -Xptxas -v,--warn-on-spills -o sass/$b.cubin $f > sass/$b.ptxas.txt 2>&1 || echo "NVCC FAIL $b"
  nvcc -std=c++17 -arch=sm_90a -ptx -o sass/$b.ptx $f 2>/dev/null
  cuobjdump -sass sass/$b.cubin > sass/$b.sass 2>/dev/null
  chain=$(python3 "$HERE/sass_chain_check.py" sass/$b.sass --ptx sass/$b.ptx | python3 -c "import sys,json; r=[json.loads(l) for l in sys.stdin]; print(','.join(sorted({f for x in r for f in x['flags']})) or ('ok' if r else '-'))")
  printf "%-45s regs=%-4s spill=%-4s GMMA=%-4s ARRIVE=%-4s perf=%-3s chain=%s\n" $b \
    "$(grep -o 'Used [0-9]* registers' sass/$b.ptxas.txt | head -1 | grep -o '[0-9]*')" \
    "$(grep -o '[0-9]* bytes spill stores' sass/$b.ptxas.txt | head -1 | grep -o '^[0-9]*')" \
    "$(grep -cE '\b[HQIB]GMMA\.' sass/$b.sass)" "$(grep -c 'WARPGROUP.ARRIVE' sass/$b.sass)" \
    "$(grep -cE '\(C75[0-9][0-9]\)' sass/$b.ptxas.txt)" "$chain"
  grep -E '\(C75[0-9][0-9]\)' sass/$b.ptxas.txt | sed 's/^/    /'
done
