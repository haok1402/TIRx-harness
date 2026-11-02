#!/bin/bash
# Compile every dumped probe .cu for sm_90a with nvcc, keep PTX + cubin + SASS and the ptxas -v report.
cd "$(dirname "$0")/out"; mkdir -p sass
for f in ${1:-*.cu}; do
  b=${f%.cu}
  nvcc -std=c++17 -arch=sm_90a -cubin -Xptxas -v,--warn-on-spills -o sass/$b.cubin $f > sass/$b.ptxas.txt 2>&1 || echo "NVCC FAIL $b"
  nvcc -std=c++17 -arch=sm_90a -ptx -o sass/$b.ptx $f 2>/dev/null
  cuobjdump -sass sass/$b.cubin > sass/$b.sass 2>/dev/null
  printf "%-45s regs=%-4s spill=%s HGMMA=%-4s warn=%s\n" $b \
    "$(grep -o 'Used [0-9]* registers' sass/$b.ptxas.txt | head -1 | grep -o '[0-9]*')" \
    "$(grep -o '[0-9]* bytes spill stores' sass/$b.ptxas.txt | head -1 | grep -o '^[0-9]*')" \
    "$(grep -c HGMMA sass/$b.sass)" \
    "$(grep -i -c 'wgmma\|serializ' sass/$b.ptxas.txt)"
done
