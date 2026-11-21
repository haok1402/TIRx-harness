"""Instruction mix of fa3_forward_sm90 (non-causal), whole kernel and the hottest loop body."""
import collections, re, subprocess, sys, tempfile
from tirx_kernels.attention import fa3_forward_sm90 as fa
from tirx_kernels.runner import compile_kernel
from tvm_tirx.backend.cuda.runtime import occupancy
bc = {"cuda": {"arch": "sm_90a"}}
ex = compile_kernel(fa.get_kernel(4, False, backend_config=bc, pingpong=sys.argv[1] == "pp"), backend_config=bc)
binary, _ = occupancy.kernel_binary(ex)
with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
    f.write(binary); f.flush()
    sass = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
open(sys.argv[2], "w").write(sass)
lines = [l for l in sass.splitlines() if re.search(r"/\*[0-9a-f]{4}\*/", l)]
ops = [re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+)", l) for l in lines]
ops = [o[0] for o in ops if o]
# the loop body: between the 2nd and 3rd WARPGROUP.ARRIVE-led chain pairs is too fiddly; take the span
# between the first backward branch target and the branch (largest BRA backwards)
c = collections.Counter(ops)
print("total", len(ops), {k: c[k] for k in ("HGMMA", "MUFU", "FFMA", "FMUL", "FADD", "FMNMX", "F2FP", "F2F", "SHFL", "BAR", "SYNCS", "IMAD", "IADD3", "LDS", "STG", "R2UR", "WARPGROUP", "WARPSYNC") if c[k]})
