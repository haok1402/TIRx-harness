import os
os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
import tvm_tirx
from tvm_tirx.script import tirx as T

def h_plain():
    T.ptx.wgmma.fence.sync.aligned()
    T.ptx["bar.sync"](T.uint32(1), T.uint32(256))

def h_eval():
    T.evaluate(T.ptx.wgmma.commit_group.sync.aligned())
    T.evaluate(T.ptx["bar.arrive"](T.uint32(2), T.uint32(256)))

@T.function
def k(A: T.Tensor((128,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    h_plain()
    h_eval()
    A[0] = 1.0

mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": k}), target=tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"}), pipeline="tirx")
src = mod.mod.imports[0].inspect_source()
body = src[src.index('__global__'):]
import re
print([l.strip() for l in body.splitlines() if re.search(r'(wgmma|bar_|barrier)\w*\(', l) and 'asm' not in l][:20])
