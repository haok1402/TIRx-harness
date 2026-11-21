import os, re, json, sys
sys.path.insert(0, os.environ["WT"] + "/tests/python/backend/cuda")
import test_setmaxnreg_contract as t
from tvm_tirx.script import tirx as T
import tvm_tirx
from tvm_tirx.backend.cuda.backend_config import compiler_options, resolve_backend_config
from tvm_tirx.support import nvcc

def kern(block, mb, dec, inc):
    @T.function
    def main(A: T.Tensor((1024,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=block), kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=mb))
        tx = T.cuda.thread_idx("x")
        if tx < 128:
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](dec)
            A[tx] = A[tx] + 1.0
        else:
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](inc)
            A[tx] = A[tx] * 2.0
    return main
out = []
for arch in ("sm_90a",):
    for block in (128, 192, 256, 320, 384, 512, 640):
        for mb in (1, 2, 3):
            for dec, inc in ((40, 232), (24, 240), (56, 200)):
                src = t._compile(kern(block, mb, dec, inc), arch).mod.imports[0].inspect_source()
                _, log = nvcc._compile_cuda_nvrtc(src, "cubin", arch, compiler_options(resolve_backend_config({"arch": arch})))
                used = int(re.search(r"Used (\d+) registers", log).group(1))
                out.append(dict(block=block, mb=mb, dec=dec, inc=inc, used=used, raw=65536 // (block * mb), c7508="C7508" in log))
for r in out: print(json.dumps(r))
