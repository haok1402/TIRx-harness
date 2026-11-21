import os, re, subprocess, sys, tempfile, collections
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
from tirx_kernels.runner import compile_kernel
from tvm_tirx.backend.cuda.runtime import occupancy
BC = {"cuda": {"arch": "sm_90a"}}
os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
exf = compile_kernel(g.make_kernel(4096, 7168, 2048, bn=int(sys.argv[1]), backend_config=BC), backend_config=BC)
del os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"]
src = exf.mod.imports[0].inspect_source()
body = src[src.find("main_kernel(const"):]
i = body.find("wait_group_sync_aligned_0()")
print(body[i - 3000:i + 1500])
ex = compile_kernel(g.make_kernel(4096, 7168, 2048, bn=int(sys.argv[1]), backend_config=BC), backend_config=BC)
binary, _ = occupancy.kernel_binary(ex)
with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
    f.write(binary); f.flush()
    sass = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
ops = collections.Counter(m.split(".")[0] for m in re.findall(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9.]+)", sass))
print(dict(ops.most_common(30)))
