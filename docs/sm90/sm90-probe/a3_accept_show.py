from tvm_tirx.backend.cuda.transform import PTXTargetError
from tirx_kernels.gemm import fp16_bf16_gemm as g
from tirx_kernels.runner import compile_kernel
bc = {"cuda": {"arch": "sm_90a"}}
try:
    compile_kernel(g.get_kernel("bf16", 4096, 4096, 4096, backend_config=bc), backend_config=bc)
except PTXTargetError as e:
    lines = str(e).splitlines()
    print(len(lines) - 1, "violations")
    for l in lines:
        if "tcgen05.mma" in l: print(l.replace("/mnt/weka/home/hao.kang/kernel-evolve/", ""))
