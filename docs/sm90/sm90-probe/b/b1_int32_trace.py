import traceback, sys
from tirx_kernels.gemm import fp16_bf16_gemm_sm90 as g
from tirx_kernels.runner import compile_kernel
BC = {"cuda": {"arch": "sm_90a"}}
try:
    compile_kernel(g.make_kernel("bf16", 4096, 11008, 4096, backend_config=BC), backend_config=BC)
except Exception as e:
    tb = traceback.format_exc()
    print("\n".join(l for l in tb.splitlines() if ("transform" in l.lower() or "Pass" in l or "pass" in l) and "File" in l)[:3000])
    print(str(e)[:3000])
