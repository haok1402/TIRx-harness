"""A3 acceptance: the sm_100 fp16_bf16_gemm compiled for sm_90a raises PTXTargetError naming
tcgen05.mma and a kernel source line, and never reaches NVRTC."""
from tvm_tirx.backend.cuda import compiler
from tvm_tirx.backend.cuda.transform import PTXTargetError
from tirx_kernels.gemm import fp16_bf16_gemm as g
from tirx_kernels.runner import compile_kernel

reached = []
compiler.compile_source = lambda *a, **k: reached.append(1) or (_ for _ in ()).throw(AssertionError("reached NVRTC"))
bc = {"cuda": {"arch": "sm_90a"}}
try:
    compile_kernel(g.get_kernel("bf16", 4096, 4096, 4096, backend_config=bc), backend_config=bc)
    print("FAIL: compiled")
except PTXTargetError as e:
    msg = str(e)
    print(msg[:1500])
    ok = "tcgen05.mma" in msg and "fp16_bf16_gemm.py:" in msg and not reached
    print("ACCEPT" if ok else "FAIL: message/NVRTC check", "nvrtc_reached=", bool(reached))
except Exception as e:  # noqa: BLE001
    print("FAIL: other", type(e).__name__, str(e)[:800], "nvrtc_reached=", bool(reached))
