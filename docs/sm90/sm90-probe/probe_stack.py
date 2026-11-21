# SPDX-License-Identifier: Apache-2.0
"""Does the TIRx-harness benchmark stack (torch cu134, NVRTC >= 13.4, cupti-python 13.4,
DeepGEMM/FlashKDA/flashinfer wheels, apache-tvm-tirx 0.1.0 wheel) work on an H200 whose
driver (595.91.07) supports CUDA up to 13.2?  One JSON line per check."""
import importlib, json, os, sys, traceback

def check(name, fn):
    try:
        r = fn()
        print(json.dumps({"check": name, "ok": True, "info": r}), flush=True)
    except BaseException as e:  # noqa: BLE001
        print(json.dumps({"check": name, "ok": False, "err": f"{type(e).__name__}: {e}"[:600]}), flush=True)

def torch_basic():
    import torch
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    c = a @ a
    torch.cuda.synchronize()
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "dev": torch.cuda.get_device_name(),
            "cap": torch.cuda.get_device_capability(), "arch_list": torch.cuda.get_arch_list(),
            "driver": torch._C._cuda_getDriverVersion() if hasattr(torch._C, "_cuda_getDriverVersion") else None,
            "sum": float(c.float().abs().sum())}

def torch_compile():
    import torch
    f = torch.compile(lambda x: torch.nn.functional.silu(x) * 2 + 1)
    x = torch.randn(1 << 20, device="cuda")
    y = f(x); torch.cuda.synchronize()
    return {"maxerr": float((y - (torch.nn.functional.silu(x) * 2 + 1)).abs().max())}

def nvrtc_version():
    from cuda.bindings import nvrtc
    err, maj, mnr = nvrtc.nvrtcVersion()
    import cuda.pathfinder as pf
    return {"nvrtc": f"{maj}.{mnr}", "lib": str(pf.load_nvidia_dynamic_lib("nvrtc").abs_path)}

def tirx_kernel():
    import torch, tvm_tirx
    from tvm_tirx.script import tirx as T
    @T.function
    def k(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tx = T.cuda.thread_idx("x")
        B[tx] = A[tx] * 2.0
    tgt = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": k}), target=tgt, pipeline="tirx")
    a = torch.arange(128, device="cuda", dtype=torch.float32); b = torch.zeros_like(a)
    mod(a, b); torch.cuda.synchronize()
    return {"tvm_tirx": getattr(tvm_tirx, "__version__", "?"), "ok": bool(torch.equal(b, a * 2))}

def cupti():
    import cupti  # cupti-python
    from cupti import cupti as c
    v = c.get_version() if hasattr(c, "get_version") else None
    import torch
    c.activity_enable(c.ActivityKind.CONCURRENT_KERNEL)
    bufs = []
    c.activity_register_callbacks(lambda: (8 << 20, 0), lambda acts: bufs.extend(acts))
    x = torch.randn(1 << 20, device="cuda"); (x * 2).sum(); torch.cuda.synchronize()
    c.activity_flush_all(1); c.activity_disable(c.ActivityKind.CONCURRENT_KERNEL)
    return {"cupti_version": v, "records": len(bufs)}

def imp(mod):
    def f():
        m = importlib.import_module(mod)
        return {"version": getattr(m, "__version__", "?"), "file": getattr(m, "__file__", "?")}
    return f

def deepgemm_run():
    import torch, deep_gemm
    m, n, k = 4096, 4096, 4096
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    deep_gemm.bf16_gemm_nt(a, b, d)
    torch.cuda.synchronize()
    return {"maxerr": float((d.float() - a.float() @ b.float().T).abs().max())}

checks = sys.argv[1:] or ["torch_basic", "torch_compile", "nvrtc_version", "tirx_kernel", "cupti",
                          "deep_gemm", "flash_kda", "flashinfer", "fla", "tirx_kernels", "deepgemm_run"]
for c_ in checks:
    fn = {"torch_basic": torch_basic, "torch_compile": torch_compile, "nvrtc_version": nvrtc_version,
          "tirx_kernel": tirx_kernel, "cupti": cupti, "deepgemm_run": deepgemm_run,
          "deep_gemm": imp("deep_gemm"), "flash_kda": imp("flash_kda"), "flashinfer": imp("flashinfer"),
          "fla": imp("fla"), "tirx_kernels": imp("tirx_kernels")}[c_]
    check(c_, fn)
