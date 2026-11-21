# SPDX-License-Identifier: Apache-2.0
"""Version-pair check: trace and NVRTC-compile (no launch) the first CONFIG of every registered
TIRx-kernels kernel for its first runtime arch, against the installed tvm-tirx. One JSON line each."""
import json, sys, time, traceback
import tvm_tirx
from tirx_kernels.registry import discover_kernels
from tirx_kernels.runner import compile_kernel

def funcs(obj):
    if isinstance(obj, tvm_tirx.tirx.Function):
        yield obj
    elif isinstance(obj, tvm_tirx.IRModule):
        for _, f in obj.functions.items():
            if isinstance(f, tvm_tirx.tirx.Function):
                yield f
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from funcs(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from funcs(v)

ok = bad = 0
for name, mod in sorted(discover_kernels().items()):
    arch = mod.KERNEL_META["runtime_cuda_archs"][0]
    cfg = dict(mod.CONFIGS[0]) if mod.CONFIGS else {}
    cfg.pop("label", None)
    bc = {"cuda": {"arch": arch}}
    t0 = time.time(); stage = "trace"
    try:
        k = mod.get_kernel(backend_config=bc, **cfg)
        stage = "compile"
        n = 0
        for f in funcs(k):
            compile_kernel(f, backend_config=bc); n += 1
        ok += 1
        print(json.dumps({"kernel": name, "arch": arch, "ok": True, "funcs": n, "s": round(time.time() - t0, 1)}), flush=True)
    except BaseException as e:  # noqa: BLE001
        bad += 1
        msg = f"{type(e).__name__}: {e}".splitlines()
        print(json.dumps({"kernel": name, "arch": arch, "ok": False, "stage": stage, "err": " | ".join(msg[:3])[:400]}), flush=True)
print(f"SUMMARY ok={ok} fail={bad}")
