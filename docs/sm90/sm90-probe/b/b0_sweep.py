import json, os, sys, importlib
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import gpu_state, interleaved
import flashinfer
BC = {"cuda": {"arch": "sm_90a"}}
print(json.dumps({"state": gpu_state()}), flush=True)
for k in (1, 2, 4, 8):
    os.environ["TIRX_RMSNORM_SM90_CTAS_PER_SM"] = str(k)
    from tirx_kernels.norm import rmsnorm as m
    from tirx_kernels.runner import compile_kernel
    for hs, bs in [(4096, 128), (4096, 4113), (4096, 16384), (8192, 4113), (8192, 16384), (5120, 16384)]:
        ex = compile_kernel(m.get_kernel(hs, backend_config=BC), backend_config=BC)
        x, w = m.prepare_data(bs, hs); out = torch.empty_like(x); ref = torch.empty_like(x)
        args = m._kernel_args(x, w, out)
        t = interleaved({"tirx": lambda: ex(*args), "fi": lambda: flashinfer.norm.rmsnorm(x, w, m.eps, enable_pdl=False, out=ref)})
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        print(json.dumps({"ctas_per_sm": k, "hs": hs, "bs": bs, "tirx_us": round(t["tirx"], 2), "fi_us": round(t["fi"], 2), "ratio": round(t["fi"] / t["tirx"], 3)}), flush=True)
