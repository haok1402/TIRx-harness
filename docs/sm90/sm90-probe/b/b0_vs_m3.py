"""Folded B0 (tirx_kernels rmsnorm sm_90 path) vs the original M3 candidate vs flashinfer, interleaved, same process."""
import importlib.util, json, sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import gpu_state, interleaved
import flashinfer
from tirx_kernels.norm import rmsnorm as m
from tirx_kernels.runner import compile_kernel

spec = importlib.util.spec_from_file_location("m3cand", sys.argv[1])
cand = importlib.util.module_from_spec(spec); spec.loader.exec_module(cand)
BC = {"cuda": {"arch": "sm_90a"}}
print(json.dumps({"state": gpu_state()}), flush=True)
sums = {"folded": 0.0, "m3": 0.0, "flashinfer": 0.0}
for hs in (4096, 5120, 8192):
    for bs in (1, 32, 128, 4113, 16384):
        ex = compile_kernel(m.get_kernel(hs, bs, backend_config=BC), backend_config=BC)
        x, w = m.prepare_data(bs, hs)
        out, ref, y3 = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
        args = m._kernel_args(x, w, out)
        run3 = cand.setup({"X": x, "W": w, "Y": y3}, bs, hs)
        t = interleaved({"folded": lambda: ex(*args), "m3": run3,
                         "flashinfer": lambda: flashinfer.norm.rmsnorm(x, w, m.eps, enable_pdl=False, out=ref)})
        torch.cuda.synchronize()
        for k in sums: sums[k] += t[k]
        print(json.dumps({"hs": hs, "bs": bs, **{f"{k}_us": round(v, 2) for k, v in t.items()},
                          "folded_vs_fi": round(t["flashinfer"] / t["folded"], 3), "m3_vs_fi": round(t["flashinfer"] / t["m3"], 3),
                          "folded_eq_m3": torch.equal(out, y3)}), flush=True)
print(json.dumps({"summed": {k: round(sums["flashinfer"] / v, 3) for k, v in sums.items() if k != "flashinfer"}}))
