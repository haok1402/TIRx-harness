"""B0 gate on H200: interleaved timing vs flashinfer, per (hidden, batch), each with its tuned sm_90 plan."""
import json, sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import gpu_state, interleaved
import flashinfer
from tirx_kernels.norm import rmsnorm as m
from tirx_kernels.runner import compile_kernel

BC = {"cuda": {"arch": "sm_90a"}}
print(json.dumps({"state": gpu_state()}), flush=True)
tir_sum = fi_sum = 0.0
gate = []
for hs in (4096, 5120, 8192):
    for bs in (1, 32, 128, 4113, 16384):
        ex = compile_kernel(m.get_kernel(hs, bs, backend_config=BC), backend_config=BC)
        x, w = m.prepare_data(bs, hs)
        out = torch.empty_like(x); ref = torch.empty_like(x)
        args = m._kernel_args(x, w, out)
        t = interleaved({"tirx": lambda: ex(*args),
                         "flashinfer": lambda: flashinfer.norm.rmsnorm(x, w, m.eps, enable_pdl=False, out=ref)})
        torch.cuda.synchronize()
        exact = torch.equal(out, ref)
        maxdiff = (out.float() - ref.float()).abs().max().item()
        r = t["flashinfer"] / t["tirx"]
        tir_sum += t["tirx"]; fi_sum += t["flashinfer"]
        if bs >= 4113:
            gate.append(r)
        print(json.dumps({"hs": hs, "bs": bs, "plan": m._sm90.plan(hs, bs)["kind"], "tirx_us": round(t["tirx"], 2),
                          "flashinfer_us": round(t["flashinfer"], 2), "ratio": round(r, 3),
                          "bit_exact_vs_flashinfer": exact, "max_abs_diff": maxdiff}), flush=True)
print(json.dumps({"summed_time_ratio": round(fi_sum / tir_sum, 3), "gate_min_bs_ge_4113": round(min(gate), 3),
                  "gate_shapes": [round(g, 3) for g in gate]}))
print(json.dumps({"state_after": gpu_state()}), flush=True)
