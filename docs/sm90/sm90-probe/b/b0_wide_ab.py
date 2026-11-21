"""RMSNorm sm_90: cost of 64-bit element offsets (wide) and of an acq_rel completion atomic, interleaved."""
import json, sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import interleaved
from tirx_kernels.norm import _rmsnorm_sm90 as K, rmsnorm as m
from tirx_kernels.runner import compile_kernel

BC = {"cuda": {"arch": "sm_90a"}}
c = lambda f: compile_kernel(f.func, backend_config=BC)
tot = {}
for hs in (4096, 5120, 8192):
    for bs in (1, 32, 128, 4113, 16384):
        p = K.plan(hs, bs)
        x, w = m.prepare_data(bs, hs); y = torch.empty_like(x); xv, yv = x.view(-1), y.view(-1)
        if p["kind"] == "small":
            v = {"narrow": c(K._make_small(hs, p["warps"], p["ld_w"], wide=False)), "wide": c(K._make_small(hs, p["warps"], p["ld_w"], wide=True))}
            grid = bs
        else:
            v = {"narrow": c(K._make_ring(hs, p["warps"], p["stages"], wide=False)),
                 "wide": c(K._make_ring(hs, p["warps"], p["stages"], wide=True))}
            grid = min(bs, 132 * p["ctas_per_sm"])
        ctrs = {k: torch.zeros(2, dtype=torch.uint32, device="cuda") for k in v}
        t = interleaved({k: (lambda k=k: v[k](xv, w, yv, ctrs[k], bs, grid)) for k in v})
        for k, us in t.items(): tot[k] = tot.get(k, 0) + us
        print(json.dumps({"hs": hs, "bs": bs, "kind": p["kind"], **{k: round(us, 2) for k, us in t.items()},
                          "ctr_zero": all(ct.cpu().tolist() == [0, 0] for ct in ctrs.values())}), flush=True)
print(json.dumps({"total_us": {k: round(us, 1) for k, us in tot.items()}}))
