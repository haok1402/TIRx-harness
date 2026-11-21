"""Over 2**31 elements: the 32-bit ring (negative control) vs the 64-bit ring; hs128 port timing, old vs new tree.

argv: old_tree (a TIRx-kernels checkout with the 32-bit offsets, e.g. 1242ce8)"""
import importlib, json, sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import interleaved
from tirx_kernels.norm import _rmsnorm_sm90 as K, rmsnorm as m
from tirx_kernels.runner import compile_kernel

BC = {"cuda": {"arch": "sm_90a"}}
hs, bs = 8192, 270000
p = K.plan(hs, bs)
# 64-bit first: the 32-bit kernel's wrapped stores can land in buffers allocated before its output,
# so each variant gets freshly generated tensors and the corrupting one runs last.
for wide in (True, False):
    x, w = m.prepare_data(bs, hs)
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + m.eps) * w.float()).half()
    del xf
    ex = compile_kernel(K._make_ring(hs, p["warps"], p["stages"], wide=wide).func, backend_config=BC)
    y = torch.full_like(x, float("nan"))
    ctr = torch.zeros(2, dtype=torch.uint32, device="cuda")
    ex(x.view(-1), w, y.view(-1), ctr, bs, min(bs, 132 * p["ctas_per_sm"]))
    torch.cuda.synchronize()
    wrong = (((y.float() - ref.float()).abs() > 1e-3 + 1e-3 * ref.float().abs()) | torch.isnan(y)).any(-1)
    rows = wrong.nonzero().flatten()
    print(json.dumps({"ring_wide": wide, "rows_wrong": int(wrong.sum()),
                      "first_wrong": int(rows[0]) if len(rows) else None}), flush=True)
    del x, w, y, ref, wrong
    torch.cuda.empty_cache()
old_tree = sys.argv[1]
for k in [k for k in sys.modules if k.startswith("tirx_kernels")]:
    del sys.modules[k]
sys.path.insert(0, old_tree)
mo = importlib.import_module("tirx_kernels.norm.rmsnorm")
from tirx_kernels.runner import compile_kernel as compile_old
sys.path.remove(old_tree)
for k in [k for k in sys.modules if k.startswith("tirx_kernels")]:
    del sys.modules[k]
mn = importlib.import_module("tirx_kernels.norm.rmsnorm")
assert mo.__file__ != mn.__file__, (mo.__file__, mn.__file__)
for b in (1, 128, 4113, 16384, 65536):
    exo = compile_old(mo.get_kernel(128, backend_config=BC), backend_config=BC)
    exn = compile_kernel(mn.get_kernel(128, b, backend_config=BC), backend_config=BC)
    x, w = mn.prepare_data(b, 128)
    yo, yn = torch.empty_like(x), torch.empty_like(x)
    ao, an = mo._kernel_args(x, w, yo), mn._kernel_args(x, w, yn)
    t = interleaved({"old_int32": lambda: exo(*ao), "new_int64": lambda: exn(*an)})
    torch.cuda.synchronize()
    print(json.dumps({"hs": 128, "bs": b, **{k: round(v, 2) for k, v in t.items()}, "equal": torch.equal(yo, yn)}), flush=True)
