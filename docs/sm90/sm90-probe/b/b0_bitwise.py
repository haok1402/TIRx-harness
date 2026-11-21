"""Outputs of the rmsnorm kernel from two TIRx-kernels trees must be bit-identical on sm_90a. argv: tree_a tree_b"""
import importlib, sys
import torch
tree_a, tree_b = sys.argv[1:3]
BC = {"cuda": {"arch": "sm_90a"}}
def build(tree):
    for k in [k for k in sys.modules if k.startswith("tirx_kernels")]:
        del sys.modules[k]
    sys.path.insert(0, tree)
    try:
        m = importlib.import_module("tirx_kernels.norm.rmsnorm")
        from tirx_kernels.runner import compile_kernel
        return m, {hs: compile_kernel(m.get_kernel(hs, backend_config=BC), backend_config=BC) for hs in (128, 4096, 5120, 8192)}
    finally:
        sys.path.remove(tree)
(ma, ea), (mb, eb) = build(tree_a), build(tree_b)
bad = 0
for hs in (128, 4096, 5120, 8192):
    for bs in (1, 32, 128, 4113, 16384):
        torch.manual_seed(hs * 7 + bs)
        x = torch.randn(bs, hs, device="cuda", dtype=torch.float16) * 3
        w = torch.randn(hs, device="cuda", dtype=torch.float16)
        oa, ob = torch.full_like(x, float("nan")), torch.full_like(x, float("nan"))
        ea[hs](*ma._kernel_args(x, w, oa)); eb[hs](*mb._kernel_args(x, w, ob)); torch.cuda.synchronize()
        same = torch.equal(oa.view(torch.int16), ob.view(torch.int16))
        bad += not same
        print(hs, bs, "bit-identical" if same else "DIFFERENT")
print("bitwise", "ok" if bad == 0 else f"{bad} DIFFERENT")
