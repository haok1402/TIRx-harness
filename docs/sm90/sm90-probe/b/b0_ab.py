"""A/B of the folded sm_90 RMSNorm kernels against the M3 candidate's, same process, interleaved."""
import importlib.util, json, sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import interleaved
from tirx_kernels.norm import _rmsnorm_sm90 as K, rmsnorm as m
from tirx_kernels.runner import compile_kernel

spec = importlib.util.spec_from_file_location("m3cand", sys.argv[1]); cand = importlib.util.module_from_spec(spec); spec.loader.exec_module(cand)
BC = {"cuda": {"arch": "sm_90a"}}
c = lambda f: compile_kernel(f.func, backend_config=BC)
for hs, bs in ((4096, 1), (4096, 128), (8192, 32)):
    warps, ldw = K.SMALL_CFG[hs]
    x, w = m.prepare_data(bs, hs); y = torch.empty_like(x)
    ctr = torch.zeros(4, dtype=torch.uint32, device="cuda")
    xv, yv = x.view(-1), y.view(-1)
    single, orig = c(K._make_small(hs, warps, ldw)), c(cand.make_small_kernel(hs, warps, cand.LD_NC, ldw))
    t = interleaved({"single": lambda: single(xv, w, yv, ctr, bs, bs),
                     "m3": lambda: orig(xv, w, yv, bs)})
    print(json.dumps({"small": [hs, bs], **{k: round(v, 2) for k, v in t.items()}}), flush=True)
for hs, bs in ((4096, 4113), (4096, 16384)):
    warps, k, S = K.RING_CFG[hs]
    x, w = m.prepare_data(bs, hs); y = torch.empty_like(x); xv, yv = x.view(-1), y.view(-1)
    grid = min(bs, 132 * k)
    mine = c(K._make_ring(hs, warps, S)); orig = c(cand.make_dyn_kernel(hs, warps, S))
    c64 = torch.zeros(4, dtype=torch.uint32, device="cuda"); c32 = torch.zeros(2, dtype=torch.uint32, device="cuda")
    t = interleaved({"mine_u64": lambda: mine(xv, w, yv, c64, bs, grid), "m3_u32": lambda: orig(xv, w, yv, c32, bs, grid)})
    print(json.dumps({"ring": [hs, bs], **{k: round(v, 2) for k, v in t.items()}, "ctr64": c64.cpu().tolist(), "ctr32": c32.cpu().tolist()}), flush=True)
