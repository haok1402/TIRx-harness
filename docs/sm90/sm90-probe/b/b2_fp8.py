"""B2: fp8_blockscale_gemm_sm90 vs DeepGEMM fp8_gemm_nt (sm90 1D2D) on H200: calc_diff vs fp32 reference and vs
DeepGEMM, remarks, interleaved L2-flushed timing.  argv: [quick|gated|all] [bn...]"""
import json, math, sys, time
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import gpu_state, interleaved
import deep_gemm
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g

mode = sys.argv[1] if len(sys.argv) > 1 else "gated"
import re
def _cfg(x):
    if x == "auto":
        return None
    m = re.fullmatch(r"(\d+)x(\d+)([mn]?)(?:r(\d+))?L?C?B?", x)
    return int(m[1]), int(m[2]), (m[3] or None)
def _kw(x):
    kw = {}
    if x.endswith("B"):  # cluster index from block_idx // 2
        kw["_cid_from_bid"] = True
        x = x[:-1]
    if x.endswith("C"):  # non-cluster kernel, launched with cluster dims 2
        kw["_launch_cluster"] = 2
        x = x[:-1]
    if x.endswith("L"):  # cluster launch, local loads (no multicast)
        kw["_multicast"] = False
        x = x[:-1]
    m = re.search(r"r(\d+)$", x)
    if m:
        kw["regs"] = (int(m[1]), (168 * 384 - int(m[1]) * 128) // 256 // 8 * 8)
    return kw
bns = sys.argv[2:] or ["auto"]
shapes = {"quick": [(4096, 7168, 2048)], "gated": g.GATED_SHAPES, "few": [(4096, 7168, 2048), (4096, 4096, 7168), (4096, 32768, 512)], "all": g.GATED_SHAPES + g.TRACKED_SHAPES}[mode]
BC = {"cuda": {"arch": "sm_90a"}}
print(json.dumps({"state": gpu_state(), "deep_gemm": getattr(deep_gemm, "__version__", "?")}), flush=True)
ratios = {bn: [] for bn in bns}
for M, N, K in shapes:
    torch.manual_seed(0)
    a8, sfa, b8, sfb, d = g.prepare_data(M, N, K)
    ref = g.reference(a8, sfa, b8, sfb)
    dd = torch.empty_like(d)
    sa_dg = deep_gemm.get_mn_major_tma_aligned_tensor(sfa)
    f_dg = lambda: deep_gemm.fp8_gemm_nt((a8, sa_dg), (b8, sfb), dd)
    f_dg(); torch.cuda.synchronize()
    for bn in bns:
        rec = {"bn": bn, "cfg": list(_cfg(bn) or g.pick_config(M, N, K)), "shape": [M, N, K]}
        try:
            t0 = time.time()
            ex = g.compile_persistent(M, N, K, config=_cfg(bn), backend_config=BC, **_kw(bn))
            rec["compile_s"] = round(time.time() - t0, 1)
            rec["remarks"] = [str(r) for r in getattr(ex, "perf_remarks", ())]
            sa_t = g.sfa_mn_major(sfa)
            d.fill_(float("nan"))
            ex(a8, sa_t, b8, sfb, d); torch.cuda.synchronize()
            rec["nan"] = int(d.isnan().sum())
            rec["diff_vs_ref"] = g.calc_diff(d, ref)
            rec["diff_deepgemm_vs_ref"] = g.calc_diff(dd, ref)
            rec["diff_vs_deepgemm"] = g.calc_diff(d, dd)
            t = interleaved({"tirx": lambda: ex(a8, sa_t, b8, sfb, d), "deepgemm": f_dg}, reps=30)
            fl = 2 * M * N * K
            rec.update(tirx_us=round(t["tirx"], 1), dg_us=round(t["deepgemm"], 1),
                       tirx_tflops=round(fl / t["tirx"] / 1e6, 1), dg_tflops=round(fl / t["deepgemm"] / 1e6, 1),
                       ratio=round(t["deepgemm"] / t["tirx"], 3))
            if (M, N, K) in g.GATED_SHAPES:
                ratios[bn].append(rec["ratio"])
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[-1500:]
        print(json.dumps(rec), flush=True)
for bn, r in ratios.items():
    if r:
        print(json.dumps({"bn": bn, "gated_geomean": round(math.exp(sum(map(math.log, r)) / len(r)), 3), "n": len(r), "min": min(r)}), flush=True)
print(json.dumps({"state_after": gpu_state()}), flush=True)
