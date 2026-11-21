"""B2 diagnosis: promoted kernel vs the same kernel without promotion (plain fp8 accumulate) vs DeepGEMM."""
import json, sys, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import interleaved
import deep_gemm
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
from tirx_kernels.runner import compile_kernel
BC = {"cuda": {"arch": "sm_90a"}}
for M, N, K in [(4096, 7168, 2048), (4096, 4096, 7168)]:
    a8, sfa, b8, sfb, d = g.prepare_data(M, N, K)
    sa_t = g.sfa_mn_major(sfa); dd = torch.empty_like(d)
    sa_dg = deep_gemm.get_mn_major_tma_aligned_tensor(sfa)
    fns = {"deepgemm": lambda: deep_gemm.fp8_gemm_nt((a8, sa_dg), (b8, sfb), dd)}
    for bn in (128, 192):
        for pr in (True, False):
            ex = compile_kernel(g.make_kernel(M, N, K, bn=bn, backend_config=BC, _promote=pr), backend_config=BC)
            fns[f"bn{bn}_{'promote' if pr else 'nopromote'}"] = (lambda ex=ex: ex(a8, sa_t, b8, sfb, d))
    t = interleaved(fns, reps=30)
    fl = 2 * M * N * K
    print(json.dumps({"shape": [M, N, K], **{k: round(fl / v / 1e6) for k, v in t.items()}}), flush=True)
