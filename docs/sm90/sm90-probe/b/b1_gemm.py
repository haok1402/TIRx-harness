# SPDX-License-Identifier: Apache-2.0
"""B1: TIRx-kernels fp16_bf16_gemm_sm90 on H200 vs cuBLAS. Correctness, ptxas remarks, interleaved L2-flushed timing.
argv: [quick|gated|all] [dtype...]"""
import json, math, sys, time
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import gpu_state, interleaved
from tirx_kernels.gemm import fp16_bf16_gemm_sm90 as g

mode = sys.argv[1] if len(sys.argv) > 1 else "gated"
dtypes = sys.argv[2:] or ["bf16", "fp16"]
shapes = {"quick": [(4096, 4096, 4096)], "gated": g.GATED_SHAPES, "all": g.GATED_SHAPES + g.TRACKED_SHAPES}[mode]
BC = {"cuda": {"arch": "sm_90a"}}
print(json.dumps({"state": gpu_state()}), flush=True)
ratios = {d: [] for d in dtypes}
for d in dtypes:
    for M, N, K in shapes:
        rec = {"dtype": d, "shape": [M, N, K]}
        try:
            t0 = time.time()
            ex, clusters = g.compile_persistent(d, M, N, K, backend_config=BC)
            rec["compile_s"] = round(time.time() - t0, 1); rec["clusters"] = clusters
            rec["remarks"] = [str(r) for r in getattr(ex, "perf_remarks", ())]
            A, B, C = g.prepare_data(d, M, N, K)
            ex(A, B, C); torch.cuda.synchronize()
            ref = A @ B.T
            rec["rel_err_vs_cublas"] = float((C.float() - ref.float()).norm() / ref.float().norm())
            rec["bit_exact"] = bool(torch.equal(C, ref))
            Cr = torch.empty_like(C)
            t = interleaved({"tirx": lambda: ex(A, B, C), "cublas": lambda: torch.matmul(A, B.T, out=Cr)}, reps=30)
            fl = 2 * M * N * K
            rec.update(tirx_us=round(t["tirx"], 1), cublas_us=round(t["cublas"], 1),
                       tirx_tflops=round(fl / t["tirx"] / 1e6, 1), cublas_tflops=round(fl / t["cublas"] / 1e6, 1),
                       ratio=round(t["cublas"] / t["tirx"], 3))
            if (M, N, K) in g.GATED_SHAPES:
                ratios[d].append(rec["ratio"])
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:1500]
        print(json.dumps(rec), flush=True)
for d, r in ratios.items():
    if r:
        print(json.dumps({"dtype": d, "gated_geomean": round(math.exp(sum(map(math.log, r)) / len(r)), 3), "n": len(r), "min": min(r)}), flush=True)
print(json.dumps({"state_after": gpu_state()}), flush=True)
