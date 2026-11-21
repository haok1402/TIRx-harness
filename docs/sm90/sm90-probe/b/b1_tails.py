"""B1 tails: odd shapes vs cuBLAS (and an fp32 reference), plus timing at each."""
import json, sys, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import interleaved
from tirx_kernels.gemm import fp16_bf16_gemm_sm90 as g
BC = {"cuda": {"arch": "sm_90a"}}
for d in ("bf16", "fp16"):
    for M, N, K in g.TAIL_SHAPES:
        rec = {"dtype": d, "shape": [M, N, K]}
        try:
            ex, _ = g.compile_persistent(d, M, N, K, backend_config=BC)
            A, B, C = g.prepare_data(d, M, N, K)
            C.fill_(float("nan"))  # every element must be written
            ex(A, B, C); torch.cuda.synchronize()
            ref32 = A.float() @ B.float().T
            ref = A @ B.T
            rec["nan"] = int(C.isnan().sum())
            rec["rel_err_vs_fp32"] = float((C.float() - ref32).norm() / ref32.norm())
            rec["cublas_rel_err_vs_fp32"] = float((ref.float() - ref32).norm() / ref32.norm())
            rec["bit_exact_cublas"] = bool(torch.equal(C, ref))
            g.run_test(d, M, N, K, backend_config=BC)
            rec["run_test"] = "pass"
            Cr = torch.empty_like(C)
            t = interleaved({"tirx": lambda: ex(A, B, C), "cublas": lambda: torch.matmul(A, B.T, out=Cr)}, reps=30)
            rec["ratio"] = round(t["cublas"] / t["tirx"], 3)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[-800:]
        print(json.dumps(rec), flush=True)
