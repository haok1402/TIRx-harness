"""C8: synccheck/racecheck on the B2 FP8 GEMM (per-warp stage release), small config with stage reuse."""
import numpy as np, ml_dtypes
from tirx_harness import synccheck, racecheck
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
M, N, K = 256, 256, 1024  # 8 k-blocks over 3 stages
for cfg in [(256, 128, None), (128, 192, None)]:
    kern = g.make_kernel(M, N, K, bm=cfg[0], bn=cfg[1], cluster=cfg[2], num_clusters=1)
    kb, ms, nb = K // 128, g.sfa_stride(M), -(-N // 128)
    inputs = {"A": np.ones((M, K), ml_dtypes.float8_e4m3fn), "SFA": np.ones((kb, ms), np.float32),
              "B": np.ones((N, K), ml_dtypes.float8_e4m3fn), "SFB": np.ones((nb, kb), np.float32),
              "D": np.zeros((M, N), ml_dtypes.bfloat16)}
    for name, fn in (("synccheck", synccheck), ("racecheck", racecheck)):
        r = fn(kern, inputs=inputs)
        kinds = [getattr(f, "kind", "?") for f in r.findings]
        print(f"B2 {cfg} {name}: verdict {r.verdict} findings {kinds}")
        for f in r.findings:
            if getattr(f, "kind", "") != "unmodelled_operations":
                print("     ", str(getattr(f, "message", f))[:400])
