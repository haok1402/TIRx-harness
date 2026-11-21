"""C8: what synccheck / racecheck / NumSim say about the sm_90a GEMM (B1) and FP8 GEMM (B2)."""
import sys, traceback
import numpy as np
from tirx_harness import synccheck
try:
    from tirx_harness import racecheck
except ImportError:
    racecheck = None
from tirx_kernels.gemm import fp16_bf16_gemm_sm90 as g

M = N = 256; K = 128
kernel = g.make_kernel("bf16", M, N, K, num_clusters=1)
rng = np.random.default_rng(0)
import ml_dtypes
inputs = {"A": rng.standard_normal((M, K)).astype(ml_dtypes.bfloat16), "B": rng.standard_normal((N, K)).astype(ml_dtypes.bfloat16),
          "C": np.zeros((M, N), dtype=ml_dtypes.bfloat16)}
for name, fn in (("synccheck", synccheck), ("racecheck", racecheck)):
    if fn is None:
        continue
    try:
        report = fn(kernel, inputs=inputs)
        print(f"== {name}: verdict {report.verdict}")
        for f in report.findings[:12]:
            print("   ", getattr(f, "kind", "?"), "|", str(getattr(f, "message", f))[:200])
    except Exception:
        print(f"== {name}: raised"); traceback.print_exc(limit=3)
