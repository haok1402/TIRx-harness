"""C8: with WGMMA skipped, synccheck must still catch a real barrier bug in an SM90 kernel."""
import importlib.util, sys
import numpy as np, ml_dtypes
from tirx_harness import synccheck
spec = importlib.util.spec_from_file_location("b1_broken", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
M = N = 256; K = int(sys.argv[2]) if len(sys.argv) > 2 else 512
inputs = {"A": np.ones((M, K), ml_dtypes.bfloat16), "B": np.ones((N, K), ml_dtypes.bfloat16), "C": np.zeros((M, N), ml_dtypes.bfloat16)}
report = synccheck(m.make_kernel("bf16", M, N, K, num_clusters=1), inputs=inputs)
print("verdict", report.verdict)
for f in report.findings[:6]:
    print("   ", getattr(f, "kind", "?"), "|", str(getattr(f, "message", f))[:1500])
