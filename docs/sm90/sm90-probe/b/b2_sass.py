"""B2 acceptance: every pick_config branch compiles with perf_remarks="error"; SASS chain structure."""
import subprocess, sys, tempfile
from tvm_tirx.backend.cuda.runtime import occupancy
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
BC = {"cuda": {"arch": "sm_90a", "perf_remarks": "error"}}
for M, N, K in [(4096, 2112, 7168), (4096, 24576, 1536), (4096, 7168, 2048)]:
    ex = g.compile_persistent(M, N, K, backend_config=BC)
    binary, _ = occupancy.kernel_binary(ex)
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(binary); f.flush()
        out = subprocess.run([sys.executable, sys.argv[1], f.name, "--min-chain", "4"], capture_output=True, text=True)
    print(g.pick_config(M, N, K), "perf_remarks=error: compiled; chain check rc", out.returncode, out.stdout.strip())
