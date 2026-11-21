import json, subprocess, sys, tempfile
from tvm_tirx.backend.cuda.runtime import occupancy
from tirx_kernels.gemm import fp16_bf16_gemm_sm90 as g
BC = {"cuda": {"arch": "sm_90a", "perf_remarks": "error"}}
for d in ("bf16", "fp16"):
    ex, _ = g.compile_persistent(d, 4096, 4096, 4096, backend_config=BC)  # error mode: raises on any remark
    binary, name = occupancy.kernel_binary(ex)
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(binary); f.flush()
        out = subprocess.run([sys.executable, sys.argv[1], f.name, "--min-chain", "4"], capture_output=True, text=True)
    print(d, "perf_remarks=error: compiled;", "chain check rc", out.returncode, out.stdout.strip())
