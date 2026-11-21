"""B2-design: compile-only register budget sweep for fp8_blockscale_gemm_sm90 at the real 384-thread launch.
Reports ptxas resource usage, remarks (A10), stages and SASS chain structure per N tile."""
import json, re, subprocess, sys, tempfile
from tvm_tirx.backend.cuda.runtime import occupancy
from tirx_kernels.gemm import fp8_blockscale_gemm_sm90 as g
from tirx_kernels.runner import compile_kernel
BC = {"cuda": {"arch": "sm_90a", "perf_remarks": "ignore"}}
for bn in [int(x) for x in sys.argv[2:]] or (128, 192, 256):
    rec = {"bn": bn, "stages": g._stages(bn), "acc_regs_per_thread": 2 * bn // 2}
    try:
        ex = compile_kernel(g.make_kernel(4096, 7168, 2048, bn=bn, backend_config=BC), backend_config=BC)
        rec["remarks"] = [f"{r.code}: {r.message}"[:160] for r in getattr(ex, "perf_remarks", ())]
        binary, name = occupancy.kernel_binary(ex)
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(binary); f.flush()
            res = subprocess.run(["cuobjdump", "-res-usage", f.name], capture_output=True, text=True).stdout
            chk = subprocess.run([sys.executable, sys.argv[1], f.name, "--min-chain", "4"], capture_output=True, text=True)
        m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", res)
        rec.update(reg=int(m[1]), stack=int(m[2]), local=int(m[4])) if m else rec.update(res=res[-300:])
        rec["chains"] = chk.stdout.strip()[:200]
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[-1200:]
    print(json.dumps(rec), flush=True)
