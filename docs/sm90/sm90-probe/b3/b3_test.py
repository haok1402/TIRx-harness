"""B3 correctness: every CONFIG of fa3_forward_sm90 with perf_remarks="error", plus register usage."""
import json, subprocess, sys, tempfile, traceback
import torch
from tirx_kernels.attention import fa3_forward_sm90 as fa
from tirx_kernels.runner import compile_kernel
from tvm_tirx.backend.cuda.runtime import occupancy

BC = {"cuda": {"arch": "sm_90a", "perf_remarks": "error"}}
only = sys.argv[1] if len(sys.argv) > 1 else None
for causal in (False, True):
    try:
        ex = compile_kernel(fa.get_kernel(4, causal, backend_config=BC), backend_config=BC)
        binary, _ = occupancy.kernel_binary(ex)
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(binary); f.flush()
            res = subprocess.run(["cuobjdump", "-res-usage", f.name], capture_output=True, text=True).stdout
        print(json.dumps({"compile": "clean", "causal": causal, "res": [l.strip() for l in res.splitlines() if "REG" in l]}), flush=True)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"compile": "FAILED", "causal": causal, "error": str(e)[-3000:]}), flush=True)
        sys.exit(1)
if only == "compile":
    sys.exit(0)
fails = 0
for cfg in fa.CONFIGS:
    if only and only not in cfg["label"]:
        continue
    try:
        fa.run_test(cfg["seq_lens"], cfg["num_heads"], cfg["causal"], backend_config=BC)
        print("PASS", cfg["label"], flush=True)
    except Exception as e:  # noqa: BLE001
        fails += 1
        print("FAIL", cfg["label"], str(e)[:600].replace("\n", " | "), flush=True)
print(f"b3 tests: {fails} failed")
