# SPDX-License-Identifier: Apache-2.0
"""Launch the warp-specialized setmaxnreg variant given on argv (from probe_setmaxnreg.py) and report whether it
completes. Run under `timeout`: an over-budget setmaxnreg.inc blocks forever."""
import sys, time, torch, tvm_tirx
sys.argv, case = sys.argv[:1], sys.argv[1]
import importlib.util
spec = importlib.util.spec_from_file_location("p", __file__.replace("_run.py", ".py"))
src = open(spec.origin).read().split("VARIANTS = [")[0]          # reuse make() without running the sweep
ns = {}; exec(compile(src, spec.origin, "exec"), ns)
attrs = {"ws384_minblocks1": {"min_blocks_per_sm": 1}, "ws384_minblocks2": {"min_blocks_per_sm": 2},
         "ws384_default": None}[case]
mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": ns["make"](384, attrs, "ws")}), target=ns["TGT"], pipeline="tirx")
a = torch.zeros(1024, device="cuda")
t = time.time(); mod(a); torch.cuda.synchronize()
print(f"{case}: completed in {time.time() - t:.3f}s, A[0]={a[0].item()} A[200]={a[200].item()}", flush=True)
