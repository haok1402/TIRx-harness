# SPDX-License-Identifier: Apache-2.0
"""B0 (smoke): TIRx-kernels rmsnorm on sm_90a. Correctness on every CONFIG vs flashinfer, then interleaved
timing vs flashinfer and torch. One JSON line per row."""
import json, sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import gpu_state, interleaved
import flashinfer
from tirx_kernels.norm import rmsnorm as m
from tirx_kernels.runner import compile_kernel

BC = {"cuda": {"arch": "sm_90a"}}
print(json.dumps({"state": gpu_state()}), flush=True)
fails = 0
for cfg in m.CONFIGS:
    try:
        m.run_test(cfg["hidden_size"], cfg["batch_size"], backend_config=BC)
    except Exception as e:  # noqa: BLE001
        fails += 1
        print(json.dumps({"test": cfg["label"], "ok": False, "err": str(e)[:300]}), flush=True)
print(json.dumps({"tests": len(m.CONFIGS), "failed": fails}), flush=True)

shapes = [(hs, bs) for hs in (4096, 5120, 8192) for bs in (1, 32, 128, 4113, 16384)]
exes = {}
for hs, bs in shapes:
    if hs not in exes:
        exes[hs] = compile_kernel(m.get_kernel(hs, backend_config=BC), backend_config=BC)
    x, w = m.prepare_data(bs, hs)
    out = torch.empty_like(x); ref = torch.empty_like(x)
    args = m._kernel_args(x, w, out)
    t = interleaved({
        "tirx": lambda: exes[hs](*args),
        "flashinfer": lambda: flashinfer.norm.rmsnorm(x, w, m.eps, enable_pdl=False, out=ref),
        "torch": lambda: torch.nn.functional.rms_norm(x, (hs,), w, m.eps),
    })
    gb = 2 * x.numel() * 2 / 1e9
    print(json.dumps({"hs": hs, "bs": bs, **{f"{k}_us": round(v, 2) for k, v in t.items()},
                      "ratio_vs_flashinfer": round(t["flashinfer"] / t["tirx"], 3),
                      "tirx_TBps": round(gb / t["tirx"] * 1e3, 2)}), flush=True)
print(json.dumps({"state_after": gpu_state()}), flush=True)
