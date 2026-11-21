"""B0 correctness by calling rmsnorm.run_test directly for every CONFIG on sm_90a.

This BYPASSES tirx_kernels.test's reference-requirement gate (pinned flashinfer git commit, CUTLASS DSL
version), so it prints the exact environment it ran in: what the reference was is part of the result."""
import importlib.metadata as md, json, sys, traceback
import torch
from tirx_kernels.norm import rmsnorm as m
from tirx_kernels.reference_requirements import unmet_reference_requirements

BC = {"cuda": {"arch": "sm_90a"}}
def _version(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None
env = {d: _version(d) for d in ("torch", "flashinfer-python", "nvidia-cutlass-dsl", "apache-tvm-tirx")}
import flashinfer, tvm_tirx
env["flashinfer_direct_url"] = (md.distribution("flashinfer-python").read_text("direct_url.json") or "").strip()
env["tvm_tirx_from"] = tvm_tirx.__file__
print(json.dumps({"env": env, "python": sys.executable, "gpu": torch.cuda.get_device_name(),
                  "unmet_reference_requirements": unmet_reference_requirements(m.KERNEL_META["reference_requirements"])}))
fails = 0
for cfg in m.CONFIGS:
    try:
        m.run_test(cfg["hidden_size"], cfg["batch_size"], backend_config=BC)
        print("PASS", cfg["label"], flush=True)
    except Exception as e:  # noqa: BLE001
        fails += 1
        print("FAIL", cfg["label"], str(e)[:400], flush=True)
print(f"direct run_test: {len(m.CONFIGS) - fails}/{len(m.CONFIGS)} passed")
