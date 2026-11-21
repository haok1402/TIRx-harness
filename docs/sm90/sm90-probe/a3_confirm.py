"""A3: every registry row VerifyPTXTarget rejects must also fail today's real compile (NVRTC) on main.

Run with upstream main on PYTHONPATH and without TVM_TIRX_COMPILE_FORCE_FALLBACK, so device code really
goes through NVRTC. argv: registry_a3.jsonl out.jsonl. Env JOBS (default 12).
"""

import json
import multiprocessing
import os
import sys


def _compile(func, bc):
    # Not imported from a3_registry: importing it turns on the source-only fallback.
    import tvm_tirx
    from tirx_kernels.runner import compile_kernel, cuda_target

    if isinstance(func, tvm_tirx.IRModule):
        return tvm_tirx.tirx.compile(func, target=cuda_target(backend_config=bc), pipeline="tirx")
    return compile_kernel(func, backend_config=bc)


def one(row):
    from tirx_kernels.registry import discover_kernels

    mod = discover_kernels()[row["kernel"]]
    cfg = next(c for c in mod.CONFIGS if c.get("label", "default") == row["config"])
    params = {k: v for k, v in cfg.items() if k != "label"}
    bc = {"cuda": {"arch": row["arch"]}}
    out = {k: row[k] for k in ("kernel", "config", "arch")}
    try:
        funcs = mod.get_kernel(**params, backend_config=bc)
        for f in funcs if isinstance(funcs, (list, tuple)) else [funcs]:
            _compile(f, bc)
        out["main"] = "compiled"  # A3 rejected something that compiles today: a false rejection
    except Exception as e:  # noqa: BLE001
        text = str(e)
        nvrtc = "ptxas" in text or "NVRTC" in text or "nvrtc" in text
        out["main"] = "nvrtc_rejects" if nvrtc else f"other:{type(e).__name__}"
        out["msg"] = next((l for l in text.splitlines() if "error" in l.lower()), text[:200])[:200]
    return out


if __name__ == "__main__":
    assert not os.environ.get("TVM_TIRX_COMPILE_FORCE_FALLBACK"), "needs the real NVRTC path"
    rows = [r for r in map(json.loads, open(sys.argv[1])) if r["status"] == "PTXTargetError"]
    sample = int(os.environ.get("SAMPLE", "0"))  # per kernel: first, middle, last ... config
    if sample:
        by_kernel = {}
        for r in rows:
            by_kernel.setdefault(r["kernel"], []).append(r)
        rows = []
        for group in by_kernel.values():
            picks = sorted({round(i * (len(group) - 1) / max(sample - 1, 1)) for i in range(sample)})
            rows += [group[i] for i in picks]
    with multiprocessing.get_context("spawn").Pool(int(os.environ.get("JOBS", "12"))) as pool, open(
        sys.argv[2], "w"
    ) as out:
        for r in pool.imap_unordered(one, rows):
            out.write(json.dumps(r) + "\n")
            out.flush()
    print("rejected rows checked", len(rows))
