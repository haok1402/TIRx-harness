"""A3 registry check (EXTRA_ARCHS=sm_90a adds archs beyond runtime_cuda_archs): for every TIRx-kernels registry kernel x CONFIG x runtime arch, generate the CUDA
source with whichever tvm-tirx is on PYTHONPATH and record (status, sha1 of source). Run once per tree,
then diff. argv: out.jsonl [max_configs_per_kernel]"""
import hashlib, json, multiprocessing, os, sys, traceback
os.environ.setdefault("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")


def _compile(func, bc):
    import tvm_tirx
    from tirx_kernels.runner import compile_kernel, cuda_target
    if isinstance(func, tvm_tirx.IRModule):  # some kernels return a whole module
        return tvm_tirx.tirx.compile(func, target=cuda_target(backend_config=bc), pipeline="tirx")
    return compile_kernel(func, backend_config=bc)


def one(name):
    from tirx_kernels.registry import discover_kernels
    from tirx_kernels.runner import compile_kernel
    mod = discover_kernels()[name]
    rows = []
    extra = [a for a in os.environ.get("EXTRA_ARCHS", "").split(",") if a]
    for arch in list(mod.KERNEL_META.get("runtime_cuda_archs", [])) + extra:
        for cfg in mod.CONFIGS[: int(sys.argv[2]) if len(sys.argv) > 2 else None]:
            params = {k: v for k, v in cfg.items() if k != "label"}
            bc = {"cuda": {"arch": arch}}
            rec = {"kernel": name, "config": cfg.get("label", "default"), "arch": arch}
            try:
                funcs = mod.get_kernel(**params, backend_config=bc)
                funcs = funcs if isinstance(funcs, (list, tuple)) else [funcs]
                src = "\n".join(_compile(f, bc).mod.imports[0].inspect_source() for f in funcs)
                rec.update(status="ok", sha=hashlib.sha1(src.encode()).hexdigest()[:16])
            except Exception as e:  # noqa: BLE001
                rec.update(status=type(e).__name__, msg=str(e).splitlines()[0][:160] if str(e) else "")
            rows.append(rec)
    return rows


if __name__ == "__main__":
    from tirx_kernels.registry import discover_kernels
    names = sorted(discover_kernels())
    if os.environ.get("KERNELS"):
        names = [n for n in names if n in os.environ["KERNELS"].split(",")]
    with multiprocessing.get_context("spawn").Pool(int(os.environ.get("JOBS", "32"))) as pool, open(sys.argv[1], "w") as out:
        for rows in pool.imap_unordered(one, names):
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
    print("kernels", len(names))
