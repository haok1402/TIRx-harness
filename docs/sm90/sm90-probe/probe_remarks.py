# SPDX-License-Identifier: Apache-2.0
"""A10 controls: TIRx kernels that make ptxas print (or not print) performance remarks, compiled
through TIRx's own NVRTC driver for sm_90a. Prints the raw NVRTC logs, the SASS chain structure, and
(when the tree has A10) the remarks / warnings / error-mode behaviour."""
import json, os, re, subprocess, sys, tempfile, warnings
import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.support import nvcc
from tvm_tirx.backend.cuda.backend_config import compiler_options, resolve_backend_config

N = 128
ACC = N // 2  # f32 accumulators per thread for m64nNk16
MMA = f"wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16"


def chain_kernel(hazard: bool, k_steps: int = 2):
    """One warpgroup, one WGMMA chain of k_steps MMAs on (uninitialized) smem; compile-only control.
    hazard=True puts a non-wgmma write of an accumulator register between the first two MMAs."""

    # fmt: off
    @T.function
    def main(Out: T.Tensor((128, ACC), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tx = T.cuda.thread_idx("x")
        A_smem = T.alloc_shared((64, 64), "bfloat16", align=1024)
        B_smem = T.alloc_shared((N, 64), "bfloat16", align=1024)
        descA: T.uint64
        descB: T.uint64
        acc = T.alloc_tensor(ty_args=[T.Tensor((ACC,), "float32", scope="local")])
        for i in T.serial(0, ACC):
            acc[i] = T.float32(0.0)
        T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descA), A_smem.data, 1, 64, 3)
        T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descB), B_smem.data, 1, 64, 3)
        T.ptx.wgmma.fence.sync.aligned()
        for k in T.unroll(0, k_steps):
            T.ptx[MMA](*[acc[i] for i in range(ACC)], descA + T.uint64(2 * k), descB + T.uint64(2 * k), True, 1, 1, 0, 0)
            if T.constexpr(hazard):
                if k == 0:
                    acc[0] = acc[0] + T.Cast("float32", tx >> 31)
        T.ptx.wgmma.commit_group.sync.aligned()
        T.ptx.wgmma.wait_group.sync.aligned(0)
        for i in T.serial(0, ACC):
            Out[tx, i] = acc[i]
    # fmt: on
    return main


def local_mem_kernel():
    # fmt: off
    @T.function
    def main(Out: T.Tensor((128,), "float32"), idx: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tx = T.cuda.thread_idx("x")
        buf = T.alloc_local((256,), "float32")
        for i in T.serial(0, 256):
            buf[(i * idx + tx) % 256] = T.Cast("float32", tx + i)
        Out[tx] = buf[(tx + idx) % 256]
    # fmt: on
    return main


def reg_pressure_kernel(min_blocks=None):
    """128 live f32 values per thread, all unrolled: needs > 128 registers. min_blocks_per_sm=16 with
    128 threads caps ptxas at 32 registers (__launch_bounds__), which forces spills."""
    attrs = {} if min_blocks is None else {"kernel_attrs": T.cuda.KernelAttributes(min_blocks_per_sm=min_blocks)}

    # fmt: off
    @T.function
    def main(A: T.Tensor((128, 128), "float32"), Out: T.Tensor((128, 128), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128), **attrs)
        tx = T.cuda.thread_idx("x")
        v = T.alloc_local((128,), "float32")
        for i in T.unroll(0, 128):
            v[i] = A[tx, i]
        for r in T.serial(0, 4):
            for i in T.unroll(0, 128):
                v[i] = v[i] * v[(i + 1) % 128] + v[(i + 7) % 128]
        for i in T.unroll(0, 128):
            Out[tx, i] = v[i]
    # fmt: on
    return main


def cuda_source(func):
    with tempfile.TemporaryDirectory():
        os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
        try:
            mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"}), pipeline="tirx")
        finally:
            del os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"]
    return mod.mod.imports[0].inspect_source("cuda") or mod.mod.imports[0].inspect_source()


def sass_chain(cubin: bytes):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin); f.flush()
        sass = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
    g = [l for l in sass.splitlines() if "GMMA" in l]
    return {"gmma": len(g), "gmma_gsb0": sum("gsb0" in l for l in g), "wg_arrive": sass.count("WARPGROUP.ARRIVE")}


CASES = {
    "chain2_clean": (lambda: chain_kernel(False, 2), {}),
    "chain4_clean": (lambda: chain_kernel(False, 4), {}),
    "chain2_hazard": (lambda: chain_kernel(True, 2), {}),
    "local_mem": (local_mem_kernel, {}),
    "regs_default": (reg_pressure_kernel, {}),
    "regs_minblocks16_spill": (lambda: reg_pressure_kernel(16), {}),
}

for name, (make, extra) in CASES.items():
    rec = {"case": name}
    try:
        src = cuda_source(make())
        if os.environ.get("PROBE_CU_OUT"):
            os.makedirs(os.environ["PROBE_CU_OUT"], exist_ok=True)
            open(os.path.join(os.environ["PROBE_CU_OUT"], name + ".cu"), "w").write(src)
        cfg = resolve_backend_config({"arch": "sm_90a", **extra})
        binary, log = nvcc._compile_cuda_nvrtc(src, "cubin", "sm_90a", compiler_options(cfg))
        rec.update(sass_chain(bytes(binary)))
        rec["log"] = [l for l in log.splitlines() if "ptxas" in l or "bytes stack" in l]
        try:  # A10, when present in this tree
            from tvm_tirx.backend.cuda.compiler import compile_source
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                res = compile_source(src, {"cuda": {"arch": "sm_90a", **extra}})
            rec["remarks"] = [str(r) for r in getattr(res, "remarks", ())]
            rec["warnings"] = [f"{x.category.__name__}: {x.message}" for x in w]
            try:
                compile_source(src, {"cuda": {"arch": "sm_90a", "perf_remarks": "error", **extra}})
                rec["error_mode"] = "compiled"
            except Exception as e:  # noqa: BLE001
                rec["error_mode"] = f"raised {type(e).__name__}: {str(e).splitlines()[0][:160]}"
        except Exception as e:  # noqa: BLE001
            rec["a10"] = f"{type(e).__name__}: {e}"[:200]
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:600]
    print(json.dumps(rec), flush=True)
