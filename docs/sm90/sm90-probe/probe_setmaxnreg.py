# SPDX-License-Identifier: Apache-2.0
"""When does ptxas ignore setmaxnreg (C7508) in TIRx-generated code? Compile variants for sm_90a through TIRx's
NVRTC path and report: the kernel signature attributes, ptxas remarks, SASS USETMAXREG count, entry registers."""
import json, os, re, subprocess, tempfile, warnings
import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.backend.cuda.compiler import compile_source

TGT = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})


def make(block, attrs, body):
    kw = {} if attrs is None else {"kernel_attrs": T.cuda.KernelAttributes(**attrs)}

    # fmt: off
    @T.function
    def main(A: T.Tensor((1024,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=block), **kw)
        tx = T.cuda.thread_idx("x")
        wg = tx // 128
        if T.constexpr(body == "flat_dec"):
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](32)
        elif T.constexpr(body == "flat_inc"):
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](232)
        elif T.constexpr(body == "ws"):          # warpgroup-uniform branches, as in a WS kernel
            if wg == 0:
                T.ptx["setmaxnreg.dec.sync.aligned.u32"](40)
                A[tx] = A[tx] + 1.0
            else:
                T.ptx["setmaxnreg.inc.sync.aligned.u32"](232)
                A[tx] = A[tx] * 2.0
        elif T.constexpr(body == "divergent"):   # data-dependent (non-warpgroup-uniform) branch
            if A[tx] > 0.0:
                T.ptx["setmaxnreg.inc.sync.aligned.u32"](232)
            A[tx] = A[tx] + 1.0
        A[tx + 512] = A[tx]
    # fmt: on
    return main


def source(func):
    os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=TGT, pipeline="tirx")
    finally:
        del os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"]
    return mod.mod.imports[0].inspect_source()


VARIANTS = [
    # name, block, kernel_attrs, body
    ("test_ptx_setmaxnreg_dec", 128, None, "flat_dec"),
    ("test_ptx_setmaxnreg_inc", 128, None, "flat_inc"),
    ("flat_dec_minblocks1", 128, {"min_blocks_per_sm": 1}, "flat_dec"),
    ("flat_dec_maxnreg64", 128, {"max_registers_per_thread": 64}, "flat_dec"),
    ("ws384_default", 384, None, "ws"),
    ("ws384_minblocks1", 384, {"min_blocks_per_sm": 1}, "ws"),
    ("ws384_minblocks2", 384, {"min_blocks_per_sm": 2}, "ws"),
    ("ws384_maxnreg168", 384, {"max_registers_per_thread": 168}, "ws"),
    ("divergent384_minblocks1", 384, {"min_blocks_per_sm": 1}, "divergent"),
]
for name, block, attrs, body in VARIANTS:
    rec = {"case": name}
    try:
        src = source(make(block, attrs, body))
        m = re.search(r'extern "C" __global__ void([^(]*)main_kernel', src)
        rec["signature_attrs"] = (m.group(1).strip() if m else "?")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = compile_source(src, {"cuda": {"arch": "sm_90a", "perf_remarks": "ignore"}})
        rec["remarks"] = [str(r) for r in res.remarks]
        rec["entry_regs"] = (re.findall(r"Used (\d+) registers", res.log) or ["?"])[0]
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(res.binary); f.flush()
            sass = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
        rec["USETMAXREG"] = [l.split(";")[0].split("*/")[-1].strip() for l in sass.splitlines() if "SETMAXREG" in l]
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:400]
    print(json.dumps(rec), flush=True)
