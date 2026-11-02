# SPDX-License-Identifier: Apache-2.0
"""SM90 probe: what happens when existing tirx-lite kernels (written for sm_100a)
are compiled for sm_90a, and whether tirx-lite's arch-neutral helpers
(Pipeline(full="tma", empty="mbar"), specialize roles with setmaxnreg, smem_pool)
work on H200.

Records, per case, where the failure surfaces (trace / TIRx lowering / NVRTC / ptxas)
and the first diagnostic lines.
"""

import json
import os
import re
import sys

import torch

import tirx_kernels.tirx_lite as txl
import tvm_tirx

CFG = {"cuda": {"arch": "sm_90a"}}
TARGET = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})


def classify(err):
    import traceback
    tb = "".join(traceback.format_exception(err))
    print(tb[-3000:], file=sys.stderr)
    s = str(err)
    if "NVRTC" in s or "nvrtc" in s:
        stage = "nvrtc/ptxas"
    elif "ptxas" in s:
        stage = "ptxas"
    else:
        stage = "tirx"
    lines = sorted(set(re.findall(r"error[^\n]{0,160}", s)))[:6]
    return {"stage": stage, "type": type(err).__name__, "errors": lines or [s[:400]]}


def case(name, fn):
    rec = {"case": name}
    try:
        rec.update(fn())
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, **classify(e))
    print(json.dumps(rec), flush=True)
    return rec


def rmsnorm_sm90():
    from tirx_kernels.norm import rmsnorm as m
    from tirx_kernels.runner import compile_kernel

    k = m.get_kernel(4096, backend_config=CFG)
    with TARGET:
        ex = compile_kernel(k, backend_config=CFG)
    x, w = m.prepare_data(64, 4096)
    out = torch.empty_like(x)
    ex(*m._kernel_args(x, w, out))
    torch.cuda.synchronize()
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + m.eps) * w.float()).half()
    err = (out.float() - ref.float()).abs().max().item()
    return {"ok": err < 1e-2, "max_abs_err": err}


def rmsnorm_trace_sm100_control():
    """Same factory traced for sm_100a (trace only, no GPU run): is the TypeError arch-specific?"""
    from tirx_kernels.norm import rmsnorm as m

    m.get_kernel(4096, backend_config={"cuda": {"arch": "sm_100a"}})
    return {"ok": True}


def gemm_sm100_on_sm90():
    from tirx_kernels.gemm import fp16_bf16_gemm as m
    from tirx_kernels.runner import compile_kernel

    k = m.get_kernel("bf16", 4096, 4096, 4096, backend_config=CFG)
    with TARGET:
        compile_kernel(k, backend_config=CFG)
    return {"ok": True}


def txl_pipeline_specialize():
    """A small warp-specialized TMA->smem->global copy written only with tirx-lite
    helpers: smem_pool, Pipeline(full="tma", empty="mbar"), specialize() roles with
    setmaxnreg, PipelineState. 1 producer warpgroup, 1 consumer warpgroup."""
    ROWS, COLS, TILE_R, STAGES = 1024, 64, 64, 3
    NT = ROWS // TILE_R
    TX = TILE_R * COLS * 2

    @txl.kernel()
    def copy_kernel(
        src: txl.Tensor((ROWS, COLS), "bfloat16"),
        dst: txl.Tensor((ROWS, COLS), "bfloat16"),
    ):
        tmap = txl.stack_alloca("tensormap", 1)
        txl.cu_tensor_map_encode_tiled(
            tmap, "bfloat16", 2, src.data, COLS, ROWS, COLS * 2, COLS, TILE_R, 1, 1, 0, 3, 2, 0
        )
        txl.device_entry(
            launch=txl.cuda.LaunchConfig(grid=1, block=256),
            kernel_attrs=txl.cuda.KernelAttributes(min_blocks_per_sm=1),
            backend_config=CFG,
        )
        smem = txl.smem_pool()
        pipe = txl.Pipeline(smem, STAGES, full="tma", empty="mbar", init_empty=128)
        smem.pool.move_base_to(1024)
        buf = smem.alloc((STAGES, TILE_R, COLS), "bfloat16",
                         swizzle=txl.SwizzleMode.SWIZZLE_128B_ATOM)
        txl.ptx.fence.proxy.async_.shared__cta()
        txl.cuda.cta_sync()
        sp = txl.specialize()
        prod = sp.role("producer", warps=range(4, 8), regs=40)
        cons = sp.role("consumer", warps=range(0, 4), regs=232)
        with prod:
            st = txl.PipelineState(STAGES, 1)
            with txl.if_(txl.thread_id() == 128), txl.then_():
                with txl.serial(NT) as t:
                    pipe.empty.wait(st.stage, st.phase)
                    txl.ptx[
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    ](buf.buf.ptr_to([st.stage, 0, 0]), txl.address_of(tmap), txl.int32(0),
                      txl.cast(t * TILE_R, ty="int32"), pipe.full.buf.ptr_to([st.stage]))
                    pipe.full.arrive(st.stage, TX)
                    st.advance()
        with cons:
            st = txl.PipelineState(STAGES, 0)
            with txl.serial(NT) as t:
                pipe.full.wait(st.stage, st.phase)
                tid = txl.thread_id()
                word = txl.alloc_local((1,), "uint32")
                with txl.serial(TILE_R * COLS // 2 // 128) as j:
                    e = (j * 128 + tid) * 2
                    r = e // COLS
                    c = e % COLS
                    txl.ptx.ld.shared.b32(word[0], buf.buf.ptr_to([st.stage, r, c]))
                    txl.ptx.st.global_.b32(dst.ptr_to([t * TILE_R + r, c]), word[0])
                pipe.empty.arrive(st.stage)
                st.advance()

    ex = copy_kernel.compile(target=TARGET)
    x = torch.randn(ROWS, COLS, device="cuda", dtype=torch.bfloat16)
    y = torch.zeros_like(x)
    ex(x, y)
    torch.cuda.synchronize()
    return {"ok": bool(torch.equal(x, y))}


if __name__ == "__main__":
    sel = sys.argv[1:]
    cases = {"rmsnorm_sm90": rmsnorm_sm90, "rmsnorm_trace_sm100_control": rmsnorm_trace_sm100_control,
             "gemm_sm100_on_sm90": gemm_sm100_on_sm90,
             "txl_pipeline_specialize": txl_pipeline_specialize}
    for n, f in cases.items():
        if not sel or n in sel:
            case(n, f)
