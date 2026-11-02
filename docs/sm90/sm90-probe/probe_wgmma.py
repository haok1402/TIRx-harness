# SPDX-License-Identifier: Apache-2.0
"""SM90 probe: WGMMA (ss and rs forms) over bf16 / fp16 / fp8 e4m3 / fp8 e5m2,
f32 and f16 accumulators, with a K loop fed by TMA (128B swizzle) + mbarrier,
and async commit/wait groups.

Each case: one CTA, one warpgroup, C[64, N] = A[64, K] @ B[N, K]^T, both operands
K-major (the only layout fp8 WGMMA allows). Prints one JSON line per case.
"""

# ruff: noqa: F841
import json
import os
import sys

import torch

import tvm_tirx
from tvm_tirx.script import tirx as T

ARCH = os.environ.get("PROBE_ARCH", "sm_90a")
OUT = os.environ.get("PROBE_OUT", os.path.join(os.path.dirname(__file__), "out"))
os.makedirs(OUT, exist_ok=True)

DT = {
    "bf16": ("bfloat16", torch.bfloat16, 2, "bf16"),
    "fp16": ("float16", torch.float16, 2, "f16"),
    "e4m3": ("float8_e4m3fn", torch.float8_e4m3fn, 1, "e4m3"),
    "e5m2": ("float8_e5m2", torch.float8_e5m2, 1, "e5m2"),
}


def build(dt, acc, form, N, K, staged_wait):
    tdtype, _, eb, ptx_t = DT[dt]
    M = 64
    KB = 128 // eb  # elements per 128-byte swizzle row = one k-block
    MMA_K = 32 // eb  # k16 for 16-bit, k32 for fp8
    STEPS = KB // MMA_K  # 4
    NKB = K // KB
    acc_regs = N // 4 if acc == "f16" else N // 2
    acc_dtype = "uint32" if acc == "f16" else "float32"
    ptx_d = "f16" if acc == "f16" else "f32"
    mma = f"wgmma.mma_async.sync.aligned.m64n{N}k{MMA_K}.{ptx_d}.{ptx_t}.{ptx_t}"
    a_bytes = M * KB * eb
    b_bytes = N * KB * eb
    out_cols = N // 2 if acc == "f16" else N
    out_dtype = "uint32" if acc == "f16" else "float32"
    # fp8 WGMMA has no transpose immediates (K-major only).
    TRANS_SS = [0, 0] if eb == 2 else []
    TRANS_RS = [0] if eb == 2 else []

    # fmt: off
    @T.function
    def main(
        A: T.Tensor((M, K), dtype=tdtype, align=16),
        B: T.Tensor((N, K), dtype=tdtype, align=16),
        C: T.Tensor((M, out_cols), dtype=out_dtype, align=16),
    ):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, tdtype, 2, A.data,
                      K, M, K * eb, KB, M, 1, 1, 0, 3, 0, 0)
        B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, tdtype, 2, B.data,
                      K, N, K * eb, KB, N, 1, 1, 0, 3, 0, 0)

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tx = T.cuda.thread_idx("x")
        A_smem = T.alloc_tensor(ty_args=[T.Tensor((M, KB), tdtype, scope="shared", align=1024)])
        B_smem = T.alloc_tensor(ty_args=[T.Tensor((N, KB), tdtype, scope="shared", align=1024)])
        bar = T.shared_scalar("uint64")
        descA: T.uint64
        descB: T.uint64
        C_local = T.alloc_tensor(ty_args=[T.Tensor((acc_regs,), acc_dtype, scope="local")])
        A_reg = T.alloc_tensor(ty_args=[T.Tensor((4,), "uint32", scope="local")])

        if tx == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(bar), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        for i in T.serial(0, acc_regs):
            C_local[i] = T.Cast(acc_dtype, 0)
            T.cuda.wgmma.noop_barrier(C_local[i])

        for kb in T.serial(0, NKB):
            if tx == 0:
                T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                    A_smem.data, T.address_of(A_map), kb * KB, 0, T.address_of(bar))
                T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                    B_smem.data, T.address_of(B_map), kb * KB, 0, T.address_of(bar))
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(bar), T.uint32(a_bytes + b_bytes))
            T.cuda.mbarrier_wait(T.address_of(bar), kb % 2)
            T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descA), A_smem.data, 1, 64, 3)
            T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descB), B_smem.data, 1, 64, 3)
            for s in T.unroll(STEPS):
                if T.constexpr(form == "rs"):
                    # A fragment straight from smem-equivalent global layout:
                    # warp w owns rows 16w..16w+15; reg r = (row + 8*(r%2), k-half r//2).
                    w = T.meta_var(tx // 32)
                    lane = T.meta_var(tx % 32)
                    for r in T.unroll(4):
                        row = T.meta_var(w * 16 + lane // 4 + 8 * (r % 2))
                        kcol = T.meta_var(kb * KB + s * MMA_K + (r // 2) * (MMA_K // 2) + (lane % 4) * (4 // eb))
                        T.ptx.ld.global_.b32(A_reg[r], A.ptr_to([row, kcol]))
                        T.cuda.wgmma.noop_barrier(A_reg[r])
                T.ptx.wgmma.fence.sync.aligned()
                if T.constexpr(form == "ss"):
                    T.ptx[mma](*[C_local[i] for i in range(acc_regs)],
                               descA + T.Cast("uint64", s * 2), descB + T.Cast("uint64", s * 2),
                               True, 1, 1, *TRANS_SS)
                else:
                    T.ptx[mma](*[C_local[i] for i in range(acc_regs)],
                               *[A_reg[i] for i in range(4)],
                               descB + T.Cast("uint64", s * 2), True, 1, 1, *TRANS_RS)
                T.ptx.wgmma.commit_group.sync.aligned()
                if T.constexpr(staged_wait):
                    T.ptx.wgmma.wait_group.sync.aligned(1)
            T.ptx.wgmma.wait_group.sync.aligned(0)
            for i in T.serial(0, acc_regs):
                T.cuda.wgmma.noop_barrier(C_local[i])
            T.cuda.cta_sync()

        w = T.meta_var(tx // 32)
        lane = T.meta_var(tx % 32)
        row = T.meta_var(w * 16 + lane // 4)
        if T.constexpr(acc == "f16"):
            for i in T.serial(0, N // 8):
                C[row, i * 4 + lane % 4] = C_local[i * 2]
                C[row + 8, i * 4 + lane % 4] = C_local[i * 2 + 1]
        else:
            for i in T.serial(0, N // 8):
                col = T.meta_var(i * 8 + (lane % 4) * 2)
                C[row, col] = C_local[i * 4]
                C[row, col + 1] = C_local[i * 4 + 1]
                C[row + 8, col] = C_local[i * 4 + 2]
                C[row + 8, col + 1] = C_local[i * 4 + 3]
    # fmt: on
    return main


def run_case(dt, acc, form, N=128, K=512, staged_wait=False):
    name = f"wgmma_{form}_{dt}_{acc}acc_n{N}" + ("_staged" if staged_wait else "")
    rec = {"case": name}
    try:
        func = build(dt, acc, form, N, K, staged_wait)
        target = tvm_tirx.target.Target({"kind": "cuda", "arch": ARCH})
        mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        open(os.path.join(OUT, name + ".cu"), "w").write(src)
        tdtype, tt, eb, _ = DT[dt]
        torch.manual_seed(0)
        A = (torch.randn(64, K, device="cuda") * 0.5).to(tt)
        B = (torch.randn(N, K, device="cuda") * 0.5).to(tt)
        if acc == "f16":
            C = torch.zeros(64, N // 2, dtype=torch.uint32, device="cuda")
        else:
            C = torch.zeros(64, N, dtype=torch.float32, device="cuda")
        mod(A, B, C)
        torch.cuda.synchronize()
        out = C.view(torch.int32).view(torch.float16).float() if acc == "f16" else C
        ref = A.float() @ B.float().T
        err = (out - ref).abs().max().item()
        tol = 0.5 if acc == "f16" else 1e-2 * (1 + ref.abs().max().item()) * (4 if eb == 1 else 1)
        rec.update(ok=bool(err <= tol), max_abs_err=err, tol=tol,
                   wgmma_in_src=src.count("wgmma.mma_async"))
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:1500]}")
    print(json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__":
    cases = [
        ("bf16", "f32", "ss"), ("fp16", "f32", "ss"), ("fp16", "f16", "ss"),
        ("e4m3", "f32", "ss"), ("e5m2", "f32", "ss"), ("e4m3", "f16", "ss"),
        ("bf16", "f32", "rs"), ("e4m3", "f32", "rs"),
    ]
    sel = sys.argv[1:]
    res = []
    for c in cases:
        if sel and "_".join(c) not in sel:
            continue
        res.append(run_case(*c))
    res.append(run_case("bf16", "f32", "ss", N=256))
    res.append(run_case("bf16", "f32", "ss", staged_wait=True))
    print("SUMMARY", sum(r["ok"] for r in res), "/", len(res))
