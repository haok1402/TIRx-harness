# SPDX-License-Identifier: Apache-2.0
"""SM90 probe: a Hopper-style warp-specialized bf16 GEMM in raw TIRx script.

Exercises together: TMA tiled loads (128B swizzle), TMA multicast across a
2-CTA cluster, mbarrier full/empty ring with cross-CTA (mapa) releases,
fence.mbarrier_init + barrier.cluster, setmaxnreg producer/consumer split,
WGMMA m64n256k16 ss with wait_group(1) overlap, and a register epilogue.

C[M, N] = A[M, K] @ B[N, K]^T (bf16 in, f32 accumulate, bf16 out).
CTA tile 128x256x64, 2 consumer warpgroups (64 rows each), 1 producer warpgroup.
Cluster (2,1): the two CTAs share the N tile; each loads half of B and multicasts.
"""

# ruff: noqa: F841
import json
import os
import subprocess
import sys

import torch

import tvm_tirx
from tvm_tirx.script import tirx as T

ARCH = os.environ.get("PROBE_ARCH", "sm_90a")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

BM, BN, BK = 128, 256, 64
ACC = BN // 2  # f32 accumulator registers per consumer thread


def build(M, N, K, STAGES=4, multicast=True, pdl=False):
    MT, NT, NK = M // BM, N // BN, K // BK
    HALF_B = BN // 2
    STAGE_TX = (BM * BK + BN * BK) * 2
    B_ROWS_PER_CTA = HALF_B if multicast else BN
    cluster = 2 if multicast else 1
    tma = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    tma_mc = tma + ".multicast::cluster"
    mma = "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16"
    launch_kw = dict(grid=MT * NT, block=384)
    if multicast:
        launch_kw["cluster"] = 2
    if pdl:
        launch_kw["programmatic_stream_serialization"] = True

    # fmt: off
    @T.function
    def main(
        A: T.Tensor((M, K), dtype="bfloat16", align=16),
        B: T.Tensor((N, K), dtype="bfloat16", align=16),
        C: T.Tensor((M, N), dtype="bfloat16", align=16),
    ):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "bfloat16", 2, A.data,
                      K, M, K * 2, BK, BM, 1, 1, 0, 3, 2, 0)
        B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "bfloat16", 2, B.data,
                      K, N, K * 2, BK, B_ROWS_PER_CTA, 1, 1, 0, 3, 2, 0)

        T.device_entry(launch=T.cuda.LaunchConfig(**launch_kw),
                       kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1))
        tx = T.cuda.thread_idx("x")
        bid = T.cuda.block_idx("x")
        A_smem = T.alloc_tensor(ty_args=[T.Tensor((STAGES, BM, BK), "bfloat16", scope="shared", align=1024)])
        B_smem = T.alloc_tensor(ty_args=[T.Tensor((STAGES, BN, BK), "bfloat16", scope="shared", align=1024)])
        full = T.alloc_tensor(ty_args=[T.Tensor((STAGES,), "uint64", scope="shared", align=8)])
        empty = T.alloc_tensor(ty_args=[T.Tensor((STAGES,), "uint64", scope="shared", align=8)])
        m_blk = T.meta_var(bid % MT)
        n_blk = T.meta_var(bid // MT)
        if T.constexpr(multicast):
            rank = T.cuda.cluster_cta_id("x")

        if tx == 0:
            T.ptx.prefetch.tensormap(T.address_of(A_map))
            T.ptx.prefetch.tensormap(T.address_of(B_map))
            for s in T.unroll(STAGES):
                T.ptx.mbarrier.init.shared.b64(T.address_of(full[s]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(T.address_of(empty[s]), T.uint32(2 * cluster))
        if T.constexpr(multicast):
            T.ptx.fence.mbarrier_init.release.cluster()
            T.cuda.cluster_sync()
        else:
            T.ptx.fence.proxy.async_.shared__cta()
            T.cuda.cta_sync()
        if T.constexpr(pdl):
            T.ptx.griddepcontrol.wait()

        if tx < 128:
            # ---------------- producer warpgroup ----------------
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](40)
            if tx == 0:
                for kt in T.serial(0, NK):
                    stage = T.meta_var(kt % STAGES)
                    T.cuda.mbarrier_wait(T.address_of(empty[stage]), ((kt // STAGES) & 1) ^ 1)
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(full[stage]), T.uint32(STAGE_TX))
                    T.ptx[tma](A_smem.ptr_to([stage, 0, 0]), T.address_of(A_map),
                               kt * BK, m_blk * BM, T.address_of(full[stage]))
                    if T.constexpr(multicast):
                        T.ptx[tma_mc](B_smem.ptr_to([stage, rank * HALF_B, 0]), T.address_of(B_map),
                                      kt * BK, n_blk * BN + rank * HALF_B, T.address_of(full[stage]),
                                      T.uint16(3))
                    else:
                        T.ptx[tma](B_smem.ptr_to([stage, 0, 0]), T.address_of(B_map),
                                   kt * BK, n_blk * BN, T.address_of(full[stage]))
        else:
            # ---------------- consumer warpgroups ----------------
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](232)
            wg = T.meta_var(tx // 128 - 1)
            ct = T.meta_var(tx % 128)
            acc = T.alloc_tensor(ty_args=[T.Tensor((ACC,), "float32", scope="local")])
            descA: T.uint64
            descB: T.uint64
            mapped = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
            packed = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
            for i in T.unroll(ACC):
                acc[i] = T.float32(0)
            T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descA), A_smem.ptr_to([0, wg * 64, 0]), 1, 64, 3)
            T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descB), B_smem.ptr_to([0, 0, 0]), 1, 64, 3)
            for kt in T.serial(0, NK):
                stage = T.meta_var(kt % STAGES)
                T.cuda.mbarrier_wait(T.address_of(full[stage]), (kt // STAGES) & 1)
                for i in T.unroll(ACC):
                    T.cuda.wgmma.noop_barrier(acc[i])
                T.ptx.wgmma.fence.sync.aligned()
                for s in T.unroll(BK // 16):
                    T.ptx[mma](*[acc[i] for i in range(ACC)],
                               descA + T.Cast("uint64", stage * (BM * BK * 2 // 16) + s * 2),
                               descB + T.Cast("uint64", stage * (BN * BK * 2 // 16) + s * 2),
                               True, 1, 1, 0, 0)
                T.ptx.wgmma.commit_group.sync.aligned()
                T.ptx.wgmma.wait_group.sync.aligned(1)
                for i in T.unroll(ACC):
                    T.cuda.wgmma.noop_barrier(acc[i])
                if kt > 0:
                    if ct == 0:
                        prev = T.meta_var((kt + STAGES - 1) % STAGES)
                        if T.constexpr(multicast):
                            for r in T.unroll(2):
                                T.ptx.mapa.shared__cluster.u32(
                                    mapped[0], T.cuda.cvta_generic_to_shared(T.address_of(empty[prev])), T.uint32(r))
                                T.ptx.mbarrier.arrive.shared__cluster.b64(mapped[0])
                        else:
                            T.ptx.mbarrier.arrive.shared.b64(T.address_of(empty[prev]))
            T.ptx.wgmma.wait_group.sync.aligned(0)
            for i in T.unroll(ACC):
                T.cuda.wgmma.noop_barrier(acc[i])
            if T.constexpr(pdl):
                T.ptx.griddepcontrol.launch_dependents()
            w = T.meta_var(ct // 32)
            lane = T.meta_var(ct % 32)
            row = T.meta_var(m_blk * BM + wg * 64 + w * 16 + lane // 4)
            for i in T.unroll(BN // 8):
                col = T.meta_var(n_blk * BN + i * 8 + (lane % 4) * 2)
                T.ptx["cvt.rn.bf16x2.f32"](packed[0], acc[i * 4 + 1], acc[i * 4])
                T.ptx.st.global_.b32(C.ptr_to([row, col]), packed[0])
                T.ptx["cvt.rn.bf16x2.f32"](packed[0], acc[i * 4 + 3], acc[i * 4 + 2])
                T.ptx.st.global_.b32(C.ptr_to([row + 8, col]), packed[0])
        if T.constexpr(multicast):
            T.cuda.cluster_sync()
    # fmt: on
    return main


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e-3


def run(M, N, K, multicast=True, stages=4, tag=""):
    name = f"gemm_bf16_{M}x{N}x{K}_mc{int(multicast)}_s{stages}{tag}"
    rec = {"case": name}
    try:
        func = build(M, N, K, STAGES=stages, multicast=multicast)
        target = tvm_tirx.target.Target({"kind": "cuda", "arch": ARCH})
        mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        cu = os.path.join(OUT, name + ".cu")
        open(cu, "w").write(src)
        torch.manual_seed(0)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
        mod(A, B, C)
        torch.cuda.synchronize()
        ref = A @ B.T
        rel = ((C.float() - ref.float()).norm() / ref.float().norm()).item()
        ref32 = A.float() @ B.float().T
        rel32 = ((C.float() - ref32).norm() / ref32.norm()).item()
        t = bench(lambda: mod(A, B, C))
        tr = bench(lambda: torch.matmul(A, B.T, out=C.new_empty(M, N)))
        fl = 2 * M * N * K
        rec.update(ok=rel32 < 1e-2, rel_err_vs_cublas=rel, rel_err_vs_fp32=rel32, us=t * 1e6, tflops=fl / t / 1e12,
                   cublas_us=tr * 1e6, cublas_tflops=fl / tr / 1e12, ratio_vs_cublas=tr / t)
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:3000]}")
    print(json.dumps(rec), flush=True)
    return rec


if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] != "host"):
    shapes = [(4096, 4096, 4096), (8192, 8192, 8192), (2048, 4096, 8192)]
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        shapes = [(1024, 1024, 1024)]
    for shp in shapes:
        run(*shp, multicast=True)
    run(*shapes[0], multicast=False)


def host_overhead():
    """Host time per call for a tiny GEMM launch (2 cuTensorMapEncodeTiled per call
    in the generated host function) vs torch.matmul on the same shape."""
    import time

    M = N = 256
    K = 64
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": build(M, N, K, STAGES=1, multicast=False)}),
                           target=tvm_tirx.target.Target({"kind": "cuda", "arch": ARCH}),
                           pipeline="tirx")
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    out = {}
    for name, fn in (("tirx_gemm", lambda: mod(A, B, C)), ("torch_matmul", lambda: torch.matmul(A, B.T, out=C))):
        for _ in range(500):
            fn()
        torch.cuda.synchronize()
        n = 5000
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        out[name + "_host_us_per_call"] = (t1 - t0) / n * 1e6
    print(json.dumps({"case": "gemm_host_overhead_256x256x64", **out}), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "host":
    host_overhead()
