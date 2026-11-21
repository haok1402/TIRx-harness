# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Hopper (sm_90a) fp16/bf16 GEMM: C[M, N] = A[M, K] @ B[N, K]^T.

Warp-specialized and persistent, written on tvm-tirx's Hopper helpers:

* CTA tile 128 x 256 x 64. One producer warpgroup issues TMA loads into a
  ``STAGES``-deep ring (``TMABar`` full / ``MBarrier`` empty). Two consumer
  warpgroups each own 64 rows and run ``wgmma.m64n256k16`` chains
  (``lang.wgmma``) with ``wait_group 1`` overlap. ``setmaxnreg`` gives 40 / 232
  registers.
* A 2-CTA cluster along M shares the N tile. Each CTA loads half of B and
  multicasts it, and consumers release a stage in both CTAs
  (``MBarrier.arrive_multicast``).
* Persistent grid of ``max_active_clusters`` clusters (``runtime.occupancy``).
  Each walks 2 x 1 tile pairs in a grouped raster for L2 reuse.

Requires tvm-tirx with ``backend/cuda/lang/wgmma.py``, ``MBarrier.arrive_multicast``
and ``runtime/occupancy.py``.
"""

from __future__ import annotations

import torch

import tvm_tirx
from tvm_tirx.script import tirx as T

BM, BN, BK = 128, 256, 64
STAGES = 4
CLUSTER = 2
GROUP_M = 8  # tile-pair rows per L2 group
DEFAULT_CLUSTERS = 66  # H100/H200 at this tile: 132 SMs / cluster 2 (see max_active_clusters)

_DT = {"fp16": "float16", "bf16": "bfloat16"}
_CVT = {"fp16": "cvt.rn.f16x2.f32", "bf16": "cvt.rn.bf16x2.f32"}


def make_kernel(dtype, M, N, K, *, num_clusters=DEFAULT_CLUSTERS, backend_config=None, no_epi=False, simple_raster=False, no_producer=False):
    from tvm_tirx.backend.cuda.lang import SMEMPool
    from tvm_tirx.backend.cuda.lang import wgmma as W
    from tvm_tirx.backend.cuda.lang.pipeline import MBarrier, TMABar

    dt = _DT[dtype]
    NO_MMA = globals().get("NOMMA", False)
    E_one_iter, E_no_wg, E_tensor_store, E_no_cvt, E_single_store = (EPI == k for k in ("one_iter", "no_wg", "tensor_store", "no_cvt", "single_store"))
    if M % (BM * CLUSTER) or N % BN or K % BK:
        raise ValueError(f"M must be a multiple of {BM * CLUSTER}, N of {BN}, K of {BK}; got {M}x{N}x{K}")
    MP, NT, NK = M // (BM * CLUSTER), N // BN, K // BK  # tile pairs along M, tiles along N
    n_pairs = MP * NT
    rounds = (n_pairs + num_clusters - 1) // num_clusters
    half_b = BN // CLUSTER
    stage_tx = (BM * BK + BN * BK) * 2
    a_stage_bytes, b_stage_bytes = BM * BK * 2, BN * BK * 2
    op = W.WGMMA(BN, dt)
    acc_n = op.num_acc
    a_tile = W.SmemTile(None, rows=64)  # one consumer's 64 rows of A, K-major
    b_tile = W.SmemTile(None, rows=BN)
    k_steps = BK * 2 // op.k_bytes
    tma = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    tma_mc = tma + ".multicast::cluster"
    cvt = _CVT[dtype]
    entry_kw = {} if backend_config is None else {"backend_config": backend_config}

    # fmt: off
    @T.function
    def main(A: T.Tensor((M, K), dt, align=16), B: T.Tensor((N, K), dt, align=16),
             C: T.Tensor((M, N), dt, align=16)):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dt, 2, A.data, K, M, K * 2, BK, BM, 1, 1, 0, 3, 2, 0)
        B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dt, 2, B.data, K, N, K * 2, BK, half_b, 1, 1, 0, 3, 2, 0)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=num_clusters * CLUSTER, block=384, cluster=CLUSTER),
                       kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1), **entry_kw)
        tx = T.cuda.thread_idx("x")
        cid = T.cuda.block_idx("x") // CLUSTER
        rank = T.cuda.cluster_cta_id("x")
        A_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, BM, BK), dt, scope="shared", align=1024)])
        B_s = T.alloc_tensor(ty_args=[T.Tensor((STAGES, BN, BK), dt, scope="shared", align=1024)])
        pool = SMEMPool()
        full = TMABar(pool, STAGES)
        empty = MBarrier(pool, STAGES)
        pool.commit()
        full.init(1)
        empty.init(2 * CLUSTER)  # one arrive per consumer warpgroup per CTA of the cluster
        it = T.alloc_local([1], "int32")
        it[0] = 0
        if tx == 0:
            T.ptx.prefetch.tensormap(T.address_of(A_map))
            T.ptx.prefetch.tensormap(T.address_of(B_map))
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cluster_sync()

        if tx < 128:
            # ---------------- producer warpgroup ----------------
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](40)
            if tx == 0 and not no_producer:
                for j in T.serial(rounds):
                    tp = cid + j * num_clusters
                    if tp < n_pairs:
                        g = tp // (GROUP_M * NT)
                        first = g * GROUP_M
                        gsize = T.min(MP - first, GROUP_M)
                        m_blk = (first + (tp % (GROUP_M * NT)) % gsize) * CLUSTER + rank
                        n_blk = (tp % (GROUP_M * NT)) // gsize
                        for kt in T.serial(NK):
                            stage = it[0] % STAGES
                            empty.wait(stage, ((it[0] // STAGES) & 1) ^ 1)
                            full.arrive(stage, stage_tx)
                            T.ptx[tma](A_s.ptr_to([stage, 0, 0]), T.address_of(A_map), kt * BK, m_blk * BM,
                                       full.ptr_to([stage]))
                            T.ptx[tma_mc](B_s.ptr_to([stage, rank * half_b, 0]), T.address_of(B_map), kt * BK,
                                          n_blk * BN + rank * half_b, full.ptr_to([stage]), T.uint16(3))
                            it[0] = it[0] + 1
        else:
            # ---------------- consumer warpgroups ----------------
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](232)
            wg = tx // 128 - 1
            ct = tx % 128
            acc = T.alloc_tensor(ty_args=[T.Tensor((acc_n,), "float32", scope="local")])
            packed = T.alloc_local([1], "uint32")
            da0 = W.smem_descriptor(A_s.ptr_to([0, wg * 64, 0]))
            db0 = W.smem_descriptor(B_s.ptr_to([0, 0, 0]))
            for j in T.serial(rounds):
                tp = cid + j * num_clusters
                if tp < n_pairs:
                    g = tp // (GROUP_M * NT)
                    first = g * GROUP_M
                    if T.constexpr(simple_raster):
                        m_blk = (tp // NT) * CLUSTER + rank
                        n_blk = tp % NT
                    else:
                        gsize = T.min(MP - first, GROUP_M)
                        m_blk = (first + (tp % (GROUP_M * NT)) % gsize) * CLUSTER + rank
                        n_blk = (tp % (GROUP_M * NT)) // gsize
                    for kt in T.serial(0 if NO_MMA else NK):
                        stage = it[0] % STAGES
                        full.wait(stage, (it[0] // STAGES) & 1)
                        a0 = W.advance_descriptor(da0, stage * a_stage_bytes)
                        b0 = W.advance_descriptor(db0, stage * b_stage_bytes)
                        W.chain(op, acc, [a_tile.k_step(op, s, a0) for s in range(k_steps)],
                                [b_tile.k_step(op, s, b0) for s in range(k_steps)], accumulate=kt > 0)
                        W.wait(1, (acc, acc_n))
                        if kt > 0:
                            if ct == 0:
                                empty.arrive_multicast((it[0] + STAGES - 1) % STAGES, ranks=range(CLUSTER))
                        it[0] = it[0] + 1
                    W.wait(0, (acc, acc_n))
                    if ct == 0:
                        empty.arrive_multicast((it[0] + STAGES - 1) % STAGES, ranks=range(CLUSTER))
                    # epilogue: registers -> global, two halves per 8-column group
                    for r4 in T.unroll(0 if no_epi else (1 if E_one_iter else acc_n // 4)):
                        r = r4 * 4
                        row, col = W.acc_coord(ct, r)
                        if T.constexpr(E_no_wg):
                            grow = m_blk * BM + row
                        else:
                            grow = m_blk * BM + wg * 64 + row
                        gcol = n_blk * BN + col
                        if T.constexpr(E_tensor_store):
                            C[grow, gcol] = T.Cast(dt, acc[r])
                        elif T.constexpr(E_no_cvt):
                            T.ptx.st.global_.b32(C.ptr_to([grow, gcol]), T.reinterpret(acc[r], ty="uint32"))
                        else:
                            T.ptx[cvt](packed[0], acc[r + 1], acc[r])
                            T.ptx.st.global_.b32(C.ptr_to([grow, gcol]), packed[0])
                            if T.constexpr(not E_single_store):
                                T.ptx[cvt](packed[0], acc[r + 3], acc[r + 2])
                                T.ptx.st.global_.b32(C.ptr_to([grow + 8, gcol]), packed[0])
        T.cuda.cluster_sync()
    # fmt: on
    return main



EPI = "full"
if __name__ == "__main__":
    import os
    os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
    from tirx_kernels.runner import compile_kernel
    BC = {"cuda": {"arch": "sm_90a"}}
    for EPI, NOMMA in [("tensor_store", False), ("tensor_store", True)]:
        globals()["EPI"] = EPI; globals()["NOMMA"] = NOMMA
        try:
            compile_kernel(make_kernel("bf16", 4096, 11008, 4096, backend_config=BC, simple_raster=True, no_producer=True), backend_config=BC)
            print(f"{EPI} no_mma={NOMMA}: ok")
        except Exception as e:
            print(f"{EPI} no_mma={NOMMA}: FAIL {str(e).splitlines()[-1][-60:]}")
