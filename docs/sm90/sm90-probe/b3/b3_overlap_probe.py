"""B3 risk probe: FA3's intra-warpgroup overlap under perf_remarks="error" (tirx-lite).

One consumer warpgroup: Q[64, 128] against NB KV blocks of 128 rows (bf16, hd128), with online softmax.
In FA3's order (flashinfer hopper/mainloop_mma.cuh), each block j >= 1 runs:
    chain S = Q K_j^T            (in flight)
    rescale O                    <- touches the *retired* O accumulator while the S chain runs
    chain O += P_{j-1} V_{j-1}   (rs, P from registers)
    wait_group 1                 (S done, PV in flight)
    softmax on S                 <- touches the retired S accumulator while the PV chain runs
    wait_group 0
    P = bf16(S)
B2-design saw ptxas serialize every wgmma (C7514) when a retired accumulator was touched while another
chain was in flight. This compiles that order ("overlap") and a fully serialized order ("serial") in
error mode, checks both against torch, and reports the SASS GMMA chain shape.

argv: overlap|serial
"""
import ctypes, json, subprocess, sys, tempfile

import numpy as np
import torch
import tvm_ffi

import tirx_kernels.tirx_lite as txl
from tirx_kernels.runner import compile_kernel
from tvm_tirx.backend.cuda.lang import wgmma as W
from tvm_tirx.script import tirx as T

MODE = sys.argv[1] if len(sys.argv) > 1 else "overlap"
BC = {"cuda": {"arch": "sm_90a", "perf_remarks": "error"}}
M, KVB, HD = 64, 128, 128
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 3
NITER = int(sys.argv[5]) if len(sys.argv) > 5 else 8  # loop mode: blocks processed (cycling over NB)
E = 64  # bf16 per 128-byte swizzle row
KV = KVB * NB
SCALE_LOG2 = float(np.log2(np.e) / np.sqrt(HD))
TMA2D = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
qk = W.WGMMA(KVB, "bfloat16")
pv = W.WGMMA(HD, "bfloat16", a_in_registers=True)
QK_STEPS = HD * 2 // qk.k_bytes
PV_STEPS = KVB // pv.k


class _Slice:
    def __init__(self, tensor, start):
        self.tensor, self.start = tensor, start

    def __getitem__(self, i):
        return self.tensor[self.start + i]

    def __iter__(self):
        return iter(self.tensor[self.start + i] for i in range(4))

    def __len__(self):
        return 4


@txl.kernel()
def probe(
    Q_map: txl.Ptr(txl.TensorMap),
    K_map: txl.Ptr(txl.TensorMap),
    V_map: txl.Ptr(txl.TensorMap),
    Out: txl.Tensor((M * HD,), "float32"),
):
    txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=128), backend_config=BC)
    tx = txl.thread_id()
    smem = txl.smem_pool()
    q_s = smem.alloc([M * HD], "bfloat16", align=1024)
    k_s = smem.alloc([KV * HD], "bfloat16", align=1024)
    v_s = smem.alloc([KV * HD], "bfloat16", align=1024)
    bar = txl.TMABar(smem, 1)
    bar.init(1)
    txl.ptx.fence.proxy.async_.shared__cta()
    txl.cuda.cta_sync()
    s_acc = txl.alloc_local([qk.num_acc], "float32")
    o_acc = txl.alloc_local([pv.num_acc], "float32")
    p_half = txl.alloc_local([qk.num_acc], "bfloat16")
    m_run = txl.alloc_local([2], "float32")
    l_run = txl.alloc_local([2], "float32")
    alpha = txl.alloc_local([2], "float32")

    with txl.if_(tx == 0), txl.then_():
        for j in range(HD // E):
            txl.ptx[TMA2D](q_s.ptr_to([j * M * E]), txl.address_of(Q_map), txl.int32(j * E), txl.int32(0), bar.ptr_to([0]))
            for b in range(NB):
                txl.ptx[TMA2D](k_s.ptr_to([b * KVB * HD + j * KVB * E]), txl.address_of(K_map), txl.int32(j * E), txl.int32(b * KVB), bar.ptr_to([0]))
                txl.ptx[TMA2D](v_s.ptr_to([b * KVB * HD + j * KVB * E]), txl.address_of(V_map), txl.int32(j * E), txl.int32(b * KVB), bar.ptr_to([0]))
        bar.arrive(0, tx_count=(M + 2 * KV) * HD * 2)
    bar.wait(0, 0)

    q_tile = W.SmemTile(q_s.ptr_to([0]), rows=M)
    p_regs = p_half.view("uint32")

    def k_tile(b):
        return W.SmemTile(k_s.ptr_to([b * KVB * HD]), rows=KVB)

    def v_tile(b):
        return W.SmemTile(v_s.ptr_to([b * KVB * HD]), rows=KVB, major="MN")

    def qk_chain(b):
        W.chain(qk, s_acc, q_tile.k_steps(qk, QK_STEPS), k_tile(b).k_steps(qk, QK_STEPS), accumulate=False)

    def pv_chain(b, accumulate):
        W.chain(pv, o_acc, [_Slice(p_regs, s * 4) for s in range(PV_STEPS)], v_tile(b).k_steps(pv, PV_STEPS),
                accumulate=accumulate, transpose_b=True)

    def softmax_step(first):
        # m' = max(m, rowmax S); alpha = exp2((m - m') c); l = l alpha + rowsum P; P = exp2((S - m') c)
        mx = txl.alloc_local([2], "float32")
        for h in range(2):
            txl.assign(mx[h], T.float32(-3.0e38) if first else m_run[h])
        for r in range(qk.num_acc):
            h = (r % 4) // 2
            txl.assign(mx[h], T.max(mx[h], s_acc[r]))
        for h in range(2):
            for d in (1, 2):
                txl.assign(mx[h], T.max(mx[h], T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), mx[h], d, 32)))
            if first:
                txl.assign(alpha[h], T.float32(0.0))
                txl.assign(l_run[h], T.float32(0.0))
            else:
                txl.assign(alpha[h], T.exp2((m_run[h] - mx[h]) * T.float32(SCALE_LOG2)))
                txl.assign(l_run[h], l_run[h] * alpha[h])
            txl.assign(m_run[h], mx[h])
        for r in range(qk.num_acc):
            h = (r % 4) // 2
            p = T.exp2((s_acc[r] - m_run[h]) * T.float32(SCALE_LOG2))
            txl.assign(l_run[h], l_run[h] + p)
            txl.assign(p_half[r], T.Cast("bfloat16", p))

    def rescale_o():
        for r in range(pv.num_acc):
            txl.assign(o_acc[r], o_acc[r] * alpha[(r % 4) // 2])

    qk_chain(0)
    W.wait(0, (s_acc, qk.num_acc))
    if MODE == "s":  # debug: S = Q K_0^T itself
        for r in range(qk.num_acc):
            row, col = W.acc_coord(tx, r)
            txl.ptx["st.global.f32"](Out.ptr_to([row * HD + col]), s_acc[r])
        return
    softmax_step(True)
    if MODE == "ones":  # debug: P = 1, so O = column sums of V block 0
        for r in range(qk.num_acc):
            txl.assign(p_half[r], T.Cast("bfloat16", T.float32(1.0)))
        pv_chain(0, accumulate=False)
        W.wait(0, (o_acc, pv.num_acc))
        for r in range(pv.num_acc):
            row, col = W.acc_coord(tx, r)
            txl.ptx["st.global.f32"](Out.ptr_to([row * HD + col]), o_acc[r])
        return
    if MODE == "p":  # debug: P = exp2((S - rowmax) c) of block 0, before PV
        for r in range(qk.num_acc):
            row, col = W.acc_coord(tx, r)
            txl.ptx["st.global.f32"](Out.ptr_to([row * HD + col]), T.Cast("float32", p_half[r]))
        return
    if MODE == "loop":
        # FA3's steady state as a runtime loop over NITER blocks that cycle through the NB loaded ones
        qk_chain(1 % NB)
        pv_chain(0, accumulate=False)
        W.wait(1, (s_acc, qk.num_acc))
        softmax_step(False)
        W.wait(0, (o_acc, pv.num_acc))
        with txl.serial(NITER - 2) as i:
            b = i + 2
            qk_chain(b % NB)
            rescale_o()
            pv_chain((b - 1) % NB, accumulate=True)
            W.wait(1, (s_acc, qk.num_acc))
            softmax_step(False)
            W.wait(0, (o_acc, pv.num_acc))
        rescale_o()
        pv_chain((NITER - 1) % NB, accumulate=True)
        W.wait(0, (o_acc, pv.num_acc))
    for b in range(1, NB if MODE != "loop" else 1):
        if MODE == "overlap":
            qk_chain(b)
            if b > 1:
                rescale_o()
            pv_chain(b - 1, accumulate=b > 1)
            W.wait(1, (s_acc, qk.num_acc))
            softmax_step(False)
            W.wait(0, (o_acc, pv.num_acc))
        else:
            if b > 1:
                rescale_o()
            pv_chain(b - 1, accumulate=b > 1)
            W.wait(0, (o_acc, pv.num_acc))
            qk_chain(b)
            W.wait(0, (s_acc, qk.num_acc))
            softmax_step(False)
    if MODE != "loop":
        if NB > 1:
            rescale_o()
        pv_chain(NB - 1, accumulate=NB > 1)
        W.wait(0, (o_acc, pv.num_acc))
    # l is accumulated per lane (its own columns) in the loop, as FA3 does; the row sum is the quad's.
    for h in range(2):
        for d in (1, 2):
            txl.assign(l_run[h], l_run[h] + T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), l_run[h], d, 32))
    for r in range(pv.num_acc):
        row, col = W.acc_coord(tx, r)
        txl.ptx["st.global.f32"](Out.ptr_to([row * HD + col]), o_acc[r] / l_run[(r % 4) // 2])


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        self.ptr = ctypes.c_void_p((ctypes.addressof(self._storage) + 63) & ~63)


def encode2d(t, box_rows):
    desc = _AlignedTensorMap()
    rows, cols = t.shape
    tvm_ffi.get_global_func("runtime.cuTensorMapEncodeTiled")(
        desc.ptr, "bfloat16", 2, ctypes.c_void_p(int(t.data_ptr())),
        cols, rows, cols * 2, E, box_rows, 1, 1, 0, 3, 2, 0,
    )
    return desc


def main():
    if len(sys.argv) > 4 and sys.argv[4] == "regs":
        bc = {"cuda": {"arch": "sm_90a"}}
        ex = compile_kernel(probe.func, backend_config=bc)
        from tvm_tirx.backend.cuda.runtime import occupancy
        binary, _ = occupancy.kernel_binary(ex)
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(binary); f.flush()
            print(subprocess.run(["cuobjdump", "-res-usage", f.name], capture_output=True, text=True).stdout.strip()[-400:])
        src = ex.mod.imports[0].inspect_source()
        print([l for l in src.splitlines() if "launch_bounds" in l or "__global__" in l][:3])
        return 0
    try:
        ex = compile_kernel(probe.func, backend_config=BC)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"nb": NB, "mode": MODE, "compile": "FAILED", "error": str(e)[-2500:]}))
        return 1
    torch.manual_seed(0)
    q = torch.randn(M, HD, device="cuda").to(torch.bfloat16)
    k = torch.randn(KV, HD, device="cuda").to(torch.bfloat16)
    v = torch.randn(KV, HD, device="cuda").to(torch.bfloat16)
    o = torch.zeros(M, HD, device="cuda")
    maps = [encode2d(q, M), encode2d(k, KVB), encode2d(v, KVB)]
    ex(*[mp.ptr for mp in maps], o.view(-1))
    torch.cuda.synchronize()
    if MODE == "loop":
        idx = torch.cat([torch.arange(b % NB * KVB, (b % NB + 1) * KVB) for b in range(NITER)]).cuda()
        ref = torch.softmax(q.float() @ k[idx].float().t() / np.sqrt(HD), dim=-1) @ v[idx].float()
    else:
        ref = torch.softmax(q.float() @ k.float().t() / np.sqrt(HD), dim=-1) @ v.float()
    if MODE == "ones":
        ref = v[:KVB].float().sum(0, keepdim=True).expand(M, HD)
        d = (o - ref).abs()
        print("ONES max err", d.max().item(), "bad cols", sorted(set((d > 0.1).nonzero()[:, 1].tolist()))[:20], "o row0[:8]", o[0, :8].tolist(), "ref[:8]", ref[0, :8].tolist())
    if MODE == "p":
        sref = q.float() @ k[:KVB].float().t() / np.sqrt(HD)
        ref = torch.exp(sref - sref.max(-1, keepdim=True).values)
        bad = ((o - ref).abs() > 0.02).nonzero()
        print("P bad count", len(bad), "first", bad[:8].tolist())
    if MODE == "s":
        ref = q.float() @ k[:KVB].float().t()
        bad = ((o - ref).abs() > 0.05).nonzero()
        print("S bad count", len(bad), "first", bad[:8].tolist())
    err = (o - ref).abs().max().item()
    from tvm_tirx.backend.cuda.runtime import occupancy
    binary, _ = occupancy.kernel_binary(ex)
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(binary); f.flush()
        chk = subprocess.run([sys.executable, sys.argv[2], f.name], capture_output=True, text=True) if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    print(json.dumps({"nb": NB, "mode": MODE, "compile": "clean under perf_remarks=error", "max_abs_err": round(err, 4),
                      "ok": err < 2e-2, "sass_chains": chk.stdout.strip() if chk else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
