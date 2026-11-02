# SPDX-License-Identifier: Apache-2.0
"""SM90 probe: thread-block clusters + distributed shared memory, and PDL.

dsmem_ld      : 2-CTA cluster; each CTA reads its peer's smem via mapa + ld.shared::cluster.
dsmem_bulk    : CTA0 pushes an smem buffer into CTA1 with cp.async.bulk.shared::cluster.shared::cta
                completing on CTA1's mbarrier (expect_tx on the receiver).
dsmem_st_async: CTA0 writes words into CTA1 with st.async ... mbarrier::complete_tx::bytes.
pdl           : producer kernel triggers griddepcontrol.launch_dependents, consumer launched with
                programmatic_stream_serialization waits on griddepcontrol.wait.
host_call_us  : host time per call of a tiny TIRx kernel invoked from Python through tvm-ffi.
"""

# ruff: noqa: F841
import json
import os
import time

import torch

import tvm_tirx
from tvm_tirx.script import tirx as T

ARCH = os.environ.get("PROBE_ARCH", "sm_90a")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
NW = 256  # words per CTA buffer


def compile_fn(func, name):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": ARCH})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    open(os.path.join(OUT, name + ".cu"), "w").write(mod.mod.imports[0].inspect_source())
    return mod


# fmt: off
@T.function
def dsmem_ld(Out: T.Tensor((2, NW), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=NW, cluster=2))
    rank = T.cuda.cluster_cta_id("x")
    tx = T.cuda.thread_idx("x")
    buf = T.alloc_tensor(ty_args=[T.Tensor((NW,), "uint32", scope="shared", align=16)])
    remote = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
    val = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
    buf[tx] = T.Cast("uint32", rank * 100000 + tx)
    T.cuda.cluster_sync()
    T.ptx.mapa.shared__cluster.u32(remote[0], T.cuda.cvta_generic_to_shared(T.address_of(buf[tx])),
                                    T.Cast("uint32", 1 - rank))
    T.ptx.ld.shared__cluster.u32(val[0], remote[0])
    Out[rank, tx] = val[0]
    T.cuda.cluster_sync()


@T.function
def dsmem_bulk(A: T.Tensor((NW,), "uint32"), Out: T.Tensor((NW,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=NW, cluster=2))
    rank = T.cuda.cluster_cta_id("x")
    tx = T.cuda.thread_idx("x")
    src = T.alloc_tensor(ty_args=[T.Tensor((NW,), "uint32", scope="shared", align=128)])
    dst = T.alloc_tensor(ty_args=[T.Tensor((NW,), "uint32", scope="shared", align=128)])
    bar = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint64", scope="shared", align=8)])
    rdst = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
    rbar = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
    if tx == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(bar[0]), T.uint32(1))
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if rank == 0:
        src[tx] = A[tx]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if tx == 0:
            T.ptx.mapa.shared__cluster.u32(rdst[0], T.cuda.cvta_generic_to_shared(T.address_of(dst[0])), T.uint32(1))
            T.ptx.mapa.shared__cluster.u32(rbar[0], T.cuda.cvta_generic_to_shared(T.address_of(bar[0])), T.uint32(1))
            T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
                rdst[0], T.cuda.cvta_generic_to_shared(T.address_of(src[0])), T.uint32(NW * 4), rbar[0])
    else:
        if tx == 0:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(bar[0]), T.uint32(NW * 4))
        T.cuda.mbarrier_wait(T.address_of(bar[0]), 0)
        Out[tx] = dst[tx]
    T.cuda.cluster_sync()


@T.function
def dsmem_st_async(Out: T.Tensor((NW,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=NW, cluster=2))
    rank = T.cuda.cluster_cta_id("x")
    tx = T.cuda.thread_idx("x")
    dst = T.alloc_tensor(ty_args=[T.Tensor((NW,), "uint32", scope="shared", align=16)])
    bar = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint64", scope="shared", align=8)])
    rdst = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
    rbar = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
    if tx == 0:
        T.ptx.mbarrier.init.shared.b64(T.address_of(bar[0]), T.uint32(1))
    T.ptx.fence.mbarrier_init.release.cluster()
    T.cuda.cluster_sync()
    if rank == 1:
        if tx == 0:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(bar[0]), T.uint32(NW * 4))
    T.cuda.cluster_sync()
    if rank == 0:
        T.ptx.mapa.shared__cluster.u32(rdst[0], T.cuda.cvta_generic_to_shared(T.address_of(dst[tx])), T.uint32(1))
        T.ptx.mapa.shared__cluster.u32(rbar[0], T.cuda.cvta_generic_to_shared(T.address_of(bar[0])), T.uint32(1))
        T.ptx.st_async.shared__cluster.mbarrier__complete_tx__bytes.b32(
            rdst[0], T.Cast("uint32", tx * 7 + 3), rbar[0])
    else:
        T.cuda.mbarrier_wait(T.address_of(bar[0]), 0)
        Out[tx] = dst[tx]
    T.cuda.cluster_sync()


@T.function
def pdl_producer(X: T.Tensor((NW * 1024,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1024, block=NW))
    i = T.cuda.block_idx("x") * NW + T.cuda.thread_idx("x")
    T.ptx.griddepcontrol.wait()
    X[i] = T.Cast("float32", i) * T.float32(2.0)
    T.ptx.griddepcontrol.launch_dependents()


@T.function
def pdl_consumer(X: T.Tensor((NW * 1024,), "float32"), Y: T.Tensor((NW * 1024,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1024, block=NW, programmatic_stream_serialization=True))
    i = T.cuda.block_idx("x") * NW + T.cuda.thread_idx("x")
    T.ptx.griddepcontrol.wait()
    Y[i] = X[i] + T.float32(1.0)
    T.ptx.griddepcontrol.launch_dependents()


@T.function
def pref_cluster(Out: T.Tensor((8,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=8, block=32, cluster=[2, 1], preferred_cluster=[4, 1]))
    if T.cuda.thread_idx("x") == 0:
        Out[T.cuda.block_idx("x")] = T.Cast("uint32", T.cuda.cluster_cta_id("x") + 1)


@T.function
def tiny(X: T.Tensor((128,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
    X[T.cuda.thread_idx("x")] = X[T.cuda.thread_idx("x")] + T.float32(1.0)
# fmt: on


def case(name, fn):
    rec = {"case": name}
    try:
        rec.update(fn())
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:2500]}")
    print(json.dumps(rec), flush=True)
    return rec


def t_dsmem_ld():
    mod = compile_fn(dsmem_ld, "dsmem_ld")
    out = torch.zeros(2, NW, dtype=torch.uint32, device="cuda")
    mod(out)
    o = out.view(torch.int32).cpu()
    ar = torch.arange(NW, dtype=torch.int32)
    return {"ok": bool((o[0] == 100000 + ar).all() and (o[1] == ar).all())}


def t_dsmem_bulk():
    mod = compile_fn(dsmem_bulk, "dsmem_bulk")
    a = torch.randint(0, 2**30, (NW,), dtype=torch.int32, device="cuda")
    out = torch.zeros(NW, dtype=torch.int32, device="cuda")
    mod(a.view(torch.uint32), out.view(torch.uint32))
    return {"ok": bool((a == out).all().item())}


def t_dsmem_st_async():
    mod = compile_fn(dsmem_st_async, "dsmem_st_async")
    out = torch.zeros(NW, dtype=torch.int32, device="cuda")
    mod(out.view(torch.uint32))
    return {"ok": bool((out.cpu() == torch.arange(NW) * 7 + 3).all())}


def t_pdl():
    p = compile_fn(pdl_producer, "pdl_producer")
    c = compile_fn(pdl_consumer, "pdl_consumer")
    n = NW * 1024
    x = torch.zeros(n, device="cuda")
    y = torch.zeros(n, device="cuda")
    ok = True
    for _ in range(20):
        x.zero_()
        y.zero_()
        p(x)
        c(x, y)
        torch.cuda.synchronize()
        ok &= bool((y == torch.arange(n, device="cuda") * 2.0 + 1).all().item())
    src = open(os.path.join(OUT, "pdl_consumer.cu")).read()
    return {"ok": ok, "griddepcontrol_in_src": src.count("griddepcontrol")}


def t_pref_cluster():
    mod = compile_fn(pref_cluster, "pref_cluster")
    out = torch.zeros(8, dtype=torch.int32, device="cuda")
    mod(out.view(torch.uint32))
    torch.cuda.synchronize()
    return {"ok": bool((out > 0).all().item()), "cluster_ranks": out.tolist()}


def t_host_call():
    mod = compile_fn(tiny, "tiny")
    x = torch.zeros(128, device="cuda")
    f = mod["main"] if hasattr(mod, "__getitem__") else mod
    for _ in range(1000):
        f(x)
    torch.cuda.synchronize()
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        f(x)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    tt0 = time.perf_counter()
    for _ in range(n):
        x.add_(1.0)
    tt1 = time.perf_counter()
    torch.cuda.synchronize()
    return {"ok": True, "tirx_host_us_per_call": (t1 - t0) / n * 1e6,
            "torch_add_host_us_per_call": (tt1 - tt0) / n * 1e6}


if __name__ == "__main__":
    res = [case("dsmem_ld", t_dsmem_ld), case("dsmem_bulk", t_dsmem_bulk),
           case("dsmem_st_async", t_dsmem_st_async), case("pdl", t_pdl), case("preferred_cluster", t_pref_cluster),
           case("host_call_us", t_host_call)]
    print("SUMMARY", sum(r.get("ok", False) for r in res), "/", len(res))
