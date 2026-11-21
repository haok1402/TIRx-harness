# SPDX-License-Identifier: Apache-2.0
"""H200 baseline numbers for the SM90 plan's kernel targets. One JSON line per case.
Timing: CUDA events, 256 MB L2 flush before every rep, median of reps."""
import json, math, sys, traceback
import torch

dev = torch.device("cuda", 0)
FLUSH = torch.empty(256 << 20, dtype=torch.uint8, device=dev)

def bench(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        FLUSH.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]

def emit(**kw):
    print(json.dumps(kw), flush=True)

def run(name, f):
    try:
        f()
    except BaseException as ex:  # noqa: BLE001
        emit(case=name, ok=False, err=f"{type(ex).__name__}: {ex}"[:500])

def gemm_bf16():
    for (m, n, k) in [(4096,)*3, (8192,)*3, (2048, 4096, 8192), (8192, 8192, 1024), (4096, 7168, 2048), (16384, 2112, 7168)]:
        a = torch.randn(m, k, device=dev, dtype=torch.bfloat16); b = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
        us = bench(lambda: a @ b.T)
        emit(case="cublas_bf16", shape=[m, n, k], us=round(us, 2), tflops=round(2*m*n*k/us/1e6, 1))

def gemm_fp8():
    import deep_gemm
    for (m, n, k) in [(4096, 7168, 2048), (4096, 2112, 7168), (4096, 24576, 1536), (4096, 4096, 7168), (4096, 7168, 16384), (128, 7168, 2048), (8192, 8192, 8192)]:
        a = torch.randn(m, k, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn)
        sa = torch.ones(m, k // 128, device=dev, dtype=torch.float32)
        sb = torch.ones(n // 128 if n % 128 == 0 else n // 128 + 1, k // 128, device=dev, dtype=torch.float32)
        d = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        sa_t = deep_gemm.get_mn_major_tma_aligned_tensor(sa) if hasattr(deep_gemm, "get_mn_major_tma_aligned_tensor") else sa
        f = lambda: deep_gemm.fp8_gemm_nt((a, sa_t), (b, sb), d)
        us = bench(f)
        emit(case="deepgemm_fp8_1x128_128x128", shape=[m, n, k], us=round(us, 2), tflops=round(2*m*n*k/us/1e6, 1))

def gemm_fp8_torch():
    for (m, n, k) in [(4096, 7168, 2048), (4096, 4096, 7168), (8192, 8192, 8192)]:
        a = torch.randn(m, k, device=dev).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn)
        sa = torch.ones(m, 1, device=dev); sb = torch.ones(1, n, device=dev)
        f = lambda: torch._scaled_mm(a, b.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        us = bench(f)
        emit(case="cublas_fp8_rowwise(_scaled_mm)", shape=[m, n, k], us=round(us, 2), tflops=round(2*m*n*k/us/1e6, 1))

def rmsnorm():
    import flashinfer
    for hs in [4096, 8192]:
        for bs in [128, 4113, 16384]:
            x = torch.randn(bs, hs, device=dev, dtype=torch.float16); w = torch.randn(hs, device=dev, dtype=torch.float16)
            out = torch.empty_like(x)
            us = bench(lambda: flashinfer.norm.rmsnorm(x, w, 1e-6, out=out))
            ust = bench(lambda: torch.nn.functional.rms_norm(x, (hs,), w, 1e-6))
            gb = 2 * x.numel() * 2 / 1e9
            emit(case="rmsnorm", hs=hs, bs=bs, flashinfer_us=round(us, 2), flashinfer_TBps=round(gb/us*1e3, 2),
                 torch_us=round(ust, 2), torch_TBps=round(gb/ust*1e3, 2))

def attn():
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for causal in [False, True]:
        for (b, h, s, d) in [(4, 16, 4096, 128), (1, 16, 16384, 128), (2, 16, 8192, 128), (8, 32, 1024, 128)]:
            q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16); k = torch.randn_like(q); v = torch.randn_like(q)
            flops = 4 * b * h * s * s * d / (2 if causal else 1)
            for be in [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]:
                try:
                    with sdpa_kernel([be]):
                        us = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
                    emit(case=f"sdpa_{be.name}", causal=causal, shape=[b, h, s, d], us=round(us, 1), tflops=round(flops/us/1e6, 1))
                except BaseException as ex:  # noqa: BLE001
                    emit(case=f"sdpa_{be.name}", ok=False, err=str(ex)[:200])
            try:
                import flashinfer
                qf = q[0].transpose(0, 1).contiguous() if b == 1 else None
                if qf is not None:
                    kf = k[0].transpose(0, 1).contiguous(); vf = v[0].transpose(0, 1).contiguous()
                    for backend in ["fa3", "fa2"]:
                        f = lambda: flashinfer.single_prefill_with_kv_cache(qf, kf, vf, causal=causal, backend=backend)
                        us = bench(f, reps=10)
                        emit(case=f"flashinfer_{backend}", causal=causal, shape=[b, h, s, d], us=round(us, 1), tflops=round(flops/us/1e6, 1))
            except BaseException as ex:  # noqa: BLE001
                emit(case="flashinfer_attn", ok=False, err=f"{type(ex).__name__}: {ex}"[:300])

for c in sys.argv[1:] or ["gemm_bf16", "gemm_fp8_torch", "gemm_fp8", "rmsnorm", "attn"]:
    run(c, globals()[c])
