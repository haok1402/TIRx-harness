"""B3-baseline (plan F9): FA3 (flashinfer backend=fa3, batch ragged prefill), cuDNN SDPA and FA2 (torch flash SDPA)
at every B3 shape, bf16 hd128, causal and non-causal; same step, interleaved, L2-flushed. Outputs cross-checked."""
import json, sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_util import gpu_state, interleaved
import flashinfer

GATED = [(4, 16, 4096, 128), (2, 16, 8192, 128), (1, 16, 16384, 128), (8, 16, 4096, 128), (4, 32, 8192, 128)]
TRACKED = [(8, 32, 1024, 128), (16, 16, 2048, 128)]
dev = torch.device("cuda")
print(json.dumps({"state": gpu_state(), "flashinfer": flashinfer.__version__, "torch": torch.__version__}), flush=True)
ws = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
for (b, h, s, d) in GATED + TRACKED:
    for causal in (False, True):
        rec = {"shape": [b, h, s, d], "causal": causal, "gated": (b, h, s, d) in GATED}
        torch.manual_seed(0)
        q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        # flashinfer ragged layout: [b * s, h, d]
        qf, kf, vf = (x.transpose(1, 2).reshape(b * s, h, d).contiguous() for x in (q, k, v))
        indptr = torch.arange(0, (b + 1) * s, s, device=dev, dtype=torch.int32)
        fns, outs = {}, {}
        try:
            w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa3")
            w.plan(indptr, indptr, h, h, d, causal=causal, q_data_type=torch.bfloat16)
            fns["fa3"] = lambda: w.run(qf, kf, vf)
            outs["fa3"] = fns["fa3"]().view(b, s, h, d).transpose(1, 2)
        except Exception as e:  # noqa: BLE001
            rec["fa3_error"] = f"{type(e).__name__}: {e}"[:300]
        for name, be in (("cudnn", SDPBackend.CUDNN_ATTENTION), ("fa2", SDPBackend.FLASH_ATTENTION)):
            def f(be=be):
                with sdpa_kernel([be]):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            try:
                outs[name] = f()
                fns[name] = f
            except Exception as e:  # noqa: BLE001
                rec[f"{name}_error"] = f"{type(e).__name__}: {e}"[:200]
        ref = outs.get("cudnn", outs.get("fa2"))
        rec["max_abs_diff_vs_cudnn"] = {k2: round(float((o.float() - ref.float()).abs().max()), 4) for k2, o in outs.items()}
        t = interleaved(fns, reps=20)
        fl = 4 * b * h * s * s * d / (2 if causal else 1)
        rec["tflops"] = {k2: round(fl / us / 1e6, 1) for k2, us in t.items()}
        rec["us"] = {k2: round(us, 1) for k2, us in t.items()}
        best = max(rec["tflops"].get("fa3", 0), rec["tflops"].get("cudnn", 0))
        rec["target_0.90x_tflops"] = round(0.9 * best, 1)
        print(json.dumps(rec), flush=True)
print(json.dumps({"state_after": gpu_state()}), flush=True)
