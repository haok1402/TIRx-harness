"""B3 gate: fa3_forward_sm90 vs FA3 (flashinfer backend=fa3, ragged prefill), interleaved and L2-flushed, same step.
Same ragged NHD tensors for both; outputs cross-checked. argv: [gated|all]"""
import json, sys
import torch
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/b")
from bench_util import gpu_state, interleaved
import flashinfer
from tirx_kernels.attention import fa3_forward_sm90 as fa

GATED = [(4, 16, 4096), (2, 16, 8192), (1, 16, 16384), (8, 16, 4096), (4, 32, 8192)]
TRACKED = [(8, 32, 1024), (16, 16, 2048)]
BC = {"cuda": {"arch": "sm_90a", "perf_remarks": "error"}}
print(json.dumps({"state": gpu_state(), "flashinfer": flashinfer.__version__}), flush=True)
ws = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
launchers = {}
VARIANTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["pp"]  # pp: ping-pong, nopp: without
res = []
for (b, h, s) in GATED + (TRACKED if len(sys.argv) > 1 and sys.argv[1] == "all" else []):
    for causal in (False, True):
        lens = (s,) * b
        q, k, v = fa.prepare_data(lens, h)
        o = torch.empty_like(q)
        runs = {}
        for var in VARIANTS:
            key = (h, causal, var)
            if key not in launchers:
                launchers[key] = fa.Launcher(lens, h, causal, backend_config=BC, pingpong={"pp": True, "nopp": False, "def": None}[var])
            L = launchers[key]
            L.seq_lens = lens
            L.cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
            L.tiles, L.num_tiles = fa.tile_table(lens, h, causal)
            runs[var] = L.bind(q, k, v, o)
        run = runs[VARIANTS[0]]
        indptr = L.cu
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa3")
        w.plan(indptr, indptr, h, h, 128, causal=causal, q_data_type=torch.bfloat16)
        ref = w.run(q, k, v)
        run(); torch.cuda.synchronize()
        diff = (o.float() - ref.float()).abs().max().item()
        t = interleaved({**{f"tirx_{var}": r for var, r in runs.items()}, "fa3": lambda: w.run(q, k, v)}, reps=20)
        fl = 4 * b * h * s * s * 128 / (2 if causal else 1)
        rec = {"shape": [b, h, s], "causal": causal, "gated": (b, h, s) in GATED,
               "tflops": {k2: round(fl / us / 1e6, 1) for k2, us in t.items()},
               **{f"ratio_{var}": round(t["fa3"] / t[f"tirx_{var}"], 3) for var in runs},
               "ratio_vs_fa3": round(t["fa3"] / t[f"tirx_{VARIANTS[0]}"], 3), "max_abs_diff_vs_fa3": round(diff, 4)}
        res.append(rec)
        print(json.dumps(rec), flush=True)
g = [r["ratio_vs_fa3"] for r in res if r["gated"]]
print(json.dumps({"gated_min_ratio": min(g), "gated_ratios": g, "gate_0.90": min(g) >= 0.9}))
print(json.dumps({"state_after": gpu_state()}), flush=True)
