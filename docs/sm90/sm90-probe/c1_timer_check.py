"""C1: which timer scores, and the cold-L2 fallback / refusal when CUPTI is unavailable."""
import os, sys
sys.path.insert(0, os.getcwd())
mode = sys.argv[1]
if mode != "cupti":
    import flashinfer.testing as ft
    def broken(*a, **k):
        raise RuntimeError("CUPTI disabled for this check")
    ft.bench_gpu_time_with_cupti = broken
    if mode == "refuse":
        os.environ["BENCH_REQUIRE_CUPTI"] = "1"
from evolution.benchmark import adapter
try:
    rows = adapter.run_version("fp16_gemm_floor", None, "baseline", warmup=3, repeat=10, shape_mode="all")
except Exception as e:  # noqa: BLE001
    print(f"RESULT {mode}: raised {type(e).__name__}: {str(e)[:160]}")
else:
    r = rows[0]
    print(f"RESULT {mode}: rows={len(rows)} timer={r.get('timer')} host={r.get('host')} sm={r.get('sm_clock_mhz')}/{r.get('sm_clock_max_mhz')} "
          f"throttle={r.get('throttle_reasons')} speedup={r.get('speedup')} after={r.get('gpu_state_after', {}).get('sm_clock_mhz')}")
