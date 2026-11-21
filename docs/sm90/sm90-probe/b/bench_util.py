# SPDX-License-Identifier: Apache-2.0
"""Same-step, interleaved, L2-flushed timing (plan v2 rule) + hostname/clock/throttle provenance."""
import socket, subprocess, statistics
import torch

_FLUSH = None


def gpu_state():
    q = "clocks.sm,clocks.max.sm,clocks_throttle_reasons.active,power.draw,temperature.gpu"
    idx = torch.cuda.current_device()
    try:
        out = subprocess.run(["nvidia-smi", f"--id={idx}", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:  # noqa: BLE001
        out = f"unavailable: {e}"
    return {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(), "smi": out}


def interleaved(fns: dict, reps=50, warmup=5, flush_mb=256):
    """Median microseconds per fn; fns alternate call by call; L2 flushed before every call."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(flush_mb << 20, dtype=torch.uint8, device="cuda")
    for _ in range(warmup):
        for f in fns.values():
            f()
    torch.cuda.synchronize()
    times = {k: [] for k in fns}
    for _ in range(reps):
        for k, f in fns.items():
            _FLUSH.zero_()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); f(); e.record()
            torch.cuda.synchronize()
            times[k].append(s.elapsed_time(e) * 1e3)
    return {k: statistics.median(v) for k, v in times.items()}
