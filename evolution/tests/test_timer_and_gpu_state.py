"""Scoring timer fallback (cold-L2 CUDA events, BENCH_REQUIRE_CUPTI) and per-score GPU provenance."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
from evolution.benchmark.flashinfer_bench_evolve import benchmark_common as bc  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture
def no_cupti(monkeypatch):
    fake = types.ModuleType("flashinfer.testing")

    def broken(**kw):
        raise RuntimeError("CUPTI disabled for this test")

    fake.bench_gpu_time_with_cupti = broken
    monkeypatch.setitem(sys.modules, "flashinfer.testing", fake)


@needs_gpu
def test_fallback_times_each_call_with_a_cold_l2(no_cupti, monkeypatch, capsys):
    monkeypatch.delenv("BENCH_REQUIRE_CUPTI", raising=False)
    device = torch.device("cuda", torch.cuda.current_device())
    flushes = []
    real_flush = bc._flush_l2
    monkeypatch.setattr(bc, "_flush_l2", lambda d: (flushes.append(d), real_flush(d)))
    x = torch.randn(1 << 20, device=device)
    calls = []

    def fn(t):
        calls.append(1)
        t.mul_(1.0)

    ms, timer = bc.time_fn(fn, [x], device, warmup=2, iters=7)
    assert timer == "cuda_event_cold_l2"
    assert ms > 0
    assert len(calls) == 2 + 7
    assert len(flushes) == 7  # one flush before every timed call, none during warmup
    assert "using cold-L2 CUDA events" in capsys.readouterr().out
    buf = bc._L2_FLUSH[device.index]
    assert buf.numel() >= 2 * torch.cuda.get_device_properties(device).L2_cache_size


@needs_gpu
def test_require_cupti_refuses_to_score(no_cupti, monkeypatch):
    monkeypatch.setenv("BENCH_REQUIRE_CUPTI", "1")
    device = torch.device("cuda", torch.cuda.current_device())
    with pytest.raises(RuntimeError, match="BENCH_REQUIRE_CUPTI=1"):
        bc.time_fn(lambda: None, [], device, warmup=1, iters=1)


class _Props:
    name = "NVIDIA H200"
    pci_domain_id = 0
    pci_bus_id = 0xB7
    pci_device_id = 0


def test_gpu_state_queries_the_gpu_by_pci_bus_id(monkeypatch):
    monkeypatch.setattr(bc.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(bc.torch.cuda, "get_device_properties", lambda i: _Props())
    seen = []

    def smi(cmd, **kw):
        seen.append(cmd)
        return "1980, 1980, 0x0000000000000000, 41, 452.18\n"

    monkeypatch.setattr(bc.subprocess, "check_output", smi)
    state = bc.gpu_state()
    assert seen[0][:3] == ["nvidia-smi", "-i", "00000000:b7:00.0"]
    assert state["gpu"] == "NVIDIA H200"
    assert (state["sm_clock_mhz"], state["sm_clock_max_mhz"]) == (1980, 1980)
    assert state["throttle_reasons"] == "0x0000000000000000"
    assert (state["temperature_c"], state["power_w"]) == (41, 452.18)
    assert state["host"]


def test_gpu_state_never_breaks_a_score(monkeypatch):
    monkeypatch.setattr(bc.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(bc.torch.cuda, "get_device_properties", lambda i: _Props())

    def smi(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(bc.subprocess, "check_output", smi)
    state = bc.gpu_state()
    assert "gpu_state_error" in state and "nvidia-smi" in state["gpu_state_error"]
    assert state["host"] and state["gpu"] == "NVIDIA H200"


@needs_gpu
def test_gpu_state_on_this_gpu():
    state = bc.gpu_state(torch.device("cuda", torch.cuda.current_device()))
    assert "gpu_state_error" not in state, state
    assert state["sm_clock_max_mhz"] >= state["sm_clock_mhz"] > 0
