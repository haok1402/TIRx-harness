"""H200 task set: registration, targets, and the ptxas performance-remark hook."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from evolution.benchmark import adapter
from evolution.preparation.declare import load_task, task_path

REPO_ROOT = Path(__file__).resolve().parents[2]
H200_TASKS = {
    "rmsnorm_h200": "rmsnorm_h200",
    "fp16_gemm_h200": "fp16_gemm_floor",
    "fp8_blockscale_gemm_h200": "fp8_blockscale_gemm_h200",
}


def test_fp8_spec_states_untimed_scale_layout():
    spec = load_task(task_path(REPO_ROOT, "fp8_blockscale_gemm_h200")).spec
    assert "Scale transposition" in spec and "excluded from timing on both sides" in spec


def test_fp8_oracle_bound_accepts_rounding_and_rejects_errors():
    torch = pytest.importorskip("torch")
    from evolution.benchmark.flashinfer_bench_evolve.tasks.fp8_blockscale_gemm_h200 import (
        benchmark as fp8,
    )

    entry = {"axes": {"M": 300, "N": 520, "K": 256}}
    args = fp8.make_inputs(entry, torch.device("cpu"))
    assert args[1].shape == (2, 300)  # SFA: [K/128, align(M, 4)], MN-major
    expected, bound = fp8.reference(*args)
    assert fp8.compare(expected, (expected, bound))[0]
    off = expected.clone()
    off[3, 5] = (off[3, 5].view(torch.int16) + 1).view(torch.bfloat16)  # one ulp: still fine
    assert fp8.compare(off, (expected, bound))[0]
    bad = expected.clone()
    bad[-1] = bad[-2]  # a copied row
    assert not fp8.compare(bad, (expected, bound))[0]


@pytest.mark.parametrize(("task_name", "benchmark"), sorted(H200_TASKS.items()))
def test_h200_tasks_are_registered_for_sm90_only(task_name, benchmark):
    task = load_task(task_path(REPO_ROOT, task_name))
    assert task.target_archs == ["sm_90a"]
    assert task.workload_dir == f"candidates/{task_name}"
    assert adapter.PACKAGED[task_name][0] == benchmark
    assert "{{gpu}}" in task.spec


def test_perf_remarks_are_printed_with_the_score(monkeypatch, capsys):
    from evolution.benchmark.flashinfer_bench_evolve import benchmark_common as bc

    class Collector:
        def __init__(self):
            self.remarks = [
                "[C7514] in main_kernel (sm_90a): wgmma serialized",
                "[spill] in main_kernel",
            ]

    class Ctx:
        def __enter__(self):
            return Collector()

        def __exit__(self, *exc):
            return False

    fake = types.ModuleType("tvm_tirx.backend.cuda.perf_remarks")
    fake.collect_perf_remarks = Ctx
    monkeypatch.setitem(sys.modules, "tvm_tirx.backend.cuda.perf_remarks", fake)
    assert bc.prepare_with_perf_remarks(lambda x: x + 1, 41) == 42
    out = capsys.readouterr().out
    assert "perf_remark: [C7514] in main_kernel (sm_90a): wgmma serialized" in out
    assert "perf_remark: [spill] in main_kernel" in out


def test_perf_remark_hook_is_a_plain_call_without_a_reporting_tvm_tirx(monkeypatch, capsys):
    from evolution.benchmark.flashinfer_bench_evolve import benchmark_common as bc

    monkeypatch.setitem(sys.modules, "tvm_tirx.backend.cuda.perf_remarks", None)  # ImportError
    assert bc.prepare_with_perf_remarks(lambda: "ok") == "ok"
    assert capsys.readouterr().out == ""
