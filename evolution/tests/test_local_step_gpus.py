"""Local mode inside a job step: the step's GPUs are used as-is, and CUDA_VISIBLE_DEVICES may hold UUIDs."""

from __future__ import annotations

from pathlib import Path

import pytest

from evolution import cli
from evolution.preparation.declare import load_task, load_toolset, task_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prompt(tmp_path, **kw):
    task = load_task(task_path(REPO_ROOT, "fp16_gemm_floor"))
    toolset = load_toolset(REPO_ROOT / "evolution/toolsets/kda_flow.yaml")
    return cli._prompt(task=task, toolset=toolset, worktree=tmp_path, **kw)


def test_shared_machine_prompt_still_picks_a_gpu(tmp_path):
    prompt = _prompt(tmp_path)
    assert "You MUST pick an idle one" in prompt
    assert "env CUDA_VISIBLE_DEVICES=<picked>" in prompt


def test_step_prompt_forbids_rewriting_visible_devices(tmp_path):
    prompt = _prompt(tmp_path, allocated_gpus="0")
    assert "Slurm sets `CUDA_VISIBLE_DEVICES=0` for the step" in prompt
    assert "Do NOT set, change, or unset `CUDA_VISIBLE_DEVICES`" in prompt
    assert "<picked>" not in prompt and "pick an idle one" not in prompt
    assert "{{" not in prompt


@pytest.mark.parametrize(
    ("flag", "env", "remote", "expected"),
    [
        (None, {}, False, None),  # shared machine
        (None, {"SLURM_STEP_GPUS": "7", "CUDA_VISIBLE_DEVICES": "0"}, False, "0"),  # auto in a step
        (None, {"SLURM_JOB_GPUS": "2,3", "CUDA_VISIBLE_DEVICES": "0,1"}, False, "0,1"),
        ("auto", {"CUDA_VISIBLE_DEVICES": "GPU-438f"}, False, "GPU-438f"),
        ("1", {}, False, "1"),
        (None, {"SLURM_STEP_GPUS": "7", "CUDA_VISIBLE_DEVICES": "0"}, True, None),  # remote: n/a
    ],
)
def test_allocated_gpus_detection(monkeypatch, flag, env, remote, expected):
    for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert cli._allocated_gpus(flag, remote=remote) == expected


SMI = (
    "0, GPU-aaaa1111-0000-0000-0000-000000000000, 70000, 90\n"
    "1, GPU-bbbb2222-0000-0000-0000-000000000000, 10, 0\n"
    "2, GPU-cccc3333-0000-0000-0000-000000000000, 5, 0\n"
)


@pytest.mark.parametrize(
    ("visible", "expected_local"),
    [
        (None, 2),  # all visible: least-loaded physical GPU 2 is cuda:2
        ("0,1", 1),  # numeric: GPU 1 is the second visible device
        ("GPU-cccc3333-0000-0000-0000-000000000000,GPU-aaaa1111-0000-0000-0000-000000000000", 0),
        ("GPU-aaaa,GPU-bbbb", 1),  # UUID prefixes, as CUDA accepts them
        ("MIG-ffff0000-1111", 0),  # unmatched (MIG): trust the allocation, first visible device
    ],
)
def test_choose_device_accepts_uuids(monkeypatch, visible, expected_local):
    from evolution.benchmark.flashinfer_bench_evolve import benchmark_common as bc

    monkeypatch.setattr(bc.subprocess, "check_output", lambda *a, **k: SMI)
    chosen = []
    monkeypatch.setattr(bc.torch.cuda, "set_device", lambda device: chosen.append(device))
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    device = bc.choose_device("auto")
    assert device.index == expected_local
    assert chosen == [device]
