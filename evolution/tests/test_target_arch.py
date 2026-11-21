"""Target architecture as task data: declaration, refusal, and spec/prompt rendering."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evolution import cli
from evolution.preparation import target as gpu_target
from evolution.preparation.declare import DEFAULT_TARGET_ARCHS, load_task, load_toolset, task_path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = sorted((REPO_ROOT / "evolution/tasks").glob("*.yaml"))
H200 = gpu_target.GpuTarget("sm_90a", "H200", 132, source="remote")
B200 = gpu_target.GpuTarget("sm_100a", "B200", 148, source="remote")


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.stem)
def test_every_spec_renders_without_hardcoded_gpu(path):
    task = load_task(path)
    assert task.target_archs, path
    for target in (B200, H200):
        text = task.render_spec(target)
        assert "{{" not in text, f"{path.name}: unfilled slot"
    # The GPU is data now: no spec names B200 / sm_100 literally.
    assert not re.search(r"B200\s+\(sm_100\)", task.spec), path.name


def test_undeclared_tasks_keep_their_b200_target():
    task = load_task(task_path(REPO_ROOT, "kda_forward"))
    assert task.target_archs == list(DEFAULT_TARGET_ARCHS) == ["sm_100a"]
    assert "Kimi Delta Attention FORWARD on B200 (sm_100)" in task.render_spec(B200)


def test_b200_only_task_is_refused_on_h200():
    task = load_task(task_path(REPO_ROOT, "kda_forward"))
    with pytest.raises(ValueError, match=r"targets sm_100a, but the run's GPU is H200 \(sm_90\)"):
        task.check_target(H200)
    task.check_target(B200)


def test_gemm_floor_runs_on_both_and_renders_h200():
    task = load_task(task_path(REPO_ROOT, "fp16_gemm_floor"))
    task.check_target(H200)
    task.check_target(B200)
    assert task.render_spec(H200).startswith("Implement FP16 GEMM on H200 (sm_90)")


def test_unknown_arch_and_slot_are_rejected(tmp_path):
    base = (REPO_ROOT / "evolution/tasks/fp16_gemm_floor.yaml").read_text()
    bad_arch = tmp_path / "bad_arch.yaml"
    bad_arch.write_text(base.replace("[sm_100a, sm_90a]", "[sm_80]"))
    with pytest.raises(ValueError, match="unknown target_archs"):
        load_task(bad_arch)
    bad_slot = tmp_path / "bad_slot.yaml"
    bad_slot.write_text(base.replace("{{gpu}}", "{{gpu_model}}"))
    with pytest.raises(ValueError, match="unknown spec slots"):
        load_task(bad_slot)


def test_prompt_renders_target(tmp_path):
    task = load_task(task_path(REPO_ROOT, "fp16_gemm_floor"))
    toolset = load_toolset(REPO_ROOT / "evolution/toolsets/kda_flow.yaml")
    prompt = cli._prompt(task=task, toolset=toolset, worktree=tmp_path, target=H200)
    assert "FP16 GEMM on H200 (sm_90)" in prompt
    assert "B200" not in prompt and "{{" not in prompt


def test_task_authored_prompt_names_the_gpu_family():
    from evolution.prompts.render import kernel_search_references

    task = next(t for t in map(load_task, TASKS) if t.kernel_authoring == "task")
    assert "study Hopper optimization patterns" in kernel_search_references(task, H200.family)
    assert "study Blackwell optimization patterns" in kernel_search_references(task, B200.family)


def test_resolve_prefers_flags_and_falls_back_to_task_default(monkeypatch, capsys):
    assert gpu_target.resolve(task_archs=["sm_100a"], remote_url=None, arch="sm_90a").label == (
        "H200 (sm_90)"
    )

    def no_gpu():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu_target, "detect_local", no_gpu)
    fallback = gpu_target.resolve(task_archs=["sm_100a"], remote_url=None)
    assert (fallback.arch, fallback.source) == ("sm_100a", "default")
    assert "could not detect the target GPU" in capsys.readouterr().err


def test_from_properties_maps_hopper():
    t = gpu_target._from_properties(
        {"name": "NVIDIA H200", "major": 9, "minor": 0, "sms": 132}, "local"
    )
    assert (t.arch, t.name, t.num_sms, t.family) == ("sm_90a", "H200", 132, "Hopper")


def test_b300_runs_sm100_tasks_as_before():
    b300 = gpu_target.GpuTarget("sm_103a", "B300", 148, source="remote")
    task = load_task(task_path(REPO_ROOT, "kda_forward"))
    task.check_target(b300)
    assert "on B300 (sm_103)" in task.render_spec(b300)
    with pytest.raises(ValueError):
        load_task(task_path(REPO_ROOT, "kda_forward")).check_target(H200)


def test_detected_unknown_arch_is_an_error_not_a_b200_fallback(monkeypatch):
    def blackwell_consumer():
        return gpu_target._from_properties(
            {"name": "NVIDIA RTX 5090", "major": 12, "minor": 0}, "local"
        )

    monkeypatch.setattr(gpu_target, "detect_local", blackwell_consumer)
    with pytest.raises(gpu_target.UnknownArchError, match="sm_120a"):
        gpu_target.resolve(task_archs=["sm_100a"], remote_url=None)


def test_wrapped_specs_keep_their_line_break():
    rendered = load_task(task_path(REPO_ROOT, "moe")).render_spec(B200)
    assert "Mixture-of-Experts on B200\n(sm_100) and beat" in rendered
