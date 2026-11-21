#!/usr/bin/env python3
"""Standalone benchmark for H200 RMSNorm candidates (FlashInfer baseline)."""

from __future__ import annotations

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    config_from_env,
    load_candidate_override,
    load_task_workloads,
    prepare_with_perf_remarks,
    run_benchmark,
)

TASK_NAME = "rmsnorm_h200"
EPS = 1e-6

# fp16 output of an fp32-accumulated normalization: 1e-3 is about one fp16 ulp at |y| ~ 1.
ATOL = 1e-3
RTOL = 1e-3
REQUIRED_MATCHED_RATIO = None


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=10, iters=100, trials=3)


def make_workloads(config: BenchConfig | None = None):
    cfg = config or default_config()
    official = []
    if cfg.include_official:
        for row in load_task_workloads(
            TASK_NAME,
            cfg.max_official,
            shape_mode=cfg.shape_mode,
            shape_axes=("batch_size", "hidden_size"),
        ):
            workload = row["workload"]
            official.append(
                {
                    "suite": "official",
                    "id": workload["uuid"],
                    "axes": workload["axes"],
                    "raw": workload,
                }
            )
    return official


def make_inputs(entry, device):
    axes = entry["axes"]
    torch.manual_seed(0)
    return [
        torch.randn((axes["batch_size"], axes["hidden_size"]), dtype=torch.float16, device=device),
        torch.randn((axes["hidden_size"],), dtype=torch.float16, device=device),
    ]


def tirx_prepare(solution_module, x, w):
    """Compile and bind a TIRx RMSNorm candidate outside timing."""

    batch_size, hidden_size = x.shape
    data = {"X": x, "W": w, "Y": torch.empty_like(x)}
    kernel_fn = prepare_with_perf_remarks(solution_module.setup, data, batch_size, hidden_size)
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["Y"]


def _print_aggregate(rows):
    valid = [
        row
        for row in rows
        if row["passed"] and row["baseline_ms"] is not None and row["kernel_ms"] is not None
    ]
    total_baseline_ms = sum(row["baseline_ms"] for row in valid)
    total_kernel_ms = sum(row["kernel_ms"] for row in valid)
    large = [row for row in valid if row["axes"]["batch_size"] >= 4113]
    if total_kernel_ms > 0:
        speedup = total_baseline_ms / total_kernel_ms
        latency = f"{total_kernel_ms * 1000:.0f}us"
        speedup_text = f"{speedup:.3f}x"
    else:
        latency = "999999999us"
        speedup_text = "0.000x"
    print()
    print("# Aggregate result")
    print(f"| candidate | candidate ({len(valid)}/{len(rows)} PASS) | {latency} | {speedup_text} |")
    if large:
        big = sum(r["baseline_ms"] for r in large) / sum(r["kernel_ms"] for r in large)
        print(f"batch_size >= 4113 speedup: {big:.3f}x ({len(large)} rows)")


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
):
    """Run the correctness gate and timing sweep for this task."""

    from . import baseline as baseline_module

    baseline_rmsnorm = baseline_module.run
    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_RMSNORM_H200_KERNEL", baseline_rmsnorm, "rmsnorm_h200_candidate"
        )

    baseline_kwargs = {"baseline_fn": baseline_rmsnorm}
    if candidate is not baseline_rmsnorm:
        baseline_kwargs = {
            "baseline_fn": baseline_module.run_prepared,
            "baseline_prepare_fn": baseline_module.prepare,
        }

    rows = run_benchmark(
        name="rmsnorm_h200",
        workloads=make_workloads(cfg) if workloads is None else workloads,
        make_inputs=make_inputs,
        candidate_fn=candidate,
        candidate_prepare_fn=candidate_prepare_fn,
        **baseline_kwargs,
        device=device,
        warmup=cfg.warmup,
        iters=cfg.iters,
        trials=cfg.trials,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=REQUIRED_MATCHED_RATIO,
        group_axis="hidden_size",
        correctness_runs=cfg.correctness_runs,
        check_after_timing=cfg.check_after_timing,
        require_repeatable_outputs=cfg.require_repeatable_outputs,
    )
    _print_aggregate(rows)
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
