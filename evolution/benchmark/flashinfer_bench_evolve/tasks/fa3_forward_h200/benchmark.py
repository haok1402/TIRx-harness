#!/usr/bin/env python3
"""Standalone benchmark for H200 attention-forward candidates (FA3 baseline, ragged NHD, varlen)."""

from __future__ import annotations

import hashlib
import itertools
import math

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

TASK_NAME = "fa3_forward_h200"
HEAD_DIM = 128

# bf16 output; FA3 itself sits within ~4e-3 of a float32 reference on these inputs.
ATOL = 2e-2
RTOL = 2e-2
REQUIRED_MATCHED_RATIO = None


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=5, iters=30, trials=3)


def make_workloads(config: BenchConfig | None = None):
    cfg = config or default_config()
    official = []
    if cfg.include_official:
        for row in load_task_workloads(
            TASK_NAME,
            cfg.max_official,
            shape_mode=cfg.shape_mode,
            shape_axes=("total_tokens", "num_heads", "causal"),
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


def _entry_seed(entry):
    raw = f"fa3-forward-h200:{entry['id']}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def cu_seqlens(seq_lens, device):
    offsets = [0]
    for n in seq_lens:
        offsets.append(offsets[-1] + int(n))
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def make_inputs(entry, device):
    """[Q, K, V (bf16 [total_tokens, heads, 128]), cu_seqlens (int32), causal (bool)]."""

    axes = entry["axes"]
    seq_lens = [int(n) for n in axes["seq_lens"]]
    total, heads = sum(seq_lens), int(axes["num_heads"])
    torch.manual_seed(_entry_seed(entry))
    q, k, v = (
        torch.randn(total, heads, HEAD_DIM, device=device, dtype=torch.bfloat16) for _ in range(3)
    )
    return [q, k, v, cu_seqlens(seq_lens, device), bool(axes["causal"])]


def tirx_prepare(solution_module, q, k, v, cu, causal):
    """Compile and bind a TIRx candidate outside timing."""

    offsets = cu.tolist()
    seq_lens = [b - a for a, b in itertools.pairwise(offsets)]
    data = {"Q": q, "K": k, "V": v, "cu_seqlens": cu, "O": torch.empty_like(q)}
    kernel_fn = prepare_with_perf_remarks(
        solution_module.setup, data, seq_lens, q.shape[1], bool(causal)
    )
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["O"]


def _print_aggregate(rows):
    valid = [
        row
        for row in rows
        if row["passed"] and row["baseline_ms"] is not None and row["kernel_ms"] is not None
    ]
    if valid:
        ratios = [r["baseline_ms"] / r["kernel_ms"] for r in valid]
        speedup = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        latency = f"{sum(r['kernel_ms'] for r in valid) * 1000:.0f}us"
        speedup_text = f"{speedup:.3f}x"
        worst = min(zip(ratios, valid), key=lambda t: t[0])
        worst_text = f"minimum per-shape speedup: {worst[0]:.3f}x ({worst[1]['id']})"
    else:
        latency = "999999999us"
        speedup_text = "0.000x"
        worst_text = "minimum per-shape speedup: n/a"
    print()
    print("# Aggregate result (speedup = geomean over shapes)")
    print(f"| candidate | candidate ({len(valid)}/{len(rows)} PASS) | {latency} | {speedup_text} |")
    print(worst_text)


def run_suite(
    config: BenchConfig | None = None,
    candidate_fn=None,
    workloads=None,
    candidate_prepare_fn=None,
):
    """Run the correctness gate and timing sweep for this task."""

    from . import baseline as baseline_module

    cfg = config or default_config()
    device = choose_device(cfg.device)
    candidate = candidate_fn
    if candidate is None:
        candidate = load_candidate_override(
            "BENCH_FA3_FORWARD_H200_KERNEL", baseline_module.run, "fa3_forward_h200_candidate"
        )
    baseline_kwargs = {"baseline_fn": baseline_module.run}
    if candidate is not baseline_module.run:
        baseline_kwargs = {
            "baseline_fn": baseline_module.run_prepared,
            "baseline_prepare_fn": baseline_module.prepare,
        }
    rows = run_benchmark(
        name=TASK_NAME,
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
        group_axis="num_heads",
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
