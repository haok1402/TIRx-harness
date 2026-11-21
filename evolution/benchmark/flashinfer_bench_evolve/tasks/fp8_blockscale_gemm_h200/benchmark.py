#!/usr/bin/env python3
"""Standalone benchmark for H200 FP8 block-scaled GEMM candidates (DeepGEMM sm90 baseline)."""

from __future__ import annotations

import math

import torch

from ...benchmark_common import (
    BenchConfig,
    choose_device,
    compare_tensors,
    config_from_env,
    load_candidate_override,
    load_task_workloads,
    prepare_with_perf_remarks,
    run_benchmark,
)

TASK_NAME = "fp8_blockscale_gemm_h200"
GRAN = 128  # 1x128 activation scales, 128x128 weight scales
FP8_MAX = 448.0
# The per-element bound replaces atol/rtol (see reference()).
ATOL = 0.0
RTOL = 0.0


def default_config() -> BenchConfig:
    """This task's configuration, read from the environment at call time."""

    return config_from_env(warmup=10, iters=50, trials=3)


def make_workloads(config: BenchConfig | None = None):
    cfg = config or default_config()
    official = []
    if cfg.include_official:
        for row in load_task_workloads(
            TASK_NAME,
            cfg.max_official,
            shape_mode=cfg.shape_mode,
            shape_axes=("M", "N", "K"),
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


def sfa_stride(m: int) -> int:
    """Row stride of the MN-major A-scale storage: M rounded up to 4 (16-byte TMA stride)."""
    return -(-m // 4) * 4


def make_inputs(entry, device):
    """DeepGEMM's FP8 recipe (fp32 scales, no UE8M0), from bf16 N(0, 1) operands.

    Returns [A (e4m3 [M, K]), SFA (fp32 [K/128, align(M, 4)], MN-major), B (e4m3 [N, K]),
    SFB (fp32 [ceil(N/128), K/128])].
    """
    m, n, k = (entry["axes"][key] for key in ("M", "N", "K"))
    if k % GRAN:
        raise ValueError("K must be a multiple of 128")
    torch.manual_seed(0)
    kb, nb = k // GRAN, -(-n // GRAN)
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    sfa = a.view(m, kb, GRAN).abs().float().amax(-1).clamp(1e-4) / FP8_MAX
    a8 = (a.view(m, kb, GRAN) / sfa.unsqueeze(-1)).to(torch.float8_e4m3fn).view(m, k)
    bp = torch.zeros(nb * GRAN, k, device=device, dtype=torch.bfloat16)
    bp[:n] = b
    sfb = bp.view(nb, GRAN, kb, GRAN).abs().float().amax(dim=(1, 3)).clamp(1e-4) / FP8_MAX
    b8 = (bp.view(nb, GRAN, kb, GRAN) / sfb[:, None, :, None]).to(torch.float8_e4m3fn)
    b8 = b8.view(nb * GRAN, k)[:n].contiguous()
    sfa_mn = torch.zeros(kb, sfa_stride(m), device=device, dtype=torch.float32)
    sfa_mn[:, :m] = sfa.t()
    return [a8.contiguous(), sfa_mn, b8, sfb.contiguous()]


def _dequantized(a8, sfa_mn, b8, sfb, *, absolute=False):
    m, k = a8.shape
    n = b8.shape[0]
    kb = k // GRAN
    f64 = torch.float64
    a = a8.to(f64).abs() if absolute else a8.to(f64)
    b = b8.to(f64).abs() if absolute else b8.to(f64)
    sa = sfa_mn[:, :m].t().to(f64)
    sb = sfb.to(f64).repeat_interleave(GRAN, dim=0)[:n]
    if absolute:
        sa, sb = sa.abs(), sb.abs()
    a = (a.view(m, kb, GRAN) * sa.unsqueeze(-1)).view(m, k)
    b = (b.view(n, kb, GRAN) * sb.unsqueeze(-1)).view(n, k)
    return a, b


@torch.no_grad()
def reference(a8, sfa_mn, b8, sfb):
    """The oracle: the exact product (float64), rounded to bf16, and a per-element bound.

    The bound is one bf16 ulp of the exact value plus the Hopper fp8 tensor-core accumulation error
    inside a 128-wide k-block, which scales with the operand magnitudes: 2^-7 |exact| + 2^-12
    (|A| @ |B|^T). DeepGEMM's own outputs sit about 2^-15 of the second term away.
    """
    a, b = _dequantized(a8, sfa_mn, b8, sfb)
    exact = a @ b.t()
    a_abs, b_abs = _dequantized(a8, sfa_mn, b8, sfb, absolute=True)
    bound = 2.0**-7 * exact.abs() + 2.0**-12 * (a_abs @ b_abs.t())
    return exact.to(torch.bfloat16), bound.float()


def compare(
    candidate,
    reference,
    atol=ATOL,
    rtol=RTOL,
    required_matched_ratio=None,
    required_rms_error_ratios=None,
):
    """The oracle returns (expected, per-element bound), outside timing."""
    expected, bound = reference
    return compare_tensors(
        candidate,
        expected,
        atol,
        rtol,
        required_matched_ratio,
        required_rms_error_ratios,
        absolute_bounds=(bound,),
    )


def tirx_prepare(solution_module, a8, sfa_mn, b8, sfb):
    """Compile and bind a TIRx candidate outside timing."""

    m, k = a8.shape
    n = b8.shape[0]
    data = {
        "A": a8,
        "SFA": sfa_mn,
        "B": b8,
        "SFB": sfb,
        "D": torch.empty((m, n), dtype=torch.bfloat16, device=a8.device),
    }
    kernel_fn = prepare_with_perf_remarks(solution_module.setup, data, m, n, k)
    return kernel_fn, data


def tirx_run(kernel_fn, data):
    kernel_fn()
    return data["D"]


def _print_aggregate(rows):
    valid = [
        row
        for row in rows
        if row["passed"] and row["baseline_ms"] is not None and row["kernel_ms"] is not None
    ]
    if valid:
        speedup = math.exp(
            sum(math.log(r["baseline_ms"] / r["kernel_ms"]) for r in valid) / len(valid)
        )
        latency = f"{sum(r['kernel_ms'] for r in valid) * 1000:.0f}us"
        speedup_text = f"{speedup:.3f}x"
    else:
        latency = "999999999us"
        speedup_text = "0.000x"
    print()
    print("# Aggregate result (speedup = geomean over shapes)")
    print(f"| candidate | candidate ({len(valid)}/{len(rows)} PASS) | {latency} | {speedup_text} |")


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
            "BENCH_FP8_BLOCKSCALE_GEMM_H200_KERNEL", baseline_module.run, "fp8_gemm_h200_candidate"
        )

    rows = run_benchmark(
        name=TASK_NAME,
        workloads=make_workloads(cfg) if workloads is None else workloads,
        make_inputs=make_inputs,
        candidate_fn=candidate,
        candidate_prepare_fn=candidate_prepare_fn,
        baseline_fn=baseline_module.run_prepared,
        baseline_prepare_fn=baseline_module.prepare,
        device=device,
        warmup=cfg.warmup,
        iters=cfg.iters,
        trials=cfg.trials,
        atol=ATOL,
        rtol=RTOL,
        required_matched_ratio=None,
        # DeepGEMM is the timing baseline; correctness uses the exact oracle and its bound.
        reference_fn=reference,
        compare_fn=compare,
        group_axis="N",
        correctness_runs=cfg.correctness_runs,
        check_after_timing=cfg.check_after_timing,
        require_repeatable_outputs=cfg.require_repeatable_outputs,
    )
    for row in rows:
        row["correctness_policy"] = "exact_fp64_bf16ulp_plus_fp8_accumulation_bound"
    _print_aggregate(rows)
    return rows


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    run_suite()


if __name__ == "__main__":
    main()
