#!/usr/bin/env python3
"""DeepGEMM baseline (its public fp8_gemm_nt API, sm90 1D2D kernel) for the H200 FP8 GEMM task.

The A scales arrive already in DeepGEMM's MN-major, TMA-aligned layout (``SFA`` storage
[K/128, align(M, 4)]). ``prepare`` only takes the [M, K/128] strided view DeepGEMM expects, so
neither side times a transposition.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def prepare(a, sfa_mn, b, sfb):
    m = a.shape[0]
    sfa = sfa_mn[:, :m].t()  # [M, K/128] view, stride (1, align(M, 4)): DeepGEMM's required layout
    output = torch.empty((m, b.shape[0]), dtype=torch.bfloat16, device=a.device)
    return a, sfa, b, sfb, output


@torch.no_grad()
def run_prepared(a, sfa, b, sfb, output):
    import deep_gemm

    deep_gemm.fp8_gemm_nt((a, sfa), (b, sfb), output)
    return output


@torch.no_grad()
def run(a, sfa_mn, b, sfb):
    """Run the complete public task contract."""

    return run_prepared(*prepare(a, sfa_mn, b, sfb))
