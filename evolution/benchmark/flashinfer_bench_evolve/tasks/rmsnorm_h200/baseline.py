#!/usr/bin/env python3
"""FlashInfer baseline for the H200 RMSNorm task."""

from __future__ import annotations

import torch

EPS = 1e-6


@torch.no_grad()
def prepare(x, w):
    """Allocate the output once so timing covers only the kernel."""

    return x, w, torch.empty_like(x)


@torch.no_grad()
def run_prepared(x, w, out):
    import flashinfer

    flashinfer.norm.rmsnorm(x, w, EPS, out=out, enable_pdl=False)
    return out


@torch.no_grad()
def run(x, w):
    """Run the complete public task contract."""

    return run_prepared(*prepare(x, w))
