#!/usr/bin/env python3
"""FA3 baseline (FlashInfer ``backend="fa3"``, batch ragged prefill) for the H200 attention task.

``prepare`` allocates the workspace and plans outside timing. The timed call is ``wrapper.run``.
"""

from __future__ import annotations

import torch

HEAD_DIM = 128


@torch.no_grad()
def prepare(q, k, v, cu_seqlens, causal):
    import flashinfer

    workspace = torch.empty(512 << 20, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend="fa3")
    heads = q.shape[1]
    wrapper.plan(
        cu_seqlens,
        cu_seqlens,
        heads,
        heads,
        HEAD_DIM,
        causal=bool(causal),
        q_data_type=torch.bfloat16,
    )
    return wrapper, q, k, v, workspace


@torch.no_grad()
def run_prepared(wrapper, q, k, v, workspace):
    return wrapper.run(q, k, v)


@torch.no_grad()
def run(q, k, v, cu_seqlens, causal):
    """Run the complete public task contract."""

    return run_prepared(*prepare(q, k, v, cu_seqlens, causal))
