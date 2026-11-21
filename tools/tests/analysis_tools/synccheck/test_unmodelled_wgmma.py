"""WGMMA fails soft: checkers run with it skipped and report it; NumSim refuses numerics.

The kernel is a minimal warpgroup-MMA protocol (operand fence, wgmma.fence, commit, wait, fence)
around a register accumulator, plus an ordinary barrier that the checks must still see.
"""

from __future__ import annotations

import numpy as np
import pytest

from tirx_harness.numsim import Engine, UnmodeledTIRxFormError, transpile
from tirx_harness.numsim.checkers import _run_racecheck as racecheck
from tirx_harness.numsim.checkers import _run_synccheck as synccheck
from tvm_tirx.script import tirx as T


def _kernel(bar_count: int):
    @T.function
    def wgmma_protocol(output: T.Tensor((128,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tx = T.cuda.thread_idx("x")
        acc = T.alloc_local([4], "float32")
        for i in T.unroll(4):
            T.cuda.wgmma.noop_barrier(acc[i])
        T.ptx.wgmma.fence.sync.aligned()
        T.ptx.wgmma.commit_group.sync.aligned()
        T.ptx.wgmma.wait_group.sync.aligned(0)
        for i in T.unroll(4):
            T.cuda.wgmma.noop_barrier(acc[i])
        T.ptx.bar.sync(T.uint32(1), T.uint32(bar_count))
        output[tx] = acc[0] + acc[1] + acc[2] + acc[3]

    return wgmma_protocol


def _inputs():
    return {"output": np.zeros((128,), np.float32)}


def _kinds(report):
    return [finding.kind for finding in report.findings]


@pytest.mark.parametrize("checker", [synccheck, racecheck], ids=["synccheck", "racecheck"])
def test_wgmma_kernel_is_checked_and_reported_as_unmodelled(checker):
    report = checker(_kernel(128), inputs=_inputs())
    assert report.verdict == "review", report.to_dict()
    assert _kinds(report) == ["unmodelled_operations"]
    (advisory,) = report.findings
    message = str(getattr(advisory, "message", advisory))
    assert "tirx.ptx.wgmma_fence" in message and "not modelled" in message
    # The skipped ops' register operands are defined: no spurious uninitialized reads.
    assert "uninitialized_read" not in _kinds(report)


def test_a_barrier_bug_next_to_wgmma_is_still_caught():
    report = synccheck(_kernel(160), inputs=_inputs())  # 160 participants, 128 threads arrive
    assert report.verdict == "error", report.to_dict()
    assert "unmodelled_operations" in _kinds(report)
    assert any(kind != "unmodelled_operations" for kind in _kinds(report))


def test_numsim_refuses_numerics_for_unmodelled_ops():
    module = transpile(_kernel(128))
    assert module.spec.unmodelled
    with pytest.raises(UnmodeledTIRxFormError, match="not modelled"):
        Engine().run(module, _inputs())
