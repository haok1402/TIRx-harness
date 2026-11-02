"""Pytest plugin: drop the blanket 'requires compute capability 10.0' skip that
tvm-tirx's codegen/tile conftests add on non-SM100 devices, so per-test
`has_cuda_compute(9)` gates decide instead (SM90 evidence runs)."""
import pytest

REASON = "requires a CUDA compute capability 10.0 device"


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
    for item in items:
        item.own_markers[:] = [
            m for m in item.own_markers
            if not (m.name == "skip" and m.kwargs.get("reason") == REASON)
        ]
