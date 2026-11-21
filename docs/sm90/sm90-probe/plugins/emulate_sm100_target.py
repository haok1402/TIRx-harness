"""Pytest plugin: make an H200 session select tests and compile kernels as a B200 (sm_100a) would.

- tvm_tirx.testing.env reports compute capability 10.0 and arch sm_100a, so every collection gate,
  skipif and env.runnable_cuda_arch() resolves as on a B200;
- Target("cuda") / {"kind": "cuda"} without an arch resolve to sm_100a.
Compilation and lowering then follow the B200 path. Anything that *launches* a kernel fails to
load on the H200 (CUDA_ERROR_NO_BINARY_FOR_GPU); read only compile/lowering outcomes."""
import tvm_tirx.target as _tmod
from tvm_tirx.testing import env

_Orig = _tmod.Target
_orig_init = _Orig.__init__


def _init(self, target, host=None):
    if target == "cuda":
        target = {"kind": "cuda", "arch": "sm_100a"}
    elif isinstance(target, dict) and target.get("kind") == "cuda" and "arch" not in target:
        target = {**target, "arch": "sm_100a"}
    _orig_init(self, target, host)


def _has_cuda_compute(major, minor=0, exact=False):
    return (10, 0) == (major, minor) if exact else (10, 0) >= (major, minor)


def pytest_configure(config):
    _Orig.__init__ = _init
    env.has_cuda_compute = _has_cuda_compute
    env.cuda_arch = lambda device_id=0: "sm_100a"


def pytest_report_header(config):
    return ["emulate_sm100_target: env reports compute 10.0 / sm_100a; Target('cuda') -> sm_100a"]
