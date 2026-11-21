"""Pytest plugin: print which tvm_tirx source tree and C++ library dir this session imports, so
before/after logs prove the worktree (not the editable install) was tested."""
import os


def pytest_report_header(config):
    import tvm_tirx

    return [f"tvm_tirx: {tvm_tirx.__file__}",
            f"TVM_TIRX_LIBRARY_PATH={os.environ.get('TVM_TIRX_LIBRARY_PATH', '')} "
            f"SKBUILD_EDITABLE_SKIP={os.environ.get('SKBUILD_EDITABLE_SKIP', '')}"]
