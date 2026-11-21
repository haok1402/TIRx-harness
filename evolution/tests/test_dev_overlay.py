"""Development overlay: local checkouts' Python copied over the pinned packages, wheel libs kept."""

from __future__ import annotations

import json

import pytest

from evolution.preparation import dev_overlay


def _site(tmp_path):
    site = tmp_path / "site"
    (site / "tvm_tirx" / "lib").mkdir(parents=True)
    (site / "tvm_tirx" / "lib" / "libtvm_runtime.so").write_text("wheel binary")
    (site / "tvm_tirx" / "__init__.py").write_text("RELEASE = True\n")
    (site / "tvm_tirx" / "wheel_only.py").write_text("x = 1\n")
    (site / "tirx_kernels").mkdir()
    (site / "tirx_kernels" / "__init__.py").write_text("RELEASE = True\n")
    (site / "tirx_kernels" / "old_kernel.py").write_text("x = 1\n")
    return site


def _checkouts(tmp_path):
    tvm = tmp_path / "tvm-tirx"
    (tvm / "python" / "tvm_tirx" / "backend").mkdir(parents=True)
    (tvm / "python" / "tvm_tirx" / "__init__.py").write_text("DEV = True\n")
    (tvm / "python" / "tvm_tirx" / "backend" / "wgmma.py").write_text("NEW = 1\n")
    (tvm / "python" / "tvm_tirx" / "lib").mkdir()
    (tvm / "python" / "tvm_tirx" / "lib" / "libtvm_runtime.so").write_text("checkout build")
    kernels = tmp_path / "TIRx-kernels"
    (kernels / "tirx_kernels" / "gemm").mkdir(parents=True)
    (kernels / "tirx_kernels" / "__init__.py").write_text("DEV = True\n")
    (kernels / "tirx_kernels" / "gemm" / "fp16_bf16_gemm_sm90.py").write_text("B1 = 1\n")
    return tvm, kernels


def test_overlay_copies_python_and_keeps_wheel_libraries(tmp_path):
    site = _site(tmp_path)
    tvm, kernels = _checkouts(tmp_path)
    record = dev_overlay.overlay(site, tvm, kernels)
    assert (site / "tvm_tirx" / "__init__.py").read_text() == "DEV = True\n"
    assert (site / "tvm_tirx" / "backend" / "wgmma.py").is_file()
    # the wheel's compiled library is kept, never replaced by a checkout's build tree
    assert (site / "tvm_tirx" / "lib" / "libtvm_runtime.so").read_text() == "wheel binary"
    # tirx_kernels is replaced as a whole: no stale release modules
    assert (site / "tirx_kernels" / "gemm" / "fp16_bf16_gemm_sm90.py").is_file()
    assert not (site / "tirx_kernels" / "old_kernel.py").exists()
    marker = json.loads((site / dev_overlay.MARKER).read_text())
    assert set(marker) == set(record) == {"tvm_tirx", "tirx_kernels"}


def test_environment_variables_select_the_overlay(tmp_path, monkeypatch):
    site = _site(tmp_path)
    _, kernels = _checkouts(tmp_path)
    monkeypatch.delenv(dev_overlay.ENV_TVM_TIRX, raising=False)
    monkeypatch.delenv(dev_overlay.ENV_TIRX_KERNELS, raising=False)
    assert dev_overlay.from_environment(site) == {}
    assert (site / "tirx_kernels" / "old_kernel.py").exists()
    monkeypatch.setenv(dev_overlay.ENV_TIRX_KERNELS, str(kernels))
    record = dev_overlay.from_environment(site)
    assert list(record) == ["tirx_kernels"]
    assert (site / "tvm_tirx" / "__init__.py").read_text() == "RELEASE = True\n"


def test_bad_checkout_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no python/tvm_tirx package"):
        dev_overlay.overlay(_site(tmp_path), tvm_tirx=tmp_path / "missing")
