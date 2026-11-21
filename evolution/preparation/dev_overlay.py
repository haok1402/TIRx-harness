"""Development mode: run the harness against local tvm-tirx / tirx-kernels checkouts instead of the pins.

Until releases contain the work under development (e.g. the SM90 helpers and kernels), a run venv or a
kcoral server venv can take the Python sources of local checkouts:

* ``tvm-tirx``: the checkout's ``python/tvm_tirx`` tree is copied over the installed package. The
  installed wheel's compiled libraries (``tvm_tirx/lib``) are kept, so only Python-level changes are
  picked up. C++ changes need a wheel built from the checkout.
* ``tirx-kernels``: the checkout's ``tirx_kernels`` package replaces the installed one.

The sources are *copied* into site-packages (not linked or put on ``sys.path``), so a task's banned
paths still apply to them: ``Environment.sanitize`` removes banned kernel files from the copy.

``evolve init`` applies it when ``TIRX_HARNESS_DEV_TVM_TIRX`` / ``TIRX_HARNESS_DEV_TIRX_KERNELS`` name
the checkouts, and records it in ``manifest.json``. For another venv (a kcoral server's)::

    python -m evolution.preparation.dev_overlay --venv .venv --tvm-tirx ~/tvm-tirx --tirx-kernels ~/TIRx-kernels
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ENV_TVM_TIRX = "TIRX_HARNESS_DEV_TVM_TIRX"
ENV_TIRX_KERNELS = "TIRX_HARNESS_DEV_TIRX_KERNELS"
MARKER = "tirx_harness_dev_overlay.json"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "lib", "*.so")


def _git(path: Path) -> dict:
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def overlay(site: Path, tvm_tirx: Path | None = None, tirx_kernels: Path | None = None) -> dict:
    """Copy the given checkouts' Python packages into ``site`` (a site-packages dir); return the record."""
    site = Path(site)
    record = {}
    if tvm_tirx is not None:
        src = Path(tvm_tirx).expanduser().resolve() / "python" / "tvm_tirx"
        dst = site / "tvm_tirx"
        if not (src / "__init__.py").is_file():
            raise ValueError(f"{ENV_TVM_TIRX}: no python/tvm_tirx package under {tvm_tirx}")
        if not dst.is_dir():
            raise ValueError(f"tvm_tirx is not installed in {site}; sync the venv first")
        shutil.copytree(src, dst, ignore=_IGNORE, dirs_exist_ok=True)
        record["tvm_tirx"] = {"source": str(src.parent.parent), **_git(src)}
    if tirx_kernels is not None:
        root = Path(tirx_kernels).expanduser().resolve()
        src = root / "tirx_kernels"
        dst = site / "tirx_kernels"
        if not (src / "__init__.py").is_file():
            raise ValueError(f"{ENV_TIRX_KERNELS}: no tirx_kernels package under {tirx_kernels}")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        record["tirx_kernels"] = {"source": str(root), **_git(root)}
    if record:
        (site / MARKER).write_text(json.dumps(record, indent=2) + "\n")
    return record


def from_environment(site: Path) -> dict:
    """Apply the overlay the ``TIRX_HARNESS_DEV_*`` variables ask for (none: no change)."""
    tvm = os.environ.get(ENV_TVM_TIRX) or None
    kernels = os.environ.get(ENV_TIRX_KERNELS) or None
    if tvm is None and kernels is None:
        return {}
    record = overlay(site, tvm, kernels)
    print(f"dev overlay: {json.dumps(record)}", file=sys.stderr)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--venv", type=Path, required=True, help="the venv to overlay")
    parser.add_argument("--tvm-tirx", type=Path, help="tvm-tirx checkout (its python/ is copied)")
    parser.add_argument("--tirx-kernels", type=Path, help="TIRx-kernels checkout")
    args = parser.parse_args(argv)
    if args.tvm_tirx is None and args.tirx_kernels is None:
        parser.error("give --tvm-tirx and/or --tirx-kernels")
    python = args.venv / "bin" / "python"
    site = Path(
        subprocess.run(
            [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    record = overlay(site, args.tvm_tirx, args.tirx_kernels)
    # Import check: the overlaid Python must load the installed compiled libraries.
    subprocess.run(
        [str(python), "-I", "-c", "import tvm_tirx, tirx_kernels; print('dev overlay imports ok')"],
        check=True,
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
