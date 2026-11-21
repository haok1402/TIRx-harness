#!/usr/bin/env bash
# Build DeepGEMM from source into the active environment, for hosts where the pinned wheel cannot load.
#
# The `benchmark` group pins a DeepGEMM 559d79f wheel built against glibc 2.38 and Torch 2.14. Hosts with an
# older glibc (Ubuntu 22.04: 2.35, e.g. the H200 nodes) fail `import deep_gemm` with a GLIBC_2.38 symbol error;
# a source build of the same commit against the pinned Torch works. Typical use, on a GPU-less compute node:
#
#   uv sync --locked --group benchmark --no-install-package deep-gemm
#   scripts/build_deepgemm.sh            # builds into the uv environment
#
# Env: DG_COMMIT (default 559d79f), DG_SRC (checkout dir, default .cache/deepgemm-$DG_COMMIT), CUDA_HOME.
# Needs git, a C++20 compiler and the CUDA toolkit (nvcc). Re-running reuses the checkout.
set -euo pipefail

DG_COMMIT=${DG_COMMIT:-559d79f}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DG_SRC=${DG_SRC:-$ROOT/.cache/deepgemm-$DG_COMMIT}
PY=${PYTHON:-$ROOT/.venv/bin/python}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

if [ ! -d "$DG_SRC/.git" ]; then
  git clone --quiet https://github.com/deepseek-ai/DeepGEMM.git "$DG_SRC"
fi
git -C "$DG_SRC" fetch --quiet origin "$DG_COMMIT" 2>/dev/null || true
git -C "$DG_SRC" checkout --quiet --force "$DG_COMMIT"
git -C "$DG_SRC" submodule update --quiet --init --recursive

# Torch >= 2.15 headers need C++20; 559d79f's setup.py asks for C++17.
sed -i "s/'-std=c++17'/'-std=c++20'/" "$DG_SRC/setup.py"

# Build-time tools (not runtime dependencies, so not in uv.lock); the build must see the pinned Torch.
uv pip install --python "$PY" --quiet setuptools wheel ninja
(cd "$DG_SRC" && uv pip install --python "$PY" --no-build-isolation --no-deps --reinstall .)
"$PY" - <<'PY'
import deep_gemm, torch
print(f"deep_gemm {getattr(deep_gemm, '__version__', '?')} OK (torch {torch.__version__})")
PY
