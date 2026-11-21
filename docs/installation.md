# Installation

```{container} lead
Install the components for your own agent workflow or a kcoral GPU server.
```

For [optimization runs](optimization-runs.md), setup prepares the packages and
skills automatically.

To run an exported kernel with `tvm-ffi`, follow
[Export a kernel](development/export-kernel.md) for the consumer dependencies.

## Before you start

- Linux x86_64, Python 3.12 or 3.13, and pip 25.1 or later.
- The CUDA Toolkit (`nvcc`, `ptxas`) where GPU kernels are compiled, and a
  compatible NVIDIA driver where they are run.
- Cargo, **Rust 1.95.0 or later**, and a C linker for numerical simulation and
  correctness checks with either installation method.

## Install Python packages

Activate your agent's Python environment and choose one method.

### Install from PyPI

Install the released package without cloning this repository:

```bash
python -m pip install tirx-harness
```

Wheels include the native frontend. If pip builds from a source distribution,
the build tools below are required.

### Build from source

Source builds require Git, C/C++ build tools, Python development headers, and
the Rust toolchain listed above.

```bash
git clone https://github.com/mlc-ai/TIRx-harness.git
cd TIRx-harness
python -m pip install .
```

#### Optional: install with uv

From the initialized checkout, build the harness with dependencies from `uv.lock`:

```bash
uv sync --locked
source .venv/bin/activate
```

### Verify the installation

After any of these methods, check imports (this does not run checks or GPU kernels):

```bash
python -c "import tvm_tirx, tvm_ffi, tirx_kernels.tirx_lite, tirx_harness; print('Core imports OK')"
```

## Install kcoral server dependencies

From the repository root, run `python -m pip install --group server`, or
`uv sync --locked --only-group server` and activate `.venv`. These install only
the server dependencies. See {ref}`remote execution <select-remote-execution>`
to start the server.

## Development mode: local tvm-tirx and tirx-kernels checkouts

Until releases contain work under development, a run venv can take the Python sources of local
checkouts instead of the pinned `apache-tvm-tirx` and `tirx-kernels`. Set the variables before
`evolve init`:

```bash
export TIRX_HARNESS_DEV_TVM_TIRX=/path/to/tvm-tirx        # its python/tvm_tirx is copied over the wheel
export TIRX_HARNESS_DEV_TIRX_KERNELS=/path/to/TIRx-kernels
uv run --package tirx-evolution evolve init --task fp16_gemm_h200 --remote @$HOME/kcoral.url
```

For the kcoral server's venv, apply the same overlay directly:

```bash
python -m evolution.preparation.dev_overlay --venv .venv \
  --tvm-tirx /path/to/tvm-tirx --tirx-kernels /path/to/TIRx-kernels
```

How the overlay works:

- The sources are copied into site-packages, not linked, so a task's banned paths still apply to them.
- The installed wheel's compiled libraries (`tvm_tirx/lib`) are kept, so only Python-level tvm-tirx
  changes are picked up. C++ changes need a wheel built from the checkout.
- `manifest.json` records each overlay's source path, commit, branch and dirty state, and so does
  `site-packages/tirx_harness_dev_overlay.json`.

## Install agent skills

| Skill | Purpose |
| --- | --- |
| {repo}`tirx-wiki <skills/tirx-wiki/SKILL.md>` | Find TIRx-lite APIs, canonical kernels, and GPU references. |
| {repo}`tirx-debug-kernel <skills/tirx-debug-kernel/SKILL.md>` | Check correctness, investigate findings, and verify fixes. |
| {repo}`tirx-profile-kernel <skills/tirx-profile-kernel/SKILL.md>` | Measure performance and use profiler evidence to guide changes. |

The package bundles the skills. Install them into the directory your agent
reads, such as `.agents/skills` or `.claude/skills`:

```bash
tirx-harness skills install --dest /absolute/path/to/your/project/.agents/skills
```

The command also downloads the wiki manuals and reference repositories,
including `tirx-kernels`. Use `--no-fetch` to skip the download and `--force`
to replace an earlier installation. Editable installs, such as `uv sync`, do
not bundle the skills; copy them from `skills/` in the checkout instead.

Continue to [Quick Start](quick-start.md) for a concrete example.
