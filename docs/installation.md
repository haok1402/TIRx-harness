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

### Hosts with glibc older than 2.38 (e.g. H200 nodes on Ubuntu 22.04)

The pinned DeepGEMM wheel needs glibc 2.38, so `import deep_gemm` fails on these hosts. Skip it, then build the
same commit from source into the environment:

```bash
uv sync --locked --group benchmark --no-install-package deep-gemm
scripts/build_deepgemm.sh
```

The build needs git, a C++20 compiler and the CUDA toolkit; it reuses its checkout under `.cache/`. A plain
`uv sync --locked --group benchmark` reinstalls the wheel, so keep `--no-install-package deep-gemm` on these
hosts.

Installing the harness project itself also builds its Rust frontend (NumSim), which needs `cargo`. A benchmark or
kcoral server environment does not use it: on hosts without a Rust toolchain, add `--no-install-project` to the
`uv sync` above.

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
