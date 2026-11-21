"""The GPU an optimization run targets: detected from the benchmark server (or the local GPU), or given.

A task declares the architectures it supports (``target_archs`` in its YAML). ``evolve init`` resolves the
run's :class:`GpuTarget`, refuses tasks that do not support it, and renders the target into the task
spec and the prompts. Spec slots: ``{{gpu}}`` ("B200 (sm_100)", "H200 (sm_90)": the arch without its
``a`` suffix, as the specs always wrote it, so B200 prompts are byte-identical to the literal ones),
``{{gpu_name}}`` and ``{{gpu_arch}}`` (its two halves, for a spec that wraps between them), ``{{arch}}``
(exact, e.g. "sm_90a"), ``{{gpu_family}}`` and ``{{num_sms}}``.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

# Architectures the harness knows how to describe. The name and SM count are only the defaults used when
# detection cannot see the GPU; a detected device always wins.
KNOWN_ARCHS = {
    "sm_90a": ("Hopper", "H200", 132),
    "sm_100a": ("Blackwell", "B200", 148),
    "sm_103a": ("Blackwell", "B300", 148),
}
# A GPU may run tasks written for another arch: B300 (sm_103a) runs every sm_100a task, as before
# target_archs existed (kcoral reports sm_103a there).
COMPATIBLE_ARCHS = {"sm_103a": ("sm_100a",)}


class UnknownArchError(ValueError):
    """The run's GPU was detected, but the harness does not know its architecture."""


@dataclass(frozen=True)
class GpuTarget:
    arch: str  # e.g. "sm_90a", as kcoral's Client.target() reports it
    name: str  # short marketing name, e.g. "H200"
    num_sms: int
    source: str = "flag"  # "remote" (kcoral server), "local" (nvidia-smi), "flag" or "default"

    def __post_init__(self):
        if self.arch not in KNOWN_ARCHS:
            raise UnknownArchError(
                f"unknown GPU architecture {self.arch!r}; known: {sorted(KNOWN_ARCHS)}"
            )

    @property
    def family(self) -> str:
        return KNOWN_ARCHS[self.arch][0]

    @property
    def display_arch(self) -> str:
        return self.arch.removesuffix("a")

    @property
    def label(self) -> str:
        return f"{self.name} ({self.display_arch})"

    def runs(self, task_archs) -> bool:
        """Whether this GPU runs a task written for ``task_archs``."""
        return any(arch in task_archs for arch in (self.arch, *COMPATIBLE_ARCHS.get(self.arch, ())))

    def slots(self) -> dict[str, str]:
        return {
            "gpu": self.label,
            "gpu_name": self.name,
            "gpu_arch": self.display_arch,
            "arch": self.arch,
            "gpu_family": self.family,
            "num_sms": str(self.num_sms),
        }

    def as_dict(self) -> dict:
        return {
            "arch": self.arch,
            "name": self.name,
            "num_sms": self.num_sms,
            "source": self.source,
        }


def from_flags(arch: str, name: str | None = None, num_sms: int | None = None) -> GpuTarget:
    if arch not in KNOWN_ARCHS:
        raise ValueError(f"--target-arch must be one of {sorted(KNOWN_ARCHS)}, got {arch!r}")
    _, default_name, default_sms = KNOWN_ARCHS[arch]
    return GpuTarget(arch, name or default_name, num_sms or default_sms, source="flag")


def default_for(task_archs: list[str]) -> GpuTarget:
    """The task's first architecture, used when the GPU cannot be detected (no flag given)."""
    arch = task_archs[0]
    _, name, sms = KNOWN_ARCHS[arch]
    return GpuTarget(arch, name, sms, source="default")


def _from_properties(info: dict, source: str) -> GpuTarget:
    major, minor = int(info["major"]), int(info["minor"])
    arch = info.get("arch") or (f"sm_{major}{minor}a" if major >= 9 else f"sm_{major}{minor}")
    name = str(info["name"]).removeprefix("NVIDIA ").strip()
    sms = info.get("sms") or KNOWN_ARCHS.get(arch, (None, None, 0))[2]
    return GpuTarget(arch, name, int(sms), source=source)


def detect_local() -> GpuTarget:
    """The first visible local GPU, from nvidia-smi (the evolve CLI's environment has no torch).

    nvidia-smi does not report an SM count; the architecture's default is used (override: --num-sms).
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    name, cap = (part.strip() for part in out.strip().splitlines()[0].split(","))
    major, minor = cap.split(".")
    return _from_properties({"name": name, "major": major, "minor": minor}, "local")


def _remote_probe_source() -> str:
    return (
        "def main():\n"
        "    import torch\n"
        "    p = torch.cuda.get_device_properties(torch.cuda.current_device())\n"
        "    return {'name': p.name, 'major': p.major, 'minor': p.minor, "
        "'sms': p.multi_processor_count}\n"
    )


def detect_remote(url: str, timeout_seconds: int = 120) -> GpuTarget:
    """The kcoral server's GPU: its reported compilation target plus the device's name and SM count.

    ``url`` may be ``@URL_FILE`` (scripts/slurm_kcoral.sh); the current URL and its bearer token are
    read from the file and ``URL_FILE.token`` through the shared ``kcoral_connection`` helper.
    """
    from kcoral import Program

    from evolution.remote.kcoral_connection import client, resolve_url

    remote = url
    url = resolve_url(remote)
    if not url:
        raise RuntimeError(f"no server URL published in {remote[1:]}")
    program = Program()
    module = program.upload(id="probe", kind="module", source=_remote_probe_source())
    fn = program.get_function(id="main", module=module, name="main")
    program.return_(key="device", value=program.run(id="device", fn=fn, args=[]))
    with client(url, remote) as connection:
        arch = connection.target()["arch"]
        result = connection.execute(program, timeout_seconds=timeout_seconds)
    if not result.completed:
        raise RuntimeError(f"device probe on {url} failed: {result.error}")
    return _from_properties({**result.results["device"], "arch": arch}, "remote")


def resolve(
    *,
    task_archs: list[str],
    remote_url: str | None,
    arch: str | None = None,
    name: str | None = None,
    num_sms: int | None = None,
) -> GpuTarget:
    """Flags first, then the benchmark server or local GPU; if neither is reachable, the task's first arch.

    A GPU that is reachable but has an architecture the harness does not know is an error
    (UnknownArchError), never silently treated as the task's default.
    """
    if arch is not None:
        return from_flags(arch, name, num_sms)
    try:
        return detect_remote(remote_url) if remote_url is not None else detect_local()
    except UnknownArchError:
        raise
    except (
        Exception
    ) as exc:  # no GPU or server at init time: keep the task's own (historical) target
        target = default_for(task_archs)
        where = remote_url or "a local GPU"
        print(
            f"warning: could not detect the target GPU from {where} ({type(exc).__name__}: {exc}); "
            f"assuming {target.label}. Pass --target-arch to set it explicitly.",
            file=sys.stderr,
        )
        return target
