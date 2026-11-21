"""The GPU an optimization run targets: detected from the benchmark server (or the local GPU), or given.

A task declares the architectures it supports (``target_archs`` in its YAML). ``evolve init`` resolves the
run's :class:`GpuTarget`, refuses tasks that do not support it, and renders the target into the task
spec (``{{gpu}}``, ``{{arch}}``, ``{{gpu_family}}``, ``{{num_sms}}``) and the prompts.
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


@dataclass(frozen=True)
class GpuTarget:
    arch: str  # e.g. "sm_90a", as kcoral's Client.target() reports it
    name: str  # short marketing name, e.g. "H200"
    num_sms: int
    source: str = "flag"  # "remote" (kcoral server), "local" (nvidia-smi), "flag" or "default"

    def __post_init__(self):
        if self.arch not in KNOWN_ARCHS:
            raise ValueError(
                f"unknown GPU architecture {self.arch!r}; known: {sorted(KNOWN_ARCHS)}"
            )

    @property
    def family(self) -> str:
        return KNOWN_ARCHS[self.arch][0]

    @property
    def label(self) -> str:
        return f"{self.name} ({self.arch})"

    def slots(self) -> dict[str, str]:
        return {
            "gpu": self.label,
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
    """The kcoral server's GPU: its reported compilation target plus the device's name and SM count."""
    from kcoral import Client, Program

    program = Program()
    module = program.upload(id="probe", kind="module", source=_remote_probe_source())
    fn = program.get_function(id="main", module=module, name="main")
    program.return_(key="device", value=program.run(id="device", fn=fn, args=[]))
    with Client(url) as client:
        arch = client.target()["arch"]
        result = client.execute(program, timeout_seconds=timeout_seconds)
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
    """Flags first, then the benchmark server or local GPU; if neither is reachable, the task's first arch."""
    if arch is not None:
        return from_flags(arch, name, num_sms)
    try:
        return detect_remote(remote_url) if remote_url is not None else detect_local()
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
