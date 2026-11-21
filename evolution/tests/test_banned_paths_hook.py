"""The generated banned-paths hook: path lists are checked entry by entry.

A run lives inside the checkout that prepared it (``<checkout>/kda_flow_runs/<run>/worktree``), and the
checkout itself is banned. So ``$PATH``, which holds the run's own ``.venv/bin``, must not
substring-match that ancestor ban once expanded, while a banned entry in a path list still blocks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from evolution.preparation.guards import BANNED_PATHS_HOOK_NAME, _write_banned_paths_hook


@pytest.fixture
def run(tmp_path):
    checkout = tmp_path / "checkout"
    worktree = checkout / "kda_flow_runs" / "run" / "worktree"
    hooks = worktree / ".claude" / "hooks"
    (worktree / ".venv" / "bin").mkdir(parents=True)
    hooks.mkdir(parents=True)
    (checkout / "secret.py").write_text("")
    _write_banned_paths_hook(
        hooks,
        [f"{checkout}/**", "tirx-kernels/banned/**"],
        allowed_prefixes=[str(worktree)],
    )
    return checkout, worktree, hooks / BANNED_PATHS_HOOK_NAME


def _bash(hook, worktree, command, **env):
    payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(worktree)}
    result = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=worktree,
        env={**os.environ, **env},
        check=False,
    )
    return result.returncode, result.stderr


def test_path_holding_the_runs_own_venv_is_not_banned(run):
    _, worktree, hook = run
    # As in a real run: tool directories first, the activated run venv later.
    path = f"/usr/local/bin:{worktree}/.venv/bin:/usr/bin"
    assert _bash(hook, worktree, "echo $PATH; ls", PATH=path) == (0, "")
    assert _bash(hook, worktree, 'env PYTHONPATH="$PATH" python -V', PATH=path)[0] == 0


@pytest.mark.parametrize(
    "command, env",
    [
        ("cat {checkout}/secret.py", {}),  # the ancestor ban itself
        ("echo $EXTRA", {"EXTRA": "/usr/bin:{checkout}/secret.py"}),  # banned entry in a list
        ("ls /usr/bin:{checkout}", {}),
        ("cat tirx-kernels/banned/x.py", {}),  # a run-relative ban
    ],
)
def test_banned_entries_still_block(run, command, env):
    checkout, worktree, hook = run
    command = command.format(checkout=checkout)
    env = {k: v.format(checkout=checkout) for k, v in env.items()}
    code, stderr = _bash(hook, worktree, command, **env)
    assert code == 2 and "banned path access blocked" in stderr
