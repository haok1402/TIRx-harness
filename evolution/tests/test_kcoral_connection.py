"""Every local kcoral client authenticates the same way (kcoral_connection), tools and health included."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

kcoral = pytest.importorskip("kcoral")
from kcoral.errors import TransportError  # noqa: E402

from evolution.remote import kcoral_compute_sanitizer, kcoral_iket, kcoral_ncu, kcoral_python  # noqa: E402
from evolution.remote import kcoral_remote as kr  # noqa: E402
from evolution.remote import kcoral_tool as tool  # noqa: E402

REMOTE_DIR = Path(kr.__file__).parent


@pytest.fixture
def url_file(tmp_path):
    path = tmp_path / "kcoral.url"
    path.write_text("http://node-a:1000\n")
    path.with_name("kcoral.url.token").write_text("s3cret\n")
    return path


@pytest.fixture
def fake_kcoral(monkeypatch):
    calls = []

    def install(*, health=None, result=None):
        class Client:
            def __init__(self, url, headers=None):
                calls.append((url, headers))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def health(self):
                if isinstance(health, Exception):
                    raise health
                return health

            def execute(self, program, **kw):
                return result

        monkeypatch.setitem(sys.modules, "kcoral", SimpleNamespace(Client=Client))
        return calls

    return install


def _result():
    return SimpleNamespace(
        completed=True,
        stdout="",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
        results={"outcome": {"returncode": 0}},
    )


def test_tool_clis_accept_url_file_and_send_its_token(url_file, fake_kcoral):
    calls = fake_kcoral(result=_result())
    outcome = tool.execute(SimpleNamespace(remote=f"@{url_file}", timeout=30), program=None)
    assert outcome == {"returncode": 0}
    assert calls == [("http://node-a:1000", {"Authorization": "Bearer s3cret"})]


@pytest.mark.parametrize(
    "cli, options, forwarded",
    [
        (kcoral_python, [], ["s.py"]),
        (kcoral_ncu, ["-o", "report"], ["python", "s.py"]),
        (kcoral_iket, ["--output-dir", "out"], ["profile", "--", "python", "s.py"]),
        (kcoral_compute_sanitizer, [], ["--tool", "memcheck", "python", "s.py"]),
    ],
)
def test_every_tool_cli_takes_at_url_file(cli, options, forwarded, url_file):
    parsed = cli.parse_args(["--remote", f"@{url_file}", *options, "--", *forwarded])
    assert parsed[0].remote == f"@{url_file}"


def test_plain_url_takes_token_from_env(tmp_path, monkeypatch, fake_kcoral):
    token = tmp_path / "token"
    token.write_text("t0k\n")
    monkeypatch.setenv(kr.TOKEN_FILE_ENV, str(token))
    calls = fake_kcoral(result=_result())
    tool.execute(SimpleNamespace(remote="http://n:1", timeout=30), program=None)
    assert calls == [("http://n:1", {"Authorization": "Bearer t0k"})]


def test_health_is_authenticated(url_file, fake_kcoral, capsys):
    calls = fake_kcoral(health={"status": "ok", "gpus": [{"gpu_id": 0}]})
    assert kr.print_health(f"@{url_file}") == 0
    assert calls == [("http://node-a:1000", {"Authorization": "Bearer s3cret"})]
    assert '"status": "ok"' in capsys.readouterr().out


@pytest.mark.parametrize("health, code", [({"status": "starting"}, 1), (TransportError("down"), 1)])
def test_health_failures_exit_nonzero(url_file, fake_kcoral, health, code):
    fake_kcoral(health=health)
    assert kr.print_health(f"@{url_file}") == code


def test_health_without_published_url(tmp_path, fake_kcoral, capsys):
    fake_kcoral(health={"status": "ok"})
    assert kr.print_health(f"@{tmp_path / 'missing.url'}") == 1
    assert "no server URL published" in capsys.readouterr().err


def test_only_kcoral_connection_builds_clients():
    """One shared helper: no other local client module may construct a kcoral Client itself."""
    offenders = [
        path.name
        for path in REMOTE_DIR.glob("kcoral_*.py")
        if path.name not in {"kcoral_connection.py", "kcoral_auth_proxy.py"}
        and re.search(r"\bClient\(", path.read_text())
    ]
    assert offenders == []
