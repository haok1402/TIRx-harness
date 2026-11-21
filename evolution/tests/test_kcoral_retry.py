"""kcoral_remote: a preempted or restarting benchmark server is retried; candidate failures are not."""

from __future__ import annotations

import types

import pytest

kcoral = pytest.importorskip("kcoral")
from kcoral.errors import KCoralError, TransportError  # noqa: E402

from evolution.remote import kcoral_connection as kc  # noqa: E402
from evolution.remote import kcoral_remote as kr  # noqa: E402


class _Result:
    def __init__(self, completed=True, message=None):
        self.completed = completed
        self.status = "COMPLETED" if completed else "FAILED"
        self.error = None if completed else {"kind": "runtime", "message": message}
        self.stdout = "out\n"
        self.stderr = ""
        self.results = {"rows": [1]}


seen_headers = []


def _fake_client(script, seen):
    """A Client whose execute() plays ``script`` (exceptions or results), recording the URLs used."""

    class Client:
        def __init__(self, url, headers=None):
            self.url = url
            self.headers = headers
            seen_headers.append(headers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, program, **kw):
            seen.append(self.url)
            step = script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step

        def health(self):
            if script and isinstance(script[0], str) and script[0] == "health-down":
                script.pop(0)
                raise TransportError("down")
            return {"status": "ok"}

    return Client


@pytest.fixture(autouse=True)
def short_gone_poll(monkeypatch):
    monkeypatch.setattr(kc, "GONE_POLL_SECONDS", 0)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(kc.time, "sleep", lambda s: None)


def _install(monkeypatch, script, seen):
    monkeypatch.setitem(
        __import__("sys").modules,
        "kcoral",
        types.SimpleNamespace(Client=_fake_client(script, seen)),
    )


def test_url_file_is_reread_and_transport_errors_retried(tmp_path, monkeypatch, no_sleep):
    url_file = tmp_path / "kcoral.url"
    url_file.write_text("http://node-a:1000\n")
    seen = []
    script = [TransportError("connection refused"), TransportError("reset"), _Result()]

    def move_server(*a):  # after the first failure the server republishes on another node
        url_file.write_text("http://node-b:2000\n")

    monkeypatch.setattr(kc.time, "sleep", move_server)
    _install(monkeypatch, script, seen)
    result = kr._execute(f"@{url_file}", object(), 60)
    assert result.results == {"rows": [1]}
    assert seen == ["http://node-a:1000", "http://node-b:2000", "http://node-b:2000"]


def test_missing_url_file_waits_for_a_server(tmp_path, monkeypatch):
    url_file = tmp_path / "kcoral.url"
    seen = []

    def publish(*a):
        url_file.write_text("http://node-c:3000")

    monkeypatch.setattr(kc.time, "sleep", publish)
    _install(monkeypatch, [_Result()], seen)
    kr._execute(f"@{url_file}", object(), 60)
    assert seen == ["http://node-c:3000"]


def test_busy_server_is_retried_but_timeout_is_not(monkeypatch, no_sleep):
    seen = []
    _install(monkeypatch, [KCoralError(503, "busy"), _Result()], seen)
    kr._execute("http://n:1", object(), 60)
    assert len(seen) == 2
    _install(monkeypatch, [KCoralError(504, "execution timed out")], seen)
    with pytest.raises(KCoralError):
        kr._execute("http://n:1", object(), 60)


def test_worker_killed_by_preemption_is_resent(monkeypatch, no_sleep):
    seen = []
    script = [
        _Result(False, "worker exited while executing the instruction"),
        "health-down",
        _Result(),
    ]
    _install(monkeypatch, script, seen)
    assert kr._execute("http://n:1", object(), 60).completed
    assert len(seen) == 2


def test_candidate_crash_with_healthy_server_is_reported(monkeypatch, no_sleep):
    seen = []
    _install(monkeypatch, [_Result(False, "worker exited while executing the instruction")], seen)
    with pytest.raises(SystemExit, match="FAILED"):
        kr._execute("http://n:1", object(), 60)
    assert len(seen) == 1


def test_retry_budget_is_bounded(monkeypatch, no_sleep):
    monkeypatch.setenv(kr.RETRY_SECONDS_ENV, "0")
    _install(monkeypatch, [TransportError("refused")] * 3, [])
    with pytest.raises(SystemExit, match="unavailable for 0s after 1 attempts"):
        kr._execute("http://n:1", object(), 60)


def test_token_is_read_next_to_the_url_file_and_sent(tmp_path, monkeypatch, no_sleep):
    url_file = tmp_path / "kcoral.url"
    url_file.write_text("http://node-a:1000\n")
    (tmp_path / "kcoral.url.token").write_text("s3cr3t-token\n")
    seen_headers.clear()
    _install(monkeypatch, [_Result()], [])
    kr._execute(f"@{url_file}", object(), 60)
    assert seen_headers[-1] == {"Authorization": "Bearer s3cr3t-token"}


def test_rejected_token_is_reported_not_retried(monkeypatch, no_sleep):
    seen = []
    _install(monkeypatch, [KCoralError(401, "unauthorized")], seen)
    with pytest.raises(SystemExit, match="rejected the token"):
        kr._execute("http://n:1", object(), 60)
    assert len(seen) == 1


def test_worker_crash_500_is_retried_only_when_the_server_went_away(monkeypatch, no_sleep):
    seen = []
    _install(monkeypatch, [KCoralError(500, "worker crashed"), "health-down", _Result()], seen)
    assert kr._execute("http://n:1", object(), 60).completed
    _install(monkeypatch, [KCoralError(500, "worker crashed")], seen)
    with pytest.raises(KCoralError):
        kr._execute("http://n:1", object(), 60)


def test_slow_shutdown_is_still_seen_as_gone(monkeypatch, no_sleep):
    """Healthy at the first poll (SIGTERM not yet acted on), gone at a later one: retried."""
    monkeypatch.setattr(kc, "GONE_POLL_SECONDS", 3600)
    clock = iter(range(0, 10**6, 10))
    monkeypatch.setattr(kc.time, "monotonic", lambda: next(clock))
    seen = []
    script = [_Result(False, "worker exited while executing the instruction"), "ok", "health-down"]
    script.append(_Result())
    real = _fake_client(script, seen)

    class Client(real):
        def health(self):
            if script and script[0] == "ok":
                script.pop(0)
                return {"status": "ok"}
            return super().health()

    monkeypatch.setitem(__import__("sys").modules, "kcoral", types.SimpleNamespace(Client=Client))
    assert kr._execute("http://n:1", object(), 60).completed
    assert len(seen) == 2
