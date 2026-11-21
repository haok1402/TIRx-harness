"""The one way every local kcoral client reaches the server: URL, token, retries, health.

``kcoral_remote.py`` (scoring, ``--cmd``, ``--health``) and the dedicated tool CLIs (``kcoral_ncu``,
``kcoral_iket``, ``kcoral_compute_sanitizer``, ``kcoral_python``) all go through ``client()`` and
``request()``, so each accepts ``--remote URL`` or ``--remote @URL_FILE`` and sends the bearer token
that ``scripts/slurm_kcoral.sh`` writes next to the URL file (``kcoral_auth_proxy.py`` checks it).
CPU-only and light: it imports nothing from the benchmark package.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

RETRY_SECONDS_ENV = "KCORAL_RETRY_SECONDS"  # total time to keep retrying an unavailable server
# Gaps between preemptible array tasks here can exceed 30 min: wait hours, not minutes.
DEFAULT_RETRY_SECONDS = 6 * 3600
_RETRY_HTTP = {502, 503}  # bad gateway / busy or shutting down; 504 is the request's own timeout
TOKEN_FILE_ENV = (
    "KCORAL_TOKEN_FILE"  # bearer token for a plain-URL --remote (see kcoral_auth_proxy.py)
)
GONE_POLL_SECONDS = (
    45  # Slurm's KillWait (30 s) plus slack: a SIGTERMed server may answer health a while
)
REMOTE_HELP = (
    "the kcoral server URL, or @FILE holding the current URL (re-read on every attempt, with its "
    "bearer token from FILE.token; scripts/slurm_kcoral.sh publishes both). For a plain URL the "
    f"token comes from ${TOKEN_FILE_ENV}, if set"
)


def resolve_url(remote: str) -> str:
    """``--remote`` is a URL, or ``@FILE`` naming a file that holds the current URL.

    A preemptible server (``scripts/slurm_kcoral.sh``) moves between nodes and ports; it republishes
    its URL in that file, and every attempt re-reads it.
    """
    if not remote.startswith("@"):
        return remote
    try:
        return Path(remote[1:]).expanduser().read_text().strip()
    except FileNotFoundError:
        return ""  # no server published right now (e.g. waiting for the next Slurm task)


def token_for(remote: str) -> str | None:
    """The bearer token: ``URL_FILE.token`` for ``@URL_FILE``, else ``$KCORAL_TOKEN_FILE``.

    scripts/slurm_kcoral.sh writes a fresh token (mode 0600) with every launch; it is re-read on
    every attempt, like the URL.
    """
    path = Path(remote[1:] + ".token") if remote.startswith("@") else None
    if path is None and os.environ.get(TOKEN_FILE_ENV):
        path = Path(os.environ[TOKEN_FILE_ENV])
    if path is None:
        return None
    try:
        return path.expanduser().read_text().strip() or None
    except FileNotFoundError:
        return None


def client(url: str, remote: str):
    """A ``kcoral.Client`` for ``url`` carrying ``remote``'s bearer token, if any."""
    from kcoral import Client

    token = token_for(remote)
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return Client(url, headers=headers)


def health(remote: str) -> dict:
    """The server's ``/health`` document, authenticated like every other request."""
    url = resolve_url(remote)
    if not url:
        raise ConnectionError(f"no server URL published in {remote[1:]}")
    with client(url, remote) as connection:
        return connection.health()


def server_gone(remote: str, url: str) -> bool:
    """After a worker died mid-request: did the server go away (retry) or the candidate crash?

    A preempted step is SIGTERMed and may keep answering health for up to Slurm's KillWait, so
    poll for GONE_POLL_SECONDS: gone as soon as the URL file moves or health fails; a candidate
    crash only if the same server stays healthy the whole time.
    """
    from kcoral.errors import KCoralError, ProtocolError, TransportError

    deadline = time.monotonic() + GONE_POLL_SECONDS
    while True:
        time.sleep(5)
        if resolve_url(remote) != url:
            return True
        try:
            with client(url, remote) as connection:
                connection.health()
        except (TransportError, KCoralError, ProtocolError):
            return True
        if time.monotonic() >= deadline:
            return False


def request(
    remote: str,
    program,
    timeout_seconds: int,
    output_limit_bytes: int | None = None,
    retry_seconds: float | None = None,
):
    """Send ``program`` and return the server's result, re-sending while the server is away.

    Requests are self-contained, so a request that the server could not finish because the
    server went away (connection refused or reset, HTTP 502/503, or its worker killed mid-run by a
    preemption) is re-sent to the server ``remote`` currently resolves to. Retries back off from
    5 s to 60 s, for up to ``retry_seconds`` (``--retry-seconds``, else ``$KCORAL_RETRY_SECONDS``,
    default 6 h) in total. Candidate failures, timeouts and protocol errors are not retried: their
    result (completed or not) is returned for the caller to report.
    """

    from kcoral.errors import KCoralError, TransportError

    if retry_seconds is None:
        retry_seconds = float(os.environ.get(RETRY_SECONDS_ENV, DEFAULT_RETRY_SECONDS))
    budget = float(retry_seconds)
    deadline = time.monotonic() + budget
    delay = 5.0
    attempt = 0
    while True:
        attempt += 1
        url = resolve_url(remote)
        reason = None
        result = None
        if not url:
            reason = f"no server URL published in {remote[1:]}"
        else:
            try:
                with client(url, remote) as connection:
                    result = connection.execute(
                        program,
                        timeout_seconds=timeout_seconds,
                        output_limit_bytes=output_limit_bytes,
                    )
            except TransportError as exc:
                reason = f"{url} unreachable ({exc})"
            except KCoralError as exc:
                crashed = exc.status_code == 500 and "worker crashed" in str(exc.message)
                if exc.status_code == 401:
                    raise SystemExit(
                        f"benchmark server {url} rejected the token (HTTP 401): "
                        f"check {remote}.token or ${TOKEN_FILE_ENV}"
                    ) from exc
                if exc.status_code not in _RETRY_HTTP and not (
                    crashed and server_gone(remote, url)
                ):
                    raise
                reason = f"{url} answered HTTP {exc.status_code} ({exc.message})"
        if result is not None:
            error = getattr(result, "error", None) or {}
            worker_died = "worker exited" in str(error.get("message", ""))
            if result.completed or not worker_died or not server_gone(remote, url):
                break
            reason = f"{url} went away while running the request ({error.get('message')})"
        if time.monotonic() + delay > deadline:
            raise SystemExit(
                f"benchmark server unavailable for {budget:.0f}s after {attempt} attempts: {reason}"
            )
        print(f"benchmark server: {reason}; retrying in {delay:.0f}s", file=sys.stderr, flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 60.0)

    if attempt > 1:
        print(f"benchmark server: succeeded on attempt {attempt} via {url}", file=sys.stderr)
    return result
