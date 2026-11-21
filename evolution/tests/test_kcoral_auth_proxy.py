"""The token proxy in front of a loopback kcoral server: no token, no request reaches kcoral."""

from __future__ import annotations

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")
from evolution.remote import kcoral_auth_proxy as proxy  # noqa: E402

TOKEN = "t" * 40


def _run(app, method, path, headers=None, body=b""):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            return await client.request(method, path, headers=headers or {}, content=body)

    return asyncio.run(go())


def _upstream(calls):
    def handler(request):
        calls.append(request)
        payload = json.dumps({"path": request.url.path, "body": request.content.decode()})
        return httpx.Response(200, stream=httpx.ByteStream(payload.encode()))

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}, {"X-Token": TOKEN}],
)
def test_requests_without_the_token_never_reach_kcoral(headers):
    calls = []
    app = proxy.make_app(TOKEN, "http://127.0.0.1:1", transport=_upstream(calls))
    for method, path in (("GET", "/health"), ("POST", "/execute")):
        response = _run(app, method, path, headers, b"payload")
        assert response.status_code == 401
    assert calls == []


def test_authorized_requests_are_forwarded_without_the_token():
    calls = []
    app = proxy.make_app(TOKEN, "http://127.0.0.1:1", transport=_upstream(calls))
    response = _run(
        app, "POST", "/execute?x=1", {"Authorization": f"Bearer {TOKEN}"}, b"program-bytes"
    )
    assert response.status_code == 200
    assert response.json() == {"path": "/execute", "body": "program-bytes"}
    (request,) = calls
    assert "authorization" not in request.headers
    assert request.url.query == b"x=1"


def test_short_tokens_and_non_loopback_upstreams_are_refused(monkeypatch):
    with pytest.raises(ValueError, match="at least 32"):
        proxy.make_app("short", "http://127.0.0.1:1")
    monkeypatch.setenv(proxy.TOKEN_ENV, TOKEN)
    assert proxy.main(["--listen", "0.0.0.0:1", "--upstream", "10.0.0.5:2"]) == 2
    monkeypatch.delenv(proxy.TOKEN_ENV)
    assert proxy.main(["--listen", "0.0.0.0:1", "--upstream", "127.0.0.1:2"]) == 2
