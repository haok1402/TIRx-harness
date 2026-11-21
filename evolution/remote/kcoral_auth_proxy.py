#!/usr/bin/env python3
"""Bearer-token reverse proxy in front of a loopback-only kcoral server.

kcoral executes arbitrary code for any client and has no client authentication: never expose
`kcoral server` itself on a network interface. scripts/slurm_kcoral.sh runs the server on 127.0.0.1
and this proxy on the node's network address; the proxy forwards only requests that carry

    Authorization: Bearer <token>

(exactly one such header) with the per-launch token from ``$KCORAL_PROXY_TOKEN`` (never on the
command line, where every user could read it from ``ps``), and only origin-form targets
("/path"). Every other request gets 401 (or 400) without reaching kcoral. Bodies are
streamed in both directions and responses have no read timeout (a scoring request can run for an
hour). The Authorization header is not forwarded.

    KCORAL_PROXY_TOKEN=... python kcoral_auth_proxy.py \
        --listen 0.0.0.0:PORT --upstream 127.0.0.1:PORT2
"""

from __future__ import annotations

import argparse
import hmac
import os
import sys

TOKEN_ENV = "KCORAL_PROXY_TOKEN"
_HOP = {
    b"host",
    b"authorization",
    b"connection",
    b"keep-alive",
    b"transfer-encoding",
    b"te",
    b"upgrade",
    b"proxy-authorization",
    b"content-length",
}


def make_app(token: str, upstream: str, *, transport=None):
    """The proxy ASGI app. ``transport`` replaces the network transport to the upstream (tests)."""
    import httpx

    if len(token) < 32:
        raise ValueError("the proxy token must be at least 32 characters")
    expected = f"Bearer {token}".encode()
    client = httpx.AsyncClient(
        base_url=upstream, timeout=httpx.Timeout(None, connect=10.0), transport=transport
    )

    async def reject(send, status, message):
        body = message.encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"text/plain"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await client.aclose()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        auth = [v for k, v in scope["headers"] if k.lower() == b"authorization"]
        # Exactly one Authorization header, equal to the token (constant-time compare).
        if len(auth) != 1 or not hmac.compare_digest(auth[0], expected):
            await reject(send, 401, "unauthorized")
            return
        target = scope.get("raw_path") or scope["path"].encode()
        # Origin-form only ("/path"): an absolute-form ("http://host/...") or scheme-relative
        # ("//host/...") target would make the upstream client leave the loopback kcoral.
        if not target.startswith(b"/") or target.startswith(b"//"):
            await reject(send, 400, "only origin-form request targets are accepted")
            return

        async def body():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if chunk:
                    yield chunk
                if not message.get("more_body", False):
                    return

        forward = [(k, v) for k, v in scope["headers"] if k.lower() not in _HOP]
        path = target
        if scope.get("query_string"):
            path += b"?" + scope["query_string"]
        request = client.build_request(
            scope["method"], path.decode("latin-1"), headers=forward, content=body()
        )
        try:
            response = await client.send(request, stream=True)
        except httpx.TransportError as exc:
            await reject(send, 502, f"upstream unavailable: {exc}")
            return
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": response.status_code,
                    "headers": [
                        (k.encode(), v.encode())
                        for k, v in response.headers.multi_items()
                        if k.lower().encode() not in _HOP
                    ],
                }
            )
            async for chunk in response.aiter_raw():
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b""})
        finally:
            await response.aclose()

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--listen", required=True, help="HOST:PORT to accept clients on")
    parser.add_argument("--upstream", required=True, help="HOST:PORT of the loopback kcoral server")
    args = parser.parse_args(argv)
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"{TOKEN_ENV} is not set", file=sys.stderr)
        return 2
    upstream_host = args.upstream.rsplit(":", 1)[0]
    if upstream_host not in ("127.0.0.1", "localhost", "::1"):
        print("--upstream must be a loopback address", file=sys.stderr)
        return 2
    import uvicorn

    host, port = args.listen.rsplit(":", 1)
    app = make_app(token, f"http://{args.upstream}")
    uvicorn.run(app, host=host, port=int(port), log_level="warning", lifespan="on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
