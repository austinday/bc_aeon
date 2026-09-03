"""Generic least-busy load balancer for dual-copy model deployments.

Generalizes the old hardcoded gemma_lb.py: the backend node URLs and the listen
port come from the environment, so the same router fronts any Tier-A (dual-copy)
model. Both copies are symmetric (same context), so routing is pure
least-active-requests with per-node health tracking and graceful failover.

Env:
  AEON_LB_NODES   comma-separated backend base URLs (e.g. "http://127.0.0.1:8014,http://127.0.0.1:8015")
  AEON_LB_PORT    port to listen on (default 8013)
"""
import asyncio
import os
import time
import logging
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response


def _validated_loopback_node(value: str) -> str:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("AEON_LB_NODES contains an invalid port") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or port is None
        or not 1024 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("AEON_LB_NODES must contain exact loopback HTTP origins")
    canonical = (
        f"http://[::1]:{port}"
        if parsed.hostname == "::1"
        else f"http://127.0.0.1:{port}"
    )
    if value not in {canonical, canonical + "/"}:
        raise RuntimeError("AEON_LB_NODES contains a non-canonical origin")
    return canonical


NODES = [
    _validated_loopback_node(value.strip())
    for value in os.environ.get("AEON_LB_NODES", "").split(",")
    if value.strip()
]
LB_PORT = int(os.environ.get("AEON_LB_PORT", "8013"))
HEALTH_INTERVAL = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("adaptive-lb")

app = FastAPI()
client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0),
    http2=False,
    trust_env=False,
    follow_redirects=False,
)

active = {u: 0 for u in NODES}
healthy = {u: True for u in NODES}


def pick_node() -> str:
    up = [u for u in NODES if healthy[u]]
    pool = up or NODES
    return min(pool, key=lambda u: active[u])


async def health_loop():
    while True:
        await asyncio.sleep(HEALTH_INTERVAL)
        for u in NODES:
            try:
                r = await client.get(f"{u}/health", timeout=5)
                healthy[u] = r.status_code == 200
            except Exception:
                healthy[u] = False


@app.on_event("startup")
async def _startup():
    logger.info(f"adaptive-lb listening on :{LB_PORT}, nodes={NODES}")
    asyncio.create_task(health_loop())


@app.get("/health")
async def health():
    return Response(status_code=200 if any(healthy.values()) else 503)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    body = await request.body()
    target = pick_node()
    url = f"{target}/{path}"
    headers = dict(request.headers)
    for h in ("host", "content-length", "content-encoding"):
        headers.pop(h, None)

    req = client.build_request(request.method, url, content=body, headers=headers, params=request.query_params)
    active[target] += 1
    try:
        resp = await client.send(req, stream=True)

        async def gen():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                active[target] -= 1

        out_headers = {k: v for k, v in resp.headers.items()
                       if k.lower() not in ("content-encoding", "content-length", "transfer-encoding", "connection")}
        return StreamingResponse(gen(), status_code=resp.status_code, headers=out_headers)
    except Exception as e:
        active[target] -= 1
        healthy[target] = False
        logger.error(f"proxy error to {target}: {e}")
        return Response(content=f"Backend Error: {e}", status_code=502)


if __name__ == "__main__":
    if not NODES:
        raise SystemExit("AEON_LB_NODES is empty")
    uvicorn.run(app, host="0.0.0.0", port=LB_PORT, log_level="info")
