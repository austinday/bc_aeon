import asyncio
import json
import logging
import random
import subprocess
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import httpx
import uvicorn

# Configuration
NODE0_URL = "http://127.0.0.1:8014"
NODE1_URL = "http://127.0.0.1:8015"
CONTEXT_LIMIT_NODE1 = 89984
CHAR_PER_TOKEN = 4
HEALTH_CHECK_INTERVAL = 30  # seconds
IDLE_SHUTDOWN_TIMEOUT = 300  # 5 minutes of no requests -> shutdown nodes
START_SCRIPT_NODE0 = "/home/aday/NexusAgentDashboard/bc_aeon/scripts/debug/start_gemma_node0.sh"
START_SCRIPT_NODE1 = "/home/aday/NexusAgentDashboard/bc_aeon/scripts/debug/start_gemma_node1.sh"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gemma-lb")

app = FastAPI()
client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0),
    http2=False,
    trust_env=False,
    follow_redirects=False,
)

active_requests = {NODE0_URL: 0, NODE1_URL: 0}
last_request_time = time.time()
node_health = {NODE0_URL: True, NODE1_URL: True}
restart_locks = {NODE0_URL: asyncio.Lock(), NODE1_URL: asyncio.Lock()}

def estimate_tokens(request_body: dict) -> int:
    total_chars = 0
    for msg in request_body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
    return total_chars // CHAR_PER_TOKEN

async def health_check_loop():
    global last_request_time, node_health
    while True:
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        for url in [NODE0_URL, NODE1_URL]:
            try:
                resp = await client.get(f"{url}/health", timeout=5)
                node_health[url] = resp.status_code == 200
            except Exception:
                node_health[url] = False
                logger.warning(f"Node {url} unhealthy - attempting restart")
                script = START_SCRIPT_NODE0 if url == NODE0_URL else START_SCRIPT_NODE1
                try:
                    subprocess.Popen(["bash", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    await asyncio.sleep(30)  # give time to boot
                except Exception as e:
                    logger.error(f"Restart failed for {url}: {e}")

        # Idle shutdown
        if time.time() - last_request_time > IDLE_SHUTDOWN_TIMEOUT:
            for url in [NODE0_URL, NODE1_URL]:
                if node_health[url]:
                    logger.info(f"Idle timeout - shutting down {url}")
                    # In real setup this would call docker stop or pkill on the node process
                    node_health[url] = False

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(health_check_loop())

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    global last_request_time
    last_request_time = time.time()

    target_url = NODE0_URL
    body_bytes = await request.body()

    if request.method == "POST" and body_bytes:
        try:
            body = json.loads(body_bytes)
            tokens = estimate_tokens(body)
            if tokens > CONTEXT_LIMIT_NODE1:
                target_url = NODE0_URL
            else:
                healthy_nodes = [u for u, h in node_health.items() if h]
                if not healthy_nodes:
                    target_url = NODE0_URL
                elif len(healthy_nodes) == 1:
                    target_url = healthy_nodes[0]
                else:
                    target_url = NODE1_URL if active_requests[NODE1_URL] < active_requests[NODE0_URL] else NODE0_URL
        except Exception:
            target_url = NODE0_URL

    url = f"{target_url}/{path}"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-encoding", None)

    req = client.build_request(request.method, url, content=body_bytes, headers=headers, params=request.query_params)
    active_requests[target_url] += 1
    try:
        response = await client.send(req, stream=True)
        async def stream_generator():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                active_requests[target_url] -= 1
        resp_headers = {k: v for k, v in response.headers.items() if k.lower() not in ("content-encoding", "content-length", "transfer-encoding", "connection")}
        return StreamingResponse(stream_generator(), status_code=response.status_code, headers=resp_headers)
    except Exception as e:
        active_requests[target_url] -= 1
        node_health[target_url] = False
        logger.error(f"Proxy error to {target_url}: {e}")
        # Instant per-node self-heal for crashed GPU
        script = START_SCRIPT_NODE0 if target_url == NODE0_URL else START_SCRIPT_NODE1
        try:
            subprocess.Popen(["bash", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Instant restart triggered for crashed node {target_url}")
        except Exception as ex:
            logger.error(f"Instant restart failed for {target_url}: {ex}")
        return Response(content=f"Backend Error: {e}", status_code=502)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8013, log_level="info")
