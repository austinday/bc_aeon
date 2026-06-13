import asyncio
import json
import logging
import random
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import httpx
import uvicorn

# Configuration - vLLM endpoints (OpenAI compatible)
NODE0_URL = "http://127.0.0.1:8016"  # 256k context GPU0
NODE1_URL = "http://127.0.0.1:8017"  # ~88k context GPU1
CONTEXT_LIMIT_NODE1 = 89984
CHAR_PER_TOKEN = 4

import logging
from logging.handlers import RotatingFileHandler

# Setup logging to both console and file
logger = logging.getLogger("vllm-lb")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler - write to /app/vllm_lb.log inside container
try:
    fh = RotatingFileHandler("/app/vllm_lb.log", maxBytes=10*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
except Exception as e:
    print(f"Failed to setup file logger: {e}")

app = FastAPI()
client = httpx.AsyncClient(timeout=None)

active_requests = {
    NODE0_URL: 0,
    NODE1_URL: 0
}

def estimate_tokens(request_body: dict) -> int:
    total_chars = 0
    messages = request_body.get("messages", [])
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
    return total_chars // CHAR_PER_TOKEN

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    target_url = NODE0_URL

    body_bytes = await request.body()

    if request.method == "POST":
        try:
            if body_bytes:
                body = json.loads(body_bytes)
                tokens = estimate_tokens(body)

                if tokens > CONTEXT_LIMIT_NODE1:
                    target_url = NODE0_URL
                    logger.info(f"Routing large request ({tokens} tokens) -> Node 0 (256k)")
                else:
                    if active_requests[NODE1_URL] < active_requests[NODE0_URL]:
                        target_url = NODE1_URL
                    elif active_requests[NODE1_URL] == active_requests[NODE0_URL]:
                        target_url = random.choice([NODE0_URL, NODE1_URL])
                    else:
                        target_url = NODE0_URL
                    logger.info(f"Routing request ({tokens} tokens) -> {target_url} (Active: N0={active_requests[NODE0_URL]}, N1={active_requests[NODE1_URL]})")
        except Exception as e:
            logger.warning(f"Could not parse body: {e}. Defaulting to Node 0.")
            target_url = NODE0_URL
    else:
        target_url = NODE0_URL if random.random() > 0.5 else NODE1_URL

    url = f"{target_url}/{path}"

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-encoding", None)

    req = client.build_request(
        request.method,
        url,
        content=body_bytes,
        headers=headers,
        params=request.query_params
    )

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

        resp_headers = {}
        for k, v in response.headers.items():
            if k.lower() not in ("content-encoding", "content-length", "transfer-encoding", "connection"):
                resp_headers[k] = v

        return StreamingResponse(
            stream_generator(),
            status_code=response.status_code,
            headers=resp_headers
        )
    except Exception as e:
        active_requests[target_url] -= 1
        logger.error(f"Proxy error to {target_url}: {e}")
        return Response(content=f"Backend Error: {e}", status_code=502)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8018, log_level="info")