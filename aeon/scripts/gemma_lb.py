import asyncio
import json
import logging
import random
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import httpx
import uvicorn

# Configuration
NODE0_URL = "http://127.0.0.1:8014"
NODE1_URL = "http://127.0.0.1:8015"
CONTEXT_LIMIT_NODE1 = 89984  # 88k
# Rough estimate: 4 characters per token
CHAR_PER_TOKEN = 4

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gemma-lb")

app = FastAPI()
client = httpx.AsyncClient(timeout=None)

# Track active requests for load balancing
active_requests = {
    NODE0_URL: 0,
    NODE1_URL: 0
}

def estimate_tokens(request_body: dict) -> int:
    """Roughly estimate tokens in the request based on character count."""
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
    # 1. Determine the target node
    target_url = NODE0_URL # Default
    
    body_bytes = await request.body()
    
    if request.method == "POST":
        try:
            if body_bytes:
                body = json.loads(body_bytes)
                tokens = estimate_tokens(body)
                
                if tokens > CONTEXT_LIMIT_NODE1:
                    target_url = NODE0_URL
                    logger.info(f"Routing large request ({tokens} tokens) -> Node 0")
                else:
                    # Load balancing for small requests: pick the node with fewer active requests
                    if active_requests[NODE1_URL] < active_requests[NODE0_URL]:
                        target_url = NODE1_URL
                    elif active_requests[NODE1_URL] == active_requests[NODE0_URL]:
                        target_url = random.choice([NODE0_URL, NODE1_URL])
                    else:
                        target_url = NODE0_URL
                    logger.info(f"Routing request ({tokens} tokens) -> {target_url} (Active: N0={active_requests[NODE0_URL]}, N1={active_requests[NODE1_URL]})")
        except Exception as e:
            logger.warning(f"Could not parse body for token estimation: {e}. Defaulting to Node 0.")
            target_url = NODE0_URL
    else:
        # Non-POST requests (like /health) go to Node 0 or are balanced
        target_url = NODE0_URL if random.random() > 0.5 else NODE1_URL

    # 2. Proxy the request with streaming support
    url = f"{target_url}/{path}"
    
    # Strip problematic hop-by-hop headers from incoming request
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
        # Use send() with stream=True instead of request()
        response = await client.send(req, stream=True)
        
        async def stream_generator():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                active_requests[target_url] -= 1

        # Strip hop-by-hop headers from the response to prevent chunking conflicts
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
        logger.error(f"Proxy error connecting to {target_url}: {e}")
        return Response(content=f"Backend Error: {e}", status_code=502)

if __name__ == "__main__":
    # Run on port 8013 as specified in the cluster config
    uvicorn.run(app, host="0.0.0.0", port=8013, log_level="info")
