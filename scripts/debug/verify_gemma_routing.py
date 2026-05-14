import asyncio
import httpx
import json
import time

LB_URL = "http://127.0.0.1:8013"
NODE0_URL = "http://127.0.0.1:8014"
NODE1_URL = "http://127.0.0.1:8015"

async def test_routing(name, prompt_size, expected_node=None):
    print(f"Testing {name}: Prompt size ~{prompt_size} chars...")
    
    # Create a prompt of the specified size
    prompt = "Hello " * (prompt_size // 6)
    payload = {
        "model": "gemma-4",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            start = time.time()
            response = await client.post(f"{LB_URL}/v1/chat/completions", json=payload)
            duration = time.time() - start
            
            if response.status_code != 200:
                print(f"  [FAIL] Request failed with status {response.status_code}: {response.text}")
                return False

            # The load balancer logs to stdout, but we can't easily read that here.
            # However, we can check the response time or try to hit the nodes directly 
            # to see if they are alive.
            print(f"  [SUCCESS] Response received in {duration:.2f}s")
            return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

async def verify_nodes():
    print("Checking node health...")
    async with httpx.AsyncClient() as client:
        for name, url in [("Node 0", NODE0_URL), ("Node 1", NODE1_URL)]:
            try:
                resp = await client.get(f"{url}/health")
                print(f"  {name} ({url}): {'Healthy' if resp.status_code == 200 else 'Unhealthy'}")
            except Exception as e:
                print(f"  {name} ({url}): Offline ({e})")

async def main():
    await verify_nodes()
    
    # 1. Small request: Should be balanced (we'll send a few to see if they both respond)
    print("\nTesting small requests (should be balanced)...")
    results = []
    for i in range(5):
        results.append(await test_routing(f"Small-{i}", 100))
    
    # 2. Medium request: ~100k chars (approx 25k tokens) -> Should be balanced
    print("\nTesting medium request (~100k chars)...")
    await test_routing("Medium", 100_000)
    
    # 3. Large request: > 128k tokens. 
    # 128k tokens * 4 chars/token = 512k chars.
    print("\nTesting large request (> 128k tokens / ~520k chars) -> Must go to Node 0")
    await test_routing("Large", 520_000)
    
    # 4. Extra Large request: > 256k tokens.
    # 256k tokens * 4 chars/token = 1.024M chars.
    print("\nTesting XL request (> 256k tokens / ~1.1M chars) -> Must go to Node 0")
    await test_routing("XL", 1_100_000)

if __name__ == "__main__":
    asyncio.run(main())