import asyncio
import httpx
import time
import subprocess
import re

# Configuration
BASE_URL = "http://localhost:8001"
TEST_URL = "https://www.google.com"
SESSIONS_TO_TEST = 5
TABS_PER_SESSION = 3

async def get_container_mem():
    """Gets the current memory usage of the browser container in MB."""
    try:
        # Target the specific measurement container by name
        cmd = "docker stats --no-stream --format '{{.MemUsage}}' aeon_browser_measure"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        # Output format is usually "123.4MiB / 221GiB"
        match = re.search(r'([0-9.]+)([a-zA-Z]+)', output)
        if match:
            val, unit = match.groups()
            val = float(val)
            if unit == 'GiB': val *= 1024
            if unit == 'KiB': val /= 1024
            return val
    except Exception as e:
        print(f"Error getting memory: {e}")
    return 0

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("--- Browser Resource Measurement ---")
        
        # 1. Baseline
        print("Measuring baseline memory...")
        await asyncio.sleep(5) # Wait for container to settle
        baseline = await get_container_mem()
        print(f"Baseline RAM: {baseline:.2f} MB")

        # 2. Session Cost
        session_mems = []
        for i in range(SESSIONS_TO_TEST):
            sid = f"session_{i}"
            print(f"Creating session {sid}...")
            await client.post(f"{BASE_URL}/navigate", json={"url": TEST_URL, "session_id": sid})
            await asyncio.sleep(2)
            mem = await get_container_mem()
            session_mems.append(mem)
        
        avg_session_cost = (session_mems[-1] - baseline) / SESSIONS_TO_TEST if session_mems else 0
        print(f"Average RAM cost per session (1 tab): {avg_session_cost:.2f} MB")

        # 3. Tab Cost
        tab_mems = []
        # Use the last session created
        sid = f"session_{SESSIONS_TO_TEST-1}"
        for j in range(1, TABS_PER_SESSION + 1):
            tid = f"tab_{j}"
            print(f"Creating tab {tid} in {sid}...")
            await client.post(f"{BASE_URL}/navigate", json={"url": TEST_URL, "session_id": sid, "tab_id": tid})
            await asyncio.sleep(2)
            mem = await get_container_mem()
            tab_mems.append(mem)
            
        avg_tab_cost = (tab_mems[-1] - session_mems[-1]) / (TABS_PER_SESSION - 1) if TABS_PER_SESSION > 1 else 0
        print(f"Average RAM cost per additional tab: {avg_tab_cost:.2f} MB")

        print("\n--- Summary ---")
        print(f"Baseline: {baseline:.2f} MB")
        print(f"Per Session (1st tab): {avg_session_cost:.2f} MB")
        print(f"Per Additional Tab: {avg_tab_cost:.2f} MB")

if __name__ == "__main__":
    asyncio.run(main())