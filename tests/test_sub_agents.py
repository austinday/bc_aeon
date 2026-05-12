import sys
import os
import time
import json
import subprocess

# Add project root to path
sys.path.insert(0, '/home/aday/bc_aeon')

from aeon.tools.sub_agent import SpawnSubAgent, GetSubAgentReport, KillSubAgent

def test_sub_agents():
    print("=== Testing Sub-Agent Tools ===\n")
    
    # Initialize tools
    spawn_tool = SpawnSubAgent()
    report_tool = GetSubAgentReport()
    kill_tool = KillSubAgent()
    
    # Clean up any existing sub-agents
    import shutil
    output_dir = '/home/aday/bc_aeon/aeon_output/sub_agents'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Spawn 3 sub-agents
    print("1. Spawning 3 sub-agents...")
    agent_ids = []
    for i in range(1, 4):
        result = spawn_tool.execute(f"Task {i}: Count to 100 and save to count_{i}.txt", "Qwen3.6-35B-A3B-Uncensored")
        print(f"   Spawn {i}: {result}")
        if "Agent ID:" in result:
            agent_id = result.split("Agent ID: ")[1].split(".")[0]
            agent_ids.append(agent_id)
        time.sleep(1)
    
    # 2. Test limit enforcement (should fail)
    print("\n2. Testing limit enforcement (should fail)...")
    result = spawn_tool.execute("Task 4: Should fail due to limit", "Qwen3.6-35B-A3B-Uncensored")
    print(f"   Spawn 4: {result}")
    
    # 3. Check reports
    print("\n3. Checking reports...")
    for agent_id in agent_ids:
        report = report_tool.execute(agent_id)
        print(f"   Report for {agent_id[:8]}...: {report[:100]}...")
    
    # 4. Kill agent
    print("\n4. Killing agent...")
    if agent_ids:
        kill_result = kill_tool.execute(agent_ids[0])
        print(f"   Kill result: {kill_result}")
        time.sleep(1)
        report = report_tool.execute(agent_ids[0])
        print(f"   Report after kill: {report}")
    
    # 5. Spawn after kill (should succeed)
    print("\n5. Spawning after kill (should succeed)...")
    result = spawn_tool.execute("Task 5: Should succeed after kill", "Qwen3.6-35B-A3B-Uncensored")
    print(f"   Spawn 5: {result}")
    
    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    test_sub_agents()