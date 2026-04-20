import os
import sys
import json
import argparse
import time
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

def main():
    parser = argparse.ArgumentParser(description="Aeon Sub-Agent Wrapper")
    parser.add_argument("--agent_id", required=True, help="Unique ID for the sub-agent")
    parser.add_argument("--objective", required=True, help="The task the sub-agent must complete")
    parser.add_argument("--model_config", required=True, help="JSON string of the model configuration")
    parser.add_argument("--workspace", required=True, help="Path to the read-only workspace")
    parser.add_argument("--output_dir", required=True, help="Path to the read-write output directory")
    parser.add_argument("--max_iterations", type=int, default=20, help="Max iterations for the sub-agent")
    parser.add_argument("--read_only", action="store_true", help="Mount workspace as read-only")
    args = parser.parse_args()

    output_path = Path(args.output_dir) / "output.json"
    status_path = Path(args.output_dir) / "status.txt"
    log_path = Path(args.output_dir) / "agent.log"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{ts}] {message}"
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    try:
        log(f"Initializing sub-agent {args.agent_id}...")
        
        config = json.loads(args.model_config)
        
        llm_client = LLMClient(strong_config=config, weak_config=config)
        
        deps = {'llm_client': llm_client}
        worker = Worker(llm_client=llm_client, debug_mode=True)
        
        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        worker.register_tools(tools)
        
        worker.model_name = config.get('model', 'unknown')

        # Change to workspace directory (read-only)
        os.chdir(args.workspace)
        log(f"Changed working directory to workspace: {args.workspace}")
        
        if args.read_only:
            os.chmod(args.workspace, 0o555)
        log("Workspace set to read-only.")
        log(f"Starting execution of objective: {args.objective}")
        with open(status_path, "w") as f:
            f.write("RUNNING")

        default_instruction = "When you finish, provide a detailed, informative report of your findings, actions taken, and final result. This report will be read by the main agent."
        objective = f"{default_instruction}\n\n{args.objective}"
        worker.run(objective, max_iterations=args.max_iterations)
        
        if args.read_only:
            os.chmod(args.workspace, 0o755)
            
        final_report = {
            "agent_id": args.agent_id,
            "status": "COMPLETED",
            "result": worker.last_observation,
            "plan": worker.current_plan,
            "memories": worker.memories
        }
        
        with open(output_path, "w") as f:
            json.dump(final_report, f, indent=2)
        
        with open(status_path, "w") as f:
            f.write("COMPLETED")
            
        log("Task completed successfully.")

    except Exception as e:
        log(f"CRITICAL ERROR: {str(e)}")
        with open(status_path, "w") as f:
            f.write(f"FAILED: {str(e)}")
        
        error_report = {
            "agent_id": args.agent_id,
            "status": "FAILED",
            "error": str(e)
        }
        with open(output_path, "w") as f:
            json.dump(error_report, f, indent=2)
        sys.exit(1)

if __name__ == "__main__":
    main()