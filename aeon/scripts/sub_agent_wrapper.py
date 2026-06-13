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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    output_path = Path(args.output_dir) / "output.json"
    status_path = Path(args.output_dir) / "status.txt"
    log_path = Path(args.output_dir) / "agent.log"
    telemetry_path = Path(args.output_dir) / "telemetry.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def update_telemetry(iteration, display_max, step_description):
        try:
            telemetry = {
                "agent_id": args.agent_id,
                "iteration": iteration,
                "current_step": step_description,
                "timestamp": time.time()
            }
            with open(telemetry_path, "w") as f:
                json.dump(telemetry, f, indent=2)
        except Exception as e:
            log(f"Telemetry update failed: {e}")

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
        
        from aeon.main import register_models_for_agent, unregister_models_for_agent
        register_models_for_agent([config.get('model')])

        deps = {'llm_client': llm_client}
        worker = Worker(llm_client=llm_client, debug_mode=args.debug)
        
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
        
        # We need to hook into the worker's loop to update telemetry.
        # Since worker.run is a blocking call, we wrap the iteration logic if possible,
        # or we can use a simple approach: the worker's state is accessible if we 
        # can modify the worker or if we run it in a way that we can poll it.
        # However, the simplest way is to let the worker handle its own telemetry 
        # if we modify the Worker class, but we are modifying the wrapper.
        # As a workaround in the wrapper, we can't easily hook into worker.run() 
        # without modifying aeon/core/worker.py. 
        # Let's check if we can pass a callback to worker.run or if we should 
        # modify the Worker class instead.
        
        # Actually, the most robust way is to modify the Worker class to accept 
        # a telemetry callback. But for now, I will implement a basic 
        # 'start' telemetry entry and then consider modifying the Worker.
        
        # Pass the update_telemetry function as a callback to the worker
        worker.run(
            objective, 
            max_iterations=args.max_iterations, 
            step_callback=update_telemetry
        )
        
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
        
        from aeon.main import unregister_models_for_agent
        unregister_models_for_agent([config.get('model')])

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
            
        from aeon.main import unregister_models_for_agent
        unregister_models_for_agent([config.get('model')])
        sys.exit(1)

if __name__ == "__main__":
    main()
