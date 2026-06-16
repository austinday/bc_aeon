import os
import sys
import json
import argparse
import time
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory

# Tools a sub-agent is NOT allowed to have. Sub-agents must not be able to spawn
# their own sub-agents (no runaway recursion / GPU oversubscription) and must not
# be able to self-modify or restart the framework.
SUB_AGENT_FORBIDDEN_TOOLS = {
    "spawn_sub_agent",
    "gather_sub_agents",
    "get_sub_agent_report",
    "kill_sub_agent",
    "steer_sub_agent",
    "get_sub_agent_status",
    "verify_self_modification",
    "restart_aeon",
}

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
    steering_path = Path(args.output_dir) / "steering.txt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def update_telemetry(iteration, display_max, step_description):
        # Telemetry checkpoint AND steering injection point. The primary agent's
        # steer_sub_agent tool writes guidance to steering.txt; we read it here at
        # the start of each iteration, fold it into the worker's last_observation so
        # the model actually sees it, then consume (delete) the file so the same
        # guidance is not re-applied every turn.
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

        try:
            if steering_path.exists():
                guidance = steering_path.read_text(encoding="utf-8").strip()
                if guidance:
                    worker.last_observation = (
                        f"[STEERING GUIDANCE FROM PRIMARY AGENT] {guidance}\n\n"
                        f"{worker.last_observation}"
                    )
                    log(f"Applied steering guidance: {guidance[:120]}")
                try:
                    steering_path.unlink()
                except Exception:
                    pass
        except Exception as e:
            log(f"Steering check failed: {e}")

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

        # Sub-agents inherit the SAME model as the primary (single served model).
        worker.model_name = config.get('model', 'unknown')
        worker.model_config = config

        tools = load_tools_from_directory("aeon.tools", dependencies=deps)
        # Strip tools that sub-agents must not have (no recursion, no self-mod).
        tools = [t for t in tools if getattr(t, "name", None) not in SUB_AGENT_FORBIDDEN_TOOLS]
        worker.register_tools(tools)

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
