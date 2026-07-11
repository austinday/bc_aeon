import os
import sys
import json
import argparse
import time
import threading
import signal
from pathlib import Path
from aeon.core.worker import Worker
from aeon.core.llm import LLMClient
from aeon.tools.loader import load_tools_from_directory
from aeon.core import runtime_signals as rt

# Tools a sub-agent is NOT allowed to have: no recursive spawning (runaway GPU
# oversubscription) and no self-modification/restart of the framework.
SUB_AGENT_FORBIDDEN_TOOLS = {
    "spawn_sub_agent",
    "gather_sub_agents",
    "get_sub_agent_report",
    "kill_sub_agent",
    "steer_sub_agent",
    "get_sub_agent_status",
    "verify_self_modification",
    "restart_aeon",
    # No detached background jobs either: a sub-agent is short-lived and exists to
    # return a report, so a job that would outlive it (and whose workload group
    # would orphan when the wrapper exits) has no owner. Use run_command instead.
    "run_command_async",
    "job_output",
    "kill_job",
    # No interactive user: stdin is detached (DEVNULL) and the terminal belongs
    # to the principal. A sub-agent must report blockers in its final report,
    # not sit waiting for input that can never arrive.
    "get_user_input",
}


def _release_browser_profile(agent_id):
    """Best-effort: tell the browser service to close this sub-agent's isolated
    context so its Chrome doesn't linger after the task ends. No-op if the sub
    agent never browsed (service not running / profile never created)."""
    try:
        import requests
        requests.post("http://localhost:8030/release_profile",
                      json={"profile": f"agent-{agent_id}"}, timeout=5)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Aeon Sub-Agent Wrapper")
    parser.add_argument("--agent_id", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--stall_timeout", type=int, default=600,
                        help="Hard-terminate if no liveness signal for this many seconds.")
    parser.add_argument("--max_wallclock", type=int, default=2400,
                        help="Absolute wall-clock cap in seconds.")
    parser.add_argument("--read_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.json"
    status_path = output_dir / "status.txt"
    log_path = output_dir / "agent.log"
    telemetry_path = output_dir / "telemetry.json"
    progress_path = output_dir / "progress.json"
    steering_path = output_dir / "steering.jsonl"
    steering_offset_path = output_dir / ".steering_offset"

    def log(message):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    # ----- liveness / watchdog state -----
    done_event = threading.Event()
    rt.reset()
    started_at = time.time()
    current_step = {"iteration": 0, "step": "initializing", "stuck_reason": None}

    # Only ever group-kill if WE are the group leader. verify_self_modification
    # launches this wrapper inside the PRIMARY's process group; a killpg there
    # would kill the primary. In that (non-leader) case fall back to exiting self.
    try:
        is_group_leader = (os.getpgrp() == os.getpid())
    except Exception:
        is_group_leader = False

    def write_terminal(status_value, payload):
        # Durable record FIRST (output.json is never deleted and is read first by
        # the principal via resolve()), THEN the status file. Invariant: a terminal
        # status always implies the result/error is already on disk.
        try:
            rt.atomic_write_json(output_path, payload)
        except Exception as e:
            log(f"failed to write output.json: {e}")
        try:
            rt.atomic_write_text(status_path, status_value)
        except Exception as e:
            log(f"failed to write status.txt: {e}")

    def publish_progress():
        try:
            rt.atomic_write_json(progress_path, {
                "agent_id": args.agent_id,
                "alive": True,
                "started_at": started_at,
                "updated_at": time.time(),
                "activity_age": round(rt.activity_age(), 1),
                "wallclock": round(time.time() - started_at, 1),
                "iteration": current_step["iteration"],
                "step": current_step["step"],
                "stuck_reason": current_step.get("stuck_reason"),
            })
        except Exception:
            pass

    def watchdog():
        # Daemon thread. Heartbeats progress.json so the principal can tell this
        # agent is alive (and detect a whole-process freeze via the file mtime),
        # and HARD-terminates on stall or wall-clock breach so the principal is
        # never blocked indefinitely. os._exit/killpg work even if the main thread
        # is wedged in a C call -- the canonical hang case.
        publish_progress()
        while not done_event.wait(timeout=5):
            age = rt.activity_age()
            wall = time.time() - started_at
            stalled = age > args.stall_timeout
            expired = wall > args.max_wallclock
            if not (stalled or expired):
                publish_progress()
                continue
            reason = (f"no progress for {age:.0f}s (stall_timeout={args.stall_timeout}s)"
                      if stalled else
                      f"exceeded wall-clock budget {wall:.0f}s (max_wallclock={args.max_wallclock}s)")
            log(f"[WATCHDOG] Hard-terminating sub-agent: {reason}")
            write_terminal(f"FAILED: watchdog - {reason}", {
                "agent_id": args.agent_id,
                "status": "FAILED",
                "error": f"Watchdog terminated the sub-agent: {reason}",
                "note": ("Stalled or over budget; force-terminated so the principal agent is "
                         "never blocked. Re-spawn with a larger time_budget_minutes if the task "
                         "is legitimately long, or refine the objective if it got stuck."),
                "last_step": dict(current_step),
            })
            try:
                sys.stderr.flush()
            except Exception:
                pass
            if is_group_leader:
                try:
                    os.killpg(os.getpgrp(), signal.SIGKILL)  # self + all command children
                except Exception:
                    os._exit(1)
            else:
                os._exit(1)

    threading.Thread(target=watchdog, daemon=True, name="aeon-subagent-watchdog").start()

    def consume_steering():
        # Queued, ordered, never-lost/duplicated steering: read only the bytes
        # appended since our last offset, then advance the offset.
        if not steering_path.exists():
            return []
        try:
            offset = 0
            if steering_offset_path.exists():
                try:
                    offset = int(steering_offset_path.read_text().strip())
                except Exception:
                    offset = 0
            with open(steering_path, "r", encoding="utf-8") as f:
                f.seek(offset)
                chunk = f.read()
                new_offset = f.tell()
            rt.atomic_write_text(steering_offset_path, str(new_offset))
            messages = []
            for line in chunk.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line).get("guidance", line))
                except Exception:
                    messages.append(line)
            return messages
        except Exception as e:
            log(f"steering read failed: {e}")
            return []

    worker = None  # bound below; update_telemetry reads it as a free variable

    def update_telemetry(iteration, display_max, step_description):
        # Heartbeat + progress checkpoint + steering injection point, fired by the
        # worker at every iteration.
        rt.touch()
        current_step["iteration"] = iteration
        current_step["step"] = step_description
        # Surface the sub-agent's OWN loop-detector to the principal: a student
        # spinning on an identical command keeps touching the heartbeat, so it
        # never looks "stalled" -- this is how the principal learns it's looping.
        current_step["stuck_reason"] = getattr(worker, "stuck_reason", None)
        try:
            rt.atomic_write_json(telemetry_path, {
                "agent_id": args.agent_id,
                "iteration": iteration,
                "current_step": step_description,
                "timestamp": time.time(),
            })
        except Exception as e:
            log(f"telemetry write failed: {e}")
        publish_progress()
        if worker is not None:
            for guidance in consume_steering():
                worker.last_observation = (
                    f"[STEERING GUIDANCE FROM PRINCIPAL AGENT] {guidance}\n\n"
                    f"{worker.last_observation}"
                )
                log(f"applied steering guidance: {guidance[:120]}")

    try:
        log(f"Initializing sub-agent {args.agent_id}...")
        config = json.loads(args.model_config)
        llm_client = LLMClient(config)

        from aeon.main import register_models_for_agent, unregister_models_for_agent
        register_models_for_agent([config.get("model")])

        worker = Worker(llm_client=llm_client, debug_mode=args.debug)
        worker.model_name = config.get("model", "unknown")
        worker.model_config = config
        # Sub-agents share the principal's cwd (workspace symlink): with
        # persistence on they'd clobber the principal's session_state.json every
        # iteration and inherit its memories/plan at boot.
        worker.persist_session = False
        # Browse as an ISOLATED identity: each sub-agent gets its own browser
        # context (own cookies/session/fingerprint) instead of sharing the
        # principal's profile, so parallel agents don't collide.
        worker.browser_profile = f"agent-{args.agent_id}"

        tools = load_tools_from_directory(
            "aeon.tools", dependencies={"llm_client": llm_client, "worker": worker}
        )
        tools = [t for t in tools if getattr(t, "name", None) not in SUB_AGENT_FORBIDDEN_TOOLS]
        worker.register_tools(tools)

        os.chdir(args.workspace)
        log(f"Changed working directory to workspace: {args.workspace}")
        log(f"Starting execution of objective: {args.objective}")

        rt.atomic_write_text(status_path, "RUNNING")
        publish_progress()

        default_instruction = (
            "When you finish, provide a detailed, informative report of your findings, actions "
            "taken, and final result. This report will be read by the principal agent."
        )
        objective = f"{default_instruction}\n\n{args.objective}"

        worker.run(objective, max_iterations=args.max_iterations, step_callback=update_telemetry)

        done_event.set()  # stand the watchdog down BEFORE the final writes
        # The report the principal reads: prefer the sub-agent's final say_to_user
        # message (the instructed deliverable). last_observation is the fallback,
        # but on the terminal turn it still holds the PREVIOUS turn's output — the
        # say_to_user text itself is never echoed into it.
        final_report = getattr(worker, "last_say_to_user", None) or worker.last_observation
        write_terminal("COMPLETED", {
            "agent_id": args.agent_id,
            "status": "COMPLETED",
            "result": final_report,
            "plan": worker.current_plan,
            "memories": worker.memories,
        })
        log("Task completed successfully.")
        unregister_models_for_agent([config.get("model")])
        _release_browser_profile(args.agent_id)

    except Exception as e:
        done_event.set()
        log(f"CRITICAL ERROR: {e}")
        write_terminal(f"FAILED: {e}", {
            "agent_id": args.agent_id,
            "status": "FAILED",
            "error": str(e),
        })
        try:
            from aeon.main import unregister_models_for_agent
            unregister_models_for_agent([json.loads(args.model_config).get("model")])
        except Exception:
            pass
        _release_browser_profile(args.agent_id)
        sys.exit(1)


if __name__ == "__main__":
    main()
