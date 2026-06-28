import os
import sys
import json
import uuid
import time
import signal
import ctypes
import subprocess
from pathlib import Path

from aeon.tools.base import BaseTool
from aeon.core import runtime_signals as rt
from aeon.core.sub_agent_state import resolve, norm_status, group_kill


def _resolve_agent_dir(base_dir, agent_id):
    """Resolve an agent_id (full UUID, an unambiguous prefix, or a full directory
    name) to its actual sub-agent directory. gather_sub_agents shows operators a
    SHORT id, so the model frequently passes a prefix back to report/kill/steer;
    an exact-match lookup then fails with 'not found'. Matches:
      1. exact directory name (fast path)
      2. unique prefix match
      3. unique substring match (covers labelled dirs like 'verify_<uuid>')
    Returns (path, error_string). Exactly one of the two is None.
    """
    base_dir = Path(base_dir)
    if not agent_id:
        return None, "No agent_id provided."
    if not base_dir.exists():
        return None, "No sub-agents have been spawned in this session."

    agent_id = str(agent_id).strip()
    exact = base_dir / agent_id
    if exact.exists() and exact.is_dir():
        return exact, None

    dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    prefix = [d for d in dirs if d.name.startswith(agent_id)]
    if len(prefix) == 1:
        return prefix[0], None
    if len(prefix) > 1:
        opts = ", ".join(sorted(d.name[:12] for d in prefix))
        return None, (f"Ambiguous agent id '{agent_id}' matches multiple sub-agents ({opts}). "
                      f"Use more characters of the id.")

    sub = [d for d in dirs if agent_id in d.name]
    if len(sub) == 1:
        return sub[0], None
    if len(sub) > 1:
        opts = ", ".join(sorted(d.name[:12] for d in sub))
        return None, (f"Ambiguous agent id '{agent_id}' matches multiple sub-agents ({opts}). "
                      f"Use more characters of the id.")

    available = sorted(d.name[:12] for d in dirs if (d / "pid.txt").exists())
    hint = f" Known sub-agents: {', '.join(available)}." if available else ""
    return None, f"Agent '{agent_id}' not found.{hint}"


def uncollected_sub_agents(base_dir, notified_set):
    """Return short ids of sub-agents that have a terminal result which was never
    surfaced to the principal (i.e. never gathered/reported). Used to stop the
    primary from abandoning a dispatched researcher at task_complete."""
    base_dir = Path(base_dir)
    out = []
    if not base_dir.exists():
        return out
    for d in base_dir.iterdir():
        if not (d.is_dir() and (d / "pid.txt").exists()):
            continue
        is_term, status, _ = resolve(d)
        if not is_term:
            # Still running but spawned this session and never harvested -> also worth flagging.
            out.append((d.name.split("-")[0], "RUNNING"))
            continue
        key = f"{d.name}_{norm_status(status)}"
        if key not in (notified_set or set()):
            out.append((d.name.split("-")[0], norm_status(status)))
    return out


class SpawnSubAgent(BaseTool):
    MAX_CONCURRENT = 5
    DEFAULT_BUDGET_MIN = 40
    DEFAULT_STALL = 600
    HARD_WALLCLOCK_CEILING = 7200
    HARD_STALL_CEILING = 1800

    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="spawn_sub_agent",
            description=(
                "Dispatch a background sub-agent (a graduate student) to work an INDEPENDENT thread in "
                "parallel (a research avenue, a separate module, an isolated experiment). You are its "
                "advisor: spawn the batch, then each turn watch the SUB-AGENTS section of your context, "
                "steer the ones drifting (steer_sub_agent), relay useful cross-findings, and do your own "
                "orthogonal work meanwhile -- never sit idle waiting. NEVER spawn a sub-agent for something "
                "you can do yourself in 1-2 commands, and NEVER finish a task while a sub-agent you spawned "
                "is still running or unread: collect its report (get_sub_agent_report), or kill it if you no "
                "longer need its work, before task_complete.\n"
                "Each sub-agent runs your model, shares the workspace, CANNOT spawn its own sub-agents, and "
                "is GUARANTEED to reach a terminal state within its budget (an internal watchdog enforces "
                "this). Its final report reaches you via get_sub_agent_report, so make the deliverable "
                "explicit in the objective.\n"
                "Schema:\n"
                "  objective (str, required): a SELF-CONTAINED task with clear, explicitly-stated deliverables.\n"
                "  time_budget_minutes (int, optional, default=40): wall-clock budget. Deep research: 60-90; "
                "quick lookups: 10-15. Capped at 120.\n"
                "  max_iterations (int, optional, default=20): planning-step cap.\n"
                "  stall_timeout_seconds (int, optional, default=600): kill if it makes no progress this long.\n"
                "Example: {\"tool_name\": \"spawn_sub_agent\", \"parameters\": {\"objective\": \"Map aeon/tools: "
                "list every tool, its base class, and its upstream/downstream imports; report as a structured "
                "summary with risks.\", \"time_budget_minutes\": 30}}"
            ),
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / self.worker.instance_id / "sub_agents"

    def _running_count(self):
        if not self.output_dir.exists():
            return 0
        n = 0
        for d in self.output_dir.iterdir():
            if d.is_dir() and (d / "pid.txt").exists() and not resolve(d)[0]:
                n += 1
        return n

    def execute(self, objective, time_budget_minutes=None, max_iterations=None, stall_timeout_seconds=None):
        if not self.worker:
            return "COMMAND FAILED: Worker context missing."

        running = self._running_count()
        if running >= self.MAX_CONCURRENT:
            return (f"COMMAND FAILED: Maximum concurrent sub-agents ({self.MAX_CONCURRENT}) reached. "
                    f"Wait/collect with gather_sub_agents or free one with kill_sub_agent.")

        model_cfg = getattr(self.worker, "model_config", None)
        if not model_cfg:
            return ("COMMAND FAILED: No model_config on the primary worker, so a sub-agent cannot be "
                    "configured with the active model.")

        try:
            budget_min = int(time_budget_minutes) if time_budget_minutes else self.DEFAULT_BUDGET_MIN
        except (TypeError, ValueError):
            budget_min = self.DEFAULT_BUDGET_MIN
        max_wallclock = max(60, min(budget_min * 60, self.HARD_WALLCLOCK_CEILING))

        try:
            stall = int(stall_timeout_seconds) if stall_timeout_seconds else self.DEFAULT_STALL
        except (TypeError, ValueError):
            stall = self.DEFAULT_STALL
        stall = max(60, min(stall, self.HARD_STALL_CEILING))

        try:
            iters = int(max_iterations) if max_iterations else 20
        except (TypeError, ValueError):
            iters = 20
        iters = max(1, min(iters, 100))

        agent_id = str(uuid.uuid4())
        agent_dir = self.output_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        workspace_path = Path(os.getcwd())
        symlink_path = agent_dir / "workspace"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(workspace_path)

        coordinated_objective = (
            f"{objective}\n\n"
            f"[COORDINATION] You are one of several parallel sub-agents sharing this task. BEFORE starting "
            f"a self-contained chunk of work, call blackboard_read to check whether a sibling already "
            f"produced that result or already hit that dead end. When you produce something reusable, call "
            f"blackboard_post.\n\n"
            f"[REPORTING] Your final say_to_user message IS your report back to the principal agent and is "
            f"the deliverable. Before you call task_complete, deliver your COMPLETE findings via say_to_user "
            f"as a structured report (not a one-line summary, not just a log of what you opened). The "
            f"principal cannot see your internal thoughts or your task_complete reason."
        )

        cmd = [
            sys.executable, "-m", "aeon.scripts.sub_agent_wrapper",
            "--agent_id", agent_id,
            "--objective", coordinated_objective,
            "--model_config", json.dumps(model_cfg),
            "--workspace", str(symlink_path),
            "--output_dir", str(agent_dir),
            "--max_iterations", str(iters),
            "--stall_timeout", str(stall),
            "--max_wallclock", str(max_wallclock),
        ]
        if getattr(self.worker, "debug_mode", False):
            cmd.append("--debug")

        def set_pdeathsig():
            try:
                ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
            except Exception:
                pass

        log_fd = open(agent_dir / "agent.log", "a")
        try:
            process = subprocess.Popen(
                cmd,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                preexec_fn=set_pdeathsig,
                start_new_session=True,
            )
        except Exception as e:
            log_fd.close()
            return f"COMMAND FAILED: could not launch sub-agent process: {e}"

        rt.atomic_write_text(agent_dir / "pid.txt", str(process.pid))
        rt.atomic_write_text(agent_dir / "status.txt", "RUNNING")

        short_id = agent_id[:8]
        return (f"Sub-agent spawned. Agent ID: {agent_id} (refer to it as '{short_id}' in steer/report/kill). "
                f"Budget: {max_wallclock // 60} min wall-clock, {stall}s stall, {iters} max iterations. "
                f"It will now appear LIVE in the SUB-AGENTS section of your context every turn -- watch it "
                f"there, steer_sub_agent if it drifts, and meanwhile advance your own orthogonal work. You "
                f"must collect its report with get_sub_agent_report (or kill_sub_agent if you no longer need "
                f"it) before you can task_complete.")


class GatherSubAgents(BaseTool):
    DEFAULT_TIMEOUT = 0
    HARD_MAX_TIMEOUT = 120
    STALL_FLAG_SECONDS = 120
    FREEZE_SECONDS = 60
    POLL_INTERVAL = 3

    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="gather_sub_agents",
            description=(
                "Snapshot check-in on your sub-agents (your graduate students). For each you get its short id, "
                "status, time since last progress, current step, and any stall/loop/freeze flag, plus a "
                "recommended action. NOTE: you ALSO see this automatically in the SUB-AGENTS section of your "
                "context every turn -- so as an advisor you should mostly be ACTING on that (steer_sub_agent, "
                "get_sub_agent_report, kill_sub_agent) and doing your own orthogonal work, not repeatedly "
                "polling here.\n"
                "By default this returns an INSTANT snapshot and does NOT block. Only pass a non-zero timeout "
                "when you have genuinely nothing else to do and want to pause until something changes; even "
                "then it returns the moment any agent finishes/freezes (capped at 120s). Do NOT use it as an "
                "idle wait loop -- supervising and doing parallel work is the whole point of dispatching them.\n"
                "Schema:\n"
                "  agent_ids (list[str], optional): specific ids; omit for all running sub-agents.\n"
                "  timeout (int, optional, default=0): 0 = instant snapshot (recommended). Non-zero = wait up "
                "to this many seconds (capped 120) for a change.\n"
                "  stall_threshold (int, optional, default=120): flag an agent showing no progress this long.\n"
                "Example: {\"tool_name\": \"gather_sub_agents\", \"parameters\": {}}"
            ),
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / self.worker.instance_id / "sub_agents"

    def _short_id(self, dir_name):
        return dir_name.split("-")[0]

    def _progress(self, agent_dir):
        """Delegate to the shared reader so gather and the principal's always-on
        digest never disagree about what a student is doing."""
        from aeon.core.sub_agent_state import read_progress
        return read_progress(agent_dir, freeze_seconds=self.FREEZE_SECONDS)

    def execute(self, agent_ids=None, timeout=None, stall_threshold=None):
        if not self.worker:
            return "Error: Worker context missing."
        try:
            timeout = self.DEFAULT_TIMEOUT if timeout is None else int(timeout)
        except (TypeError, ValueError):
            timeout = self.DEFAULT_TIMEOUT
        timeout = max(0, min(self.HARD_MAX_TIMEOUT, timeout))
        try:
            stall_threshold = self.STALL_FLAG_SECONDS if stall_threshold is None else int(stall_threshold)
        except (TypeError, ValueError):
            stall_threshold = self.STALL_FLAG_SECONDS

        base = self.output_dir
        if not base.exists():
            return "No sub-agents have been spawned in this session."

        missing = []
        if agent_ids:
            if isinstance(agent_ids, str):
                agent_ids = [agent_ids]
            targets = []
            for aid in agent_ids:
                d, err = _resolve_agent_dir(base, aid)
                if d:
                    targets.append(d)
                else:
                    missing.append(str(aid))
            if not targets:
                return f"None of the requested sub-agents were found: {agent_ids}"
        else:
            targets = [d for d in base.iterdir()
                       if d.is_dir() and ((d / "status.txt").exists()
                                          or (d / "output.json").exists()
                                          or (d / "pid.txt").exists())]
            if not targets:
                return "No sub-agents found to gather."

        initially_running = {d.name for d in targets if not resolve(d)[0]}
        start = time.time()
        while (time.time() - start) < timeout:
            running_now = {d.name for d in targets if not resolve(d)[0]}
            if running_now != initially_running:
                break
            if not running_now:
                break
            if any(self._progress(d)["frozen"] for d in targets if d.name in running_now):
                break
            time.sleep(self.POLL_INTERVAL)

        completed = failed = killed = stalled = frozen = looping = healthy = 0
        lines = []
        for d in targets:
            is_term, status, report = resolve(d)
            sid = self._short_id(d.name)
            if is_term:
                base_status = norm_status(status)
                self.worker.notified_sub_agents.add(f"{d.name}_{base_status}")
                if base_status == "COMPLETED":
                    completed += 1
                    lines.append(f"[{sid}] COMPLETED\n  {(report or '')[:800]}\n"
                                 f"  (full findings: get_sub_agent_report(agent_id='{sid}'))")
                elif base_status == "KILLED":
                    killed += 1
                    lines.append(f"[{sid}] KILLED")
                else:
                    failed += 1
                    lines.append(f"[{sid}] {status}\n  {(report or '')[:600]}")
                continue
            pr = self._progress(d)
            age, step, it, is_frozen, stuck = pr["age"], pr["step"], pr["iteration"], pr["frozen"], pr["stuck_reason"]
            age_str = f"{age:.0f}s ago" if age is not None else "unknown"
            sfx = (f" on '{step}'" if step else "") + (f" (iter {it})" if it else "")
            if is_frozen:
                frozen += 1
                lines.append(f"[{sid}] FROZEN - watchdog stopped responding (whole-process freeze). "
                             f"It cannot self-recover; kill_sub_agent(agent_id='{sid}') and proceed.")
            elif stuck:
                looping += 1
                lines.append(f"[{sid}] LOOPING - {stuck} It is burning budget without progress; "
                             f"steer_sub_agent(agent_id='{sid}') with a new approach, or kill_sub_agent.")
            elif age is not None and age > stall_threshold:
                stalled += 1
                lines.append(f"[{sid}] POSSIBLY STALLED - no progress for {age:.0f}s{sfx}. "
                             f"Confirm with get_sub_agent_report(agent_id='{sid}'), then steer or kill.")
            else:
                healthy += 1
                lines.append(f"[{sid}] RUNNING (healthy) - last progress {age_str}{sfx}. "
                             f"Do other orthogonal work, or gather_sub_agents again with a non-zero timeout to wait.")

        header = (f"Check-in: {completed} completed, {failed} failed, {killed} killed, "
                  f"{stalled} possibly stalled, {looping} looping, {frozen} frozen, "
                  f"{healthy} healthy & running.")
        if missing:
            header += f" (Requested but not found: {missing})"
        if frozen:
            footer = "\n\nAction: kill the FROZEN agent(s) - they cannot recover - then continue."
        elif looping:
            footer = ("\n\nAction: a LOOPING agent self-reported it is repeating itself. Steer it with a "
                      "concretely different approach, or kill it if its work is no longer needed.")
        elif stalled:
            footer = ("\n\nAction: confirm stalls with get_sub_agent_report before acting (an agent may be on "
                      "a long legitimate step). If truly stuck, steer with a corrected approach or kill.")
        elif healthy:
            footer = ("\n\nAction: agents are still running and you can see them live in your SUB-AGENTS "
                      "section each turn. Advance your OWN orthogonal work and steer them as needed; collect "
                      "each report (get_sub_agent_report) before you finish the task. Don't idle-poll.")
        else:
            footer = ""
        return header + "\n\n" + "\n\n".join(lines) + footer


class GetSubAgentReport(BaseTool):
    MAX_RESULT_CHARS = 8000

    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="get_sub_agent_report",
            description=(
                "Read a sub-agent in depth. If finished, returns its FULL findings (fold these into your "
                "synthesis and spawn follow-ups for leads it surfaced). If still running, returns a live "
                "analysis of its recent activity. Accepts the short id shown by gather_sub_agents or a full "
                "UUID. Don't call this every turn for a running agent - prefer gather_sub_agents for batch "
                "check-ins.\n"
                "Schema:\n"
                "  agent_id (str, required): short id or full UUID.\n"
                "  specific_question (str, optional): a targeted question about a running agent's progress.\n"
                "Example: {\"tool_name\": \"get_sub_agent_report\", \"parameters\": {\"agent_id\": \"a44fa909\"}}"
            ),
            underlying_model=llm_client.model if llm_client else None,
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / self.worker.instance_id / "sub_agents"

    def execute(self, agent_id, specific_question=None):
        agent_dir, err = _resolve_agent_dir(self.output_dir, agent_id)
        if err:
            return err

        is_term, status, report = resolve(agent_dir)
        base_status = norm_status(status)

        if is_term:
            self.worker.notified_sub_agents.add(f"{agent_dir.name}_{base_status}")
            if base_status == "COMPLETED":
                result = report or "N/A"
                tail = ""
                if len(result) > self.MAX_RESULT_CHARS:
                    tail = (f"\n\n[... truncated at {self.MAX_RESULT_CHARS} chars; full text in "
                            f"{agent_dir / 'output.json'} ...]")
                    result = result[:self.MAX_RESULT_CHARS]
                return f"Agent {agent_dir.name[:8]} Status: COMPLETED\n\n--- FINDINGS ---\n{result}{tail}"
            return f"Agent {agent_dir.name[:8]} Status: {status}\n\n{report or ''}"

        report_str = f"Agent {agent_dir.name[:8]} Status: RUNNING"
        log_path = agent_dir / "agent.log"
        log_tail = ""
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_tail = "".join(f.readlines()[-150:])
            except Exception as e:
                log_tail = f"(Could not read log: {e})"

        if self.llm_client and log_tail:
            prompt = (
                f"You are a principal agent monitoring a research sub-agent's progress.\n"
                f"Analyze this recent log tail from sub-agent '{agent_dir.name[:8]}'.\n"
                f"1. What concrete progress has it made recently?\n"
                f"2. Is it stuck, looping, or blocked?\n"
                f"3. Any critical errors?\n"
                f"4. Recommendation: keep waiting, steer it, or kill it?\n"
            )
            if specific_question:
                prompt += f"\nAlso answer this specific question: {specific_question}\n"
            prompt += f"\n--- RECENT LOG TAIL ---\n{log_tail}\n--- END LOG ---"
            try:
                resp = self.llm_client.client.chat.completions.create(
                    model=self.llm_client.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                report_str += f"\n\n[LIVE PROGRESS ANALYSIS]\n{resp.choices[0].message.content}"
            except Exception as e:
                report_str += f"\n\n[LIVE PROGRESS ANALYSIS FAILED]: {e}\nRaw log tail:\n{log_tail[-1000:]}"
        elif log_tail:
            report_str += f"\n\n[RECENT LOG TAIL]\n{log_tail[-1500:]}"
        else:
            report_str += "\n\n[No log data found yet.]"

        report_str += ("\n\n[GUIDANCE] Still running. You see its live status every turn in your SUB-AGENTS "
                       "section, so don't re-poll here each turn - advance your own orthogonal work. If it is "
                       "drifting, steer_sub_agent; if its work is no longer needed, kill_sub_agent. Do not "
                       "finish the task with this agent's report uncollected.")
        return report_str


class KillSubAgent(BaseTool):
    def __init__(self, worker=None, llm_client=None):
        super().__init__(
            name="kill_sub_agent",
            description=(
                "Terminate a sub-agent and its child processes when it is stuck, frozen, or no longer needed. "
                "Kills the whole process group so nothing leaks. Accepts the short id shown by "
                "gather_sub_agents or a full UUID.\n"
                "Schema:\n  agent_id (str, required): short id or full UUID.\n"
                "Example: {\"tool_name\": \"kill_sub_agent\", \"parameters\": {\"agent_id\": \"a44fa909\"}}"
            ),
        )
        self.worker = worker
        self.llm_client = llm_client

    @property
    def output_dir(self):
        return Path(os.getcwd()) / "aeon_output" / self.worker.instance_id / "sub_agents"

    def execute(self, agent_id):
        agent_dir, err = _resolve_agent_dir(self.output_dir, agent_id)
        if err:
            return err

        rt.atomic_write_json(agent_dir / "output.json", {
            "agent_id": agent_dir.name,
            "status": "KILLED",
            "result": "Terminated by the principal agent before completion.",
        })
        rt.atomic_write_text(agent_dir / "status.txt", "KILLED")
        self.worker.notified_sub_agents.add(f"{agent_dir.name}_KILLED")

        pid_path = agent_dir / "pid.txt"
        if not pid_path.exists():
            return f"Sub-agent {agent_dir.name[:8]} marked KILLED (no PID file; process may have already exited)."
        try:
            pid = int(pid_path.read_text().strip())
        except Exception:
            return f"Sub-agent {agent_dir.name[:8]} marked KILLED (PID file unreadable)."
        group_kill(pid)
        return f"Sub-agent {agent_dir.name[:8]} (PID {pid}) terminated (process group killed) and marked KILLED."
