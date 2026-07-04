import os
import sys
import json
import uuid
import subprocess
from pathlib import Path
from .base import BaseTool

class VerifySelfModificationTool(BaseTool):
    """A tool to safely test self-modifications by running the new code in a sandboxed sub-agent."""
    
    def __init__(self, worker=None):
        super().__init__(
            name="verify_self_modification",
            description=(
                "Tests code modifications by spawning a sub-agent with the new code to complete a test objective. "
                "Use this BEFORE restart_aeon to ensure your changes work and won't crash the main process.\n"
                "Schema:\n"
                "  test_objective (str, required): A specific task for the sub-agent to perform to exercise your newly added code/tool.\n"
                "  timeout (int, optional, default=180): Max seconds to wait for the test to complete.\n"
                "Example: {\"tool_name\": \"verify_self_modification\", \"parameters\": {\"test_objective\": \"Use the newly created calculator tool to compute 125 * 3.4\"}}"
            )
        )
        self.worker = worker

    def _run_test_gate(self, workspace: str, timeout: int = 120):
        """Run smoke_test then test_core. Returns None if both pass (or are
        absent), or a failure message string to short-circuit verification."""
        gates = [
            ("smoke test", [sys.executable, "-B", "-m", "aeon.smoke_test"], "aeon/smoke_test.py"),
            ("unit tests", [sys.executable, "-B", "-m", "aeon.tests.test_core"], "aeon/tests/test_core.py"),
        ]
        for label, cmd, marker in gates:
            if not os.path.exists(os.path.join(workspace, marker)):
                continue  # gate not present in this code version — skip gracefully
            print(f"{self.C_CYAN}[VERIFY] Running {label}...{self.C_RESET}")
            try:
                res = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return (f"VERIFICATION FAILED: the {label} timed out after {timeout}s — your change likely "
                        f"introduced a hang at import/collection time. Fix it before retrying.")
            if res.returncode != 0:
                out = ((res.stdout or "") + (res.stderr or "")).strip()
                return (
                    f"VERIFICATION FAILED at the {label} (before any sub-agent ran).\n\n"
                    f"--- Output (tail) ---\n{out[-1500:]}\n\n"
                    f"Action Required: fix the failure above with str_replace/write_file, then run "
                    f"verify_self_modification again. (Caught cheaply — no sub-agent was spawned.)"
                )
            print(f"{self.C_GREEN}[VERIFY] {label} passed.{self.C_RESET}")
        return None

    def _aeon_source_root(self) -> str:
        """The Aeon source/install root (dir containing setup.py), independent of
        the current workspace. Self-modifications live here, so pip install and the
        test gates MUST run against this tree — NOT os.getcwd(), which is the user's
        project when aeon is run portably from another directory (installing that by
        mistake, and testing an unchanged aeon)."""
        try:
            from ..core.paths import PROJECT_ROOT
            return str(PROJECT_ROOT)
        except Exception:
            return os.getcwd()

    def execute(self, test_objective: str, timeout: int = 180) -> str:
        if not self.worker:
            return "Error: Worker context missing."

        # The sub-agent runs in the user's workspace (where the real task lives),
        # but the code under test — and thus pip install and the gates — is the
        # aeon source tree, which may be a different directory entirely.
        workspace = os.getcwd()
        aeon_root = self._aeon_source_root()

        # 1. Pip install the AEON SOURCE so entry points/cache reflect the change.
        print(f"{self.C_CYAN}[VERIFY] Applying changes to sub-environment (pip install . in {aeon_root})...{self.C_RESET}")
        pip_res = subprocess.run(
            [sys.executable, "-m", "pip", "install", ".", "--quiet"],
            cwd=aeon_root, capture_output=True, text=True
        )
        if pip_res.returncode != 0:
            return f"Verification failed during pip install:\n{pip_res.stderr}\nFix the syntax/build errors before continuing."

        # 1b. FAIL FAST: run the cheap, deterministic test gate (smoke + unit
        # tests) BEFORE spinning up an expensive sub-agent (LLM + GPU). A syntax
        # error, broken import, or parser regression is caught here in ~1s with
        # a precise message instead of after a multi-minute sub-agent run. The
        # gate files (smoke_test.py, tests/) live in the aeon source tree.
        gate = self._run_test_gate(aeon_root, timeout=120)
        if gate is not None:
            return gate

        # 2. Setup isolated sub-agent output directory
        agent_id = f"verify_{uuid.uuid4().hex[:8]}"
        instance_id = getattr(self.worker, 'instance_id', 'default')
        output_dir = os.path.join(workspace, "aeon_output", instance_id, "sub_agents", agent_id)
        os.makedirs(output_dir, exist_ok=True)

        # Extract model config from the parent worker safely
        model_cfg = getattr(self.worker, 'model_config', {})
        if not model_cfg:
            return "Error: Could not retrieve model configuration from primary agent."

        # 3. Build command for sub_agent_wrapper
        cmd = [
            sys.executable, "-m", "aeon.scripts.sub_agent_wrapper",
            "--agent_id", agent_id,
            "--objective", test_objective,
            "--model_config", json.dumps(model_cfg),
            "--workspace", workspace,
            "--output_dir", output_dir,
            "--max_iterations", "5"  # Short iteration limit to prevent infinite loops during tests
        ]
        if getattr(self.worker, 'debug_mode', False):
            cmd.append("--debug")

        print(f"{self.C_CYAN}[VERIFY] Spawning test sub-agent. Objective: '{test_objective}'{self.C_RESET}")

        try:
            # 4. Run synchronously
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return (
                f"Verification timed out after {timeout} seconds. "
                f"The modification might have caused an infinite loop or hang.\n\n"
                f"Sub-agent Stdout (Tail):\n{e.stdout[-1000:] if e.stdout else 'N/A'}\n\n"
                f"Sub-agent Stderr:\n{e.stderr[-1000:] if e.stderr else 'N/A'}\n\n"
                f"Action Required: Fix the hang/loop in your code and try again."
            )

        # 5. Read outputs
        status_file = os.path.join(output_dir, "status.txt")
        output_file = os.path.join(output_dir, "output.json")
        log_file = os.path.join(output_dir, "agent.log")

        status = "UNKNOWN"
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = f.read().strip()

        final_report = "No output.json generated."
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    if "error" in data:
                        final_report = f"Error: {data['error']}"
                    else:
                        final_report = f"Result: {data.get('result', 'N/A')}"
            except Exception as parse_e:
                final_report = f"Failed to parse output.json: {parse_e}"

        log_tail = ""
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                lines = f.readlines()
                log_tail = "".join(lines[-50:])

        # 6. Evaluate success
        if res.returncode != 0 or status in ["FAILED", "UNKNOWN"]:
            return (
                f"VERIFICATION FAILED (Status: {status}, Exit Code: {res.returncode})\n\n"
                f"--- Sub-agent Report ---\n{final_report}\n\n"
                f"--- Sub-agent Stderr ---\n{res.stderr}\n\n"
                f"--- Log Tail ---\n{log_tail}\n\n"
                f"Action Required: Review the errors above, fix your code using str_replace/write_file, and run `verify_self_modification` again."
            )

        return (
            f"VERIFICATION SUCCESSFUL!\n"
            f"Sub-agent completed the objective without crashing.\n\n"
            f"--- Sub-agent Report ---\n{final_report}\n\n"
            f"--- Sub-agent Stdout (Tail) ---\n{res.stdout[-1000:] if res.stdout else 'N/A'}\n\n"
            f"If this result confirms your modification works as intended, you may now safely call `restart_aeon` to integrate the changes into your primary process."
        )
