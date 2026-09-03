import ctypes
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

from aeon.core import runtime_signals as rt
from aeon.core.fleet_backend import validate_loopback_endpoint
from aeon.core.sub_agent_environment import (
    CHILD_FLEET_CONFIGURATION_KEYS,
    VERIFICATION_PREBOUND_NONCE_ENV,
    VERIFICATION_PREBOUND_RECEIPT,
    SubAgentFleetCompute,
    bounded_sub_agent_environment,
)
from aeon.core.sub_agent_state import (
    CPU_SANDBOX_SLICE_ENV,
    ProcessIdentityError,
    assert_sub_agent_systemd_units_available,
    capture_sub_agent_process,
    sub_agent_systemd_command,
    sub_agent_systemd_units,
    terminate_sub_agent,
)
from aeon.core.utils.io import read_bounded_fd

from .base import BaseTool

_VERIFICATION_MODEL_KEYS = frozenset(
    {"provider", "model", "api_model", "base_url", "context_limit", "multimodal"}
)


def _model_verification_boundary_available() -> bool:
    """Whether the host can confine a child to one exact loopback endpoint.

    Current enforcement can deny all networking or restrict a TCP *port*, but
    cannot prove the destination address is loopback.  A preconnected-FD proxy
    (or equivalent actively-probed destination confinement) is required before
    model-driven candidate code may run.
    """

    return False


def _verification_parent_death_signal():
    """Ensure a verifier child cannot outlive a suddenly-dead principal."""

    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
    except Exception:
        pass


def _read_child_file(path: Path, *, limit: int, tail: bool = False) -> str:
    """Read one exact child-owned regular file without following symlinks."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise OSError("child output is not an exact owner-owned regular file")
        if tail and metadata.st_size > limit:
            os.lseek(descriptor, metadata.st_size - limit, os.SEEK_SET)
        data = read_bounded_fd(descriptor, limit)
        if not tail and len(data) > limit:
            raise OSError("child output exceeded its fixed read bound")
        if tail and len(data) > limit:
            data = data[-limit:]
        return data.decode("utf-8", "replace")
    finally:
        os.close(descriptor)

class VerifySelfModificationTool(BaseTool):
    """Test source changes without installing them into the live environment."""
    
    def __init__(self, worker=None):
        super().__init__(
            name="verify_self_modification",
            description=(
                "Fail-closed self-modification verifier. It never pip-installs into the live environment. "
                "On this host, candidate execution remains blocked until an actively-probed masked-home "
                "sandbox and exact loopback-only preconnected model transport are installed; port-only "
                "filtering is not accepted. Report that operator blocker instead of retrying.\n"
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
                env = dict(os.environ)
                env["PYTHONPATH"] = workspace + (
                    os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
                )
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                res = subprocess.run(
                    cmd, cwd=workspace, env=env, capture_output=True, text=True, timeout=timeout
                )
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
        the current workspace. Self-modifications and deterministic gates live in
        this tree; the child imports it through PYTHONPATH without installation."""
        try:
            from ..core.paths import PROJECT_ROOT
            return str(PROJECT_ROOT)
        except Exception:
            return os.getcwd()

    def execute(self, test_objective: str, timeout: int = 180) -> str:
        if not self.worker:
            return "Error: Worker context missing."
        if not isinstance(test_objective, str) or not test_objective.strip():
            return "Error: test_objective must be a non-empty string."
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            return "Error: timeout must be an integer number of seconds."
        if isinstance(timeout, bool) or timeout < 1 or timeout > 1800:
            return "Error: timeout must be between 1 and 1800 seconds."
        if not _model_verification_boundary_available():
            return (
                "VERIFICATION BLOCKED: this host cannot yet confine untrusted "
                "candidate code to only its exact Fleet-issued loopback model "
                "endpoint. Port-only filtering could expose non-loopback hosts, "
                "so Aeon refuses to launch the verification child. Install a "
                "reviewed preconnected-FD/proxy boundary, then retry."
            )

        # The sub-agent runs in the user's workspace, while PYTHONPATH binds it to
        # the exact modified Aeon source tree.
        workspace = str(Path.cwd().resolve())
        aeon_root = str(Path(self._aeon_source_root()).resolve())

        # 1. FAIL FAST: run the cheap, deterministic test gate (smoke + unit
        # tests) BEFORE spinning up an expensive sub-agent (LLM + GPU). A syntax
        # error, broken import, or parser regression is caught here in ~1s with
        # a precise message instead of after a multi-minute sub-agent run. The
        # gate files (smoke_test.py, tests/) live in the aeon source tree.
        gate = self._run_test_gate(aeon_root, timeout=120)
        if gate is not None:
            return gate

        # 2. Reserve one canonical UUID-derived systemd scope/slice before any
        # candidate module is imported in a child.
        agent_id = str(uuid.uuid4())
        try:
            scope_unit, slice_unit = sub_agent_systemd_units(agent_id)
            assert_sub_agent_systemd_units_available(agent_id)
        except ProcessIdentityError as exc:
            return f"VERIFICATION BLOCKED: could not reserve exact child units: {exc}"
        output_dir = Path(self.worker.sub_agent_output_dir()) / agent_id
        output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(output_dir, 0o700)

        # Copy only inference fields. Runtime/container/claim metadata from the
        # principal is neither needed nor allowed in the candidate process.
        parent_model_cfg = getattr(self.worker, "model_config", {})
        if not isinstance(parent_model_cfg, dict) or not parent_model_cfg:
            return "Error: Could not retrieve model configuration from primary agent."
        model_cfg = {
            key: parent_model_cfg[key]
            for key in _VERIFICATION_MODEL_KEYS
            if key in parent_model_cfg
        }

        # 3. Build the candidate wrapper argv. For local Qwen, base_url is
        # replaced below with this verifier's broker-returned endpoint before
        # this argv can reach Popen.
        cmd = [
            sys.executable, "-m", "aeon.scripts.sub_agent_wrapper",
            "--agent_id", agent_id,
            "--objective", test_objective,
            "--model_config", json.dumps(model_cfg),
            "--workspace", workspace,
            "--output_dir", str(output_dir),
            "--max_iterations", "5",
            "--read_only",
        ]
        if getattr(self.worker, 'debug_mode', False):
            cmd.append("--debug")

        print(f"{self.C_CYAN}[VERIFY] Spawning test sub-agent. Objective: '{test_objective}'{self.C_RESET}")

        request_id = str(getattr(self.worker, "request_id", "") or "unscoped")
        blackboard_path = self.worker.blackboard_path()
        env = bounded_sub_agent_environment()
        # Unlike an ordinary delegated agent, the verification candidate is not
        # a Fleet client. The trusted parent owns its ticket, and gives the child
        # only the resulting endpoint through its bounded model config.
        for key in CHILD_FLEET_CONFIGURATION_KEYS:
            env.pop(key, None)
        env.update({
            "PYTHONPATH": aeon_root + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "AEON_READ_ONLY": "1",
            "AEON_PARENT_INSTANCE_ID": str(self.worker.instance_id),
            "AEON_PARENT_REQUEST_ID": request_id,
            "AEON_BLACKBOARD_PATH": str(blackboard_path),
            CPU_SANDBOX_SLICE_ENV: slice_unit,
        })

        compute = SubAgentFleetCompute(
            agent_id=agent_id,
            model_config=model_cfg,
        )
        process = None
        process_receipted = False
        compute_started = False
        capability_path = output_dir / VERIFICATION_PREBOUND_RECEIPT
        stdout = ""
        stderr = ""
        failure = None
        lifecycle_failure = None
        release_failure = None
        deadline = time.monotonic() + timeout

        try:
            # 4. The trusted, already-loaded parent obtains and renews the local
            # Fleet demand before candidate Python is imported. No ticket, claim,
            # broker socket, profile, or device authority is serialized to the
            # child. External providers (if ever re-enabled) create no demand.
            if compute.required:
                try:
                    endpoint = validate_loopback_endpoint(compute.start())
                    compute_started = True
                    compute.assert_prebound_endpoint_healthy(endpoint)
                except Exception as exc:
                    failure = (
                        "VERIFICATION DEFERRED: independent Fleet compute could not "
                        f"be prepared safely ({type(exc).__name__}: {exc})."
                    )
                else:
                    nonce = secrets.token_hex(32)
                    rt.atomic_write_json(capability_path, {
                        "schema": 1,
                        "kind": "aeon-verification-prebound-fleet",
                        "agent_id": agent_id,
                        "scope_unit": scope_unit,
                        "slice_unit": slice_unit,
                        "endpoint": endpoint,
                        "nonce": nonce,
                    })
                    os.chmod(capability_path, 0o600)
                    env[VERIFICATION_PREBOUND_NONCE_ENV] = nonce
                    # Replace the principal URL only after exact acquisition.
                    cmd[cmd.index("--model_config") + 1] = json.dumps(model_cfg)

            if failure is None:
                scoped_cmd = sub_agent_systemd_command(agent_id, cmd)
                process = subprocess.Popen(
                    scoped_cmd,
                    cwd=workspace,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=_verification_parent_death_signal,
                    start_new_session=True,
                )
                process_ref = capture_sub_agent_process(
                    output_dir,
                    process.pid,
                    scope_unit=scope_unit,
                    slice_unit=slice_unit,
                )
                rt.atomic_write_json(output_dir / "process.json", process_ref)
                rt.atomic_write_text(output_dir / "pid.txt", str(process.pid))
                process_receipted = True

                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        failure = (
                            f"Verification timed out after {timeout} seconds. "
                            "The modification might have caused an infinite loop or hang."
                        )
                        break
                    try:
                        stdout, stderr = process.communicate(
                            timeout=min(1.0, remaining)
                        )
                        break
                    except subprocess.TimeoutExpired:
                        if compute_started:
                            try:
                                compute.assert_prebound_endpoint_healthy(endpoint)
                            except Exception as exc:
                                failure = (
                                    "VERIFICATION FAILED: the parent-owned Fleet "
                                    "endpoint changed or lost renewal while candidate "
                                    f"code was running ({type(exc).__name__}: {exc})."
                                )
                                break
        except Exception as exc:
            failure = (
                "VERIFICATION FAILED before the candidate result was trusted: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            # Retire the exact recursive slice before releasing its model ticket;
            # this catches descendants even if the wrapper already exited.
            if process is not None:
                if process_receipted:
                    try:
                        terminate_sub_agent(output_dir)
                    except Exception as exc:
                        lifecycle_failure = (
                            "exact verification slice retirement was not proven: "
                            f"{type(exc).__name__}: {exc}"
                        )
                else:
                    try:
                        process.kill()
                        process.wait(timeout=5)
                    except Exception:
                        pass
                    lifecycle_failure = (
                        "verification launcher identity could not be committed; "
                        "the pinned launcher was stopped"
                    )
                try:
                    final_stdout, final_stderr = process.communicate(timeout=5)
                    stdout = final_stdout or stdout
                    stderr = final_stderr or stderr
                except Exception:
                    pass
            try:
                capability_path.unlink(missing_ok=True)
            except OSError as exc:
                lifecycle_failure = lifecycle_failure or (
                    f"verification capability cleanup failed: {exc}"
                )
            if compute.required:
                last_error = None
                release_proof = None
                for _attempt in range(2):
                    try:
                        release_proof = compute.close()
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                if last_error is not None:
                    release_failure = (
                        "exact verification Fleet ticket release remains unresolved: "
                        f"{type(last_error).__name__}: {last_error}"
                    )
                elif compute_started and release_proof != {
                    "state": "released", "compute_state": "inactive"
                }:
                    release_failure = (
                        "verification Fleet broker did not return exact release proof"
                    )

        if release_failure is not None:
            return f"VERIFICATION FAILED: {release_failure}"
        if lifecycle_failure is not None:
            return f"VERIFICATION FAILED: {lifecycle_failure}"
        if failure is not None:
            return (
                f"{failure}\n\n"
                f"Sub-agent Stdout (Tail):\n{stdout[-1000:] if stdout else 'N/A'}\n\n"
                f"Sub-agent Stderr:\n{stderr[-1000:] if stderr else 'N/A'}\n\n"
                f"Action Required: Fix the hang/loop in your code and try again."
            )

        # 5. Read outputs
        status_file = output_dir / "status.txt"
        output_file = output_dir / "output.json"
        log_file = output_dir / "agent.log"

        status = "UNKNOWN"
        try:
            status = _read_child_file(status_file, limit=4096).strip()
        except OSError:
            pass

        final_report = "No output.json generated."
        try:
            data = json.loads(_read_child_file(output_file, limit=1024 * 1024))
            if not isinstance(data, dict):
                raise ValueError("result is not an object")
            if "error" in data:
                final_report = f"Error: {str(data['error'])[:10000]}"
            else:
                final_report = f"Result: {str(data.get('result', 'N/A'))[:100000]}"
        except Exception as parse_e:
            if output_file.exists() or output_file.is_symlink():
                final_report = f"Failed to parse output.json: {parse_e}"

        log_tail = ""
        try:
            log_tail = "".join(
                _read_child_file(log_file, limit=64 * 1024, tail=True).splitlines(True)[-50:]
            )
        except OSError:
            pass

        # 6. Evaluate success
        if process is None or process.returncode != 0 or status != "COMPLETED":
            return (
                f"VERIFICATION FAILED (Status: {status}, Exit Code: "
                f"{process.returncode if process is not None else 'N/A'})\n\n"
                f"--- Sub-agent Report ---\n{final_report}\n\n"
                f"--- Sub-agent Stderr ---\n{stderr[-10000:]}\n\n"
                f"--- Log Tail ---\n{log_tail}\n\n"
                f"Action Required: Review the errors above, fix your code using str_replace/write_file, and run `verify_self_modification` again."
            )

        return (
            f"VERIFICATION SUCCESSFUL!\n"
            f"Sub-agent completed the objective without crashing.\n\n"
            f"--- Sub-agent Report ---\n{final_report}\n\n"
            f"--- Sub-agent Stdout (Tail) ---\n{stdout[-1000:] if stdout else 'N/A'}\n\n"
            f"If this result confirms your modification works as intended, you may now safely call `restart_aeon` to integrate the changes into your primary process."
        )
