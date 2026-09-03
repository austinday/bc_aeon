from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from aeon.core.model_identity import AEON_DEFAULT_MODEL_NAME
from aeon.harnesses.launch import build_harness_argv
from aeon.harnesses.model_proxy import FleetModelProxy
from aeon.harnesses.opencode_config import (
    _atomic_private_bytes,
    isolated_environment,
    materialize_authority,
    materialize_config,
)
from aeon.harnesses.opencode_runtime import OpenCodeTurnRunner, _state_root


def test_harness_launch_is_fixed_and_modular() -> None:
    assert build_harness_argv(
        "/usr/bin/python3",
        "opencode",
        AEON_DEFAULT_MODEL_NAME,
        resume_unfinished=True,
        start_objective="inspect",
    ) == [
        "/usr/bin/python3",
        "-m",
        "aeon.harnesses.opencode_runtime",
        "--model",
        AEON_DEFAULT_MODEL_NAME,
        "--resume-unfinished",
        "--start",
        "inspect",
    ]
    assert build_harness_argv(
        "/usr/bin/python3", "legacy-aeon", AEON_DEFAULT_MODEL_NAME
    )[2] == "aeon.main"
    with pytest.raises(ValueError):
        build_harness_argv("/usr/bin/python3", "invented", AEON_DEFAULT_MODEL_NAME)


def test_isolated_config_disables_bypass_tools(tmp_path: Path) -> None:
    os.chmod(tmp_path, 0o700)
    instructions = _atomic_private_bytes(tmp_path, "instructions.md", b"safe\n")
    authority = materialize_authority(tmp_path, "Inspect this workspace")
    config_path = materialize_config(
        tmp_path,
        base_url="http://127.0.0.1:19001/v1",
        bearer_token="test-token",
        instruction_path=instructions,
        max_steps=7,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["model"] == "nexus-fleet/qwen"
    assert config["autoupdate"] is False
    assert config["share"] == "disabled"
    assert config["agent"]["aeon"]["steps"] == 7
    assert config["mcp"]["aeon"]["type"] == "local"
    for name in (
        "bash",
        "read",
        "glob",
        "grep",
        "edit",
        "write",
        "patch",
        "apply_patch",
        "webfetch",
        "websearch",
        "task",
        "todowrite",
        "skill",
        "question",
        "lsp",
        "plan_exit",
        "execute",
    ):
        assert config["tools"][name] is False
        assert config["permission"][name] == "deny"
    assert config["permission"]["*"] == "deny"
    assert config["permission"]["aeon_*"] == "allow"
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600

    environment = isolated_environment(
        {
            "PATH": "/usr/bin",
            "AEON_COMPUTE_BACKEND": "broker",
            "AEON_FLEET_SOCKET": "/run/user/1000/fleet.sock",
            "AEON_STATE_DIR": "/private/worker-state",
            "AEON_BROWSER_PROFILE": "reviewed-profile",
            "AEON_BROWSER_SESSION_ID": "oc-" + "a" * 32,
            "AEON_REMOTE_INSTANCE_ID": "b" * 32,
            "NEXUS_SELF_SETTINGS_TOKEN_FILE": "/private/token-file",
            "OPENAI_API_KEY": "must-not-leak",
            "NEXUS_DATABASE_PASSWORD": "must-not-leak",
            "LD_PRELOAD": "/tmp/must-not-load.so",
            "PYTHONPATH": "/tmp/must-not-import",
            "NODE_OPTIONS": "--require=/tmp/must-not-load.js",
            "OPENCODE_CONFIG": "/tmp/attacker.json",
            "OPENCODE_CONFIG_CONTENT": '{"tools":{"bash":true}}',
            "OPENCODE_SERVER_PASSWORD": "must-not-leak",
        },
        directory=tmp_path,
        config_path=config_path,
        authority_path=authority,
        base_url="http://127.0.0.1:19001/v1",
        bearer_token="test-token",
        logical_model=AEON_DEFAULT_MODEL_NAME,
        wire_model="Qwen3.8-27B-ARA-NVFP4-MTP",
    )
    assert environment["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert environment["OPENCODE_DISABLE_DEFAULT_PLUGINS"] == "1"
    assert environment["AEON_OPENCODE_AUTHORITY_FILE"] == str(authority)
    assert environment["AEON_COMPUTE_BACKEND"] == "broker"
    assert environment["AEON_FLEET_SOCKET"] == "/run/user/1000/fleet.sock"
    assert environment["AEON_BROWSER_PROFILE"] == "reviewed-profile"
    assert environment["AEON_BROWSER_SESSION_ID"] == "oc-" + "a" * 32
    assert environment["AEON_REMOTE_INSTANCE_ID"] == "b" * 32
    assert environment["AEON_STATE_DIR"] == "/private/worker-state"
    assert environment["NEXUS_SELF_SETTINGS_TOKEN_FILE"] == "/private/token-file"
    assert environment["PYTHONPATH"] == str(Path(__file__).resolve().parents[2])
    for name in (
        "OPENAI_API_KEY",
        "NEXUS_DATABASE_PASSWORD",
        "LD_PRELOAD",
        "NODE_OPTIONS",
        "OPENCODE_CONFIG_CONTENT",
        "OPENCODE_SERVER_PASSWORD",
    ):
        assert name not in environment
    assert environment["OPENCODE_CONFIG"] == str(config_path)
    assert environment["CUDA_VISIBLE_DEVICES"] == "void"
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"


def test_turn_runner_propagates_browser_profile_and_stable_worker_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.chmod(tmp_path, 0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("AEON_REMOTE_INSTANCE_ID", "a" * 32)
    monkeypatch.setenv("AEON_BROWSER_SESSION_ID", "oc-" + "b" * 32)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", "must-not-override")
    state_root = tmp_path / "opencode-state"
    state_root.mkdir(mode=0o700)
    runner = OpenCodeTurnRunner(
        binary=tmp_path / "opencode",
        root=state_root,
        proxy=SimpleNamespace(
            base_url="http://127.0.0.1:19001/v1",
            token="test-token",
            wire_model="Qwen3.8-27B-ARA-NVFP4-MTP",
        ),
        logical_model=AEON_DEFAULT_MODEL_NAME,
        max_steps=4,
        resume=False,
        browser_profile="profile with spaces",
    )

    environment = runner._environment("Inspect this workspace")

    assert environment["AEON_BROWSER_PROFILE"] == "profile-with-spaces"
    assert environment["AEON_BROWSER_SESSION_ID"] == "oc-" + "b" * 32
    assert environment["AEON_OPENCODE_INSTANCE_ID"] == "a" * 32
    assert "OPENAI_API_KEY" not in environment
    assert "OPENCODE_CONFIG_CONTENT" not in environment


def test_direct_turn_runner_keeps_reviewed_runtime_config_but_not_shell_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.chmod(tmp_path, 0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("AEON_REMOTE_INSTANCE_ID", raising=False)
    monkeypatch.setenv("AEON_COMPUTE_BACKEND", "broker")
    monkeypatch.setenv("AEON_FLEET_PROFILE", "aeon-qwen38-standard")
    monkeypatch.setenv("AEON_STATE_DIR", str(tmp_path / "parent-state"))
    monkeypatch.setenv("AEON_BROWSER_TOKEN_FILE", str(tmp_path / "browser-token"))
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    state_root = tmp_path / "opencode-state"
    state_root.mkdir(mode=0o700)
    runner = OpenCodeTurnRunner(
        binary=tmp_path / "opencode",
        root=state_root,
        proxy=SimpleNamespace(
            base_url="http://127.0.0.1:19001/v1",
            token="test-token",
            wire_model="Qwen3.8-27B-ARA-NVFP4-MTP",
        ),
        logical_model=AEON_DEFAULT_MODEL_NAME,
        max_steps=4,
        resume=False,
        browser_profile="direct-profile",
    )

    environment = runner._environment("Inspect this workspace")

    assert environment["AEON_COMPUTE_BACKEND"] == "broker"
    assert environment["AEON_FLEET_PROFILE"] == "aeon-qwen38-standard"
    assert environment["AEON_BROWSER_TOKEN_FILE"] == str(tmp_path / "browser-token")
    assert environment["AEON_BROWSER_PROFILE"] == "direct-profile"
    assert environment["AEON_OPENCODE_PROXY_TOKEN"] == "test-token"
    assert environment["AEON_STATE_DIR"] == str(tmp_path / "parent-state")
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "GITHUB_TOKEN" not in environment


def test_direct_processes_in_one_workspace_get_disjoint_private_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    state = tmp_path / "state"
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("AEON_CHAT_TRANSCRIPT_PATH", raising=False)
    monkeypatch.delenv("AEON_REMOTE_INSTANCE_ID", raising=False)
    monkeypatch.setenv("AEON_STATE_DIR", str(state))
    identities = iter(("a" * 32, "b" * 32))
    monkeypatch.setattr(
        "aeon.harnesses.opencode_runtime.process_instance_id",
        lambda: next(identities),
    )

    first = _state_root()
    second = _state_root()

    assert first != second
    assert first.parent == second.parent
    assert first.name == "a" * 32
    assert second.name == "b" * 32
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert stat.S_IMODE(second.stat().st_mode) == 0o700


def test_turn_runner_terminates_child_when_stream_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aeon.harnesses import opencode_runtime

    os.chmod(tmp_path, 0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("AEON_REMOTE_INSTANCE_ID", "c" * 32)
    real_popen = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def launch_sleep(*_args, **_kwargs):
        child = real_popen(
            ["/usr/bin/sleep", "60"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        children.append(child)
        return child

    class FailingSelector:
        def register(self, *_args, **_kwargs):
            raise OSError("selector setup failed")

        def close(self):
            return None

    monkeypatch.setattr(opencode_runtime.subprocess, "Popen", launch_sleep)
    monkeypatch.setattr(opencode_runtime.selectors, "DefaultSelector", FailingSelector)
    state_root = tmp_path / "opencode-state"
    state_root.mkdir(mode=0o700)
    runner = OpenCodeTurnRunner(
        binary=tmp_path / "opencode",
        root=state_root,
        proxy=SimpleNamespace(
            base_url="http://127.0.0.1:19001/v1",
            token="test-token",
            wire_model="Qwen3.8-27B-ARA-NVFP4-MTP",
        ),
        logical_model=AEON_DEFAULT_MODEL_NAME,
        max_steps=4,
        resume=False,
    )

    with pytest.raises(OSError, match="selector setup failed"):
        runner.run("Inspect this workspace")

    assert len(children) == 1
    assert children[0].poll() is not None


def test_fresh_mcp_process_restores_reviewed_worker_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aeon.harnesses.opencode_mcp import _build_worker

    os.chmod(tmp_path, 0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    state = tmp_path / "worker-state"
    state.mkdir(mode=0o700)
    bridge = tmp_path / "bridge"
    bridge.mkdir(mode=0o700)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("AEON_STATE_DIR", str(state))
    monkeypatch.setenv("AEON_OPENCODE_PROXY_URL", "http://127.0.0.1:19001/v1")
    monkeypatch.setenv("AEON_OPENCODE_PROXY_TOKEN", "test-token")
    monkeypatch.setenv("AEON_OPENCODE_LOGICAL_MODEL", AEON_DEFAULT_MODEL_NAME)
    monkeypatch.setenv(
        "AEON_OPENCODE_WIRE_MODEL", "Qwen3.8-27B-ARA-NVFP4-MTP"
    )
    monkeypatch.setenv("AEON_OPENCODE_INSTANCE_ID", "d" * 32)

    authority = materialize_authority(bridge, "Remember a project fact")
    monkeypatch.setenv("AEON_OPENCODE_AUTHORITY_FILE", str(authority))
    first, _tools = _build_worker()
    receipts, _interrupted, _restart = first._execute_protocol_actions(
        {
            "intent": "remember the fact",
            "actions": [
                {
                    "tool_name": "memorize",
                    "parameters": {
                        "key": "project-color",
                        "value": "blue",
                        "scope": "project",
                    },
                }
            ],
        },
        1,
    )
    assert receipts[0].successful
    first._persist_session_state()

    materialize_authority(bridge, "List the saved project facts")
    second, _tools = _build_worker()

    assert second.instance_id == "d" * 32
    assert second.memories["project-color"]["value"] == "blue"


def test_console_front_door_defaults_to_opencode_and_preserves_legacy_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aeon import cli as console_cli
    from aeon.harnesses import opencode_runtime
    from aeon import main as legacy_runtime

    opencode_calls: list[list[str]] = []
    legacy_calls: list[list[str]] = []
    adoption_calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(opencode_runtime, "main", lambda argv: opencode_calls.append(argv) or 0)
    monkeypatch.setattr(legacy_runtime, "cli", lambda argv: legacy_calls.append(argv) or 0)
    monkeypatch.setattr(
        legacy_runtime,
        "_auto_adopt_tmux",
        lambda _options, *, cli_args, harness: adoption_calls.append(
            (list(cli_args), harness)
        )
        or False,
    )

    assert console_cli.main(["-n", "--start", "inspect"]) == 0
    assert opencode_calls == [[
        "--model",
        AEON_DEFAULT_MODEL_NAME,
        "--start",
        "inspect",
        "--non-interactive",
    ]]
    assert legacy_calls == []
    assert adoption_calls == [(opencode_calls[0], "opencode")]

    assert console_cli.main(
        ["--harness", "legacy-aeon", "-n", "--start", "inspect"]
    ) == 0
    assert legacy_calls == [["-n", "--start", "inspect"]]

    assert console_cli.main(
        ["--harness=legacy-aeon", "-n", "--start", "inspect again"]
    ) == 0
    assert legacy_calls[-1] == ["-n", "--start", "inspect again"]


class _UpstreamHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["content-length"])
        document = json.loads(self.rfile.read(length))
        type(self).requests.append(document)
        body = json.dumps(
            {
                "id": "test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


class _FakeFleetSession:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.runtime_profiles = ("aeon-qwen38-standard",)
        self.guards = 0
        self.handler = None

    def set_endpoint_change_handler(self, handler) -> None:
        self.handler = handler

    def ensure_ready(self) -> None:
        self.guards += 1


def test_model_proxy_authenticates_guards_and_rewrites_model() -> None:
    _UpstreamHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    fleet = _FakeFleetSession(f"http://127.0.0.1:{upstream.server_port}/v1")
    proxy = FleetModelProxy(fleet)
    proxy.start()
    try:
        refused = requests.get(proxy.base_url + "/models", timeout=5)
        assert refused.status_code == 401
        accepted = requests.post(
            proxy.base_url + "/chat/completions",
            headers={"Authorization": f"Bearer {proxy.token}"},
            json={"model": "model-controlled-value", "messages": []},
            timeout=5,
        )
        assert accepted.status_code == 200
        assert fleet.guards == 1
        assert _UpstreamHandler.requests[-1]["model"] == "Qwen3.8-27B-ARA-NVFP4-MTP"
    finally:
        proxy.close()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_browser_session_can_follow_stable_supervisor_identity(monkeypatch) -> None:
    from aeon.tools import browser

    monkeypatch.setenv("AEON_BROWSER_SESSION_ID", "oc-" + "a" * 32)
    assert browser._session_id() == "oc-" + "a" * 32
    monkeypatch.setenv("AEON_BROWSER_SESSION_ID", "../unsafe")
    assert browser._session_id() == str(os.getpid())
