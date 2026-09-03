"""Hermetic checks for Aeon's explicit tool compute-route manifest."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from aeon.core.agent_protocol import RequestContract, RequestMode
from aeon.core.tool_resources import (
    ToolComputeRoute,
    declared_tool_names,
    tool_resource_policy,
)
from aeon.core.worker import Worker
from aeon.remote.project_manager import PROJECT_MANAGER_INSTANCE_ID
from aeon.remote.self_settings import (
    SELF_SETTINGS_TOKEN_FILE_ENV,
    SELF_SETTINGS_URL_ENV,
)
from aeon.tools.base import BaseTool
from aeon.tools.analyzers.sub_agent_monitor import SubAgentMonitor
from aeon.tools.loader import load_tools_from_directory


class _LLM:
    model = "fixture"
    context_limit = 100_000
    last_reasoning_content = ""
    last_generation_performance = None

    def set_action_schema(self, _schema):
        pass


class ToolResourceManifestTests(unittest.TestCase):
    def test_legacy_nested_analyzer_is_explicitly_non_model_facing(self):
        monitor = SubAgentMonitor(worker=types.SimpleNamespace())
        self.assertTrue(monitor.is_internal)
        self.assertNotIn(monitor.name, declared_tool_names())

    def test_project_manager_inventory_is_fully_declared(self):
        llm = _LLM()
        worker = types.SimpleNamespace(llm_client=llm)
        errors: list[str] = []
        with tempfile.TemporaryDirectory() as external_state:
            with patch.dict(
                "os.environ",
                {
                    "AEON_REMOTE_INSTANCE_ID": PROJECT_MANAGER_INSTANCE_ID,
                    SELF_SETTINGS_URL_ENV: (
                        "http://127.0.0.1:8765/internal/agent/job-role"
                    ),
                    SELF_SETTINGS_TOKEN_FILE_ENV: "/owner/private/self-settings.token",
                    # Pin the one opt-in tool instead of inheriting an owner's
                    # persisted external-expert setting from ~/.aeon.
                    "AEON_EXTERNAL_EXPERT_ENABLED": "1",
                    "AEON_EXTERNAL_EXPERT_STATE_DIR": external_state,
                },
                clear=False,
            ):
                tools = load_tools_from_directory(
                    "aeon.tools",
                    dependencies={"worker": worker, "llm_client": llm},
                    errors_out=errors,
                )
        names = {tool.name for tool in tools}

        self.assertFalse(errors)
        self.assertEqual(len(tools), len(names))
        self.assertEqual(len(names), 82)
        self.assertIn("integrate_sub_agent_changes", names)
        # These capability-gated tools are declared in the resource manifest
        # but are absent unless their exact launch-bound identity is active.
        self.assertEqual(
            names,
            declared_tool_names()
            - {"benchmark_workflow", "send_collaborator_handoff"},
        )
        for tool in tools:
            self.assertEqual(tool.resource_policy, tool_resource_policy(tool.name))

        # This is the exact loader -> registration shape used by main.py. A
        # redundant manual append must fail this regression instead of aborting a
        # real agent during startup with an ambiguous executable capability.
        runtime_worker = Worker(llm_client=llm, print_func=lambda *_args: None)
        runtime_worker.persist_session = False
        runtime_worker.register_tools(tools)
        self.assertEqual(set(runtime_worker.tools), names)

    def test_unknown_tool_fails_closed_before_registration(self):
        class UnreviewedTool(BaseTool):
            def __init__(self):
                super().__init__("unreviewed_accelerator", "fixture")

            def execute(self):
                return "should not run"

        tool = UnreviewedTool()
        worker = Worker(llm_client=_LLM(), print_func=lambda *_args: None)
        with self.assertRaisesRegex(ValueError, "no reviewed compute-route"):
            worker.register_tools([tool])

        with self.assertRaisesRegex(ValueError, "no reviewed compute-route"):
            Worker(llm_client=_LLM(), tools=[tool], print_func=lambda *_args: None)

    def test_duplicate_worker_tool_name_fails_closed(self):
        class ReviewedTool(BaseTool):
            def __init__(self):
                super().__init__("open_file", "fixture")

            def execute(self, path: str):
                return path

        first = ReviewedTool()
        second = ReviewedTool()
        with self.assertRaisesRegex(ValueError, "duplicate tool name 'open_file'"):
            Worker(
                llm_client=_LLM(),
                tools=[first, second],
                print_func=lambda *_args: None,
            )

        worker = Worker(
            llm_client=_LLM(), tools=[ReviewedTool()], print_func=lambda *_args: None
        )
        with self.assertRaisesRegex(ValueError, "duplicate tool name 'open_file'"):
            worker.register_tools([ReviewedTool()])

    def test_duplicate_loader_tool_name_removes_both_implementations(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "duplicate_tools"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            source = (
                "from aeon.tools.base import BaseTool\n"
                "class {class_name}(BaseTool):\n"
                "    def __init__(self):\n"
                "        super().__init__('open_file', 'fixture')\n"
                "    def execute(self, path: str):\n"
                "        return path\n"
            )
            (package / "first.py").write_text(
                source.format(class_name="FirstTool"), encoding="utf-8"
            )
            (package / "second.py").write_text(
                source.format(class_name="SecondTool"), encoding="utf-8"
            )
            sys.path.insert(0, temporary)
            try:
                importlib.invalidate_caches()
                errors: list[str] = []
                tools = load_tools_from_directory(
                    "duplicate_tools", dependencies={}, errors_out=errors
                )
            finally:
                sys.path.remove(temporary)
                for module_name in list(sys.modules):
                    if module_name == "duplicate_tools" or module_name.startswith(
                        "duplicate_tools."
                    ):
                        del sys.modules[module_name]

        self.assertEqual(tools, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("duplicate tool name 'open_file'", errors[0])
        self.assertIn("refusing all implementations", errors[0])

    def test_auto_external_consult_uses_runtime_compute_preflight(self):
        class RecordingExpert(BaseTool):
            def __init__(self):
                super().__init__("consult_external_expert", "fixture")
                self.calls = 0

            def execute(self, **_kwargs):
                self.calls += 1
                return "expert result"

        llm = _LLM()
        llm.provider = "vllm"
        expert = RecordingExpert()
        worker = Worker(llm_client=llm, tools=[expert], print_func=lambda *_args: None)
        worker.persist_session = False
        worker.model_config = {"provider": "vllm"}
        worker.request_contract = RequestContract.from_request(
            "Consult the external expert now.",
            forced_mode=RequestMode.EXTERNAL_ACTION,
        )
        worker._failures_since_external_consult = 2

        with patch.dict(os.environ, {"AEON_AUTO_EXTERNAL_CONSULT": "1"}):
            blocked = worker._maybe_auto_consult_external(
                objective="repair", intent="retry", actions=["run_command"],
                latest_result="failed",
            )
        self.assertIn("no Fleet ticket guard", blocked)
        self.assertEqual(expert.calls, 0)
        self.assertEqual(worker._failures_since_external_consult, 2)

        guard_calls = []
        worker.compute_guard = lambda: guard_calls.append("ready")
        worker._failures_since_external_consult = 2
        with patch.dict(os.environ, {"AEON_AUTO_EXTERNAL_CONSULT": "1"}):
            result = worker._maybe_auto_consult_external(
                objective="repair", intent="retry", actions=["run_command"],
                latest_result="failed",
            )
        self.assertEqual(result, "expert result")
        self.assertEqual(guard_calls, ["ready"])
        self.assertEqual(expert.calls, 1)

    def test_routes_name_gpu_backed_tools_explicitly(self):
        self.assertEqual(
            tool_resource_policy("generate_video").fleet_service,
            "aeon-video-comfyui",
        )
        self.assertEqual(
            tool_resource_policy("analyze_image").route,
            ToolComputeRoute.ACTIVE_MODEL,
        )
        self.assertEqual(
            tool_resource_policy("spawn_sub_agent").route,
            ToolComputeRoute.FLEET_CHILD,
        )
        self.assertEqual(
            tool_resource_policy("run_command").route,
            ToolComputeRoute.DYNAMIC_COMMAND,
        )
        self.assertEqual(
            tool_resource_policy("browser_navigate").route,
            ToolComputeRoute.HOST_SERVICE,
        )
        self.assertEqual(
            tool_resource_policy("browser_navigate").host_service,
            "aeon-browser",
        )
        search = tool_resource_policy("search_web")
        self.assertEqual(search.route, ToolComputeRoute.HOST_SERVICE)
        self.assertEqual(search.host_service, "aeon-searxng")
        self.assertTrue(search.requires_primary_compute_guard)
        for name in (
            "huggingface_model_search",
            "huggingface_model_info",
            "huggingface_repo_file",
        ):
            self.assertEqual(
                tool_resource_policy(name).route,
                ToolComputeRoute.EXTERNAL_PROVIDER,
            )
            self.assertFalse(
                tool_resource_policy(name).requires_primary_compute_guard
            )
        self.assertEqual(
            tool_resource_policy("set_job_role").route,
            ToolComputeRoute.NEXUS_LIFECYCLE,
        )
        for name in (
            "fleet_batch_capabilities",
            "fleet_submit_batch_job",
            "fleet_batch_job_status",
        ):
            self.assertEqual(
                tool_resource_policy(name).route,
                ToolComputeRoute.FLEET_BATCH,
            )

    def test_local_model_tool_requires_and_calls_compute_guard(self):
        worker = Worker(llm_client=_LLM(), print_func=lambda *_args: None)
        worker.persist_session = False
        worker.model_config = {"provider": "vllm"}
        tool = types.SimpleNamespace(
            name="analyze_image",
            resource_policy=tool_resource_policy("analyze_image")
        )

        self.assertIn("no Fleet ticket guard", worker._tool_resource_error(tool))
        calls = []
        worker.compute_guard = lambda: calls.append("ready")
        self.assertEqual(worker._tool_resource_error(tool), "")
        self.assertEqual(calls, ["ready"])

        worker.model_config = {"provider": "llamacpp"}
        self.assertEqual(worker._tool_resource_error(tool), "")
        self.assertEqual(calls, ["ready", "ready"])

    def test_external_model_does_not_require_owner_fleet_ticket(self):
        worker = Worker(llm_client=_LLM(), print_func=lambda *_args: None)
        worker.persist_session = False
        worker.model_config = {"provider": "openai"}
        tool = types.SimpleNamespace(
            name="think",
            resource_policy=tool_resource_policy("think")
        )
        self.assertEqual(worker._tool_resource_error(tool), "")

    def test_missing_unknown_or_inconsistent_provider_fails_closed(self):
        worker = Worker(llm_client=_LLM(), print_func=lambda *_args: None)
        tool = types.SimpleNamespace(
            name="think",
            resource_policy=tool_resource_policy("think"),
        )

        worker.model_config = {}
        self.assertIn("provider is missing", worker._tool_resource_error(tool))
        worker.model_config = {"provider": "mystery-local-runtime"}
        self.assertIn("no reviewed", worker._tool_resource_error(tool))
        worker.model_config = {"provider": "openai"}
        worker.llm_client.provider = "vllm"
        self.assertIn("providers disagree", worker._tool_resource_error(tool))

    def test_runtime_route_tampering_is_blocked(self):
        worker = Worker(llm_client=_LLM(), print_func=lambda *_args: None)
        tool = types.SimpleNamespace(
            name="generate_video",
            resource_policy=tool_resource_policy("open_file"),
        )
        self.assertIn("does not match", worker._tool_resource_error(tool))

    def test_prompt_label_exposes_enforced_route(self):
        tool = types.SimpleNamespace(name="generate_image")
        self.assertEqual(
            Worker._tool_resource_label(tool),
            "[compute-route: fleet_service:aeon-comfyui]",
        )


if __name__ == "__main__":
    unittest.main()
