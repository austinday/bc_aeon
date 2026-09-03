"""Hermetic tests for the opt-in external expert tool; no network calls occur."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aeon.core.agent_protocol import RequestContract, RequestMode
from aeon.core.llm import LLMClient
from aeon.core.worker import Worker
from aeon.main import select_model
from aeon.tools.loader import load_tools_from_directory
from aeon.tools.external_expert import (
    ConsultExternalExpertTool,
    ExternalExpertConfig,
    load_external_expert_settings,
    redact_sensitive,
    save_external_expert_settings,
)
from aeon.core.external_expert_setup import configure_external_expert_interactive


class FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Try invariant X."))],
            usage=SimpleNamespace(total_tokens=321),
        )


class FakeClient:
    def __init__(self):
        self.request = None
        self.chat = SimpleNamespace(completions=FakeCompletions(self))


class FakeCommandRunner:
    def __init__(self, stdout="Advisory answer", returncode=0, last_message=None):
        self.stdout = stdout
        self.returncode = returncode
        self.last_message = last_message
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        if self.last_message is not None and "--output-last-message" in args:
            path = Path(args[args.index("--output-last-message") + 1])
            path.write_text(self.last_message, encoding="utf-8")
        return subprocess.CompletedProcess(
            args, self.returncode, self.stdout, "" if self.returncode == 0 else "failed"
        )


class FakeLocalDisclosureReviewer:
    def __init__(self, decision="ALLOW", reason="No sensitive content detected."):
        self.decision = decision
        self.reason = reason
        self.prompts = []

    def review_external_disclosure(self, candidate_prompt):
        self.prompts.append(candidate_prompt)
        return {"decision": self.decision, "reason": self.reason}


class FakeLocalReviewCompletions:
    def __init__(self, content):
        self.content = content
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.content)
        )])


class TestExternalExpert(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.secret = "sk-this-is-a-test-secret-123456"
        self.config = ExternalExpertConfig(
            enabled=True,
            model="strong-test-model",
            base_url="https://expert.invalid/v1",
            api_key_env="TEST_EXPERT_API_KEY",
            state_dir=Path(self.temp.name),
            max_calls_per_run=1,
            max_calls_per_day=2,
            max_total_tokens_per_day=20000,
            max_input_chars=4000,
            max_output_tokens=1000,
        )
        self.local_reviewer = FakeLocalDisclosureReviewer()
        self.worker = SimpleNamespace(
            stuck_reason="three approaches failed",
            _stuck_banner="STUCK",
            _loop_blocked_fingerprint="x",
            _no_progress_streak=3,
            llm_client=self.local_reviewer,
        )
        self.client = FakeClient()
        self.env = patch.dict(os.environ, {"TEST_EXPERT_API_KEY": self.secret})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def tool(self, **overrides):
        config = overrides.pop("config", self.config)
        worker = overrides.pop("worker", self.worker)
        return ConsultExternalExpertTool(
            worker=worker,
            config=config,
            client_factory=lambda **kwargs: self.client,
            **overrides,
        )

    def test_disabled_configuration_is_hidden(self):
        disabled = ExternalExpertConfig(
            enabled=False,
            model="",
            base_url="",
            api_key_env="TEST_EXPERT_API_KEY",
            state_dir=Path(self.temp.name),
        )
        self.assertTrue(self.tool(config=disabled).is_internal)

    def test_api_endpoint_must_be_public_https(self):
        for endpoint in (
            "http://expert.example/v1",
            "https://127.0.0.1/v1",
            "https://10.42.0.9/v1",
            "https://[::1]/v1",
            "https://localhost/v1",
            "https://service.internal/v1",
            "https://user@example.com/v1",
            "https://expert.example:8443/v1",
            "https://expert.example:not-a-port/v1",
        ):
            with self.subTest(endpoint=endpoint):
                config = ExternalExpertConfig(
                    enabled=True,
                    model="strong-test-model",
                    base_url=endpoint,
                    api_key_env="TEST_EXPERT_API_KEY",
                    state_dir=Path(self.temp.name),
                    allow_insecure_http=True,
                )
                self.assertIsNotNone(config.problem())

        self.assertIsNone(self.config.problem())

    def test_production_cli_requires_a_safe_official_executable_identity(self):
        executable = Path(self.temp.name) / "codex"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o777)
        config = ExternalExpertConfig(
            enabled=True,
            model="gpt-5.6-sol",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="codex",
            executable=str(executable),
            reasoning_effort="high",
        )
        self.assertIn("identity is unsafe", config.problem())
        executable.chmod(0o700)
        self.assertIsNone(config.problem())

    def test_production_api_dns_must_resolve_only_to_public_addresses(self):
        tool = ConsultExternalExpertTool(worker=self.worker, config=self.config)
        private_answer = [
            (2, 1, 6, "", ("127.0.0.1", 443)),
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]
        with patch("aeon.tools.external_expert.socket.getaddrinfo", return_value=private_answer):
            self.assertIn("local/private", tool._resolved_endpoint_problem())
        public_answer = [(2, 1, 6, "", ("93.184.216.34", 443))]
        with patch("aeon.tools.external_expert.socket.getaddrinfo", return_value=public_answer):
            self.assertIsNone(tool._resolved_endpoint_problem())

    def test_loader_exposes_tool_only_after_explicit_enablement(self):
        with patch.dict(os.environ, {"AEON_EXTERNAL_EXPERT_ENABLED": "0"}):
            disabled_names = {
                tool.name for tool in load_tools_from_directory(
                    "aeon.tools", dependencies={"worker": self.worker}
                )
            }
        configured = {
            "AEON_EXTERNAL_EXPERT_ENABLED": "1",
            "AEON_EXTERNAL_EXPERT_MODEL": "strong-test-model",
            "AEON_EXTERNAL_EXPERT_BASE_URL": "https://expert.invalid/v1",
            "AEON_EXTERNAL_EXPERT_API_KEY_ENV": "TEST_EXPERT_API_KEY",
            "AEON_EXTERNAL_EXPERT_STATE_DIR": self.temp.name,
        }
        with patch.dict(os.environ, configured):
            enabled_names = {
                tool.name for tool in load_tools_from_directory(
                    "aeon.tools", dependencies={"worker": self.worker}
                )
            }
        self.assertNotIn("consult_external_expert", disabled_names)
        self.assertIn("consult_external_expert", enabled_names)

    def test_refuses_before_stall_by_default(self):
        clear_worker = SimpleNamespace(
            stuck_reason=None,
            _stuck_banner="",
            _loop_blocked_fingerprint=None,
            _no_progress_streak=0,
        )
        result = self.tool(worker=clear_worker).execute("p", "a", "q")
        self.assertIn("stall detector has not fired", result)
        self.assertIsNone(self.client.request)

    def test_two_local_failures_open_the_external_expert_gate(self):
        failed_worker = SimpleNamespace(
            stuck_reason=None,
            _stuck_banner="",
            _loop_blocked_fingerprint=None,
            _no_progress_streak=0,
            _failures_since_external_consult=2,
            llm_client=self.local_reviewer,
        )
        result = self.tool(worker=failed_worker).execute("p", "a", "q")
        self.assertIn("EXTERNAL EXPERT ADVICE", result)
        self.assertIsNotNone(self.client.request)

    def test_redacts_credentials_and_records_only_usage_metadata(self):
        result = self.tool().execute(
            problem=f"Service fails with token={self.secret}",
            attempts="Tried a restart and sk-another-secret-987654321.",
            question="What invariant should I inspect?",
        )
        self.assertIn("EXTERNAL EXPERT ADVICE", result)
        request_text = self.client.request["messages"][1]["content"]
        self.assertIs(self.client.request["store"], False)
        self.assertNotIn(self.secret, request_text)
        self.assertNotIn("sk-another-secret", request_text)
        usage = json.loads(self.config.usage_path.read_text(encoding="utf-8"))
        self.assertEqual(usage[0]["tokens"], 321)
        serialized = json.dumps(usage)
        self.assertNotIn("Service fails", serialized)
        self.assertNotIn(self.secret, serialized)
        self.assertEqual(self.config.usage_path.stat().st_mode & 0o777, 0o600)

    def test_private_data_requires_operator_opt_in(self):
        result = self.tool().execute("p", "a", "q", sensitivity="private")
        self.assertIn("Private data transmission is disabled", result)
        self.assertIsNone(self.client.request)

    def test_local_model_blocks_sensitive_prompt_before_external_call(self):
        self.local_reviewer.decision = "BLOCK"
        self.local_reviewer.reason = "The task may need an uncensored model."
        result = self.tool().execute(
            problem="Analyze refusal-sensitive material",
            attempts="The local uncensored model made two attempts",
            question="What should happen next?",
        )
        self.assertIn("EXTERNAL DISCLOSURE BLOCKED BY LOCAL MODEL", result)
        self.assertIn("Continue troubleshooting with the local uncensored model", result)
        self.assertIsNone(self.client.request)
        self.assertEqual(len(self.local_reviewer.prompts), 1)
        self.assertFalse(self.config.usage_path.exists())

    def test_local_review_sees_exact_redacted_outbound_prompt(self):
        self.tool().execute(
            problem=f"Failure with token={self.secret}",
            attempts="Tried locally",
            question="What next?",
        )
        reviewed = self.local_reviewer.prompts[0]
        sent = self.client.request["messages"][1]["content"]
        self.assertEqual(reviewed, sent)
        self.assertNotIn(self.secret, reviewed)

    def test_ambiguous_or_failed_local_review_blocks_fail_closed(self):
        ambiguous = self.tool(local_reviewer=lambda _prompt: {
            "decision": "MAYBE", "reason": "uncertain"
        })
        result = ambiguous.execute("p", "a", "q")
        self.assertIn("EXTERNAL DISCLOSURE BLOCKED", result)
        self.assertIsNone(self.client.request)

        def broken_reviewer(_prompt):
            raise RuntimeError("local model unavailable")

        failed = self.tool(local_reviewer=broken_reviewer)
        result = failed.execute("p2", "a2", "q2")
        self.assertIn("local disclosure reviewer failed", result)
        self.assertIsNone(self.client.request)

    def test_missing_local_reviewer_blocks_fail_closed(self):
        worker = SimpleNamespace(
            stuck_reason="stuck",
            _stuck_banner="STUCK",
            _loop_blocked_fingerprint="x",
            _no_progress_streak=3,
        )
        result = self.tool(worker=worker).execute("p", "a", "q")
        self.assertIn("No local disclosure reviewer is available", result)
        self.assertIsNone(self.client.request)

    def test_local_classifier_requires_explicit_valid_json_allow(self):
        completions = FakeLocalReviewCompletions(
            json.dumps({"decision": "ALLOW", "reason": "ordinary debugging"})
        )
        llm = object.__new__(LLMClient)
        llm.api_model = "local-qwen"
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        llm.logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
        decision = llm.review_external_disclosure("candidate text")
        self.assertEqual(decision["decision"], "ALLOW")
        request = completions.request
        self.assertEqual(request["model"], "local-qwen")
        self.assertEqual(request["temperature"], 0)
        review_prompt = request["messages"][0]["content"]
        self.assertIn("needs an uncensored model", review_prompt)
        self.assertIn("large technology company", review_prompt)
        self.assertIn("candidate text", review_prompt)

        completions.content = "probably okay"
        decision = llm.review_external_disclosure("candidate text")
        self.assertEqual(decision["decision"], "BLOCK")

    def test_per_run_budget_stops_second_call(self):
        tool = self.tool()
        self.assertIn("EXTERNAL EXPERT ADVICE", tool.execute("p", "a", "q"))
        result = tool.execute("different", "attempt", "question")
        self.assertIn("Per-run", result)

    def test_daily_reservation_budget_is_enforced(self):
        first = self.tool()
        second = self.tool()
        third = self.tool()
        self.assertIn("EXTERNAL EXPERT ADVICE", first.execute("p1", "a", "q"))
        self.assertIn("EXTERNAL EXPERT ADVICE", second.execute("p2", "a", "q"))
        self.assertIn("daily external-expert call budget", third.execute("p3", "a", "q"))

    def test_generic_redaction(self):
        value = redact_sensitive("Authorization: Bearer abcdefghijklmnop password=hunter22")
        self.assertNotIn("abcdefghijklmnop", value)
        self.assertNotIn("hunter22", value)

    def test_persistent_provider_choice_contains_no_credentials(self):
        save_external_expert_settings(
            Path(self.temp.name),
            {
                "enabled": True,
                "backend": "codex",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
                "credential": self.secret,
            },
        )
        stored = load_external_expert_settings(Path(self.temp.name))
        self.assertEqual(stored["backend"], "codex")
        self.assertEqual(stored["reasoning_effort"], "high")
        self.assertNotIn("credential", stored)
        self.assertNotIn(self.secret, json.dumps(stored))
        self.assertEqual((Path(self.temp.name) / "config.json").stat().st_mode & 0o777, 0o600)

    def test_codex_subscription_adapter_is_ephemeral_and_disables_tools(self):
        config = ExternalExpertConfig(
            enabled=True,
            model="gpt-5.6-sol",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="codex",
            executable="/test/codex",
            reasoning_effort="high",
            max_output_tokens=1000,
        )
        runner = FakeCommandRunner()
        tool = ConsultExternalExpertTool(
            worker=self.worker, config=config, command_runner=runner
        )
        result = tool.execute("hard problem", "three failures", "what assumption?")
        self.assertIn("Advisory answer", result)
        args, kwargs = runner.calls[0]
        self.assertEqual(args[0], "/home/aday/bin/fleet-low-priority")
        self.assertEqual(args[1:3], ["/test/codex", "exec"])
        self.assertIn("--ephemeral", args)
        self.assertIn("--ignore-user-config", args)
        self.assertIn("read-only", args)
        self.assertEqual(args[args.index("--model") + 1], "gpt-5.6-sol")
        self.assertEqual(
            args[args.index("--config") + 1], 'model_reasoning_effort="high"'
        )
        self.assertEqual(args[-1], "-")
        self.assertIn("PROBLEM\nhard problem", kwargs["input"])
        self.assertGreaterEqual(args.count("--disable"), 7)
        self.assertNotIn(self.secret, kwargs["env"].values())
        self.assertTrue(Path(kwargs["cwd"]).is_dir())

    def test_subscription_cli_cannot_inherit_local_compute_authority(self):
        config = ExternalExpertConfig(
            enabled=True,
            model="gpt-5.6-sol",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="codex",
            executable="/test/codex",
            reasoning_effort="high",
            max_output_tokens=1000,
        )
        runner = FakeCommandRunner()
        with patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "GPU-principal",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "GPU_AGENT_CLAIM_ID": "claim-principal",
                "GPU_MEM_LIMIT_GB": "48",
                "AEON_FLEET_TICKET": "fd-secret",
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
                "LD_PRELOAD": "/tmp/untrusted.so",
                "PYTHONPATH": "/tmp/untrusted-python",
            },
            clear=False,
        ):
            ConsultExternalExpertTool(
                worker=self.worker, config=config, command_runner=runner
            ).execute("hard problem", "three failures", "what assumption?")

        environment = runner.calls[0][1]["env"]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "void")
        self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], "void")
        for key in (
            "GPU_AGENT_CLAIM_ID",
            "GPU_MEM_LIMIT_GB",
            "AEON_FLEET_TICKET",
            "DBUS_SESSION_BUS_ADDRESS",
            "LD_PRELOAD",
            "PYTHONPATH",
        ):
            self.assertNotIn(key, environment)

    def test_codex_uses_final_message_file_and_jsonl_usage(self):
        config = ExternalExpertConfig(
            enabled=True,
            model="gpt-5.6-sol",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="codex",
            executable="/test/codex",
            reasoning_effort="high",
            max_output_tokens=1000,
        )
        stdout = "\n".join([
            json.dumps({
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "JSONL fallback"},
            }),
            json.dumps({
                "type": "turn.completed",
                "usage": {"input_tokens": 120, "output_tokens": 30},
            }),
        ])
        runner = FakeCommandRunner(
            stdout=stdout,
            last_message="Authoritative final advice",
        )
        result = ConsultExternalExpertTool(
            worker=self.worker, config=config, command_runner=runner
        ).execute("hard problem", "three failures", "what assumption?")
        self.assertIn("Authoritative final advice", result)
        self.assertNotIn("JSONL fallback", result)
        args, _kwargs = runner.calls[0]
        self.assertIn("--json", args)
        output_path = Path(args[args.index("--output-last-message") + 1])
        self.assertFalse(output_path.exists())
        usage = json.loads(config.usage_path.read_text(encoding="utf-8"))
        self.assertEqual(usage[0]["tokens"], 150)

    def test_worker_consults_only_after_explicit_auto_opt_in_and_authority(self):
        class RecordingExpert:
            name = "consult_external_expert"

            def __init__(self):
                self.calls = []

            def execute(self, **kwargs):
                self.calls.append(kwargs)
                return "EXTERNAL EXPERT ADVICE\nTry a different invariant."

        expert = RecordingExpert()
        worker = Worker(
            SimpleNamespace(context_limit=131072, provider="vllm"),
            tools=[expert],
            print_func=lambda _message: None,
        )
        worker.model_config = {"provider": "vllm"}
        worker.compute_guard = lambda: None
        worker.current_plan = "Reproduce, isolate, then repair."
        worker.action_log = [
            "[Iter 1]\n- Intent: reproduce\n- Actions: run_command\n"
            "- Result: ERROR — first failure"
        ]
        worker.request_contract = RequestContract.from_request(
            "Consult the configured external expert about this failure.",
            forced_mode=RequestMode.EXTERNAL_ACTION,
        )

        # External disclosure/spend is never an invisible default recovery step.
        worker._failures_since_external_consult = 2
        self.assertEqual(worker._maybe_auto_consult_external(
            objective="Fix the failing service",
            intent="Retry the health check",
            actions=["run_command(health check)"],
            latest_result="Error: connection refused",
        ), "")
        self.assertEqual(expert.calls, [])

        worker._failures_since_external_consult = 1
        with patch.dict(os.environ, {"AEON_AUTO_EXTERNAL_CONSULT": "1"}):
            self.assertEqual(worker._maybe_auto_consult_external(
                objective="Fix the failing service",
                intent="Retry the health check",
                actions=["run_command(health check)"],
                latest_result="Error: connection refused",
            ), "")
            worker._failures_since_external_consult = 2
            result = worker._maybe_auto_consult_external(
                objective="Fix the failing service",
                intent="Retry the health check",
                actions=["run_command(health check)"],
                latest_result="Error: connection refused",
            )
        self.assertIn("Try a different invariant", result)
        self.assertEqual(len(expert.calls), 1)
        call = expert.calls[0]
        self.assertIn("Fix the failing service", call["problem"])
        self.assertIn("Retry the health check", call["problem"])
        self.assertIn("Reproduce, isolate, then repair", call["attempts"])
        self.assertIn("first failure", call["attempts"])
        self.assertIn("Error: connection refused", call["attempts"])
        self.assertIn("2 consecutive local failures", call["attempts"])
        self.assertEqual(worker._failures_since_external_consult, 0)

    def test_external_consultation_is_not_task_progress(self):
        worker = object.__new__(Worker)
        actions = [{
            "tool_name": "consult_external_expert",
            "parameters": {"problem": "p", "attempts": "a", "question": "q"},
        }]
        self.assertEqual(worker._consequential_fp(actions), "")

    def test_plain_error_output_counts_as_a_failed_local_turn(self):
        output = "Action 1: run_command\nError: connection refused"
        self.assertTrue(Worker._turn_made_no_progress(output, consequential=True))

    def test_codex_configuration_requires_explicit_model_and_effort(self):
        config = ExternalExpertConfig(
            enabled=True,
            model="",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="codex",
            executable="/test/codex",
        )
        self.assertIn("model has not been selected", config.problem())

    def test_claude_subscription_adapter_has_empty_tool_set(self):
        config = ExternalExpertConfig(
            enabled=True,
            model="",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="claude",
            executable="/test/claude",
            max_output_tokens=1000,
        )
        runner = FakeCommandRunner(stdout=json.dumps({
            "result": "Claude advice", "usage": {"total_tokens": 99}
        }))
        tool = ConsultExternalExpertTool(
            worker=self.worker, config=config, command_runner=runner
        )
        result = tool.execute("hard problem", "three failures", "what assumption?")
        self.assertIn("Claude advice", result)
        args, kwargs = runner.calls[0]
        tools_index = args.index("--tools")
        self.assertEqual(args[tools_index + 1], "")
        self.assertIn("--safe-mode", args)
        self.assertNotIn("ANTHROPIC_API_KEY", kwargs["env"])

    def test_claude_subscription_prompts_never_enter_argv(self):
        config = ExternalExpertConfig(
            enabled=True,
            model="",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="claude",
            executable="/test/claude",
            max_output_tokens=1000,
        )
        observed = {}

        def runner(args, **kwargs):
            prompt_path = Path(args[args.index("--system-prompt-file") + 1])
            observed.update(
                command=list(args),
                kwargs=dict(kwargs),
                path=prompt_path,
                content=prompt_path.read_text(encoding="utf-8"),
                mode=prompt_path.stat().st_mode & 0o777,
            )
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps({"result": "Claude advice", "usage": {"total_tokens": 9}}),
                "",
            )

        system_body = "SYSTEM-PROMPT-SENTINEL private system context"
        user_body = "USER-PROMPT-SENTINEL private failure context"
        tool = ConsultExternalExpertTool(
            worker=self.worker, config=config, command_runner=runner
        )
        answer, tokens = tool._run_cli(system_body, user_body)

        self.assertEqual(answer, "Claude advice")
        self.assertEqual(tokens, 9)
        self.assertEqual(observed["content"], system_body)
        self.assertEqual(observed["mode"], 0o600)
        self.assertEqual(observed["path"].parent, config.state_dir / "adviser_workspace")
        self.assertFalse(observed["path"].exists())
        rendered = "\x00".join(observed["command"])
        self.assertNotIn(system_body, rendered)
        self.assertNotIn(user_body, rendered)
        self.assertEqual(observed["kwargs"]["input"], user_body)

    def test_claude_system_prompt_file_is_removed_when_cli_fails(self):
        config = ExternalExpertConfig(
            enabled=True,
            model="",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="claude",
            executable="/test/claude",
        )
        observed_path = None

        def runner(args, **_kwargs):
            nonlocal observed_path
            observed_path = Path(args[args.index("--system-prompt-file") + 1])
            self.assertTrue(observed_path.is_file())
            raise subprocess.TimeoutExpired(args, 1)

        tool = ConsultExternalExpertTool(
            worker=self.worker, config=config, command_runner=runner
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            tool._run_cli("private system", "private user")
        self.assertIsNotNone(observed_path)
        self.assertFalse(observed_path.exists())

    def test_gemini_subscription_adapter_uses_plan_mode(self):
        config = ExternalExpertConfig(
            enabled=True,
            model="",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="gemini",
            executable="/test/gemini",
            max_output_tokens=1000,
        )
        runner = FakeCommandRunner(stdout=json.dumps({
            "response": "Gemini advice", "stats": {"totalTokens": 88}
        }))
        tool = ConsultExternalExpertTool(
            worker=self.worker, config=config, command_runner=runner
        )
        result = tool.execute("hard problem", "three failures", "what assumption?")
        self.assertIn("Gemini advice", result)
        args, kwargs = runner.calls[0]
        self.assertEqual(args[1], "/test/gemini")
        self.assertEqual(args[args.index("--approval-mode") + 1], "plan")
        self.assertEqual(args[args.index("--output-format") + 1], "json")
        self.assertEqual(args[args.index("--prompt") + 1], "")
        self.assertIn("PROBLEM\nhard problem", kwargs["input"])
        self.assertNotIn("hard problem", "\x00".join(args))

    def test_subscription_advice_is_bounded_before_entering_context(self):
        config = ExternalExpertConfig(
            enabled=True,
            model="gpt-5.6-sol",
            base_url="",
            api_key_env="UNUSED",
            state_dir=Path(self.temp.name),
            backend="codex",
            executable="/test/codex",
            reasoning_effort="low",
            max_output_tokens=128,
        )
        runner = FakeCommandRunner(stdout="x" * 1000)
        result = ConsultExternalExpertTool(
            worker=self.worker, config=config, command_runner=runner
        ).execute("hard problem", "three failures", "what assumption?")
        self.assertIn("[external advice truncated]", result)
        self.assertLess(len(result), 700)

    def test_startup_wizard_reuses_official_codex_login(self):
        runner = FakeCommandRunner(stdout=json.dumps({"models": [
            {
                "slug": "gpt-test-fast",
                "display_name": "GPT Test Fast",
                "description": "Fast test model.",
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [
                    {"effort": "low"}, {"effort": "medium"}, {"effort": "high"}
                ],
                "visibility": "list",
            },
            {
                "slug": "gpt-test-strong",
                "display_name": "GPT Test Strong",
                "description": "Strong test model.",
                "default_reasoning_level": "high",
                "supported_reasoning_levels": [
                    {"effort": "medium"}, {"effort": "high"}, {"effort": "xhigh"}
                ],
                "visibility": "list",
            },
        ]}))
        answers = iter(["1", "", "2", "3"])
        messages = []
        with patch.dict(
            os.environ,
            {"AEON_EXTERNAL_EXPERT_STATE_DIR": self.temp.name},
        ):
            changed = configure_external_expert_interactive(
                input_fn=lambda _prompt: next(answers),
                print_fn=messages.append,
                runner=runner,
                which=lambda name: f"/test/{name}",
            )
        self.assertTrue(changed)
        settings = load_external_expert_settings(Path(self.temp.name))
        self.assertEqual(settings["backend"], "codex")
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["model"], "gpt-test-strong")
        self.assertEqual(settings["reasoning_effort"], "xhigh")
        self.assertTrue(any(call[0][1:3] == ["debug", "models"]
                            for call in runner.calls))
        self.assertFalse(any(call[0][1:3] == ["login", "--device-auth"]
                             for call in runner.calls))

    def test_model_picker_configuration_action_returns_to_local_models(self):
        local = {"label": "Local Qwen", "model": "Qwen-local", "provider": "vllm"}
        menu = [
            {"label": "--- Local Models ---", "is_header": True},
            local,
            {"label": "--- Optional Escalation ---", "is_header": True},
            {"label": "External expert account", "menu_action": "external_expert"},
        ]
        with patch("builtins.input", side_effect=["2", "1"]), patch(
            "aeon.core.external_expert_setup.configure_external_expert_interactive"
        ) as configure, patch("aeon.main.build_model_menu", return_value=menu):
            selected = select_model(menu, "Select Model", default_model="Qwen-local")
        configure.assert_called_once_with()
        self.assertIs(selected, local)


if __name__ == "__main__":
    unittest.main(verbosity=2)
