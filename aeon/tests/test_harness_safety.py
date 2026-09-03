"""Hermetic safety and evidence-boundary regressions for the Aeon harness."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from aeon.core.agent_protocol import (
    RequestContract,
    RequestMode,
    SideEffect,
    ToolStatus,
    effective_tool_effect,
    infer_tool_policy,
    normalize_tool_result,
)
from aeon.core.worker import Worker
from aeon.tools.base import BaseTool
from aeon.tools.file_io import OpenFileTool, StrReplaceTool, WriteFileTool
from aeon.tools.memory import ListMemoriesTool, MemorizeTool


class FakeLLM:
    context_limit = 100_000
    last_reasoning_content = ""

    def set_action_schema(self, schema):
        self.action_schema = schema


def make_worker(*tools):
    worker = Worker(FakeLLM(), tools=list(tools), print_func=lambda *_: None)
    worker.persist_session = False
    return worker


class FileWorker:
    def __init__(self, workspace_root):
        self.open_files = {}
        self.workspace_root = Path(workspace_root).resolve()
        metadata = self.workspace_root.stat()
        self.workspace_root_identity = (int(metadata.st_dev), int(metadata.st_ino))

    def is_file_open(self, path):
        return os.path.abspath(path) in self.open_files

    def update_open_file(self, path, content):
        self.open_files[os.path.abspath(path)] = content

    def close_file(self, path):
        return self.open_files.pop(os.path.abspath(path), None) is not None


class FileEditSafetyScenarios(unittest.TestCase):
    def test_file_tools_are_confined_to_exact_launch_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.txt"
            outside.write_text("private\n", encoding="utf-8")
            worker = FileWorker(workspace)

            opened = OpenFileTool(worker).execute(str(outside))
            written = WriteFileTool(worker).execute(str(outside), "changed\n")

            self.assertEqual(opened.status, ToolStatus.BLOCKED)
            self.assertIn("outside this agent's launch workspace", opened.summary)
            self.assertIn("outside this agent's launch workspace", written)
            self.assertEqual(outside.read_text(encoding="utf-8"), "private\n")

    def test_symlink_ancestors_and_outside_hardlinks_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            outside_dir = root / "outside"
            workspace.mkdir()
            outside_dir.mkdir()
            secret = outside_dir / "secret.txt"
            secret.write_text("private\n", encoding="utf-8")
            (workspace / "escape").symlink_to(outside_dir, target_is_directory=True)
            linked = workspace / "linked.txt"
            os.link(secret, linked)
            worker = FileWorker(workspace)

            through_symlink = OpenFileTool(worker).execute("escape/secret.txt")
            overwrite = WriteFileTool(worker).execute("escape/new.txt", "changed\n")
            through_hardlink = OpenFileTool(worker).execute("linked.txt")

            self.assertEqual(through_symlink.status, ToolStatus.BLOCKED)
            self.assertIn("symlink", through_symlink.summary.lower())
            self.assertIn("symlink", overwrite.lower())
            self.assertEqual(through_hardlink.status, ToolStatus.BLOCKED)
            self.assertIn("multiply-linked", through_hardlink.summary)
            self.assertEqual(secret.read_text(encoding="utf-8"), "private\n")
            self.assertFalse((outside_dir / "new.txt").exists())

    def test_sensitive_state_paths_are_not_direct_file_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("token=secret\n", encoding="utf-8")
            (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
            worker = FileWorker(root)

            git_result = OpenFileTool(worker).execute(".git/config")
            env_result = OpenFileTool(worker).execute(".env")

            self.assertEqual(git_result.status, ToolStatus.BLOCKED)
            self.assertIn("reviewed tool", git_result.summary)
            self.assertEqual(env_result.status, ToolStatus.BLOCKED)
            self.assertIn("credential-like", env_result.summary)

    def test_nested_new_file_uses_workspace_bound_atomic_write(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = FileWorker(directory)
            target = Path(directory) / "nested" / "module.py"
            result = WriteFileTool(worker).execute("nested/module.py", "value = 1\n")
            self.assertIn("Created", result)
            self.assertEqual(target.read_text(encoding="utf-8"), "value = 1\n")

    def test_open_receipt_binds_exact_edit_and_stale_receipt_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "module.py"
            path.write_text("value = 1\n", encoding="utf-8")
            worker = FileWorker(directory)
            opened = OpenFileTool(worker).execute(str(path))
            receipt = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(opened.status, ToolStatus.OK)
            self.assertIn(receipt, opened.summary)

            tool = StrReplaceTool(worker)
            missing = tool.execute(str(path), old_str="value = 1", new_str="value = 2")
            self.assertIn("expected_sha256 is required", missing)
            changed = tool.execute(
                str(path), old_str="value = 1", new_str="value = 2",
                expected_sha256=receipt,
            )
            self.assertIn("Successfully applied", changed)
            self.assertEqual(path.read_text(encoding="utf-8"), "value = 2\n")
            stale = tool.execute(
                str(path), old_str="value = 2", new_str="value = 3",
                expected_sha256=receipt,
            )
            self.assertIn("changed since", stale)
            self.assertEqual(path.read_text(encoding="utf-8"), "value = 2\n")

    def test_open_file_content_cannot_reclassify_its_typed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.txt"
            content = (
                "Error: this is example documentation, not a tool failure\n"
                "permission denied is text being investigated\n"
                "status: running is another literal fixture\n"
            )
            path.write_text(content, encoding="utf-8")
            worker = FileWorker(directory)

            opened = OpenFileTool(worker).execute(str(path))
            normalized = normalize_tool_result(
                "open_file",
                opened,
                policy=infer_tool_policy("open_file"),
                parameters={"file_path": str(path)},
            )
            legacy_text_status = normalize_tool_result(
                "open_file",
                opened.summary,
                policy=infer_tool_policy("open_file"),
                parameters={"file_path": str(path)},
            ).status

            self.assertIs(normalized, opened)
            self.assertEqual(opened.status, ToolStatus.OK)
            self.assertEqual(legacy_text_status, ToolStatus.OK)
            self.assertFalse(opened.changed)
            self.assertIn("Error: this is example documentation", opened.summary)
            working_copy = worker.open_files[str(path.resolve())]
            self.assertIn("Error: this is example documentation", working_copy)
            self.assertIn("permission denied is text being investigated", working_copy)
            self.assertIn("status: running is another literal fixture", working_copy)

    def test_invalid_syntax_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "module.py"
            path.write_text("def f():\n    return 1\n", encoding="utf-8")
            receipt = hashlib.sha256(path.read_bytes()).hexdigest()
            result = StrReplaceTool(FileWorker(directory)).execute(
                str(path), old_str="def f():", new_str="def f(:",
                expected_sha256=receipt,
            )
            self.assertIn("Refusing", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "def f():\n    return 1\n")

            new_path = Path(directory) / "broken.json"
            result = WriteFileTool(FileWorker(directory)).execute(str(new_path), "{broken")
            self.assertIn("Refusing", result)
            self.assertFalse(new_path.exists())

    def test_blind_overwrite_and_default_fuzzy_edit_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("alpha beta gamma\n", encoding="utf-8")
            writer = WriteFileTool(FileWorker(directory))
            self.assertIn(
                "will not blindly overwrite",
                writer.execute(str(path), "replacement"),
            )
            receipt = hashlib.sha256(path.read_bytes()).hexdigest()
            fuzzy = StrReplaceTool(FileWorker(directory)).execute(
                str(path), old_str="alpha beta gammx", new_str="changed",
                expected_sha256=receipt,
            )
            self.assertIn("Could not find one exact", fuzzy)
            self.assertEqual(path.read_text(encoding="utf-8"), "alpha beta gamma\n")


class MemoryBoundaryScenarios(unittest.TestCase):
    def test_secret_values_are_rejected_without_echo(self):
        worker = types.SimpleNamespace(memories={})
        token = "ghp_" + "A" * 32
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = MemorizeTool(worker).execute("github access", token, "credentials")
        self.assertIn("BLOCKED", result)
        self.assertNotIn(token, result + stream.getvalue())
        self.assertEqual(worker.memories, {})

    def test_memory_is_bounded_and_requires_concise_values(self):
        worker = types.SimpleNamespace(memories={})
        tool = MemorizeTool(worker)
        result = tool.execute("oversized", "x" * (tool.MAX_VALUE_CHARS + 1))
        self.assertIn("must be concise", result)
        self.assertEqual(worker.memories, {})

        worker.memories = {f"key-{index}": "value" for index in range(tool.MAX_ITEMS)}
        result = tool.execute("one-too-many", "value")
        self.assertIn("item limit", result)

    def test_legacy_secret_is_withheld_from_prompt_and_listing(self):
        token = "ghp_" + "B" * 32
        worker = Worker.__new__(Worker)
        worker.memories = {
            "github_token": {"value": token, "category": "credentials"},
            "project_root": {"value": "/work", "category": "path", "scope": "project"},
        }
        prompt = worker._format_memories()
        listing = ListMemoriesTool(worker).execute()
        self.assertNotIn(token, prompt + listing)
        self.assertIn("withheld", (prompt + listing).lower())
        self.assertIn("/work", prompt + listing)


class ToolSchemaScenarios(unittest.TestCase):
    class TypedTool(BaseTool):
        def __init__(self):
            super().__init__("typed", "typed")

        def execute(self, path: str, count: int = 1, enabled: bool = False):
            return "ok"

    def test_signature_schema_rejects_unknown_and_wrong_types(self):
        tool = self.TypedTool()
        schema = tool.parameter_schema()
        self.assertEqual(schema["required"], ["path"])
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("unknown parameter", tool.validate_parameters({"path": "x", "extra": 1}))
        self.assertIn("wrong JSON type", tool.validate_parameters({"path": "x", "count": "1"}))
        self.assertEqual(tool.validate_parameters({"path": "x", "count": 2}), "")


class StateAndDelegationBoundaryScenarios(unittest.TestCase):
    def test_lifetime_history_and_checkpoint_rewrites_remain_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = make_worker()
            worker.persist_session = True
            state_path = Path(directory) / "session.json"
            worker._session_state_path = types.MethodType(
                lambda _self: state_path, worker
            )
            encoded_sizes = []
            for index in range(1000):
                worker._history_messages.extend([
                    {"role": "user", "content": f"request-{index}-" + "u" * 300},
                    {"role": "assistant", "content": f"answer-{index}-" + "a" * 300},
                ])
                worker._trim_history()
                if index % 100 == 0:
                    encoded_sizes.append(len(json.dumps(worker.serialize_state())))
                    worker._persist_session_state()

            self.assertLess(max(encoded_sizes), 512_000)
            self.assertLess(max(encoded_sizes) - min(encoded_sizes[-3:]), 256_000)
            self.assertLess(state_path.stat().st_size, 512_000)
            restored = worker._read_bounded_state(state_path)
            self.assertRegex(restored["history_archive_digest"], r"^[0-9a-f]{64}$")
            self.assertGreater(restored["history_archive_messages"], 0)

    def test_oversized_checkpoint_is_refused_before_json_decode(self):
        from aeon.core.worker import MAX_PERSISTED_STATE_BYTES

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.json"
            with path.open("wb") as stream:
                stream.truncate(MAX_PERSISTED_STATE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "bounded file contract"):
                Worker._read_bounded_state(path)

    def test_session_state_is_outside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            state = Path(directory) / "private-state"
            workspace.mkdir()
            previous = os.getcwd()
            os.chdir(workspace)
            try:
                worker = Worker.__new__(Worker)
                worker.instance_id = "instance"
                with mock.patch.dict(os.environ, {"AEON_STATE_DIR": str(state)}):
                    path = worker._session_state_path()
            finally:
                os.chdir(previous)
            self.assertTrue(str(path).startswith(str(state)))
            self.assertNotIn(str(workspace / "aeon_output"), str(path))

    def test_read_only_subagent_is_agent_state_not_project_mutation(self):
        policy = infer_tool_policy("spawn_sub_agent")
        self.assertEqual(
            effective_tool_effect(policy, {"read_only": True}), SideEffect.AGENT_STATE
        )
        self.assertEqual(
            effective_tool_effect(policy, {"read_only": False}), SideEffect.LOCAL_MUTATION
        )
        inspect = RequestContract.from_request("Audit the repository")
        self.assertEqual(inspect.mode, RequestMode.INSPECT)
        self.assertEqual(inspect.authorization_error(policy, {"read_only": True}), "")
        self.assertIn("does not authorize", inspect.authorization_error(policy, {"read_only": False}))

    def test_history_trimming_keeps_tool_call_and_receipt_atomic(self):
        worker = make_worker()
        worker._history_messages = [
            {"role": "user", "content": "old " * 5000},
            {
                "role": "assistant",
                "content": "tool decision",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "open_file", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "open_file",
                "content": "observed",
            },
            {"role": "user", "content": "new request"},
        ]
        worker._trim_history(max_tokens=200)
        roles = [message["role"] for message in worker._history_messages]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)
        assistant_index = roles.index("assistant")
        self.assertEqual(roles[assistant_index + 1], "tool")

    def test_waiting_request_contract_survives_state_round_trip(self):
        from aeon.core.agent_protocol import ExecutionState

        worker = make_worker()
        worker.request_contract = RequestContract.from_request(
            "Can you create an agent?", forced_mode=RequestMode.PLAN
        )
        worker.request_contract.state = ExecutionState.WAITING_USER
        worker.request_contract.pending_question = "Should I create it now?"
        worker.execution_state = ExecutionState.WAITING_USER
        worker.pending_question = worker.request_contract.pending_question
        worker.request_id = worker.request_contract.request_id
        worker._history_messages = [
            {"role": "user", "content": "Can you create an agent?"},
            {"role": "assistant", "content": "Should I create it now?"},
        ]

        restored = make_worker()
        restored.restore_state(worker.serialize_state())
        self.assertEqual(restored.execution_state, ExecutionState.WAITING_USER)
        self.assertEqual(restored.request_contract.mode, RequestMode.PLAN)
        self.assertEqual(restored.pending_question, "Should I create it now?")

    def test_read_only_subagent_stays_visible_without_authorizing_mutable_child(self):
        class Tool(BaseTool):
            def __init__(self, name):
                super().__init__(name, name)

            def execute(self, **kwargs):
                return "ok"

        worker = make_worker(Tool("spawn_sub_agent"), Tool("write_file"))
        worker.request_contract = RequestContract.from_request(
            "How would you investigate this?", forced_mode=RequestMode.PLAN
        )
        self.assertIn("spawn_sub_agent", worker._active_tool_names())
        self.assertNotIn("write_file", worker._active_tool_names())
        policy = worker._tool_policy("spawn_sub_agent")
        self.assertEqual(worker.request_contract.authorization_error(policy, {"read_only": True}), "")
        self.assertIn(
            "does not authorize",
            worker.request_contract.authorization_error(policy, {"read_only": False}),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
