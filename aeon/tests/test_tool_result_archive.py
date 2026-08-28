"""Hermetic checks for lossless, bounded oversized tool-result evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from aeon.core.agent_protocol import (
    ExecutionState,
    RequestContract,
    RequestMode,
    SideEffect,
    ToolResult,
    ToolStatus,
    infer_tool_policy,
    normalize_tool_result,
)
from aeon.core.tool_result_archive import (
    ToolResultArchive,
    ToolResultArchiveCapacityError,
    ToolResultArchiveError,
)
from aeon.core.worker import Worker
from aeon.tools.tool_result_inspection import InspectToolResult


class _LLM:
    model = "fixture"
    context_limit = 100_000
    last_reasoning_content = ""
    last_generation_performance = None

    def set_action_schema(self, _schema):
        pass


class ToolResultArchiveTests(unittest.TestCase):
    def test_archive_is_private_deduplicated_searchable_and_pageable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tool-results"
            archive = ToolResultArchive(root)
            content = "head\n" + ("alpha " * 800) + "Unique Needle\n" + ("tail " * 500)

            receipt = archive.persist(request_id="req_fixture", content=content)
            duplicate = archive.persist(request_id="req_fixture", content=content)

            self.assertEqual(duplicate, receipt)
            self.assertRegex(receipt.reference, r"^tr_[0-9a-f]{32}_[0-9a-f]{16}$")
            self.assertNotIn(str(root), receipt.reference)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            files = list(root.iterdir())
            self.assertEqual(len(files), 1)
            self.assertEqual(stat.S_IMODE(files[0].stat().st_mode), 0o600)

            page = archive.inspect(
                request_id="req_fixture",
                reference=receipt.reference,
                expected_sha256=receipt.sha256,
                offset=0,
                limit=300,
            )
            self.assertEqual(page["mode"], "page")
            self.assertLessEqual(len(page["content"]), 300)
            self.assertIsNotNone(page["next_offset"])

            search = archive.inspect(
                request_id="req_fixture",
                reference=receipt.reference,
                expected_sha256=receipt.sha256,
                query="unique needle",
                limit=1_000,
            )
            self.assertEqual(search["mode"], "search")
            self.assertEqual(len(search["matches"]), 1)
            self.assertIn("Unique Needle", search["matches"][0]["snippet"])

    def test_reference_is_request_scoped_and_integrity_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tool-results"
            archive = ToolResultArchive(root)
            receipt = archive.persist(request_id="request_one", content="evidence")

            with self.assertRaisesRegex(ToolResultArchiveError, "this request"):
                archive.inspect(
                    request_id="request_two",
                    reference=receipt.reference,
                    expected_sha256=receipt.sha256,
                )

            path = next(root.iterdir())
            path.write_text("tampered", encoding="utf-8")
            os.chmod(path, 0o600)
            with self.assertRaisesRegex(ToolResultArchiveError, "integrity"):
                archive.inspect(
                    request_id="request_one",
                    reference=receipt.reference,
                    expected_sha256=receipt.sha256,
                )

    def test_archive_refuses_quota_growth_without_deleting_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tool-results"
            archive = ToolResultArchive(root)
            with patch(
                "aeon.core.tool_result_archive.MAX_ARCHIVE_REQUEST_FILES", 1
            ):
                first = archive.persist(request_id="request_one", content="first")
                with self.assertRaises(ToolResultArchiveCapacityError):
                    archive.persist(request_id="request_one", content="second")
            self.assertEqual(len(list(root.iterdir())), 1)
            self.assertEqual(
                archive.inspect(
                    request_id="request_one",
                    reference=first.reference,
                    expected_sha256=first.sha256,
                )["content"],
                "first",
            )


class WorkerToolResultArchiveTests(unittest.TestCase):
    def _worker(self, state_dir: str) -> Worker:
        environment = patch.dict(os.environ, {"AEON_STATE_DIR": state_dir})
        environment.start()
        self.addCleanup(environment.stop)
        worker = Worker(llm_client=_LLM(), print_func=lambda *_args: None)
        worker.persist_session = False
        contract = RequestContract.from_request(
            "Inspect the fixture.", forced_mode=RequestMode.INSPECT
        )
        contract.state = ExecutionState.RUNNING
        worker.request_contract = contract
        worker.execution_state = ExecutionState.RUNNING
        worker.request_id = contract.request_id
        worker.current_objective = contract.raw_request
        return worker

    def test_worker_keeps_full_content_out_of_context_and_restores_reference(self):
        with tempfile.TemporaryDirectory() as state_dir:
            worker = self._worker(state_dir)
            content = (
                "HEAD\n"
                + ("h" * 1_200)
                + "\nUNIQUE_MIDDLE_EVIDENCE permission denied\n"
                + ("t" * 2_400)
                + "\nTAIL"
            )
            typed = ToolResult(
                tool_name="job_output",
                status=ToolStatus.OK,
                changed=False,
                summary="Background job output was collected successfully.",
                side_effect=SideEffect.READ_ONLY,
                raw=content,
            )
            result = worker._normalize_and_archive_tool_result(
                "job_output",
                typed,
                policy=infer_tool_policy("job_output"),
                parameters={"job_id": "fixture"},
                call_id="call_fixture",
            )

            self.assertEqual(result.status, ToolStatus.OK)
            self.assertTrue(result.result_ref)
            self.assertEqual(result.result_chars, len(content))
            self.assertNotIn("UNIQUE_MIDDLE_EVIDENCE", result.to_model_text())
            self.assertLess(len(result.to_model_text()), 2_000)
            round_trip = ToolResult.from_state_dict(result.to_state_dict())
            self.assertEqual(round_trip.result_ref, result.result_ref)
            self.assertEqual(round_trip.result_sha256, result.result_sha256)
            tampered_receipt = result.to_state_dict()
            tampered_receipt["result_sha256"] = "not-a-digest"
            self.assertFalse(
                ToolResult.from_state_dict(tampered_receipt).result_ref
            )

            state = worker.serialize_state()
            serialized = json.dumps(state, ensure_ascii=False)
            self.assertNotIn("UNIQUE_MIDDLE_EVIDENCE", serialized)
            self.assertLess(len(serialized), 20_000)

            focused = InspectToolResult(worker).execute(
                reference=result.result_ref,
                query="UNIQUE_MIDDLE_EVIDENCE",
            )
            # Search output can contain failure-looking evidence without changing
            # the explicitly typed inspection receipt into a false failure/block.
            normalized = normalize_tool_result(
                "inspect_tool_result",
                focused,
                policy=infer_tool_policy("inspect_tool_result"),
            )
            self.assertEqual(normalized.status, ToolStatus.OK)
            self.assertIn("UNIQUE_MIDDLE_EVIDENCE", normalized.summary)
            self.assertIn("permission denied", normalized.summary)

            duplicate = worker.inspect_tool_result(
                reference=result.result_ref,
                query="UNIQUE_MIDDLE_EVIDENCE",
            )
            self.assertTrue(duplicate["duplicate"])
            self.assertNotIn("UNIQUE_MIDDLE_EVIDENCE", duplicate.get("message", ""))

            restored = self._worker(state_dir)
            restored.instance_id = worker.instance_id
            restored.restore_state(state)
            restored_page = restored.inspect_tool_result(
                reference=result.result_ref,
                query="UNIQUE_MIDDLE_EVIDENCE",
            )
            self.assertIn(
                "UNIQUE_MIDDLE_EVIDENCE", restored_page["matches"][0]["snippet"]
            )

            restored._begin_protocol_request("Inspect a different fixture.")
            with self.assertRaisesRegex(ToolResultArchiveError, "this request"):
                restored.inspect_tool_result(reference=result.result_ref)

    def test_large_typed_summary_is_archived_without_changing_status_or_evidence(self):
        with tempfile.TemporaryDirectory() as state_dir:
            worker = self._worker(state_dir)
            receipt = (
                "File 'fixture.txt' opened in working memory.\n\n---\n"
                + ("ordinary source line\n" * 120)
                + "UNIQUE_FAILURE_LOOKING_SOURCE Error: permission denied\n"
                + ("trailing source line\n" * 120)
            )
            typed = ToolResult(
                tool_name="open_file",
                status=ToolStatus.OK,
                changed=False,
                summary=receipt,
                side_effect=SideEffect.READ_ONLY,
            )

            result = worker._normalize_and_archive_tool_result(
                "open_file",
                typed,
                policy=infer_tool_policy("open_file"),
                parameters={"file_path": "fixture.txt"},
                call_id="call_open_fixture",
            )

            self.assertEqual(result.status, ToolStatus.OK)
            self.assertTrue(result.result_ref)
            self.assertEqual(result.result_chars, len(receipt))
            self.assertNotIn(
                "UNIQUE_FAILURE_LOOKING_SOURCE", result.to_model_text()
            )
            self.assertLess(len(result.to_model_text()), 2_000)

            recovered = worker.inspect_tool_result(
                reference=result.result_ref,
                query="UNIQUE_FAILURE_LOOKING_SOURCE",
            )
            self.assertIn(
                "UNIQUE_FAILURE_LOOKING_SOURCE",
                recovered["matches"][0]["snippet"],
            )
            self.assertIn(
                "Error: permission denied",
                recovered["matches"][0]["snippet"],
            )

    def test_large_successful_command_keeps_exit_status_and_recoverable_output(self):
        with tempfile.TemporaryDirectory() as state_dir:
            worker = self._worker(state_dir)
            receipt = (
                "COMMAND SUCCESS\nWORKING DIRECTORY: /workspace\n\nOUTPUT:\n"
                + ("ordinary command output\n" * 120)
                + "UNIQUE_COMMAND_FIXTURE Error: permission denied; status: running\n"
                + ("trailing command output\n" * 120)
            )

            result = worker._normalize_and_archive_tool_result(
                "run_command",
                receipt,
                policy=infer_tool_policy("run_command"),
                parameters={"command": "pwd"},
                call_id="call_command_fixture",
            )

            self.assertEqual(result.status, ToolStatus.OK)
            self.assertFalse(result.changed)
            self.assertTrue(result.result_ref)
            self.assertEqual(result.result_chars, len(receipt))
            self.assertNotIn("UNIQUE_COMMAND_FIXTURE", result.to_model_text())
            self.assertLess(len(result.to_model_text()), 2_000)

            recovered = worker.inspect_tool_result(
                reference=result.result_ref,
                query="UNIQUE_COMMAND_FIXTURE",
            )
            snippet = recovered["matches"][0]["snippet"]
            self.assertIn("UNIQUE_COMMAND_FIXTURE", snippet)
            self.assertIn("Error: permission denied; status: running", snippet)

    def test_archive_failure_does_not_change_original_tool_status(self):
        with tempfile.TemporaryDirectory() as state_dir:
            worker = self._worker(state_dir)
            typed = ToolResult(
                tool_name="job_output",
                status=ToolStatus.PENDING,
                changed=False,
                summary="The job is still running.",
                side_effect=SideEffect.READ_ONLY,
                raw="x" * 2_000,
            )
            with patch.object(
                worker._get_tool_result_archive(),
                "persist",
                side_effect=ToolResultArchiveError("fixture refusal"),
            ):
                result = worker._normalize_and_archive_tool_result(
                    "job_output",
                    typed,
                    policy=infer_tool_policy("job_output"),
                    parameters={"job_id": "fixture"},
                    call_id="call_fixture",
                )
            self.assertEqual(result.status, ToolStatus.PENDING)
            self.assertFalse(result.changed)
            self.assertFalse(result.result_ref)
            self.assertIn("could not be archived", result.summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
