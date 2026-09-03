from __future__ import annotations

import fcntl
import os
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import aeon.core.chat_transcript as chat_transcript
from aeon.core.chat_attachments import (
    ATTACHMENT_DIRECTORY,
    ChatAttachmentError,
    resolve_chat_attachment,
    store_chat_attachments,
    store_generated_chat_attachments,
)

from aeon.core.chat_transcript import (
    CHAT_DELIVERY_FAILURES_FILENAME,
    CHAT_TRANSCRIPT_ENV,
    CHAT_TRANSCRIPT_FILENAME,
    CHAT_WRITER_PID_ENV,
    CHAT_DELIVERY_PREFIX,
    ChatTranscriptError,
    append_assistant_message_from_environment,
    append_chat_message,
    append_plan_message_from_environment,
    append_progress_message_from_environment,
    clear_chat_messages,
    clear_chat_messages_from_environment,
    commit_chat_delivery,
    prepare_chat_delivery,
    read_chat_messages,
    wait_for_chat_delivery_consumed,
)
from aeon.core.console import ConsoleInput
from aeon.core.orchestrator_instructions import (
    MAIN_ORCHESTRATOR_ENV,
    main_orchestrator_instruction_section,
)


class ChatTranscriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / "instance"
        self.directory.mkdir(mode=0o700)
        self.path = self.directory / CHAT_TRANSCRIPT_FILENAME

    def tearDown(self):
        self.temporary.cleanup()

    def test_private_transcript_round_trip(self):
        user = append_chat_message(self.path, role="user", content="Hello")
        assistant = append_chat_message(
            self.path, role="assistant", content="Hello back"
        )

        self.assertEqual(read_chat_messages(self.path), [user, assistant])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.directory / "chat-transcript.lock").stat().st_mode & 0o777, 0o600)

    def test_caller_supplied_message_identity_is_persisted(self):
        message_id = "msg-" + "a" * 32
        message = append_chat_message(
            self.path,
            role="user",
            content="Durable voice turn",
            message_id=message_id,
        )

        self.assertEqual(message["id"], message_id)
        self.assertEqual(read_chat_messages(self.path), [message])

    def test_managed_delivery_waits_for_exact_transcript_commit(self):
        message_id = "msg-" + "d" * 32
        content = "This line must not reach the worker before commit."
        envelope = prepare_chat_delivery(self.path, message_id, content)
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console._typeahead = True
        started = threading.Event()

        def dispatch():
            started.set()
            input_console._dispatch_line(envelope)

        with patch.dict(
            os.environ,
            {CHAT_TRANSCRIPT_ENV: str(self.path)},
            clear=False,
        ):
            thread = threading.Thread(target=dispatch)
            thread.start()
            self.assertTrue(started.wait(timeout=1))
            time.sleep(0.03)
            self.assertFalse(input_console.has_pending())
            committed = commit_chat_delivery(
                self.path,
                message_id,
                content,
            )
            thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(input_console.take_pending(), content)
        self.assertEqual(read_chat_messages(self.path), [committed])
        self.assertTrue(
            wait_for_chat_delivery_consumed(
                self.path, message_id, content, timeout_seconds=0.01
            )
        )

    def test_delivery_keeps_private_transport_out_of_visible_transcript(self):
        message_id = "msg-" + "1" * 32
        visible = "Please inspect the attached image."
        private_path = str(self.directory / "chat-attachments" / "owner-only.png")
        transport = (
            f"{visible}\n\nNexus supplied a private attachment at {private_path}.\n"
            "Use analyze_image on that exact path before answering."
        )
        public_attachment = {
            "id": "att-" + "2" * 32,
            "name": "screen.png",
            "media_type": "image",
            "mime_type": "image/png",
            "size_bytes": 123,
        }
        envelope = prepare_chat_delivery(
            self.path,
            message_id,
            transport,
            visible_content=visible,
            attachments=[public_attachment],
        )
        committed = commit_chat_delivery(
            self.path,
            message_id,
            transport,
            visible_content=visible,
            attachments=[public_attachment],
        )
        self.assertFalse(
            wait_for_chat_delivery_consumed(
                self.path,
                message_id,
                transport,
                visible_content=visible,
                attachments=[public_attachment],
                timeout_seconds=0.01,
            )
        )
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console._typeahead = True
        with patch.dict(
            os.environ, {CHAT_TRANSCRIPT_ENV: str(self.path)}, clear=False
        ):
            input_console._dispatch_line(envelope)

        self.assertEqual(input_console.take_pending(), transport)
        self.assertEqual(committed["content"], visible)
        self.assertEqual(committed["attachments"], [public_attachment])
        self.assertEqual(read_chat_messages(self.path), [committed])
        persisted = self.path.read_text(encoding="utf-8")
        self.assertNotIn(private_path, persisted)
        self.assertNotIn("Use analyze_image on that exact path", persisted)

    def test_commit_recovers_once_after_transcript_append_state_write_crash(self):
        message_id = "msg-" + "3" * 32
        content = "Recover this exact committed turn once."
        envelope = prepare_chat_delivery(self.path, message_id, content)
        original_write = chat_transcript._write_delivery_entries_locked
        failed = False

        def fail_after_append(directory_fd, entries):
            nonlocal failed
            if not failed and entries[message_id]["state"] == "committed":
                failed = True
                raise ChatTranscriptError("injected state write failure")
            return original_write(directory_fd, entries)

        with patch.object(
            chat_transcript,
            "_write_delivery_entries_locked",
            side_effect=fail_after_append,
        ):
            with self.assertRaisesRegex(ChatTranscriptError, "injected"):
                commit_chat_delivery(self.path, message_id, content)

        self.assertEqual(len(read_chat_messages(self.path)), 1)
        recovered = commit_chat_delivery(self.path, message_id, content)
        self.assertEqual(read_chat_messages(self.path), [recovered])

        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console._typeahead = True
        with patch.dict(
            os.environ, {CHAT_TRANSCRIPT_ENV: str(self.path)}, clear=False
        ):
            input_console._dispatch_line(envelope)
            input_console._dispatch_line(envelope)
        self.assertEqual(input_console.take_pending(), content)
        self.assertFalse(input_console.has_pending())

    def test_corrupt_delivery_state_is_quarantined_after_finite_attempts(self):
        message_id = "msg-" + "4" * 32
        content = "Never dispatch through corrupt delivery state."
        envelope = prepare_chat_delivery(self.path, message_id, content)
        state_path = self.directory / "chat-delivery-state.json"
        state_path.write_text("{not-json", encoding="utf-8")
        self.assertFalse(
            wait_for_chat_delivery_consumed(
                self.path, message_id, content, timeout_seconds=0.02
            )
        )
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console._typeahead = True
        started = time.monotonic()
        with (
            patch.dict(
                os.environ, {CHAT_TRANSCRIPT_ENV: str(self.path)}, clear=False
            ),
            patch.object(chat_transcript, "CHAT_DELIVERY_COMMIT_WAIT_SECONDS", 0),
            patch.object(chat_transcript, "CHAT_DELIVERY_MAX_ATTEMPTS", 3),
            self.assertWarnsRegex(RuntimeWarning, "envelope was dropped"),
        ):
            input_console._dispatch_line(envelope)

        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(input_console.has_pending())
        failure_path = self.directory / CHAT_DELIVERY_FAILURES_FILENAME
        self.assertTrue(failure_path.is_file())
        self.assertIn(message_id, failure_path.read_text(encoding="utf-8"))
        self.assertEqual(failure_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            len(list(self.directory.glob("chat-delivery-state.quarantine-*.json"))),
            1,
        )
        self.assertEqual(read_chat_messages(self.path), [])

    def test_receiver_lock_contention_is_finite_and_never_dispatches(self):
        message_id = "msg-" + "5" * 32
        content = "Do not block ConsoleInput behind a transcript writer."
        envelope = prepare_chat_delivery(self.path, message_id, content)
        lock_descriptor = os.open(
            self.directory / "chat-transcript.lock", os.O_RDWR | os.O_CLOEXEC
        )
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console._typeahead = True
        started = time.monotonic()
        try:
            self.assertFalse(
                wait_for_chat_delivery_consumed(
                    self.path, message_id, content, timeout_seconds=0.03
                )
            )
            with (
                patch.dict(
                    os.environ, {CHAT_TRANSCRIPT_ENV: str(self.path)}, clear=False
                ),
                patch.object(
                    chat_transcript, "CHAT_DELIVERY_COMMIT_WAIT_SECONDS", 0.03
                ),
                patch.object(chat_transcript, "CHAT_DELIVERY_MAX_ATTEMPTS", 4),
                patch.object(
                    chat_transcript,
                    "CHAT_DELIVERY_QUARANTINE_LOCK_WAIT_SECONDS",
                    0.02,
                ),
                self.assertWarnsRegex(RuntimeWarning, "envelope was dropped"),
            ):
                input_console._dispatch_line(envelope)
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)

        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(input_console.has_pending())
        self.assertEqual(read_chat_messages(self.path), [])

    def test_persistent_consume_error_is_durably_dropped_after_deadline(self):
        message_id = "msg-" + "6" * 32
        content = "Never dispatch after persistent consume errors."
        envelope = prepare_chat_delivery(self.path, message_id, content)
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console._typeahead = True
        with (
            patch.dict(
                os.environ, {CHAT_TRANSCRIPT_ENV: str(self.path)}, clear=False
            ),
            patch.object(chat_transcript, "CHAT_DELIVERY_COMMIT_WAIT_SECONDS", 0.02),
            patch.object(chat_transcript, "CHAT_DELIVERY_MAX_ATTEMPTS", 4),
            patch.object(
                chat_transcript,
                "_consume_chat_delivery",
                side_effect=ChatTranscriptError("injected persistent consume error"),
            ),
            self.assertWarnsRegex(RuntimeWarning, "envelope was dropped"),
        ):
            input_console._dispatch_line(envelope)

        self.assertFalse(input_console.has_pending())
        failure_path = self.directory / CHAT_DELIVERY_FAILURES_FILENAME
        self.assertIn(message_id, failure_path.read_text(encoding="utf-8"))
        state = (self.directory / "chat-delivery-state.json").read_text(
            encoding="utf-8"
        )
        self.assertIn('"state":"abandoned"', state)
        self.assertFalse(
            wait_for_chat_delivery_consumed(
                self.path, message_id, content, timeout_seconds=0.01
            )
        )

    def test_uncommitted_managed_delivery_is_dropped_not_exposed(self):
        content = "Never expose this uncommitted transport line."
        message_id = "msg-" + "e" * 32
        envelope = prepare_chat_delivery(self.path, message_id, content)
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console._typeahead = True
        with (
            patch.dict(
                os.environ,
                {CHAT_TRANSCRIPT_ENV: str(self.path)},
                clear=False,
            ),
            patch(
                "aeon.core.chat_transcript.CHAT_DELIVERY_COMMIT_WAIT_SECONDS",
                0.03,
            ),
        ):
            input_console._dispatch_line(envelope)

        self.assertFalse(input_console.has_pending())
        self.assertEqual(read_chat_messages(self.path), [])
        with self.assertRaisesRegex(ChatTranscriptError, "abandoned"):
            commit_chat_delivery(self.path, message_id, content)
        self.assertEqual(read_chat_messages(self.path), [])
        self.assertFalse(
            wait_for_chat_delivery_consumed(
                self.path, message_id, content, timeout_seconds=0.01
            )
        )

    def test_managed_delivery_decodes_once_and_raw_clear_can_commit(self):
        content = (
            CHAT_DELIVERY_PREFIX
            + "\n/clear-looking collaborator text stays literal."
        )
        message_id = "msg-" + "f" * 32
        envelope = prepare_chat_delivery(self.path, message_id, content)
        commit_chat_delivery(self.path, message_id, content)
        input_console = ConsoleInput()
        input_console._tty = True
        input_console._started = True
        input_console._typeahead = True
        with patch.dict(
            os.environ,
            {CHAT_TRANSCRIPT_ENV: str(self.path)},
            clear=False,
        ):
            input_console._dispatch_line(envelope)
            input_console._dispatch_line(envelope)

        self.assertEqual(input_console.take_pending(), content)
        self.assertFalse(input_console.has_pending())

        clear_path = self.directory / CHAT_TRANSCRIPT_FILENAME
        clear_chat_messages(clear_path)
        clear_id = "msg-" + "0" * 32
        clear_envelope = prepare_chat_delivery(clear_path, clear_id, "/clear")
        commit_chat_delivery(clear_path, clear_id, "/clear")
        with patch.dict(
            os.environ,
            {CHAT_TRANSCRIPT_ENV: str(clear_path)},
            clear=False,
        ):
            input_console._dispatch_line(clear_envelope)
        self.assertEqual(input_console.take_pending(), "/clear")

    def test_media_attachment_round_trip_stays_private_and_transcript_bounded(self):
        image_bytes = BytesIO()
        Image.new("RGB", (8, 6), "purple").save(image_bytes, format="PNG")
        upload = SimpleNamespace(
            filename="../screen shot.png",
            content_type="image/png",
            file=BytesIO(image_bytes.getvalue()),
        )

        attachments = store_chat_attachments(self.directory, [upload])
        public = attachments[0].public()
        message = append_chat_message(
            self.path,
            role="user",
            content="What is in this image?",
            attachments=[public],
        )

        self.assertEqual(message["attachments"][0]["name"], "screen shot.png")
        self.assertNotIn(str(self.directory), repr(message))
        self.assertEqual(
            resolve_chat_attachment(self.directory, public), attachments[0].path
        )
        attachment_directory = self.directory / ATTACHMENT_DIRECTORY
        self.assertEqual(attachment_directory.stat().st_mode & 0o777, 0o700)
        self.assertEqual(attachments[0].path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(read_chat_messages(self.path), [message])

    def test_attachment_magic_mismatch_is_rejected_without_retaining_bytes(self):
        upload = SimpleNamespace(
            filename="not-really.png",
            content_type="image/png",
            file=BytesIO(b"not an image"),
        )
        with self.assertRaises(ChatAttachmentError):
            store_chat_attachments(self.directory, [upload])
        directory = self.directory / ATTACHMENT_DIRECTORY
        retained = [
            path for path in directory.iterdir()
            if path.name != "chat-attachments.lock"
        ]
        self.assertEqual(retained, [])

    def test_generated_video_is_copied_and_attached_only_by_primary_writer(self):
        source = Path(self.temporary.name) / "render.mp4"
        source.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"verified-render" * 4)
        stored = store_generated_chat_attachments(self.directory, [source])
        self.assertEqual(stored[0].public()["media_type"], "video")
        stored[0].path.unlink()

        environment = {
            CHAT_TRANSCRIPT_ENV: str(self.path),
            CHAT_WRITER_PID_ENV: str(os.getpid()),
        }
        with patch.dict(os.environ, environment, clear=False):
            self.assertTrue(
                append_assistant_message_from_environment(
                    "Here is the final cut.", artifact_paths=[str(source)]
                )
            )
        messages = read_chat_messages(self.path)
        self.assertEqual(messages[0]["role"], "assistant")
        attachment = messages[0]["attachments"][0]
        self.assertEqual(attachment["name"], "render.mp4")
        copied = resolve_chat_attachment(self.directory, attachment)
        self.assertNotEqual(copied, source)
        self.assertEqual(copied.read_bytes(), source.read_bytes())

    def test_generated_attachment_rejects_symlink_source(self):
        source = Path(self.temporary.name) / "real.mp4"
        source.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"video" * 8)
        link = Path(self.temporary.name) / "linked.mp4"
        link.symlink_to(source)
        with self.assertRaises(ChatAttachmentError):
            store_generated_chat_attachments(self.directory, [link])

    def test_assistant_performance_is_validated_and_persisted(self):
        message = append_chat_message(
            self.path,
            role="assistant",
            content="Done.",
            performance={"tokens_per_second": 123.456, "completion_tokens": 789},
        )
        self.assertEqual(
            message["performance"],
            {"tokens_per_second": 123.46, "completion_tokens": 789},
        )
        self.assertEqual(read_chat_messages(self.path), [message])

    def test_assistant_performance_preserves_distinct_live_measurements(self):
        performance = {
            "tokens_per_second": 108.237,
            "decode_tokens_per_second": 108.237,
            "end_to_end_tokens_per_second": 37.894,
            "inference_tokens_per_second": 74.126,
            "completion_tokens": 321,
            "prompt_tokens": 8_192,
            "cached_prompt_tokens": 6_144,
            "time_to_first_token_seconds": 1.2378,
            "prefill_time_to_first_token_seconds": 0.4518,
            "queue_seconds": 0.0838,
            "mean_inter_token_seconds": 0.009247,
            "decode_seconds": 2.9664,
            "end_to_end_seconds": 8.4712,
            "reasoning_effort": "xhigh",
            "served_model": "Qwen3-Coder-Next-FP8",
            "measurement": "vllm_per_request_metrics",
            "speculative_method": "mtp",
            "speculative_tokens": 3,
        }

        message = append_chat_message(
            self.path,
            role="assistant",
            content="Measured response.",
            performance=performance,
        )

        self.assertEqual(
            message["performance"],
            {
                "tokens_per_second": 108.24,
                "completion_tokens": 321,
                "decode_tokens_per_second": 108.24,
                "end_to_end_tokens_per_second": 37.89,
                "inference_tokens_per_second": 74.13,
                "time_to_first_token_seconds": 1.238,
                "prefill_time_to_first_token_seconds": 0.452,
                "queue_seconds": 0.084,
                "mean_inter_token_seconds": 0.0092,
                "decode_seconds": 2.966,
                "end_to_end_seconds": 8.471,
                "prompt_tokens": 8_192,
                "cached_prompt_tokens": 6_144,
                "speculative_tokens": 3,
                "reasoning_effort": "xhigh",
                "served_model": "Qwen3-Coder-Next-FP8",
                "measurement": "vllm_per_request_metrics",
                "speculative_method": "mtp",
            },
        )

    def test_assistant_performance_rejects_impossible_cache_count(self):
        with self.assertRaisesRegex(ChatTranscriptError, "performance is invalid"):
            append_chat_message(
                self.path,
                role="assistant",
                content="Impossible measurement.",
                performance={
                    "tokens_per_second": 100,
                    "completion_tokens": 10,
                    "prompt_tokens": 100,
                    "cached_prompt_tokens": 101,
                },
            )

    def test_progress_round_trip_is_redacted_and_primary_pid_only(self):
        environment = {
            CHAT_TRANSCRIPT_ENV: str(self.path),
            CHAT_WRITER_PID_ENV: str(os.getpid()),
        }
        with patch.dict(os.environ, environment, clear=False):
            self.assertTrue(
                append_progress_message_from_environment(
                    "Step 2\nInspect https://example.test token=super-secret-value"
                )
            )
        messages = read_chat_messages(self.path)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "progress")
        self.assertNotIn("example.test", messages[0]["content"])
        self.assertNotIn("super-secret-value", messages[0]["content"])
        self.assertNotIn("\n", messages[0]["content"])

        with patch.dict(
            os.environ,
            {
                CHAT_TRANSCRIPT_ENV: str(self.path),
                CHAT_WRITER_PID_ENV: str(os.getpid() + 1),
            },
            clear=False,
        ):
            self.assertFalse(append_progress_message_from_environment("Do not append"))
        self.assertEqual(read_chat_messages(self.path), messages)

    def test_plan_round_trip_preserves_checklist_and_supports_clear_marker(self):
        environment = {
            CHAT_TRANSCRIPT_ENV: str(self.path),
            CHAT_WRITER_PID_ENV: str(os.getpid()),
        }
        with patch.dict(os.environ, environment, clear=False):
            self.assertTrue(
                append_plan_message_from_environment(
                    "- [x] Inspect current code\n"
                    "- [ ] Test https://example.test token=super-secret-value"
                )
            )
            self.assertTrue(append_plan_message_from_environment(""))

        messages = read_chat_messages(self.path)
        self.assertEqual([message["role"] for message in messages], ["plan", "plan"])
        self.assertIn("- [x] Inspect current code", messages[0]["content"])
        self.assertIn("\n- [ ] Test [url]", messages[0]["content"])
        self.assertNotIn("super-secret-value", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "")

    def test_latest_plan_is_retained_when_the_visible_suffix_is_full(self):
        plan = append_chat_message(
            self.path,
            role="plan",
            content="- [ ] Keep this checklist pinned",
        )
        for index in range(3):
            append_chat_message(
                self.path,
                role="progress",
                content=f"Safe progress {index}",
            )

        with patch("aeon.core.chat_transcript.MAX_CHAT_MESSAGES", 3):
            messages = read_chat_messages(self.path)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0], plan)
        self.assertEqual(
            [message["content"] for message in messages[1:]],
            ["Safe progress 1", "Safe progress 2"],
        )

    def test_collaborator_rolling_suffix_preserves_active_turn_order(self):
        with (
            patch(
                "aeon.core.chat_transcript.COLLABORATOR_CHAT_TRANSCRIPT_BYTES",
                1_800,
            ),
            patch(
                "aeon.core.chat_transcript.COLLABORATOR_CHAT_RETAIN_BYTES",
                900,
            ),
        ):
            for index in range(8):
                append_chat_message(
                    self.path,
                    role="user",
                    content=f"Old public question {index} " + "q" * 90,
                    rolling=True,
                )
                append_chat_message(
                    self.path,
                    role="assistant",
                    content=f"Old public answer {index} " + "a" * 90,
                    rolling=True,
                )
            active = append_chat_message(
                self.path,
                role="user",
                content="This unmatched requirement must survive compaction.",
                message_id="msg-" + "c" * 32,
                rolling=True,
            )
            for index in range(5):
                append_chat_message(
                    self.path,
                    role="progress",
                    content=f"Bounded liaison progress {index} " + "p" * 80,
                    rolling=True,
                )
            answer = append_chat_message(
                self.path,
                role="assistant",
                content="The active requirement was captured.",
                rolling=True,
            )

        messages = read_chat_messages(self.path)
        active_index = next(
            index for index, message in enumerate(messages) if message["id"] == active["id"]
        )
        answer_index = next(
            index for index, message in enumerate(messages) if message["id"] == answer["id"]
        )
        self.assertLess(active_index, answer_index)
        self.assertLessEqual(self.path.stat().st_size, 1_800)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_inherited_subprocess_cannot_write_as_the_primary(self):
        environment = {
            CHAT_TRANSCRIPT_ENV: str(self.path),
            CHAT_WRITER_PID_ENV: str(os.getpid() + 1),
        }
        with patch.dict(os.environ, environment, clear=False):
            self.assertFalse(
                append_assistant_message_from_environment("must not be written")
            )
        self.assertEqual(read_chat_messages(self.path), [])

    def test_clear_transcript_keeps_private_file_ready_for_new_messages(self):
        append_chat_message(self.path, role="user", content="Forget this")
        clear_chat_messages(self.path)
        self.assertEqual(read_chat_messages(self.path), [])
        replacement = append_chat_message(
            self.path, role="assistant", content="Fresh context"
        )
        self.assertEqual(read_chat_messages(self.path), [replacement])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_clear_control_record_is_never_returned_as_conversation(self):
        append_chat_message(self.path, role="user", content="/CLEAR")
        self.assertEqual(read_chat_messages(self.path), [])

    def test_only_exact_primary_process_can_clear_transcript(self):
        message = append_chat_message(self.path, role="user", content="Keep this")
        with patch.dict(
            os.environ,
            {
                CHAT_TRANSCRIPT_ENV: str(self.path),
                CHAT_WRITER_PID_ENV: str(os.getpid() + 1),
            },
            clear=False,
        ):
            self.assertFalse(clear_chat_messages_from_environment())
        self.assertEqual(read_chat_messages(self.path), [message])

        with patch.dict(
            os.environ,
            {
                CHAT_TRANSCRIPT_ENV: str(self.path),
                CHAT_WRITER_PID_ENV: str(os.getpid()),
            },
            clear=False,
        ):
            self.assertTrue(clear_chat_messages_from_environment())
        self.assertEqual(read_chat_messages(self.path), [])

    def test_control_characters_and_non_private_directories_are_rejected(self):
        with self.assertRaises(ChatTranscriptError):
            append_chat_message(self.path, role="user", content="bad\x1bmessage")
        self.directory.chmod(0o755)
        with self.assertRaises(ChatTranscriptError):
            append_chat_message(self.path, role="user", content="private only")

    def test_main_orchestrator_role_requires_the_server_marker(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main_orchestrator_instruction_section(), "")
        with patch.dict(os.environ, {MAIN_ORCHESTRATOR_ENV: "1"}, clear=True):
            instructions = main_orchestrator_instruction_section()
        self.assertIn("primary, persistent orchestrator", instructions)
        self.assertIn("/home/aday", instructions)
        self.assertIn("renter-first compute policy", instructions)
        self.assertIn("DO NOT\nauthorize creation", instructions)
        self.assertIn("explicitly tell you to start it", instructions)
        self.assertIn("can you make me an agent to take care of this site?", instructions)
        self.assertIn("do not call tools, create files, or change the site", instructions)
        self.assertIn("durable Nexus-managed Aeon instance by default", instructions)
        self.assertIn("Never reinterpret it as a Python script, Ollama application", instructions)
        self.assertIn("persistent memory is never evidence", instructions)
        self.assertIn("inspect the actual\nreturned observation", instructions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
