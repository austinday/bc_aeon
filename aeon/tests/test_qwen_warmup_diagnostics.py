#!/usr/bin/env python3
"""Bounded diagnostics for the Qwen release warmup boundary."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from aeon.core import qwen_runtime as runtime
from aeon.scripts import warmup_qwen38_vllm as warmup
from aeon.tests.test_qwen_runtime import durable_runtime_state


class WarmupProcessDiagnosticsTests(unittest.TestCase):
    def _run_main(self, side_effect):
        output = io.StringIO()
        errors = io.StringIO()
        with tempfile.TemporaryFile(mode="w+b") as failure:
            with (
                patch.object(warmup, "_assert_staged_imports", return_value=None),
                patch.object(warmup, "warm", side_effect=side_effect),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(errors),
            ):
                result = warmup.main(
                    [
                        "--base-url",
                        "http://localhost:1",
                        "--model",
                        "qwen",
                        "--failure-fd",
                        str(failure.fileno()),
                    ]
                )
            failure.seek(0)
            receipt = failure.read().decode("ascii")
            mode = os.fstat(failure.fileno()).st_mode & 0o777
        return result, output.getvalue(), errors.getvalue(), receipt, mode

    def test_main_emits_only_the_allowlisted_failure_envelope(self):
        def fail_vision(_base_url, _model, *, include_image=False):
            if include_image:
                raise warmup.WarmupFailure(
                    "vision", "turn_action", "RAW_MODEL_OUTPUT_MUST_NOT_ESCAPE"
                )
            return {"completion_tokens": 1}

        result, output, errors, receipt, mode = self._run_main(fail_vision)

        self.assertEqual(result, 1)
        self.assertEqual(output, "")
        self.assertEqual(errors, "")
        self.assertEqual(
            json.loads(receipt),
            {"schema_version": 1, "stage": "vision", "code": "turn_action"},
        )
        self.assertEqual(mode, 0o600)
        self.assertNotIn("RAW_MODEL_OUTPUT", receipt)

    def test_unexpected_failure_is_reduced_to_a_safe_stage_code(self):
        result, output, errors, receipt, _mode = self._run_main(
            RuntimeError("RAW_EXCEPTION_AND_RESPONSE_BODY")
        )

        self.assertEqual(result, 1)
        self.assertEqual(output, "")
        self.assertEqual(errors, "")
        self.assertEqual(
            json.loads(receipt),
            {"schema_version": 1, "stage": "text", "code": "internal"},
        )
        self.assertNotIn("RAW_EXCEPTION", receipt)

    def test_transport_timeout_has_a_stable_text_code(self):
        with patch.object(
            warmup.requests,
            "post",
            side_effect=requests.Timeout("RAW_TRANSPORT_DETAIL"),
        ):
            with self.assertRaises(warmup.WarmupFailure) as raised:
                warmup.warm("http://localhost:1", "qwen")

        self.assertEqual((raised.exception.stage, raised.exception.code), ("text", "http_timeout"))
        self.assertNotIn("RAW_TRANSPORT_DETAIL", str(raised.exception))

    def test_wrong_action_has_a_stable_semantic_code(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "kind": "tool_calls",
                "intent": warmup.MARKER,
                "message": "",
                "actions": [],
            })}}],
        }
        with patch.object(warmup.requests, "post", return_value=response):
            with self.assertRaises(warmup.WarmupFailure) as raised:
                warmup.warm("http://localhost:1", "qwen", include_image=False)

        self.assertEqual((raised.exception.stage, raised.exception.code), ("text", "turn_action"))

    def test_canonical_turn_example_and_text_vision_requests_are_exact(self):
        expected_turn = {
            "kind": "tool_calls",
            "intent": warmup.MARKER,
            "message": "",
            "actions": [
                {
                    "tool_name": warmup.TOOL_NAME,
                    "parameters": {"reason": warmup.REASON},
                    "goal_refs": [],
                }
            ],
        }
        canonical = json.dumps(
            expected_turn,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        self.assertEqual(warmup.CANONICAL_TURN_JSON, canonical)
        self.assertEqual(json.loads(canonical), expected_turn)
        self.assertLessEqual(
            len(canonical.encode("ascii")), warmup.CANONICAL_TURN_MAX_BYTES
        )

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": canonical}}],
            "usage": {"completion_tokens": 7},
        }
        data_url = "data:image/png;base64,AA=="
        with (
            patch.object(warmup, "_vision_data_url", return_value=data_url),
            patch.object(warmup.requests, "post", return_value=response) as post,
        ):
            self.assertEqual(
                warmup.warm("http://localhost:1/", "qwen", include_image=False),
                {"completion_tokens": 7},
            )
            self.assertEqual(
                warmup.warm("http://localhost:1/", "qwen", include_image=True),
                {"completion_tokens": 7},
            )

        prompt = (
            "Verify the runtime is ready. Return only this exact JSON object with "
            "no additional fields or prose:\n"
            f"{canonical}"
        )
        expected_common = {
            "model": "qwen",
            "temperature": warmup.QWEN_CONTROL_TEMPERATURE,
            "top_p": warmup.QWEN_CONTROL_TOP_P,
            "top_k": warmup.QWEN_CONTROL_TOP_K,
            "min_p": 0.0,
            "reasoning_effort": "medium",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "aeon_runtime_warmup",
                    "strict": True,
                    "schema": warmup.SCHEMA,
                },
            },
            "seed": 1701,
            "max_tokens": 2048,
        }
        system_message = {
            "role": "system",
            "content": (
                "You are Aeon's local reasoner. Think privately, then return only "
                "the schema-constrained final object."
            ),
        }
        text_call, vision_call = post.call_args_list
        for call in (text_call, vision_call):
            self.assertEqual(call.args, ("http://localhost:1/v1/chat/completions",))
            self.assertEqual(call.kwargs["timeout"], (15, 240))
            self.assertEqual(
                {key: value for key, value in call.kwargs["json"].items() if key != "messages"},
                expected_common,
            )
        self.assertEqual(
            text_call.kwargs["json"]["messages"],
            [system_message, {"role": "user", "content": prompt}],
        )
        self.assertEqual(
            vision_call.kwargs["json"]["messages"],
            [
                system_message,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        self.assertEqual(prompt.count(canonical), 1)

    def test_turn_shape_failures_have_distinct_content_free_codes(self):
        complete = {
            "kind": "tool_calls",
            "intent": warmup.MARKER,
            "message": "",
            "actions": [],
        }
        cases = (
            ([], "turn_not_object"),
            ({"kind": "tool_calls"}, "turn_missing_required"),
            ({**complete, "RAW_SECRET_FIELD": "RAW_SECRET_VALUE"}, "turn_unexpected_fields"),
        )
        for parsed, expected_code in cases:
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "choices": [{"message": {"content": json.dumps(parsed)}}],
            }
            with self.subTest(code=expected_code), patch.object(
                warmup.requests, "post", return_value=response
            ), self.assertRaises(warmup.WarmupFailure) as raised:
                warmup.warm("http://localhost:1", "qwen")
            self.assertEqual(
                (raised.exception.stage, raised.exception.code),
                ("text", expected_code),
            )
            self.assertNotIn("RAW_", raised.exception.code)

    def test_success_leaves_the_failure_receipt_empty(self):
        output = io.StringIO()
        with (
            tempfile.TemporaryFile(mode="w+b") as failure,
            patch.object(warmup, "_assert_staged_imports", return_value=None),
            patch.object(
                warmup,
                "warm",
                side_effect=[
                    {"completion_tokens": 10},
                    {"completion_tokens": 20},
                ],
            ),
            contextlib.redirect_stdout(output),
        ):
            result = warmup.main(
                [
                    "--base-url",
                    "http://localhost:1",
                    "--model",
                    "qwen",
                    "--failure-fd",
                    str(failure.fileno()),
                ]
            )
            self.assertEqual(os.fstat(failure.fileno()).st_size, 0)
        self.assertEqual(result, 0)
        self.assertEqual(
            output.getvalue(),
            "QWEN38_WARMUP_OK text_completion_tokens=10 "
            "vision_completion_tokens=20\n",
        )

    def test_success_redacts_malformed_server_usage(self):
        output = io.StringIO()
        with (
            tempfile.TemporaryFile(mode="w+b") as failure,
            patch.object(warmup, "_assert_staged_imports", return_value=None),
            patch.object(
                warmup,
                "warm",
                side_effect=[
                    {"completion_tokens": "RAW_SERVER_VALUE"},
                    {"completion_tokens": -1},
                ],
            ),
            contextlib.redirect_stdout(output),
        ):
            result = warmup.main(
                [
                    "--base-url",
                    "http://localhost:1",
                    "--model",
                    "qwen",
                    "--failure-fd",
                    str(failure.fileno()),
                ]
            )
            self.assertEqual(os.fstat(failure.fileno()).st_size, 0)
        self.assertEqual(result, 0)
        self.assertEqual(
            output.getvalue(),
            "QWEN38_WARMUP_OK text_completion_tokens=0 "
            "vision_completion_tokens=0\n",
        )
        self.assertNotIn("RAW_", output.getvalue())


class RuntimeWarmupReceiptTests(unittest.TestCase):
    @staticmethod
    def _failure(stage: str, code: str, **extra):
        return {"schema_version": 1, "stage": stage, "code": code, **extra}

    @classmethod
    def _wire(cls, stage: str, code: str, **extra) -> str:
        return json.dumps(
            cls._failure(stage, code, **extra),
            sort_keys=True,
            separators=(",", ":"),
        )

    def test_runtime_accepts_every_declared_stage_code_pair(self):
        self.assertEqual(
            runtime._WARMUP_FAILURE_CODES_BY_STAGE,
            warmup.FAILURE_CODES_BY_STAGE,
        )
        self.assertEqual(
            runtime._WARMUP_FAILURE_SCHEMA_VERSION,
            warmup.FAILURE_SCHEMA_VERSION,
        )
        self.assertEqual(runtime._WARMUP_FAILURE_MAX_BYTES, warmup.FAILURE_MAX_BYTES)
        for stage, codes in warmup.FAILURE_CODES_BY_STAGE.items():
            for code in codes:
                with self.subTest(stage=stage, code=code):
                    self.assertEqual(
                        runtime._validated_warmup_failure(
                            self._failure(stage, code)
                        ),
                        {"schema_version": 1, "stage": stage, "code": code},
                    )

    def test_malformed_or_unreleased_diagnostics_use_one_safe_fallback(self):
        fallback = {
            "schema_version": 1,
            "stage": "runner",
            "code": "invalid_diagnostic",
        }
        malformed = (
            None,
            "",
            self._failure("text", "turn_action", raw="RAW_RESPONSE_BODY"),
            {"schema_version": True, "stage": "text", "code": "turn_action"},
            {"schema_version": 2, "stage": "text", "code": "turn_action"},
            self._failure("unknown", "turn_action"),
            self._failure("preflight", "http_timeout"),
            self._failure("text", "turn_envelope"),
            self._failure("t\u00e9xt", "turn_action"),
        )
        for value in malformed:
            with self.subTest(value=repr(value)[:80]):
                received = runtime._validated_warmup_failure(value)
                self.assertEqual(received, fallback)
                self.assertNotIn("RAW_", json.dumps(received))

    def test_private_reader_is_size_bounded_and_requires_canonical_json(self):
        expected = self._failure("text", "turn_action")
        malformed = (
            b"{",
            b"x" * 257,
            self._wire("text", "turn_action").encode() + b"\nRAW_TRAILING_OUTPUT",
            b'{"stage":"text","code":"turn_action","schema_version":1}',
            b'{"code":"turn_json","code":"turn_action","schema_version":1,"stage":"text"}',
            '{"code":"turn_action","schema_version":1,"stage":"t\u00e9xt"}'.encode(),
        )
        fallback = self._failure("runner", "invalid_diagnostic")
        with tempfile.TemporaryFile(mode="w+b") as valid:
            valid.write((self._wire("text", "turn_action") + "\n").encode())
            valid.flush()
            self.assertEqual(
                runtime._read_warmup_failure(valid.fileno()), expected
            )
        for index, payload in enumerate(malformed):
            with tempfile.TemporaryFile(mode="w+b") as receipt:
                receipt.write(payload)
                receipt.flush()
                with self.subTest(index=index):
                    received = runtime._read_warmup_failure(receipt.fileno())
                    self.assertEqual(received, fallback)
                    self.assertNotIn("RAW_", json.dumps(received))

    def test_recorded_failure_and_exception_contain_only_the_validated_contract(self):
        cases = (
            (
                self._failure("vision", "http_timeout"),
                {"schema_version": 1, "stage": "vision", "code": "http_timeout"},
            ),
            (
                self._failure("vision", "http_timeout", raw="RAW_MODEL_OUTPUT"),
                {"schema_version": 1, "stage": "runner", "code": "invalid_diagnostic"},
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            for index, (stdout, expected) in enumerate(cases):
                state_path = Path(temp) / str(index) / "runtime.json"
                with self.subTest(expected=expected), self.assertRaises(
                    runtime.QwenRuntimeError
                ) as raised:
                    runtime._record_warmup_failure(
                        {"phase": "launching", "updated_at": 1.0},
                        state_path,
                        stdout,
                    )
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["phase"], "launching")
                self.assertEqual(saved["warmup_failure"], expected)
                self.assertEqual(
                    str(raised.exception),
                    "Qwen structured release warmup failed "
                    f"[v1:{expected['stage']}:{expected['code']}]",
                )
                serialized = json.dumps(saved) + str(raised.exception)
                self.assertNotIn("RAW_MODEL_OUTPUT", serialized)

    def test_runner_boundary_sanitizes_nonzero_malformed_timeout_and_exec_errors(self):
        scenarios = (
            (
                "valid",
                self._failure("vision", "turn_action"),
                self._failure("vision", "turn_action"),
            ),
            (
                "malformed",
                b"RAW_MODEL_OUTPUT_AND_RESPONSE_BODY",
                self._failure("runner", "invalid_diagnostic"),
            ),
            ("timeout", None, self._failure("runner", "timeout")),
            ("exec_error", None, self._failure("runner", "exec_error")),
            (
                "result_mismatch",
                self._failure("vision", "turn_action"),
                self._failure("runner", "result_mismatch"),
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, (scenario, diagnostic, expected) in enumerate(scenarios):
                run_dir = root / f"run-{index}"
                run_dir.mkdir(mode=0o700)
                state_path = run_dir / "runtime.json"

                def runner(_command, **kwargs):
                    self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
                    self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
                    descriptor = kwargs["pass_fds"][0]
                    self.assertIn(str(descriptor), _command)
                    if scenario == "timeout":
                        raise subprocess.TimeoutExpired(
                            ["RAW_COMMAND_AND_ID"], 600, output="RAW_MODEL_OUTPUT"
                        )
                    if scenario == "exec_error":
                        raise OSError("RAW_PATH_AND_ID")
                    if isinstance(diagnostic, dict):
                        payload = (
                            json.dumps(
                                diagnostic, sort_keys=True, separators=(",", ":")
                            )
                            + "\n"
                        ).encode("ascii")
                    else:
                        payload = diagnostic
                    os.write(descriptor, payload)
                    os.fsync(descriptor)
                    return subprocess.CompletedProcess(
                        [], 0 if scenario == "result_mismatch" else 1
                    )

                with self.subTest(scenario=scenario), self.assertRaises(
                    runtime.QwenRuntimeError
                ) as raised:
                    runtime._run_structured_warmup(
                        ["RAW_COMMAND_AND_ID"],
                        cwd=run_dir,
                        environment={"RAW_ENV": "RAW_VALUE"},
                        receipt_dir=run_dir,
                        state={"phase": "launching", "updated_at": 1.0},
                        state_path=state_path,
                        command_runner=runner,
                    )
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["warmup_failure"], expected)
                serialized = json.dumps(saved) + str(raised.exception)
                self.assertNotIn("RAW_", serialized)
                formatted = "".join(
                    traceback.format_exception(
                        type(raised.exception),
                        raised.exception,
                        raised.exception.__traceback__,
                    )
                )
                self.assertNotIn("RAW_", formatted)

    def test_runner_boundary_redacts_diagnostic_write_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with (
                patch.object(
                    runtime,
                    "_private_json_write",
                    side_effect=OSError("RAW_RECEIPT_PATH"),
                ),
                self.assertRaises(runtime.QwenRuntimeError) as raised,
            ):
                runtime._run_structured_warmup(
                    ["RAW_COMMAND"],
                    cwd=root,
                    environment={},
                    receipt_dir=root,
                    state={"phase": "launching"},
                    state_path=root / "runtime.json",
                    command_runner=Mock(
                        side_effect=subprocess.TimeoutExpired(["RAW_COMMAND"], 600)
                    ),
                )
        self.assertEqual(
            str(raised.exception),
            "Qwen structured release warmup diagnostic write failed",
        )
        self.assertNotIn("RAW_", str(raised.exception))

    def test_runner_success_has_no_diagnostic_receipt_or_output_pipe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def runner(_command, **kwargs):
                self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
                self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
                self.assertEqual(
                    runtime._warmup_receipt_size(kwargs["pass_fds"][0]), 0
                )
                return subprocess.CompletedProcess([], 0)

            runtime._run_structured_warmup(
                ["warmup"],
                cwd=root,
                environment={},
                receipt_dir=root,
                state={"phase": "launching"},
                state_path=root / "runtime.json",
                command_runner=runner,
            )
            self.assertFalse((root / "runtime.json").exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_current_state_accepts_only_optional_exact_failure_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            state = {
                **durable_runtime_state(Path(temp)),
                "phase": "launching",
                "warmup_failure": self._failure("vision", "http_timeout"),
            }
            with patch.object(runtime, "_private_json_read", return_value=state):
                self.assertEqual(runtime.current_runtime_state(Path("/not/read")), state)
            releasing = {**state, "phase": "releasing"}
            with patch.object(runtime, "_private_json_read", return_value=releasing):
                self.assertEqual(
                    runtime.current_runtime_state(Path("/not/read")), releasing
                )
            malformed = (
                {**state, "phase": "ready"},
                {**state, "warmup_failure": self._failure("vision", "unknown")},
                {
                    **state,
                    "warmup_failure": {
                        "schema_version": True,
                        "stage": "vision",
                        "code": "http_timeout",
                    },
                },
                {
                    **state,
                    "warmup_failure": self._failure(
                        "vision", "http_timeout", raw="RAW_MODEL_OUTPUT"
                    ),
                },
            )
            for changed in malformed:
                with self.subTest(changed=changed["warmup_failure"]), patch.object(
                    runtime, "_private_json_read", return_value=changed
                ), self.assertRaisesRegex(
                    runtime.QwenRuntimeError, "warmup failure receipt is invalid"
                ):
                    runtime.current_runtime_state(Path("/not/read"))


if __name__ == "__main__":
    unittest.main()
