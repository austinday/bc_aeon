#!/usr/bin/env python3
"""
Unit tests for Aeon's pure-logic components.

These tests deliberately exercise ONLY the deterministic, model-free machinery
(JSON/block parsing, output truncation, token estimation, loop detection). They
must run fast with no GPU, no model server, and no network so they can be used
as a fast pre-restart gate alongside smoke_test.py.

Run with:  python3 -m aeon.tests.test_core
"""
import sys
import unittest
from pathlib import Path

# Ensure the local package wins over any installed copy.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _bare_llm_client():
    """Build an LLMClient without running __init__ (which needs a model config).

    The parsing/repair helpers we test only depend on self.logger, so we attach
    one and leave the rest unset.
    """
    from aeon.core.llm import LLMClient
    from aeon.core.logger import get_logger
    client = LLMClient.__new__(LLMClient)
    client.logger = get_logger()
    return client


class TestJsonExtraction(unittest.TestCase):
    def setUp(self):
        self.c = _bare_llm_client()

    def test_find_json_end_simple(self):
        raw = '{"a": 1}\ntrailing'
        end = self.c._find_json_end(raw)
        self.assertEqual(raw[:end], '{"a": 1}')

    def test_find_json_end_ignores_braces_in_strings(self):
        raw = '{"a": "}{ not real"}rest'
        end = self.c._find_json_end(raw)
        self.assertEqual(raw[:end], '{"a": "}{ not real"}')

    def test_find_json_end_nested(self):
        raw = '{"a": {"b": {"c": 1}}} after'
        end = self.c._find_json_end(raw)
        self.assertEqual(raw[:end], '{"a": {"b": {"c": 1}}}')

    def test_clean_json_strips_fences_and_think(self):
        raw = "<think>reasoning</think>```json\n{\"x\": 1}\n```"
        cleaned = self.c._clean_json_response(raw)
        import json
        self.assertEqual(json.loads(cleaned), {"x": 1})

    def test_clean_json_no_object(self):
        self.assertEqual(self.c._clean_json_response("no json here"), "{}")


class TestBlockSubstitution(unittest.TestCase):
    def setUp(self):
        self.c = _bare_llm_client()

    def test_v2_block_extraction_and_substitution(self):
        raw = (
            '{"intent": "x", "actions": [{"tool_name": "write_file", '
            '"parameters": {"content": "__BLOCK_1__"}}]}\n'
            '--- BEGIN BLOCK_1 ---\n'
            'line one\nline two\n'
            '--- END BLOCK_1 ---\n'
        )
        json_end = self.c._find_json_end(raw)
        blocks = self.c._extract_content_blocks(raw, json_end)
        self.assertIn('BLOCK_1', blocks)
        import json
        parsed = json.loads(raw[:json_end])
        missing = []
        parsed = self.c._substitute_blocks(parsed, blocks, missing)
        self.assertEqual(missing, [])
        self.assertEqual(
            parsed['actions'][0]['parameters']['content'], 'line one\nline two'
        )

    def test_missing_block_is_reported(self):
        parsed = {"actions": [{"parameters": {"content": "__BLOCK_9__"}}]}
        missing = []
        self.c._substitute_blocks(parsed, {}, missing)
        self.assertIn('BLOCK_9', missing)

    def test_braces_inside_block_not_parsed_as_json(self):
        raw = (
            '{"actions": [{"tool_name": "write_file", '
            '"parameters": {"content": "__BLOCK_1__"}}]}\n'
            '--- BEGIN BLOCK_1 ---\n'
            '{"nested": "json", "with": ["braces"]}\n'
            '--- END BLOCK_1 ---\n'
        )
        json_end = self.c._find_json_end(raw)
        blocks = self.c._extract_content_blocks(raw, json_end)
        self.assertEqual(blocks['BLOCK_1'], '{"nested": "json", "with": ["braces"]}')


class TestTruncation(unittest.TestCase):
    def test_short_text_untouched(self):
        from aeon.core.worker_utils import truncate_output
        self.assertEqual(truncate_output("hello", 100), "hello")

    def test_long_text_keeps_head_and_tail(self):
        from aeon.core.worker_utils import truncate_output
        text = "HEAD" + ("x" * 1000) + "TAILEND"
        out = truncate_output(text, 200)
        self.assertLessEqual(len(out), 200)
        self.assertTrue(out.startswith("HEAD"))
        self.assertTrue(out.endswith("TAILEND"))
        self.assertIn("TRUNCATED", out)


class TestTokenEstimation(unittest.TestCase):
    def test_returns_positive_int(self):
        from aeon.core.utils import estimate_tokens
        n = estimate_tokens("hello world, this is a test")
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_empty_string(self):
        from aeon.core.utils import estimate_tokens
        self.assertGreaterEqual(estimate_tokens(""), 0)

    def test_monotonic_with_length(self):
        from aeon.core.utils import estimate_tokens
        short = estimate_tokens("word " * 10)
        long = estimate_tokens("word " * 100)
        self.assertGreater(long, short)


class TestToolNameResolution(unittest.TestCase):
    def _worker(self):
        from aeon.core.worker import Worker
        w = Worker.__new__(Worker)
        w.tools = {"run_command": object(), "write_file": object(), "task_complete": object()}
        return w

    def test_exact_case_variant_autocorrects(self):
        w = self._worker()
        self.assertEqual(w._resolve_tool_name("Run_Command"), "run_command")
        self.assertEqual(w._resolve_tool_name("run-command"), "run_command")
        self.assertEqual(w._resolve_tool_name("WRITE FILE"), "write_file")

    def test_unknown_does_not_autocorrect(self):
        w = self._worker()
        self.assertIsNone(w._resolve_tool_name("frobnicate"))

    def test_suggestion_lists_close_match(self):
        w = self._worker()
        hint = w._suggest_tools("run_comand")
        self.assertIn("run_command", hint)

    def test_suggestion_when_no_match(self):
        w = self._worker()
        hint = w._suggest_tools("zzzzzz")
        self.assertIn("expand_tool_category", hint)


def load_tests(loader, standard_tests, pattern):
    return standard_tests


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
