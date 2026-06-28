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


class TestLocalJsonRepair(unittest.TestCase):
    def setUp(self):
        self.c = _bare_llm_client()

    def test_trailing_comma_in_object(self):
        import json
        fixed = self.c._local_json_repair('{"a": 1, "b": 2,}')
        self.assertEqual(json.loads(fixed), {"a": 1, "b": 2})

    def test_trailing_comma_in_array(self):
        import json
        fixed = self.c._local_json_repair('{"a": [1, 2, 3,]}')
        self.assertEqual(json.loads(fixed), {"a": [1, 2, 3]})

    def test_python_literals(self):
        import json
        fixed = self.c._local_json_repair('{"a": True, "b": False, "c": None}')
        self.assertEqual(json.loads(fixed), {"a": True, "b": False, "c": None})

    def test_comma_inside_string_preserved(self):
        import json
        fixed = self.c._local_json_repair('{"a": "x, y, z", "b": 1,}')
        self.assertEqual(json.loads(fixed), {"a": "x, y, z", "b": 1})

    def test_literal_word_inside_string_preserved(self):
        import json
        fixed = self.c._local_json_repair('{"msg": "set flag to True now",}')
        self.assertEqual(json.loads(fixed), {"msg": "set flag to True now"})

    def test_unrepairable_returns_none(self):
        self.assertIsNone(self.c._local_json_repair('{"a": "unterminated'))


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


class TestNormalizeActions(unittest.TestCase):
    def _worker(self):
        from aeon.core.worker import Worker
        return Worker.__new__(Worker)

    def test_list_passthrough(self):
        w = self._worker()
        acts = [{"tool_name": "x", "parameters": {}}]
        self.assertEqual(w._normalize_actions(acts), acts)

    def test_single_dict_wrapped(self):
        w = self._worker()
        out = w._normalize_actions({"tool_name": "run_command", "parameters": {"command": "ls"}})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["tool_name"], "run_command")

    def test_key_aliases(self):
        w = self._worker()
        out = w._normalize_actions([{"tool": "run_command", "args": {"command": "ls"}}])
        self.assertEqual(out[0]["tool_name"], "run_command")
        self.assertEqual(out[0]["parameters"], {"command": "ls"})

    def test_dropping_non_dicts(self):
        w = self._worker()
        out = w._normalize_actions(["garbage", {"tool_name": "x"}, 5])
        self.assertEqual(len(out), 1)

    def test_dict_of_actions(self):
        w = self._worker()
        out = w._normalize_actions({"1": {"tool_name": "a"}, "2": {"tool_name": "b"}})
        self.assertEqual(len(out), 2)


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

    def test_signature_hint_lists_required_and_optional(self):
        from aeon.core.worker import Worker
        w = Worker.__new__(Worker)

        class T:
            def execute(self, file_path, content, mode='w'):
                pass

        w.tools = {"write_file": T()}
        hint = w._tool_signature_hint("write_file")
        self.assertIn("required: file_path, content", hint)
        self.assertIn("optional: mode", hint)
        self.assertEqual(w._tool_signature_hint("missing"), "")


class TestToolLoader(unittest.TestCase):
    def test_all_tool_modules_import_cleanly(self):
        # With empty deps, dependency-bearing tools skip silently; any entry in
        # errors_out therefore signals a genuine import/instantiation bug.
        from aeon.tools.loader import load_tools_from_directory
        errors = []
        tools = load_tools_from_directory('aeon.tools', dependencies={}, errors_out=errors)
        names = [t.name for t in tools]
        self.assertIn('run_command', names)
        self.assertEqual(errors, [], f"tool loader reported errors: {errors}")

    def test_instantiation_failure_is_reported(self):
        # Build a throwaway on-disk tool package with one tool whose __init__
        # raises, and confirm the loader reports it (instead of swallowing it).
        import tempfile
        import importlib
        from aeon.tools.loader import load_tools_from_directory

        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "broken_pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("")
            (pkg / "bad.py").write_text(
                "from aeon.tools.base import BaseTool\n"
                "class BadTool(BaseTool):\n"
                "    def __init__(self):\n"
                "        raise RuntimeError('boom')\n"
                "    def execute(self, *a, **k):\n"
                "        pass\n"
            )
            sys.path.insert(0, tmp)
            try:
                importlib.invalidate_caches()
                errors = []
                tools = load_tools_from_directory("broken_pkg", dependencies={}, errors_out=errors)
                self.assertEqual(tools, [])
                self.assertTrue(any("BadTool" in e or "instantiate" in e for e in errors), errors)
            finally:
                sys.path.remove(tmp)
                for m in list(sys.modules):
                    if m == "broken_pkg" or m.startswith("broken_pkg."):
                        del sys.modules[m]


class TestStrReplaceMatchLocations(unittest.TestCase):
    def _tool(self):
        from aeon.tools.file_io import StrReplaceTool
        return StrReplaceTool.__new__(StrReplaceTool)

    def test_lists_all_match_lines(self):
        t = self._tool()
        content = "a\nfoo\nb\nfoo\nc\nfoo\n"
        hint = t._match_locations(content, "foo")
        self.assertIn("2, 4, 6", hint)

    def test_no_match_empty(self):
        t = self._tool()
        self.assertEqual(t._match_locations("abc", "xyz"), "")

    def test_caps_with_more_suffix(self):
        t = self._tool()
        content = "x\n" * 20
        hint = t._match_locations(content, "x", max_show=3)
        self.assertIn("more)", hint)


class TestImageDimNormalization(unittest.TestCase):
    def setUp(self):
        from aeon.tools.generate_image import ComfyUITool
        self.f = ComfyUITool._norm_dim

    def test_passthrough_valid(self):
        self.assertEqual(self.f(1024), 1024)

    def test_string_number(self):
        self.assertEqual(self.f("512"), 512)

    def test_rounds_to_multiple(self):
        self.assertEqual(self.f(1000) % 16, 0)

    def test_clamps_high_and_low(self):
        self.assertEqual(self.f(5000), 2048)
        self.assertEqual(self.f(10), 256)

    def test_garbage_falls_back(self):
        self.assertEqual(self.f("abc"), 1024)
        self.assertEqual(self.f(None), 1024)


class TestBrowserSnapshotFormat(unittest.TestCase):
    """The structured element snapshot is the agent's primary, lossless view."""

    def _els(self):
        return [
            {"id": 1, "role": "link", "name": "Inbox", "states": [], "inViewport": True,
             "scrollContainer": None, "value": ""},
            {"id": 23, "role": "row", "name": "Jane — Project update", "states": ["collapsed"],
             "inViewport": True, "scrollContainer": 2, "value": ""},
            {"id": 50, "role": "button", "name": "Archive", "states": ["disabled"],
             "inViewport": False, "scrollContainer": None, "value": ""},
        ]

    def test_groups_in_view_vs_offscreen(self):
        from aeon.tools.browser import _format_elements
        out = _format_elements(self._els())
        self.assertIn("IN VIEW", out)
        self.assertIn("OFF-SCREEN", out)
        self.assertIn("[23] row", out)
        self.assertIn("(collapsed)", out)
        self.assertIn("scroll-group 2", out)

    def test_empty_elements_message(self):
        from aeon.tools.browser import _format_elements
        self.assertIn("no interactive elements", _format_elements([]).lower())

    def test_element_list_is_bounded(self):
        from aeon.tools.browser import _format_elements, MAX_ELEMENTS_CHARS
        big = [{"id": i, "role": "button", "name": "x" * 200, "states": [],
                "inViewport": True, "scrollContainer": None, "value": ""} for i in range(2000)]
        out = _format_elements(big)
        self.assertLessEqual(len(out), MAX_ELEMENTS_CHARS + 200)

    def test_scroll_state(self):
        from aeon.tools.browser import _format_scroll
        self.assertIn("more below", _format_scroll({"scrollY": 0, "scrollHeight": 3000, "clientHeight": 1000}))
        self.assertIn("fits in view", _format_scroll({"scrollY": 0, "scrollHeight": 800, "clientHeight": 1000}))


class TestOscillationLogic(unittest.TestCase):
    """Mirrors the worker's 2-cycle detection predicate."""
    @staticmethod
    def _osc(cmds, outs):
        if len(cmds) >= 4:
            a, b, c, d = list(zip(cmds[-4:], outs[-4:]))
            return a == c and b == d and a != b
        return False

    def test_abab_detected(self):
        self.assertTrue(self._osc(['A', 'B', 'A', 'B'], ['1', '2', '1', '2']))

    def test_steady_not_detected(self):
        self.assertFalse(self._osc(['A', 'A', 'A', 'A'], ['1', '1', '1', '1']))

    def test_progressing_not_detected(self):
        self.assertFalse(self._osc(['A', 'B', 'C', 'D'], ['1', '2', '3', '4']))


def load_tests(loader, standard_tests, pattern):
    return standard_tests


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
