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

# Pull in the self-modification / self-improvement substrate tests so they run as
# part of this same pre-restart gate (loadTestsFromModule discovers imported cases).
from aeon.tests.test_selfimprove import *  # noqa: F401,F403,E402


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


class TestLoopFingerprint(unittest.TestCase):
    """The loop guard keys on _consequential_fp. A weak model re-decorates its own
    tool call each turn (adds/drops tab_id=default, toggles compare, restates
    expected_text); those are incidental and must NOT mint a fresh fingerprint, or
    the repeat streak never reaches the hard block and a dead action spins forever
    (the real 'clicked Next forever, only ever soft-warned' failure)."""

    def _worker(self):
        from aeon.core.worker import Worker
        return Worker.__new__(Worker)

    def _click(self, **params):
        return [{"tool_name": "browser_interact",
                 "parameters": {"action": "click", "element_id": 6, **params}}]

    def test_incidental_params_share_one_fingerprint(self):
        w = self._worker()
        base = w._consequential_fp(self._click(expected_text="Next"))
        # Same click, re-decorated the ways the stuck transcript actually varied it.
        for variant in (
            self._click(expected_text="Next", tab_id="default"),   # tab_id=default added
            self._click(expected_text="Next", compare=True),        # compare toggled
            self._click(),                                          # expected_text dropped
            self._click(tab_id="default", include_vision=True, visual="overlay"),
        ):
            self.assertEqual(w._consequential_fp(variant), base)

    def test_meaningful_param_change_is_distinct(self):
        w = self._worker()
        # A different element / different action must remain a different fingerprint.
        self.assertNotEqual(w._consequential_fp(self._click()),
                            w._consequential_fp(
                                [{"tool_name": "browser_interact",
                                  "parameters": {"action": "click", "element_id": 7}}]))
        self.assertNotEqual(w._consequential_fp(self._click()),
                            w._consequential_fp(
                                [{"tool_name": "browser_interact",
                                  "parameters": {"action": "type", "element_id": 6,
                                                 "text": "x"}}]))

    def test_non_default_tab_id_is_kept(self):
        w = self._worker()
        self.assertNotEqual(w._consequential_fp(self._click(tab_id="default")),
                            w._consequential_fp(self._click(tab_id="gmail")))

    def test_streak_reaches_hard_block_over_transcript(self):
        # The decorated clicks from the stuck run must now count as one streak and
        # cross the 3x hard-block threshold instead of resetting to 2 each time.
        w = self._worker()
        clicks = [self._click(expected_text="Next"),
                  self._click(expected_text="Next", tab_id="default"),
                  self._click(expected_text="Next", compare=True)]
        fps = [w._consequential_fp(c) for c in clicks]
        streak = 0
        for fp in fps:
            streak = streak + 1 if fp == fps[-1] else 1
        self.assertGreaterEqual(streak, 3)

    def test_pure_read_turn_is_transparent(self):
        w = self._worker()
        self.assertEqual(
            w._consequential_fp([{"tool_name": "browser_read",
                                  "parameters": {"tab_id": "default"}}]), "")

    def _search(self, q):
        return [{"tool_name": "search_web", "parameters": {"query": q}}]

    def test_distinct_searches_have_distinct_structural_fp(self):
        # Regression: the structural fingerprint (semantic-stall detector) used to
        # collapse every verb-less call to the bare tool name, so three DIFFERENT
        # web searches looked like one repeated move and tripped 'semantic stall'.
        w = self._worker()
        fps = {w._structural_fp(self._search(q))
               for q in ("pizza NYC", "tokyo weather", "python asyncio")}
        self.assertEqual(len(fps), 3)

    def test_signup_varied_detail_still_collapses_structurally(self):
        # The intended catch must survive: a verb-FUL action (type) varying only an
        # incidental value (a fresh username) still shares one structural fingerprint.
        w = self._worker()
        def typ(u):
            return [{"tool_name": "browser_interact",
                     "parameters": {"action": "type", "element_id": 3, "text": u}}]
        self.assertEqual(w._structural_fp(typ("alice1")), w._structural_fp(typ("bob2")))


class TestGroundTruthOutcome(unittest.TestCase):
    """The attempt log must record what the tool output ACTUALLY showed, not the
    model's own summary of it — a stuck agent narrates 'clicked Next' for a click
    that did nothing, so a self-narrated log hides the very no-op it needs to see."""

    def _worker(self):
        from aeon.core.worker import Worker
        return Worker.__new__(Worker)

    def test_no_op_detected_only_when_consequential(self):
        from aeon.core.worker import Worker
        banner = "URL: x\n⚠ NO CHANGE: the URL and EVERY interactive element are identical..."
        self.assertIn("NO EFFECT", Worker._derive_ground_truth_outcome(banner, consequential=True))
        # A deliberate re-read is not a no-op even if the page didn't change.
        self.assertEqual(
            Worker._derive_ground_truth_outcome(
                "(No change since your last observation.)", consequential=False), "")

    def test_error_and_block_dominate(self):
        from aeon.core.worker import Worker
        self.assertTrue(Worker._derive_ground_truth_outcome(
            "COMMAND FAILED: boom", consequential=True).startswith("ERROR"))
        self.assertTrue(Worker._derive_ground_truth_outcome(
            "** COMMAND BLOCKED (loop guard) ...", consequential=True).startswith("BLOCKED"))

    def test_effective_action_yields_no_tag(self):
        from aeon.core.worker import Worker
        # A normal, effective action flags nothing -> caller keeps the model's note.
        self.assertEqual(Worker._derive_ground_truth_outcome(
            "URL: y\nTitle: Next page\n=== INTERACTIVE ELEMENTS ===", consequential=True), "")

    def test_loop_streak_appended(self):
        from aeon.core.worker import Worker
        out = Worker._derive_ground_truth_outcome(
            "⚠ NO CHANGE: ...", consequential=True, loop_detected=True, repeat_count=4)
        self.assertIn("repeated 4x", out)

    def test_collapse_survives_reworded_agent_notes(self):
        # The core fix: identical ground-truth Result collapses even when the model
        # reworded its subordinate note each turn (what used to defeat collapse).
        w = self._worker()
        noop = ("NO EFFECT — the action did NOT change the page (URL + interactive "
                "elements identical to before).")
        notes = ["clicked Next; still on Basic information.",
                 "page did not advance after clicking Next.",
                 "attempted Next again, remained on birthday screen."]
        entries = [
            f"[Iter {26+i}]\n- Intent: advance signup\n"
            f"- Actions: browser_interact(click [6])\n- Result: {noop}\n- Agent's note: {notes[i]}"
            for i in range(3)
        ]
        collapsed = w._collapse_repeated_entries(entries)
        self.assertEqual(len(collapsed), 1)
        self.assertIn("repeated 3x", collapsed[0])

    def test_distinct_results_do_not_collapse(self):
        w = self._worker()
        e1 = ("[Iter 1]\n- Intent: go\n- Actions: browser_interact(click [6])\n"
              "- Result: NO EFFECT — no change.\n- Agent's note: a")
        e2 = ("[Iter 2]\n- Intent: go\n- Actions: browser_interact(click [6])\n"
              "- Result: FORM STILL INVALID — Month unmet.\n- Agent's note: b")
        self.assertEqual(len(w._collapse_repeated_entries([e1, e2])), 2)


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


class TestSelfModAutoDerive(unittest.TestCase):
    """restart_aeon / verify_self_modification must resolve the agent's own source
    root themselves (never force the model to supply it, never install the wrong
    tree when run from a foreign workspace)."""

    def test_restart_default_dir_is_source_root(self):
        import os
        from aeon.tools.restart import RestartAeonTool
        d = RestartAeonTool(worker=object())._default_code_dir()
        self.assertTrue(d and os.path.exists(os.path.join(d, 'setup.py')),
                        f"restart default dir must contain setup.py: {d!r}")

    def test_verify_source_root_is_source_root(self):
        import os
        from aeon.tools.verify_modification import VerifySelfModificationTool
        d = VerifySelfModificationTool(worker=object())._aeon_source_root()
        self.assertTrue(d and os.path.exists(os.path.join(d, 'setup.py')),
                        f"verify source root must contain setup.py: {d!r}")


class TestCreateSkillGuard(unittest.TestCase):
    """create_skill must only accept safe single path components (no traversal)."""

    def test_rejects_unsafe(self):
        from aeon.tools.skills_runtime import _safe_component
        for bad in ("../etc", "a/b", "a\\b", "", ".", "..", "no space", ".hidden"):
            self.assertFalse(_safe_component(bad), f"should reject {bad!r}")

    def test_accepts_safe(self):
        from aeon.tools.skills_runtime import _safe_component
        for ok in ("research", "web_research", "api-migration", "v2.step"):
            self.assertTrue(_safe_component(ok), f"should accept {ok!r}")


class TestSkillCrudTools(unittest.TestCase):
    """Skills are self-modifiable: the full create/read/modify/delete surface must
    be present and its path handling must reject malformed input."""

    def test_crud_tools_are_discoverable(self):
        from aeon.tools.loader import load_tools_from_directory
        names = {t.name for t in load_tools_from_directory('aeon.tools', dependencies={})}
        for n in ('create_skill', 'read_skill', 'delete_skill', 'activate_skill', 'deactivate_skill'):
            self.assertIn(n, names, f"{n} must be discoverable by the loader")

    def test_read_delete_require_category_path(self):
        import types
        from aeon.tools.skills_runtime import ReadSkillTool, DeleteSkillTool
        w = types.SimpleNamespace(expanded_categories=set(), active_skill=None)
        self.assertTrue(ReadSkillTool(w).execute(skill_path="noslash").startswith("Error:"))
        self.assertTrue(DeleteSkillTool(w).execute(skill_path="noslash").startswith("Error:"))

    def test_delete_rejects_traversal(self):
        import types
        from aeon.tools.skills_runtime import DeleteSkillTool
        w = types.SimpleNamespace(expanded_categories=set(), active_skill=None)
        self.assertIn("invalid", DeleteSkillTool(w).execute(skill_path="../etc/passwd").lower())


class TestHumanMotion(unittest.TestCase):
    """The browser's human-motion math (curved eased mouse paths, wheel-notch
    scrolls, keystroke cadence) is Playwright-free and must behave correctly:
    trajectories end EXACTLY on target (clicks/drags land) while looking human."""

    def setUp(self):
        import random
        from aeon.services.browser import human_motion as hm
        self.hm = hm
        self.rng = random.Random(1234)  # deterministic

    def test_ease_endpoints_and_monotonic(self):
        e = self.hm.ease_in_out
        self.assertEqual(e(0.0), 0.0)
        self.assertEqual(e(1.0), 1.0)
        prev = -1.0
        for i in range(0, 101):
            v = e(i / 100)
            self.assertGreaterEqual(v, prev)  # non-decreasing
            prev = v

    def test_mouse_path_lands_exactly_on_target(self):
        for end in [(400, 300), (10, 900), (1270, 5)]:
            path = self.hm.mouse_path((640, 400), end, rng=self.rng)
            self.assertEqual(path[-1], (float(end[0]), float(end[1])),
                             "final point must be exactly the target")
            self.assertLessEqual(len(path), self.hm._MAX_STEPS + 2)  # +overshoot pair
            self.assertGreaterEqual(len(path), 1)

    def test_mouse_path_is_curved_not_straight(self):
        # A straight line would have every interior point collinear with the
        # endpoints; a human arc must deviate from the segment somewhere.
        start, end = (100.0, 100.0), (900.0, 700.0)
        path = self.hm.mouse_path(start, end, rng=self.rng)
        def dist_to_line(p):
            (x1, y1), (x2, y2), (px, py) = start, end, p
            num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
            import math
            return num / math.hypot(y2 - y1, x2 - x1)
        self.assertTrue(any(dist_to_line(p) > 2.0 for p in path[:-1]),
                        "path should arc off the straight segment")

    def test_mouse_path_tiny_move_is_single_point(self):
        self.assertEqual(self.hm.mouse_path((10, 10), (10, 10), rng=self.rng), [(10.0, 10.0)])

    def test_scroll_ticks_sum_and_sign(self):
        for total in (600, -600, 150, -30, 2000):
            ticks = self.hm.scroll_ticks(total, rng=self.rng)
            self.assertEqual(sum(ticks), total, f"ticks must sum to {total}")
            self.assertTrue(all((t >= 0) == (total >= 0) for t in ticks),
                            "every tick shares the scroll's sign")
        self.assertEqual(self.hm.scroll_ticks(0, rng=self.rng), [])
        self.assertEqual(len(self.hm.scroll_ticks(120, rng=self.rng)), 1)  # small = one notch
        self.assertGreater(len(self.hm.scroll_ticks(1000, rng=self.rng)), 1)  # long = several

    def test_type_delays_shape(self):
        d = self.hm.type_delays("hello world", rng=self.rng)
        self.assertEqual(len(d), len("hello world"))
        self.assertTrue(all(x > 0 for x in d))

    def test_type_delays_is_human_speed_not_superhuman(self):
        # Mean per-char delay should imply a human typing speed, not faster than
        # the ~216 WPM human record. WPM = 60/(mean_sec_per_char*5).
        text = "the quick brown fox jumps over the lazy dog several times over"
        d = self.hm.type_delays(text, rng=self.rng)
        mean = sum(d) / len(d)
        wpm = 60.0 / (mean * 5)
        self.assertLess(wpm, 180, f"typing too fast to be human: {wpm:.0f} WPM")
        self.assertGreater(wpm, 40, f"implausibly slow: {wpm:.0f} WPM")

    def test_idle_drift_stays_on_screen_and_near(self):
        import math
        for cur in [(960, 540), (5, 5), (1915, 1075)]:
            for _ in range(50):
                nx, ny = self.hm.idle_drift_target(cur, 1920, 1080, rng=self.rng)
                self.assertTrue(8 <= nx <= 1912 and 8 <= ny <= 1072, "must stay on-screen with margin")
        # From a center point (no clamping), drift is a small wander (<= ~140px).
        for _ in range(50):
            nx, ny = self.hm.idle_drift_target((960, 540), 1920, 1080, rng=self.rng)
            self.assertLessEqual(math.hypot(nx - 960, ny - 540), 141)


class TestMultimodalPerception(unittest.TestCase):
    """The browser now hands the real screenshot to the deciding model. Verify the
    plumbing: the user turn becomes multimodal only when images are present, and
    the worker holds/consumes the current view without accumulating frames."""

    def _client(self):
        # Build an LLMClient without __init__ (needs a model config); the content
        # builder only touches logger + the encoder.
        from aeon.core.llm import LLMClient
        import logging
        c = LLMClient.__new__(LLMClient)
        c.logger = logging.getLogger("test")
        return c

    def test_text_only_when_no_images(self):
        c = self._client()
        self.assertEqual(c._build_user_content("hello", None), "hello")
        self.assertEqual(c._build_user_content("hello", []), "hello")

    def test_multimodal_when_image_encodes(self):
        import io, os, tempfile
        from PIL import Image
        c = self._client()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shot.jpg")
            Image.new("RGB", (64, 48), (10, 20, 30)).save(p, "JPEG")
            content = c._build_user_content("look", [p])
            self.assertIsInstance(content, list)
            self.assertEqual(content[0], {"type": "text", "text": "look"})
            self.assertEqual(content[1]["type"], "image_url")
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_bad_image_falls_back_to_text(self):
        c = self._client()
        # Non-existent path -> encoder returns None -> plain string, never raises.
        self.assertEqual(c._build_user_content("look", ["/no/such/file.png"]), "look")

    def test_jpeg_passthrough_no_reencode(self):
        # A right-sized JPEG (the browser case) must be base64'd verbatim — no
        # second lossy re-encode. A PNG must be converted to JPEG (bytes differ).
        import base64, os, tempfile
        from PIL import Image
        c = self._client()
        with tempfile.TemporaryDirectory() as d:
            jp = os.path.join(d, "shot.jpg")
            Image.new("RGB", (200, 120), (30, 60, 90)).save(jp, "JPEG", quality=90)
            with open(jp, "rb") as f:
                raw = f.read()
            url = c._encode_image_data_url(jp)
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), raw, "JPEG should pass through unchanged")

            pn = os.path.join(d, "shot.png")
            Image.new("RGB", (200, 120), (30, 60, 90)).save(pn, "PNG")
            purl = c._encode_image_data_url(pn)
            self.assertTrue(purl.startswith("data:image/jpeg;base64,"), "PNG should be re-encoded to JPEG")

    def test_worker_visual_context_replaces_not_accumulates(self):
        from aeon.core.worker import Worker
        w = Worker.__new__(Worker)
        w.visual_context = []
        w.set_visual_context("/tmp/a.jpg")
        w.set_visual_context(["/tmp/b.jpg", "/tmp/c.jpg"])  # replace by default
        self.assertEqual(w.visual_context, ["/tmp/b.jpg", "/tmp/c.jpg"])
        w.set_visual_context("/tmp/d.jpg", replace=False)   # explicit append
        self.assertEqual(w.visual_context, ["/tmp/b.jpg", "/tmp/c.jpg", "/tmp/d.jpg"])
        w.set_visual_context([None, ""])                    # empties are dropped
        self.assertEqual(w.visual_context, [])


class TestBrowserUtil(unittest.TestCase):
    """Pure browser-service decision logic (proxy parsing, destructive-dialog
    detection, timezone/locale handling). This is the code that can't be exercised
    live without the container, so it's especially important to unit-test."""

    def setUp(self):
        from aeon.services.browser import browser_util as bu
        self.bu = bu

    def test_parse_proxy(self):
        p = self.bu.parse_proxy("http://user:pass@1.2.3.4:8080")
        self.assertEqual(p, {"server": "http://1.2.3.4:8080", "username": "user", "password": "pass"})
        self.assertEqual(self.bu.parse_proxy("socks5://host:1080"), {"server": "socks5://host:1080"})
        self.assertIsNone(self.bu.parse_proxy(""))
        self.assertIsNone(self.bu.parse_proxy("   "))
        self.assertIsNone(self.bu.parse_proxy("not a url"))  # no hostname

    def test_destructive_dialog(self):
        for m in ["Delete this item?", "This CANNOT be undone", "Discard unsaved changes?",
                  "Are you sure you want to remove it"]:
            self.assertTrue(self.bu.is_destructive_dialog(m), m)
        for m in ["Reload site?", "Allow notifications?", "", "Please confirm your email"]:
            self.assertFalse(self.bu.is_destructive_dialog(m), m)

    def test_valid_timezone(self):
        self.assertTrue(self.bu.valid_timezone("America/New_York"))
        self.assertTrue(self.bu.valid_timezone("Europe/Berlin"))
        self.assertFalse(self.bu.valid_timezone("Mars/Olympus"))
        self.assertFalse(self.bu.valid_timezone(""))
        self.assertFalse(self.bu.valid_timezone(None))

    def test_primary_locale(self):
        self.assertEqual(self.bu.primary_locale("en-US,haw,fr"), "en-US")
        self.assertEqual(self.bu.primary_locale("de-DE"), "de-DE")
        self.assertEqual(self.bu.primary_locale("", "en-US"), "en-US")
        self.assertIsNone(self.bu.primary_locale(""))


class TestActionSchema(unittest.TestCase):
    """The turn schema handed to the server for grammar-constrained decoding."""

    def setUp(self):
        from aeon.core.action_schema import build_turn_schema
        self.schema = build_turn_schema(["run_command", "write_file", "think"])

    def test_required_fields_present(self):
        from aeon.core.action_schema import TURN_FIELDS_REQUIRED
        self.assertEqual(self.schema["required"], TURN_FIELDS_REQUIRED)
        for f in TURN_FIELDS_REQUIRED:
            self.assertIn(f, self.schema["properties"])

    def test_thought_generated_first(self):
        # xgrammar emits properties in schema order: reasoning must precede actions.
        self.assertEqual(next(iter(self.schema["properties"])), "thought")

    def test_updated_plan_optional(self):
        self.assertIn("updated_plan", self.schema["properties"])
        self.assertNotIn("updated_plan", self.schema["required"])

    def test_tool_name_enum_matches_tools(self):
        item = self.schema["properties"]["actions"]["items"]
        self.assertEqual(item["properties"]["tool_name"]["enum"],
                         ["run_command", "think", "write_file"])

    def test_envelope_closed_parameters_open(self):
        # Envelope/action: strictly closed. Tool parameters: free-form object.
        item = self.schema["properties"]["actions"]["items"]
        self.assertFalse(self.schema["additionalProperties"])
        self.assertFalse(item["additionalProperties"])
        self.assertTrue(item["properties"]["parameters"]["additionalProperties"])

    def test_no_tools_gives_unconstrained_name(self):
        from aeon.core.action_schema import build_turn_schema
        s = build_turn_schema([])
        self.assertNotIn("enum", s["properties"]["actions"]["items"]["properties"]["tool_name"])

    def test_schema_is_json_serializable(self):
        import json as _json
        _json.dumps(self.schema)


class TestStructuredRequestModes(unittest.TestCase):
    """Structured-output request construction + graceful downgrade tiers."""

    def setUp(self):
        self.c = _bare_llm_client()
        from aeon.core.action_schema import build_turn_schema
        self.c.action_schema = build_turn_schema(["run_command"])
        self.c._structured_mode = None

    def test_default_mode_uses_response_format(self):
        kw = self.c._structured_request_kwargs()
        self.assertIn("response_format", kw)
        self.assertEqual(kw["response_format"]["type"], "json_schema")
        self.assertIs(kw["response_format"]["json_schema"]["schema"], self.c.action_schema)

    def test_guided_json_mode(self):
        self.c._structured_mode = "guided_json"
        kw = self.c._structured_request_kwargs()
        self.assertNotIn("response_format", kw)
        self.assertIs(kw["extra_body"]["guided_json"], self.c.action_schema)

    def test_legacy_mode_returns_none(self):
        self.c._structured_mode = "legacy"
        self.assertIsNone(self.c._structured_request_kwargs())

    def test_no_schema_returns_none(self):
        self.c.action_schema = None
        self.assertIsNone(self.c._structured_request_kwargs())

    def test_downgrade_ladder(self):
        err = Exception("response_format 'json_schema' is not supported")
        self.assertTrue(self.c._downgrade_structured_mode(err))
        self.assertEqual(self.c._structured_mode, "guided_json")
        err2 = Exception("guided_json is not a valid parameter")
        self.assertTrue(self.c._downgrade_structured_mode(err2))
        self.assertEqual(self.c._structured_mode, "legacy")
        # Fully downgraded: nothing further to try.
        self.assertFalse(self.c._downgrade_structured_mode(err2))

    def test_unrelated_error_does_not_downgrade(self):
        err = Exception("context length exceeded")
        self.assertFalse(self.c._downgrade_structured_mode(err))
        self.assertIsNone(self.c._structured_mode)

    def test_set_schema_reprobes_after_legacy(self):
        self.c._structured_mode = "legacy"
        self.c.set_action_schema(self.c.action_schema)
        self.assertIsNone(self.c._structured_mode)


class TestStructuredEndToEnd(unittest.TestCase):
    """get_primary_agent_response with a stubbed server: the structured fast
    path parses directly (no repair machinery), and a max_tokens truncation
    triggers a retry with a terseness note instead of a parse attempt."""

    @staticmethod
    def _chunks(text, finish_reason="stop"):
        from types import SimpleNamespace as NS
        return [
            NS(choices=[NS(delta=NS(content=text), finish_reason=None)], usage=None),
            NS(choices=[NS(delta=NS(content=None), finish_reason=finish_reason)], usage=None),
            NS(choices=[], usage=NS(completion_tokens=42)),
        ]

    def _client_returning(self, batches):
        """A bare LLMClient whose chat.completions.create pops from batches
        (each batch = (text, finish_reason)) and records the request kwargs."""
        from types import SimpleNamespace as NS
        c = _bare_llm_client()
        from aeon.core.action_schema import build_turn_schema
        c.action_schema = build_turn_schema(["run_command"])
        c._structured_mode = None
        c.debug_path = None
        c.model = c.api_model = "stub"
        c._vision_supported = True
        c.requests = []

        def create(**kwargs):
            c.requests.append(kwargs)
            text, fr = batches.pop(0)
            return iter(self._chunks(text, fr))

        c.client = NS(chat=NS(completions=NS(create=create)))
        return c

    def test_structured_fast_path(self):
        import json as _json
        good = ('{"thought": "t", "previous_result_summary": "N/A", "skill_check": "No matching skill.", '
                '"memory_check": "Nothing new.", "parallel_check": "Sequential: no parallelism available.", '
                '"intent": "run it", "actions": [{"tool_name": "run_command", '
                '"parameters": {"command": "echo hi"}}]}')
        c = self._client_returning([(good, "stop")])
        out = c.get_primary_agent_response("PROMPT")
        data = _json.loads(out)
        self.assertEqual(data["actions"][0]["tool_name"], "run_command")
        # The request actually asked for grammar-constrained decoding...
        self.assertIn("response_format", c.requests[0])
        # ...and did NOT send the JSON-corrupting accumulating penalty.
        self.assertNotIn("frequency_penalty", c.requests[0])

    def test_truncation_retries_with_terseness_note(self):
        import json as _json
        good = ('{"thought": "t", "previous_result_summary": "N/A", "skill_check": "No matching skill.", '
                '"memory_check": "Nothing new.", "parallel_check": "Sequential: no parallelism available.", '
                '"intent": "run it", "actions": [{"tool_name": "run_command", '
                '"parameters": {"command": "echo hi"}}]}')
        c = self._client_returning([('{"thought": "endless...', "length"), (good, "stop")])
        out = c.get_primary_agent_response("PROMPT")
        self.assertEqual(_json.loads(out)["intent"], "run it")
        self.assertEqual(len(c.requests), 2)
        retry_prompt = c.requests[1]["messages"][0]["content"]
        self.assertIn("CUT OFF", retry_prompt)


class TestSubAgentReportIntegrity(unittest.TestCase):
    """The principal reads a sub-agent's deliverable from output.json. Two bugs
    guarded here: (1) say_to_user must stash its message on the worker (the
    wrapper's report source — last_observation never contains it), and
    (2) kill_sub_agent must NOT clobber a finished agent's output.json."""

    def test_say_to_user_stashes_message_on_worker(self):
        import types
        from aeon.tools.communication import SayToUserTool
        w = types.SimpleNamespace(last_say_to_user=None)
        out = SayToUserTool(worker=w).execute("final findings: all good")
        self.assertIn("delivered", out.lower())
        self.assertEqual(w.last_say_to_user, "final findings: all good")

    def test_kill_does_not_clobber_completed_report(self):
        import json as _json
        import os
        import tempfile
        import types
        from pathlib import Path
        from aeon.tools.sub_agent import KillSubAgent
        from aeon.core import runtime_signals as rt

        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "aeon_output" / "inst" / "sub_agents"
            agent_dir = base / "abcd1234-0000-0000-0000-000000000000"
            agent_dir.mkdir(parents=True)
            rt.atomic_write_json(agent_dir / "output.json", {
                "agent_id": agent_dir.name, "status": "COMPLETED",
                "result": "THE REPORT",
            })
            rt.atomic_write_text(agent_dir / "status.txt", "COMPLETED")

            tool = KillSubAgent(worker=types.SimpleNamespace(
                instance_id="inst", notified_sub_agents=set()))
            old_cwd = os.getcwd()
            os.chdir(td)  # output_dir resolves relative to cwd
            try:
                out = tool.execute("abcd1234")
            finally:
                os.chdir(old_cwd)
            self.assertIn("already finished", out.lower())
            data = _json.loads((agent_dir / "output.json").read_text())
            self.assertEqual(data["result"], "THE REPORT")
            self.assertEqual(data["status"], "COMPLETED")


class TestTokenCalibration(unittest.TestCase):
    """estimate_tokens self-calibrates from the server's real prompt_tokens.
    The EMA must move toward the observed ratio, ignore absurd/short samples,
    and never poison future estimates."""

    def setUp(self):
        from aeon.core.utils import tokens
        tokens._reset_calibration()

    def tearDown(self):
        from aeon.core.utils import tokens
        tokens._reset_calibration()

    def test_calibration_scales_estimates(self):
        from aeon.core.utils import tokens
        text = "some representative agent context " * 200  # comfortably > 500 raw tokens
        base = tokens.estimate_tokens(text)
        tokens.calibrate(text, int(tokens._raw_estimate(text) * 2))  # server says 2x
        scaled = tokens.estimate_tokens(text)
        self.assertGreater(scaled, base)  # EMA moved toward 2x
        self.assertLess(scaled, base * 2)  # but not all the way in one step

    def test_absurd_and_tiny_samples_ignored(self):
        from aeon.core.utils import tokens
        text = "word " * 2000
        base = tokens.estimate_tokens(text)
        tokens.calibrate(text, int(tokens._raw_estimate(text) * 50))  # image-inflated ratio
        tokens.calibrate("short", 10_000)                             # sample too small
        self.assertEqual(tokens.estimate_tokens(text), base)


class TestSensitiveMemoryGuard(unittest.TestCase):
    """Sensitive memories must be exempt from LLM memory compression (a
    paraphrased password is silent data loss)."""

    def test_key_and_category_markers(self):
        from aeon.core.worker import Worker
        sens = Worker._is_sensitive_memory
        self.assertTrue(sens("github_password", "hunter2"))
        self.assertTrue(sens("api_key_openrouter", {"value": "sk-x", "category": "general"}))
        self.assertTrue(sens("proton_details", {"value": "u/p", "category": "credentials"}))
        self.assertFalse(sens("project_paths", {"value": "/data", "category": "general"}))
        self.assertFalse(sens("build_command", "make -j8"))


class TestFittedGpuMemUtil(unittest.TestCase):
    """A single-copy (solo/dual) plan must size gpu_memory_utilization to
    weights + KV(ctx) + headroom, NOT fill the whole card — otherwise vLLM
    pre-allocates a huge KV pool a single agent never uses. split/offload, which
    legitimately fill the GPUs, keep the launcher's tier default."""

    def _gpus(self, n=2, gib=95.6):
        from aeon.core.gpu import GpuInfo
        return [GpuInfo(index=i, name="test", total_gib=gib, free_gib=gib) for i in range(n)]

    def test_solo_util_fits_footprint_not_whole_card(self):
        from aeon.core import model_catalog as c, deploy_planner as dp
        gpus = self._gpus()
        e = c.by_name("Qwen3.6-27B-Huihui-NVFP4-MTP")
        p = dp.plan(e, gpus, mode="solo")
        self.assertEqual(p.tier, "solo")
        util = float(p.env["AEON_GPU_MEM_UTIL"])
        reserved = util * gpus[0].total_gib
        kv = p.context_limit * e.kv_gib_per_64k / 65536.0
        expected = e.weights_gib + kv + dp.KV_POOL_HEADROOM_GIB
        self.assertAlmostEqual(reserved, expected, delta=0.5)
        # Well below the old flat 0.85 fill, and the KV pool still holds full ctx.
        self.assertLess(util, 0.6)
        self.assertGreaterEqual(reserved - e.weights_gib, kv)

    def test_dual_util_is_fitted_per_gpu(self):
        from aeon.core import model_catalog as c, deploy_planner as dp
        gpus = self._gpus()
        e = c.by_name("Qwen3.6-27B-Huihui-NVFP4-MTP")
        p = dp.plan(e, gpus, mode="dual")
        self.assertEqual(p.tier, "dual")
        self.assertTrue(p.env["AEON_GPU_MEM_UTIL"])
        self.assertLess(float(p.env["AEON_GPU_MEM_UTIL"]), 0.6)

    def test_split_keeps_tier_default(self):
        from aeon.core import model_catalog as c, deploy_planner as dp
        e = c.by_name("Qwen3.5-397B-A17B-Q3K")  # force_split flagship
        p = dp.plan(e, self._gpus())
        self.assertIn(p.tier, ("split", "offload"))
        self.assertEqual(p.env["AEON_GPU_MEM_UTIL"], "")

    def test_util_never_exceeds_safety_cap(self):
        from aeon.core import model_catalog as c, deploy_planner as dp
        # Tiny cards force the footprint above the cap -> util clamps to SAFETY.
        e = c.by_name("Qwen3.6-27B-Huihui-NVFP4-MTP")
        p = dp.plan(e, self._gpus(gib=40.0), mode="solo")
        if p.tier == "solo":
            self.assertLessEqual(float(p.env["AEON_GPU_MEM_UTIL"]), dp.SAFETY)


class TestMessageHistoryMode(unittest.TestCase):
    """Opt-in message-history mode: a stable system message + a growing turn
    history + one volatile current-state message. Verifies the split (static vs
    volatile), history seeding/append/trim, and the LLM messages path."""

    def _worker(self):
        import types
        from aeon.core.worker import Worker
        w = Worker.__new__(Worker)
        w.important_reminders = "REMINDERVAL"
        w.base_directives = "BASEVAL"
        w.docker_directives = "DOCKVAL"
        w.current_plan = "PLANVAL_XZ"
        w.last_observation = "LASTOBS_XZ"
        w._stuck_banner = ""
        w.active_skill = None
        w._history_messages = []
        w._history_seeded = False
        w.action_log = []
        w.pending_iteration_state = None
        w._get_skills_description = lambda: "SKILLSVAL"
        w._format_active_skill = lambda: ""
        w.llm_client = types.SimpleNamespace(context_limit=100000)
        return w

    def test_system_static_vs_current_state_volatile(self):
        w = self._worker()
        sm = w._build_system_message("MY_OBJECTIVE_XZ", "TOOLSVAL", "TOOLDIRVAL")
        self.assertIn("BASEVAL", sm)
        self.assertIn("TOOLSVAL", sm)
        self.assertIn("MY_OBJECTIVE_XZ", sm)
        # Volatile values must NOT be in the (cacheable) system message.
        self.assertNotIn("LASTOBS_XZ", sm)
        self.assertNotIn("PLANVAL_XZ", sm)

        cm = w._build_current_state_message("TREEVAL", "STATSVAL", "MEMVAL", "FILESVAL",
                                            sub_agent_digest="SUBVAL")
        for token in ("TREEVAL", "MEMVAL", "PLANVAL_XZ", "FILESVAL", "LASTOBS_XZ", "STATSVAL",
                      "SUBVAL", "NEXT ACTION"):
            self.assertIn(token, cm)

    def test_history_seed_from_action_log(self):
        w = self._worker()
        w.action_log = ["[Iter 1]\n- Intent: x\n- Actions: a\n- Result: ok"]
        w._ensure_history_seeded()
        self.assertEqual(len(w._history_messages), 1)
        self.assertIn("EARLIER WORK", w._history_messages[0]["content"])
        # Idempotent.
        w._ensure_history_seeded()
        self.assertEqual(len(w._history_messages), 1)

    def test_append_turn_records_decision_and_brief_result(self):
        w = self._worker()
        resp = {"thought": "th", "intent": "do the thing",
                "actions": [{"tool_name": "run_command", "parameters": {"command": "x"}}]}
        w._append_history_turn(resp, "R" * 5000)
        self.assertEqual(len(w._history_messages), 2)
        self.assertEqual(w._history_messages[0]["role"], "assistant")
        self.assertIn("do the thing", w._history_messages[0]["content"])
        self.assertEqual(w._history_messages[1]["role"], "user")
        self.assertLess(len(w._history_messages[1]["content"]), 5000)  # brief result truncated

    def test_trim_history_bounds_and_notes(self):
        w = self._worker()
        for i in range(50):
            w._history_messages.append({"role": "user", "content": "x" * 4000})
        w._trim_history(max_tokens=2000)
        joined = " ".join(m["content"] for m in w._history_messages)
        self.assertIn("trimmed", joined)
        # Bounded well below the original 50 messages.
        self.assertLess(len(w._history_messages), 20)

    def test_llm_uses_message_list_and_keeps_prefix_stable(self):
        import json as _json
        import logging
        import types
        from aeon.core.llm import LLMClient
        c = LLMClient.__new__(LLMClient)
        c.logger = logging.getLogger("t")
        c.model = "M"
        c.api_model = "M"
        c.action_schema = None
        c._structured_mode = "legacy"
        c._vision_supported = True
        c.debug_path = None
        c.context_limit = 200000
        good = ('{"thought":"t","previous_result_summary":"n","skill_check":"none",'
                '"memory_check":"none","parallel_check":"none","intent":"go",'
                '"actions":[{"tool_name":"run_command","parameters":{"command":"echo hi"}}]}')
        captured = {}

        def mk(o):
            return types.SimpleNamespace(**o)

        class Stream:
            def __iter__(self):
                yield mk({"choices": [mk({"delta": mk({"content": good}), "finish_reason": "stop"})], "usage": None})
                yield mk({"choices": [], "usage": mk({"completion_tokens": 5, "prompt_tokens": 100,
                                                      "prompt_tokens_details": None})})

        class Chat:
            class completions:
                @staticmethod
                def create(**k):
                    captured["messages"] = k["messages"]
                    return Stream()

        c.client = types.SimpleNamespace(chat=Chat())
        msgs = [{"role": "system", "content": "SYS"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "STATE"}]
        out = c.get_primary_agent_response(messages=msgs)
        self.assertEqual(_json.loads(out)["intent"], "go")
        sent = captured["messages"]
        self.assertEqual(sent[0], {"role": "system", "content": "SYS"})
        self.assertEqual(sent[1], {"role": "assistant", "content": "A1"})
        self.assertEqual(sent[-1]["role"], "user")
        self.assertIn("STATE", LLMClient._msg_text(sent[-1]))


class TestInterruptionFieldCoercion(unittest.TestCase):
    """The interruption/resume integrators are asked for string fields but LLMs
    frequently return a LIST (e.g. `plan` as an array of steps). Calling .strip()
    on that raised 'list object has no attribute strip' INSIDE the Ctrl+C handler,
    which killed the whole session. Fields must be coerced, never crash."""

    def test_coerce_text_handles_list_dict_none(self):
        from aeon.core.worker import Worker
        c = Worker._coerce_text
        self.assertEqual(c(None), "")
        self.assertEqual(c("  hi  "), "hi")
        self.assertEqual(c(["step 1", "step 2"]), "step 1\nstep 2")
        self.assertEqual(c([]), "")
        self.assertEqual(c({"a": "x", "b": "y"}), "a: x\nb: y")
        self.assertEqual(c(42), "42")

    def test_guidance_with_list_plan_does_not_crash(self):
        # Reproduces the reported fatal: Ctrl+C guidance -> integrator returns
        # `plan` as a list. Must integrate cleanly instead of raising.
        import os
        import tempfile
        from aeon.core.worker import Worker

        class _StubLLM:
            def integrate_interruption(self, obj, plan, progress, inp):
                return {"mode": "REVISE",
                        "objective": "Refactor the parser and add tests",
                        "plan": ["Write failing tests", "Refactor", "Make tests pass"],
                        "directive": ["Do the tests first", "then refactor"],
                        "reasoning": "user steered toward tests"}

        w = Worker.__new__(Worker)
        w.llm_client = _StubLLM()
        w.current_plan = "old plan"
        w.action_log = []
        w.last_observation = ""
        w.pending_iteration_state = None
        w.print_func = lambda *a, **k: None
        with tempfile.TemporaryDirectory() as td:
            old = os.getcwd()
            os.chdir(td)
            try:
                obj, reset = w._integrate_user_input("Old objective", "focus on tests", 5)
            finally:
                os.chdir(old)
        self.assertEqual(obj, "Refactor the parser and add tests")
        # The list plan was joined into a multi-line string, not crashed on.
        self.assertEqual(w.current_plan, "Write failing tests\nRefactor\nMake tests pass")
        self.assertIn("Do the tests first", w.last_observation)


class TestResumePreviousSession(unittest.TestCase):
    """A stopped session writes a resumable dump; the resume_previous_session tool
    reads it and sets the loop up to continue the prior objective."""

    def _worker(self):
        from aeon.core.worker import Worker
        w = Worker.__new__(Worker)
        w.memories = {}
        w.action_log = []
        w.action_log_summary = ""
        w._summarized_upto = 0
        w.current_plan = "none"
        w.active_skill = None
        w.expanded_categories = set()
        w.open_files = {}
        w.open_files_access_order = []
        w._resume_objective = None
        return w

    def test_tool_is_discoverable_and_top_level(self):
        from aeon.tools.loader import load_tools_from_directory
        from aeon.tools.categories import TOP_LEVEL_TOOLS
        import types
        names = {t.name for t in load_tools_from_directory(
            'aeon.tools', dependencies={'worker': types.SimpleNamespace()})}
        self.assertIn('resume_previous_session', names)
        self.assertIn('resume_previous_session', TOP_LEVEL_TOOLS)

    def _write_dump(self, td):
        import json as _json
        from pathlib import Path
        dump = Path(td) / "aeon_output" / "interrupted_session.json"
        dump.parent.mkdir(parents=True)
        dump.write_text(_json.dumps({
            "objective": "Build the parser",
            "current_plan": "Focus: finish the tokenizer.",
            "action_log": ["[Iter 1]\n- Intent: start\n- Actions: x\n- Result: ok"],
            "action_log_summary": "",
            "memories": {"path": {"value": "/x", "category": "general"}},
            "open_files_list": [],
            "open_files_access_order": [],
            "summarized_upto": 0,
            "stopped_at": "2026-07-12 10:00:00",
            "stop_reason": "ctrl-c",
        }))

    def test_resume_restores_state_and_objective(self):
        # No new-session instruction / no llm_client -> falls back to the restored
        # objective verbatim (the integration path is exercised separately below).
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            old = os.getcwd()
            os.chdir(td)
            try:
                self._write_dump(td)
                w = self._worker()
                out = w.resume_from_dump()
                self.assertIn("Build the parser", out)
                self.assertEqual(w._resume_objective, "Build the parser")
                self.assertEqual(w.current_plan, "Focus: finish the tokenizer.")
                self.assertEqual(len(w.action_log), 1)
                self.assertIn("path", w.memories)
            finally:
                os.chdir(old)

    def test_resume_integrates_new_instruction(self):
        # The new-session prompt ("continue but also do X") is merged with the
        # restored objective via an llm call, and the merged objective is adopted.
        import os
        import tempfile

        class _StubLLM:
            def __init__(self, result):
                self.result = result
                self.calls = []

            def integrate_resume(self, prev_objective, prev_plan, progress, new_instruction):
                self.calls.append((prev_objective, prev_plan, progress, new_instruction))
                return self.result

        with tempfile.TemporaryDirectory() as td:
            old = os.getcwd()
            os.chdir(td)
            try:
                self._write_dump(td)
                w = self._worker()
                w.current_objective = "continue but now also add CSV export"
                w.llm_client = _StubLLM({
                    "objective": "Build the parser AND add CSV export",
                    "directive": "Keep the parser work; additionally add CSV export.",
                })
                out = w.resume_from_dump()
                self.assertEqual(w._resume_objective, "Build the parser AND add CSV export")
                self.assertEqual(len(w.llm_client.calls), 1)
                prev_objective, prev_plan, progress, new_instruction = w.llm_client.calls[0]
                self.assertEqual(prev_objective, "Build the parser")
                self.assertEqual(new_instruction, "continue but now also add CSV export")
                self.assertIn("CSV export", out)              # merged objective surfaced
                self.assertIn("additionally add CSV export", out)  # directive surfaced
            finally:
                os.chdir(old)

    def test_resume_with_no_dump_is_graceful(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            old = os.getcwd()
            os.chdir(td)
            try:
                w = self._worker()
                out = w.resume_from_dump()
                self.assertIn("no previous session", out.lower())
                self.assertIsNone(w._resume_objective)
            finally:
                os.chdir(old)


class TestBootguardMarkerStability(unittest.TestCase):
    """The boot-pending marker must live at a cwd-INDEPENDENT path: a crashed
    relaunch in workspace A has to be recoverable by a fresh start in workspace B
    (the marker itself names aeon_code_dir)."""

    def test_marker_roundtrip_under_aeon_home(self):
        import json as _json
        import os
        import tempfile
        from aeon.core import bootguard

        old_home = os.environ.get("AEON_HOME")
        with tempfile.TemporaryDirectory() as td:
            os.environ["AEON_HOME"] = td
            try:
                p = bootguard._marker_path()
                self.assertTrue(str(p).startswith(td), "marker must live under AEON_HOME, not cwd")
                bootguard.mark_pending("/some/code/dir", "aeon-ckpt/x", reason="test")
                self.assertTrue(p.exists())
                data = _json.loads(p.read_text())
                self.assertEqual(data["aeon_code_dir"], "/some/code/dir")
                self.assertEqual(data["checkpoint"], "aeon-ckpt/x")
                bootguard.mark_boot_ok()
                self.assertFalse(p.exists())
            finally:
                if old_home is None:
                    os.environ.pop("AEON_HOME", None)
                else:
                    os.environ["AEON_HOME"] = old_home


class TestLocalProviderEndpoint(unittest.TestCase):
    """Provider 'local' (Ollama) must talk to the brain's port 8000 (mapped from
    11434 in start_brain.sh), NOT 8013 — that's the llama.cpp/vLLM load balancer,
    which would silently route Ollama chats to a different model."""

    def test_local_provider_uses_ollama_port(self):
        from aeon.core.llm import LLMClient
        c = LLMClient.__new__(LLMClient)
        client = c._create_client({'provider': 'local'})
        self.assertIn(":8000", str(client.base_url))


def load_tests(loader, standard_tests, pattern):
    return standard_tests


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
