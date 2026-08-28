from __future__ import annotations

import copy
import json
import unittest

from aeon.core.context_projection import (
    project_action_log,
    project_history,
    project_open_files,
)


class HistoryProjectionTests(unittest.TestCase):
    def test_reasoning_is_removed_without_mutating_source(self):
        source = [
            {"role": "user", "content": "request"},
            {
                "role": "assistant",
                "content": "decision",
                "reasoning": "private",
                "reasoning_content": "private",
            },
        ]
        before = copy.deepcopy(source)
        result = project_history(source)
        self.assertEqual(source, before)
        rendered = json.dumps(result.messages)
        self.assertNotIn("private", rendered)
        self.assertEqual(result.stripped_reasoning_fields, 2)
        self.assertIn("AEON_CONTEXT_CHECKPOINT", rendered)

    def test_tool_call_and_all_receipts_are_kept_or_dropped_atomically(self):
        source = [{"role": "user", "content": "old" * 500}]
        source.extend(
            [
                {
                    "role": "assistant",
                    "content": "call",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "x", "arguments": "{}"}},
                        {"id": "b", "function": {"name": "y", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "one"},
                {"role": "tool", "tool_call_id": "b", "content": "two"},
            ]
        )
        result = project_history(source, max_chars=1200, max_tokens=300)
        calls = [
            message for message in result.messages if message.get("tool_calls")
        ]
        self.assertEqual(len(calls), 1)
        receipts = [
            message for message in result.messages if message.get("role") == "tool"
        ]
        self.assertEqual({item["tool_call_id"] for item in receipts}, {"a", "b"})
        self.assertLessEqual(result.char_cost, 1200)
        self.assertLessEqual(result.token_cost, 300)

    def test_trim_checkpoint_is_deterministic_and_drops_orphans(self):
        source = [
            {"role": "tool", "tool_call_id": "orphan", "content": "unsafe"},
            *[
                {"role": "user", "content": f"turn-{index}-" + "x" * 400}
                for index in range(8)
            ],
        ]
        first = project_history(source, max_chars=1200, max_tokens=300)
        second = project_history(source, max_chars=1200, max_tokens=300)
        self.assertEqual(first.messages, second.messages)
        self.assertEqual(first.omitted_sha256, second.omitted_sha256)
        self.assertEqual(first.orphan_receipts, 1)
        self.assertNotIn("unsafe", json.dumps(first.messages))
        self.assertIn("AEON_CONTEXT_CHECKPOINT", first.messages[0]["content"])

        reprojected = project_history(first.messages, max_chars=1200, max_tokens=300)
        self.assertIn(
            "AEON_CONTEXT_CHECKPOINT",
            json.dumps(reprojected.messages),
        )


class ActionLogProjectionTests(unittest.TestCase):
    def test_collapses_repeats_and_never_mutates_raw_log(self):
        source = [
            "[Iter 1]\n- Intent: inspect\n- Actions: status\n- Receipts: blocked"
        ] * 12
        before = list(source)
        result = project_action_log(source, max_chars=900, max_tokens=225)
        self.assertEqual(source, before)
        self.assertIn("repeated 12 times", result.text)
        self.assertEqual(result.collapsed_repeats, 11)
        self.assertLessEqual(result.char_cost, 900)
        self.assertLessEqual(result.token_cost, 225)

    def test_large_log_gets_digest_bound_checkpoint_and_recent_suffix(self):
        source = [
            f"[Iter {index}]\n- Intent: step {index}\n- Actions: tool({index})\n"
            f"- Receipts: {'x' * 300}"
            for index in range(20)
        ]
        result = project_action_log(
            source, max_chars=1800, max_tokens=450, recent_entries=4
        )
        self.assertIn("AEON_ACTION_CHECKPOINT", result.text)
        self.assertIn("Iter 19", result.text)
        self.assertGreater(result.omitted_entries, 0)
        self.assertEqual(len(result.omitted_sha256), 64)
        self.assertLessEqual(result.char_cost, 1800)
        self.assertLessEqual(result.token_cost, 450)


class OpenFilesProjectionTests(unittest.TestCase):
    def test_selects_newest_complete_files_and_discloses_omissions(self):
        files = {
            "/workspace/old.py": "old\n" * 150,
            "/workspace/new.py": "new\n" * 80,
            "/workspace/newest.py": "newest\n" * 30,
        }
        before = dict(files)
        result = project_open_files(
            files,
            ["/workspace/old.py", "/workspace/new.py", "/workspace/newest.py"],
            max_chars=900,
            max_tokens=225,
        )
        self.assertEqual(files, before)
        self.assertTrue(result.selected_paths)
        self.assertEqual(result.selected_paths[0], "/workspace/newest.py")
        self.assertIn("--- END FILE:", result.text)
        self.assertTrue(result.omitted)
        self.assertTrue(all(len(item.sha256) == 64 for item in result.omitted))
        self.assertIn("OMITTED OPEN FILES", result.text)
        self.assertLessEqual(result.char_cost, 900)
        self.assertLessEqual(result.token_cost, 225)

    def test_max_files_reason_is_explicit(self):
        result = project_open_files(
            {"a": "A", "b": "B", "c": "C"},
            ["a", "b", "c"],
            max_files=1,
        )
        self.assertEqual(result.selected_paths, ("c",))
        self.assertEqual({item.reason for item in result.omitted}, {"max_files"})


if __name__ == "__main__":
    unittest.main()
