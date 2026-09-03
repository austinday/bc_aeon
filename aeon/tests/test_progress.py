"""Hermetic tests for the harness-owned progress and recovery controller."""

from __future__ import annotations

import unittest

from aeon.core.progress import NoProgressSample, ProgressController


def sample(
    action: str,
    outcome: str = "failed",
    *,
    structure: str | None = None,
    blocked: bool = False,
    retryable: bool = False,
    bar_exact: bool = True,
) -> NoProgressSample:
    return NoProgressSample(
        action=action,
        structure=structure or action.split("(", 1)[0],
        outcome=outcome,
        blocked=blocked,
        retryable=retryable,
        bar_exact=bar_exact,
    )


class ProgressControllerScenarios(unittest.TestCase):
    def test_first_failure_starts_level_one_recovery_without_stopping(self):
        controller = ProgressController()

        decision = controller.observe(sample("run(a)"))

        self.assertFalse(decision.hard_stop)
        self.assertFalse(decision.block_exact_action)
        self.assertTrue(decision.recovery_required)
        self.assertEqual(decision.recovery_level, 1)
        self.assertIn("RECOVERY REQUIRED (level 1)", controller.recovery_directive())
        self.assertIn("Do not narrate recovery", controller.recovery_directive())

    def test_equivalent_failures_escalate_to_level_two_and_bar_family(self):
        controller = ProgressController()

        decisions = [
            controller.observe(
                sample(action, structure="run:pytest", outcome="same assertion")
            )
            for action in ("run(a)", "run(b)", "run(c)")
        ]

        self.assertFalse(any(item.hard_stop for item in decisions))
        self.assertEqual(decisions[-1].recovery_level, 2)
        self.assertEqual(
            set(decisions[-1].barred_actions),
            {"run(a)", "run(b)", "run(c)"},
        )
        self.assertIn("equivalent actions", decisions[-1].reason)

    def test_ab_oscillation_requires_level_three_reframe_before_stopping(self):
        controller = ProgressController()

        decisions = [
            controller.observe(sample(action, outcome=outcome))
            for action, outcome in (
                ("run(a)", "a failed"),
                ("read(b)", "b failed"),
                ("run(a)", "a failed"),
                ("read(b)", "b failed"),
            )
        ]

        self.assertFalse(any(item.hard_stop for item in decisions))
        self.assertEqual(decisions[-1].recovery_level, 3)
        self.assertEqual(
            set(decisions[-1].barred_actions), {"run(a)", "read(b)"}
        )
        self.assertIn("A/B/A/B", decisions[-1].reason)
        self.assertIn("different route", controller.recovery_directive())

    def test_exact_block_hard_stops_only_after_two_ignored_bar_reproposals(self):
        controller = ProgressController()
        refused = sample(
            "run(a)",
            outcome="policy refused exact call",
            blocked=True,
            retryable=False,
        )

        first = controller.observe(refused)
        first_ignored_bar = controller.observe(refused)
        second_ignored_bar = controller.observe(refused)

        self.assertFalse(first.hard_stop)
        self.assertTrue(first.block_exact_action)
        self.assertEqual(first.recovery_level, 1)
        self.assertFalse(first_ignored_bar.hard_stop)
        self.assertEqual(first_ignored_bar.recovery_level, 3)
        self.assertTrue(second_ignored_bar.hard_stop)
        self.assertTrue(second_ignored_bar.block_exact_action)
        self.assertEqual(second_ignored_bar.streak, 2)
        self.assertIn(
            "ignored the same harness action bar twice",
            second_ignored_bar.reason,
        )

    def test_only_explicit_verified_progress_resets_recovery_epoch(self):
        controller = ProgressController()
        controller.observe(sample("run(a)"))

        unchanged = controller.observe(None)

        self.assertTrue(unchanged.recovery_required)
        self.assertEqual(len(controller.history), 1)

        reset = controller.observe(None, made_progress=True)

        self.assertFalse(reset.recovery_required)
        self.assertEqual(controller.history, ())
        self.assertEqual(controller.barred_actions, frozenset())
        self.assertEqual(controller.recovery_directive(), "")

    def test_bar_reproposal_budget_survives_state_round_trip(self):
        refused = sample(
            "run(a)",
            outcome="nonretryable refusal",
            blocked=True,
            retryable=False,
        )
        original = ProgressController()
        original.observe(refused)
        first_ignored_bar = original.observe(refused)
        self.assertFalse(first_ignored_bar.hard_stop)

        state = original.to_state_dict()
        self.assertEqual(state["version"], 2)

        restored = ProgressController()
        restored.restore_state_dict(state)
        second_ignored_bar = restored.observe(refused)

        self.assertTrue(second_ignored_bar.hard_stop)
        self.assertTrue(second_ignored_bar.block_exact_action)
        self.assertEqual(second_ignored_bar.streak, 2)
        self.assertIn("run(a)", restored.barred_actions)
        self.assertEqual(restored.recovery_level, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
