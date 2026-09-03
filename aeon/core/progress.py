"""Harness-owned progress, recovery, and loop control for agent turns.

The model may suggest that an attempt was useful, but only typed receipts and
task obligations establish progress. This controller separates three states
that used to be conflated: an attempt failed, the strategy must change, and the
request is genuinely exhausted. It has no tool, provider, memory, or Fleet
dependencies, which keeps recovery deterministic and hermetic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class NoProgressSample:
    """One executed action set whose receipts did not prove task progress."""

    action: str
    structure: str
    outcome: str
    blocked: bool = False
    retryable: bool = False
    # Authorization/category refusals may become valid after a user response or
    # capability expansion. They require recovery but not a request-long bar.
    bar_exact: bool = True


@dataclass(frozen=True)
class ProgressDecision:
    """Harness action after observing one turn."""

    hard_stop: bool = False
    block_exact_action: bool = False
    recovery_required: bool = False
    recovery_level: int = 0
    reason: str = ""
    streak: int = 0
    barred_actions: tuple[str, ...] = ()


class ProgressController:
    """Force bounded strategy changes before declaring a task exhausted.

    A first failure starts evidence-grounded recovery. Repeated exact or
    structurally equivalent attempts are barred, and A/B oscillation triggers a
    parent-goal reframe. The controller hard-stops only after the model ignores
    an existing bar twice or after a broad, bounded strategy plateau. Verified
    obligation progress resets the recovery epoch; arbitrary successful reads do
    not erase it.
    """

    def __init__(
        self,
        *,
        exact_failure_limit: int = 3,
        equivalent_outcome_limit: int = 4,
        exhaustion_limit: int = 12,
        history_limit: int = 16,
    ) -> None:
        self.exact_failure_limit = max(2, int(exact_failure_limit))
        self.equivalent_outcome_limit = max(3, int(equivalent_outcome_limit))
        self.exhaustion_limit = max(
            self.equivalent_outcome_limit + 2, int(exhaustion_limit)
        )
        self._history: deque[NoProgressSample] = deque(
            maxlen=max(self.exhaustion_limit, int(history_limit), 4)
        )
        self._barred_actions: set[str] = set()
        self._barred_reproposals: dict[str, int] = {}
        self._recovery_level = 0
        self._recovery_reason = ""
        self._recovery_origin_actions: set[str] = set()
        self._recovery_attempts: set[str] = set()

    @property
    def history(self) -> tuple[NoProgressSample, ...]:
        return tuple(self._history)

    @property
    def barred_actions(self) -> frozenset[str]:
        return frozenset(self._barred_actions)

    @property
    def recovery_required(self) -> bool:
        return self._recovery_level > 0

    @property
    def recovery_level(self) -> int:
        return self._recovery_level

    @property
    def recovery_reason(self) -> str:
        return self._recovery_reason

    @property
    def recovery_attempt_count(self) -> int:
        return len(self._recovery_attempts)

    def reset(self) -> None:
        self._history.clear()
        self._barred_actions.clear()
        self._barred_reproposals.clear()
        self._recovery_level = 0
        self._recovery_reason = ""
        self._recovery_origin_actions.clear()
        self._recovery_attempts.clear()

    def _activate_recovery(
        self,
        reason: str,
        *,
        level: int,
        origin_actions: Iterable[str] = (),
        bar_actions: Iterable[str] = (),
    ) -> None:
        clean_origins = {str(item)[:4096] for item in origin_actions if str(item)}
        clean_bars = {str(item)[:4096] for item in bar_actions if str(item)}
        if not self.recovery_required:
            self._recovery_origin_actions = set(clean_origins)
            self._recovery_attempts.clear()
        else:
            self._recovery_origin_actions.update(clean_origins)
        self._recovery_level = max(self._recovery_level, min(3, max(1, int(level))))
        self._recovery_reason = str(reason or self._recovery_reason)[:2000]
        for action in clean_bars:
            if len(self._barred_actions) < 64 or action in self._barred_actions:
                self._barred_actions.add(action)

    def force_recovery(
        self,
        reason: str,
        *,
        level: int = 1,
        origin_actions: Iterable[str] = (),
        bar_actions: Iterable[str] = (),
    ) -> ProgressDecision:
        """Enter or escalate recovery for a harness signal outside a failed call."""

        self._activate_recovery(
            reason,
            level=level,
            origin_actions=origin_actions,
            bar_actions=bar_actions,
        )
        return self._current_decision(reason=reason)

    def note_proposed_actions(self, actions: Iterable[str]) -> None:
        """Count genuinely different post-failure strategies, including reads.

        Rewording or retrying an originating/barred exact call does not count as
        strategic diversity.
        """

        if not self.recovery_required:
            return
        for raw in actions:
            action = str(raw or "")[:4096]
            if (
                not action
                or action in self._recovery_origin_actions
                or action in self._barred_actions
            ):
                continue
            if len(self._recovery_attempts) < 64:
                self._recovery_attempts.add(action)

    def recovery_directive(self) -> str:
        """Return a compact, action-oriented recovery constraint."""

        if not self.recovery_required:
            return ""
        level = self.recovery_level
        if level == 1:
            level_action = "Check one missing precondition or use a different action."
        elif level == 2:
            level_action = "Use a different tool or mechanism from the failed family."
        else:
            level_action = "Choose a different route to an unmet owner outcome."
        barred = (
            f" {len(self._barred_actions)} exact action(s) are harness-barred."
            if self._barred_actions
            else ""
        )
        return (
            f"RECOVERY REQUIRED (level {level}): {self._recovery_reason}.{barred}\n"
            f"{level_action} Do not narrate recovery, restate the goal, or update the "
            "plan unless it changed. Take one evidence-producing action."
        )

    def to_state_dict(self) -> dict[str, object]:
        """Return bounded run-local state needed to survive a process loss."""

        return {
            "version": 2,
            "samples": [
                {
                    "action": item.action,
                    "structure": item.structure,
                    "outcome": item.outcome,
                    "blocked": item.blocked,
                    "retryable": item.retryable,
                    "bar_exact": item.bar_exact,
                }
                for item in self._history
            ],
            "barred_actions": sorted(self._barred_actions),
            "barred_reproposals": dict(self._barred_reproposals),
            "recovery_level": self._recovery_level,
            "recovery_reason": self._recovery_reason,
            "recovery_origin_actions": sorted(self._recovery_origin_actions),
            "recovery_attempts": sorted(self._recovery_attempts),
        }

    def restore_state_dict(self, state: object) -> None:
        """Restore v2 state; conservatively recover bounded v1 history."""

        self.reset()
        if not isinstance(state, Mapping) or state.get("version") not in {1, 2}:
            return
        samples = state.get("samples")
        if not isinstance(samples, list):
            return
        restored: list[NoProgressSample] = []
        for raw in samples[-self._history.maxlen :]:
            if not isinstance(raw, Mapping):
                self.reset()
                return
            action = str(raw.get("action") or "")[:4096]
            structure = str(raw.get("structure") or "")[:4096]
            outcome = str(raw.get("outcome") or "")[:4096]
            if not action or not structure or not outcome:
                self.reset()
                return
            restored.append(
                NoProgressSample(
                    action=action,
                    structure=structure,
                    outcome=outcome,
                    blocked=raw.get("blocked") is True,
                    retryable=raw.get("retryable") is True,
                    bar_exact=raw.get("bar_exact") is not False,
                )
            )
        self._history.extend(restored)
        if state.get("version") == 1:
            if restored:
                last = restored[-1]
                self._activate_recovery(
                    "restored no-progress evidence requires a different strategy",
                    level=1,
                    origin_actions=(last.action, last.structure),
                    bar_actions=(last.action,) if last.blocked and last.bar_exact else (),
                )
            return

        def bounded_strings(key: str, limit: int = 64) -> set[str]:
            raw_values = state.get(key)
            if not isinstance(raw_values, list):
                return set()
            return {
                str(item)[:4096]
                for item in raw_values[-limit:]
                if isinstance(item, str) and item
            }

        self._barred_actions = bounded_strings("barred_actions")
        self._recovery_origin_actions = bounded_strings("recovery_origin_actions")
        self._recovery_attempts = bounded_strings("recovery_attempts")
        try:
            self._recovery_level = min(
                3, max(0, int(state.get("recovery_level") or 0))
            )
        except (TypeError, ValueError):
            self._recovery_level = 0
        self._recovery_reason = str(state.get("recovery_reason") or "")[:2000]
        raw_reproposals = state.get("barred_reproposals")
        if isinstance(raw_reproposals, Mapping):
            for action, count in list(raw_reproposals.items())[-64:]:
                if not isinstance(action, str) or action not in self._barred_actions:
                    continue
                try:
                    normalized = min(2, max(0, int(count)))
                except (TypeError, ValueError):
                    continue
                if normalized:
                    self._barred_reproposals[action[:4096]] = normalized
        if self._recovery_level and not self._recovery_reason:
            self._recovery_reason = "restored no-progress recovery state"

    @staticmethod
    def _trailing_count(values: list[str], expected: str) -> int:
        count = 0
        for value in reversed(values):
            if value != expected:
                break
            count += 1
        return count

    def _current_decision(
        self,
        *,
        reason: str = "",
        streak: int = 0,
        hard_stop: bool = False,
        block_exact_action: bool = False,
    ) -> ProgressDecision:
        return ProgressDecision(
            hard_stop=hard_stop,
            block_exact_action=block_exact_action,
            recovery_required=self.recovery_required,
            recovery_level=self.recovery_level,
            reason=str(reason or self.recovery_reason),
            streak=max(0, int(streak)),
            barred_actions=tuple(sorted(self._barred_actions)),
        )

    def observe(
        self,
        sample: NoProgressSample | None,
        *,
        made_progress: bool = False,
    ) -> ProgressDecision:
        """Record a no-progress sample or reset only after typed task progress."""

        if made_progress:
            self.reset()
            return ProgressDecision()
        if sample is None or not sample.action or not sample.outcome:
            return self._current_decision()

        was_barred = sample.action in self._barred_actions
        if was_barred:
            count = min(2, self._barred_reproposals.get(sample.action, 0) + 1)
            self._barred_reproposals[sample.action] = count
            self._activate_recovery(
                "the model reproposed an action already barred by typed failure evidence",
                level=3,
                origin_actions=(sample.action,),
            )
            if count >= 2:
                return self._current_decision(
                    hard_stop=True,
                    block_exact_action=True,
                    reason="the model ignored the same harness action bar twice",
                    streak=count,
                )

        self._history.append(sample)
        history = list(self._history)
        actions = [item.action for item in history]
        exact_streak = self._trailing_count(actions, sample.action)
        same_outcome = self._trailing_count(
            [item.outcome for item in history], sample.outcome
        )
        same_structure = self._trailing_count(
            [f"{item.structure}\n{item.outcome}" for item in history],
            f"{sample.structure}\n{sample.outcome}",
        )

        soft_block = bool(
            sample.blocked and not sample.retryable and sample.bar_exact
        )
        if soft_block:
            self._activate_recovery(
                "a non-retryable exact call was refused",
                level=2 if was_barred else 1,
                origin_actions=(sample.action, sample.structure),
                bar_actions=(sample.action,),
            )

        if exact_streak >= self.exact_failure_limit:
            self._activate_recovery(
                f"the same exact action made no progress {exact_streak} times",
                level=2,
                origin_actions=(sample.action, sample.structure),
                bar_actions=(sample.action,),
            )

        # Detect A/B/A/B only when corresponding calls have the same outcomes. A
        # changing result is evidence of movement, not oscillation.
        if len(history) >= 4:
            a1, b1, a2, b2 = history[-4:]
            if (
                a1.action == a2.action
                and b1.action == b2.action
                and a1.action != b1.action
                and a1.outcome == a2.outcome
                and b1.outcome == b2.outcome
            ):
                self._activate_recovery(
                    "two no-progress approaches are oscillating A/B/A/B",
                    level=3,
                    origin_actions=(
                        a1.action, a1.structure, b1.action, b1.structure
                    ),
                    bar_actions=(a1.action, b1.action),
                )

        if same_structure >= self.exact_failure_limit:
            equivalent_actions = {
                item.action
                for item in history[-same_structure:]
                if item.structure == sample.structure
            }
            self._activate_recovery(
                f"equivalent actions produced the same outcome {same_structure} times",
                level=2,
                origin_actions=(*equivalent_actions, sample.structure),
                bar_actions=equivalent_actions,
            )
        if same_outcome >= self.equivalent_outcome_limit:
            plateau_actions = {item.action for item in history[-same_outcome:]}
            self._activate_recovery(
                f"different calls produced the same outcome {same_outcome} times",
                level=3,
                origin_actions=(
                    *plateau_actions,
                    *(item.structure for item in history[-same_outcome:]),
                ),
                bar_actions=plateau_actions,
            )

        if not self.recovery_required:
            self._activate_recovery(
                "the latest consequential action produced no typed task progress",
                level=1,
                origin_actions=(sample.action, sample.structure),
            )

        # A wide plateau is finite, but unlike the old controller it affords a
        # genuine parent-goal reframe and multiple strategy families first.
        recent = history[-self.exhaustion_limit :]
        distinct_structures = {item.structure for item in recent}
        if (
            len(recent) >= self.exhaustion_limit
            and len(distinct_structures) >= 3
            and self.recovery_attempt_count >= 2
        ):
            return self._current_decision(
                hard_stop=True,
                reason=(
                    "typed task state remained unchanged across the bounded "
                    "multi-strategy recovery budget"
                ),
                streak=len(recent),
            )

        return self._current_decision(
            reason=self.recovery_reason,
            streak=max(exact_streak, same_structure, same_outcome),
            block_exact_action=sample.action in self._barred_actions,
        )
