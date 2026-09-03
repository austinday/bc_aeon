"""Deterministic Project Manager guard for Nexus-managed agent creation.

The model may remember files called "agents" or decide that a local LLM script is
equivalent to a Nexus session.  Neither is evidence that a durable instance was
created.  This module keeps that distinction outside the model: only the typed
receipt returned by the authenticated ``start_agent_instance`` bridge can unlock a
creation success report in the current user turn.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


INTENT_NONE = "none"
INTENT_CAPABILITY = "capability"
INTENT_CREATE = "create"

_CREATION_VERB = r"(?:create|spawn|provision|set[ -]?up|add)"
_START_VERB = r"(?:start|launch)"
_AGENT_NOUN = (
    r"(?:aeon\s+instance|agent(?:\s+(?:instance|session|tab))?|"
    r"standalone\s+session|new\s+(?:agent\s+)?tab)"
)

_GENERIC_AGENT_ARTIFACT_RE = re.compile(
    r"\b(?:ai\s+|llm\s+|software\s+)?agent(?:ic)?\s+"
    r"(?:app(?:lication)?|script|program|service|framework|library|code|prototype|bot)\b",
    re.IGNORECASE,
)
_GENERIC_AGENT_ARTIFACT_REVERSE = re.compile(
    r"\b(?:app(?:lication)?|script|program|service|framework|library|code|prototype|bot)\s+"
    r"(?:for|implementing)\s+(?:an?\s+)?(?:ai\s+|llm\s+)?agent\b",
    re.IGNORECASE,
)
_AGENT_MODIFICATION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+)?make\s+"
    r"(?:(?:the|my|this)\s+agent\b|an?\s+agent\s+"
    r"(?:page|ui|interface|dashboard|behavior|api|backend|tool|prompt|harness|"
    r"model|response|workflow|better|faster|responsive|work|use)\b)",
    re.IGNORECASE,
)
_MAKE_AGENT_RE = re.compile(
    rf"\bmake\s+(?:me\s+)?(?:a|an|new|another)\b.{{0,100}}\b{_AGENT_NOUN}\b",
    re.IGNORECASE,
)
_CAPABILITY_PREFIX_RE = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"(?:can|could|would|will)\s+you|"
    r"are\s+you\s+able\s+to|do\s+you\s+know\s+how\s+to|"
    r"how\s+would\s+you|what\s+would\s+it\s+take\s+to|"
    r"is\s+it\s+possible(?:\s+for\s+you)?\s+to|"
    r"plan(?:\s+out)?\s+how\s+to"
    r")\b",
    re.IGNORECASE,
)
_EXPLICIT_CREATE_PREFIX_RE = re.compile(
    rf"^\s*(?:please\s+)?(?:now\s+|go\s+ahead\s+and\s+)?{_CREATION_VERB}\b",
    re.IGNORECASE,
)
_EXPLICIT_WANT_PREFIX_RE = re.compile(
    rf"^\s*i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+{_CREATION_VERB}\b",
    re.IGNORECASE,
)
_EXPLICIT_MAKE_PREFIX_RE = re.compile(
    rf"^\s*(?:please\s+)?(?:now\s+|go\s+ahead\s+and\s+)?"
    rf"make\s+(?:me\s+)?(?:a|an|new|another)\b.{{0,100}}\b{_AGENT_NOUN}\b",
    re.IGNORECASE,
)
_EXPLICIT_WANT_MAKE_PREFIX_RE = re.compile(
    rf"^\s*i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+"
    rf"make\s+(?:me\s+)?(?:a|an|new|another)\b.{{0,100}}\b{_AGENT_NOUN}\b",
    re.IGNORECASE,
)
_EXPLICIT_START_PREFIX_RE = re.compile(
    rf"^\s*(?:please\s+)?(?:now\s+|go\s+ahead\s+and\s+)?{_START_VERB}\s+"
    rf"(?:me\s+)?(?:a|an|new|another|standalone|aeon|nexus-managed)\b",
    re.IGNORECASE,
)
_EXPLICIT_WANT_START_PREFIX_RE = re.compile(
    rf"^\s*i\s+(?:want|need|would\s+like)\s+(?:you\s+)?to\s+{_START_VERB}\s+"
    rf"(?:me\s+)?(?:a|an|new|another|standalone|aeon|nexus-managed)\b",
    re.IGNORECASE,
)
_CONFIRMATION_RE = re.compile(
    r"^\s*(?:yes\s*[,;:-]?\s*)?(?:go\s+ahead(?:\s+and\s+(?:create|start|make)\s+it)?|"
    r"do\s+it|proceed|"
    r"create\s+it|make\s+it|start\s+it)(?:\s+now)?[.!\s]*$",
    re.IGNORECASE,
)
_NEGATED_AUTHORIZATION_RE = re.compile(
    r"\b(?:do\s+not|don['’]?t|not\s+yet|without\s+(?:creating|starting|making))\b",
    re.IGNORECASE,
)

_SUCCESS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"\b(?:i|we)\s+(?:(?:have|'ve)\s+)?(?:successfully\s+)?"
        rf"(?:created|started|launched|spawned|provisioned|registered|set[ -]?up|made|added|built)\b"
        rf".{{0,180}}\b{_AGENT_NOUN}\b",
        rf"\b{_AGENT_NOUN}\b.{{0,180}}\b(?:was|has\s+been|is\s+now|is)\s+"
        r"(?:successfully\s+)?(?:created|started|launched|spawned|provisioned|registered|"
        r"set[ -]?up|built|(?:verified\s+)?working|ready|running|live|available)\b",
        rf"\b(?:created|started|launched|spawned|provisioned|registered)\b.{{0,140}}"
        rf"\b(?:new\s+)?{_AGENT_NOUN}\b",
        r"\bnexus\s+instance(?:\s+id)?\s*[:#]\s*[a-z0-9_-]+\b",
        r"\b(?:new\s+)?agent\s+tab\s+(?:is|should\s+be)\s+(?:now\s+)?(?:ready|visible|open)\b",
        rf"\b{_AGENT_NOUN}\b.{{0,80}}\bexists\b",
    )
)
_NEGATED_SUCCESS_RE = re.compile(
    r"\b(?:not|never|no)\b(?:\W+\w+){0,4}\W+"
    r"(?:created|started|launched|spawned|provisioned|registered|built|working|ready|running|live|available|exists)\b|"
    r"\b(?:didn['’]?t|couldn['’]?t|can['’]?t|hasn['’]?t|haven['’]?t|"
    r"isn['’]?t|wasn['’]?t|unable\s+to|failed\s+to)\b(?:\W+\w+){0,5}\W+"
    r"(?:create|start|launch|spawn|provision|register|set[ -]?up|make|build|work|ready|exist)\b|"
    rf"\bno\s+(?:(?:new|aeon|nexus(?:-managed)?)\s+)?{_AGENT_NOUN}\b",
    re.IGNORECASE,
)
_TRUTHFUL_NONCOMPLETION_RE = re.compile(
    r"\b(?:not\s+created|was\s+not\s+created|did\s+not\s+create|could\s+not\s+create|"
    r"couldn['’]?t\s+create|unable\s+to\s+create|failed|blocked|refused|unavailable|"
    r"awaiting|waiting\s+for|needs?\s+(?:user\s+)?(?:input|clarification|confirmation)|"
    r"asked\s+(?:the\s+)?user|requires?\s+(?:input|clarification|confirmation)|"
    r"delivered\s+(?:the\s+)?plan|answered\s+(?:the\s+)?capability\s+question)\b",
    re.IGNORECASE,
)
_DEFERRED_ACTIVE_CLAIM_RE = re.compile(
    r"\b(?:started|launched|running|working|analy[sz]ing|executing|processing)\b",
    re.IGNORECASE,
)
_DEFERRED_ACTIVE_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|isn['’]?t|wasn['’]?t|hasn['’]?t|haven['’]?t)\b"
    r"(?:\W+\w+){0,6}\W+"
    r"(?:started|launched|running|working|analy[sz]ing|executing|processing)\b",
    re.IGNORECASE,
)
_CLARIFICATION_RE = re.compile(
    r"\?|\b(?:need|which|what|where|please\s+provide|clarif|confirm|choose|specify|"
    r"path|directory|workspace|folder)\b",
    re.IGNORECASE,
)
_CANCEL_PENDING_RE = re.compile(
    r"^\s*(?:never\s*mind|cancel|stop|forget\s+it|do\s+not|don['’]?t)\b",
    re.IGNORECASE,
)

_ALLOWED_KINDS = frozenset({"aeon", "codex", "claude", "grok"})
_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_PROJECT_ID_RE = re.compile(r"^pr-[0-9a-f]{32}$")


def classify_project_manager_agent_intent(
    text: str, *, pending_confirmation: bool = False
) -> str:
    """Classify only requests for a *Nexus session*, not agent software work."""

    value = " ".join(str(text or "").strip().split())
    if not value:
        return INTENT_NONE
    if pending_confirmation and _CONFIRMATION_RE.fullmatch(value):
        return INTENT_CREATE
    if (
        _GENERIC_AGENT_ARTIFACT_RE.search(value)
        or _GENERIC_AGENT_ARTIFACT_REVERSE.search(value)
        or _AGENT_MODIFICATION_RE.search(value)
    ):
        return INTENT_NONE
    if not re.search(rf"\b{_AGENT_NOUN}\b", value, re.IGNORECASE):
        return INTENT_NONE

    mentions_creation = bool(
        re.search(rf"\b(?:{_CREATION_VERB}|{_START_VERB})\b", value, re.IGNORECASE)
        or _MAKE_AGENT_RE.search(value)
    )
    if mentions_creation and _CAPABILITY_PREFIX_RE.search(value):
        return INTENT_CAPABILITY
    if re.search(r"\b(?:do\s+not|don['’]?t|not\s+yet)\b", value, re.IGNORECASE):
        return INTENT_CAPABILITY if mentions_creation else INTENT_NONE
    if (
        _EXPLICIT_CREATE_PREFIX_RE.search(value)
        or _EXPLICIT_WANT_PREFIX_RE.search(value)
        or _EXPLICIT_MAKE_PREFIX_RE.search(value)
        or _EXPLICIT_WANT_MAKE_PREFIX_RE.search(value)
        or _EXPLICIT_START_PREFIX_RE.search(value)
        or _EXPLICIT_WANT_START_PREFIX_RE.search(value)
    ):
        return INTENT_CREATE
    return INTENT_NONE


def claims_agent_creation_success(text: str) -> bool:
    """Return whether visible prose asserts that a durable agent now exists."""

    value = str(text or "")
    # Evaluate clauses independently so "no agent was created; I created one now"
    # still catches the positive second clause.
    clauses = re.split(r"(?:[;\n]+|(?<=[.!?])\s+|\b(?:but|however)\b)", value)
    for clause in clauses:
        if not clause.strip():
            continue
        if _NEGATED_SUCCESS_RE.search(clause):
            continue
        if any(pattern.search(clause) for pattern in _SUCCESS_PATTERNS):
            return True
    return False


def _claims_deferred_agent_is_active(text: object) -> bool:
    """Reject present/past work claims for a register-only Aeon receipt."""

    clauses = re.split(r"(?:[;\n]+|(?<=[.!?])\s+|\b(?:but|however)\b)", str(text or ""))
    return any(
        _DEFERRED_ACTIVE_CLAIM_RE.search(clause)
        and not _DEFERRED_ACTIVE_NEGATION_RE.search(clause)
        for clause in clauses
        if clause.strip()
    )


@dataclass(frozen=True)
class VerifiedNexusAgentStart:
    """Typed evidence emitted only after validating the authenticated Nexus reply."""

    instance: Mapping[str, Any]
    message: str

    def __str__(self) -> str:
        return self.message


def verified_start_receipt(
    instance: object,
    *,
    expected_name: str,
    expected_workspace: str,
    expected_kind: str,
    expected_continuous: bool = False,
    expected_goal: str = "",
    expected_project_id: str | None = None,
) -> VerifiedNexusAgentStart:
    """Validate and freeze the durable record returned by Nexus."""

    if not isinstance(instance, Mapping):
        raise ValueError("Nexus returned a malformed agent record")
    instance_id = instance.get("id")
    name = instance.get("name")
    workspace = instance.get("workspace")
    kind = instance.get("kind")
    status = instance.get("status")
    project_id = instance.get("project_id")
    if not isinstance(instance_id, str) or not _INSTANCE_ID_RE.fullmatch(instance_id):
        raise ValueError("Nexus returned a malformed agent id")
    if not isinstance(name, str) or name.strip() != str(expected_name).strip():
        raise ValueError("Nexus returned a mismatched agent name")
    if not isinstance(kind, str) or kind.strip().lower() != str(expected_kind).strip().lower():
        raise ValueError("Nexus returned a mismatched agent kind")
    if kind.strip().lower() not in _ALLOWED_KINDS:
        raise ValueError("Nexus returned an unsupported agent kind")
    if not isinstance(workspace, str) or not os.path.isabs(workspace):
        raise ValueError("Nexus returned a malformed agent workspace")
    expected_path = Path(expected_workspace).resolve(strict=False)
    returned_path = Path(workspace).resolve(strict=False)
    if returned_path != expected_path:
        raise ValueError("Nexus returned a mismatched agent workspace")
    if not isinstance(status, str) or not status.strip() or len(status) > 64:
        raise ValueError("Nexus returned a malformed agent status")
    if project_id is not None and (
        not isinstance(project_id, str) or not _PROJECT_ID_RE.fullmatch(project_id)
    ):
        raise ValueError("Nexus returned a malformed project id")
    if expected_project_id is not None:
        if (
            not isinstance(expected_project_id, str)
            or not _PROJECT_ID_RE.fullmatch(expected_project_id)
        ):
            raise ValueError("Expected project id is invalid")
        if project_id != expected_project_id:
            raise ValueError("Nexus returned a mismatched project id")
    awaiting_objective = instance.get("awaiting_objective")
    if not isinstance(awaiting_objective, bool):
        raise ValueError("Nexus returned a malformed awaiting-objective state")
    continuous = instance.get("continuous_mode")
    continuous_enabled = bool(
        isinstance(continuous, Mapping) and continuous.get("enabled") is True
    )
    if not isinstance(expected_continuous, bool):
        raise ValueError("Expected continuous-mode state is invalid")
    if continuous_enabled != expected_continuous:
        raise ValueError("Nexus returned a mismatched continuous-mode state")
    if expected_continuous:
        if kind.strip().lower() != "aeon" or awaiting_objective:
            raise ValueError("Nexus returned an inconsistent continuous Aeon state")
        if not isinstance(continuous, Mapping) or continuous.get("goal") != str(expected_goal).strip():
            raise ValueError("Nexus returned a mismatched continuous goal")
    if kind.strip().lower() == "aeon" and not awaiting_objective and not continuous_enabled:
        raise ValueError("Nexus did not confirm the deferred Aeon state")
    if awaiting_objective and (
        kind.strip().lower() != "aeon" or status.strip().lower() != "idle"
    ):
        raise ValueError("Nexus returned an inconsistent deferred Aeon state")
    if not awaiting_objective and status.strip().lower() != "running":
        raise ValueError("Nexus did not confirm that the agent is running")

    record = {
        "id": instance_id,
        "name": name.strip(),
        "workspace": str(returned_path),
        "kind": kind.strip().lower(),
        "mode": instance.get("mode"),
        "status": status.strip(),
        "awaiting_objective": awaiting_objective,
        "project_id": project_id,
        "continuous_mode": {
            "enabled": continuous_enabled,
            "goal": str(continuous.get("goal") or "") if isinstance(continuous, Mapping) else "",
        },
    }
    if continuous_enabled:
        message = (
            f"Started standalone {record['kind']} agent '{record['name']}' in "
            f"{record['workspace']}. Nexus instance: {record['id']}; "
            f"state: {record['status']}, continuous mode enabled."
        )
    elif record["awaiting_objective"]:
        message = (
            f"Registered standalone {record['kind']} agent '{record['name']}' in "
            f"{record['workspace']}. Nexus instance: {record['id']}; "
            f"state: {record['status']}, awaiting the user's first message. "
            "No Aeon process or objective has started yet."
        )
    else:
        message = (
            f"Started standalone {record['kind']} agent '{record['name']}' in "
            f"{record['workspace']}. Nexus instance: {record['id']}; "
            f"state: {record['status']}."
        )
    return VerifiedNexusAgentStart(instance=record, message=message)


@dataclass
class DurableAgentTurnGuard:
    """State machine whose receipt scope is exactly one user turn/objective."""

    project_manager: bool
    pending_confirmation: bool = False
    intent: str = INTENT_NONE
    attempted: bool = False
    verified_instance: Mapping[str, Any] | None = None
    last_attempt_error: str = ""
    awaiting_clarification: bool = False

    def to_state_dict(self) -> dict[str, Any]:
        """Persist the exact in-flight creation boundary across process loss."""

        return {
            "intent": self.intent,
            "pending_confirmation": bool(self.pending_confirmation),
            "attempted": bool(self.attempted),
            "verified_instance": (
                dict(self.verified_instance)
                if isinstance(self.verified_instance, Mapping)
                else None
            ),
            "last_attempt_error": str(self.last_attempt_error or "")[:2000],
            "awaiting_clarification": bool(self.awaiting_clarification),
        }

    def restore_state_dict(self, state: object) -> None:
        """Restore only a structurally valid guard snapshot."""

        self.reset_conversation()
        if not self.project_manager or not isinstance(state, Mapping):
            return
        intent = state.get("intent")
        if intent not in {INTENT_NONE, INTENT_CAPABILITY, INTENT_CREATE}:
            return
        verified = state.get("verified_instance")
        if verified is not None:
            if not isinstance(verified, Mapping):
                return
            instance_id = verified.get("id")
            if not isinstance(instance_id, str) or not _INSTANCE_ID_RE.fullmatch(instance_id):
                return
            verified = dict(verified)
        self.intent = str(intent)
        self.pending_confirmation = bool(state.get("pending_confirmation"))
        self.attempted = bool(state.get("attempted"))
        self.verified_instance = verified
        self.last_attempt_error = str(state.get("last_attempt_error") or "")[:2000]
        self.awaiting_clarification = bool(state.get("awaiting_clarification"))

    def reset_conversation(self) -> None:
        """Clear all ephemeral authorization/evidence state (including /clear)."""

        self.pending_confirmation = False
        self.intent = INTENT_NONE
        self.attempted = False
        self.verified_instance = None
        self.last_attempt_error = ""
        self.awaiting_clarification = False

    def begin_user_turn(self, objective: str) -> str:
        prior_pending = self.pending_confirmation
        prior_clarification = self.awaiting_clarification
        if (prior_pending or prior_clarification) and _CANCEL_PENDING_RE.search(
            str(objective or "")
        ):
            self.pending_confirmation = False
            self.intent = INTENT_NONE
            self.attempted = False
            self.verified_instance = None
            self.last_attempt_error = ""
            self.awaiting_clarification = False
            return ""
        classified = (
            classify_project_manager_agent_intent(
                objective, pending_confirmation=prior_pending
            )
            if self.project_manager
            else INTENT_NONE
        )
        # A response to an explicit native ask_user or legacy get_user_input
        # clarification remains part of the durable creation transaction even
        # when the response is just a path.
        self.intent = (
            INTENT_CREATE
            if prior_clarification and classified == INTENT_NONE
            else classified
        )
        self.attempted = False
        self.verified_instance = None
        self.last_attempt_error = ""
        self.awaiting_clarification = False
        if self.intent == INTENT_CAPABILITY:
            # A prohibition is informational but is not a latent authorization
            # that a later context-free "go ahead" may unlock.
            self.pending_confirmation = not bool(
                _NEGATED_AUTHORIZATION_RE.search(str(objective or ""))
            )
            return (
                "DURABLE AGENT POLICY: This is a capability/planning question, not "
                "authorization. Explain the Nexus/Aeon plan and request explicit "
                "confirmation; do not call tools or change state."
            )
        if self.intent == INTENT_CREATE:
            self.pending_confirmation = False
            return (
                "DURABLE AGENT POLICY: This explicitly requests a Nexus-managed agent. "
                "Only start_agent_instance can satisfy it. Existing files, scripts, "
                "Ollama processes, memories, and health checks are not creation evidence."
            )
        self.pending_confirmation = False
        return ""

    @property
    def bypass_skill_routing(self) -> bool:
        return self.project_manager and self.intent != INTENT_NONE

    def prepare_actions(
        self, actions: Sequence[Mapping[str, Any]]
    ) -> tuple[list[Mapping[str, Any]], str]:
        """Fail closed before any disallowed action has a chance to execute."""

        proposed = list(actions)
        if not self.project_manager or self.intent == INTENT_NONE:
            return proposed, ""

        names = [str(action.get("tool_name") or "").strip() for action in proposed]
        if self.intent == INTENT_CAPABILITY:
            allowed = {"say_to_user", "get_user_input", "task_complete"}
            if all(name in allowed for name in names):
                return proposed, ""
            return [], (
                "DURABLE AGENT GUARD BLOCKED ACTIONS: a can/could/would/how request "
                "is informational and does not authorize start_agent_instance or any "
                "shell, file, skill, memory, website, or process action. Respond with "
                "the plan and ask for explicit confirmation."
            )

        if self.verified_instance is not None:
            allowed_after_success = {
                "say_to_user",
                "get_user_input",
                "task_complete",
            }
            if all(name in allowed_after_success for name in names):
                return proposed, ""
            return [], (
                "DURABLE AGENT GUARD BLOCKED EXTRA ACTIONS: Nexus has already "
                "returned the verified instance record for this request. Report "
                "that exact result and finish; do not operate the site, run shell "
                "commands, or make unrelated changes in the creation transaction."
            )

        for action in proposed:
            if str(action.get("tool_name") or "").strip() == "start_agent_instance":
                # The bridge is an observation boundary. No preliminary shell/file
                # checks and no pre-composed success prose execute around it.
                return [action], ""

        if not self.attempted and self._is_clarification_wait(proposed):
            self.awaiting_clarification = True
            return proposed, ""

        if self.attempted:
            allowed_after_failure = {
                "start_agent_instance",
                "say_to_user",
                "get_user_input",
                "task_complete",
            }
            if all(name in allowed_after_failure for name in names):
                return proposed, ""

        return [], (
            "DURABLE AGENT GUARD BLOCKED ACTIONS: this turn requests a durable "
            "Nexus/Aeon instance. Before a verified start_agent_instance receipt, "
            "do not run shell, file, skill, memory, Ollama, or health-check actions "
            "and do not report completion. Call start_agent_instance, or explicitly "
            "ask for missing information and wait for the user's answer."
        )

    def prepare_ask_user(self, message: object) -> str:
        """Validate and retain a native protocol clarification boundary."""

        if (
            not self.project_manager
            or self.intent != INTENT_CREATE
            or self.verified_instance is not None
        ):
            return ""
        value = str(message or "")
        if claims_agent_creation_success(value) or not _CLARIFICATION_RE.search(value):
            return (
                "DURABLE AGENT GUARD BLOCKED QUESTION: this turn requests a "
                "durable Nexus/Aeon instance. Ask only for specific missing "
                "creation information, or call start_agent_instance; do not imply "
                "that registration already happened."
            )
        self.awaiting_clarification = True
        return ""

    def resume_waiting_request(self, objective: object, question: object) -> None:
        """Re-arm safe pending state after a worker restart while awaiting input."""

        if (
            not self.project_manager
            or self.intent != INTENT_NONE
            or self.pending_confirmation
            or self.awaiting_clarification
            or self.attempted
            or self.verified_instance is not None
        ):
            return
        self.begin_user_turn(str(objective or ""))
        if self.intent == INTENT_CREATE:
            self.prepare_ask_user(question)

    @staticmethod
    def _is_clarification_wait(actions: Sequence[Mapping[str, Any]]) -> bool:
        names = [str(action.get("tool_name") or "").strip() for action in actions]
        if names == ["get_user_input"]:
            prompt = str(actions[0].get("parameters", {}).get("prompt", ""))
            return (
                not claims_agent_creation_success(prompt)
                and bool(_CLARIFICATION_RE.search(prompt))
            )
        if names == ["say_to_user", "get_user_input"]:
            message = str(actions[0].get("parameters", {}).get("message", ""))
            prompt = str(actions[1].get("parameters", {}).get("prompt", ""))
            return (
                not claims_agent_creation_success(message)
                and bool(_CLARIFICATION_RE.search(message + " " + prompt))
            )
        return False

    def observe_tool_result(self, tool_name: str, raw_result: object) -> None:
        if (
            not self.project_manager
            or self.intent != INTENT_CREATE
            or str(tool_name).strip() != "start_agent_instance"
        ):
            return
        self.attempted = True
        if isinstance(raw_result, VerifiedNexusAgentStart):
            self.verified_instance = dict(raw_result.instance)
            self.last_attempt_error = ""
        else:
            self.verified_instance = None
            self.last_attempt_error = str(raw_result)

    def visible_claim_error(self, message: object) -> str:
        if (
            self.project_manager
            and self.intent == INTENT_CREATE
            and self.verified_instance is not None
            and self.verified_instance.get("awaiting_objective") is True
            and _claims_deferred_agent_is_active(message)
        ):
            return (
                "DURABLE AGENT GUARD BLOCKED FALSE ACTIVE STATE: Nexus registered "
                "the Aeon tab, but it is idle and awaiting the user's first message. "
                "Do not claim that it started, is running, or performed work."
            )
        if (
            self.project_manager
            and self.intent in {INTENT_CAPABILITY, INTENT_CREATE}
            and self.verified_instance is None
            and claims_agent_creation_success(str(message or ""))
        ):
            return (
                "DURABLE AGENT GUARD BLOCKED FALSE SUCCESS: no typed, verified "
                "start_agent_instance receipt exists in this user turn. The message "
                "was not printed or appended to the Nexus transcript. Existing "
                "scripts, memories, Ollama/llama processes, and old claims do not count."
            )
        return ""

    def completion_error(self, reason: object) -> str:
        if not self.project_manager or self.intent == INTENT_NONE:
            return ""
        value = str(reason or "")
        if self.verified_instance is not None:
            if (
                self.verified_instance.get("awaiting_objective") is True
                and _claims_deferred_agent_is_active(value)
            ):
                return (
                    "DURABLE AGENT GUARD BLOCKED FALSE COMPLETION: the registered "
                    "Aeon is idle and awaiting the user's first message; it has not "
                    "started or performed work."
                )
            return ""
        if claims_agent_creation_success(value):
            return (
                "DURABLE AGENT GUARD BLOCKED FALSE COMPLETION: task_complete "
                "claimed an agent exists without a typed, verified "
                "start_agent_instance receipt in this user turn."
            )
        if self.intent == INTENT_CREATE and not _TRUTHFUL_NONCOMPLETION_RE.search(value):
            return (
                "DURABLE AGENT GUARD BLOCKED COMPLETION: an explicit durable-agent "
                "creation request cannot be marked done without a typed, verified "
                "start_agent_instance receipt. Retry the bridge or report its actual "
                "failure/blocker truthfully."
            )
        return ""
