"""Deterministic control-plane types for the Aeon agent loop.

The language model proposes a turn; this module decides whether that turn is
allowed, what a tool result actually means, and whether the task may be reported
as complete.  Keeping these decisions outside the prompt prevents a weaker local
model from granting itself authority or treating persuasive prose as evidence.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import shlex
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


class RequestMode(str, Enum):
    ANSWER = "answer"
    INSPECT = "inspect"
    PLAN = "plan"
    CHANGE_LOCAL = "change_local"
    EXTERNAL_ACTION = "external_action"
    DESTRUCTIVE = "destructive"


class ExecutionState(str, Enum):
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    WAITING_COMPUTE = "waiting_compute"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TurnKind(str, Enum):
    TOOL_CALLS = "tool_calls"
    FINAL = "final"
    ASK_USER = "ask_user"
    WAIT = "wait"


class SideEffect(str, Enum):
    READ_ONLY = "read_only"
    AGENT_STATE = "agent_state"
    LOCAL_MUTATION = "local_mutation"
    EXTERNAL_MUTATION = "external_mutation"
    DESTRUCTIVE = "destructive"
    DYNAMIC = "dynamic"
    CONTROL = "control"


class CapabilityFamily(str, Enum):
    """Harness-owned identity for consequential capabilities.

    RequestMode is only a ceiling.  These families bind an explicit request to
    the *kind* of consequential action the owner named, so an unrelated tool at
    the same side-effect level cannot substitute for it.
    """

    GITHUB = "github"
    GITHUB_CREATE = "github_create"
    EXTERNAL_INTERACTION = "external_interaction"
    AGENT_INSTANCE = "agent_instance"
    EXTERNAL_EXPERT = "external_expert"
    COLLABORATION_PORTAL = "collaboration_portal"
    JOB_ROLE = "job_role"
    MCP_CONNECTION = "mcp_connection"
    DELETE_RESOURCE = "delete_resource"
    KILL_JOB = "kill_job"
    KILL_SUB_AGENT = "kill_sub_agent"
    RESTART_AEON = "restart_aeon"
    REVERT_AEON = "revert_aeon"
    SERVICE_CONTROL = "service_control"
    PROCESS_CONTROL = "process_control"
    SOURCE_REVERT = "source_revert"
    ACCESS_REVOCATION = "access_revocation"


class ToolStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    BLOCKED = "blocked"
    PENDING = "pending"
    NO_CHANGE = "no_change"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    side_effect: SideEffect
    observation_boundary: bool = False
    idempotent: bool = False
    approval_required: bool = False
    self_verifying: bool = False
    retry_limit: int = 0


@dataclass
class ToolResult:
    tool_name: str
    status: ToolStatus
    changed: bool
    summary: str
    evidence: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    error_code: str = ""
    retryable: bool = False
    side_effect: SideEffect = SideEffect.READ_ONLY
    call_id: str = ""
    result_ref: str = ""
    result_sha256: str = ""
    result_chars: int = 0
    raw: Any = field(default=None, repr=False)

    @property
    def successful(self) -> bool:
        return self.status == ToolStatus.OK

    def to_model_dict(self) -> dict[str, Any]:
        data = {
            "tool": self.tool_name,
            "call_id": self.call_id,
            "status": self.status.value,
            "changed": self.changed,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "artifacts": list(self.artifacts),
            "error_code": self.error_code or None,
            "retryable": self.retryable,
        }
        # Keep ordinary receipts byte-for-byte compatible. Archive metadata is
        # additive only when a complete oversized result actually exists.
        if self.result_ref and self.result_sha256 and self.result_chars > 0:
            data.update(
                {
                    "result_ref": self.result_ref,
                    "result_sha256": self.result_sha256,
                    "result_chars": self.result_chars,
                }
            )
        return data

    def to_model_text(self) -> str:
        return json.dumps(self.to_model_dict(), ensure_ascii=False, default=str)

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize only bounded, non-executable receipt data."""

        data = self.to_model_dict()
        data["side_effect"] = self.side_effect.value
        return data

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any]) -> "ToolResult":
        status_value = str(data.get("status") or ToolStatus.FAILED.value)
        effect_value = str(data.get("side_effect") or SideEffect.READ_ONLY.value)
        try:
            status = ToolStatus(status_value)
        except ValueError:
            status = ToolStatus.FAILED
        try:
            effect = SideEffect(effect_value)
        except ValueError:
            effect = SideEffect.READ_ONLY
        try:
            result_chars = max(
                0, min(100_000_000, int(data.get("result_chars") or 0))
            )
        except (TypeError, ValueError):
            result_chars = 0
        result_ref = str(data.get("result_ref") or "")
        result_sha256 = str(data.get("result_sha256") or "")
        if not (
            result_chars > 0
            and re.fullmatch(r"tr_[0-9a-f]{32}_[0-9a-f]{16}", result_ref)
            and re.fullmatch(r"[0-9a-f]{64}", result_sha256)
        ):
            result_ref = ""
            result_sha256 = ""
            result_chars = 0
        return cls(
            tool_name=str(data.get("tool") or "unknown")[:200],
            status=status,
            changed=bool(data.get("changed", False)),
            summary=str(data.get("summary") or "")[:1600],
            evidence=[str(item)[:500] for item in list(data.get("evidence") or [])[:8]],
            artifacts=[str(item)[:1000] for item in list(data.get("artifacts") or [])[:20]],
            error_code=str(data.get("error_code") or "")[:100],
            retryable=bool(data.get("retryable", False)),
            side_effect=effect,
            call_id=str(data.get("call_id") or "")[:200],
            result_ref=result_ref,
            result_sha256=result_sha256,
            result_chars=result_chars,
        )


@dataclass(frozen=True)
class RunOutcome:
    state: ExecutionState
    message: str = ""
    request_id: str = ""
    evidence: tuple[str, ...] = ()

    @property
    def completed(self) -> bool:
        return self.state == ExecutionState.DONE


@dataclass(frozen=True)
class ContractDelta:
    """Typed progress emitted by one observed receipt.

    Acceptance progress changes an owner-bound obligation. Information progress
    adds a genuinely new receipt digest but never substitutes for completion.
    """

    acceptance_advanced: tuple[str, ...] = ()
    information_added: tuple[str, ...] = ()
    obligations_opened: tuple[str, ...] = ()
    obligations_closed: tuple[str, ...] = ()


_DESTRUCTIVE_REQUEST_RE = re.compile(
    r"\b(delete|erase|destroy|purge|wipe|remove|drop|truncate|stop|restart|"
    r"kill|terminate|reset|revert|rollback|revoke|uninstall|prune|discard|"
    r"cancel|abort|unpublish|deactivate)\b|"
    r"\bclear\b[^?.!\n]{0,60}\b(?:cache|queue|database|table|storage|state)\b|"
    r"\bshut\s+down\b[^?.!\n]{0,60}\b(?:service|process|daemon|server|host)\b",
    re.IGNORECASE,
)
_EXTERNAL_REQUEST_RE = re.compile(
    r"\b(publish|make live|deploy publicly|schedule|book|order|pay|subscribe|git\s+push|"
    r"(?:git\s+)?push\b[^?.!\n]{0,200}\b(?:github|gitlab)\b|"
    r"push\s+(?:the\s+)?(?:github|gitlab)\s+(?:project|repo(?:sitory)?)|"
    r"(?:back\s*up|(?:commit\s+and\s+)?push)\s+"
    r"(?:all|every)\s+(?:the\s+)?(?:current\s+)?(?:files?|changes?|edits?|work)\s+"
    r"to\s+(?:github|gitlab)|push(?:\s+(?:(?:an?|the)\s+)?update)?"
    r"(?:\s+(?:this|it|the\s+(?:repo|repository|project)))?\s+to\s+(?:github|gitlab)|"
    r"update\s+(?:the\s+)?(?:github|gitlab)\s+(?:project|repo(?:sitory)?)|"
    r"update\s+(?:the\s+)?(?:github|gitlab)(?=\s*(?:[?.!,;:]|$))|"
    r"send (?:an? )?(?:email|message|tweet|post)|"
    r"(?:send|email)\s+(?:(?:the|this|an?)\s+)?"
    r"(?:report|email|message|update|summary|results?)\b|"
    r"(?:send|email|message|notify)\s+"
    r"(?:[A-Z][A-Za-z0-9_.-]{1,63}|the\s+(?:team|owner|user|client)|"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b|"
    r"reply\s+to\s+(?:[A-Z][A-Za-z0-9_.-]{1,63}|the\s+(?:team|owner|user|client))\b|"
    r"forward\b[^?.!\n]{0,100}\bto\s+"
    r"(?:[A-Z][A-Za-z0-9_.-]{1,63}|the\s+(?:team|owner|user|client))\b|"
    r"share\b[^?.!\n]{0,100}\bwith\s+"
    r"(?:[A-Z][A-Za-z0-9_.-]{1,63}|the\s+(?:team|owner|user|client))\b|"
    r"post (?:it|this|on|to)|"
    r"upload|submit|purchase|buy|log ?in|sign ?in|create (?:a |an )?(?:(?:github|gitlab) )?"
    r"(?:account|repository|repo|pull request|issue|agent|tab)|"
    r"(?:create|add|spawn|provision|set[ -]?up) "
    r"(?:(?:a|an|another) )?(?:new )?(?:agent|session|tab)|"
    r"(?:start|launch) (?:(?:a|an|another) )?(?:new )?(?:agent|session|tab)|"
    r"make (?:me |us )?(?:(?:a|an|another) )?(?:new )?"
    r"(?:agent|session|tab|repository|repo)|"
    r"consult (?:an? )?(?:external expert|codex|claude|grok|gemini)|"
    r"(?:create|request|open) (?:a |an )?(?:collaboration|collaborator) portal|"
    r"(?:change|set|update|restore) (?:my |the )?job role|"
    r"(?:connect|link|authorize) (?:my |the |an? )?(?:mcp |external )?account|"
    r"register|provision)\b",
    re.IGNORECASE,
)

# ``upload`` above deliberately remains a verb-shaped token.  The production
# continuous Hugging Face goal instead says "through making useful uploads".
# Treat that bounded action construction as external authority without turning
# every informational reference to plural uploads into permission to publish.
_ACTIONABLE_PLURAL_UPLOAD_RE = re.compile(
    r"(?:\b(?:make|create|produce|perform|do)\b|"
    r"\b(?:through|by|keep|continue|start)\s+"
    r"(?:making|creating|producing|performing|doing)\b)"
    r"[^?.!\n]{0,120}\buploads\b",
    re.IGNORECASE,
)

# A direct polite request is still authority to perform the named external
# action.  Do not demote "can you push ... now?" to a hypothetical plan merely
# because it is phrased as a question.  This is deliberately narrower than the
# generic modal-question matcher below, so "can you explain how to push" remains
# informational.
_DIRECT_POLITE_EXTERNAL_REQUEST_RE = re.compile(
    r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:"
    r"(?:git\s+)?push\b|"
    r"(?:update|back\s*up)\b[^?.!]*\b(?:github|gitlab)\b|"
    r"(?:publish|upload)\b[^?.!]*\b(?:github|gitlab|repo(?:sitory)?|project)\b"
    r")",
    re.IGNORECASE,
)
_CHANGE_REQUEST_RE = re.compile(
    r"\b(fix|implement|build|write|generate|render|commit|change|modify|edit|update|refactor|migrate|"
    r"add|create|make|apply|configure|set up|install|optimi[sz]e|integrate|repair|"
    r"replace|enable|disable|improve|enhance|strengthen|harden|streamline|clean\s+up|"
    r"persist|save|refine|revise|rework|overhaul|patch|hotfix|eliminate|introduce|"
    r"extract|inline|split|merge|consolidate|deduplicate|reformat|format|autofix|"
    r"scaffold|wire|instrument|memoize|"
    r"speed\s+up|go ahead|proceed|do it|do all)\b|"
    r"\b(?:clean|speed)\b[^?.!\n]{0,60}\bup\b|"
    r"\bget\b[^?.!\n]{0,60}\bworking\b|"
    r"\b(?:i\s+)?need\b[^?.!\n]{0,60}\bfixed\b",
    re.IGNORECASE,
)
_INSPECT_REQUEST_RE = re.compile(
    r"\b(audit|inspect|investigate|diagnose|review|analy[sz]e|check|verify|assess|"
    r"evaluate|examine|look\s+into|trace|profile|take a (?:long )?look|deep dive|"
    r"find (?:the )?(?:cause|problem|issue)|what(?:'s| is) wrong|why (?:is|does|did))\b",
    re.IGNORECASE,
)
_LEADING_INSPECTION_RE = re.compile(
    r"^\s*(?:(?:i\s+(?:want|need)\s+you\s+to|please)\s+)?"
    r"(?:audit|inspect|investigate|diagnose|review|analy[sz]e|check|verify|assess|"
    r"evaluate|examine|look\s+into|trace|profile|"
    r"take\s+a\s+(?:long\s+)?look|deep\s+dive)\b",
    re.IGNORECASE,
)
_PLAN_REQUEST_RE = re.compile(
    r"\b(plan|proposal|recommend|suggest|options|tradeoffs|how (?:would|could|can)"
    r"\s+(?:we|you|i)|without (?:changing|modifying|implementing))\b",
    re.IGNORECASE,
)

# A terse status check asks for live project/task evidence even though it often
# contains none of the verbs in _INSPECT_REQUEST_RE (the production failure was
# literally ``Hello? Status?``). Keep this narrower than a bare occurrence of
# "status" so informational questions such as "What does HTTP status mean?"
# remain ordinary answers.
_STATUS_INSPECTION_RE = re.compile(
    r"(?:^|[.!?]\s*)(?:(?:hello|hi|hey)\W+)?(?:please\s+)?"
    r"(?:(?:status|progress)(?:\s+update)?|update)(?:\s+(?:please|report))?"
    r"\s*[?!.]*\s*$|"
    r"\b(?:give|show|tell)\s+me\s+(?:(?:a|an|the)\s+)?(?:current\s+)?"
    r"(?:(?:status|progress)(?:\s+update)?|update)(?:\s+(?:on|for|about)\b|\s*[?!.]*$)|"
    r"\bwhat(?:'s| is)\s+(?:(?:the|your|our)\s+)?(?:current\s+)?"
    r"(?:(?:project|workspace|task|site|agent)\s+)?(?:status|progress)"
    r"(?:\s+(?:of|on|for)\b|\s*[?!.]*$)|"
    r"\b(?:current|project|workspace|task|site|agent)\s+(?:status|progress)"
    r"(?:\s+(?:of|on|for)\b|\s*[?!.]*$)",
    re.IGNORECASE,
)
_GITHUB_SCOPE_RE = re.compile(
    r"\b(?:github|gitlab|git\s+push|push(?:ing|ed)?)\b",
    re.IGNORECASE,
)
_COMPLETE_BACKUP_SCOPE_RE = re.compile(
    r"\b(?:all|every)\s+(?:the\s+)?(?:current\s+)?(?:files?|changes?|edits?|work)\b|"
    r"\beverything\b|\bback\s*up\s+(?:the\s+)?(?:whole|entire)\b",
    re.IGNORECASE,
)

_GITHUB_CAPABILITY_RE = re.compile(
    r"\b(?:github|gitlab|git\s+push|git\s+commit|"
    r"(?:commit\s+and\s+)?push(?:ing|ed)?\s+(?:the\s+)?(?:update|changes?|repo(?:sitory)?)?)\b",
    re.IGNORECASE,
)
_GITHUB_CREATE_CAPABILITY_RE = re.compile(
    r"\b(?:create|open)\b[^?.!]{0,80}\b(?:github|gitlab)\b[^?.!]{0,80}"
    r"\b(?:repository|repo|pull\s+request|issue)\b|"
    r"\b(?:create|open)\b[^?.!]{0,80}"
    r"\b(?:repository|repo|pull\s+request|issue)\b[^?.!]{0,80}"
    r"\b(?:on|in)\s+(?:github|gitlab)\b",
    re.IGNORECASE,
)
_LOCAL_GITHUB_COMMIT_CAPABILITY_RE = re.compile(
    r"(?:^|[.!?;]\s*|\bthen\s+)(?:please\s+)?commit\b",
    re.IGNORECASE,
)
_AGENT_INSTANCE_CAPABILITY_RE = re.compile(
    r"\b(?:(?:create|add|make|start|launch|spawn|provision|register|set[ -]?up)\s+"
    r"(?:(?:me|us)\s+)?(?:(?:a|an|another)\s+)?(?:new\s+)?"
    r"(?:(?:durable|idle|managed|aeon)\s+){0,4}"
    # A confirmation prompt naturally includes a short display name, as in
    # "register an idle Bananacoconut agent tab".  Keep this deliberately
    # bounded and require the explicit agent/session/tab noun so unrelated
    # uses of the generic verb "register" cannot acquire this capability.
    r"(?:[A-Za-z0-9][A-Za-z0-9._-]{0,63}\s+){0,2}"
    r"(?:agents?|agent\s+tabs?|agent\s+sessions?|sessions?|tabs?)|"
    r"(?:agent|session|tab)\s+(?:instance|creation|registration))\b",
    re.IGNORECASE,
)
_EXTERNAL_EXPERT_CAPABILITY_RE = re.compile(
    r"\b(?:consult|ask|use)\s+(?:(?:a|an|the)\s+)?"
    r"(?:external\s+expert|codex|claude|grok|gemini)\b",
    re.IGNORECASE,
)
_COLLABORATION_PORTAL_CAPABILITY_RE = re.compile(
    r"\b(?:collaboration|collaborator)\s+portal\b",
    re.IGNORECASE,
)
_JOB_ROLE_CAPABILITY_RE = re.compile(r"\bjob\s+role\b", re.IGNORECASE)
_MCP_CONNECTION_CAPABILITY_RE = re.compile(
    r"\b(?:connect|link|authorize)\b[^?.!]{0,80}\b(?:mcp|account|credential)\b|"
    r"\bmcp\b[^?.!]{0,80}\b(?:connect|link|authorize)\b",
    re.IGNORECASE,
)
_EXTERNAL_INTERACTION_CAPABILITY_RE = re.compile(
    r"\b(?:publish|make\s+live|deploy(?:\s+publicly)?|send|post|upload|submit|"
    r"purchase|buy|schedule|book|order|pay|subscribe|log\s*in|sign\s*in|register|create\s+(?:an?\s+)?"
    r"(?:account|repository|repo|pull\s+request|issue))\b",
    re.IGNORECASE,
)
_STRONG_EXTERNAL_INTERACTION_CAPABILITY_RE = re.compile(
    r"\b(?:publish|make\s+live|deploy(?:\s+publicly)?|send|post|upload|submit|"
    r"purchase|buy|schedule|book|order|pay|subscribe|log\s*in|sign\s*in)\b",
    re.IGNORECASE,
)

_AEON_RESTART_CAPABILITY_RE = re.compile(
    r"\b(?:restart|reload)\b[^?.!]{0,50}\baeon\b|"
    r"\baeon\b[^?.!]{0,50}\b(?:restart|reload)\b",
    re.IGNORECASE,
)
_AEON_REVERT_CAPABILITY_RE = re.compile(
    r"\b(?:revert|rollback|restore)\b[^?.!]{0,50}\baeon\b|"
    r"\baeon\b[^?.!]{0,50}\b(?:revert|rollback|restore)\b",
    re.IGNORECASE,
)
_KILL_SUB_AGENT_CAPABILITY_RE = re.compile(
    r"\b(?:kill|stop|terminate|remove)\b[^?.!]{0,50}\bsub[ -]?agent\b|"
    r"\bsub[ -]?agent\b[^?.!]{0,50}\b(?:kill|stop|terminate|remove)\b",
    re.IGNORECASE,
)
_KILL_JOB_CAPABILITY_RE = re.compile(
    r"\b(?:kill|stop|terminate)\b[^?.!]{0,50}\b(?:background\s+)?jobs?\b|"
    r"\b(?:background\s+)?jobs?\b[^?.!]{0,50}\b(?:kill|stop|terminate)\b",
    re.IGNORECASE,
)
_SKILL_STATE_DELETION_RE = re.compile(
    r"\b(?:delete|remove|erase|uninstall)\b[^?.!]{0,50}"
    r"\b(?:skill(?:[- ](?:wiki|knowledge))?(?:\s+(?:note|entry|protocol))?|"
    r"wiki\s+(?:note|entry)|knowledge\s+note)\b|"
    r"\b(?:skill(?:[- ](?:wiki|knowledge))?(?:\s+(?:note|entry|protocol))?|"
    r"wiki\s+(?:note|entry)|knowledge\s+note)\b[^?.!]{0,50}"
    r"\b(?:delete|remove|erase|uninstall)\b",
    re.IGNORECASE,
)
_SERVICE_CONTROL_CAPABILITY_RE = re.compile(
    r"\b(?:stop|restart|disable|terminate)\b[^?.!]{0,50}\bservice\b|"
    r"\bservice\b[^?.!]{0,50}\b(?:stop|restart|disable|terminate)\b|"
    r"\b(?:restart|disable)\s+(?:the\s+)?[A-Za-z0-9_.@-]+(?:\.service)?\b",
    re.IGNORECASE,
)
_PROCESS_CONTROL_CAPABILITY_RE = re.compile(
    r"\b(?:kill|stop|terminate)\b[^?.!]{0,50}\b(?:process(?:es)?|pids?|daemons?)\b|"
    r"\b(?:process(?:es)?|pids?|daemons?)\b[^?.!]{0,50}\b(?:kill|stop|terminate)\b",
    re.IGNORECASE,
)
_SOURCE_REVERT_CAPABILITY_RE = re.compile(
    r"\b(?:revert|rollback|reset|restore)\b[^?.!]{0,50}"
    r"\b(?:source|code|commit|branch|repository|repo|changes?)\b|"
    r"\b(?:source|code|commit|branch|repository|repo|changes?)\b[^?.!]{0,50}"
    r"\b(?:revert|rollback|reset|restore)\b",
    re.IGNORECASE,
)
_ACCESS_REVOCATION_CAPABILITY_RE = re.compile(
    r"\b(?:revoke|remove|disable)\b[^?.!]{0,50}"
    r"\b(?:access|credential|token|account|permission)\b|"
    r"\b(?:access|credential|token|account|permission)\b[^?.!]{0,50}"
    r"\b(?:revoke|remove|disable)\b",
    re.IGNORECASE,
)
_DELETE_RESOURCE_CAPABILITY_RE = re.compile(
    r"\b(?:delete|erase|destroy|purge|wipe|remove|drop|truncate|uninstall)\b",
    re.IGNORECASE,
)
_SOURCE_EDIT_DELETION_RE = re.compile(
    r"\b(?:delete|remove|drop)\b[^?.!\n]{0,100}\b(?:unused|obsolete|redundant|stale)?\s*"
    r"(?:import|function|method|class|argument|parameter|variable|constant|line|"
    r"code|handler|check|case|clause|dependency|setting|field|property)\b",
    re.IGNORECASE,
)
_AGENT_STATE_REQUEST_RE = re.compile(
    r"\b(?:create|add|update|revise|delete|remove|activate|deactivate)\b"
    r"[^.!?\n]{0,40}\bskill\s+"
    r"(?:[`'\"])?[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:[`'\"])?\b|"
    r"\b(?:remember|memorize|forget)\b[^.!?\n]{0,120}"
    r"\b(?:memory|fact|lesson|knowledge|preference|instruction|note|this|that|it)\b|"
    r"\b(?:delete|remove)\b[^.!?\n]{0,80}\b(?:wiki|knowledge)\s+(?:note|entry)\b|"
    r"\bmaintain\b[^.!?\n]{0,100}\b(?:learned\s+skills?|agent\s+memory)\b",
    re.IGNORECASE,
)
_LOCAL_CHANGE_IMPERATIVE_RE = re.compile(
    r"^\s*(?:(?:i\s+(?:want|need)\s+you\s+to|please)\s+)?"
    r"(?:rename|move|copy|document|correct|resolve|address|upgrade|bump|pin|port|"
    r"convert|simplify|modernize|ensure|wrap|expose|transform|reorganize|restructure|"
    r"persist|save|refine|revise|rework|overhaul|patch|hotfix|eliminate|introduce|"
    r"extract|inline|split|merge|consolidate|deduplicate|reformat|format|autofix|"
    r"scaffold|wire|instrument|memoize)\b|"
    r"^\s*(?:(?:i\s+(?:want|need)\s+you\s+to|please)\s+)?turn\b"
    r"[^?.!\n]{0,100}\binto\b",
    re.IGNORECASE,
)
_CONVERSATIONAL_ARTIFACT_RE = re.compile(
    r"^\s*(?:(?:i\s+(?:want|need)\s+you\s+to|please)\s+)?"
    r"(?:write|generate|create|build|make|draft|produce)\b[^?.!\n]{0,100}\b"
    r"(?:answer|response|poem|story|ideas?|plan|proposal|argument|recommendation|"
    r"table|list|outline|summary|explanation|description|draft|copy|text)\b",
    re.IGNORECASE,
)
_PERSISTENT_DESTINATION_RE = re.compile(
    r"\b(?:file|path|directory|folder|repo(?:sitory)?|project|workspace|codebase|"
    r"database|service|site|website|app|application|module|package)\b|"
    r"(?:^|\s)(?:\.?\.?/|~?/)[^\s]+|\b[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,12}\b",
    re.IGNORECASE,
)
_INLINE_OUTPUT_RE = re.compile(
    r"\b(?:here|in\s+(?:the|your)\s+(?:answer|response|chat))\b|"
    r"\b(?:do\s+not|don['’]?t|dont|without)\b[^?.!\n]{0,60}\b"
    r"(?:create|write|save|make)\b[^?.!\n]{0,40}\bfiles?\b|\bno\s+files?\b",
    re.IGNORECASE,
)

_EVIDENCE_ACTION_HEAD_RE = re.compile(
    r"^(?:read|look\s+at|look\s+into|dig\s+into|research|search|find|locate|list|"
    r"show(?:\s+(?:me\s+)?)?(?:the\s+)?diff|compare|diff|scan|survey|map|measure|"
    r"benchmark|test|validate|verify|reproduce|triage|debug|monitor|observe|inventory|"
    r"figure\s+out|check|assess|evaluate|examine|trace|profile|audit|inspect|"
    r"investigate|diagnose|review|analy[sz]e)\b",
    re.IGNORECASE,
)
_LOCAL_ACTION_HEAD_RE = re.compile(
    r"^(?:fix|implement|build|write|generate|render|commit|change|modify|edit|update|"
    r"refactor|migrate|add|create|make|apply|configure|set\s+up|install|optimi[sz]e|"
    r"integrate|repair|replace|enable|disable|improve|enhance|strengthen|harden|"
    r"streamline|persist|save|refine|revise|rework|overhaul|patch|hotfix|eliminate|"
    r"introduce|extract|inline|split|merge|consolidate|deduplicate|reformat|format|"
    r"autofix|scaffold|wire|instrument|memoize|rewrite|adjust|tweak|amend|alter|"
    r"bootstrap|normalize|canonicalize|serialize|type\s+annotate|lint|prettify|minify|"
    r"containerize|mock|stub|seed|populate|index|cache|regenerate|clean|rename|move|"
    r"copy|document|correct|resolve|address|upgrade|bump|pin|port|convert|simplify|"
    r"modernize|ensure|wrap|expose|transform|reorganize|restructure)\b",
    re.IGNORECASE,
)
_EXTERNAL_ACTION_HEAD_RE = re.compile(
    r"^(?:publish|deploy|upload|submit|push|send|share|email|message|notify|reply|forward|"
    r"dm|ping|invite|tweet|post|purchase|buy|register|provision|log\s*in|sign\s*in|"
    r"schedule|book|order|pay|subscribe|connect|link|authorize|"
    r"open\s+(?:a\s+)?(?:github|gitlab)\s+(?:issue|pull\s+request)|"
    r"merge\s+(?:the\s+)?(?:pr|pull\s+request))\b",
    re.IGNORECASE,
)
_DESTRUCTIVE_ACTION_HEAD_RE = re.compile(
    r"^(?:delete|erase|destroy|purge|wipe|remove|drop|truncate|stop|restart|kill|"
    r"terminate|reset|revert|rollback|revoke|uninstall|prune|discard|cancel|abort|"
    r"unpublish|deactivate|shutdown|shut\s+down|flush|empty|unlink|disconnect|suspend|"
    r"ban|unlist|detach|expire|overwrite|rebase|squash)\b",
    re.IGNORECASE,
)


def _clause_head_mode(value: str) -> RequestMode | None:
    """Classify the positive head of one clause from a shared verb taxonomy."""

    action = " ".join(str(value or "").strip().split())
    action = re.sub(
        r"^(?:(?:i\s+(?:want|need)\s+you\s+to|please|go\s+ahead\s+and|then|now)\s+)+",
        "",
        action,
        flags=re.IGNORECASE,
    )
    if re.match(
        r"^(?:change|set|update|restore)\b[^?!.]{0,80}\bjob\s+role\b|"
        r"^(?:create|add|make|start|launch|spawn|provision|register|set\s+up)\b"
        r"[^?!.]{0,100}\b(?:agent|session|tab|account|repository|repo|"
        r"pull\s+request|issue|collaboration\s+portal)\b",
        action,
        re.IGNORECASE,
    ):
        return RequestMode.EXTERNAL_ACTION
    if _EVIDENCE_ACTION_HEAD_RE.match(action):
        return RequestMode.INSPECT
    if _DESTRUCTIVE_ACTION_HEAD_RE.match(action):
        return RequestMode.DESTRUCTIVE
    if _EXTERNAL_ACTION_HEAD_RE.match(action):
        # "share your thoughts" and similar inline content are answers, not an
        # outbound effect. A recipient/platform/destination makes it external.
        if re.match(r"^(?:share|send)\b", action, re.IGNORECASE) and not re.search(
            r"\b(?:to|with|on|via)\b|@[A-Za-z0-9_]+|"
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            action,
            re.IGNORECASE,
        ):
            return None
        return RequestMode.EXTERNAL_ACTION
    if _LOCAL_ACTION_HEAD_RE.match(action):
        return RequestMode.CHANGE_LOCAL
    # Unknown positive imperative + a persistent destination is safer and more
    # useful as a bounded local change than silently treating it as prose.
    if (
        re.match(r"^[A-Za-z][A-Za-z-]{2,}\b", action)
        and not re.match(
            r"^(?:what|why|when|where|which|who|whose|how|can|could|would|"
            r"should|may|might|is|are|was|were|do|does|did)\b",
            action,
            re.IGNORECASE,
        )
        and _PERSISTENT_DESTINATION_RE.search(action)
    ):
        return RequestMode.CHANGE_LOCAL
    return None


_DIRECT_POLITE_ACTION_RE = re.compile(
    r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?(?P<action>.+)$",
    re.IGNORECASE,
)

_INFORMATIONAL_REQUEST_RE = re.compile(
    r"^\s*(?:explain|describe|summari[sz]e|outline|tell\s+me|show\s+me)\b|"
    r"^\s*(?:is|would)\s+it\s+(?:be\s+)?possible\b|"
    r"^\s*what\s+would\s+happen\b|"
    r"^\s*how\b|"
    r"^\s*(?:what|why|when|where|which)\s+"
    r"(?:is|are|was|were|would|could|can|should|do|does|did)\b|"
    r"^\s*(?:can|could|would|should|may|might)\s+(?:you|we|i)\b",
    re.IGNORECASE,
)
_EXPLICIT_IMPERATIVE_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:(?:then|and\s+then)\s+)?"
    r"(?:(?:i\s+(?:want|need)\s+you\s+to|please)\s+)?"
    r"(?:go ahead|do it|do this|proceed|"
    r"fix|implement|build|write|generate|render|commit|change|modify|edit|update|refactor|apply|"
    r"improve|enhance|strengthen|harden|streamline|clean\s+up|speed\s+up|"
    r"persist|save|refine|revise|rework|overhaul|patch|hotfix|eliminate|introduce|"
    r"extract|inline|split|merge|consolidate|deduplicate|reformat|format|autofix|"
    r"scaffold|wire|instrument|memoize|"
    r"publish|deploy|send|upload|submit|create|start|delete|remove|restart|get)\b",
    re.IGNORECASE,
)
_CONFIRMATION_RE = re.compile(
    r"^\s*(?:yes[,.! ]*)?(?:go ahead|do it|do this|proceed|apply it|"
    r"make it|start it|create it|publish it|delete it|remove it|restart it)"
    r"[.! ]*$|^\s*yes[.! ]*$",
    re.IGNORECASE,
)
_NEGATIVE_CONFIRMATION_RE = re.compile(
    r"^\s*(?:no|nope|do\s+not|don['’]?t|dont|cancel|never\s+mind)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_ADDITIVE_FOLLOWUP_RE = re.compile(
    r"\b(?:also|too|as\s+well|in\s+addition|additionally|along\s+with)\b",
    re.IGNORECASE,
)
_NEGATED_ACTION_CLAUSE_RE = re.compile(
    r"\b(?:do\s+not|don['’]?t|dont|never)\s+(?:please\s+)?"
    r"(?P<verb>do|proceed|apply|change|edit|modify|fix|delete|remove|erase|"
    r"destroy|purge|wipe|stop|restart|kill|terminate|reset|revert|rollback|"
    r"revoke|uninstall|publish|deploy|send|share|email|message|post|upload|"
    r"submit|push|purchase|buy|register|provision|connect|create|start)\b"
    r"(?P<object>[^;.!?]{0,100})",
    re.IGNORECASE,
)
_EXPLICIT_READ_ONLY_RE = re.compile(
    r"\bread[ -]?only\b|"
    r"\bno\s+(?:code\s+|file\s+|state\s+)?(?:changes?|edits?|modifications?)\b|"
    r"\bdo\s+not\s+(?:make|apply)\s+(?:any\s+)?(?:changes?|edits?|modifications?)\b|"
    r"\bdon['’]?t\s+(?:make|apply)\s+(?:any\s+)?(?:changes?|edits?|modifications?)\b|"
    r"\b(?:do\s+not|don['’]?t|dont)\s+"
    r"(?:change|edit|modify)\s+(?:anything|any\s+(?:files?|code|state))\b|"
    r"\bwithout\s+(?:making|applying)\s+(?:any\s+)?(?:changes?|edits?|modifications?)\b|"
    r"\bwithout\s+changing\s+(?:anything|any\s+(?:files?|code|state))\b",
    re.IGNORECASE,
)
_SCOPED_READ_ONLY_RE = re.compile(
    r"\bno\s+(?:code\s+|file\s+|state\s+)?changes?\s+"
    r"(?:to|outside|in)\b|"
    r"\b(?:do\s+not|don['’]?t|dont)\s+(?:make|apply)\s+"
    r"(?:any\s+)?changes?\s+to\b",
    re.IGNORECASE,
)
_LATER_EXPLICIT_ACTION_RE = re.compile(
    r"(?:^|[.!?;]\s*|,\s*(?:(?:and\s+)?then\s+|but\s+)?|\bthen\s+|"
    r"\bbut\s+|\band(?:\s+then)?\s+|\bas\s+well\s+as\s+|\bplus\s+)"
    r"(?:(?:i\s+(?:want|need)\s+you\s+to|please)\s+)?"
    r"(?:can|could|would|will)?\s*(?:you\s+)?(?:please\s+)?"
    r"(?:go\s+ahead|do\s+it|proceed|fix|implement|build|write|generate|render|commit|change|modify|edit|"
    r"update|refactor|migrate|add|create|make|apply|configure|set\s+up|install|"
    r"improve|enhance|strengthen|harden|streamline|clean\s+up|speed\s+up|get|"
    r"persist|save|refine|revise|rework|overhaul|patch|hotfix|eliminate|introduce|"
    r"extract|inline|split|merge|consolidate|deduplicate|reformat|format|autofix|"
    r"scaffold|wire|instrument|memoize|notify|reply|forward|share|message|email|"
    r"enable|disable|publish|deploy|send|post|upload|submit|push|delete|erase|"
    r"destroy|purge|wipe|remove|drop|truncate|stop|restart|kill|terminate|reset|"
    r"revert|rollback|revoke|uninstall)\b",
    re.IGNORECASE,
)

# Split only where a coordinator introduces another explicit action head.  A
# plain conjunction inside an object ("review parser and scheduler") remains a
# single clause, while compound authority ("restart nginx and restart apache")
# is evaluated one action at a time.  Clause-local parsing prevents a bounded
# regex for one action from consuming a neighbouring target or suppressing an
# independent capability family.
_ACTION_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:[;.!?]\s*|,\s*(?:(?:and\s+)?then\s+|but\s+)?|"
    r"\b(?:and(?:\s+then)?|then|also|plus|as\s+well\s+as|but)\b\s+)"
    r"(?=(?:(?:i\s+(?:want|need)\s+you\s+to|please)\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:fix|implement|build|write|generate|render|commit|change|modify|edit|"
    r"update|refactor|migrate|add|create|make|apply|configure|set\s+up|install|"
    r"improve|enhance|strengthen|harden|streamline|clean\s+up|speed\s+up|get|"
    r"persist|save|refine|revise|rework|overhaul|patch|hotfix|eliminate|introduce|"
    r"extract|inline|split|merge|consolidate|deduplicate|reformat|format|autofix|"
    r"scaffold|wire|instrument|memoize|audit|inspect|review|check|analy[sz]e|"
    r"compare|test|validate|verify|publish|deploy|send|notify|reply|forward|share|"
    r"message|email|post|upload|submit|push|purchase|buy|schedule|book|order|pay|"
    r"subscribe|register|provision|connect|link|authorize|delete|erase|destroy|"
    r"purge|wipe|remove|drop|truncate|stop|restart|kill|terminate|reset|revert|"
    r"rollback|revoke|uninstall|unpublish|deactivate)\b)",
    re.IGNORECASE,
)


def _explicit_action_clauses(text: str) -> tuple[str, ...]:
    """Return bounded explicit-action clauses without splitting object lists."""

    clauses = [
        " ".join(item.strip(" ,.;:!?\n\t").split())
        for item in _ACTION_CLAUSE_SPLIT_RE.split(str(text or ""))
    ]
    return tuple(item for item in clauses if item) or (str(text or ""),)
_PENDING_PROPOSAL_RE = re.compile(
    r"\b(?:(?:should|may|can|could)\s+(?:i|we)|"
    r"(?:would\s+you\s+like|do\s+you\s+want)\s+me\s+to)\s+"
    r"(?P<action>[^?!.]+)",
    re.IGNORECASE,
)
_PENDING_CAPABILITY_STATEMENT_RE = re.compile(
    r"(?:^|[.!?]\s*)(?:i|we)\s+(?:can|could|would)\s+"
    r"(?P<action>[^?!.]+)",
    re.IGNORECASE,
)
_NEGATED_ACTION_RE = re.compile(
    r"\bnot\s+(?:actually\s+)?(?:making|creating|producing|performing|doing)\b"
    r"[^?.!\n]{0,120}\buploads\b|"
    r"\b(?:make|create|produce|perform|do)\s+no\b"
    r"[^?.!\n]{0,120}\buploads\b|"
    r"\b(?:do\s+not|don['’]?t|dont|never)\s+"
    r"(?:clean|speed)\b[^?.!\n]{0,60}\bup\b|"
    r"\b(?:do\s+not|don['’]?t|dont|never|not\s+to|no\s+need\s+to)\s+"
    r"(?:actually\s+|ever\s+)?(?:delete|erase|destroy|purge|wipe|remove|drop|"
    r"truncate|stop|restart|kill|terminate|reset|revert|rollback|revoke|"
    r"uninstall|prune|discard|cancel|abort|unpublish|deactivate|publish|deploy|"
    r"send|notify|reply|forward|share|message|email|post|upload|submit|purchase|buy|log\s*in|"
    r"sign\s*in|register|provision|fix|implement|build|write|generate|render|commit|change|modify|edit|"
    r"update|refactor|migrate|add|create|make|apply|configure|install|improve|enhance|"
    r"strengthen|harden|streamline|clean\s+up|speed\s+up|persist|save|refine|revise|"
    r"rework|overhaul|patch|hotfix|eliminate|introduce|extract|inline|split|merge|"
    r"consolidate|deduplicate|reformat|format|autofix|scaffold|wire|instrument|memoize|"
    r"enable|disable)"
    r"(?:s|ed|ing)?\b|"
    r"\bwithout\s+(?:actually\s+|ever\s+)?(?:delete|erase|destroy|purge|wipe|"
    r"remove|dropp?|truncat|stopp?|restart|kill|terminat|reset|revert|rollback|"
    r"revoke|uninstall|publish|deploy|send|post|upload|submit|purchas|buy|"
    r"logg?\s*in|sign\s*in|register|provision|fix|implement|build|writ|generat|render|commit|chang|"
    r"modify|edit|updat|refactor|migrat|add|creat|mak|apply|configur|install|"
    r"improv|enhanc|strengthen|harden|streamlin|clean\s+up|speed\s+up|"
    r"enabl|disabl)(?:e|s|ed|ing)?\b|"
    r"\bwithout\s+(?:making|applying)\s+(?:any\s+)?(?:changes|edits|modifications)\b|"
    r"\bno\s+(?:changes|edits|modifications|deletions|external actions)\b",
    re.IGNORECASE,
)
_NEGATED_EFFECT_CLAUSE_RE = re.compile(
    r"\b(?:do\s+not|don['’]?t|dont|never|no\s+need\s+to)\b"
    r".*?(?=(?:[;!?]|\.(?=\s|$)|,\s*(?:but|then)\b|\s+(?:but|then)\b|$))|"
    r"\bwithout\b.*?"
    r"(?=(?:[;!?]|\.(?=\s|$)|,\s*|$))|"
    r"\b(?:please\s+)?avoid\b.*?"
    r"(?=(?:[;!?]|\.(?=\s|$)|,\s*(?:but|then)\b|\s+(?:but|then)\b|$))|"
    r"\b(?:leave|keep|preserve|retain)\b.*?"
    r"\b(?:unchanged|untouched|intact|as[- ]is)\b.*?"
    r"(?=(?:[;!?]|\.(?=\s|$)|,\s*(?:but|then)\b|\s+(?:but|then)\b|$))|"
    r"\bno\s+(?:code\s+|file\s+|state\s+)?changes?\s+"
    r"(?:to|outside|in)\b.*?"
    r"(?=(?:[;!?]|\.(?=\s|$)|,\s*(?:but|then)\b|\s+(?:but|then)\b|$))",
    re.IGNORECASE,
)

COLLABORATOR_HANDOFF_MARKER = "NEXUS COLLABORATOR HANDOFF"
_SYNTHETIC_CONTINUOUS_REQUEST_PREFIX = "CONTINUOUS MODE:"

# A server-marked collaborator handoff is untrusted project input, not owner
# authority. Keep its entire request contract on a deliberately tiny capability
# surface: bounded local inspection plus owner-facing dialogue. In particular,
# tools classified generically as read-only may still perform outbound requests
# or expose persistent authenticated browser/MCP state.
_COLLABORATOR_HANDOFF_ALLOWED_TOOLS = frozenset(
    {
        "think",
        "say_to_user",
        "get_user_input",
        "task_complete",
    }
)


def _is_collaborator_handoff(value: object) -> bool:
    return str(value or "").lstrip().startswith(COLLABORATOR_HANDOFF_MARKER)


def _effect_text(value: str) -> str:
    """Remove explicit prohibitions before deriving granted authority."""

    masked = _NEGATED_EFFECT_CLAUSE_RE.sub(" ", str(value or ""))
    return _NEGATED_ACTION_RE.sub(" ", masked)


_SUBORDINATE_INSPECTION_FRAME_RE = re.compile(
    r"\b(?:how(?:\s+to)?|steps?(?:\s+needed)?\s+to|ways?\s+to|"
    r"plan\s+to|approach\s+to|whether(?:\s+or\s+not)?(?:\s+to)?|"
    r"what\s+it\s+would\s+take\s+to)\b",
    re.IGNORECASE,
)


def _inspection_later_action_is_independent(
    tail: str, candidate: re.Match[str]
) -> bool:
    """Distinguish a coordinated imperative from an inspected plan/object."""

    matched = candidate.group(0).lstrip().casefold()
    if re.match(
        r"^(?:[;.!?]|,\s*(?:and\s+)?then\b|then\b|and\s+then\b|"
        r"also\s+please\b|but\b)",
        matched,
        re.IGNORECASE,
    ):
        return True
    if not re.match(r"^(?:and\b|as\s+well\s+as\b|plus\b)", matched):
        return False
    inspected_frame = tail[: candidate.start()]
    return not _SUBORDINATE_INSPECTION_FRAME_RE.search(inspected_frame)


def _has_hard_boundary_later_action(text: str) -> bool:
    """Detect a later imperative after an unmistakable clause boundary."""

    return any(
        re.match(
            r"^(?:[;.!?]|,\s*(?:and\s+)?then\b|then\b|and\s+then\b)",
            match.group(0).lstrip(),
            re.IGNORECASE,
        )
        for match in _LATER_EXPLICIT_ACTION_RE.finditer(str(text or ""))
    )


def _classify_action_intent(value: str) -> RequestMode:
    """Classify requested effects without interpreting question modality."""

    source_edit = _SOURCE_EDIT_DELETION_RE.search(value)
    if source_edit:
        # "remove the unused function" is a local edit, but it must not hide a
        # separately headed later clause such as "and restart Aeon".
        modes = [RequestMode.CHANGE_LOCAL]
        for later in _LATER_EXPLICIT_ACTION_RE.finditer(value):
            if later.start() < source_edit.end():
                continue
            later_mode = _classify_action_intent(value[later.start() :])
            if later_mode in {
                RequestMode.CHANGE_LOCAL,
                RequestMode.EXTERNAL_ACTION,
                RequestMode.DESTRUCTIVE,
            }:
                modes.append(later_mode)
        priority = {
            RequestMode.CHANGE_LOCAL: 1,
            RequestMode.EXTERNAL_ACTION: 2,
            RequestMode.DESTRUCTIVE: 3,
        }
        return max(modes, key=lambda item: priority.get(item, 0))
    if _DESTRUCTIVE_REQUEST_RE.search(value):
        return RequestMode.DESTRUCTIVE
    if _AGENT_INSTANCE_CAPABILITY_RE.search(value):
        return RequestMode.EXTERNAL_ACTION
    if (
        _EXTERNAL_REQUEST_RE.search(value)
        or _ACTIONABLE_PLURAL_UPLOAD_RE.search(value)
    ):
        return RequestMode.EXTERNAL_ACTION
    if (
        _CONVERSATIONAL_ARTIFACT_RE.search(value)
        and (
            not _PERSISTENT_DESTINATION_RE.search(value)
            or _INLINE_OUTPUT_RE.search(value)
        )
    ):
        return (
            RequestMode.PLAN
            if re.search(r"\b(?:plan|proposal|recommendation)\b", value, re.IGNORECASE)
            else RequestMode.ANSWER
        )
    head_mode = _clause_head_mode(value)
    if head_mode is not None:
        return head_mode
    if _LOCAL_CHANGE_IMPERATIVE_RE.search(value):
        return RequestMode.CHANGE_LOCAL
    if _CHANGE_REQUEST_RE.search(value):
        return RequestMode.CHANGE_LOCAL
    if _INSPECT_REQUEST_RE.search(value):
        return RequestMode.INSPECT
    if _PLAN_REQUEST_RE.search(value):
        return RequestMode.PLAN
    return RequestMode.ANSWER


def _leading_proposed_action_mode(value: str) -> RequestMode | None:
    """Classify only the leading verb in an explicit pending proposal."""

    action = " ".join(str(value or "").strip().split())
    action = re.sub(r"^(?:please|go ahead and|then|now)\s+", "", action, flags=re.IGNORECASE)
    if re.match(
        r"^(?:create|add|make|start|launch|spawn|do)\s+"
        r"(?:it|this|that|one)\b",
        action,
        re.IGNORECASE,
    ):
        return None
    shared = _clause_head_mode(action)
    if shared is not None:
        return shared
    if re.match(
        r"^(?:delete|erase|destroy|purge|wipe|remove|drop|truncate|stop|restart|"
        r"kill|terminate|reset|revert|rollback|revoke|uninstall|prune|discard|"
        r"cancel|abort|unpublish|deactivate)\b|"
        r"^(?:clear\b[^?!.]*\b(?:cache|queue|database|table|storage|state)|"
        r"shut\s+down\b[^?!.]*\b(?:service|process|daemon|server|host))\b",
        action,
        re.IGNORECASE,
    ):
        return RequestMode.DESTRUCTIVE
    if re.match(
        r"^(?:publish|deploy|send|notify|reply|forward|share|message|email|post|"
        r"upload|submit|push|purchase|buy|register|"
        r"provision|log\s*in|sign\s*in)\b",
        action,
        re.IGNORECASE,
    ):
        return RequestMode.EXTERNAL_ACTION
    if re.match(
        r"^(?:create|add|make|start|launch|spawn)\b[^?!.]*\b"
        r"(?:agent|session|tab|account|repository|repo|pull request|issue)\b",
        action,
        re.IGNORECASE,
    ):
        return RequestMode.EXTERNAL_ACTION
    if re.match(
        r"^(?:fix|implement|build|write|generate|render|commit|change|modify|edit|update|refactor|"
        r"migrate|add|create|make|apply|configure|set\s+up|install|optimi[sz]e|"
        r"integrate|repair|replace|improve|enhance|strengthen|harden|streamline|"
        r"clean\s+up|speed\s+up|rename|move|copy|document|correct|resolve|address|"
        r"upgrade|bump|pin|port|convert|simplify|modernize|ensure|wrap|expose|"
        r"transform|reorganize|restructure|persist|save|refine|revise|rework|"
        r"overhaul|patch|hotfix|eliminate|introduce|extract|inline|split|merge|"
        r"consolidate|deduplicate|reformat|format|autofix|scaffold|wire|instrument|"
        r"memoize|turn|enable|disable|get)\b",
        action,
        re.IGNORECASE,
    ):
        return RequestMode.CHANGE_LOCAL
    return None


def _pending_proposal_mode(question: str) -> RequestMode | None:
    """Return authority named by the exact outstanding confirmation prompt."""

    proposals = list(_PENDING_PROPOSAL_RE.finditer(str(question or "")))
    if not proposals:
        return None
    direct = _leading_proposed_action_mode(proposals[-1].group("action"))
    if direct is not None:
        return direct
    # Pronouns such as "Should I do it?" may refer to the immediately preceding
    # bounded capability statement.  Never rescan arbitrary historical request
    # nouns; only this exact pending question participates.
    prefix = str(question or "")[: proposals[-1].start()]
    statements = list(_PENDING_CAPABILITY_STATEMENT_RE.finditer(prefix))
    if not statements:
        return None
    return _leading_proposed_action_mode(statements[-1].group("action"))


def classify_request_mode(text: str) -> RequestMode:
    """Conservatively classify the authority granted by an exact user request."""

    value = " ".join(str(text or "").strip().split())
    if not value:
        return RequestMode.ANSWER
    if value.startswith(COLLABORATOR_HANDOFF_MARKER):
        # The manager, not the collaborator, adds this provenance marker. The
        # payload can propose work but cannot grant mutation authority.
        return RequestMode.PLAN
    if _STATUS_INSPECTION_RE.search(value):
        return RequestMode.INSPECT
    read_only_directives = list(_EXPLICIT_READ_ONLY_RE.finditer(value))
    if read_only_directives:
        scoped_directives = list(_SCOPED_READ_ONLY_RE.finditer(value))
        all_scoped_after_positive_action = bool(scoped_directives) and all(
            any(
                scoped.start() <= directive.start() < scoped.end()
                and _classify_action_intent(
                    _effect_text(value[: scoped.start()])
                )
                in {
                    RequestMode.CHANGE_LOCAL,
                    RequestMode.EXTERNAL_ACTION,
                    RequestMode.DESTRUCTIVE,
                }
                for scoped in scoped_directives
            )
            for directive in read_only_directives
        )
        # Destructive/external words often appear as the *subject* of a review
        # ("audit delete-user" or "review git push").  An explicit read-only
        # directive is a hard authority ceiling unless a later, separate
        # imperative grants mutation after that directive.
        if not all_scoped_after_positive_action:
            tail = value[read_only_directives[-1].end() :]
            later_action = _LATER_EXPLICIT_ACTION_RE.search(tail)
            if later_action is None:
                return (
                    RequestMode.INSPECT
                    if _INSPECT_REQUEST_RE.search(value)
                    else RequestMode.PLAN
                    if _PLAN_REQUEST_RE.search(value)
                    else RequestMode.INSPECT
                )
            later_mode = _classify_action_intent(
                _effect_text(tail[later_action.start() :])
            )
            if later_mode in {
                RequestMode.CHANGE_LOCAL,
                RequestMode.EXTERNAL_ACTION,
                RequestMode.DESTRUCTIVE,
            }:
                return later_mode
    effect_text = _effect_text(value)
    # Inspection verbs describe how to examine their object; nouns such as
    # "restart" or "delete-user" inside that object are not destructive authority.
    # A later, separately expressed imperative can still grant the named effect.
    leading_inspection = _LEADING_INSPECTION_RE.search(effect_text)
    if leading_inspection:
        tail = effect_text[leading_inspection.end() :]
        later_action = None
        # The inspected object may itself start with an action-shaped noun
        # ("review restart lifecycle", "audit upload handler"). It is not a
        # second imperative unless a clause boundary or coordinator separates
        # it from the inspection head. Keep scanning after that noun so a real
        # later clause ("; then restart foo") is still recognized.
        for candidate in _LATER_EXPLICIT_ACTION_RE.finditer(tail):
            if not _inspection_later_action_is_independent(tail, candidate):
                continue
            later_action = candidate
            break
        if later_action is None:
            return RequestMode.INSPECT
        later_mode = _classify_action_intent(tail[later_action.start() :])
        if later_mode in {
            RequestMode.CHANGE_LOCAL,
            RequestMode.EXTERNAL_ACTION,
            RequestMode.DESTRUCTIVE,
        }:
            return later_mode
    if (
        _DIRECT_POLITE_EXTERNAL_REQUEST_RE.search(effect_text)
        and not re.search(r"\b(?:not|never|without)\b", value, re.IGNORECASE)
    ):
        return RequestMode.EXTERNAL_ACTION
    polite = _DIRECT_POLITE_ACTION_RE.match(effect_text)
    if polite is not None:
        polite_mode = _clause_head_mode(polite.group("action"))
        if polite_mode is not None:
            # A polite compound request grants every explicitly named effect,
            # not merely the first clause. Preserve the precise leading-head
            # classification (which avoids treating nouns such as "delete
            # parser" as commands), then raise the ceiling for later explicit
            # action clauses only.
            modes = [polite_mode]
            action_text = polite.group("action")
            for later in _LATER_EXPLICIT_ACTION_RE.finditer(action_text):
                if later.start() == 0:
                    continue
                if polite_mode == RequestMode.INSPECT:
                    if not _inspection_later_action_is_independent(
                        action_text, later
                    ):
                        continue
                later_mode = _classify_action_intent(action_text[later.start() :])
                if later_mode in {
                    RequestMode.CHANGE_LOCAL,
                    RequestMode.EXTERNAL_ACTION,
                    RequestMode.DESTRUCTIVE,
                }:
                    modes.append(later_mode)
            priority = {
                RequestMode.CHANGE_LOCAL: 1,
                RequestMode.EXTERNAL_ACTION: 2,
                RequestMode.DESTRUCTIVE: 3,
            }
            return max(modes, key=lambda item: priority.get(item, 0))
    # Questions about what/how the agent *could* do are informational. A later
    # explicit confirmation can elevate the existing request contract without
    # pretending the original question itself granted mutation authority.
    if (
        _INFORMATIONAL_REQUEST_RE.search(value)
        and not _EXPLICIT_IMPERATIVE_RE.search(value)
        and not _has_hard_boundary_later_action(effect_text)
    ):
        intent = _classify_action_intent(effect_text)
        if intent in {
            RequestMode.CHANGE_LOCAL,
            RequestMode.EXTERNAL_ACTION,
            RequestMode.DESTRUCTIVE,
        }:
            return RequestMode.PLAN
        return intent
    return _classify_action_intent(effect_text)


def _requested_capability_families(
    text: str, mode: RequestMode
) -> tuple[CapabilityFamily, ...]:
    """Derive exact consequential capability identities from owner text.

    Specific, typed capabilities win over generic action verbs.  For example,
    Agent-owned skill/wiki maintenance is ordinary private agent state and does
    not acquire a destructive capability family. "Restart Aeon" still grants
    the Aeon restart boundary, not an arbitrary service or process kill.
    """

    value = _effect_text(" ".join(str(text or "").strip().split()))
    clauses = _explicit_action_clauses(value)
    families: set[CapabilityFamily] = set()

    # Capability extraction is intentionally independent for compound owner
    # requests. A higher overall mode is an authority ceiling, not permission to
    # discard lower-level obligations such as "fix ... and push ...", nor to
    # discard a push when a later destructive action raises the mode.
    if mode in {RequestMode.EXTERNAL_ACTION, RequestMode.DESTRUCTIVE}:
        external_checks = (
            (_AGENT_INSTANCE_CAPABILITY_RE, CapabilityFamily.AGENT_INSTANCE),
            (_EXTERNAL_EXPERT_CAPABILITY_RE, CapabilityFamily.EXTERNAL_EXPERT),
            (
                _COLLABORATION_PORTAL_CAPABILITY_RE,
                CapabilityFamily.COLLABORATION_PORTAL,
            ),
            (_JOB_ROLE_CAPABILITY_RE, CapabilityFamily.JOB_ROLE),
            (_MCP_CONNECTION_CAPABILITY_RE, CapabilityFamily.MCP_CONNECTION),
        )
        for pattern, family in external_checks:
            if any(pattern.search(clause) for clause in clauses):
                families.add(family)

        github_create = any(
            _GITHUB_CREATE_CAPABILITY_RE.search(clause) for clause in clauses
        )
        if github_create:
            families.add(CapabilityFamily.GITHUB_CREATE)
        if any(_GITHUB_CAPABILITY_RE.search(clause) for clause in clauses) and (
            not github_create
            or re.search(
                r"\b(?:commit|push|update|back\s*up)\b",
                value,
                re.IGNORECASE,
            )
        ):
            families.add(CapabilityFamily.GITHUB)

        actionable_plural_upload = next(
            (
                match
                for clause in clauses
                if (match := _ACTIONABLE_PLURAL_UPLOAD_RE.search(clause))
            ),
            None,
        )
        if (
            any(
                _EXTERNAL_INTERACTION_CAPABILITY_RE.search(clause)
                for clause in clauses
            )
            or actionable_plural_upload
        ) and (
            not families
            or any(
                _STRONG_EXTERNAL_INTERACTION_CAPABILITY_RE.search(clause)
                for clause in clauses
            )
            or actionable_plural_upload
        ):
            families.add(CapabilityFamily.EXTERNAL_INTERACTION)

    if mode == RequestMode.DESTRUCTIVE:
        # Removing one private learned skill or wiki note is recoverable
        # agent-state maintenance, not workspace/resource destruction. Suppress
        # only that generic delete family; still accumulate any independently
        # named process, service, GitHub, or Aeon effects.
        destructive_checks = (
            (_KILL_SUB_AGENT_CAPABILITY_RE, CapabilityFamily.KILL_SUB_AGENT),
            (_KILL_JOB_CAPABILITY_RE, CapabilityFamily.KILL_JOB),
            (_SERVICE_CONTROL_CAPABILITY_RE, CapabilityFamily.SERVICE_CONTROL),
            (_PROCESS_CONTROL_CAPABILITY_RE, CapabilityFamily.PROCESS_CONTROL),
            (_SOURCE_REVERT_CAPABILITY_RE, CapabilityFamily.SOURCE_REVERT),
            (
                _ACCESS_REVOCATION_CAPABILITY_RE,
                CapabilityFamily.ACCESS_REVOCATION,
            ),
        )
        if any(_AEON_RESTART_CAPABILITY_RE.search(clause) for clause in clauses):
            families.add(CapabilityFamily.RESTART_AEON)
        if any(_AEON_REVERT_CAPABILITY_RE.search(clause) for clause in clauses):
            families.add(CapabilityFamily.REVERT_AEON)
        for pattern, family in destructive_checks:
            candidate_clauses = clauses
            if family == CapabilityFamily.SERVICE_CONTROL:
                candidate_clauses = tuple(
                    clause
                    for clause in clauses
                    if not _AEON_RESTART_CAPABILITY_RE.search(clause)
                )
            if any(pattern.search(clause) for clause in candidate_clauses):
                families.add(family)
        resource_deletion = any(
            _DELETE_RESOURCE_CAPABILITY_RE.search(clause)
            and not _SKILL_STATE_DELETION_RE.search(clause)
            and not _SOURCE_EDIT_DELETION_RE.search(clause)
            for clause in clauses
        )
        if resource_deletion:
            families.add(CapabilityFamily.DELETE_RESOURCE)

    if mode == RequestMode.CHANGE_LOCAL and (
        _GITHUB_CAPABILITY_RE.search(value)
        or _LOCAL_GITHUB_COMMIT_CAPABILITY_RE.search(value)
    ):
        # A commit-only request is local but still binds the typed GitHub
        # gateway instead of authorizing an unrelated local capability to stand
        # in for the requested commit.
        families.add(CapabilityFamily.GITHUB)

    # A forced legacy external contract with no recognizable action identity
    # remains usable only for the generic browser/MCP interaction family. It
    # does not regain access to typed GitHub, agent, or control capabilities.
    if mode == RequestMode.EXTERNAL_ACTION and not families:
        families.add(CapabilityFamily.EXTERNAL_INTERACTION)
    return tuple(sorted(families, key=lambda family: family.value))


_READ_ONLY_TOOLS = frozenset(
    {
        "open_file",
        "search_web",
        "huggingface_model_search",
        "huggingface_model_info",
        "huggingface_repo_file",
        "browser_navigate",
        "browser_read",
        "browser_find",
        "browser_extract",
        "browser_switch_tab",
        "system_info",
        "read_skill",
        "list_skill_knowledge",
        "read_skill_knowledge",
        "search_skill_knowledge",
        "list_memories",
        "gather_sub_agents",
        "get_sub_agent_report",
        "get_sub_agent_status",
        "job_output",
        "inspect_tool_result",
        "blackboard_read",
        "analyze_image",
        "list_mcp_credentials",
        "list_provider_credentials",
        "list_payment_addresses",
        "list_mcp_tools",
        "github_repositories",
        "github_status",
        "github_verify_remote",
    }
)
_AGENT_STATE_TOOLS = frozenset(
    {
        "memorize",
        "forget",
        "activate_skill",
        "deactivate_skill",
        "create_skill",
        "delete_skill",
        "delete_skill_knowledge",
        "remember_skill_knowledge",
        "blackboard_post",
        "send_collaborator_handoff",
    }
)
_CONTROL_TOOLS = frozenset(
    {
        "think",
        "say_to_user",
        "get_user_input",
        "task_complete",
        "resume_previous_session",
        "close_file",
        "expand_tool_category",
        "collapse_tool_category",
        "expand_skill_category",
        "collapse_skill_category",
        "expand_skills_category",
        "collapse_skills_category",
    }
)
_LOCAL_MUTATION_TOOLS = frozenset(
    {
        "write_file",
        "str_replace",
        "run_command_async",
        "spawn_sub_agent",
        "integrate_sub_agent_changes",
        "generate_image",
        "edit_image",
        "composite_image",
        "generate_video",
        "verify_self_modification",
        "run_self_benchmark",
        "github_commit",
    }
)
_EXTERNAL_MUTATION_TOOLS = frozenset(
    {
        "browser_interact",
        "consult_external_expert",
        "start_agent_instance",
        "set_job_role",
        "connect_mcp_account",
        "call_mcp_tool",
        "create_collaboration_portal",
        "github_push",
    }
)
_DESTRUCTIVE_TOOLS = frozenset(
    {
        "kill_job",
        "kill_sub_agent",
        "restart_aeon",
        "revert_aeon",
    }
)
_SELF_VERIFYING_TOOLS = frozenset(
    {
        "start_agent_instance",
        "generate_image",
        "edit_image",
        "composite_image",
        "generate_video",
        "run_self_benchmark",
        "create_collaboration_portal",
        "consult_external_expert",
        "set_job_role",
        "kill_job",
        "kill_sub_agent",
        "restart_aeon",
        "revert_aeon",
        # The server receipt itself proves either durable queueing or delivery;
        # a public sibling has no broader validation capability.
        "send_collaborator_handoff",
    }
)


def infer_tool_policy(name: str) -> ToolPolicy:
    value = str(name or "").strip()
    if value in {"run_command", "spawn_sub_agent"}:
        effect = SideEffect.DYNAMIC
    elif value in _READ_ONLY_TOOLS:
        effect = SideEffect.READ_ONLY
    elif value in _AGENT_STATE_TOOLS:
        effect = SideEffect.AGENT_STATE
    elif value in _LOCAL_MUTATION_TOOLS:
        effect = SideEffect.LOCAL_MUTATION
    elif value in _EXTERNAL_MUTATION_TOOLS:
        effect = SideEffect.EXTERNAL_MUTATION
    elif value in _DESTRUCTIVE_TOOLS:
        effect = SideEffect.DESTRUCTIVE
    elif value in _CONTROL_TOOLS:
        effect = SideEffect.CONTROL
    else:
        # An unknown capability is never presumed read-only.
        effect = SideEffect.LOCAL_MUTATION
    consequential = effect in {
        SideEffect.LOCAL_MUTATION,
        SideEffect.EXTERNAL_MUTATION,
        SideEffect.DESTRUCTIVE,
        SideEffect.DYNAMIC,
    }
    return ToolPolicy(
        name=value,
        side_effect=effect,
        observation_boundary=consequential,
        idempotent=effect == SideEffect.READ_ONLY,
        approval_required=effect in {SideEffect.EXTERNAL_MUTATION, SideEffect.DESTRUCTIVE},
        # Typed private-agent-state CRUD receipts are the authoritative
        # postcondition. Requiring a separate workspace validator makes
        # create/delete skill and memory goals impossible to close (and a
        # post-delete read is expected to fail). They remain target/relevance
        # checked by the goal graph and never satisfy workspace mutations.
        self_verifying=(
            value in _SELF_VERIFYING_TOOLS or value in _AGENT_STATE_TOOLS
        ),
        retry_limit=1 if effect == SideEffect.READ_ONLY else 0,
    )


_SHELL_SPLIT_RE = re.compile(r"(?:&&|\|\||[;\n])")
_SHELL_REDIRECT_RE = re.compile(r"(?:^|\s)(?:>|>>|2>|2>>|&>)")
_SHELL_EVALUATION_RE = re.compile(r"`|\$\(|[<>]\(")
_DESTRUCTIVE_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:rm|rmdir|shred|unlink|kill|pkill|killall|truncate)\b|"
    r"\bgit\s+(?:reset|clean)\b|\bdocker\s+(?:rm|rmi|kill|stop|system\s+prune)\b|"
    r"\bsystemctl(?:\s+--user)?\s+(?:stop|disable|restart)\b",
    re.IGNORECASE,
)
_EXTERNAL_COMMAND_RE = re.compile(
    r"\bgit\s+push\b|\bgh\s+(?:repo\s+create|pr\s+create|issue\s+create|release\s+create)\b|"
    r"\b(?:curl|http)\b[^\n]*(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--data|-d\s)|"
    r"\b(?:scp|sftp)\b|\brsync\b[^\n]*:[^\s]",
    re.IGNORECASE,
)
_LOCAL_MUTATION_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:touch|mkdir|mv|cp|chmod|chown|ln|tee|install)\b|"
    r"\b(?:pip|uv|npm|pnpm|yarn|apt|dnf)\s+(?:install|add|remove|uninstall)\b|"
    r"\bgit\s+(?:add|commit|checkout|switch|merge|rebase|restore|worktree\s+add)\b",
    re.IGNORECASE,
)
_READ_ONLY_COMMANDS = frozenset(
    {
        "pwd",
        "ls",
        "rg",
        "grep",
        "sed",
        "head",
        "tail",
        "wc",
        "stat",
        "file",
        "findmnt",
        "df",
        "du",
        "free",
        "ps",
        "pgrep",
        "hostname",
        "uname",
        "readlink",
        "realpath",
        "git",
    }
)


def _sed_invocation_is_read_only(arguments: Sequence[str]) -> bool:
    """Accept only a deliberately small, output-only sed grammar.

    GNU sed's ``e`` command and substitution flag execute a shell, while ``r``
    and ``w`` cross the read boundary.  Treating arbitrary sed programs as
    readers would therefore let an INSPECT request execute code.  Richer
    inspection remains available through rg/open_file.
    """

    remaining = list(arguments)
    while remaining and remaining[0] in {"-n", "--quiet", "--silent"}:
        remaining.pop(0)
    if not remaining:
        return False
    program = remaining[0]
    return bool(
        re.fullmatch(
            r"\s*(?:(?:\d+|\$)(?:\s*,\s*(?:\d+|\$))?)?\s*(?:p|q|=)\s*",
            program,
        )
    )


def _git_subcommand(words: Sequence[str]) -> tuple[str, list[str]]:
    """Return a conservatively parsed git subcommand and its arguments."""

    arguments = list(words[1:])
    index = 0
    options_with_values = {"-C", "--git-dir", "--work-tree", "--namespace"}
    while index < len(arguments):
        word = arguments[index]
        if word == "-c":
            if (
                index + 1 < len(arguments)
                and arguments[index + 1].casefold() == "core.fsmonitor=false"
            ):
                index += 2
                continue
            return "", []
        if word == "--config-env" or word.startswith("--config-env="):
            return "", []
        if word in options_with_values:
            index += 2
            continue
        if any(
            word.startswith(prefix)
            for prefix in ("--git-dir=", "--work-tree=", "--namespace=")
        ):
            index += 1
            continue
        if word in {"-p", "--paginate", "--exec-path", "--html-path", "--man-path", "--info-path"} or any(
            word.startswith(prefix)
            for prefix in (
                "--exec-path=",
                "--html-path=",
                "--man-path=",
                "--info-path=",
            )
        ):
            return "", []
        if word.startswith("-"):
            index += 1
            continue
        return word, arguments[index + 1 :]
    return "status", []


def _git_invocation_is_read_only(
    words: Sequence[str], environment_assignments: Sequence[str]
) -> bool:
    dangerous_environment = {
        "HOME",
        "XDG_CONFIG_HOME",
        "GIT_EXTERNAL_DIFF",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_COUNT",
        "GIT_PAGER",
        "PAGER",
    }
    for assignment in environment_assignments:
        key = assignment.split("=", 1)[0]
        if key in dangerous_environment or key.startswith("GIT_"):
            return False
    subcommand, arguments = _git_subcommand(words)
    if not subcommand:
        return False
    if subcommand not in {"status", "diff", "log", "show", "rev-parse", "ls-files"}:
        return False
    if any(
        argument in {"--ext-diff", "--textconv", "--paginate"}
        or argument.startswith("--ext-diff=")
        or argument.startswith("--textconv=")
        for argument in arguments
    ):
        return False
    safe_fsmonitor = any(
        words[index] == "-c"
        and index + 1 < len(words)
        and words[index + 1].casefold() == "core.fsmonitor=false"
        for index in range(len(words))
    )
    if subcommand == "status" and not safe_fsmonitor:
        return False
    # diff/show/log may invoke repository-selected diff/textconv drivers or a
    # pager. Admit only an invocation which explicitly disables both helper
    # classes and fsmonitor; plain forms remain outside a read-only contract.
    if subcommand in {"diff", "show", "log"} and not (
        safe_fsmonitor
        and "--no-ext-diff" in arguments
        and "--no-textconv" in arguments
        and "--no-pager" in words
    ):
        return False
    return True


def classify_command_effect(command: str) -> SideEffect:
    value = str(command or "").strip()
    if not value:
        return SideEffect.DYNAMIC
    if _DESTRUCTIVE_COMMAND_RE.search(value):
        return SideEffect.DESTRUCTIVE
    if _EXTERNAL_COMMAND_RE.search(value):
        return SideEffect.EXTERNAL_MUTATION
    # Substitution can hide arbitrary effects inside an otherwise read-only
    # command, and output redirection turns readers into writers. Fail closed
    # instead of attempting to duplicate a shell parser.
    if (
        _SHELL_EVALUATION_RE.search(value)
        or ">" in value
        or _SHELL_REDIRECT_RE.search(value)
        or _LOCAL_MUTATION_COMMAND_RE.search(value)
    ):
        return SideEffect.LOCAL_MUTATION

    segments = [part.strip() for part in _SHELL_SPLIT_RE.split(value) if part.strip()]
    for segment in segments:
        # A pipe is allowed only when every stage is a known read operation.
        for stage in [piece.strip() for piece in segment.split("|") if piece.strip()]:
            try:
                words = shlex.split(stage)
            except ValueError:
                return SideEffect.DYNAMIC
            environment_assignments: list[str] = []
            while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
                environment_assignments.append(words.pop(0))
            if not words:
                continue
            command_name = words[0].rsplit("/", 1)[-1]
            if command_name not in _READ_ONLY_COMMANDS:
                return SideEffect.LOCAL_MUTATION
            if command_name == "sed" and not _sed_invocation_is_read_only(words[1:]):
                return SideEffect.LOCAL_MUTATION
            if command_name == "git" and not _git_invocation_is_read_only(
                words, environment_assignments
            ):
                return SideEffect.LOCAL_MUTATION
    return SideEffect.READ_ONLY


def effective_tool_effect(policy: ToolPolicy, parameters: Mapping[str, Any] | None) -> SideEffect:
    params = parameters if isinstance(parameters, Mapping) else {}
    if policy.name == "spawn_sub_agent":
        if bool(params.get("read_only", True)):
            # The child may write only its private run receipts; it cannot mutate
            # the target workspace. This is agent state, not a project edit.
            return SideEffect.AGENT_STATE
        return SideEffect.LOCAL_MUTATION
    if policy.side_effect != SideEffect.DYNAMIC:
        return policy.side_effect
    return classify_command_effect(str(params.get("command", "")))


_TOOL_CAPABILITY_FAMILIES: dict[str, CapabilityFamily] = {
    "github_commit": CapabilityFamily.GITHUB,
    "github_push": CapabilityFamily.GITHUB,
    "github_repositories": CapabilityFamily.GITHUB,
    "github_status": CapabilityFamily.GITHUB,
    "github_verify_remote": CapabilityFamily.GITHUB,
    "job_output": CapabilityFamily.KILL_JOB,
    "get_sub_agent_report": CapabilityFamily.KILL_SUB_AGENT,
    "get_sub_agent_status": CapabilityFamily.KILL_SUB_AGENT,
    "browser_interact": CapabilityFamily.EXTERNAL_INTERACTION,
    "call_mcp_tool": CapabilityFamily.EXTERNAL_INTERACTION,
    "start_agent_instance": CapabilityFamily.AGENT_INSTANCE,
    "consult_external_expert": CapabilityFamily.EXTERNAL_EXPERT,
    "create_collaboration_portal": CapabilityFamily.COLLABORATION_PORTAL,
    "set_job_role": CapabilityFamily.JOB_ROLE,
    "connect_mcp_account": CapabilityFamily.MCP_CONNECTION,
    "kill_job": CapabilityFamily.KILL_JOB,
    "kill_sub_agent": CapabilityFamily.KILL_SUB_AGENT,
    "restart_aeon": CapabilityFamily.RESTART_AEON,
    "revert_aeon": CapabilityFamily.REVERT_AEON,
}

_TARGET_REQUIRED_FAMILIES = frozenset(
    {
        CapabilityFamily.GITHUB,
        CapabilityFamily.EXTERNAL_INTERACTION,
        CapabilityFamily.AGENT_INSTANCE,
        CapabilityFamily.KILL_JOB,
        CapabilityFamily.KILL_SUB_AGENT,
        CapabilityFamily.RESTART_AEON,
        CapabilityFamily.REVERT_AEON,
        CapabilityFamily.SERVICE_CONTROL,
        CapabilityFamily.PROCESS_CONTROL,
        CapabilityFamily.DELETE_RESOURCE,
        CapabilityFamily.SOURCE_REVERT,
        CapabilityFamily.ACCESS_REVOCATION,
        CapabilityFamily.JOB_ROLE,
        CapabilityFamily.COLLABORATION_PORTAL,
        CapabilityFamily.MCP_CONNECTION,
    }
)
_MAX_CAPABILITY_TARGETS = 32


_EXTERNAL_OPERATION_ALIASES = {
    "buy": "purchase",
    "email": "send",
    "forward": "send",
    "invite": "send",
    "make live": "publish",
    "message": "send",
    "notify": "send",
    "ping": "send",
    "reply": "send",
}
_LOCAL_ARTIFACT_SUFFIX_RE = re.compile(
    r"(?:^|/)[A-Za-z0-9_.-]+\."
    r"(?:py|pyi|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|cs|rb|php|swift|kt|"
    r"scala|sh|bash|zsh|fish|ps1|sql|html|css|scss|vue|svelte|md|rst|txt|json|"
    r"ya?ml|toml|ini|cfg|conf|xml|proto|graphql|lock)$",
    re.IGNORECASE,
)


def _external_operation_target(value: Any) -> str:
    """Return one canonical external operation scope, never a broad effect."""

    text = " ".join(str(value or "").strip().split()).casefold()
    text = re.sub(r"[_-]+", " ", text)
    if not text:
        return ""
    match = re.search(
        r"\b(make\s+live|publish|deploy|send|email|message|notify|reply|forward|"
        r"share|post|upload|submit|purchase|buy|schedule|book|order|pay|subscribe|"
        r"log\s*in|sign\s*in|register|invite|ping|dm)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    operation = " ".join(match.group(1).casefold().split())
    if operation in {"log in", "login", "sign in", "signin"}:
        operation = "login"
    elif operation == "dm":
        operation = "send"
    operation = _EXTERNAL_OPERATION_ALIASES.get(operation, operation)
    return f"operation:{operation}"


def _external_scope_target(value: Any) -> str:
    """Normalize an explicit recipient/platform/site/account scope."""

    target = " ".join(str(value or "").strip().strip("`'\"").split()).rstrip(
        ".,;:!?"
    )
    if not target:
        return ""
    lowered = target.casefold()
    if re.fullmatch(r"(?:recipient|platform|site|account):[^\s:][^\n]{0,240}", lowered):
        return lowered
    if re.fullmatch(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", target
    ):
        return f"recipient:{lowered}"
    if _LOCAL_ARTIFACT_SUFFIX_RE.fullmatch(target):
        return ""
    parsed = urlsplit(target if "://" in target else f"https://{target}")
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if hostname and (
        "://" in target
        or re.fullmatch(r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}", target)
    ):
        return f"site:{hostname}"
    if lowered in {
        "x", "twitter", "linkedin", "facebook", "instagram", "github",
        "gitlab", "slack", "discord",
    }:
        return f"platform:{lowered}"
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,63}", target):
        return f"recipient:{lowered}"
    return ""


def _capability_binding_complete(
    family: CapabilityFamily, targets: Sequence[str] | None
) -> bool:
    values = [str(item or "") for item in (targets or ()) if str(item or "")]
    if not values:
        return False
    if family == CapabilityFamily.EXTERNAL_INTERACTION:
        return bool(
            any(item.startswith("operation:") for item in values)
            and any(
                item.startswith(("recipient:", "platform:", "site:", "account:"))
                for item in values
            )
        )
    return True


def _command_capability_family(command: str) -> CapabilityFamily | None:
    value = str(command or "")
    if re.search(r"\bgit\s+(?:add|commit|push)\b|\bgh\s+", value, re.IGNORECASE):
        return CapabilityFamily.GITHUB
    if re.search(r"\bsystemctl(?:\s+--user)?\s+(?:stop|disable|restart)\b", value, re.IGNORECASE):
        return CapabilityFamily.SERVICE_CONTROL
    if re.search(r"(?:^|\s)(?:kill|pkill|killall)\b", value, re.IGNORECASE):
        return CapabilityFamily.PROCESS_CONTROL
    if re.search(r"\bgit\s+(?:reset|clean)\b", value, re.IGNORECASE):
        return CapabilityFamily.SOURCE_REVERT
    if re.search(
        r"(?:^|\s)(?:rm|rmdir|shred|unlink|truncate)\b|"
        r"\bdocker\s+(?:rm|rmi|system\s+prune)\b",
        value,
        re.IGNORECASE,
    ):
        return CapabilityFamily.DELETE_RESOURCE
    if _EXTERNAL_COMMAND_RE.search(value):
        return CapabilityFamily.EXTERNAL_INTERACTION
    return None


def _tool_capability_family(
    policy: ToolPolicy, parameters: Mapping[str, Any] | None = None
) -> CapabilityFamily | None:
    params = parameters if isinstance(parameters, Mapping) else {}
    if policy.name == "run_command":
        return _command_capability_family(str(params.get("command", "")))
    if policy.name == "call_mcp_tool" and re.search(
        r"\b(?:revoke|remove|delete|disable).*(?:access|credential|token|permission|member)|"
        r"(?:access|credential|token|permission|member).*(?:revoke|remove|delete|disable)\b",
        str(params.get("tool_name", "")),
        re.IGNORECASE,
    ):
        return CapabilityFamily.ACCESS_REVOCATION
    if policy.name == "call_mcp_tool" and re.search(
        r"\b(?:github|gitlab)\b.*\b(?:create|open|new)\b.*"
        r"\b(?:repository|repo|issue|pull[_ -]?request|pr)\b|"
        r"\b(?:create|open|new)\b.*\b(?:repository|repo|issue|pull[_ -]?request|pr)\b",
        str((params.get("tool_name") or params.get("name") or "")),
        re.IGNORECASE,
    ):
        return CapabilityFamily.GITHUB_CREATE
    return _TOOL_CAPABILITY_FAMILIES.get(policy.name)


def _command_has_explicit_mutation(command: str) -> bool:
    """Distinguish a test runner's incidental writes from an edit+test shell."""

    value = str(command or "")
    if (
        _SHELL_EVALUATION_RE.search(value)
        or ">" in value
        or _DESTRUCTIVE_COMMAND_RE.search(value)
        or _EXTERNAL_COMMAND_RE.search(value)
        or _LOCAL_MUTATION_COMMAND_RE.search(value)
    ):
        return True
    try:
        stages = [
            shlex.split(stage.strip())
            for segment in _SHELL_SPLIT_RE.split(value)
            for stage in segment.split("|")
            if stage.strip()
        ]
    except ValueError:
        return True
    return any(
        words
        and words[0].rsplit("/", 1)[-1] == "sed"
        and not _sed_invocation_is_read_only(words[1:])
        for words in stages
    )


_FAIL_RE = re.compile(
    r"(?:^|\n)\s*(?:error|an error occurred|tool (?:execution|parameter) error|"
    r"command failed|command timed out|browser (?:action |capture )?failed|"
    r"browser error|verification failed|failed:)|"
    r"\btraceback \(most recent call last\)",
    re.IGNORECASE | re.MULTILINE,
)
_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:command|request|operation|action)\s+)?"
    r"(?:blocked|refused|denied)\b|"
    r"\b(?:permission|access)\s+denied\b|"
    r"\b(?:guard|harness|policy|fleet compute(?: policy)?)\s+"
    r"(?:blocked|refused|denied)\b|"
    r"\bnot\s+authorized\b",
    re.IGNORECASE | re.MULTILINE,
)
_PENDING_RE = re.compile(
    r"\b(?:waiting_for_compute|awaiting user input|still running|status:\s*running|"
    r"sub-agent spawned|job started|pending)\b",
    re.IGNORECASE,
)
_NO_CHANGE_RE = re.compile(
    r"\b(?:no[ -]?op|no change|no changes|changed nothing|content is identical|"
    r"was not open|was not expanded|nothing to kill)\b",
    re.IGNORECASE,
)
_SUCCESS_CHANGE_RE = re.compile(
    r"\b(?:successfully|created|updated|overwrote|wrote|applied|saved|posted|"
    r"registered|started|spawned|generated|deleted|erased|removed|terminated)\b",
    re.IGNORECASE,
)
_ANSI_ESCAPE_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))"
)


def _bounded_summary(value: Any, limit: int = 1600) -> str:
    text = str(value if value is not None else "").strip()
    if len(text) <= limit:
        return text
    head = max(1, limit // 3)
    tail = limit - head - 32
    return text[:head] + "\n...[result truncated]...\n" + text[-tail:]


def _receipt_control_line(value: str) -> str:
    """Return the first harness control line with terminal decoration removed."""

    first_line = str(value or "").lstrip().splitlines()[0:1]
    if not first_line:
        return ""
    return _ANSI_ESCAPE_RE.sub("", first_line[0]).strip()


def _run_command_envelope_status(value: str) -> ToolStatus | None:
    """Read only the trusted control line, never captured command output."""

    control_line = _receipt_control_line(value)
    if not control_line:
        return None
    line = control_line.upper()
    prefixes = (
        ("COMMAND SUCCESS", ToolStatus.OK),
        ("COMMAND FAILED", ToolStatus.FAILED),
        ("COMMAND TIMED OUT", ToolStatus.FAILED),
        ("COMMAND BLOCKED", ToolStatus.BLOCKED),
        ("COMMAND REFUSED", ToolStatus.BLOCKED),
    )
    for prefix, status in prefixes:
        if line == prefix or line.startswith((prefix + " ", prefix + ":", prefix + "(")):
            return status
    return None


def _async_control_envelope_status(tool_name: str, value: str) -> ToolStatus | None:
    """Parse only exact first-line job/sub-agent envelopes.

    Human-readable bodies may contain arbitrary status words, so only the
    harness-owned prefix participates. Unknown statuses in a recognized control
    envelope fail closed instead of becoming success evidence.
    """

    line = _receipt_control_line(value)
    if not line:
        return None
    name = str(tool_name or "").strip().lower()
    status = ""
    if name == "job_output":
        match = re.match(
            r"^Job\s+[A-Za-z0-9_.-]{1,64}\s+\[([^\]]{1,80})\](?:\s|$)",
            line,
            re.IGNORECASE,
        )
        if match is None:
            return None
        status = match.group(1).strip().upper()
    elif name in {"get_sub_agent_report", "get_sub_agent_status"}:
        match = re.match(
            r"^(?:Sub-)?Agent\s+[A-Za-z0-9_.-]{1,64}\s+Status:\s*"
            r"([A-Za-z][A-Za-z _-]{0,79})(?:\s|$)",
            line,
            re.IGNORECASE,
        )
        if match is None:
            return None
        status = match.group(1).strip().upper().replace("_", " ")
    else:
        return None

    if status in {"RUNNING", "STARTING", "PENDING", "QUEUED"}:
        return ToolStatus.PENDING
    if status in {"COMPLETED", "COMPLETE", "SUCCEEDED", "SUCCESS"}:
        return ToolStatus.OK
    if status.startswith(("FAILED", "TIMED OUT", "TIMEOUT", "KILLED", "CANCELLED")):
        return ToolStatus.FAILED
    if status.startswith(("BLOCKED", "REFUSED")):
        return ToolStatus.BLOCKED
    return ToolStatus.BLOCKED


def normalize_tool_result(
    tool_name: str,
    raw: Any,
    *,
    policy: ToolPolicy | None = None,
    parameters: Mapping[str, Any] | None = None,
    call_id: str = "",
) -> ToolResult:
    if isinstance(raw, ToolResult):
        if not raw.call_id:
            raw.call_id = call_id
        if not raw.artifacts:
            raw.artifacts = _tool_targets(tool_name, parameters)
        return raw
    policy = policy or infer_tool_policy(tool_name)
    effect = effective_tool_effect(policy, parameters)
    text = _bounded_summary(raw)
    stripped = text.strip()
    control_line = _receipt_control_line(stripped)
    envelope_status = (
        _run_command_envelope_status(stripped)
        if str(tool_name or "").strip().lower() == "run_command"
        else _async_control_envelope_status(tool_name, stripped)
    )
    if envelope_status is not None:
        status = envelope_status
    elif _BLOCK_RE.match(control_line):
        status = ToolStatus.BLOCKED
    elif _FAIL_RE.match(control_line):
        status = ToolStatus.FAILED
    elif (
        effect
        in {
            SideEffect.AGENT_STATE,
            SideEffect.LOCAL_MUTATION,
            SideEffect.EXTERNAL_MUTATION,
            SideEffect.DESTRUCTIVE,
        }
        and _NO_CHANGE_RE.match(control_line)
    ):
        status = ToolStatus.NO_CHANGE
    elif (
        str(tool_name or "").strip().lower()
        in {
            "spawn_sub_agent",
            "get_sub_agent_report",
            "get_sub_agent_status",
            "start_background_job",
            "job_output",
        }
        and _PENDING_RE.match(control_line)
    ):
        status = ToolStatus.PENDING
    else:
        status = ToolStatus.OK

    # A successful receipt from a mutating capability is stronger evidence than
    # the wording the tool happened to use. Tools report an explicit NO_CHANGE
    # status when they can prove a no-op; otherwise an OK mutating invocation is
    # recorded as a change and must still be followed by validation.
    changed = bool(
        status == ToolStatus.OK
        and effect in {
            SideEffect.AGENT_STATE,
            SideEffect.LOCAL_MUTATION,
            SideEffect.EXTERNAL_MUTATION,
            SideEffect.DESTRUCTIVE,
        }
    )
    error_code = ""
    if status == ToolStatus.FAILED:
        error_code = "tool_failed"
    elif status == ToolStatus.BLOCKED:
        error_code = "tool_blocked"
    elif status == ToolStatus.NO_CHANGE:
        error_code = "no_change"
    elif status == ToolStatus.PENDING:
        error_code = "pending"
    elif status == ToolStatus.SKIPPED:
        error_code = "skipped"
    evidence = [stripped[:500]] if status == ToolStatus.OK and stripped else []
    return ToolResult(
        tool_name=str(tool_name),
        status=status,
        changed=changed,
        summary=text or "(no output)",
        evidence=evidence,
        artifacts=_tool_targets(tool_name, parameters),
        error_code=error_code,
        retryable=status == ToolStatus.FAILED and policy.retry_limit > 0,
        side_effect=effect,
        call_id=call_id,
        raw=raw,
    )


_SUCCESS_CLAIM_RE = re.compile(
    r"\b(?:i|we)\s+(?:(?:have|'ve)\s+)?(?:successfully\s+)?"
    r"(?:fixed|changed|created|built|implemented|updated|deployed|published|sent|"
    r"uploaded|submitted|deleted|removed|installed|configured|completed|finished)\b|"
    r"\b(?:has|have)\s+been\s+(?:successfully\s+)?"
    r"(?:fixed|changed|created|implemented|updated|deployed|published|sent|deleted)\b|"
    r"\b(?:task|work|change|deployment|setup|it|this|that|everything)\s+is\s+"
    r"(?:complete|done|finished|fixed|working|ready)\b|"
    r"^\s*(?:done|fixed|completed|finished|implemented|updated|deployed|published)\s*[.!]?\s*$",
    re.IGNORECASE,
)
# A terminal blocker must be an explicit current disposition, not the mere word
# "failed" or "unavailable" in explanatory/history prose. Receipt support is
# checked separately by RequestContract.
_TRUTHFUL_BLOCK_RE = re.compile(
    r"^\s*(?:i|we|aeon)\s+"
    r"(?:(?:am|are|is|remain|remains)\s+)?(?:currently\s+)?"
    r"(?:blocked\b|"
    r"(?:cannot|can['’]?t|could\s+not|couldn['’]?t|(?:am\s+|are\s+)?unable\s+to)\s+"
    r"(?:complete|finish|continue|satisfy|perform|execute|access|inspect|change|"
    r"write|verify|validate|publish|deploy|send|delete|remove|build|implement)\b|"
    r"did\s+not\s+complete\b)|"
    r"^\s*(?:the\s+)?(?:request|task|work)\s+"
    r"(?:is|remains)\s+(?:currently\s+)?(?:blocked|incomplete)\b",
    re.IGNORECASE,
)
_TERMINAL_BLOCKER_CODES = frozenset(
    {"owner_authority_required", "verified_invariant_blocker"}
)
_ALREADY_SATISFIED_RE = re.compile(
    r"\b(?:already (?:is|was|has|exists|matches|works|satisfies)|"
    r"(?:is|was) already|no (?:code |file |state )?(?:change|edit|action)s? "
    r"(?:is |are |was |were )?(?:needed|required|necessary)|"
    r"currently (?:matches|works|satisfies|has the requested))\b",
    re.IGNORECASE,
)


def claims_success(message: Any) -> bool:
    text = str(message or "")
    if _TRUTHFUL_BLOCK_RE.search(text):
        # Evaluate clauses independently so a truthful first clause cannot mask a
        # contradictory success assertion after "but".
        clauses = re.split(r"(?:[;\n]+|\bbut\b|\bhowever\b)", text, flags=re.IGNORECASE)
        return any(_SUCCESS_CLAIM_RE.search(clause) and not _TRUTHFUL_BLOCK_RE.search(clause) for clause in clauses)
    return bool(_SUCCESS_CLAIM_RE.search(text))


def claims_already_satisfied(message: Any) -> bool:
    return bool(_ALREADY_SATISFIED_RE.search(str(message or "")))


def incomplete_final_response(message: Any) -> bool:
    """Reject a terminal lead-in whose promised body is visibly absent.

    A final colon is not a stylistic nit: in a terminal turn it means the model
    introduced content that never arrived. Structured decoding can still yield
    perfectly valid JSON containing such an incomplete message, so this check
    belongs in the semantic completion gate rather than the JSON parser.
    """

    return str(message or "").rstrip().endswith(":")


_DIRECT_VALIDATION_COMMANDS = frozenset(
    {
        "pytest",
        "py.test",
        "ruff",
        "mypy",
        "pyright",
        "tox",
        "nox",
        "ctest",
    }
)
_PYTHON_VALIDATION_MODULES = frozenset(
    {"pytest", "unittest", "compileall", "ruff", "mypy", "pyright"}
)
_VALIDATION_SCRIPT_RE = re.compile(
    r"^(?:test(?:s|[-_].*)?|check(?:s|[-_].*)?|lint(?:[-_].*)?|"
    r"verify(?:[-_].*)?|validate(?:[-_].*)?|smoke(?:[-_].*)?)"
    r"(?:\.(?:sh|py|js|ts|rb|pl))?$",
    re.IGNORECASE,
)


def _is_validation_command(command: str) -> bool:
    """Recognize an invoked validator, not a word appearing in shell data."""

    value = str(command or "").strip()
    if not value or _SHELL_EVALUATION_RE.search(value):
        return False
    try:
        stages = [
            shlex.split(stage.strip())
            for segment in _SHELL_SPLIT_RE.split(value)
            for stage in segment.split("|")
            if stage.strip()
        ]
    except ValueError:
        return False
    for words in stages:
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            words.pop(0)
        if not words:
            continue
        executable = words[0].rsplit("/", 1)[-1].casefold()
        arguments = words[1:]
        if executable in _DIRECT_VALIDATION_COMMANDS:
            return True
        if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable):
            if "-m" in arguments:
                index = arguments.index("-m")
                if (
                    index + 1 < len(arguments)
                    and arguments[index + 1].casefold()
                    in _PYTHON_VALIDATION_MODULES
                ):
                    return True
            script = next((item for item in arguments if not item.startswith("-")), "")
            if script and _VALIDATION_SCRIPT_RE.fullmatch(
                script.rsplit("/", 1)[-1]
            ):
                return True
        if executable in {"npm", "pnpm", "yarn", "bun"}:
            normalized = [item.casefold() for item in arguments]
            if normalized[:1] == ["test"] or normalized[:2] == ["run", "test"]:
                return True
        if executable == "cargo" and arguments:
            if arguments[0].casefold() in {"test", "check", "build"}:
                return True
        if executable == "go" and arguments and arguments[0].casefold() == "test":
            return True
        if executable in {"make", "gmake"}:
            targets = {item.casefold() for item in arguments if not item.startswith("-")}
            if targets & {"test", "tests", "check", "lint", "build", "verify"}:
                return True
        if executable == "dotnet" and arguments:
            if arguments[0].casefold() in {"test", "build"}:
                return True
        candidate = words[0].rsplit("/", 1)[-1]
        if "/" in words[0] and _VALIDATION_SCRIPT_RE.fullmatch(candidate):
            return True
        if executable in {"bash", "sh"} and arguments:
            script = next((item for item in arguments if not item.startswith("-")), "")
            if "/" in script and _VALIDATION_SCRIPT_RE.fullmatch(
                script.rsplit("/", 1)[-1]
            ):
                return True
    return False


def _is_git_observation_command(command: str) -> bool:
    """Identify a plain Git observer that is not acceptance validation.

    Plain ``git status``/``diff`` remains conservatively LOCAL_MUTATION for tool
    authorization because repository helpers may write incidental metadata. That
    incidental write must not be promoted into evidence that the requested code
    change occurred, nor create an unscoped deliverable obligation.
    """

    try:
        words = shlex.split(str(command or "").strip())
    except ValueError:
        return False
    while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
        words.pop(0)
    if not words or words[0].rsplit("/", 1)[-1] != "git":
        return False
    subcommand, _arguments = _git_subcommand(words)
    return subcommand in {"status", "diff", "log", "show", "rev-parse", "ls-files"}


def is_validation_call(tool_name: str, parameters: Mapping[str, Any] | None) -> bool:
    name = str(tool_name or "")
    params = parameters if isinstance(parameters, Mapping) else {}
    if name in {
        "browser_read",
        "open_file",
        "github_status",
        "github_verify_remote",
        "job_output",
    }:
        return True
    if name != "run_command":
        return False
    return _is_validation_command(str(params.get("command", "")))


_ALLOWED_EFFECTS: dict[RequestMode, frozenset[SideEffect]] = {
    RequestMode.ANSWER: frozenset(
        {SideEffect.READ_ONLY, SideEffect.AGENT_STATE, SideEffect.CONTROL}
    ),
    RequestMode.INSPECT: frozenset(
        {SideEffect.READ_ONLY, SideEffect.AGENT_STATE, SideEffect.CONTROL}
    ),
    RequestMode.PLAN: frozenset(
        {SideEffect.READ_ONLY, SideEffect.AGENT_STATE, SideEffect.CONTROL}
    ),
    RequestMode.CHANGE_LOCAL: frozenset(
        {SideEffect.READ_ONLY, SideEffect.AGENT_STATE, SideEffect.LOCAL_MUTATION, SideEffect.CONTROL}
    ),
    RequestMode.EXTERNAL_ACTION: frozenset(
        {
            SideEffect.READ_ONLY,
            SideEffect.AGENT_STATE,
            SideEffect.LOCAL_MUTATION,
            SideEffect.EXTERNAL_MUTATION,
            SideEffect.CONTROL,
        }
    ),
    RequestMode.DESTRUCTIVE: frozenset(set(SideEffect)),
}


_FILE_TARGET_TOOLS = frozenset({"open_file", "write_file", "str_replace"})
_GITHUB_TARGET_PREFIXES = ("github-local:", "github-remote:")


def _github_receipt_document(result: ToolResult | None) -> Mapping[str, Any]:
    if result is not None and isinstance(result.raw, Mapping):
        return result.raw
    return {}


def _github_repository_identity(
    parameters: Mapping[str, Any], document: Mapping[str, Any]
) -> str:
    candidate: Any = parameters.get("repository")
    receipt_repository = document.get("repository")
    if isinstance(receipt_repository, Mapping):
        candidate = receipt_repository.get("path", candidate)
    elif isinstance(receipt_repository, str):
        candidate = receipt_repository
    value = str(candidate or "").strip()
    if (
        not value
        or len(value) > 900
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        return ""
    return os.path.normpath(value)


def _github_head(document: Mapping[str, Any], *, nested: bool = False) -> str:
    source: Mapping[str, Any] = document
    if nested and isinstance(document.get("repository"), Mapping):
        source = document["repository"]
    value = str(source.get("head") or "").strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else ""


def _github_remote_name(
    parameters: Mapping[str, Any], document: Mapping[str, Any]
) -> str:
    candidate: Any = parameters.get("remote_name", "origin")
    if isinstance(document.get("remote"), Mapping):
        candidate = document["remote"].get("name", candidate)
    value = str(candidate or "origin").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", value):
        return ""
    return value


def _github_target(prefix: str, *parts: str) -> str:
    return prefix + json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))


def _parse_github_target(value: str) -> tuple[str, tuple[str, ...]] | None:
    for prefix in _GITHUB_TARGET_PREFIXES:
        if not value.startswith(prefix):
            continue
        try:
            parts = json.loads(value[len(prefix) :])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(parts, list) or not all(isinstance(item, str) for item in parts):
            return None
        return prefix, tuple(parts)
    return None


def _github_local_target_repository(value: str) -> str:
    parsed = _parse_github_target(value)
    if (
        parsed is None
        or parsed[0] != "github-local:"
        or len(parsed[1]) != 2
        or not re.fullmatch(r"[0-9a-f]{40,64}", parsed[1][1])
    ):
        return ""
    return parsed[1][0]


def _replace_github_local_target(
    targets: Sequence[str], target: str
) -> list[str]:
    repository = _github_local_target_repository(target)
    if not repository:
        return list(targets)
    return [
        existing
        for existing in targets
        if _github_local_target_repository(existing) != repository
    ] + [target]


def _github_local_target_from_remote_receipt(
    parameters: Mapping[str, Any] | None, result: ToolResult
) -> str:
    params = parameters if isinstance(parameters, Mapping) else {}
    document = _github_receipt_document(result)
    repository = _github_repository_identity(params, document)
    head = _github_head(document)
    if not repository or not head:
        return ""
    return _github_target("github-local:", repository, head)


def _tool_targets(
    tool_name: str,
    parameters: Mapping[str, Any] | None,
    *,
    result: ToolResult | None = None,
) -> list[str]:
    """Extract bounded resource identities used to correlate edit validation."""

    params = parameters if isinstance(parameters, Mapping) else {}
    values: list[str] = []
    name = str(tool_name or "")
    if name in _FILE_TARGET_TOOLS:
        candidate = str(params.get("file_path") or "").strip()
        if candidate:
            values.append(os.path.normpath(candidate)[:1000])
    elif name in {"browser_interact", "browser_read"}:
        tab_id = _bounded_target(params.get("tab_id"))
        if tab_id:
            values.append(f"browser-tab:{tab_id}"[:1000])
    elif name == "create_skill":
        category = _bounded_target(params.get("category"))
        skill_name = _bounded_target(params.get("skill_name"))
        if category and skill_name:
            values.append(f"skill:{category}/{skill_name}"[:1000])
    elif name in {"delete_skill", "read_skill"}:
        skill_path = _bounded_target(params.get("skill_path"))
        if skill_path:
            values.append(f"skill:{skill_path}"[:1000])
    elif name in {"delete_skill_knowledge", "read_skill_knowledge"}:
        note_id = _bounded_target(params.get("note_id"))
        if note_id:
            values.append(f"skill-knowledge:{note_id}"[:1000])
    elif name in {"github_commit", "github_status"}:
        document = _github_receipt_document(result)
        repository = _github_repository_identity(params, document)
        head = _github_head(document, nested=name == "github_status")
        if repository and (head or name == "github_commit"):
            values.append(
                _github_target("github-local:", repository, head or "unverified")
            )
        if name == "github_commit" and result is not None and result.successful:
            committed = document.get("committed_paths")
            if not isinstance(committed, list):
                committed = []
            for item in committed[:100]:
                path = _bounded_target(item)
                if not path:
                    continue
                values.append(os.path.normpath(path)[:1000])
                if repository and not os.path.isabs(path):
                    values.append(
                        os.path.normpath(os.path.join(repository, path))[:1000]
                    )
    elif name in {"github_push", "github_verify_remote"}:
        document = _github_receipt_document(result)
        repository = _github_repository_identity(params, document)
        remote_name = _github_remote_name(params, document)
        head = _github_head(document)
        if name == "github_verify_remote":
            remote_head = str(document.get("remote_head") or "").strip().lower()
            if (
                document.get("matches") is not True
                or not re.fullmatch(r"[0-9a-f]{40,64}", remote_head)
                or remote_head != head
            ):
                return []
        if repository and remote_name and (head or name == "github_push"):
            values.append(
                _github_target(
                    "github-remote:",
                    repository,
                    remote_name,
                    head or "unverified",
                )
            )
    # This typed in-process tool discovers exact paths only after opening its
    # harness-owned receipt. Its tool-authored artifacts are trusted evidence;
    # model-authored raw strings still cannot create them. Keep this allowlist
    # narrow: generic merging would reintroduce provisional GitHub targets that
    # normalize_tool_result attached before a remote receipt was available.
    if result is not None and name in {"integrate_sub_agent_changes"}:
        for item in result.artifacts[:100]:
            target = str(item or "").strip()
            if (
                target
                and len(target) <= 1000
                and not any(character in target for character in ("\x00", "\r", "\n"))
                and target not in values
            ):
                values.append(target)
    return values


def _targets_match(left: str, right: str) -> bool:
    left_github = _parse_github_target(str(left or ""))
    right_github = _parse_github_target(str(right or ""))
    if left_github is not None or right_github is not None:
        return left_github is not None and left_github == right_github
    a = os.path.normpath(str(left or ""))
    b = os.path.normpath(str(right or ""))
    if not a or not b:
        return False
    if a == b:
        return True
    # A relative path has no trustworthy identity without the launch workspace
    # against which it was resolved. Never equate it to an absolute path by
    # basename/suffix: /workspace/sub/x.py and a later x.py may be different.
    return False


def _bounded_target(value: Any) -> str:
    target = " ".join(str(value or "").strip().split())
    if (
        not target
        or len(target) > 1000
        or any(character in target for character in ("\x00", "\r", "\n"))
    ):
        return ""
    return target


def _canonical_workspace_root(value: Any = None) -> str:
    candidate = str(value or os.getcwd()).strip()
    if not candidate:
        candidate = os.getcwd()
    root = os.path.realpath(os.path.abspath(candidate))
    return root if os.path.isabs(root) else os.path.realpath(os.getcwd())


def _workspace_target(workspace_root: str, candidate: str) -> str:
    value = str(candidate or "").strip().strip("`'\"").rstrip(".,:;")
    if not value:
        return ""
    path = value if os.path.isabs(value) else os.path.join(workspace_root, value)
    normalized = os.path.realpath(os.path.abspath(path))
    try:
        if os.path.commonpath((workspace_root, normalized)) != workspace_root:
            return ""
    except ValueError:
        return ""
    return normalized


_REQUEST_LOCAL_FILE_RE = re.compile(
    r"(?<![A-Za-z0-9@:/])(?:`|['\"])?"
    r"(?P<path>(?:(?:\.?\.?/)?[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\."
    r"(?:py|pyi|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|cs|rb|php|swift|kt|"
    r"scala|sh|bash|zsh|fish|ps1|sql|html|css|scss|vue|svelte|md|rst|txt|json|"
    r"ya?ml|toml|ini|cfg|conf|xml|proto|graphql|lock))"
    r"(?:`|['\"])?(?=$|[.\s,;:!?()\[\]])",
    re.IGNORECASE,
)
_REQUEST_LOCAL_WELL_KNOWN_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?P<path>(?:(?:\.?\.?/)?[A-Za-z0-9_.-]+/)*"
    r"(?:Dockerfile(?:\.[A-Za-z0-9_.-]+)?|Makefile|LICENSE|NOTICE|CHANGELOG|"
    r"\.gitignore|\.gitattributes|\.editorconfig|\.dockerignore))"
    r"(?=$|[.\s,;:!?()\[\]])",
    re.IGNORECASE,
)
_REQUEST_LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.@/-])(?:`|['\"])?"
    r"(?P<path>(?:\.?\.?/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)"
    r"(?:`|['\"])?(?=$|[.\s,;:!?()\[\]])"
)
_REQUEST_LOCAL_MODULE_RE = re.compile(
    r"\bmodule\s+(?P<module>[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b",
    re.IGNORECASE,
)


def _existing_workspace_relative_target(
    workspace_root: str,
    candidate: str,
    *,
    require_exists: bool = False,
) -> str:
    value = str(candidate or "").strip().strip("`'\"")
    if not value or os.path.isabs(value):
        return ""
    normalized = os.path.normpath(value)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        return ""
    lexical = os.path.abspath(os.path.join(workspace_root, normalized))
    root = os.path.realpath(workspace_root)
    try:
        if os.path.commonpath((root, lexical)) != root:
            return ""
    except ValueError:
        return ""
    # Owner target discovery is read-only and must not follow any existing
    # symlink component. A new lexical descendant is still an exact obligation:
    # requiring the final leaf to exist would let "create docs/newguide" be
    # laundered by an unrelated write.
    current = root
    for part in normalized.split(os.sep):
        current = os.path.join(current, part)
        if os.path.lexists(current) and os.path.islink(current):
            return ""
    if os.path.realpath(lexical) != lexical:
        return ""
    if require_exists and not os.path.exists(lexical):
        return ""
    return normalized[:1000]


def _request_local_target_bindings(
    text: str, *, workspace_root: str | None = None
) -> list[str]:
    """Extract explicit owner-named local files without guessing directories."""

    source = _effect_text(str(text or ""))
    source = re.sub(r"\b(?:https?|ftp)://\S+", " ", source, flags=re.IGNORECASE)
    whole_mode = classify_request_mode(source)
    if whole_mode in {RequestMode.EXTERNAL_ACTION, RequestMode.DESTRUCTIVE}:
        local_clauses: list[str] = []
        for clause in _explicit_action_clauses(source):
            action = _GOAL_WRAPPER_RE.sub("", clause).strip()
            clause_mode = _classify_action_intent(action)
            if clause_mode not in {
                RequestMode.EXTERNAL_ACTION,
                RequestMode.DESTRUCTIVE,
            }:
                local_clauses.append(clause)
        source = "; ".join(local_clauses)
    values: list[str] = []

    def append(target: str) -> None:
        raw = str(target or "").strip()
        if not raw:
            return
        normalized = os.path.normpath(raw)[:1000]
        if (
            normalized not in {"", ".", ".."}
            and not normalized.startswith("../")
            and normalized not in values
            and len(values) < 20
        ):
            values.append(normalized)

    for match in _REQUEST_LOCAL_FILE_RE.finditer(source):
        append(match.group("path"))
        if len(values) >= 20:
            break
    for match in _REQUEST_LOCAL_WELL_KNOWN_RE.finditer(source):
        append(match.group("path"))
    if workspace_root:
        root = _canonical_workspace_root(workspace_root)
        for match in _REQUEST_LOCAL_PATH_RE.finditer(source):
            prefix = source[max(0, match.start() - 24) : match.start()]
            if re.search(
                r"\b(?:skill|wiki|knowledge|note)\s+$",
                prefix,
                re.IGNORECASE,
            ):
                continue
            append(_existing_workspace_relative_target(root, match.group("path")))
        for match in _REQUEST_LOCAL_MODULE_RE.finditer(source):
            relative = match.group("module").replace(".", "/")
            candidates = (f"{relative}.py", os.path.join(relative, "__init__.py"))
            resolved = next(
                (
                    target
                    for item in candidates
                    if (
                        target := _existing_workspace_relative_target(
                            root,
                            item,
                            require_exists=True,
                        )
                    )
                ),
                "",
            )
            append(resolved)
    return values


def _request_external_input_target_bindings(
    text: str, *, workspace_root: str
) -> list[str]:
    """Freeze owner-named local artifacts consumed by an external action."""

    root = _canonical_workspace_root(workspace_root)
    values: list[str] = []

    def append(candidate: str) -> None:
        target = _existing_workspace_relative_target(root, candidate)
        if target and target not in values and len(values) < 20:
            values.append(target)

    source = re.sub(
        r"\b(?:https?|ftp)://\S+",
        " ",
        _effect_text(str(text or "")),
        flags=re.IGNORECASE,
    )
    for clause in _explicit_action_clauses(source):
        action = _GOAL_WRAPPER_RE.sub("", clause).strip()
        if _classify_action_intent(action) != RequestMode.EXTERNAL_ACTION:
            continue
        for pattern in (_REQUEST_LOCAL_FILE_RE, _REQUEST_LOCAL_WELL_KNOWN_RE):
            for match in pattern.finditer(clause):
                append(match.group("path"))
        for match in _REQUEST_LOCAL_PATH_RE.finditer(clause):
            append(match.group("path"))
    return values


def _declared_external_input_targets(
    policy: ToolPolicy,
    parameters: Mapping[str, Any] | None,
    *,
    workspace_root: str,
) -> list[str]:
    """Return harness-only source declarations carried by an external call."""

    params = parameters if isinstance(parameters, Mapping) else {}
    raw: list[Any] = []
    declared = params.get("source_files")
    if isinstance(declared, list):
        raw.extend(declared[:20])
    elif declared is not None:
        raw.append(declared)
    if policy.name == "browser_interact" and params.get("file_path"):
        raw.append(params.get("file_path"))
    values: list[str] = []
    for item in raw:
        target = _existing_workspace_relative_target(
            _canonical_workspace_root(workspace_root), str(item or "")
        )
        if target and target not in values:
            values.append(target)
    return values


def _local_targets_match(workspace_root: str, left: str, right: str) -> bool:
    a = _workspace_target(workspace_root, left)
    b = _workspace_target(workspace_root, right)
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return bool(os.path.isdir(a) and os.path.commonpath((a, b)) == a)
    except ValueError:
        return False


def _validation_mentions_local_target(
    target: str,
    parameters: Mapping[str, Any] | None,
    result: ToolResult,
) -> bool:
    """Require a green-only broad validator to name the requested target.

    Exact readback is handled through typed artifacts. For commands, a basename
    or meaningful stem in the command/output is the minimum deterministic link;
    an unrelated generic ``pytest`` pass remains supporting evidence only.
    """

    params = parameters if isinstance(parameters, Mapping) else {}
    basename = os.path.basename(str(target or "")).casefold()
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    haystack = "\n".join(
        [
            json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
            result.summary,
            *result.artifacts,
        ]
    ).casefold()
    return bool(
        basename
        and (
            basename in haystack
            or (len(stem) >= 4 and re.search(rf"(?<![a-z0-9]){re.escape(stem)}(?![a-z0-9])", haystack))
        )
    )


def _resolve_named_workspace_target(workspace_root: str, text: str) -> str:
    """Resolve one owner-named immediate workspace directory, or fail closed."""

    names = [
        match.group("name")
        for match in re.finditer(
            r"(?:\b(?:in|at|under|from)\s+)?(?:the\s+)?"
            r"(?P<name>[A-Za-z][A-Za-z0-9_.-]{2,63})\s+"
            r"(?:directory|project|repo(?:sitory)?|workspace)\b",
            str(text or ""),
            re.IGNORECASE,
        )
        if match.group("name").casefold()
        not in {"this", "current", "same", "github", "gitlab", "the"}
    ]
    if not names:
        return ""

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    try:
        entries: list[tuple[str, str]] = []
        with os.scandir(workspace_root) as iterator:
            for index, entry in enumerate(iterator):
                if index >= 256:
                    break
                try:
                    if entry.is_dir(follow_symlinks=False):
                        entries.append((entry.name, entry.path))
                except OSError:
                    continue
    except OSError:
        return ""
    matches: set[str] = set()
    for requested_name in names:
        query = normalized(requested_name)
        if len(query) < 3:
            continue
        for entry_name, entry_path in entries:
            candidate = normalized(entry_name)
            camel_tokens = {
                normalized(token)
                for token in re.findall(
                    r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+",
                    entry_name,
                )
            }
            if candidate == query or candidate.startswith(query) or query in camel_tokens:
                matches.add(os.path.realpath(entry_path))
    return next(iter(matches)) if len(matches) == 1 else ""


def _request_capability_target_bindings(
    text: str,
    families: Sequence[CapabilityFamily],
    *,
    workspace_root: str,
) -> dict[str, list[str]]:
    """Extract only unambiguous owner-authored target identifiers."""

    original_value = str(text or "")
    value = re.sub(
        r"\n\nOWNER TARGET (?:REPLACEMENT|BINDING)\s+\([a-z_]+\):\n[^\n]*",
        "",
        original_value,
        flags=re.IGNORECASE,
    )
    value = _effect_text(value)
    bindings: dict[str, list[str]] = {}

    def bind(family: CapabilityFamily, candidate: Any) -> None:
        target = _bounded_target(candidate)
        if target:
            bindings.setdefault(family.value, [])
            if target not in bindings[family.value]:
                bindings[family.value].append(target)

    family_set = set(families)
    named_workspace_target = _resolve_named_workspace_target(workspace_root, value)
    if CapabilityFamily.GITHUB in family_set:
        paths = re.findall(
            r"(?<![A-Za-z0-9_.-])(/(?:home|workspace|srv|opt)/[A-Za-z0-9_./-]+)",
            value,
        )
        for path in paths[:_MAX_CAPABILITY_TARGETS]:
            bind(CapabilityFamily.GITHUB, os.path.normpath(path.rstrip(".,:;")))
        if not bindings.get(CapabilityFamily.GITHUB.value) and named_workspace_target:
            bind(CapabilityFamily.GITHUB, named_workspace_target)
        if (
            not bindings.get(CapabilityFamily.GITHUB.value)
            and re.search(
                r"\b(?:this|current)\s+(?:repo|repository)\b|"
                r"\bupdate\s+the\s+(?:github|gitlab)\b",
                value,
                re.IGNORECASE,
            )
            and os.path.isdir(os.path.join(workspace_root, ".git"))
        ):
            bind(CapabilityFamily.GITHUB, workspace_root)
    if CapabilityFamily.AGENT_INSTANCE in family_set:
        path_matches = list(re.finditer(
            r"\b(?:in|at|directory|workspace)\s+[`'\"]?"
            r"(?P<path>/(?:home|workspace|srv|opt)/[A-Za-z0-9_./-]+)",
            value,
            re.IGNORECASE,
        ))
        if not path_matches:
            full_match = re.fullmatch(
                r"\s*[`'\"]?(?P<path>/(?:home|workspace|srv|opt)/"
                r"[A-Za-z0-9_./-]+)[`'\"]?[.!]?\s*",
                value,
                re.IGNORECASE,
            )
            path_matches = [full_match] if full_match is not None else []
        for match in path_matches:
            bind(
                CapabilityFamily.AGENT_INSTANCE,
                os.path.normpath(match.group("path").rstrip(".,:;")),
            )
        if not bindings.get(CapabilityFamily.AGENT_INSTANCE.value) and named_workspace_target:
            bind(CapabilityFamily.AGENT_INSTANCE, named_workspace_target)
        name_matches = re.finditer(
            r"\b(?:create|add|make|start|launch|spawn|provision|register|set[ -]?up)\s+"
            r"(?:(?:an?|another|new|durable|managed|aeon)\s+)*"
            r"(?:agent(?:\s+(?:instance|tab|session))?\s+)?"
            r"(?P<name>[A-Za-z][A-Za-z0-9_.-]{1,63})\s+"
            r"(?:agent\b|in\b|at\b|under\b|for\b)",
            value,
            re.IGNORECASE,
        )
        for name_match in name_matches:
            if name_match.group("name").casefold() in {
                "agent", "instance", "session", "tab", "the", "this", "new"
            }:
                continue
            bind(
                CapabilityFamily.AGENT_INSTANCE,
                f"agent-name:{name_match.group('name').casefold()}",
            )
        plural_names = re.search(
            r"\b(?:create|add|make|start|launch|spawn|provision|register)\s+"
            r"agents?\s+(?P<names>[A-Za-z][A-Za-z0-9_.-]{1,63}"
            r"(?:\s*(?:,|and)\s*[A-Za-z][A-Za-z0-9_.-]{1,63})+)",
            value,
            re.IGNORECASE,
        )
        if plural_names:
            for name in re.findall(
                r"[A-Za-z][A-Za-z0-9_.-]{1,63}", plural_names.group("names")
            ):
                if name.casefold() != "and":
                    bind(CapabilityFamily.AGENT_INSTANCE, f"agent-name:{name.casefold()}")
    if CapabilityFamily.KILL_JOB in family_set:
        matches = re.finditer(
            r"\b(?:job(?:\s+id)?\s*[:#]?\s*)"
            r"(?P<id>(?=[A-Za-z0-9_-]{4,64}\b)(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]+)",
            value,
            re.IGNORECASE,
        )
        for match in matches:
            bind(CapabilityFamily.KILL_JOB, match.group("id").lower())
        for plural in re.finditer(
            r"\bjobs?\s+(?P<ids>[A-Za-z0-9_-]{4,64}"
            r"(?:\s*(?:,|and)\s*[A-Za-z0-9_-]{4,64})+)",
            value,
            re.IGNORECASE,
        ):
            for candidate in re.findall(r"[A-Za-z0-9_-]{4,64}", plural.group("ids")):
                if any(character.isdigit() for character in candidate):
                    bind(CapabilityFamily.KILL_JOB, candidate.casefold())
    if CapabilityFamily.KILL_SUB_AGENT in family_set:
        matches = re.finditer(
            r"\bsub[ -]?agent(?:\s+id)?\s*[:#]?\s*"
            r"(?P<id>(?=[A-Za-z0-9_-]{4,64}\b)(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]+)",
            value,
            re.IGNORECASE,
        )
        for match in matches:
            bind(CapabilityFamily.KILL_SUB_AGENT, match.group("id").lower())
    if CapabilityFamily.RESTART_AEON in family_set:
        bind(CapabilityFamily.RESTART_AEON, "aeon")
    if CapabilityFamily.REVERT_AEON in family_set:
        bind(CapabilityFamily.REVERT_AEON, "aeon")
    if CapabilityFamily.JOB_ROLE in family_set:
        role_match = re.search(
            r"\b(?:change|set|update|restore)\s+(?:my\s+|your\s+|the\s+)?job\s+role\s+"
            r"(?:to\s+)?(?P<role>[^.!?\n]{2,160})",
            value,
            re.IGNORECASE,
        )
        if role_match:
            role = " ".join(role_match.group("role").strip(" `\"'").split()).casefold()
            if role:
                bind(CapabilityFamily.JOB_ROLE, f"role:{role}")
    if CapabilityFamily.COLLABORATION_PORTAL in family_set:
        portal_matches = re.finditer(
            r"\b(?:collaboration|collaborator)\s+portal\s+"
            r"(?:named|called|for)\s+[`'\"]?(?P<name>[A-Za-z][A-Za-z0-9_.-]{1,63})",
            value,
            re.IGNORECASE,
        )
        for portal_match in portal_matches:
            bind(
                CapabilityFamily.COLLABORATION_PORTAL,
                f"portal-name:{portal_match.group('name').casefold()}",
            )
    if CapabilityFamily.SERVICE_CONTROL in family_set:
        matches = re.finditer(
            r"\b(?:restart|stop|disable|terminate)\s+(?:the\s+)?"
            r"(?:service\s+)?(?P<service>[A-Za-z0-9_.@-]+)",
            value,
            re.IGNORECASE,
        )
        for match in matches:
            if match.group("service").casefold() in {
                "service",
                "it",
                "this",
                "that",
                "aeon",
            }:
                continue
            bind(
                CapabilityFamily.SERVICE_CONTROL,
                f"service:{match.group('service').rstrip('.,:;!?').casefold()}",
            )
        for clause in _explicit_action_clauses(value):
            coordinated = re.search(
                r"\b(?:restart|stop|disable|terminate)\s+(?:the\s+)?"
                r"(?:services?\s+)?(?P<names>[A-Za-z0-9_.@-]+"
                r"(?:\s*(?:,|and)\s*[A-Za-z0-9_.@-]+)+)\s*$",
                clause,
                re.IGNORECASE,
            )
            if coordinated:
                for name in re.findall(r"[A-Za-z0-9_.@-]+", coordinated.group("names")):
                    if name.casefold() != "and":
                        bind(CapabilityFamily.SERVICE_CONTROL, f"service:{name.casefold()}")
    if CapabilityFamily.PROCESS_CONTROL in family_set:
        matches = re.finditer(
            r"\b(?:kill|stop|terminate)\s+(?:the\s+)?(?:process(?:es)?|pids?|daemons?)\s+"
            r"(?P<process>[A-Za-z0-9_.@-]+)",
            value,
            re.IGNORECASE,
        )
        for match in matches:
            if match.group("process").casefold() in {
                "process",
                "it",
                "this",
                "that",
            }:
                continue
            bind(
                CapabilityFamily.PROCESS_CONTROL,
                f"process:{match.group('process').rstrip('.,:;!?').casefold()}",
            )
        for plural in re.finditer(
            r"\b(?:process(?:es)?|pids?|daemons?)\s+"
            r"(?P<ids>[A-Za-z0-9_.@-]+(?:\s*(?:,|and)\s*[A-Za-z0-9_.@-]+)+)",
            value,
            re.IGNORECASE,
        ):
            for candidate in re.findall(r"[A-Za-z0-9_.@-]+", plural.group("ids")):
                if candidate.casefold() != "and":
                    bind(CapabilityFamily.PROCESS_CONTROL, f"process:{candidate.casefold()}")
    if CapabilityFamily.DELETE_RESOURCE in family_set:
        matches = re.finditer(
            r"\b(?:delete|erase|destroy|purge|wipe|remove|drop|truncate|uninstall)\s+"
            r"(?:the\s+)?[`'\"]?(?P<resource>[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)",
            value,
            re.IGNORECASE,
        )
        for match in matches:
            resource = match.group("resource")
            target = (
                _workspace_target(workspace_root, resource)
                if "/" in resource or "." in resource
                else ""
            )
            if target:
                bind(CapabilityFamily.DELETE_RESOURCE, f"path:{target}")
        for clause in _explicit_action_clauses(value):
            coordinated = re.search(
                r"\b(?:delete|erase|destroy|purge|wipe|remove|drop|truncate|uninstall)\s+"
                r"(?:the\s+)?(?P<resources>[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+"
                r"(?:\s*(?:,|and)\s*[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)+)\s*$",
                clause,
                re.IGNORECASE,
            )
            if coordinated:
                for resource in re.findall(
                    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+",
                    coordinated.group("resources"),
                ):
                    target = _workspace_target(workspace_root, resource)
                    if target:
                        bind(CapabilityFamily.DELETE_RESOURCE, f"path:{target}")
    if CapabilityFamily.SOURCE_REVERT in family_set:
        match = re.search(
            r"\b(?:commit|checkpoint|ref|revision)\s+[`'\"]?"
            r"(?P<ref>[A-Za-z0-9][A-Za-z0-9._/-]{3,99})",
            value,
            re.IGNORECASE,
        )
        if match:
            bind(CapabilityFamily.SOURCE_REVERT, f"source:{match.group('ref')}")
    if CapabilityFamily.ACCESS_REVOCATION in family_set:
        matches = re.finditer(
            r"(?P<access>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
            r"[A-Za-z0-9][A-Za-z0-9_-]{5,99})",
            value,
        )
        for match in matches:
            proposed = match.group("access")
            if proposed.casefold() in {
                "revoke", "remove", "disable", "access", "credential",
                "account", "permission", "token",
            }:
                continue
            bind(
                CapabilityFamily.ACCESS_REVOCATION,
                f"access:{proposed.casefold()}",
            )
    if CapabilityFamily.MCP_CONNECTION in family_set:
        match = re.search(
            r"\b(?:connect|link|authorize)\s+(?:my\s+|the\s+|an?\s+)?"
            r"(?P<service>[A-Za-z0-9_.-]+)(?:\s+(?:mcp\s+)?account)?",
            value,
            re.IGNORECASE,
        )
        if match and match.group("service").casefold() not in {"mcp", "account"}:
            bind(
                CapabilityFamily.MCP_CONNECTION,
                f"mcp:{match.group('service').casefold()}",
            )
    if CapabilityFamily.EXTERNAL_INTERACTION in family_set:
        external_clauses = _explicit_action_clauses(value)
        named_recipients: list[str] = []
        for clause in external_clauses:
            operation = _external_operation_target(clause)
            if operation:
                bind(CapabilityFamily.EXTERNAL_INTERACTION, operation)
            direct_group = re.search(
                r"\b(?:email|message|notify|dm|ping|invite)\s+"
                r"(?P<recipients>[A-Za-z][A-Za-z0-9_.-]{1,63}"
                r"(?:\s*(?:,|and)\s*[A-Za-z][A-Za-z0-9_.-]{1,63})+)"
                r"(?=\s+(?:the|a|an|this|that)\b|\s*$)",
                clause,
                re.IGNORECASE,
            )
            if direct_group:
                for recipient in re.findall(
                    r"[A-Za-z][A-Za-z0-9_.-]{1,63}",
                    direct_group.group("recipients"),
                ):
                    if recipient.casefold() != "and":
                        bind(
                            CapabilityFamily.EXTERNAL_INTERACTION,
                            f"recipient:{recipient.casefold()}",
                        )
            destination_matches = re.finditer(
                r"\b(?:publish|deploy|send|email|message|notify|reply|forward|share|"
                r"post|upload|submit)\b[^\n]{0,180}?\b(?:to|with|on|via)\s+"
                r"(?:the\s+)?"
                r"[`'\"]?(?P<target>"
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
                r"https?://[^\s`'\"<>]+|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}|"
                r"[A-Za-z][A-Za-z0-9_.-]{1,63})",
                clause,
                re.IGNORECASE,
            )
            for destination in destination_matches:
                target = _external_scope_target(destination.group("target"))
                if target and target not in {
                    "recipient:portal", "recipient:site", "recipient:account",
                    "recipient:platform",
                }:
                    bind(CapabilityFamily.EXTERNAL_INTERACTION, target)
                    tail = clause[destination.end() :].strip()
                    coordinated = re.fullmatch(
                        r"(?:(?:,|\band\b)\s+(?:the\s+)?"
                        r"(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
                        r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}|"
                        r"[A-Za-z][A-Za-z0-9_.-]{1,63}))+",
                        tail,
                        re.IGNORECASE,
                    )
                    if coordinated:
                        for extra in re.findall(
                            r"(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
                            r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}|"
                            r"[A-Za-z][A-Za-z0-9_.-]{1,63})",
                            tail,
                        ):
                            if extra.casefold() not in {"and", "the"}:
                                scoped = _external_scope_target(extra)
                                if scoped:
                                    bind(CapabilityFamily.EXTERNAL_INTERACTION, scoped)
            named_recipients.extend(
                re.findall(
                    r"\b(?:send|email|message|notify|reply|forward)\b"
                    r"[^.!?\n]{0,100}\bto\s+"
                    r"(?P<recipient>[A-Za-z][A-Za-z0-9_.-]{1,63})\b|"
                    r"\bshare\b[^.!?\n]{0,100}\b(?:to|with)\s+"
                    r"(?P<recipient_share>[A-Za-z][A-Za-z0-9_.-]{1,63})\b|"
                    r"\b(?:email|message|notify|dm|ping|invite)\s+"
                    r"(?P<recipient_direct>[A-Za-z][A-Za-z0-9_.-]{1,63})\b",
                    clause,
                    flags=re.IGNORECASE,
                )
            )
            for platform_match in re.finditer(
                r"\b(?:post|share|publish|upload|deploy)\b[^.!?\n]{0,80}"
                r"\b(?:on|to|via)\s+(?P<platform>X|Twitter|LinkedIn|Facebook|"
                r"Instagram|GitHub|GitLab|Slack|Discord)\b",
                clause,
                re.IGNORECASE,
            ):
                bind(
                    CapabilityFamily.EXTERNAL_INTERACTION,
                    f"platform:{platform_match.group('platform').casefold()}",
                )
            for site_match in re.finditer(
                r"\b(?:publish|deploy|upload|submit|make\s+live)\b[^.!?\n]{0,120}\b"
                r"(?P<site>https?://[^\s`'\"<>]+|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63})",
                clause,
                re.IGNORECASE,
            ):
                site = _external_scope_target(site_match.group("site"))
                if site:
                    bind(CapabilityFamily.EXTERNAL_INTERACTION, site)
            for account_match in re.finditer(
                r"\b(?:(?:using|from|with)\s+(?:the\s+)?"
                r"(?P<account1>[A-Za-z][A-Za-z0-9_.-]{1,63})\s+account|"
                r"account\s+(?:named|called)\s+[`'\"]?"
                r"(?P<account2>[A-Za-z][A-Za-z0-9_.-]{1,63}))",
                clause,
                re.IGNORECASE,
            ):
                account_name = account_match.group("account1") or account_match.group(
                    "account2"
                )
                if account_name and account_name.casefold() not in {
                    "a", "an", "the", "this", "that", "report", "file", "document",
                }:
                    bind(
                        CapabilityFamily.EXTERNAL_INTERACTION,
                        f"account:{account_name.casefold()}",
                    )
        recipients = re.findall(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value
        )
        for recipient in recipients[:_MAX_CAPABILITY_TARGETS]:
            bind(
                CapabilityFamily.EXTERNAL_INTERACTION,
                f"recipient:{recipient.casefold()}",
            )
        recipient_stop_words = {
            "a", "an", "the", "this", "that", "my", "our", "your", "with",
            "report", "file", "document", "message", "email", "them", "him",
            "her", "it",
            "portal", "site", "account", "platform",
        }
        flattened_recipients = [
            candidate
            for match in named_recipients
            for candidate in (match if isinstance(match, tuple) else (match,))
            if candidate
        ]
        for recipient in flattened_recipients[:_MAX_CAPABILITY_TARGETS]:
            if recipient.casefold() in recipient_stop_words:
                continue
            bind(
                CapabilityFamily.EXTERNAL_INTERACTION,
                f"recipient:{recipient.casefold()}",
            )

    # Target-only replies are persisted with a typed owner-authored marker so
    # restore can reconstruct the exact family and target without trusting
    # serialized bindings or reinterpreting a bare email/path as a new action.
    marker_re = re.compile(
        r"OWNER TARGET (?P<operation>REPLACEMENT|BINDING)\s+"
        r"\((?P<family>[a-z_]+)\):\n(?P<target>[^\n]+)",
        re.IGNORECASE,
    )
    for marker in marker_re.finditer(original_value):
        try:
            family = CapabilityFamily(marker.group("family").casefold())
        except ValueError:
            continue
        if family not in family_set:
            continue
        target = _bare_capability_target_binding(
            marker.group("target"),
            family,
            workspace_root=workspace_root,
        )
        if (
            target
            and family == CapabilityFamily.EXTERNAL_INTERACTION
            and target.startswith("recipient:")
        ):
            categories = {
                item.split(":", 1)[0]
                for item in bindings.get(family.value, [])
                if not item.startswith("operation:") and ":" in item
            }
            if len(categories) == 1 and "recipient" not in categories:
                target = f"{next(iter(categories))}:{marker.group('target').casefold()}"
        if not target:
            continue
        key = family.value
        existing = bindings.get(key, [])
        if marker.group("operation").casefold() == "replacement":
            if family == CapabilityFamily.AGENT_INSTANCE:
                existing = [
                    item for item in existing if not item.startswith("agent-name:")
                ]
            elif family == CapabilityFamily.EXTERNAL_INTERACTION:
                prefix = target.split(":", 1)[0] + ":"
                existing = [item for item in existing if not item.startswith(prefix)]
            else:
                existing = []
        if target not in existing:
            existing.append(target)
        bindings[key] = existing[:_MAX_CAPABILITY_TARGETS]
    return {
        key: values[:_MAX_CAPABILITY_TARGETS]
        for key, values in bindings.items()
    }


def _request_capability_action_bindings(
    text: str,
    families: Sequence[CapabilityFamily],
    *,
    workspace_root: str,
) -> dict[str, list[list[str]]]:
    """Preserve clause-local target pairings for compound capability calls."""

    groups: dict[str, list[list[str]]] = {}
    for clause in _explicit_action_clauses(_effect_text(str(text or ""))):
        clause_bindings = _request_capability_target_bindings(
            clause,
            families,
            workspace_root=workspace_root,
        )
        for family, targets in clause_bindings.items():
            if not targets:
                continue
            existing = groups.setdefault(family, [])
            unique = list(dict.fromkeys(targets))[:_MAX_CAPABILITY_TARGETS]
            family_kind = CapabilityFamily(family)
            shared: list[str] = []
            alternatives: list[str] = []
            if family_kind == CapabilityFamily.EXTERNAL_INTERACTION:
                shared = [item for item in unique if item.startswith("operation:")]
                scoped = [item for item in unique if not item.startswith("operation:")]
                categories: dict[str, list[str]] = {}
                for item in scoped:
                    categories.setdefault(item.split(":", 1)[0], []).append(item)
                multi = [values for values in categories.values() if len(values) > 1]
                if len(multi) == 1:
                    alternatives = multi[0]
                    shared.extend(
                        item for values in categories.values() if len(values) == 1 for item in values
                    )
            elif family_kind == CapabilityFamily.AGENT_INSTANCE:
                alternatives = [item for item in unique if item.startswith("agent-name:")]
                shared = [item for item in unique if not item.startswith("agent-name:")]
            elif len(unique) > 1:
                alternatives = unique
            candidate_groups = (
                [shared + [item] for item in alternatives]
                if alternatives
                else [unique]
            )
            for group in candidate_groups:
                if group and group not in existing:
                    existing.append(group)
    return groups


def _capability_call_matches_group(
    family: CapabilityFamily,
    group: Sequence[str],
    observed: Sequence[str],
) -> bool:
    if not group or not observed:
        return False
    scoped_observed = list(observed)
    if (
        family == CapabilityFamily.AGENT_INSTANCE
        and not any(bound.startswith("agent-name:") for bound in group)
    ):
        scoped_observed = [
            target
            for target in scoped_observed
            if not target.startswith("agent-name:")
        ]
    outside = [
        target
        for target in scoped_observed
        if not any(
            _capability_targets_match(bound, target, family=family)
            for bound in group
        )
    ]
    if outside:
        return False
    if family == CapabilityFamily.EXTERNAL_INTERACTION:
        categories = {
            bound.split(":", 1)[0]
            for bound in group
            if ":" in bound
        }
        return all(
            any(
                target.startswith(f"{category}:")
                and any(
                    _capability_targets_match(bound, target, family=family)
                    for bound in group
                    if bound.startswith(f"{category}:")
                )
                for target in scoped_observed
            )
            for category in categories
        )
    return all(
        any(
            _capability_targets_match(bound, target, family=family)
            for target in scoped_observed
        )
        for bound in group
    )


def _bare_capability_target_binding(
    text: str,
    family: CapabilityFamily,
    *,
    workspace_root: str,
) -> str:
    """Bind one owner reply only when one pending family supplies its meaning."""

    value = " ".join(str(text or "").strip().strip("`'\"").split()).rstrip(
        ".,;:!?"
    )
    if not value or value.casefold() in {
        "it", "this", "that", "the one", "same", "yes", "no", "default"
    }:
        if family == CapabilityFamily.JOB_ROLE and value.casefold() == "default":
            return "role:default"
        return ""
    if family in {CapabilityFamily.KILL_JOB, CapabilityFamily.KILL_SUB_AGENT}:
        return value.casefold() if re.fullmatch(r"(?=[A-Za-z0-9_-]{4,64}$)(?=.*\d)[A-Za-z0-9_-]+", value) else ""
    if family == CapabilityFamily.GITHUB:
        if re.fullmatch(
            r"/(?:home|workspace|srv|opt)(?:/[A-Za-z0-9_.-]+)+", value
        ):
            return os.path.normpath(value)
        return ""
    if family == CapabilityFamily.SERVICE_CONTROL:
        return f"service:{value.casefold()}" if re.fullmatch(r"[A-Za-z0-9_.@-]{2,100}", value) else ""
    if family == CapabilityFamily.PROCESS_CONTROL:
        return f"process:{value.casefold()}" if re.fullmatch(r"[A-Za-z0-9_.@-]{2,100}", value) else ""
    if family == CapabilityFamily.MCP_CONNECTION:
        return f"mcp:{value.casefold()}" if re.fullmatch(r"[A-Za-z0-9_.-]{2,100}", value) else ""
    if family == CapabilityFamily.COLLABORATION_PORTAL:
        return f"portal-name:{value.casefold()}" if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,63}", value) else ""
    if family == CapabilityFamily.JOB_ROLE:
        return f"role:{value.casefold()}" if 2 <= len(value) <= 160 else ""
    if family == CapabilityFamily.AGENT_INSTANCE:
        target = _workspace_target(workspace_root, value)
        if value.startswith("/") and target:
            return target
        return (
            f"agent-name:{value.casefold()}"
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,63}", value)
            else ""
        )
    if family == CapabilityFamily.DELETE_RESOURCE:
        target = _workspace_target(workspace_root, value)
        return f"path:{target}" if target and ("/" in value or "." in value) else ""
    if family == CapabilityFamily.SOURCE_REVERT:
        return f"source:{value}" if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{3,99}", value) else ""
    if family == CapabilityFamily.ACCESS_REVOCATION:
        if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value):
            return f"access:{value.casefold()}"
        if re.fullmatch(r"(?=[A-Za-z0-9_-]{6,100}$)(?=.*\d)[A-Za-z0-9_-]+", value):
            return f"access:{value.casefold()}"
    if family == CapabilityFamily.EXTERNAL_INTERACTION:
        return _external_scope_target(value)
    return ""


def _tool_capability_targets(
    family: CapabilityFamily | None,
    policy: ToolPolicy,
    parameters: Mapping[str, Any] | None,
    *,
    workspace_root: str,
) -> list[str]:
    params = parameters if isinstance(parameters, Mapping) else {}
    if family == CapabilityFamily.GITHUB:
        repository = _github_repository_identity(params, {})
        return [repository] if repository else []
    if family == CapabilityFamily.AGENT_INSTANCE:
        targets: list[str] = []
        directory = _bounded_target(params.get("directory"))
        name = _bounded_target(params.get("name"))
        if directory:
            targets.append(os.path.normpath(directory))
        if name:
            targets.append(f"agent-name:{name.casefold()}")
        return targets
    if family == CapabilityFamily.KILL_JOB:
        target = _bounded_target(params.get("job_id")).lower()
        return [target] if target else []
    if family == CapabilityFamily.KILL_SUB_AGENT:
        target = _bounded_target(params.get("agent_id")).lower()
        return [target] if target else []
    if family in {CapabilityFamily.RESTART_AEON, CapabilityFamily.REVERT_AEON}:
        return ["aeon"]
    if family == CapabilityFamily.JOB_ROLE:
        if params.get("use_default") is True:
            return ["role:default"]
        role = _bounded_target(params.get("job_role"))
        return [f"role:{role.casefold()}"] if role else []
    if family == CapabilityFamily.COLLABORATION_PORTAL:
        name = _bounded_target(params.get("name"))
        return [f"portal-name:{name.casefold()}"] if name else []
    if family == CapabilityFamily.COLLABORATION_PORTAL:
        target = _bounded_target(params.get("name"))
        return [target] if target else []
    if family == CapabilityFamily.MCP_CONNECTION:
        target = _bounded_target(
            params.get("credential_id") or params.get("service")
        )
        return [f"mcp:{target.casefold()}"] if target else []
    if family == CapabilityFamily.EXTERNAL_INTERACTION:
        arguments = params.get("arguments")
        source = arguments if isinstance(arguments, Mapping) else params
        targets: list[str] = []
        # Generic browser calls carry harness-only scope declarations. MCP calls
        # must derive scope from the real provider tool name and forwarded
        # arguments; otherwise a caller can label delete_account as "send".
        declared_scope = params if policy.name != "call_mcp_tool" else {}
        for key in ("authority_target", "authority_targets", "target_scope"):
            candidate = declared_scope.get(key)
            if isinstance(candidate, list):
                targets.extend(
                    normalized
                    for item in candidate[:8]
                    if (normalized := _external_scope_target(item))
                )
            else:
                normalized = _external_scope_target(candidate)
                if normalized:
                    targets.append(normalized)
        for key in ("to", "recipient", "email"):
            candidate = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(candidate, str) and _bounded_target(candidate):
                targets.append(f"recipient:{candidate.strip().casefold()}")
            elif isinstance(candidate, list):
                targets.extend(
                    f"recipient:{str(item).strip().casefold()}"
                    for item in candidate[:8]
                    if _bounded_target(item)
                )
        for key in ("platform", "provider", "service"):
            candidate = source.get(key) if isinstance(source, Mapping) else None
            target = _bounded_target(candidate)
            if target:
                targets.append(f"platform:{target.casefold()}")
        account = source.get("account") if isinstance(source, Mapping) else None
        account_target = _bounded_target(account)
        if account_target:
            targets.append(f"account:{account_target.casefold()}")
        for key in ("url", "domain", "site"):
            candidate = source.get(key) if isinstance(source, Mapping) else None
            normalized = _external_scope_target(candidate)
            if normalized and normalized.startswith("site:"):
                targets.append(normalized)
        operation_source = (
            params.get("tool_name")
            if policy.name == "call_mcp_tool"
            else params.get("authority_operation")
        )
        operation = _external_operation_target(operation_source)
        if operation:
            targets.append(operation)
        return list(dict.fromkeys(targets))
    if family == CapabilityFamily.ACCESS_REVOCATION:
        arguments = params.get("arguments")
        source = arguments if isinstance(arguments, Mapping) else params
        for key in (
            "email",
            "user",
            "member",
            "credential_id",
            "token_id",
            "permission_id",
        ):
            candidate = source.get(key) if isinstance(source, Mapping) else None
            target = _bounded_target(candidate)
            if target:
                return [f"access:{target.casefold()}"]
    if policy.name == "run_command":
        command = str(params.get("command", ""))
        if family == CapabilityFamily.SERVICE_CONTROL:
            try:
                words = shlex.split(command)
            except ValueError:
                return []
            for index, word in enumerate(words):
                if word in {"stop", "disable", "restart"} and index + 1 < len(words):
                    target = _bounded_target(words[index + 1])
                    return [f"service:{target.casefold()}"] if target else []
        if family == CapabilityFamily.PROCESS_CONTROL:
            match = re.search(
                r"(?:^|\s)(?:kill|pkill|killall)\s+(?:-[A-Za-z0-9]+\s+)?([^\s;&|]+)",
                command,
                re.IGNORECASE,
            )
            target = _bounded_target(match.group(1) if match else "")
            return [f"process:{target.casefold()}"] if target else []
        if family == CapabilityFamily.DELETE_RESOURCE:
            try:
                words = shlex.split(command)
            except ValueError:
                return []
            command_index = next(
                (
                    index
                    for index, word in enumerate(words)
                    if word.rsplit("/", 1)[-1]
                    in {"rm", "rmdir", "shred", "unlink", "truncate"}
                ),
                -1,
            )
            if command_index >= 0:
                targets = []
                for word in words[command_index + 1 :]:
                    if word.startswith("-"):
                        continue
                    target = _workspace_target(workspace_root, word)
                    if target:
                        targets.append(f"path:{target}")
                return list(dict.fromkeys(targets))[:_MAX_CAPABILITY_TARGETS]
        if family == CapabilityFamily.SOURCE_REVERT:
            try:
                words = shlex.split(command)
            except ValueError:
                return []
            for verb in ("reset", "clean"):
                if verb not in words:
                    continue
                candidates = [
                    word
                    for word in words[words.index(verb) + 1 :]
                    if not word.startswith("-")
                ]
                if candidates:
                    return [f"source:{candidates[0]}"]
    return []


def _capability_targets_match(
    left: str,
    right: str,
    *,
    family: CapabilityFamily | None = None,
) -> bool:
    a = _bounded_target(left)
    b = _bounded_target(right)
    if not a or not b:
        return False
    if os.path.isabs(a) or os.path.isabs(b):
        return _targets_match(a, b)
    if a.casefold() == b.casefold():
        return True
    # Only durable job/sub-agent identifiers intentionally accept unambiguous
    # short forms. Prefix matching for services/processes made `foo` authorize
    # `foobar`, defeating the owner-bound target contract.
    if (
        family in {CapabilityFamily.KILL_JOB, CapabilityFamily.KILL_SUB_AGENT}
        and len(a) >= 4
        and len(b) >= 4
        and (a.casefold().startswith(b.casefold()) or b.casefold().startswith(a.casefold()))
    ):
        return True
    return False


_GOAL_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:acceptance\s+criteria|requirements?|deliverables?|"
    r"done\s+when|success\s+criteria)\s*:?[ \t]*$",
    re.IGNORECASE,
)
_GOAL_BULLET_RE = re.compile(
    r"^\s*(?:[-*+]\s+(?:\[[ xX]\]\s+)?|\d+[.)]\s+)(?P<body>\S.*)$"
)
_GOAL_ACTION_HEAD = (
    r"(?:fix|implement|build|write|generate|render|add|create|change|modify|edit|"
    r"update|apply|configure|install|integrate|back\s*up|improve|harden|refactor|"
    r"research|audit|inspect|review|check|double\s+check|analy[sz]e|compare|"
    r"explain|describe|summari[sz]e|test|validate|verify|"
    r"document|diagnose|investigate|repair|replace|streamline|optimi[sz]e|"
    r"strengthen|rewrite|ensure|preserve|retain|leave|keep|avoid|maintain|"
    r"commit|push|publish|deploy|"
    r"upload|submit|send|share|email|message|notify|reply|forward|post|purchase|"
    r"buy|schedule|book|order|pay|subscribe|register|provision|connect|link|"
    r"authorize|delete|erase|destroy|purge|wipe|remove|drop|truncate|stop|"
    r"restart|kill|terminate|reset|revert|rollback|revoke|uninstall|unpublish|"
    r"deactivate|activate|remember|memorize|forget)"
)
_COMPLEX_MULTI_ACTION_RE = re.compile(
    rf"\b{_GOAL_ACTION_HEAD}\b[^.!?\n]{{0,180}}"
    rf"\b(?:and|then|also|plus|as\s+well\s+as)\b"
    rf"\s*(?:please\s+)?{_GOAL_ACTION_HEAD}\b",
    re.IGNORECASE,
)
_GOAL_WRAPPER_RE = re.compile(
    r"^\s*(?:(?:also|additionally|then|next)\s*[,;:]?\s*)?"
    r"(?:(?:i\s+(?:want|need)\s+you\s+to|please|you\s+should|"
    r"(?:can|could|would|will)\s+you(?:\s+please)?)\s+)?",
    re.IGNORECASE,
)
_MAX_GOAL_LEAVES = 32
_MAX_GOAL_RECORDS = _MAX_GOAL_LEAVES + 1


def _explicit_goal_items(text: str) -> list[tuple[str, str]]:
    """Compile only unmistakably owner-authored structured requirements."""

    lines = str(text or "").splitlines()
    items: list[tuple[str, str]] = []
    under_heading = False
    for line in lines:
        if _GOAL_HEADING_RE.fullmatch(line):
            under_heading = True
            continue
        match = _GOAL_BULLET_RE.match(line)
        task_list = bool(re.match(r"^\s*[-*+]\s+\[[ xX]\]\s+", line))
        if match and (under_heading or task_list):
            body = " ".join(match.group("body").split())[:1000]
            if body:
                kind = _goal_kind(body)
                items.append((kind, body))
                if len(items) > _MAX_GOAL_LEAVES:
                    break
            continue
        if under_heading and line.strip():
            under_heading = False
    return items


def _goal_kind(description: str, *, default: str = "change") -> str:
    """Classify an owner criterion without turning it into a mutation by default."""

    value = _GOAL_WRAPPER_RE.sub(
        "", " ".join(str(description or "").split())
    )
    if re.match(
        r"(?:all\b.{0,80})?\btests?\b.{0,80}\b(?:pass|succeed|remain\s+green)\b|"
        r"no\s+[^.!?]{0,80}regressions?\b|"
        r"(?:validate|verify)\b|"
        r"(?:preserve|maintain|verify|ensure)\b.*\b(?:compatibility|api|behavior|"
        r"invariant|tests?|validation)\b|(?:validation|verification)\b",
        value,
        re.IGNORECASE,
    ):
        return "validation"
    if re.match(
        r"(?:do\s+not|don['’]?t|dont|never|must\s+not|avoid|without)\b|"
        r"no\s+(?:code\s+|file\s+|state\s+)?changes?\b|"
        r"(?:preserve|retain|leave|keep)\b",
        value,
        re.IGNORECASE,
    ):
        return "invariant"
    if re.match(
        r"(?:audit|inspect|research|review|check|double\s+check|analy[sz]e|"
        r"compare|explain|describe|summari[sz]e|deep\s+dive|"
        r"take\s+a\s+(?:long\s+)?look|"
        r"assess|evaluate|examine|diagnose|investigate|trace|measure)\b",
        value,
        re.IGNORECASE,
    ):
        return "inspect"
    return default


def _unstructured_goal_items(
    text: str, *, default_kind: str
) -> list[tuple[str, str]]:
    """Conservatively freeze independent owner-authored action clauses.

    This is deliberately not a free-form semantic decomposition. It recognizes
    only direct imperatives, repeated action heads, explicit prohibitions, and
    comma lists governed by one improvement verb. Quoted examples and fenced
    code are removed before parsing so logs cannot become acceptance criteria.
    """

    safe = re.sub(r"```.*?```", " ", str(text or ""), flags=re.DOTALL)
    safe = re.sub(r"`[^`\n]{1,300}`", " ", safe)
    segments = re.split(r"(?:\r?\n)+|(?<=[.!?])\s+|\s*;\s*", safe)
    items: list[tuple[str, str]] = []

    def append(description: str) -> None:
        compact = " ".join(description.strip(" ,.;:-").split())[:1000]
        if not compact:
            return
        normalized = compact.casefold()
        if any(existing.casefold() == normalized for _, existing in items):
            return
        items.append((_goal_kind(compact, default=default_kind), compact))

    for segment in segments:
        clause = _GOAL_WRAPPER_RE.sub("", " ".join(segment.split()))
        if not clause:
            continue
        # Repeated, explicitly headed clauses are independent obligations.
        parts = re.split(
            rf"\s+(?:and(?:\s+then)?|then|also|but|plus|as\s+well\s+as)\s+"
            rf"(?=(?:do\s+not|don['’]?t|dont|never|must\s+not|"
            rf"no\s+(?:code\s+|file\s+|state\s+)?changes?|{_GOAL_ACTION_HEAD})\b)|"
            r"\s+(?=without\s+(?:changing|making|applying)\b)",
            clause,
            flags=re.IGNORECASE,
        )
        if len(parts) > 1 and any(
            len(_GOAL_WRAPPER_RE.sub("", item).strip(" .!?;").split()) < 2
            or re.fullmatch(
                rf"{_GOAL_ACTION_HEAD}\s+(?:it|this|that|them|the\s+result|"
                r"the\s+changes?)\b[.!?]*",
                _GOAL_WRAPPER_RE.sub("", item).strip(),
                re.IGNORECASE,
            )
            for item in parts
        ):
            parts = [clause]
        for part in parts:
            part = _GOAL_WRAPPER_RE.sub("", part).strip()
            if not part:
                continue
            # A leading improvement verb governing a 3+ item comma list yields
            # one leaf per exact item; this closes the common laundering hole
            # without splitting arbitrary prose on every conjunction.
            governed = re.match(
                r"(?P<head>improve|harden|strengthen|streamline|optimi[sz]e|fix)\s+"
                r"(?P<body>[^.!?]{1,900})$",
                part.strip(" .!?"),
                re.IGNORECASE,
            )
            if governed and governed.group("body").count(",") >= 2:
                components = re.split(
                    r"\s*,\s*|\s*,?\s+and\s+",
                    governed.group("body"),
                    flags=re.IGNORECASE,
                )
                components = [
                    re.sub(r"^and\s+", "", " ".join(component.split()), flags=re.IGNORECASE)
                    for component in components
                    if 1 <= len(component.split()) <= 24
                ]
                if len(components) >= 3:
                    for component in components:
                        append(f"{governed.group('head')} {component}")
                    continue
            actionable = bool(
                re.match(
                    rf"(?:do\s+not|don['’]?t|dont|never|must\s+not|{_GOAL_ACTION_HEAD})\b",
                    part,
                    re.IGNORECASE,
                )
                or _LEADING_INSPECTION_RE.match(part)
                or _goal_kind(part, default="") in {"validation", "invariant"}
            )
            if actionable:
                append(part)
        if len(items) > _MAX_GOAL_LEAVES:
            break
    return items


def _authority_goal_sections(text: str) -> tuple[str, list[str]]:
    """Return the stable base request and explicit additive owner follow-ups."""

    exact = str(text or "")
    marker = "\n\nUSER RESPONSE:\n"
    if marker not in exact:
        return exact, []
    parts = exact.split(marker)
    base = parts[0]
    additions = [
        " ".join(part.split())[:1000]
        for part in parts[1:]
        if _ADDITIVE_FOLLOWUP_RE.search(part)
    ]
    return base, additions[:_MAX_GOAL_LEAVES]


def _semantic_evidence_gate_required(
    text: str,
    mode: RequestMode,
    explicit_items: Sequence[tuple[str, str]],
) -> bool:
    if mode not in {
        RequestMode.INSPECT,
        RequestMode.CHANGE_LOCAL,
        RequestMode.EXTERNAL_ACTION,
        RequestMode.DESTRUCTIVE,
    }:
        return False
    value = str(text or "")
    return bool(
        explicit_items
        or len(value) >= 320
        or value.count(",") >= 2
        or _goal_agent_state_targets(value)
        or re.search(
            r"\b(?:remember|memorize|forget|activate|deactivate)\b",
            _effect_text(value),
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:all|both)\s+(?:of\s+)?(?:these|the\s+following|requested)\b",
            value,
            re.IGNORECASE,
        )
    )


_GOAL_EVIDENCE_STOP_WORDS = frozenset(
    {
        "about", "after", "again", "also", "anything", "apply", "authorize",
        "before", "better", "book", "build", "buy", "change", "changes",
        "check", "code", "commit", "complete", "connect", "create", "deactivate",
        "deep", "delete", "deploy", "destroy", "deliverable", "done", "drop",
        "edit", "email", "erase", "everything", "following", "forward", "implement",
        "improve", "include", "including", "issue", "kill", "link", "make",
        "message", "modify", "must", "need", "notify", "order", "pay", "please",
        "post", "project", "publish", "purchase", "purge", "push", "remove",
        "request", "requested", "requirements", "restart", "revert", "revoke",
        "rollback", "schedule", "send", "share", "should", "stop", "submit",
        "subscribe", "task", "terminate", "test", "tests", "that", "their",
        "these", "this", "through", "truncate", "uninstall", "unpublish",
        "update", "upload", "using", "validate", "verify", "wipe", "with",
        "without", "work", "working", "would", "your",
    }
)


def _goal_keywords(description: str) -> tuple[str, ...]:
    values: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(description or "")):
        normalized = token.casefold().replace("_", "-")
        if len(normalized) < 4 or normalized in _GOAL_EVIDENCE_STOP_WORDS:
            continue
        if normalized not in values:
            values.append(normalized)
        if len(values) >= 32:
            break
    return tuple(values)


def _goal_evidence_haystack(
    tool_name: str,
    parameters: Mapping[str, Any] | None,
    result: ToolResult,
    *,
    include_edit_fragments: bool,
) -> str:
    params = parameters if isinstance(parameters, Mapping) else {}
    safe_params: dict[str, Any] = {}
    for key, value in params.items():
        if key in {"content", "body", "message", "prompt", "reason"}:
            continue
        if key in {"old_str", "new_str", "old_text", "new_text", "replacement"}:
            if include_edit_fragments:
                safe_params[key] = str(value)[:2000]
            continue
        safe_params[str(key)] = value
    return "\n".join(
        [
            str(tool_name or ""),
            json.dumps(safe_params, sort_keys=True, ensure_ascii=False, default=str)[:5000],
            result.summary[:2000],
            "\n".join(result.artifacts[:20]),
        ]
    ).casefold().replace("_", "-")


def _goal_evidence_relevant(
    description: str,
    tool_name: str,
    parameters: Mapping[str, Any] | None,
    result: ToolResult,
    *,
    include_edit_fragments: bool = False,
) -> bool:
    keywords = _goal_keywords(description)
    if not keywords:
        return False
    haystack = _goal_evidence_haystack(
        tool_name,
        parameters,
        result,
        include_edit_fragments=include_edit_fragments,
    )
    matches = [
        keyword
        for keyword in keywords
        if re.search(
            rf"(?<![a-z0-9]){re.escape(keyword)}(?:s|ed|ing)?(?![a-z0-9])",
            haystack,
        )
    ]
    # A single rare keyword is enough for a narrow criterion; broad root prose
    # needs two independent lexical hooks before a green-only receipt can close.
    required = 1 if len(keywords) <= 5 else 2
    return len(matches) >= required


def _goal_agent_state_targets(description: str) -> tuple[str, ...]:
    """Freeze explicit private agent-state identities named by one goal leaf."""

    value = " ".join(str(description or "").strip().split())
    targets: list[str] = []
    for pattern, prefix in (
        (
            r"\b(?:create|add|update|delete|remove)\s+(?:the\s+)?skill\s+"
            r"(?P<target>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
            "skill:",
        ),
        (
            r"\b(?:delete|remove)\s+(?:(?:skill[- ]knowledge|wiki)\s+)?"
            r"(?:note|entry)\s+(?P<target>[A-Za-z0-9_.-]{2,100})",
            "skill-knowledge:",
        ),
    ):
        for match in re.finditer(pattern, value, re.IGNORECASE):
            target = prefix + match.group("target")
            if target not in targets:
                targets.append(target)
    return tuple(targets[:_MAX_CAPABILITY_TARGETS])


def _goal_receipt_relevant(
    description: str,
    policy: ToolPolicy,
    parameters: Mapping[str, Any] | None,
    result: ToolResult,
    *,
    workspace_root: str,
    include_edit_fragments: bool = False,
) -> bool:
    """Require lexical relevance plus exact frozen targets where available."""

    lexical_relevant = _goal_evidence_relevant(
        description,
        result.tool_name,
        parameters,
        result,
        include_edit_fragments=include_edit_fragments,
    )
    exact_target_relevant = False
    family = _tool_capability_family(policy, parameters)
    if family is not None:
        goal_mode = classify_request_mode(description)
        goal_families = _requested_capability_families(description, goal_mode)
        if family in goal_families:
            bindings = _request_capability_target_bindings(
                description,
                goal_families,
                workspace_root=workspace_root,
            ).get(family.value, [])
            if bindings:
                observed = _tool_capability_targets(
                    family,
                    policy,
                    parameters,
                    workspace_root=workspace_root,
                )
                if not all(
                    any(
                        _capability_targets_match(bound, target, family=family)
                        for target in observed
                    )
                    for bound in bindings
                ):
                    return False
                exact_target_relevant = True
    state_targets = _goal_agent_state_targets(description)
    if state_targets:
        observed_state = _tool_targets(
            result.tool_name,
            parameters,
            result=result,
        )
        if not all(target in observed_state for target in state_targets):
            return False
        exact_target_relevant = True
    local_targets = _request_local_target_bindings(
        description,
        workspace_root=workspace_root,
    )
    if local_targets:
        observed_local = _tool_targets(
            result.tool_name,
            parameters,
            result=result,
        )
        exact_local = all(
            any(
                _local_targets_match(workspace_root, bound, target)
                for target in observed_local
            )
            or (
                result.tool_name == "run_command"
                and is_validation_call(result.tool_name, parameters)
                and _validation_mentions_local_target(bound, parameters, result)
            )
            for bound in local_targets
        )
        if not exact_local:
            return False
        exact_target_relevant = True
    return bool(lexical_relevant or exact_target_relevant)


def _targetless_inspect_goal_accepts_bound_read(
    description: str, *, workspace_root: str
) -> bool:
    """Allow explicit evidence binding for a genuinely high-level inspect leaf."""

    if _request_local_target_bindings(
        description,
        workspace_root=workspace_root,
    ):
        return False
    mode = classify_request_mode(description)
    families = _requested_capability_families(description, mode)
    if _request_capability_target_bindings(
        description,
        families,
        workspace_root=workspace_root,
    ):
        return False
    if _goal_agent_state_targets(description):
        return False
    substantive = {
        keyword
        for keyword in _goal_keywords(description)
        if keyword not in {
            "anything", "double", "further", "look", "long", "more", "else",
        }
    }
    return not substantive


def _validation_goal_relevant(
    description: str,
    tool_name: str,
    parameters: Mapping[str, Any] | None,
    result: ToolResult,
) -> bool:
    if _goal_evidence_relevant(description, tool_name, parameters, result):
        return True
    value = " ".join(str(description or "").split()).casefold()
    command = str((parameters or {}).get("command", "")).strip().casefold()
    global_criterion = bool(
        re.search(r"\ball\b.{0,80}\btests?\b|\bno\b.{0,80}\bregressions?\b", value)
    )
    full_suite = bool(
        re.fullmatch(r"(?:python(?:3)?\s+-m\s+)?pytest(?:\s+-[^\s]+)*\s*", command)
        or re.fullmatch(r"(?:npm|pnpm|yarn)\s+test(?:\s+--[^\s]+)*\s*", command)
        or re.fullmatch(r"cargo\s+test(?:\s+--[^\s]+)*\s*", command)
        or re.fullmatch(r"go\s+test\s+\.\.\.(?:\s+.*)?", command)
        or re.fullmatch(r"make\s+(?:test|check)(?:\s+.*)?", command)
    )
    return bool(global_criterion and full_suite and result.successful)


def _compile_goal_anchors(
    text: str, mode: RequestMode
) -> tuple[dict[str, str], dict[str, str], bool]:
    exact = str(text or "")
    base, additions = _authority_goal_sections(exact)
    explicit = _explicit_goal_items(base)
    default_kind = "change" if mode in {
        RequestMode.CHANGE_LOCAL,
        RequestMode.EXTERNAL_ACTION,
        RequestMode.DESTRUCTIVE,
    } else "inspect"
    children = list(explicit) or _unstructured_goal_items(
        base, default_kind=default_kind
    )
    if mode == RequestMode.INSPECT:
        # The read-only effect ceiling already enforces no-mutation clauses.
        # A deictic phrase such as "without changing it" is otherwise an
        # impossible independent evidence leaf in an inspection-only turn.
        children = [item for item in children if item[0] != "invariant"]
    # A lone parsed clause is equivalent to the immutable root and needlessly
    # adds a second ID. Atomization matters only when it reveals real plurality.
    if len(children) == 1 and not explicit:
        child = " ".join(children[0][1].strip(" ,.;:!?-").split()).casefold()
        root = " ".join(
            _GOAL_WRAPPER_RE.sub("", base).strip(" ,.;:!?-").split()
        ).casefold()
        # Collapse only when the parser retained the whole root. A lone suffix
        # leaf (for example a final "then push") is not equivalent to the owner
        # request and must remain visible instead of laundering omitted clauses.
        if child == root:
            children = []
    if additions:
        # Once an owner adds a second independent obligation, retain the first
        # request as its own child even when it was originally unstructured.
        if not children:
            children.append((_goal_kind(base, default=default_kind), base[:1000]))
        for item in additions:
            parsed = _unstructured_goal_items(item, default_kind=default_kind)
            if parsed:
                children.extend(parsed)
            else:
                children.append((_goal_kind(item, default=default_kind), item))
    overflow = len(children) > _MAX_GOAL_LEAVES
    if overflow:
        # Never silently erase owner obligations. Reserve the final leaf for a
        # deterministic compiler-overflow blocker; the user can split an
        # unusually large request without the agent falsely claiming completion.
        children = children[: _MAX_GOAL_LEAVES - 1]
        children.append(
            (
                "overflow",
                "Goal compiler capacity exceeded; additional owner criteria remain "
                "outside the bounded evidence graph.",
            )
        )
    anchors = {"G0": exact[:4000] or "Exact owner request"}
    kinds = {"G0": "aggregate" if children else default_kind}
    for index, (kind, description) in enumerate(children, 1):
        goal_id = f"G{index}"
        anchors[goal_id] = description
        kinds[goal_id] = kind
    return anchors, kinds, _semantic_evidence_gate_required(
        exact, mode, children
    )


def _goal_probe_fingerprint(
    tool_name: str, parameters: Mapping[str, Any] | None
) -> str:
    payload = {
        "tool": str(tool_name or ""),
        "parameters": dict(parameters) if isinstance(parameters, Mapping) else {},
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
        ).encode("utf-8", errors="replace")
    ).hexdigest()


def _goal_receipt_identity(result: ToolResult, sequence: int) -> str:
    return str(result.call_id or f"receipt-{sequence}")[:200]


@dataclass
class RequestContract:
    raw_request: str
    mode: RequestMode
    workspace_root: str = field(default_factory=_canonical_workspace_root)
    # Text shown as the current request and text that grants authority are
    # normally identical.  Synthetic continuous turns are the one deliberate
    # exception: ``raw_request`` is the safety-preserving scheduler prompt while
    # ``authority_request`` is only the authenticated owner-configured goal.
    # Keeping both durable prevents wrapper/recovery prose from widening a
    # restored contract.
    authority_request: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: ExecutionState = ExecutionState.RUNNING
    results: list[ToolResult] = field(default_factory=list)
    pending_question: str = ""
    changed: bool = False
    satisfied: bool = False
    needs_verification: bool = False
    verified_after_change: bool = False
    external_action_satisfied: bool = False
    github_clean_required: bool = False
    github_clean_satisfied: bool = False
    github_commit_required: bool = False
    github_committed_targets: list[str] = field(default_factory=list)
    github_backup_targets: list[str] = field(default_factory=list)
    github_clean_targets: list[str] = field(default_factory=list)
    pending_validation_targets: list[str] = field(default_factory=list)
    pending_external_validation_targets: list[str] = field(default_factory=list)
    unscoped_mutation_pending: bool = False
    # Owner-named files are immutable completion obligations. Unrelated edits
    # remain auditable receipts but cannot satisfy or validate these targets.
    local_target_bindings: list[str] = field(default_factory=list)
    input_local_target_bindings: list[str] = field(default_factory=list)
    observed_input_local_targets: list[str] = field(default_factory=list)
    inspected_local_targets: list[str] = field(default_factory=list)
    changed_local_targets: list[str] = field(default_factory=list)
    validated_local_targets: list[str] = field(default_factory=list)
    goal_anchors: dict[str, str] = field(default_factory=dict)
    goal_kinds: dict[str, str] = field(default_factory=dict)
    semantic_evidence_required: bool = False
    goal_mutation_evidence: dict[str, list[str]] = field(default_factory=dict)
    goal_relevant_mutation_evidence: dict[str, list[str]] = field(default_factory=dict)
    goal_validation_evidence: dict[str, list[str]] = field(default_factory=dict)
    goal_information_evidence: dict[str, list[str]] = field(default_factory=dict)
    goal_invariant_violations: dict[str, list[str]] = field(default_factory=dict)
    goal_failed_probes: dict[str, dict[str, int]] = field(default_factory=dict)
    goal_last_mutation_sequence: dict[str, int] = field(default_factory=dict)
    observation_sequence: int = 0
    capability_families: list[str] = field(default_factory=list)
    satisfied_capability_families: list[str] = field(default_factory=list)
    capability_target_bindings: dict[str, list[str]] = field(default_factory=dict)
    # Family-level booleans are insufficient for compound requests: one
    # successful kill/push must not satisfy two separately named IDs/repos.
    # Store the exact immutable owner bindings proven by typed receipts.
    satisfied_capability_targets: dict[str, list[str]] = field(default_factory=dict)
    agent_state_requested: bool = False
    untrusted_collaborator_handoff: bool = False

    def __post_init__(self) -> None:
        # Preserve the historical direct-constructor contract. Factory-created
        # synthetic turns always pass the authenticated goal explicitly.
        if self.authority_request is None:
            self.authority_request = str(self.raw_request or "")

    @classmethod
    def from_request(
        cls,
        raw_request: str,
        *,
        forced_mode: RequestMode | str | None = None,
        authority_request: str | None = None,
        workspace_root: str | None = None,
    ) -> "RequestContract":
        exact_request = str(raw_request or "")
        authority_text = (
            exact_request
            if authority_request is None
            else str(authority_request or "")
        )
        if forced_mode is None:
            mode = classify_request_mode(authority_text)
        else:
            mode = forced_mode if isinstance(forced_mode, RequestMode) else RequestMode(str(forced_mode))
        exact_workspace_root = _canonical_workspace_root(workspace_root)
        clean_required = bool(
            mode == RequestMode.EXTERNAL_ACTION
            and _GITHUB_SCOPE_RE.search(authority_text)
            and _COMPLETE_BACKUP_SCOPE_RE.search(authority_text)
        )
        local_targets = _request_local_target_bindings(
            authority_text,
            workspace_root=exact_workspace_root,
        )
        commit_required = bool(
            _GITHUB_SCOPE_RE.search(authority_text)
            and (
                re.search(
                    r"\bcommit\b", _effect_text(authority_text), re.IGNORECASE
                )
                or local_targets
                or clean_required
            )
        )
        capability_families = _requested_capability_families(authority_text, mode)
        capability_bindings = _request_capability_target_bindings(
            authority_text,
            capability_families,
            workspace_root=exact_workspace_root,
        )
        if clean_required and CapabilityFamily.GITHUB in capability_families:
            capability_bindings.setdefault(
                CapabilityFamily.GITHUB.value, [exact_workspace_root]
            )
        goal_anchors, goal_kinds, semantic_required = _compile_goal_anchors(
            authority_text, mode
        )
        return cls(
            raw_request=exact_request,
            mode=mode,
            authority_request=authority_text,
            workspace_root=exact_workspace_root,
            github_clean_required=clean_required,
            github_commit_required=commit_required,
            capability_families=[family.value for family in capability_families],
            capability_target_bindings=capability_bindings,
            local_target_bindings=local_targets,
            input_local_target_bindings=_request_external_input_target_bindings(
                authority_text,
                workspace_root=exact_workspace_root,
            ),
            goal_anchors=goal_anchors,
            goal_kinds=goal_kinds,
            semantic_evidence_required=semantic_required,
            agent_state_requested=bool(
                _AGENT_STATE_REQUEST_RE.search(_effect_text(authority_text))
            ),
            untrusted_collaborator_handoff=_is_collaborator_handoff(exact_request),
        )

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "raw_request": self.raw_request,
            # Avoid duplicating every ordinary request in the bounded session
            # checkpoint. Absence means authority==raw for non-continuous
            # records; synthetic continuous records always differ and persist
            # the authenticated goal explicitly.
            **(
                {"authority_request": self.authority_request}
                if self.authority_request != self.raw_request
                else {}
            ),
            "mode": self.mode.value,
            "workspace_root": self.workspace_root,
            "request_id": self.request_id,
            "state": self.state.value,
            "results": [item.to_state_dict() for item in self.results[-100:]],
            "pending_question": self.pending_question,
            "changed": self.changed,
            "satisfied": self.satisfied,
            "needs_verification": self.needs_verification,
            "verified_after_change": self.verified_after_change,
            "external_action_satisfied": self.external_action_satisfied,
            "github_clean_required": self.github_clean_required,
            "github_clean_satisfied": self.github_clean_satisfied,
            "github_commit_required": self.github_commit_required,
            "github_committed_targets": list(self.github_committed_targets[:20]),
            "github_backup_targets": list(self.github_backup_targets[-50:]),
            "github_clean_targets": list(self.github_clean_targets[-50:]),
            "pending_validation_targets": list(self.pending_validation_targets[-50:]),
            "pending_external_validation_targets": list(
                self.pending_external_validation_targets[-50:]
            ),
            "unscoped_mutation_pending": self.unscoped_mutation_pending,
            "local_target_bindings": list(self.local_target_bindings[:20]),
            "input_local_target_bindings": list(
                self.input_local_target_bindings[:20]
            ),
            "observed_input_local_targets": list(
                self.observed_input_local_targets[:20]
            ),
            "inspected_local_targets": list(self.inspected_local_targets[:20]),
            "changed_local_targets": list(self.changed_local_targets[:20]),
            "validated_local_targets": list(self.validated_local_targets[:20]),
            "goal_anchors": dict(list(self.goal_anchors.items())[:_MAX_GOAL_RECORDS]),
            "goal_kinds": dict(list(self.goal_kinds.items())[:_MAX_GOAL_RECORDS]),
            "semantic_evidence_required": self.semantic_evidence_required,
            "goal_mutation_evidence": {
                key: list(values[-20:])
                for key, values in list(self.goal_mutation_evidence.items())[:_MAX_GOAL_RECORDS]
            },
            "goal_relevant_mutation_evidence": {
                key: list(values[-20:])
                for key, values in list(self.goal_relevant_mutation_evidence.items())[:_MAX_GOAL_RECORDS]
            },
            "goal_validation_evidence": {
                key: list(values[-20:])
                for key, values in list(self.goal_validation_evidence.items())[:_MAX_GOAL_RECORDS]
            },
            "goal_information_evidence": {
                key: list(values[-20:])
                for key, values in list(self.goal_information_evidence.items())[:_MAX_GOAL_RECORDS]
            },
            "goal_invariant_violations": {
                key: list(values[-20:])
                for key, values in list(self.goal_invariant_violations.items())[:_MAX_GOAL_RECORDS]
            },
            "goal_failed_probes": {
                key: dict(list(values.items())[-20:])
                for key, values in list(self.goal_failed_probes.items())[:_MAX_GOAL_RECORDS]
            },
            "goal_last_mutation_sequence": dict(self.goal_last_mutation_sequence),
            "observation_sequence": min(1000000, max(0, self.observation_sequence)),
            "capability_families": list(self.capability_families[:20]),
            "satisfied_capability_families": list(
                self.satisfied_capability_families[:20]
            ),
            "capability_target_bindings": {
                family: list(targets[:_MAX_CAPABILITY_TARGETS])
                for family, targets in list(self.capability_target_bindings.items())[:20]
            },
            "satisfied_capability_targets": {
                family: list(targets[:_MAX_CAPABILITY_TARGETS])
                for family, targets in list(
                    self.satisfied_capability_targets.items()
                )[:20]
            },
            "agent_state_requested": self.agent_state_requested,
            "untrusted_collaborator_handoff": self.untrusted_collaborator_handoff,
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any]) -> "RequestContract":
        try:
            mode = RequestMode(str(data.get("mode") or RequestMode.ANSWER.value))
        except ValueError:
            mode = RequestMode.ANSWER
        request_id = str(data.get("request_id") or uuid.uuid4().hex)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
            request_id = uuid.uuid4().hex
        try:
            state = ExecutionState(str(data.get("state") or ExecutionState.RUNNING.value))
        except ValueError:
            state = ExecutionState.RUNNING
        raw_results = data.get("results") if isinstance(data.get("results"), list) else []
        results = [
            ToolResult.from_state_dict(item)
            for item in raw_results[-100:]
            if isinstance(item, Mapping)
        ]
        raw_request = str(data.get("raw_request") or "")
        stored_authority = data.get("authority_request")
        unbound_synthetic_continuous = bool(
            raw_request.lstrip().startswith(
                _SYNTHETIC_CONTINUOUS_REQUEST_PREFIX
            )
            and (
                "authority_request" not in data
                or not isinstance(stored_authority, str)
                or not stored_authority.strip()
            )
        )
        if unbound_synthetic_continuous:
            # Checkpoints written before the authority/request split derived a
            # destructive mode from scheduler prose such as "stop merely".
            # The old record contains no authenticated copy of the stored goal,
            # so it cannot be safely reconstructed here. Resume that one cycle
            # read-only; the next fresh cycle can rebind from the owner-private
            # continuous control file through prepare_continuous_turn().
            mode = RequestMode.ANSWER
            authority_request = ""
        else:
            authority_request = (
                str(data.get("authority_request") or "")
                if "authority_request" in data
                else raw_request
            )
        workspace_root = _canonical_workspace_root(data.get("workspace_root"))
        derived_families = _requested_capability_families(authority_request, mode)
        derived_family_values = {family.value for family in derived_families}
        stored_family_values = {
            str(item)
            for item in (
                data.get("capability_families")
                if isinstance(data.get("capability_families"), list)
                else []
            )[:20]
            if str(item) in {family.value for family in CapabilityFamily}
        }
        restored_family_values = (
            stored_family_values & derived_family_values
            if stored_family_values
            else derived_family_values
        )
        restored_family_values = set(sorted(restored_family_values))
        restored_clean_required = bool(
            data.get("github_clean_required", False)
            or (
                mode == RequestMode.EXTERNAL_ACTION
                and _GITHUB_SCOPE_RE.search(authority_request)
                and _COMPLETE_BACKUP_SCOPE_RE.search(authority_request)
            )
        )
        derived_commit_required = bool(
            _GITHUB_SCOPE_RE.search(authority_request)
            and (
                bool(data.get("github_commit_required", False))
                or
                re.search(
                    r"\bcommit\b", _effect_text(authority_request), re.IGNORECASE
                )
                or _request_local_target_bindings(
                    authority_request, workspace_root=workspace_root
                )
                or restored_clean_required
            )
        )
        restored_backup_targets = [
            str(item)[:1000]
            for item in (
                data.get("github_backup_targets")
                if isinstance(data.get("github_backup_targets"), list)
                else []
            )[-50:]
            if _github_local_target_repository(str(item))
        ]
        restored_clean_targets = [
            str(item)[:1000]
            for item in (
                data.get("github_clean_targets")
                if isinstance(data.get("github_clean_targets"), list)
                else []
            )[-50:]
            if _github_local_target_repository(str(item))
        ]
        stored_satisfied_present = isinstance(
            data.get("satisfied_capability_families"), list
        )
        stored_satisfied_values = {
            str(item)
            for item in (
                data.get("satisfied_capability_families")
                if isinstance(data.get("satisfied_capability_families"), list)
                else []
            )[:20]
        }
        restored_satisfied_values = stored_satisfied_values & restored_family_values
        if not stored_satisfied_present:
            for item in results:
                family = _TOOL_CAPABILITY_FAMILIES.get(item.tool_name)
                if (
                    family is not None
                    and family.value in restored_family_values
                    and item.side_effect == SideEffect.EXTERNAL_MUTATION
                    and (item.changed or item.status == ToolStatus.NO_CHANGE)
                ):
                    restored_satisfied_values.add(family.value)
        raw_bindings = (
            data.get("capability_target_bindings")
            if isinstance(data.get("capability_target_bindings"), Mapping)
            else {}
        )
        restored_bindings: dict[str, list[str]] = {}
        for family, targets in list(raw_bindings.items())[:20]:
            family_value = str(family)
            if family_value not in restored_family_values or not isinstance(targets, list):
                continue
            cleaned = [
                target
                for target in (
                    _bounded_target(item)
                    for item in targets[:_MAX_CAPABILITY_TARGETS]
                )
                if target
            ]
            if cleaned:
                restored_bindings[family_value] = list(dict.fromkeys(cleaned))
        required_bindings = _request_capability_target_bindings(
            authority_request,
            [CapabilityFamily(value) for value in sorted(restored_family_values)],
            workspace_root=workspace_root,
        )
        if (
            restored_clean_required
            and CapabilityFamily.GITHUB.value in restored_family_values
        ):
            required_bindings.setdefault(
                CapabilityFamily.GITHUB.value, [workspace_root]
            )
        for family, targets in required_bindings.items():
            restored_bindings.setdefault(family, [])
            for target in targets:
                if target not in restored_bindings[family]:
                    restored_bindings[family].append(target)
        raw_satisfied_targets = (
            data.get("satisfied_capability_targets")
            if isinstance(data.get("satisfied_capability_targets"), Mapping)
            else {}
        )
        restored_satisfied_targets: dict[str, list[str]] = {}
        for family_value, targets in list(raw_satisfied_targets.items())[:20]:
            family_name = str(family_value)
            if (
                family_name not in restored_satisfied_values
                or family_name not in restored_bindings
                or not isinstance(targets, list)
            ):
                continue
            family = CapabilityFamily(family_name)
            proven: list[str] = []
            for bound in restored_bindings[family_name]:
                if any(
                    _capability_targets_match(
                        bound,
                        _bounded_target(candidate),
                        family=family,
                    )
                    for candidate in targets[:_MAX_CAPABILITY_TARGETS]
                ):
                    proven.append(bound)
            if proven:
                restored_satisfied_targets[family_name] = proven[
                    :_MAX_CAPABILITY_TARGETS
                ]

        def restored_family_satisfied(family_value: str) -> bool:
            if family_value not in restored_satisfied_values:
                return False
            bindings = restored_bindings.get(family_value, [])
            if not bindings:
                return True
            try:
                family = CapabilityFamily(family_value)
            except ValueError:
                return False
            proven = restored_satisfied_targets.get(family_value, [])
            return all(
                any(
                    _capability_targets_match(bound, target, family=family)
                    for target in proven
                )
                for bound in bindings
            )
        external_required = {
            value
            for value in restored_family_values
            if value
            in {
                CapabilityFamily.GITHUB.value,
                CapabilityFamily.GITHUB_CREATE.value,
                CapabilityFamily.EXTERNAL_INTERACTION.value,
                CapabilityFamily.AGENT_INSTANCE.value,
                CapabilityFamily.EXTERNAL_EXPERT.value,
                CapabilityFamily.COLLABORATION_PORTAL.value,
                CapabilityFamily.JOB_ROLE.value,
                CapabilityFamily.MCP_CONNECTION.value,
            }
        }
        restored_external = bool(
            mode == RequestMode.EXTERNAL_ACTION
            and external_required
            and all(restored_family_satisfied(value) for value in external_required)
        )
        derived_local_targets = _request_local_target_bindings(
            authority_request,
            workspace_root=workspace_root,
        )
        derived_input_targets = _request_external_input_target_bindings(
            authority_request,
            workspace_root=workspace_root,
        )
        stored_committed = {
            os.path.normpath(str(item))
            for item in (
                data.get("github_committed_targets")
                if isinstance(data.get("github_committed_targets"), list)
                else []
            )[:20]
        }
        restored_committed = [
            target
            for target in restored_bindings.get(CapabilityFamily.GITHUB.value, [])
            if os.path.normpath(target) in stored_committed
        ]
        stored_inputs = {
            os.path.normpath(str(item))
            for item in (
                data.get("observed_input_local_targets")
                if isinstance(data.get("observed_input_local_targets"), list)
                else []
            )[:20]
        }
        restored_observed_inputs = [
            target for target in derived_input_targets if target in stored_inputs
        ]

        def restored_local_evidence(key: str) -> list[str]:
            raw_values = data.get(key)
            if not isinstance(raw_values, list):
                return []
            stored = {os.path.normpath(str(item)) for item in raw_values[:20]}
            return [target for target in derived_local_targets if target in stored]

        restored_changed_local = restored_local_evidence("changed_local_targets")
        restored_inspected_local = restored_local_evidence("inspected_local_targets")
        restored_validated_local = [
            target
            for target in restored_local_evidence("validated_local_targets")
            if target in restored_changed_local
        ]
        derived_goal_anchors, derived_goal_kinds, semantic_required = (
            _compile_goal_anchors(authority_request, mode)
        )
        valid_goal_ids = set(derived_goal_anchors)
        valid_receipt_ids = {
            str(item.call_id)
            for item in results
            if str(item.call_id or "")
        }

        def restored_goal_receipts(key: str) -> dict[str, list[str]]:
            raw_map = data.get(key)
            if not isinstance(raw_map, Mapping):
                return {}
            restored: dict[str, list[str]] = {}
            for goal_id in valid_goal_ids:
                values = raw_map.get(goal_id)
                if not isinstance(values, list):
                    continue
                clean = [
                    str(item)[:200]
                    for item in values[-20:]
                    if isinstance(item, str)
                    and (not valid_receipt_ids or str(item) in valid_receipt_ids)
                ]
                if clean:
                    restored[goal_id] = list(dict.fromkeys(clean))
            return restored

        restored_mutation_evidence = restored_goal_receipts(
            "goal_mutation_evidence"
        )
        restored_relevant_mutation_evidence = restored_goal_receipts(
            "goal_relevant_mutation_evidence"
        )
        restored_validation_evidence = restored_goal_receipts(
            "goal_validation_evidence"
        )
        restored_invariant_violations = restored_goal_receipts(
            "goal_invariant_violations"
        )
        raw_information = data.get("goal_information_evidence")
        restored_information: dict[str, list[str]] = {}
        if isinstance(raw_information, Mapping):
            for goal_id in valid_goal_ids:
                values = raw_information.get(goal_id)
                if isinstance(values, list):
                    clean = [
                        str(item)
                        for item in values[-20:]
                        if re.fullmatch(r"[0-9a-f]{64}", str(item))
                    ]
                    if clean:
                        restored_information[goal_id] = list(dict.fromkeys(clean))
        try:
            observation_sequence = max(
                0, min(1000000, int(data.get("observation_sequence") or 0))
            )
        except (TypeError, ValueError):
            observation_sequence = 0
        restored_last_mutation: dict[str, int] = {}
        raw_last_mutation = data.get("goal_last_mutation_sequence")
        if isinstance(raw_last_mutation, Mapping):
            for goal_id in valid_goal_ids:
                try:
                    value = max(0, int(raw_last_mutation.get(goal_id) or 0))
                except (TypeError, ValueError):
                    continue
                if value <= observation_sequence and value:
                    restored_last_mutation[goal_id] = value
        restored_failed_probes: dict[str, dict[str, int]] = {}
        raw_failed_probes = data.get("goal_failed_probes")
        if isinstance(raw_failed_probes, Mapping):
            for goal_id in valid_goal_ids:
                values = raw_failed_probes.get(goal_id)
                if not isinstance(values, Mapping):
                    continue
                clean: dict[str, int] = {}
                for fingerprint, raw_sequence in list(values.items())[-20:]:
                    if not re.fullmatch(r"[0-9a-f]{64}", str(fingerprint)):
                        continue
                    try:
                        sequence = max(0, int(raw_sequence))
                    except (TypeError, ValueError):
                        continue
                    if sequence <= observation_sequence:
                        clean[str(fingerprint)] = sequence
                if clean:
                    restored_failed_probes[goal_id] = clean
        return cls(
            raw_request=raw_request,
            mode=mode,
            authority_request=authority_request,
            workspace_root=workspace_root,
            request_id=request_id,
            state=state,
            results=results,
            pending_question=str(data.get("pending_question") or "")[:4000],
            changed=bool(data.get("changed", False)),
            satisfied=bool(data.get("satisfied", False)),
            needs_verification=bool(data.get("needs_verification", False)),
            verified_after_change=bool(data.get("verified_after_change", False)),
            external_action_satisfied=restored_external,
            github_clean_required=restored_clean_required,
            github_clean_satisfied=bool(
                data.get("github_clean_satisfied", False)
                and restored_backup_targets
                and all(
                    target in restored_clean_targets
                    for target in restored_backup_targets
                )
            ),
            github_commit_required=derived_commit_required,
            github_committed_targets=restored_committed,
            github_backup_targets=restored_backup_targets,
            github_clean_targets=restored_clean_targets,
            pending_validation_targets=[
                str(item)[:1000]
                for item in (
                    data.get("pending_validation_targets")
                    if isinstance(data.get("pending_validation_targets"), list)
                    else []
                )[-50:]
            ],
            pending_external_validation_targets=[
                str(item)[:1000]
                for item in (
                    data.get("pending_external_validation_targets")
                    if isinstance(
                        data.get("pending_external_validation_targets"), list
                    )
                    else []
                )[-50:]
            ],
            unscoped_mutation_pending=bool(
                data.get("unscoped_mutation_pending", False)
            ),
            local_target_bindings=derived_local_targets,
            input_local_target_bindings=derived_input_targets,
            observed_input_local_targets=restored_observed_inputs,
            inspected_local_targets=restored_inspected_local,
            changed_local_targets=restored_changed_local,
            validated_local_targets=restored_validated_local,
            goal_anchors=derived_goal_anchors,
            goal_kinds=derived_goal_kinds,
            semantic_evidence_required=semantic_required,
            goal_mutation_evidence=restored_mutation_evidence,
            goal_relevant_mutation_evidence=restored_relevant_mutation_evidence,
            goal_validation_evidence=restored_validation_evidence,
            goal_information_evidence=restored_information,
            goal_invariant_violations=restored_invariant_violations,
            goal_failed_probes=restored_failed_probes,
            goal_last_mutation_sequence=restored_last_mutation,
            observation_sequence=observation_sequence,
            capability_families=sorted(restored_family_values),
            satisfied_capability_families=sorted(restored_satisfied_values),
            capability_target_bindings=restored_bindings,
            satisfied_capability_targets=restored_satisfied_targets,
            agent_state_requested=bool(
                _AGENT_STATE_REQUEST_RE.search(_effect_text(authority_request))
            ),
            untrusted_collaborator_handoff=(
                bool(data.get("untrusted_collaborator_handoff"))
                or _is_collaborator_handoff(raw_request)
            ),
        )

    @property
    def mutation_requested(self) -> bool:
        return self.mode in {
            RequestMode.CHANGE_LOCAL,
            RequestMode.EXTERNAL_ACTION,
            RequestMode.DESTRUCTIVE,
        }

    def _capability_binding_was_satisfied(
        self,
        family: CapabilityFamily,
        bound: str,
        observed: Sequence[str],
    ) -> bool:
        if any(
            _capability_targets_match(bound, target, family=family)
            for target in observed
        ):
            return True
        if family == CapabilityFamily.GITHUB and self.github_clean_required:
            # A complete-backup contract binds the owner workspace and may
            # discover multiple repositories beneath it. Exact per-repository
            # closure remains enforced by github_backup/clean target receipts.
            return any(
                os.path.isabs(bound)
                and os.path.isabs(target)
                and os.path.commonpath((bound, target)) == bound
                for target in observed
            )
        return False

    def _capability_family_satisfied(self, family: CapabilityFamily) -> bool:
        if family.value not in self.satisfied_capability_families:
            return False
        bindings = self.capability_target_bindings.get(family.value, [])
        if not bindings:
            return True
        satisfied = self.satisfied_capability_targets.get(family.value, [])
        return all(
            any(
                _capability_targets_match(bound, target, family=family)
                for target in satisfied
            )
            for bound in bindings
        )

    def _record_capability_satisfaction(
        self,
        family: CapabilityFamily,
        policy: ToolPolicy,
        parameters: Mapping[str, Any] | None,
    ) -> None:
        """Record the exact owner-bound targets proven by one typed action."""

        if family.value not in self.satisfied_capability_families:
            self.satisfied_capability_families.append(family.value)
        bindings = self.capability_target_bindings.get(family.value, [])
        if not bindings:
            return
        observed = _tool_capability_targets(
            family,
            policy,
            parameters,
            workspace_root=self.workspace_root,
        )
        proven = self.satisfied_capability_targets.setdefault(family.value, [])
        for bound in bindings:
            if (
                self._capability_binding_was_satisfied(family, bound, observed)
                and bound not in proven
            ):
                proven.append(bound)
        del proven[_MAX_CAPABILITY_TARGETS:]

    @property
    def capability_obligation_satisfied(self) -> bool:
        required = [
            CapabilityFamily(value)
            for value in self.capability_families
            if value in {family.value for family in CapabilityFamily}
        ]
        if not required:
            return not self.mutation_requested or self.mode == RequestMode.CHANGE_LOCAL
        return all(self._capability_family_satisfied(family) for family in required)

    def _external_capability_obligation_satisfied(self) -> bool:
        if self.mode != RequestMode.EXTERNAL_ACTION:
            return False
        required = [
            CapabilityFamily(value)
            for value in self.capability_families
            if value in {family.value for family in CapabilityFamily}
        ]
        return bool(required and all(
            self._capability_family_satisfied(family) for family in required
        ))

    def normalized_goal_refs(self, values: object) -> tuple[str, ...]:
        if not isinstance(values, (list, tuple)):
            values = ()
        refs: list[str] = []
        for item in values[:13]:
            goal_id = str(item or "").upper()
            if goal_id in self.goal_anchors and goal_id not in refs:
                refs.append(goal_id)
        # A single owner goal has only one possible pre-execution binding. Let
        # the harness supply it so routine tasks do not spend an extra turn
        # parroting G0; multi-goal work remains explicit and auditable.
        if not refs and len(self.goal_anchors) == 1:
            refs.append("G0")
        return tuple(refs)

    def goal_ref_error(
        self,
        policy: ToolPolicy,
        parameters: Mapping[str, Any] | None,
        goal_refs: object,
    ) -> str:
        """Require complex-task evidence bindings before an action executes."""

        supplied = (
            [str(item or "").upper() for item in goal_refs[:13]]
            if isinstance(goal_refs, (list, tuple))
            else []
        )
        unknown = [item for item in supplied if item not in self.goal_anchors]
        if unknown:
            return (
                "HARNESS BLOCKED: action goal_refs contain unknown owner-goal IDs: "
                + ", ".join(unknown[:8])
                + ". Use only IDs shown in TASK ACCEPTANCE."
            )
        aggregate_refs = [
            item
            for item in supplied
            if self.goal_kinds.get(item) in {"aggregate", "overflow"}
        ]
        if aggregate_refs:
            return (
                "HARNESS BLOCKED: aggregate/overflow goal IDs cannot receive tool evidence: "
                + ", ".join(aggregate_refs)
                + ". Bind each action to the specific leaf criteria it advances."
            )
        if not self.semantic_evidence_required:
            return ""
        effect = effective_tool_effect(policy, parameters)
        validator = is_validation_call(policy.name, parameters)
        generic_validator = bool(
            validator
            and policy.name == "run_command"
            and not _command_has_explicit_mutation(
                str((parameters or {}).get("command", ""))
            )
        )
        consequential_mutation = bool(
            effect
            in {
                SideEffect.LOCAL_MUTATION,
                SideEffect.EXTERNAL_MUTATION,
                SideEffect.DESTRUCTIVE,
            }
            and not generic_validator
        )
        if (
            (validator or consequential_mutation)
            and not supplied
            and len(self.goal_anchors) > 1
        ):
            return (
                "HARNESS BLOCKED: this complex request requires pre-execution "
                "goal_refs on mutations and validators. Bind the action to one or "
                "more TASK ACCEPTANCE IDs; evidence cannot be assigned retroactively."
            )
        return ""

    def goal_acceptance_marker(self) -> str:
        payload = {
            "relevant_mutation": {
                key: tuple(values)
                for key, values in sorted(
                    self.goal_relevant_mutation_evidence.items()
                )
            },
            "validation": {
                key: tuple(values)
                for key, values in sorted(self.goal_validation_evidence.items())
            },
            "invariant_violations": {
                key: tuple(values)
                for key, values in sorted(self.goal_invariant_violations.items())
            },
            "local_changed": tuple(sorted(self.changed_local_targets)),
            "local_validated": tuple(sorted(self.validated_local_targets)),
            "capabilities": tuple(sorted(self.satisfied_capability_families)),
            "capability_targets": {
                key: tuple(sorted(values))
                for key, values in sorted(self.satisfied_capability_targets.items())
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def goal_information_marker(self) -> str:
        payload = {
            key: tuple(values)
            for key, values in sorted(self.goal_information_evidence.items())
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _goal_verified(self, goal_id: str) -> bool:
        kind = self.goal_kinds.get(goal_id, "change")
        if kind == "aggregate":
            return all(
                self._goal_verified(item_id)
                for item_id in self.goal_anchors
                if item_id != "G0"
            )
        if kind == "overflow":
            return False
        if kind == "inspect":
            evidence_count = len(self.goal_information_evidence.get(goal_id, ()))
            required = 2 if (
                re.search(
                    r"\b(?:deep|thorough|comprehensive|audit|research|"
                    r"double\s+check)\b",
                    self.goal_anchors.get(goal_id, ""),
                    re.IGNORECASE,
                )
            ) else 1
            return evidence_count >= required
        if kind == "validation":
            return bool(self.goal_validation_evidence.get(goal_id))
        if kind == "invariant":
            return bool(
                self.goal_validation_evidence.get(goal_id)
                and not self.goal_invariant_violations.get(goal_id)
            )
        return bool(
            self.goal_relevant_mutation_evidence.get(goal_id)
            and self.goal_validation_evidence.get(goal_id)
        )

    def goal_acceptance_summary(self) -> str:
        if not self.semantic_evidence_required:
            return (
                "No semantic evidence graph is required for this proportional task; "
                "the typed request contract still enforces effects, targets, and validation."
            )
        lines = [
            "Harness-owned goal evidence (plan checkboxes do not alter it):"
        ]
        for goal_id, description in self.goal_anchors.items():
            verified = self._goal_verified(goal_id)
            if verified:
                state = "verified"
            elif self.goal_invariant_violations.get(goal_id):
                state = "violated by an observed mutation"
            elif self.goal_relevant_mutation_evidence.get(goal_id):
                state = "changed; outcome proof pending"
            elif self.goal_mutation_evidence.get(goal_id):
                state = "activity observed; relevant change pending"
            elif self.goal_information_evidence.get(goal_id):
                state = "evidence gathered; change pending"
            else:
                state = "pending"
            compact = " ".join(description.split())[:500]
            lines.append(f"- [{goal_id}] {state}: {compact}")
        lines.append(
            "Change goals need a relevant pre-bound mutation plus a targeted "
            "validator; validation-only and invariant goals need their own bound "
            "checks. Same-probe fail→change→pass is stronger evidence when available."
        )
        return "\n".join(lines)

    def goal_completion_error(self) -> str:
        if not self.semantic_evidence_required:
            return ""
        pending = [
            goal_id for goal_id in self.goal_anchors if not self._goal_verified(goal_id)
        ]
        if not pending:
            return ""
        violations = [
            goal_id
            for goal_id in pending
            if self.goal_invariant_violations.get(goal_id)
        ]
        if violations:
            return (
                "COMPLETION BLOCKED: an owner-authored invariant was violated by "
                "an observed mutation: " + ", ".join(violations) + "."
            )
        overflow = [
            goal_id for goal_id in pending if self.goal_kinds.get(goal_id) == "overflow"
        ]
        if overflow:
            return (
                "COMPLETION BLOCKED: the owner request contains more independently "
                "verifiable criteria than one bounded goal graph can represent. Ask "
                "the owner to split the request; no criteria were silently treated as done."
            )
        return (
            "COMPLETION BLOCKED: owner-goal evidence remains unsupported: "
            + ", ".join(pending)
            + ". A model-authored plan/checkmark or generic green suite is not proof. "
            "Bind goal_refs before acting, make a lexically/target-relevant change, "
            "and run a targeted validator. Prefer same-probe fail→change→pass when practical."
        )

    def _observe_goal_evidence(
        self,
        result: ToolResult,
        *,
        policy: ToolPolicy,
        parameters: Mapping[str, Any] | None,
        goal_refs: object,
    ) -> ContractDelta:
        before_verified = {
            goal_id for goal_id in self.goal_anchors if self._goal_verified(goal_id)
        }
        before_mutated = {
            goal_id
            for goal_id, values in self.goal_relevant_mutation_evidence.items()
            if values
        }
        before_information = {
            goal_id: set(values)
            for goal_id, values in self.goal_information_evidence.items()
        }
        self.observation_sequence = min(1000000, self.observation_sequence + 1)
        sequence = self.observation_sequence
        refs = self.normalized_goal_refs(goal_refs)
        if not refs:
            return ContractDelta()

        params = parameters if isinstance(parameters, Mapping) else {}
        effect = effective_tool_effect(policy, params)
        validator = is_validation_call(result.tool_name, params)
        generic_validator = bool(
            validator
            and result.tool_name == "run_command"
            and not _command_has_explicit_mutation(str(params.get("command", "")))
        )
        receipt_id = _goal_receipt_identity(result, sequence)

        if result.successful and effect == SideEffect.READ_ONLY:
            digest_payload = {
                "tool": result.tool_name,
                "targets": _tool_targets(
                    result.tool_name, params, result=result
                ),
                "summary": result.summary[:1600],
                "sha256": result.result_sha256,
            }
            digest = hashlib.sha256(
                json.dumps(
                    digest_payload,
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8", errors="replace")
            ).hexdigest()
            for goal_id in refs:
                relevant_read = _goal_receipt_relevant(
                    self.goal_anchors.get(goal_id, ""),
                    policy,
                    params,
                    result,
                    workspace_root=self.workspace_root,
                )
                if (
                    not relevant_read
                    and self.goal_kinds.get(goal_id) == "inspect"
                    and _targetless_inspect_goal_accepts_bound_read(
                        self.goal_anchors.get(goal_id, ""),
                        workspace_root=self.workspace_root,
                    )
                    and (
                        result.tool_name == "open_file"
                        or (
                            result.tool_name == "github_status"
                            and any(
                                _parse_github_target(target) is not None
                                for target in _tool_targets(
                                    result.tool_name, params, result=result
                                )
                            )
                        )
                    )
                ):
                    relevant_read = True
                if not relevant_read:
                    continue
                values = self.goal_information_evidence.setdefault(goal_id, [])
                if digest not in values:
                    values.append(digest)
                    del values[:-20]

        probe_fingerprint = (
            _goal_probe_fingerprint(result.tool_name, params) if validator else ""
        )
        if validator and result.status == ToolStatus.FAILED:
            for goal_id in refs:
                if not _goal_receipt_relevant(
                    self.goal_anchors.get(goal_id, ""),
                    policy,
                    params,
                    result,
                    workspace_root=self.workspace_root,
                ):
                    continue
                probes = self.goal_failed_probes.setdefault(goal_id, {})
                probes[probe_fingerprint] = sequence
                while len(probes) > 20:
                    probes.pop(next(iter(probes)))

        consequential_mutation = bool(
            result.changed
            and effect
            in {
                SideEffect.AGENT_STATE,
                SideEffect.LOCAL_MUTATION,
                SideEffect.EXTERNAL_MUTATION,
                SideEffect.DESTRUCTIVE,
            }
            and not generic_validator
            and not (
                policy.name == "run_command"
                and _is_git_observation_command(str(params.get("command", "")))
            )
        )
        if consequential_mutation:
            # Outcome proof is always relative to the final workspace/external
            # state, but independent one-shot effects must remain independently
            # satisfiable. Reopen global validation criteria and goals explicitly
            # bound to or lexically affected by this mutation; do not erase a
            # disjoint self-verifying receipt merely because another goal acted
            # later. Complex turns must declare goal_refs before execution.
            for goal_id, kind in self.goal_kinds.items():
                affects_goal = bool(
                    kind == "validation"
                    or goal_id in refs
                    or (
                        not refs
                        and _goal_receipt_relevant(
                            self.goal_anchors.get(goal_id, ""),
                            policy,
                            params,
                            result,
                            workspace_root=self.workspace_root,
                            include_edit_fragments=True,
                        )
                    )
                )
                if kind not in {"aggregate", "invariant"} and affects_goal:
                    self.goal_validation_evidence.pop(goal_id, None)
            # Negative owner constraints apply to every mutation, even if the
            # model omits that invariant from goal_refs. A forbidden target
            # cannot be hidden by binding the action only to a positive goal.
            for invariant_id, kind in self.goal_kinds.items():
                if kind != "invariant":
                    continue
                if not _goal_receipt_relevant(
                    self.goal_anchors.get(invariant_id, ""),
                    policy,
                    params,
                    result,
                    workspace_root=self.workspace_root,
                    include_edit_fragments=True,
                ):
                    continue
                violations = self.goal_invariant_violations.setdefault(
                    invariant_id, []
                )
                if receipt_id not in violations:
                    violations.append(receipt_id)
                    del violations[:-20]
            for goal_id in refs:
                evidence = self.goal_mutation_evidence.setdefault(goal_id, [])
                if receipt_id not in evidence:
                    evidence.append(receipt_id)
                    del evidence[:-20]
                self.goal_last_mutation_sequence[goal_id] = sequence
                self.goal_relevant_mutation_evidence.pop(goal_id, None)
                relevant_mutation = _goal_receipt_relevant(
                    self.goal_anchors.get(goal_id, ""),
                    policy,
                    params,
                    result,
                    workspace_root=self.workspace_root,
                    include_edit_fragments=True,
                )
                if relevant_mutation:
                    relevant = self.goal_relevant_mutation_evidence.setdefault(
                        goal_id, []
                    )
                    if receipt_id not in relevant:
                        relevant.append(receipt_id)
                        del relevant[:-20]
                # Any later relevant mutation invalidates an older proof.
                self.goal_validation_evidence.pop(goal_id, None)
                if (
                    (policy.self_verifying or policy.name == "github_commit")
                    and relevant_mutation
                    and self.goal_kinds.get(goal_id) != "invariant"
                ):
                    self.goal_validation_evidence[goal_id] = [receipt_id]

        if validator and result.successful:
            for goal_id in refs:
                goal_kind = self.goal_kinds.get(goal_id, "change")
                description = self.goal_anchors.get(goal_id, "")
                validation_relevant = (
                    _validation_goal_relevant(
                        description,
                        result.tool_name,
                        params,
                        result,
                    )
                    if goal_kind == "validation"
                    else _goal_receipt_relevant(
                        description,
                        policy,
                        params,
                        result,
                        workspace_root=self.workspace_root,
                    )
                )
                failed_at = self.goal_failed_probes.get(goal_id, {}).get(
                    probe_fingerprint, 0
                )
                changed_at = self.goal_last_mutation_sequence.get(goal_id, 0)
                red_green = bool(
                    validation_relevant
                    and failed_at
                    and failed_at < changed_at < sequence
                )
                targeted_green = bool(
                    validation_relevant
                    and self.goal_relevant_mutation_evidence.get(goal_id)
                )
                criterion_check = bool(
                    (goal_kind == "validation" and validation_relevant)
                    or (goal_kind == "invariant" and validation_relevant)
                )
                exact_self_verify = bool(
                    policy.self_verifying
                    and (goal_kind != "invariant" or validation_relevant)
                )
                if (
                    red_green
                    or targeted_green
                    or exact_self_verify
                    or criterion_check
                ):
                    evidence = self.goal_validation_evidence.setdefault(goal_id, [])
                    if receipt_id not in evidence:
                        evidence.append(receipt_id)
                        del evidence[:-20]

        if (
            result.status == ToolStatus.NO_CHANGE
            and effect
            in {
                SideEffect.AGENT_STATE,
                SideEffect.LOCAL_MUTATION,
                SideEffect.EXTERNAL_MUTATION,
                SideEffect.DESTRUCTIVE,
            }
        ):
            for goal_id in refs:
                self.goal_mutation_evidence[goal_id] = [receipt_id]
                if _goal_receipt_relevant(
                    self.goal_anchors.get(goal_id, ""),
                    policy,
                    params,
                    result,
                    workspace_root=self.workspace_root,
                    include_edit_fragments=True,
                ):
                    self.goal_relevant_mutation_evidence[goal_id] = [receipt_id]
                    self.goal_validation_evidence[goal_id] = [receipt_id]

        after_verified = {
            goal_id for goal_id in self.goal_anchors if self._goal_verified(goal_id)
        }
        after_mutated = {
            goal_id
            for goal_id, values in self.goal_relevant_mutation_evidence.items()
            if values
        }
        info_added = tuple(
            goal_id
            for goal_id in refs
            if set(self.goal_information_evidence.get(goal_id, ()))
            != before_information.get(goal_id, set())
        )
        return ContractDelta(
            acceptance_advanced=tuple(sorted((after_mutated - before_mutated) | (after_verified - before_verified))),
            information_added=tuple(sorted(info_added)),
            obligations_opened=tuple(sorted(after_mutated - before_mutated)),
            obligations_closed=tuple(sorted(after_verified - before_verified)),
        )

    def _capability_target_error(
        self,
        family: CapabilityFamily,
        policy: ToolPolicy,
        parameters: Mapping[str, Any] | None,
    ) -> str:
        bindings = self.capability_target_bindings.get(family.value, [])
        if not bindings:
            return ""
        if family == CapabilityFamily.GITHUB and self.github_clean_required:
            # An explicit whole-workspace backup may legitimately span the
            # workspace's independently rooted repositories. Each resulting
            # commit/push is still bound to exact repository+HEAD receipts by
            # the GitHub validation state machine.
            observed = _tool_capability_targets(
                family,
                policy,
                parameters,
                workspace_root=self.workspace_root,
            )
            if observed and all(
                any(
                    os.path.isabs(bound)
                    and os.path.isabs(target)
                    and os.path.commonpath((bound, target)) == bound
                    for bound in bindings
                )
                for target in observed
            ):
                return ""
        observed = _tool_capability_targets(
            family,
            policy,
            parameters,
            workspace_root=self.workspace_root,
        )
        try:
            requested_families = [
                CapabilityFamily(value)
                for value in self.capability_families
                if value in {item.value for item in CapabilityFamily}
            ]
        except ValueError:
            requested_families = []
        action_groups = _request_capability_action_bindings(
            self.authority_request or self.raw_request,
            requested_families,
            workspace_root=self.workspace_root,
        ).get(family.value, [])
        if action_groups:
            if any(
                _capability_call_matches_group(family, group, observed)
                for group in action_groups
            ):
                return ""
            expected_groups = " or ".join(
                "[" + ", ".join(group) + "]"
                for group in action_groups[:4]
            )
            return (
                f"HARNESS BLOCKED '{policy.name}': the '{family.value}' call does "
                "not name the same exact typed target bound to scope. It must match "
                "one exact owner-authored action/target group: "
                f"{expected_groups}. Targets from different clauses cannot be mixed."
            )
        if family == CapabilityFamily.EXTERNAL_INTERACTION:
            # Recipient/site/account/platform and operation are independent
            # parts of the owner's scope. Requiring every bound category keeps
            # a recovery turn from changing either where the action lands or
            # what external effect it performs.
            missing = [
                bound
                for bound in bindings
                if not any(
                    _capability_targets_match(bound, target, family=family)
                    for target in observed
                )
            ]
            outside = [
                target
                for target in observed
                if not any(
                    _capability_targets_match(bound, target, family=family)
                    for bound in bindings
                )
            ]
            if not missing and not outside:
                return ""
            expected = ", ".join(bindings[:6])
            return (
                f"HARNESS BLOCKED '{policy.name}': the external action is bound to "
                f"scope(s) {expected}; this call must declare every matching "
                "authority_target and authority_operation before it can act."
            )
        if family == CapabilityFamily.AGENT_INSTANCE and observed:
            # A display name is an implementation choice when the owner bound
            # only the workspace. If the owner named both, both must match.
            if all(
                any(
                    _capability_targets_match(bound, target, family=family)
                    for target in observed
                )
                for bound in bindings
            ):
                return ""
        if observed and all(
            any(
                _capability_targets_match(bound, target, family=family)
                for bound in bindings
            )
            for target in observed
        ):
            return ""
        expected = ", ".join(bindings[:4])
        return (
            f"HARNESS BLOCKED '{policy.name}': the '{family.value}' capability is "
            f"bound to target(s) {expected}; this call does not name the same exact "
            "typed target. Obtain a new explicit user request before changing scope."
        )

    def _bind_capability_target(
        self,
        family: CapabilityFamily,
        policy: ToolPolicy,
        parameters: Mapping[str, Any] | None,
    ) -> None:
        targets = _tool_capability_targets(
            family,
            policy,
            parameters,
            workspace_root=self.workspace_root,
        )
        if not targets:
            return
        existing = self.capability_target_bindings.setdefault(family.value, [])
        if existing and not (
            family == CapabilityFamily.GITHUB and self.github_clean_required
        ):
            return
        for target in targets:
            if (
                target not in existing
                and len(existing) < _MAX_CAPABILITY_TARGETS
            ):
                existing.append(target)

    def invariant_mutation_error(
        self,
        policy: ToolPolicy,
        parameters: Mapping[str, Any] | None = None,
        *,
        artifacts: Sequence[str] | None = None,
    ) -> str:
        """Return an owner-invariant conflict for a concrete mutation target.

        Ordinary tools expose targets in their proposed parameters. Receipt-
        backed capabilities such as mutable sub-agent integration discover the
        exact artifacts only inside the trusted tool and call this method again
        before performing any write.
        """

        effect = effective_tool_effect(policy, parameters)
        invariant_relevant_mutation = not (
            is_validation_call(policy.name, parameters)
            and policy.name == "run_command"
            and not _command_has_explicit_mutation(
                str((parameters or {}).get("command", ""))
            )
        )
        if not invariant_relevant_mutation or effect not in {
            SideEffect.LOCAL_MUTATION,
            SideEffect.EXTERNAL_MUTATION,
            SideEffect.DESTRUCTIVE,
        }:
            return ""
        proposed = ToolResult(
            tool_name=policy.name,
            status=ToolStatus.OK,
            changed=True,
            summary="",
            artifacts=list(artifacts or ()),
            side_effect=effect,
        )
        for goal_id, kind in self.goal_kinds.items():
            if kind != "invariant":
                continue
            if _goal_evidence_relevant(
                self.goal_anchors.get(goal_id, ""),
                policy.name,
                parameters,
                proposed,
            ):
                return (
                    f"the proposed target conflicts with owner invariant {goal_id}. "
                    "Choose a route that leaves that resource unchanged."
                )
        return ""

    def authorization_error(
        self,
        policy: ToolPolicy,
        parameters: Mapping[str, Any] | None = None,
        *,
        validate_target: bool = True,
    ) -> str:
        if (
            self.untrusted_collaborator_handoff
            and policy.name not in _COLLABORATOR_HANDOFF_ALLOWED_TOOLS
        ):
            return (
                f"HARNESS BLOCKED '{policy.name}': an authenticated collaborator "
                "handoff is untrusted project input, not owner authority. This "
                "contract permits only reasoning and owner-facing dialogue; "
                "finish it and obtain a new explicit "
                "owner-authored request before using network, persistent agent "
                "state, session control, or mutation capabilities."
            )
        if policy.name == "call_mcp_tool":
            arguments = (parameters or {}).get("arguments")
            if isinstance(arguments, Mapping) and any(
                str(key).casefold().startswith("authority_") for key in arguments
            ):
                return (
                    "HARNESS BLOCKED 'call_mcp_tool': authority_* fields are "
                    "harness metadata and may not be forwarded as provider "
                    "arguments. Scope is derived from the real MCP tool name and "
                    "its actual recipient/site fields."
                )
        effect = effective_tool_effect(policy, parameters)
        if effect not in _ALLOWED_EFFECTS[self.mode]:
            return (
                f"HARNESS BLOCKED '{policy.name}': this request is mode '{self.mode.value}' "
                f"and does not authorize a '{effect.value}' action. Answer/inspect/plan "
                "requests are read-only; obtain a new explicit user request before mutating state."
            )
        invariant_error = self.invariant_mutation_error(
            policy,
            parameters,
            artifacts=_tool_targets(policy.name, parameters),
        )
        if invariant_error:
            return f"HARNESS BLOCKED '{policy.name}': {invariant_error}"

        if (
            validate_target
            and
            policy.name == "github_commit"
            and self.local_target_bindings
            and not self.github_clean_required
        ):
            params = parameters if isinstance(parameters, Mapping) else {}
            raw_paths = params.get("paths")
            provided = [
                os.path.normpath(str(item))
                for item in (raw_paths if isinstance(raw_paths, list) else [])
                if _bounded_target(item)
            ]
            missing = [
                target
                for target in self.local_target_bindings
                if not any(
                    _local_targets_match(self.workspace_root, target, item)
                    for item in provided
                )
            ]
            outside = [
                item
                for item in provided
                if not any(
                    _local_targets_match(self.workspace_root, target, item)
                    for target in self.local_target_bindings
                )
            ]
            if missing or outside:
                return (
                    "HARNESS BLOCKED 'github_commit': commit paths must match the "
                    "owner-named local targets exactly. Missing: "
                    + (", ".join(missing[:8]) or "none")
                    + "; outside scope: "
                    + (", ".join(outside[:8]) or "none")
                    + "."
                )

        family = _tool_capability_family(policy, parameters)
        consequential_family_call = bool(
            family is not None
            and (
                effect
                in {
                    SideEffect.EXTERNAL_MUTATION,
                    SideEffect.DESTRUCTIVE,
                }
                or policy.name == "github_commit"
                or (
                    policy.name == "run_command"
                    and effect
                    in {
                        SideEffect.LOCAL_MUTATION,
                        SideEffect.EXTERNAL_MUTATION,
                        SideEffect.DESTRUCTIVE,
                    }
                )
            )
        )
        if consequential_family_call:
            if family.value not in self.capability_families:
                granted = ", ".join(self.capability_families) or "none"
                return (
                    f"HARNESS BLOCKED '{policy.name}': this request authorizes capability "
                    f"family/families [{granted}], not '{family.value}'. A capability at "
                    "the same side-effect level cannot substitute for the action the user named."
                )
            if policy.name == "run_command" and family == CapabilityFamily.GITHUB:
                return (
                    "HARNESS BLOCKED 'run_command': GitHub mutations require the typed "
                    "github_commit/github_push gateway so repository, credential, HEAD, and "
                    "remote identity remain receipt-bound."
                )
            if (
                family == CapabilityFamily.EXTERNAL_INTERACTION
                and self.input_local_target_bindings
            ):
                unread = [
                    target
                    for target in self.input_local_target_bindings
                    if target not in self.observed_input_local_targets
                ]
                declared = _declared_external_input_targets(
                    policy,
                    parameters,
                    workspace_root=self.workspace_root,
                )
                missing_sources = [
                    target
                    for target in self.input_local_target_bindings
                    if not any(
                        _local_targets_match(self.workspace_root, target, item)
                        for item in declared
                    )
                ]
                if unread or missing_sources:
                    return (
                        f"HARNESS BLOCKED '{policy.name}': owner-named external "
                        "source artifacts must be read exactly and declared through "
                        "source_files (or browser upload file_path). Unread: "
                        + (", ".join(unread[:8]) or "none")
                        + "; undeclared: "
                        + (", ".join(missing_sources[:8]) or "none")
                        + "."
                    )
            if validate_target and (
                family in _TARGET_REQUIRED_FAMILIES
                and not _capability_binding_complete(
                    family, self.capability_target_bindings.get(family.value)
                )
            ):
                preflight = (
                    " Ask the owner for the exact job ID, then inspect that ID with job_output."
                    if family == CapabilityFamily.KILL_JOB
                    else " Ask the owner for the exact sub-agent ID, then inspect that ID with get_sub_agent_report."
                    if family == CapabilityFamily.KILL_SUB_AGENT
                    else " Obtain an exact owner-authored target before acting."
                )
                return (
                    f"HARNESS BLOCKED '{policy.name}': the '{family.value}' capability "
                    "has no exact bound target; a model-selected target cannot silently "
                    "become owner authority."
                    + preflight
                )
            if validate_target:
                target_error = self._capability_target_error(family, policy, parameters)
                if target_error:
                    return target_error
            if (
                validate_target
                and policy.name == "github_push"
                and self.github_commit_required
            ):
                repository = _github_repository_identity(
                    parameters if isinstance(parameters, Mapping) else {}, {}
                )
                if not repository or not any(
                    _targets_match(repository, target)
                    for target in self.github_committed_targets
                ):
                    return (
                        "HARNESS BLOCKED 'github_push': this request includes local "
                        "work that must be captured by a typed github_commit after "
                        "the latest edit before any push can satisfy it."
                    )
        elif (
            validate_target
            and family is not None
            and family.value in self.capability_target_bindings
        ):
            # Read-only typed validators remain bound to the same exact target
            # once a consequential action establishes it.
            target_error = self._capability_target_error(family, policy, parameters)
            if target_error:
                return target_error
        return ""

    def observe(
        self,
        result: ToolResult,
        *,
        policy: ToolPolicy,
        parameters: Mapping[str, Any] | None = None,
        goal_refs: object = None,
    ) -> ContractDelta:
        prior_validation_targets = list(self.pending_validation_targets)
        prior_unscoped_mutation = self.unscoped_mutation_pending
        self.results.append(result)
        effect = effective_tool_effect(policy, parameters)
        agent_state_change = bool(
            result.changed and effect == SideEffect.AGENT_STATE
        )
        family = _tool_capability_family(policy, parameters)
        capability_authorized = bool(
            family is not None
            and family.value in self.capability_families
            and not self.authorization_error(policy, parameters)
        )
        params = parameters if isinstance(parameters, Mapping) else {}
        receipt_targets = _tool_targets(
            result.tool_name, parameters, result=result
        )
        incidental_git_observer = bool(
            result.successful
            and policy.name == "run_command"
            and _is_git_observation_command(str(params.get("command", "")))
        )
        if result.successful and effect == SideEffect.READ_ONLY:
            for required in self.input_local_target_bindings:
                if any(
                    _local_targets_match(self.workspace_root, required, observed)
                    for observed in receipt_targets
                ) and required not in self.observed_input_local_targets:
                    self.observed_input_local_targets.append(required)

        # A push proves only the commit that existed at that moment. Any later
        # project mutation (including a new commit) reopens the GitHub obligation
        # so an old remote receipt can never launder newer local work.
        if (
            result.changed
            and effect == SideEffect.LOCAL_MUTATION
            and not incidental_git_observer
            and CapabilityFamily.GITHUB.value in self.capability_families
        ):
            self.github_commit_required = True
            self.github_committed_targets.clear()
            self.github_clean_satisfied = False
            self.github_clean_targets.clear()
            if CapabilityFamily.GITHUB.value in self.satisfied_capability_families:
                self.satisfied_capability_families.remove(
                    CapabilityFamily.GITHUB.value
                )
            self.satisfied_capability_targets.pop(
                CapabilityFamily.GITHUB.value, None
            )
            self.external_action_satisfied = False
            self.pending_external_validation_targets.clear()
            for goal_id, description in self.goal_anchors.items():
                goal_mode = classify_request_mode(description)
                if CapabilityFamily.GITHUB in _requested_capability_families(
                    description, goal_mode
                ):
                    self.goal_mutation_evidence.pop(goal_id, None)
                    self.goal_relevant_mutation_evidence.pop(goal_id, None)
                    self.goal_validation_evidence.pop(goal_id, None)

        if (
            result.changed
            and effect == SideEffect.LOCAL_MUTATION
            and not incidental_git_observer
        ):
            self.observed_input_local_targets = [
                target
                for target in self.observed_input_local_targets
                if not any(
                    _local_targets_match(self.workspace_root, target, observed)
                    for observed in receipt_targets
                )
            ]

        if (
            policy.name == "github_commit"
            and capability_authorized
            and (result.changed or result.status == ToolStatus.NO_CHANGE)
        ):
            repository = _github_repository_identity(params, _github_receipt_document(result))
            if repository and repository not in self.github_committed_targets:
                self.github_committed_targets.append(repository)
        consequential_family_call = bool(
            family is not None
            and (
                effect in {SideEffect.EXTERNAL_MUTATION, SideEffect.DESTRUCTIVE}
                or policy.name == "github_commit"
                or (
                    policy.name == "run_command"
                    and effect
                    in {
                        SideEffect.LOCAL_MUTATION,
                        SideEffect.EXTERNAL_MUTATION,
                        SideEffect.DESTRUCTIVE,
                    }
                )
            )
        )
        if capability_authorized and (
            consequential_family_call
        ):
            self._bind_capability_target(family, policy, parameters)
        if (
            self.github_clean_required
            and family == CapabilityFamily.GITHUB
            and capability_authorized
        ):
            backup_target = ""
            if result.tool_name == "github_commit" and result.changed:
                backup_target = next(
                    (
                        target
                        for target in _tool_targets(
                            result.tool_name, parameters, result=result
                        )
                        if _github_local_target_repository(target)
                    ),
                    "",
                )
            elif result.tool_name == "github_push" and (
                result.changed
                or result.status == ToolStatus.NO_CHANGE
                or result.error_code == "remote_outcome_ambiguous"
            ):
                backup_target = _github_local_target_from_remote_receipt(
                    parameters, result
                )
            if backup_target:
                repository = _github_local_target_repository(backup_target)
                previous_target = next(
                    (
                        target
                        for target in self.github_backup_targets
                        if _github_local_target_repository(target) == repository
                    ),
                    "",
                )
                self.github_backup_targets = _replace_github_local_target(
                    self.github_backup_targets, backup_target
                )
                if previous_target != backup_target:
                    self.github_clean_targets = [
                        target
                        for target in self.github_clean_targets
                        if _github_local_target_repository(target) != repository
                    ]

            if result.successful and result.tool_name == "github_status":
                status_target = next(
                    (
                        target
                        for target in _tool_targets(
                            result.tool_name, parameters, result=result
                        )
                        if _github_local_target_repository(target)
                    ),
                    "",
                )
                document = _github_receipt_document(result)
                repository_receipt = document.get("repository")
                if status_target and not self.github_backup_targets:
                    self.github_backup_targets = [status_target]
                if status_target in self.github_backup_targets:
                    repository = _github_local_target_repository(status_target)
                    self.github_clean_targets = [
                        target
                        for target in self.github_clean_targets
                        if _github_local_target_repository(target) != repository
                    ]
                    if (
                        isinstance(repository_receipt, Mapping)
                        and repository_receipt.get("dirty") is False
                    ):
                        self.github_clean_targets.append(status_target)
            self.github_clean_satisfied = bool(
                self.github_backup_targets
                and all(
                    target in self.github_clean_targets
                    for target in self.github_backup_targets
                )
            )
        if (
            effect == SideEffect.EXTERNAL_MUTATION
            and family == CapabilityFamily.GITHUB
            and capability_authorized
            and result.status == ToolStatus.FAILED
            and result.error_code == "remote_outcome_ambiguous"
        ):
            ambiguous_document = _github_receipt_document(result)
            ambiguous_head = _github_head(ambiguous_document)
            observed_remote_head = str(
                ambiguous_document.get("remote_head") or ""
            ).strip().lower()
            ambiguous_targets = (
                _tool_targets(result.tool_name, parameters, result=result)
                if (
                    ambiguous_document.get("outcome_ambiguous") is True
                    and ambiguous_document.get("verification_required") is True
                    and ambiguous_head
                    and "remote_head" in ambiguous_document
                    and (
                        ambiguous_document.get("remote_head") is None
                        or (
                            re.fullmatch(r"[0-9a-f]{40,64}", observed_remote_head)
                            and observed_remote_head != ambiguous_head
                        )
                    )
                )
                else []
            )
            for target in ambiguous_targets:
                if target not in self.pending_validation_targets:
                    self.pending_validation_targets.append(target)
                if target not in self.pending_external_validation_targets:
                    self.pending_external_validation_targets.append(target)
            if ambiguous_targets:
                self.needs_verification = True
        if (
            capability_authorized
            and family is not None
            and effect == SideEffect.EXTERNAL_MUTATION
            and (result.changed or result.status == ToolStatus.NO_CHANGE)
        ):
            self._record_capability_satisfaction(family, policy, parameters)
        if (
            capability_authorized
            and family is not None
            and (
                effect == SideEffect.DESTRUCTIVE
                or (
                    self.mode == RequestMode.CHANGE_LOCAL
                    and policy.name == "github_commit"
                )
            )
            and (result.changed or result.status == ToolStatus.NO_CHANGE)
        ):
            self._record_capability_satisfaction(family, policy, parameters)
        self.external_action_satisfied = (
            self._external_capability_obligation_satisfied()
        )

        if (
            self.mode == RequestMode.INSPECT
            and result.successful
            and effect == SideEffect.READ_ONLY
        ):
            for required in self.local_target_bindings:
                if any(
                    _local_targets_match(self.workspace_root, required, observed)
                    for observed in receipt_targets
                ) and required not in self.inspected_local_targets:
                    self.inspected_local_targets.append(required)
        matched_owner_local_targets = [
            required
            for required in self.local_target_bindings
            if any(
                _local_targets_match(self.workspace_root, required, observed)
                for observed in receipt_targets
            )
        ]
        if (
            result.changed
            and not agent_state_change
            and effect
            in {
                SideEffect.LOCAL_MUTATION,
                SideEffect.EXTERNAL_MUTATION,
                SideEffect.DESTRUCTIVE,
            }
        ):
            for target in matched_owner_local_targets:
                if target not in self.changed_local_targets:
                    self.changed_local_targets.append(target)
                # A later edit reopens semantic/readback debt for that target.
                if target in self.validated_local_targets:
                    self.validated_local_targets.remove(target)
                if (
                    result.tool_name == "github_commit"
                    and result.successful
                    and target not in self.validated_local_targets
                ):
                    # The typed gateway's committed_paths + HEAD receipt proves
                    # that this exact owner-named file was included in the commit.
                    self.validated_local_targets.append(target)
        generic_mutating_validator = bool(
            result.successful
            and policy.name == "run_command"
            and effect == SideEffect.LOCAL_MUTATION
            and is_validation_call(result.tool_name, parameters)
            and not _command_has_explicit_mutation(str(params.get("command", "")))
        )
        incidental_git_observer = bool(
            result.successful
            and policy.name == "run_command"
            and effect == SideEffect.LOCAL_MUTATION
            and _is_git_observation_command(str(params.get("command", "")))
        )
        prior_scoped_validation_obligation = bool(
            prior_validation_targets and not prior_unscoped_mutation
        )
        track_current_mutation = bool(
            result.changed
            and not agent_state_change
            and not (
                generic_mutating_validator and prior_scoped_validation_obligation
            )
        )
        if result.changed and not incidental_git_observer:
            if agent_state_change:
                # Private skill/memory CRUD is bounded agent state. A successful
                # typed receipt completes that operation without inventing a
                # workspace validation obligation only when the owner actually
                # requested agent-state work. Opportunistic memory/skill upkeep
                # must never launder an unrelated code, external, or destructive
                # obligation through the coarse changed/satisfied flags.
                if self.agent_state_requested:
                    self.changed = True
                    self.satisfied = True
            elif policy.self_verifying and (
                not consequential_family_call or capability_authorized
            ):
                self.changed = True
                self.satisfied = True
            elif track_current_mutation:
                self.changed = True
                mutation_targets = _tool_targets(
                    result.tool_name, parameters, result=result
                )
                if result.tool_name == "github_commit":
                    # committed_paths are exact membership evidence for owner
                    # local-target accounting, not separate post-commit debts.
                    # HEAD/status verification is tracked by github-local IDs.
                    mutation_targets = [
                        target
                        for target in mutation_targets
                        if _github_local_target_repository(target)
                    ]
                for target in mutation_targets:
                    if target not in self.pending_validation_targets:
                        self.pending_validation_targets.append(target)
                if not mutation_targets:
                    self.unscoped_mutation_pending = True
                self.needs_verification = True
        if (
            result.successful
            and is_validation_call(result.tool_name, parameters)
            and (effect == SideEffect.READ_ONLY or generic_mutating_validator)
        ):
            validation_targets = _tool_targets(
                result.tool_name, parameters, result=result
            )
            resolved_external_targets = [
                target
                for target in self.pending_external_validation_targets
                if any(
                    _targets_match(target, observed)
                    for observed in validation_targets
                )
            ]
            if resolved_external_targets:
                self.pending_external_validation_targets = [
                    target
                    for target in self.pending_external_validation_targets
                    if target not in resolved_external_targets
                ]
                # The push pre-read proved the branch did not have the expected
                # HEAD; this fresh exact receipt now proves that it does.  Only
                # that typed transition can resolve an ambiguous external push.
                if (
                    capability_authorized
                ):
                    self._record_capability_satisfaction(
                        CapabilityFamily.GITHUB,
                        policy,
                        parameters,
                    )
                self.external_action_satisfied = (
                    self._external_capability_obligation_satisfied()
                )
                self.changed = True
            if self.pending_validation_targets and validation_targets:
                remaining = [
                    target
                    for target in self.pending_validation_targets
                    if not any(
                        _targets_match(target, observed)
                        for observed in validation_targets
                    )
                ]
                self.pending_validation_targets = remaining
                relevant = not remaining and not self.unscoped_mutation_pending
            elif any(
                target.startswith(_GITHUB_TARGET_PREFIXES)
                for target in self.pending_validation_targets
            ):
                # A generic successful test or `run_command git status` cannot
                # prove an exact Nexus-mediated local commit or GitHub remote
                # identity. Only the typed status/remote receipts above carry
                # the bound repository, HEAD and remote name.
                relevant = False
            elif self.unscoped_mutation_pending:
                # No validator can prove an unnamed mutation by coincidence.
                # The mutating capability must return a bounded target or a
                # self-verifying typed receipt.
                relevant = False
            else:
                # Tests/health probes validate a broader postcondition and do
                # not necessarily name every touched file in their arguments.
                relevant = True
                self.pending_validation_targets = []
            if relevant and (
                not consequential_family_call or capability_authorized
            ):
                self.satisfied = True
                self.verified_after_change = True
                self.needs_verification = False
            for target in self.changed_local_targets:
                exact_target_evidence = any(
                    _local_targets_match(self.workspace_root, target, observed)
                    for observed in validation_targets
                )
                targeted_validator_evidence = bool(
                    result.tool_name == "run_command"
                    and _validation_mentions_local_target(target, params, result)
                )
                if (
                    (exact_target_evidence or targeted_validator_evidence)
                    and target not in self.validated_local_targets
                ):
                    self.validated_local_targets.append(target)
        elif result.status == ToolStatus.NO_CHANGE and effect in {
            SideEffect.AGENT_STATE,
            SideEffect.LOCAL_MUTATION,
            SideEffect.EXTERNAL_MUTATION,
            SideEffect.DESTRUCTIVE,
        }:
            # A mutating tool's explicit no-change receipt can prove the desired
            # state was already present, but final prose must say that plainly.
            if (
                (effect != SideEffect.AGENT_STATE or self.agent_state_requested)
                and (not consequential_family_call or capability_authorized)
            ):
                self.satisfied = True
            if (
                not self.pending_validation_targets
                and not self.unscoped_mutation_pending
            ):
                self.needs_verification = False
            for target in matched_owner_local_targets:
                if target not in self.changed_local_targets:
                    self.changed_local_targets.append(target)
                if target not in self.validated_local_targets:
                    self.validated_local_targets.append(target)
        elif result.changed and not agent_state_change:
            self.verified_after_change = bool(
                policy.self_verifying
                and (not consequential_family_call or capability_authorized)
            )
            self.needs_verification = not self.verified_after_change
            if self.verified_after_change:
                self.pending_validation_targets = []
        return self._observe_goal_evidence(
            result,
            policy=policy,
            parameters=parameters,
            goal_refs=goal_refs,
        )

    def continue_with(self, user_response: str) -> str:
        """Continue a waiting request without discarding prior receipts.

        A reply may grant additional authority (for example, "yes, apply it") or
        explicitly revoke it (for example, "do not change anything; explain").
        Evidence remains monotonic, while the newest exact owner instruction is
        the executable authority ceiling. The response also remains a separate
        user-role chat message.
        """

        response = str(user_response or "")
        confirmed_proposal = str(self.pending_question or "")
        existing_families: list[CapabilityFamily] = []
        for family_value in self.capability_families:
            try:
                existing_families.append(CapabilityFamily(family_value))
            except ValueError:
                continue

        # Short owner replies frequently answer an exact target question. Do
        # this before generic classification: `bar.service` and email addresses
        # otherwise look like local artifacts and accidentally replace the
        # authorized capability family.
        target_reply = re.sub(
            r"^\s*(?:actually\s*[,;:]?\s*|use\s+|call\s+it\s+|set\s+it\s+to\s+)",
            "",
            response,
            flags=re.IGNORECASE,
        )
        target_reply = re.sub(
            r"\s+(?:instead|rather)\s*[.!]?\s*$",
            "",
            target_reply,
            flags=re.IGNORECASE,
        )
        target_correction = bool(
            re.search(r"\b(?:instead|rather)\b|^\s*actually\b", response, re.IGNORECASE)
            or re.match(r"^\s*(?:use|call\s+it|set\s+it\s+to)\b", response, re.IGNORECASE)
        )
        explicit_action_head = bool(
            _EVIDENCE_ACTION_HEAD_RE.match(response.strip())
            or _LOCAL_ACTION_HEAD_RE.match(response.strip())
            or _EXTERNAL_ACTION_HEAD_RE.match(response.strip())
            or _DESTRUCTIVE_ACTION_HEAD_RE.match(response.strip())
        )
        early_binding_family: CapabilityFamily | None = None
        early_binding_target = ""
        candidate_families = [
            family
            for family in existing_families
            if family in _TARGET_REQUIRED_FAMILIES
            or (
                family == CapabilityFamily.EXTERNAL_INTERACTION
                and (
                    target_correction
                    or self.capability_target_bindings.get(family.value)
                )
            )
        ]
        if (
            len(candidate_families) == 1
            and not _NEGATED_ACTION_CLAUSE_RE.search(response)
            and (target_correction or not explicit_action_head)
        ):
            family = candidate_families[0]
            existing_targets = self.capability_target_bindings.get(family.value, [])
            needs_binding = not _capability_binding_complete(
                family, existing_targets
            ) or (
                family == CapabilityFamily.AGENT_INSTANCE
                and not any(item.startswith("agent-name:") for item in existing_targets)
            )
            if target_correction or needs_binding:
                target = _bare_capability_target_binding(
                    target_reply,
                    family,
                    workspace_root=self.workspace_root,
                )
                if (
                    target
                    and family == CapabilityFamily.EXTERNAL_INTERACTION
                    and target.startswith("recipient:")
                ):
                    categories = {
                        item.split(":", 1)[0]
                        for item in existing_targets
                        if not item.startswith("operation:") and ":" in item
                    }
                    if len(categories) == 1 and "recipient" not in categories:
                        target = f"{next(iter(categories))}:{target_reply.casefold()}"
                if target:
                    early_binding_family = family
                    early_binding_target = target
        target_replacement_epoch = bool(
            target_correction and early_binding_family and early_binding_target
        )
        bare_target_clarification = bool(
            not target_replacement_epoch and early_binding_family and early_binding_target
        )
        confirmation = bool(_CONFIRMATION_RE.fullmatch(response.strip()))
        if confirmation:
            # "yes" and "go ahead" contain no action identity of their own.
            # They can confirm only the exact outstanding proposal, never
            # rescan historical request nouns such as restart/delete/push.
            authority_text = self.pending_question
            next_mode = _pending_proposal_mode(authority_text) or self.mode
        elif bare_target_clarification or target_replacement_epoch:
            authority_text = response
            next_mode = self.mode
        else:
            # A new explicit imperative carries its own authority and does not
            # depend on an earlier proposal.
            authority_text = response
            next_mode = classify_request_mode(response)
        negated = _NEGATED_ACTION_CLAUSE_RE.search(response)
        later_positive = bool(
            negated
            and _LATER_EXPLICIT_ACTION_RE.search(response[negated.end() :])
        )
        negated_current = bool(
            negated
            and not later_positive
            and (
                self.mode in {RequestMode.EXTERNAL_ACTION, RequestMode.DESTRUCTIVE}
                or re.search(
                    r"\b(?:it|this|that|anything|the\s+(?:task|change|action))\b|"
                    r"\b(?:just\s+)?explain(?:\s+instead)?\b",
                    negated.group("object") or response,
                    re.IGNORECASE,
                )
            )
        )
        if negated_current:
            next_mode = (
                RequestMode.INSPECT
                if re.search(r"\b(?:inspect|review|explain|analy[sz]e)\b", response, re.IGNORECASE)
                else RequestMode.ANSWER
            )
        explicit_revocation = bool(
            (
                _EXPLICIT_READ_ONLY_RE.search(response)
                or _NEGATIVE_CONFIRMATION_RE.fullmatch(response)
                or negated_current
            )
            and next_mode
            not in {
                RequestMode.CHANGE_LOCAL,
                RequestMode.EXTERNAL_ACTION,
                RequestMode.DESTRUCTIVE,
            }
        )
        mutating_modes = {
            RequestMode.CHANGE_LOCAL,
            RequestMode.EXTERNAL_ACTION,
            RequestMode.DESTRUCTIVE,
        }
        replacement_epoch = bool(
            not confirmation
            and not bare_target_clarification
            and not target_replacement_epoch
            and next_mode in mutating_modes
            and not _ADDITIVE_FOLLOWUP_RE.search(response)
        )
        additive_epoch = bool(
            not confirmation
            and not bare_target_clarification
            and not target_replacement_epoch
            and _ADDITIVE_FOLLOWUP_RE.search(response)
            and (
                next_mode != RequestMode.ANSWER
                or _goal_kind(response, default="")
                in {"inspect", "validation", "invariant"}
            )
        )
        rank = {
            RequestMode.ANSWER: 0,
            RequestMode.INSPECT: 1,
            RequestMode.PLAN: 1,
            RequestMode.CHANGE_LOCAL: 2,
            RequestMode.EXTERNAL_ACTION: 3,
            RequestMode.DESTRUCTIVE: 4,
        }

        def reset_completion_epoch() -> None:
            # Retain the append-only receipt audit, but make it impossible for a
            # prior target's edit/test to satisfy a newly substituted request.
            self.changed = False
            self.satisfied = False
            self.needs_verification = False
            self.verified_after_change = False
            self.external_action_satisfied = False
            self.github_clean_required = False
            self.github_clean_satisfied = False
            self.github_commit_required = False
            self.github_committed_targets.clear()
            self.github_backup_targets.clear()
            self.github_clean_targets.clear()
            self.pending_validation_targets.clear()
            self.pending_external_validation_targets.clear()
            self.unscoped_mutation_pending = False
            self.satisfied_capability_families.clear()
            self.satisfied_capability_targets.clear()
            self.local_target_bindings.clear()
            self.input_local_target_bindings.clear()
            self.observed_input_local_targets.clear()
            self.inspected_local_targets.clear()
            self.changed_local_targets.clear()
            self.validated_local_targets.clear()
            self.goal_mutation_evidence.clear()
            self.goal_relevant_mutation_evidence.clear()
            self.goal_validation_evidence.clear()
            self.goal_information_evidence.clear()
            self.goal_invariant_violations.clear()
            self.goal_failed_probes.clear()
            self.goal_last_mutation_sequence.clear()
            self.observation_sequence = 0

        new_families = _requested_capability_families(authority_text, next_mode)
        if bare_target_clarification or target_replacement_epoch:
            # A target-only reply narrows the existing family; it cannot add a
            # generic external capability merely because the current mode is
            # external.
            new_families = ()
        if explicit_revocation:
            # Evidence is monotonic; executable authority is not. A newer exact
            # owner instruction can revoke a prior mutation ceiling immediately.
            reset_completion_epoch()
            self.mode = next_mode
            self.capability_families.clear()
            self.capability_target_bindings.clear()
        elif confirmation and next_mode in mutating_modes:
            # Confirmation starts an executable epoch for the exact pending
            # proposal. Earlier read/plan receipts remain in the audit log but
            # cannot satisfy the newly authorized mutation.
            reset_completion_epoch()
            self.mode = next_mode
            self.capability_families = sorted(
                family.value for family in new_families
            )
            self.capability_target_bindings.clear()
        elif target_replacement_epoch and early_binding_family is not None:
            reset_completion_epoch()
            family_key = early_binding_family.value
            if early_binding_family == CapabilityFamily.AGENT_INSTANCE:
                self.capability_target_bindings[family_key] = [
                    item
                    for item in self.capability_target_bindings.get(family_key, [])
                    if not item.startswith("agent-name:")
                ]
            elif early_binding_family == CapabilityFamily.EXTERNAL_INTERACTION:
                prefix = early_binding_target.split(":", 1)[0] + ":"
                self.capability_target_bindings[family_key] = [
                    item
                    for item in self.capability_target_bindings.get(family_key, [])
                    if not item.startswith(prefix)
                ]
            else:
                self.capability_target_bindings.pop(family_key, None)
        elif replacement_epoch:
            # A reply to an outstanding question starts a new executable
            # authority epoch by default. "Use B instead" must replace A even
            # when B is lower-ranked or belongs to another capability family.
            # Explicit additive words (also/as well/in addition) are the only
            # way to union the old and new scope.
            reset_completion_epoch()
            self.mode = next_mode
            self.capability_families = sorted(
                family.value for family in new_families
            )
            self.capability_target_bindings.clear()
        elif rank[next_mode] > rank[self.mode]:
            self.mode = next_mode
        if next_mode in mutating_modes:
            for family in new_families:
                if family.value not in self.capability_families:
                    self.capability_families.append(family.value)
            self.capability_families.sort()
            self.external_action_satisfied = (
                self._external_capability_obligation_satisfied()
            )
        # A reply to a pending target question often consists only of a path or
        # identifier and therefore classifies as ANSWER.  It may bind an already
        # authorized family, but it must never introduce a new family merely
        # because a model mentioned one in its question.
        binding_families = list(new_families)
        if not binding_families:
            for family_value in self.capability_families:
                try:
                    binding_families.append(CapabilityFamily(family_value))
                except ValueError:
                    continue
        new_bindings = _request_capability_target_bindings(
            authority_text,
            binding_families,
            workspace_root=self.workspace_root,
        )
        if early_binding_family is not None and early_binding_target:
            new_bindings = dict(new_bindings)
            new_bindings[early_binding_family.value] = [early_binding_target]
        if not new_bindings and not explicit_action_head and not negated:
            unbound = [
                family
                for family in binding_families
                if family in _TARGET_REQUIRED_FAMILIES
                and not _capability_binding_complete(
                    family, self.capability_target_bindings.get(family.value)
                )
            ]
            if len(unbound) == 1:
                target = _bare_capability_target_binding(
                    response,
                    unbound[0],
                    workspace_root=self.workspace_root,
                )
                if target:
                    new_bindings = {unbound[0].value: [target]}
        for family, targets in new_bindings.items():
            if family not in self.capability_families:
                continue
            existing = self.capability_target_bindings.setdefault(family, [])
            for target in targets:
                if (
                    target not in existing
                    and len(existing) < _MAX_CAPABILITY_TARGETS
                ):
                    existing.append(target)
        self.raw_request = f"{self.raw_request}\n\nUSER RESPONSE:\n{response}".strip()
        if confirmation:
            self.authority_request = (
                "CONFIRMED PROPOSAL:\n"
                + confirmed_proposal[:4000]
                + "\n\nOWNER CONFIRMATION:\n"
                + response[:1000]
            ).strip()
        elif explicit_revocation or replacement_epoch:
            self.authority_request = response.strip()
        elif target_replacement_epoch and early_binding_family is not None:
            self.authority_request = (
                f"{self.authority_request}\n\nOWNER TARGET REPLACEMENT "
                f"({early_binding_family.value}):\n{target_reply}"
            ).strip()
        elif bare_target_clarification and early_binding_family is not None:
            self.authority_request = (
                f"{self.authority_request}\n\nOWNER TARGET BINDING "
                f"({early_binding_family.value}):\n{target_reply}"
            ).strip()
        else:
            self.authority_request = (
                f"{self.authority_request}\n\nUSER RESPONSE:\n{response}".strip()
            )
        self.agent_state_requested = bool(
            _AGENT_STATE_REQUEST_RE.search(
                _effect_text(self.authority_request or "")
            )
        )
        disposition = (
            "confirmation"
            if confirmation
            else "revocation"
            if explicit_revocation
            else "replacement"
            if replacement_epoch or target_replacement_epoch
            else "additive"
            if additive_epoch
            else "clarification"
        )
        if disposition in {"replacement", "revocation", "additive", "confirmation"}:
            previous_changed_targets = set(self.changed_local_targets)
            previous_validated_targets = set(self.validated_local_targets)
            previous_goal_evidence: dict[tuple[str, str], dict[str, Any]] = {}
            if disposition == "additive":
                for goal_id, description in self.goal_anchors.items():
                    key = (
                        self.goal_kinds.get(goal_id, "change"),
                        " ".join(description.split()),
                    )
                    previous_goal_evidence[key] = {
                        "mutation": list(self.goal_mutation_evidence.get(goal_id, ())),
                        "relevant": list(
                            self.goal_relevant_mutation_evidence.get(goal_id, ())
                        ),
                        "validation": list(
                            self.goal_validation_evidence.get(goal_id, ())
                        ),
                        "information": list(
                            self.goal_information_evidence.get(goal_id, ())
                        ),
                        "violations": list(
                            self.goal_invariant_violations.get(goal_id, ())
                        ),
                        "failed": dict(self.goal_failed_probes.get(goal_id, {})),
                        "last_mutation": self.goal_last_mutation_sequence.get(goal_id, 0),
                    }
                previous_sequence = self.observation_sequence
            anchors, kinds, semantic_required = _compile_goal_anchors(
                self.authority_request, self.mode
            )
            self.goal_anchors = anchors
            self.goal_kinds = kinds
            self.semantic_evidence_required = semantic_required
            self.goal_mutation_evidence.clear()
            self.goal_relevant_mutation_evidence.clear()
            self.goal_validation_evidence.clear()
            self.goal_information_evidence.clear()
            self.goal_invariant_violations.clear()
            self.goal_failed_probes.clear()
            self.goal_last_mutation_sequence.clear()
            self.observation_sequence = 0
            if disposition == "additive":
                for goal_id, description in self.goal_anchors.items():
                    prior = previous_goal_evidence.get(
                        (
                            self.goal_kinds.get(goal_id, "change"),
                            " ".join(description.split()),
                        )
                    )
                    if not prior:
                        continue
                    for field_name, key_name in (
                        ("goal_mutation_evidence", "mutation"),
                        ("goal_relevant_mutation_evidence", "relevant"),
                        ("goal_validation_evidence", "validation"),
                        ("goal_information_evidence", "information"),
                        ("goal_invariant_violations", "violations"),
                    ):
                        values = prior[key_name]
                        if values:
                            getattr(self, field_name)[goal_id] = list(values)
                    if prior["failed"]:
                        self.goal_failed_probes[goal_id] = dict(prior["failed"])
                    if prior["last_mutation"]:
                        self.goal_last_mutation_sequence[goal_id] = int(
                            prior["last_mutation"]
                        )
                self.observation_sequence = previous_sequence
            self.local_target_bindings = _request_local_target_bindings(
                self.authority_request,
                workspace_root=self.workspace_root,
            )
            self.input_local_target_bindings = (
                _request_external_input_target_bindings(
                    self.authority_request,
                    workspace_root=self.workspace_root,
                )
            )
            if disposition == "additive":
                self.changed_local_targets = [
                    target
                    for target in self.local_target_bindings
                    if target in previous_changed_targets
                ]
                self.validated_local_targets = [
                    target
                    for target in self.local_target_bindings
                    if target in previous_validated_targets
                    and target in self.changed_local_targets
                ]
        elif disposition == "clarification":
            # A bare path/identifier can narrow an existing request without
            # introducing a capability family. Recompile from the same durable
            # authority text used on restore; narrowed criteria must not reuse
            # prior completion evidence by accident.
            anchors, kinds, semantic_required = _compile_goal_anchors(
                self.authority_request, self.mode
            )
            self.goal_anchors = anchors
            self.goal_kinds = kinds
            self.semantic_evidence_required = semantic_required
            self.goal_mutation_evidence.clear()
            self.goal_relevant_mutation_evidence.clear()
            self.goal_validation_evidence.clear()
            self.goal_information_evidence.clear()
            self.goal_invariant_violations.clear()
            self.goal_failed_probes.clear()
            self.goal_last_mutation_sequence.clear()
            self.observation_sequence = 0
            derived_local = _request_local_target_bindings(
                self.authority_request,
                workspace_root=self.workspace_root,
            )
            for target in derived_local:
                if target not in self.local_target_bindings:
                    self.local_target_bindings.append(target)
            self.input_local_target_bindings = (
                _request_external_input_target_bindings(
                    self.authority_request,
                    workspace_root=self.workspace_root,
                )
            )
        if (
            self.mode == RequestMode.EXTERNAL_ACTION
            and _GITHUB_SCOPE_RE.search(self.authority_request)
            and _COMPLETE_BACKUP_SCOPE_RE.search(self.authority_request)
        ):
            self.github_clean_required = True
        if _GITHUB_SCOPE_RE.search(self.authority_request or "") and (
            self.github_clean_required
            or self.local_target_bindings
            or re.search(
                r"\bcommit\b",
                _effect_text(self.authority_request or ""),
                re.IGNORECASE,
            )
        ):
            self.github_commit_required = True
        self.state = ExecutionState.RUNNING
        self.pending_question = ""
        return disposition

    def _supported_terminal_blocker(self, message: Any) -> bool:
        """Return whether explicit blocker prose has a typed invariant receipt."""

        if not _TRUTHFUL_BLOCK_RE.search(str(message or "")):
            return False
        latest = self.results[-1] if self.results else None
        return bool(
            latest is not None
            and latest.status == ToolStatus.BLOCKED
            and not latest.retryable
            and latest.error_code in _TERMINAL_BLOCKER_CODES
        )

    def ask_user_error(self, message: Any) -> str:
        """Reject permission-seeking while allowing concrete missing input."""

        value = " ".join(str(message or "").split())
        if re.search(
            r"\b(?:should|may|can|could)\s+i\s+(?:proceed|continue|start|"
            r"go\s+ahead|do\s+it|apply|make\s+the\s+change)\b|"
            r"\bwould\s+you\s+like\s+me\s+to\b",
            value,
            re.IGNORECASE,
        ):
            return (
                "HARNESS BLOCKED QUESTION: the exact owner request already grants "
                "the classified authority. Do not ask generic permission to proceed; "
                "take the next safe evidence-producing action."
            )
        concrete_field = bool(
            re.search(
                r"\b(?:which|what|provide|specify|need)\b[^?]{0,100}\b"
                r"(?:file|path|directory|repository|repo|branch|service|process|"
                r"job|sub-agent|agent|name|id|recipient|email|account|credential|"
                r"token|role|portal|url|domain|target|format|value|choice)\b|"
                r"\b(?:file|path|directory|repository|repo|branch|service|process|"
                r"job|sub-agent|agent|name|id|recipient|email|account|credential|"
                r"token|role|portal|url|domain|target)\s+(?:should|do)\b",
                value,
                re.IGNORECASE,
            )
        )
        unresolved_target = any(
            family in _TARGET_REQUIRED_FAMILIES
            and not _capability_binding_complete(
                family, self.capability_target_bindings.get(family.value)
            )
            for family in (
                CapabilityFamily(item)
                for item in self.capability_families
                if item in {family.value for family in CapabilityFamily}
            )
        )
        agent_name_unresolved = bool(
            CapabilityFamily.AGENT_INSTANCE.value in self.capability_families
            and re.search(r"\b(?:agent\s+)?name\b", value, re.IGNORECASE)
            and not any(
                item.startswith("agent-name:")
                for item in self.capability_target_bindings.get(
                    CapabilityFamily.AGENT_INSTANCE.value, []
                )
            )
        )
        latest = self.results[-1] if self.results else None
        typed_missing_input = bool(
            latest
            and latest.status in {ToolStatus.BLOCKED, ToolStatus.FAILED}
            and latest.error_code
            in {
                "owner_authority_required",
                "confirmation_required",
                "missing_credential",
                "credential_required",
                "missing_target",
                "missing_required_input",
            }
        )
        intrinsically_unbound = bool(
            self.mutation_requested
            and not self.local_target_bindings
            and not self.capability_families
            and not self.results
        )
        if concrete_field and (
            unresolved_target
            or agent_name_unresolved
            or typed_missing_input
            or intrinsically_unbound
        ):
            return ""
        return (
            "HARNESS BLOCKED QUESTION: no typed unresolved owner input supports "
            "yielding this task. Inspect current evidence, choose a bounded reversible "
            "next step, or continue the recovery ladder. Ask only for a concrete "
            "missing target/credential/value that the harness cannot discover safely."
        )

    def completion_error(self, message: Any) -> str:
        text = str(message or "").strip()
        blocker_claim = bool(_TRUTHFUL_BLOCK_RE.search(text))
        truthful_block = self._supported_terminal_blocker(text)
        latest = self.results[-1] if self.results else None
        if blocker_claim and not truthful_block:
            return (
                "COMPLETION BLOCKED: the proposed final says the task is blocked, but no "
                "latest typed non-retryable invariant receipt supports ending the parent "
                "goal. Take a materially different safe approach, ask for missing owner "
                "input, or continue until the harness exhausts bounded recovery."
            )
        if self.mode == RequestMode.INSPECT and not any(
            result.successful and result.side_effect == SideEffect.READ_ONLY
            for result in self.results
        ) and not truthful_block:
            return (
                "COMPLETION BLOCKED: this was an inspection request, but this request has no "
                "successful read-only tool receipt. Inspect current evidence or report why "
                "inspection is blocked; do not claim a review from intent alone."
            )
        if (
            self.mode == RequestMode.INSPECT
            and self.local_target_bindings
            and not truthful_block
        ):
            missing_inspections = [
                target
                for target in self.local_target_bindings
                if target not in self.inspected_local_targets
            ]
            if missing_inspections:
                return (
                    "COMPLETION BLOCKED: the owner named exact inspection target(s), "
                    "but no successful typed read receipt is bound to: "
                    + ", ".join(missing_inspections[:8])
                    + ". Unrelated reads cannot satisfy this inspection."
                )
        if incomplete_final_response(text):
            return (
                "COMPLETION BLOCKED: the proposed final response ends with a lead-in colon, "
                "but the promised body is missing. Provide the complete user-facing answer "
                "in this final message; do not publish an introduction by itself."
            )
        if latest and latest.status in {
            ToolStatus.FAILED,
            ToolStatus.BLOCKED,
            ToolStatus.PENDING,
            ToolStatus.SKIPPED,
        } and not truthful_block:
            return (
                "COMPLETION BLOCKED: the proposed final answer is terminal, but "
                f"the latest observed tool result is '{latest.status.value}'. Report the actual "
                "receipt while continuing with a materially different verified step, or use "
                "ask_user when exact owner input/authority is genuinely required. Ordinary "
                "failure/refusal is not a terminal parent-goal blocker."
            )
        if (
            self.mode == RequestMode.EXTERNAL_ACTION
            and not self.external_action_satisfied
            and not truthful_block
        ):
            return (
                "COMPLETION BLOCKED: this request explicitly required an external action, "
                "but no successful external-mutation or explicit external no-change receipt "
                "proves it occurred. Local edits, tests, or git status cannot satisfy a "
                "requested push, upload, publication, or message."
            )
        if (
            self.mutation_requested
            and self.capability_families
            and not self.capability_obligation_satisfied
            and not truthful_block
        ):
            pending: list[str] = []
            for value in self.capability_families:
                try:
                    family = CapabilityFamily(value)
                except ValueError:
                    continue
                if self._capability_family_satisfied(family):
                    continue
                bindings = self.capability_target_bindings.get(value, [])
                satisfied_targets = self.satisfied_capability_targets.get(value, [])
                missing_targets = [
                    target
                    for target in bindings
                    if not any(
                        _capability_targets_match(target, item, family=family)
                        for item in satisfied_targets
                    )
                ]
                pending.extend(
                    f"{value}:{target}" for target in missing_targets
                )
                if not missing_targets:
                    pending.append(value)
            return (
                "COMPLETION BLOCKED: no successful receipt from the exact requested "
                "capability family/families proves the requested action occurred. "
                "Still required: "
                + ", ".join(pending)
                + ". An unrelated mutation cannot satisfy this contract."
            )
        if (
            self.github_clean_required
            and not self.github_clean_satisfied
            and not truthful_block
        ):
            return (
                "COMPLETION BLOCKED: this request required backing up all current Git "
                "changes, but no final typed github_status receipt proves the repository "
                "is clean. Commit the remaining eligible paths or report exact bounded "
                "exclusions; a commit, push, or remote verification alone is incomplete."
            )
        if self.github_commit_required and not truthful_block:
            required_repositories = self.capability_target_bindings.get(
                CapabilityFamily.GITHUB.value, []
            )
            missing_commits = [
                repository
                for repository in required_repositories
                if not any(
                    _targets_match(repository, committed)
                    for committed in self.github_committed_targets
                )
            ]
            if missing_commits:
                return (
                    "COMPLETION BLOCKED: local work must be captured by a typed "
                    "github_commit after the latest edit before its push is current. "
                    "Still uncommitted for: "
                    + ", ".join(missing_commits[:8])
                    + "."
                )
        goal_error = self.goal_completion_error()
        if goal_error and not truthful_block:
            return goal_error
        if (
            self.mutation_requested
            and self.local_target_bindings
            and not truthful_block
        ):
            missing_changes = [
                target
                for target in self.local_target_bindings
                if target not in self.changed_local_targets
            ]
            if missing_changes:
                return (
                    "COMPLETION BLOCKED: the owner named exact local target(s), but no "
                    "mutation/no-change receipt is bound to: "
                    + ", ".join(missing_changes[:8])
                    + ". Unrelated files cannot satisfy this request."
                )
            missing_validation = [
                target
                for target in self.local_target_bindings
                if target not in self.validated_local_targets
            ]
            if missing_validation:
                return (
                    "COMPLETION BLOCKED: the owner-named target(s) lack later exact "
                    "readback or a validator that names the target: "
                    + ", ".join(missing_validation[:8])
                    + ". A generic unrelated green suite is supporting evidence only."
                )
        if self.mutation_requested and not self.changed:
            if self.satisfied and claims_already_satisfied(text):
                return ""
            if truthful_block:
                return ""
            return (
                "COMPLETION BLOCKED: this request required a state change, but no receipt "
                "proves a change. If current evidence proves the state was already correct, "
                "say explicitly that no change was needed; otherwise report the blocker or act."
            )
        if self.mutation_requested and self.needs_verification and not truthful_block:
            scope = ""
            if self.pending_validation_targets:
                scope = " Pending target(s): " + ", ".join(
                    self.pending_validation_targets[:5]
                )
            return (
                "COMPLETION BLOCKED: a change was made but no later validation receipt proves "
                "the requested end state. Run a targeted test or fresh observation first."
                + scope
            )
        return ""

    def final_state(self, message: Any = "") -> ExecutionState:
        """Return the truthful terminal state after an accepted final message."""

        latest = self.results[-1] if self.results else None
        if self._supported_terminal_blocker(message):
            return ExecutionState.BLOCKED
        if latest and latest.status in {
            ToolStatus.FAILED,
            ToolStatus.BLOCKED,
            ToolStatus.PENDING,
            ToolStatus.SKIPPED,
        }:
            return ExecutionState.BLOCKED
        if self.mode == RequestMode.INSPECT and not any(
            result.successful and result.side_effect == SideEffect.READ_ONLY
            for result in self.results
        ):
            return ExecutionState.BLOCKED
        if self.mode == RequestMode.INSPECT and any(
            target not in self.inspected_local_targets
            for target in self.local_target_bindings
        ):
            return ExecutionState.BLOCKED
        if not self.mutation_requested:
            return ExecutionState.DONE
        if self.capability_families and not self.capability_obligation_satisfied:
            return ExecutionState.BLOCKED
        if (
            self.mode == RequestMode.EXTERNAL_ACTION
            and not self.external_action_satisfied
        ):
            return ExecutionState.BLOCKED
        if self.github_clean_required and not self.github_clean_satisfied:
            return ExecutionState.BLOCKED
        if self.github_commit_required and any(
            not any(
                _targets_match(repository, committed)
                for committed in self.github_committed_targets
            )
            for repository in self.capability_target_bindings.get(
                CapabilityFamily.GITHUB.value, []
            )
        ):
            return ExecutionState.BLOCKED
        if self.goal_completion_error():
            return ExecutionState.BLOCKED
        if self.mutation_requested and any(
            target not in self.changed_local_targets
            or target not in self.validated_local_targets
            for target in self.local_target_bindings
        ):
            return ExecutionState.BLOCKED
        if not self.changed:
            if self.satisfied and claims_already_satisfied(message):
                return ExecutionState.DONE
            return ExecutionState.BLOCKED
        if self.needs_verification:
            return ExecutionState.BLOCKED
        if latest and latest.status in {
            ToolStatus.FAILED,
            ToolStatus.BLOCKED,
            ToolStatus.PENDING,
            ToolStatus.SKIPPED,
        }:
            return ExecutionState.BLOCKED
        return ExecutionState.DONE

    def prompt_summary(self) -> str:
        effects = ", ".join(effect.value for effect in sorted(_ALLOWED_EFFECTS[self.mode], key=lambda e: e.value))
        verification = (
            "required after the latest mutation"
            if self.needs_verification
            else "currently satisfied or not yet applicable"
        )
        external_obligation = (
            "required and satisfied"
            if self.mode == RequestMode.EXTERNAL_ACTION
            and self.external_action_satisfied
            else "required and not yet satisfied"
            if self.mode == RequestMode.EXTERNAL_ACTION
            else "not required"
        )
        github_scope = (
            "all-change backup required and clean status observed"
            if self.github_clean_required and self.github_clean_satisfied
            else "all-change backup required; final clean github_status missing"
            if self.github_clean_required
            else "not required"
        )
        provenance = (
            "\nProvenance: untrusted collaborator handoff; only reasoning and "
            "owner-facing dialogue are authorized for this whole contract."
            if self.untrusted_collaborator_handoff
            else ""
        )
        capabilities = ", ".join(self.capability_families) or "none"
        capability_targets = "; ".join(
            f"{family}={','.join(targets[:4])}"
            for family, targets in sorted(self.capability_target_bindings.items())
            if targets
        ) or "none"
        local_targets = ", ".join(self.local_target_bindings) or "none"
        local_target_evidence = ", ".join(
            f"{target}={'verified' if target in self.validated_local_targets else 'changed' if target in self.changed_local_targets else 'pending'}"
            for target in self.local_target_bindings
        ) or "none"
        return (
            f"Request ID: {self.request_id}\n"
            f"Mode: {self.mode.value}\n"
            f"Authorized effects: {effects}\n"
            f"Completion evidence: {verification}\n"
            f"External action: {external_obligation}\n"
            f"GitHub backup scope: {github_scope}\n"
            f"Authorized capability families: {capabilities}\n"
            f"Bound capability targets: {capability_targets}"
            f"\nOwner-named local targets: {local_targets}"
            f"\nLocal target evidence: {local_target_evidence}"
            f"{provenance}"
        )


def normalize_turn_envelope(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the new four-field protocol and safely adapt legacy envelopes."""

    raw = dict(data or {})
    actions = raw.get("actions", [])
    if isinstance(actions, Mapping):
        actions = [actions] if ("tool_name" in actions or "tool" in actions) else list(actions.values())
    if not isinstance(actions, list):
        actions = []
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in {item.value for item in TurnKind}:
        # Compatibility for a legacy response while old sessions roll over.
        names = [str(a.get("tool_name") or a.get("tool") or "") for a in actions if isinstance(a, Mapping)]
        if "get_user_input" in names:
            kind = TurnKind.ASK_USER.value
        elif "task_complete" in names:
            kind = TurnKind.FINAL.value
        else:
            kind = TurnKind.TOOL_CALLS.value
    message = str(raw.get("message") or "")
    if not message:
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            params = action.get("parameters") if isinstance(action.get("parameters"), Mapping) else {}
            name = str(action.get("tool_name") or action.get("tool") or "")
            if name == "say_to_user":
                message = str(params.get("message") or "")
            elif name == "get_user_input" and not message:
                message = str(params.get("prompt") or "")
            elif name == "task_complete" and not message:
                message = str(params.get("reason") or "")
    return {
        "kind": kind,
        "intent": str(raw.get("intent") or "").strip(),
        "message": message.strip(),
        "updated_plan": raw.get("updated_plan"),
        "actions": actions,
    }


def turn_semantic_error(turn: Mapping[str, Any]) -> str:
    kind = str(turn.get("kind") or "")
    actions = turn.get("actions") if isinstance(turn.get("actions"), list) else []
    message = str(turn.get("message") or "").strip()
    if kind == TurnKind.TOOL_CALLS.value:
        if not actions:
            return "A tool_calls turn requires at least one action."
        if message:
            return "A tool_calls turn must keep message empty; observe tools before composing prose."
        return ""
    if actions:
        return f"A {kind} turn cannot include tool calls."
    if kind in {TurnKind.FINAL.value, TurnKind.ASK_USER.value, TurnKind.WAIT.value} and not message:
        return f"A {kind} turn requires a visible message."
    if kind not in {item.value for item in TurnKind}:
        return f"Unknown turn kind: {kind!r}."
    return ""


def bound_actions_for_observation(
    actions: Sequence[Mapping[str, Any]], policies: Mapping[str, ToolPolicy]
) -> tuple[list[Mapping[str, Any]], int]:
    """Batch independent reads; stop before any result-dependent later action."""

    proposed = list(actions)
    accepted: list[Mapping[str, Any]] = []
    def is_boundary(action: Mapping[str, Any]) -> bool:
        name = str(action.get("tool_name") or "").strip()
        policy = policies.get(name) or infer_tool_policy(name)
        parameters = action.get("parameters")
        effect = effective_tool_effect(
            policy, parameters if isinstance(parameters, Mapping) else {}
        )
        return bool(
            policy.observation_boundary
            and effect
            in {
                SideEffect.LOCAL_MUTATION,
                SideEffect.EXTERNAL_MUTATION,
                SideEffect.DESTRUCTIVE,
            }
        )

    for action in proposed:
        if accepted and (
            is_boundary(action)
            or any(is_boundary(accepted_action) for accepted_action in accepted)
        ):
            break
        accepted.append(action)
        if is_boundary(action):
            break
    return accepted, max(0, len(proposed) - len(accepted))
