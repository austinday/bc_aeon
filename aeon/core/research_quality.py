"""Deterministic evidence gates for continuous Hugging Face scouting claims.

The model may always report a provisional lead, unknowns, blockers, or negative
search results scoped to the exact query.  Strong promotion and ecosystem-coverage
language is different: it is accepted only when this cycle's typed Hub receipts
cover the facts the harness can actually verify.

This module deliberately does not interpret licenses, predict demand, estimate a
toolchain's peak memory, or validate benchmark claims.  Because no typed receipt
currently proves those classes, it does not allow ``winner`` or ``decision-ready``
claims at all; the truthful supported label is ``provisional lead``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping


_HF_CONTEXT_RE = re.compile(r"\b(?:hugging\s*face|huggingface|hf\s+hub)\b", re.IGNORECASE)
_SCOUT_CONTEXT_RE = re.compile(
    r"\b(?:candidate|target|opportunit|gap|missing|derivative|quantiz|fine[- ]?tun|"
    r"popular|recent|model\s+lineage|optimization|speedup|scout)\w*\b",
    re.IGNORECASE,
)
_UPLOAD_SCOUT_CONTEXT_RE = re.compile(
    r"\b(?:useful|model|dataset|adapter|hugging\s*face)\s+uploads?\b",
    re.IGNORECASE,
)
_RESEARCH_CLAUSE_RE = re.compile(
    r"\b(?:candidate|opportunit|derivative|compet(?:itor|ition)|ecosystem|modality|"
    r"(?:research|model|format|open|feasible)\s+gaps?|"
    r"(?:research|model|video|image|audio|speech|asr|dataset|adapter)\s+lanes?|"
    r"model\s+(?:target|lineage|repository|repo)|"
    r"hub\s+(?:search|survey|sample|quer(?:y|ies)|result|metadata|"
    r"repository|repositories))\w*\b",
    re.IGNORECASE,
)
_STRONG_PROMOTION_RE = re.compile(r"\b(?:validated|confirmed)\b", re.IGNORECASE)
_DECISION_RE = re.compile(
    r"\b(?:decision[- ]ready|clear(?:ly)?\s+(?:best|winner)|winner(?:s)?)\b",
    re.IGNORECASE,
)
_COVERAGE_RE = re.compile(
    r"\b(?:covered|closed|no\s+feasible\s+(?:open\s+)?gap)\b",
    re.IGNORECASE,
)
_ABSENCE_RE = re.compile(
    r"\b(?:gap|absen(?:t|ce)|no\s+(?:existing|competing|alternative)|"
    r"missing\s+(?:derivative|format|version)|nothing\s+exists)\b",
    re.IGNORECASE,
)
_NEGATED_CLAIM_RE = re.compile(
    r"\b(?:unvalidated|unconfirmed|unknown|insufficient\s+evidence(?:\s+to)?|"
    r"not(?!\s+only)\b[^.;\n]{0,100}\b"
    r"(?:validated|confirmed|covered|closed|decision[- ]ready|"
    r"(?:a\s+)?(?:clear\s+)?winner)|"
    r"no\b[^.;\n]{0,80}\b(?:candidate\s+)?(?:is|was|can\s+be)?\s*"
    r"(?:validated|confirmed|covered|closed|decision[- ]ready)|"
    r"(?:cannot|can['’]?t)\s+(?:validate|confirm)|"
    r"(?:cannot|can['’]?t)\s+(?:call|declare)\b[^.;\n]{0,60}\b"
    r"(?:validated|confirmed|covered|closed|decision[- ]ready|(?:clear\s+)?winner))\b",
    re.IGNORECASE,
)
_UNPROVABLE_CLOSURE_RE = re.compile(
    r"\b(?:closed|no\s+feasible\s+(?:open\s+)?gap)\b", re.IGNORECASE
)
_COVERAGE_SCOPE_RE = re.compile(
    r"\b(?:within|among|limited\s+to)\b[^.;]{0,120}\b"
    r"(?:sample|quer(?:y|ies)|page|result|repositor(?:y|ies))\b|"
    r"\b(?:sampled|surveyed|inspected)\b[^.;]{0,100}\b"
    r"(?:quer(?:y|ies)|page|result|repositor(?:y|ies))\b",
    re.IGNORECASE,
)
_SCOPED_RECEIPT_GROUPS = (
    re.compile(
        r"(?=[^.;]*\b(?:exact\s+)?(?:hub\s+)?(?:identity|revision)\b)"
        r"(?=[^.;]*\b(?:creation|created|modification|modified)\b[^.;]{0,40}\b"
        r"(?:timestamp|metadata|date)s?\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?=[^.;]*\barchitecture\b)"
        r"(?=[^.;]*\b(?:parameter\s+(?:count|metadata)|safetensors\s+total)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?=[^.;]*\blicense[- ]tag\b)"
        r"(?=[^.;]*(?:\blicense[- ](?:file|text)\b[^.;]{0,40}\b(?:retriev|fetch|receipt)\w*\b|"
        r"\b(?:retriev|fetch)\w*\b[^.;]{0,40}\blicense[- ](?:file|text)\b))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:sampled|bounded)\b[^.;]{0,50}\b"
        r"(?:competition|competitor|alternative|derivative|search)\w*\b|"
        r"\b(?:competition|competitor|alternative|derivative)\w*\b"
        r"[^.;]{0,50}\b(?:sample|search\s+receipt)\w*\b",
        re.IGNORECASE,
    ),
)
_REPO_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"([A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95})"
    r"(?![A-Za-z0-9_./-])"
)
_COMMIT_RE = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
_HUB_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})"
)
_LICENSE_PATH_RE = re.compile(
    r"(?:^|/)(?:license(?:\.[A-Za-z0-9._-]+)?|notice(?:\.[A-Za-z0-9._-]+)?|"
    r"copying(?:\.[A-Za-z0-9._-]+)?|terms(?:\.[A-Za-z0-9._-]+)?)$",
    re.IGNORECASE,
)
_GOAL_RE = re.compile(
    r"\bGOAL:\s*\n(?P<goal>.*?)(?:\n\nThis is the same durable goal|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_MODEL_TOKENS = frozenset(
    {
        "base",
        "chat",
        "checkpoint",
        "fp8",
        "gguf",
        "instruct",
        "model",
        "models",
        "quantized",
    }
)


def _normalized(value: Any, maximum: int = 1000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:maximum]


def _campaign_goal(request: str) -> str:
    text = str(request or "")
    match = _GOAL_RE.search(text)
    return _normalized(match.group("goal") if match else text, 20_000)


def _json_document(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str) or len(raw) > 4 * 1024 * 1024:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _result_is_successful(result: Any) -> bool:
    status = getattr(result, "status", None)
    return bool(getattr(result, "successful", False)) or getattr(status, "value", status) == "ok"


def _claim_clauses(message: str) -> list[str]:
    return [
        part.strip(" -*#\t")
        for part in re.split(
            r"\n+|(?<=[.!?;])\s+|\s+\b(?:but|however)\b\s+",
            str(message or ""),
            flags=re.IGNORECASE,
        )
        if part.strip(" -*#\t")
    ]


def _candidate_tokens(repo_id: str) -> set[str]:
    basename = str(repo_id).split("/", 1)[-1]
    return {
        token
        for token in re.findall(r"[a-z0-9]+", basename.lower())
        if len(token) >= 3
        and token not in _GENERIC_MODEL_TOKENS
        and not re.fullmatch(r"v?\d+[a-z]?", token)
    }


@dataclass(frozen=True)
class _ModelEvidence:
    repo_id: str
    sha: str
    identity: bool
    timestamps: bool
    architecture: bool
    parameter_count: bool
    license_tag: bool
    relation_text: str


@dataclass(frozen=True)
class _SearchEvidence:
    key: str
    label: str
    query_text: str
    result_count: int
    result_text: str


class ResearchQualityGuard:
    """Gate strong scouting claims and retain a bounded branch ledger."""

    STATE_VERSION = 1
    MAX_BRANCHES = 128

    def __init__(self) -> None:
        self._campaign_goal_sha256 = ""
        self._branches: list[dict[str, str]] = []
        self._active = False
        self._infos: dict[str, _ModelEvidence] = {}
        self._license_commits: dict[str, set[str]] = {}
        self._searches: dict[str, _SearchEvidence] = {}

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._campaign_goal_sha256 = ""
        self._branches = []
        self._reset_cycle()

    def _reset_cycle(self) -> None:
        self._active = False
        self._infos = {}
        self._license_commits = {}
        self._searches = {}

    def begin_cycle(self, request: str) -> None:
        """Start a current-receipt epoch while retaining same-goal branches."""

        self._reset_cycle()
        text = str(request or "")
        goal_sha256 = hashlib.sha256(_campaign_goal(text).encode("utf-8")).hexdigest()
        same_goal = bool(
            self._campaign_goal_sha256
            and goal_sha256 == self._campaign_goal_sha256
        )
        if self._campaign_goal_sha256 and not same_goal:
            self._branches = []
        self._campaign_goal_sha256 = goal_sha256
        prior_hub_campaign = bool(
            same_goal
            and any(
                item.get("kind") in {"model_info", "model_search", "license_file"}
                for item in self._branches
            )
        )
        self._active = bool(
            (_HF_CONTEXT_RE.search(text) and _SCOUT_CONTEXT_RE.search(text))
            or _UPLOAD_SCOUT_CONTEXT_RE.search(text)
            or prior_hub_campaign
        )
        if not self._active:
            return

    def _add_branch(self, key: str, kind: str, label: str) -> None:
        record = {
            "key": str(key)[:64],
            "kind": str(kind)[:32],
            "label": _normalized(label, 300),
        }
        self._branches = [item for item in self._branches if item.get("key") != record["key"]]
        self._branches.append(record)
        self._branches = self._branches[-self.MAX_BRANCHES :]

    def observe_turn(self, turn: Mapping[str, Any], results: list[Any]) -> None:
        """Record only successful typed Hub receipts and their exact parameters."""

        if not self._active:
            # A goal may describe "public model uploads" without naming the Hub.
            # Executing a typed Hugging Face evidence tool is an unambiguous,
            # harness-observed way to enter this narrow research campaign.
            self._active = any(
                str(getattr(result, "tool_name", "")).startswith("huggingface_")
                and _result_is_successful(result)
                for result in results
            )
            if not self._active:
                return
        actions = {
            str(action.get("_call_id") or ""): action
            for action in list(turn.get("actions") or [])
            if isinstance(action, Mapping)
        }
        for result in results:
            if not _result_is_successful(result):
                continue
            name = str(getattr(result, "tool_name", ""))
            if name not in {
                "huggingface_model_info",
                "huggingface_model_search",
                "huggingface_repo_file",
            }:
                continue
            action = actions.get(str(getattr(result, "call_id", ""))) or {}
            params = action.get("parameters") if isinstance(action.get("parameters"), Mapping) else {}
            document = _json_document(getattr(result, "raw", None))
            if not document:
                continue
            if name == "huggingface_model_info":
                self._observe_model_info(params, document)
            elif name == "huggingface_repo_file":
                self._observe_repo_file(params, document)
            else:
                self._observe_search(params, document)

    def _observe_model_info(self, params: Mapping[str, Any], document: dict[str, Any]) -> None:
        requested = _normalized(params.get("repo_id"), 200)
        metadata = document.get("metadata") if isinstance(document.get("metadata"), Mapping) else {}
        observed = _normalized(metadata.get("id") or metadata.get("modelId"), 200)
        if (
            not _REPO_ID_RE.fullmatch(requested)
            or observed.lower() != requested.lower()
        ):
            return
        sha = _normalized(metadata.get("sha"), 64).lower()
        config = document.get("config") if isinstance(document.get("config"), Mapping) else {}
        card = document.get("card_data") if isinstance(document.get("card_data"), Mapping) else {}
        safetensors = document.get("safetensors")
        parameter_count = bool(
            isinstance(safetensors, Mapping)
            and _positive_number(safetensors.get("total"))
        )
        architecture = bool(config.get("architectures") or config.get("model_type"))
        relation_text = _normalized(
            " ".join(
                (
                    observed,
                    " ".join(str(tag) for tag in list(metadata.get("tags") or [])[:100]),
                    json.dumps(card.get("base_model"), ensure_ascii=False, default=str),
                )
            ),
            4000,
        ).lower()
        evidence = _ModelEvidence(
            repo_id=observed,
            sha=sha,
            identity=bool(_COMMIT_RE.fullmatch(sha)),
            timestamps=bool(
                _HUB_TIMESTAMP_RE.fullmatch(str(metadata.get("createdAt") or ""))
                and _HUB_TIMESTAMP_RE.fullmatch(str(metadata.get("lastModified") or ""))
            ),
            architecture=architecture,
            parameter_count=parameter_count,
            license_tag=bool(card.get("license") or card.get("license_name")),
            relation_text=relation_text,
        )
        self._infos[observed.lower()] = evidence
        key = hashlib.sha256(f"info\0{observed.lower()}\0{sha}".encode("utf-8")).hexdigest()
        self._add_branch(key, "model_info", f"exact model metadata: {observed}@{sha[:12]}")

    def _observe_repo_file(self, params: Mapping[str, Any], document: dict[str, Any]) -> None:
        repo_id = _normalized(params.get("repo_id"), 200)
        path = _normalized(params.get("path"), 512)
        commit = _normalized(document.get("repo_commit"), 64).lower()
        content = str(document.get("content") or "")
        if (
            not repo_id
            or not _LICENSE_PATH_RE.search(path)
            or not _COMMIT_RE.fullmatch(commit)
            or document.get("truncated") is True
            or len(content.strip()) < 20
        ):
            return
        self._license_commits.setdefault(repo_id.lower(), set()).add(commit)
        key = hashlib.sha256(
            f"file\0{repo_id.lower()}\0{commit}\0{path.lower()}".encode("utf-8")
        ).hexdigest()
        self._add_branch(key, "license_file", f"license text: {repo_id}@{commit[:12]}:{path}")

    def _observe_search(self, params: Mapping[str, Any], document: dict[str, Any]) -> None:
        if not isinstance(document.get("results"), list):
            return
        selected = {
            key: _normalized(params.get(key), 300)
            for key in (
                "query",
                "filter_tag",
                "author",
                "pipeline_tag",
                "next_page_url",
            )
            if params.get(key) not in {None, ""}
        }
        if not selected:
            return
        canonical = json.dumps(selected, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(f"search\0{canonical}".encode("utf-8")).hexdigest()
        try:
            result_count = max(0, int(document.get("result_count") or 0))
        except (TypeError, ValueError):
            return
        result_text = " ".join(
            _normalized(
                f"{item.get('id') or item.get('modelId') or ''} "
                f"{' '.join(str(tag) for tag in list(item.get('tags') or [])[:100])}",
                2000,
            )
            for item in document.get("results")[:50]
            if isinstance(item, Mapping)
        )
        label = ", ".join(f"{name}={value}" for name, value in selected.items())
        self._searches[key] = _SearchEvidence(
            key=key,
            label=label,
            query_text=" ".join(selected.values()).lower(),
            result_count=result_count,
            result_text=result_text.lower(),
        )
        self._add_branch(key, "model_search", f"Hub search: {label}")

    def campaign_summary(self) -> str:
        """Return bounded strategy history; never present it as current evidence."""

        if not self._branches:
            return ""
        recent = self._branches[-8:]
        lines = [
            "RESEARCH CAMPAIGN LEDGER (strategy history only; does not satisfy this "
            "cycle's evidence gate):",
            f"- {len(self._branches)} distinct Hub branch(es) explored across this goal.",
        ]
        lines.extend(f"- {item['label']}" for item in recent)
        return "\n".join(lines)[:2400]

    def _candidate_searches(self, evidence: _ModelEvidence) -> list[_SearchEvidence]:
        tokens = _candidate_tokens(evidence.repo_id)
        exact = evidence.repo_id.lower()
        matches = []
        for search in self._searches.values():
            query_tokens = set(re.findall(r"[a-z0-9]+", search.query_text))
            relevant = bool(
                exact in search.query_text
                or exact in search.result_text
                or (tokens and tokens.issubset(query_tokens))
                or (tokens and all(token in search.result_text for token in tokens))
            )
            if relevant:
                matches.append(search)
        return matches

    def _candidate_error(self, clause: str, *, decision_claim: bool) -> str:
        mentioned = [
            evidence
            for repo_id, evidence in self._infos.items()
            if re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(repo_id)}(?![A-Za-z0-9_./-])", clause, re.IGNORECASE)
        ]
        if len(mentioned) != 1:
            return (
                "RESEARCH CLAIM BLOCKED: a strong candidate conclusion must name one "
                "exact org/repository id in the same clause and have current-cycle "
                "huggingface_model_info evidence for it. Use 'provisional lead' while "
                "identity or evidence remains incomplete."
            )
        candidate = mentioned[0]
        missing: list[str] = []
        if not (candidate.identity and candidate.timestamps):
            missing.append(
                "exact identity/revision plus ISO Hub creation/modification timestamps"
            )
        if not (candidate.architecture and candidate.parameter_count):
            missing.append("architecture plus a numeric safetensors parameter total")
        license_commits = self._license_commits.get(candidate.repo_id.lower(), set())
        if not candidate.license_tag or candidate.sha not in license_commits:
            missing.append("license tag plus untruncated same-revision license text")
        searches = self._candidate_searches(candidate)
        if len({item.key for item in searches}) < 2:
            missing.append("two distinct candidate-relevant Hub competition searches")
        candidate_tokens = _candidate_tokens(candidate.repo_id)
        competitors = [
            item
            for key, item in self._infos.items()
            if key != candidate.repo_id.lower()
            and item.identity
            and (
                candidate.repo_id.lower() in item.relation_text
                or (
                    candidate_tokens
                    and all(token in item.relation_text for token in candidate_tokens)
                )
            )
        ]
        if not competitors:
            missing.append("one exact candidate-related competing repository inspection")
        if missing:
            return (
                f"RESEARCH CLAIM BLOCKED for {candidate.repo_id}: current-cycle typed "
                "receipts are missing " + "; ".join(missing) + ". Report the candidate "
                "as provisional and list the missing gates instead of promoting it."
            )
        if decision_claim:
            return (
                f"RESEARCH CLAIM BLOCKED for {candidate.repo_id}: Hub receipts now cover "
                "identity, metadata, license-file retrieval, and sampled competition, but "
                "no typed current-cycle receipt proves real hardware/toolchain feasibility, "
                "differentiated user value, or a reproducible benchmark. Call it a "
                "provisional lead, not a winner or decision-ready."
            )
        if sum(bool(pattern.search(clause)) for pattern in _SCOPED_RECEIPT_GROUPS) < len(
            _SCOPED_RECEIPT_GROUPS
        ):
            return (
                f"RESEARCH CLAIM BLOCKED for {candidate.repo_id}: 'validated' or "
                "'confirmed' must be scoped in the same clause to exact Hub "
                "identity/timestamps, architecture/parameter metadata, license-file "
                "retrieval, and a bounded competition-search sample. This validates "
                "only that receipt set, not the whole candidate."
            )
        return ""

    def _blocked_claim(self, clause: str, error: str) -> str:
        normalized = _normalized(clause, 500)
        key = hashlib.sha256(f"demotion\0{normalized.lower()}".encode("utf-8")).hexdigest()
        self._add_branch(key, "demotion", f"demoted unsupported claim: {normalized}")
        return error

    def completion_error(self, message: Any) -> str:
        """Reject unsupported strong wording while allowing provisional reporting."""

        if not self._active:
            return ""
        for clause in _claim_clauses(str(message or "")):
            unnegated = _NEGATED_CLAIM_RE.sub("", clause)
            strong_promotion = bool(_STRONG_PROMOTION_RE.search(unnegated))
            decision_claim = bool(_DECISION_RE.search(unnegated))
            coverage_claim = bool(_COVERAGE_RE.search(unnegated))
            if not (strong_promotion or decision_claim or coverage_claim):
                continue
            # Words such as "confirmed" and "covered" remain useful for
            # credential access, checksums, tests, and ordinary files. Campaign
            # activation alone must not reinterpret those unrelated clauses as
            # model-candidate promotions.
            if not (
                decision_claim
                or _REPO_ID_RE.search(clause)
                or _RESEARCH_CLAUSE_RE.search(clause)
            ):
                continue
            if (strong_promotion or decision_claim) and _ABSENCE_RE.search(clause):
                return self._blocked_claim(
                    clause,
                    "RESEARCH CLAIM BLOCKED: a bounded Hub search cannot confirm absence "
                    "or validate an open gap. State the exact query/page coverage and call "
                    "the gap provisional.",
                )
            if coverage_claim:
                if _UNPROVABLE_CLOSURE_RE.search(unnegated):
                    return self._blocked_claim(
                        clause,
                        "RESEARCH CLAIM BLOCKED: a bounded Hub survey cannot establish "
                        "that an opportunity space is closed or has no feasible open gap. "
                        "Report only the finite sampled results and remaining unknowns.",
                    )
                if not _COVERAGE_SCOPE_RE.search(clause):
                    return self._blocked_claim(
                        clause,
                        "RESEARCH CLAIM BLOCKED: 'covered', 'closed', and ecosystem-wide "
                        "no-gap conclusions require explicit finite survey scope. A bounded "
                        "Hub sample cannot establish global coverage; report exactly what "
                        "the sampled queries and repositories showed.",
                    )
                if len(self._searches) < 4 or len(self._infos) < 2:
                    return self._blocked_claim(
                        clause,
                        "RESEARCH CLAIM BLOCKED: even a scoped coverage statement requires "
                        "at least four distinct current-cycle Hub searches and two exact "
                        "repository inspections. Report the current sample as incomplete.",
                    )
            if strong_promotion or decision_claim:
                error = self._candidate_error(clause, decision_claim=decision_claim)
                if error:
                    return self._blocked_claim(clause, error)
        return ""

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "campaign_goal_sha256": self._campaign_goal_sha256,
            "explored_branches": [dict(item) for item in self._branches[-self.MAX_BRANCHES :]],
        }

    def restore_state_dict(self, state: Any) -> None:
        self.reset()
        if not isinstance(state, Mapping) or state.get("version") != self.STATE_VERSION:
            return
        digest = str(state.get("campaign_goal_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            return
        raw_branches = state.get("explored_branches")
        if not isinstance(raw_branches, list):
            return
        restored: list[dict[str, str]] = []
        for raw in raw_branches[-self.MAX_BRANCHES :]:
            if not isinstance(raw, Mapping):
                self.reset()
                return
            key = str(raw.get("key") or "")
            kind = str(raw.get("kind") or "")
            label = _normalized(raw.get("label"), 300)
            if not re.fullmatch(r"[0-9a-f]{64}", key) or not re.fullmatch(r"[a-z_]{1,32}", kind) or not label:
                self.reset()
                return
            restored.append({"key": key, "kind": kind, "label": label})
        self._campaign_goal_sha256 = digest
        self._branches = restored
