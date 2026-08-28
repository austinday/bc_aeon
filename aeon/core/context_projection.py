"""Deterministic projections for long-running agent context.

These helpers construct bounded model views without an extra summarizer-model
call and without splitting assistant/tool protocol groups.  The worker also uses
the projection boundary to retain a bounded durable suffix plus a chained digest
of omitted groups, rather than rewriting an ever-growing lifetime transcript.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


TokenCounter = Callable[[str], int]
_CHECKPOINT_PREFIX = "[AEON_CONTEXT_CHECKPOINT]\n"


def deterministic_token_estimate(text: str) -> int:
    """Return a conservative deterministic token estimate without provider code."""

    return max(1, (len(str(text)) + 3) // 4)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _fits(
    text: str, *, max_chars: int, max_tokens: int, token_counter: TokenCounter
) -> bool:
    return len(text) <= max_chars and token_counter(text) <= max_tokens


def _message_text(message: Mapping[str, Any]) -> str:
    return _canonical(message)


def _strip_reasoning(
    message: Mapping[str, Any], *, include_hidden_reasoning: bool
) -> tuple[dict[str, Any], int]:
    result = copy.deepcopy(dict(message))
    stripped = 0
    if not include_hidden_reasoning:
        for key in ("reasoning", "reasoning_content"):
            if key in result:
                result.pop(key, None)
                stripped += 1
    return result, stripped


def strip_hidden_reasoning(
    message: Mapping[str, Any], include_hidden_reasoning: bool = False
) -> dict[str, Any]:
    """Return a deep copy with provider-native hidden reasoning aliases removed."""

    return _strip_reasoning(
        message, include_hidden_reasoning=include_hidden_reasoning
    )[0]


@dataclass(frozen=True)
class HistoryProjection:
    messages: tuple[dict[str, Any], ...]
    source_messages: int
    kept_messages: int
    omitted_messages: int
    source_groups: int
    kept_groups: int
    stripped_reasoning_fields: int
    orphan_receipts: int
    repaired_assistants: int
    omitted_sha256: str
    char_cost: int
    token_cost: int


@dataclass(frozen=True)
class _HistoryGroup:
    indices: tuple[int, ...]
    messages: tuple[dict[str, Any], ...]

    @property
    def text(self) -> str:
        return "\n".join(_message_text(message) for message in self.messages)


def _history_groups(
    messages: Sequence[Mapping[str, Any]], *, include_hidden_reasoning: bool
) -> tuple[list[_HistoryGroup], int, int, int]:
    groups: list[_HistoryGroup] = []
    orphan_receipts = 0
    repaired_assistants = 0
    stripped_fields = 0
    index = 0
    while index < len(messages):
        source = messages[index]
        if not isinstance(source, Mapping):
            index += 1
            continue
        role = str(source.get("role") or "")
        if role == "tool":
            orphan_receipts += 1
            index += 1
            continue

        message, removed = _strip_reasoning(
            source, include_hidden_reasoning=include_hidden_reasoning
        )
        stripped_fields += removed
        calls = message.get("tool_calls") if role == "assistant" else None
        if not isinstance(calls, list) or not calls:
            groups.append(_HistoryGroup((index,), (message,)))
            index += 1
            continue

        expected = {
            str(call.get("id") or "")
            for call in calls
            if isinstance(call, Mapping) and str(call.get("id") or "")
        }
        group_indices = [index]
        receipts: list[dict[str, Any]] = []
        seen: set[str] = set()
        cursor = index + 1
        while cursor < len(messages):
            candidate = messages[cursor]
            if not isinstance(candidate, Mapping) or candidate.get("role") != "tool":
                break
            receipt, removed = _strip_reasoning(
                candidate, include_hidden_reasoning=include_hidden_reasoning
            )
            stripped_fields += removed
            receipt_id = str(receipt.get("tool_call_id") or "")
            group_indices.append(cursor)
            if receipt_id in expected and receipt_id not in seen:
                receipts.append(receipt)
                seen.add(receipt_id)
            else:
                orphan_receipts += 1
            cursor += 1

        if expected and seen == expected:
            groups.append(
                _HistoryGroup(tuple(group_indices), tuple([message, *receipts]))
            )
        else:
            repaired_assistants += 1
            safe = dict(message)
            safe.pop("tool_calls", None)
            if str(safe.get("content") or "").strip():
                groups.append(_HistoryGroup((index,), (safe,)))
        index = cursor
    return groups, stripped_fields, orphan_receipts, repaired_assistants


def project_history(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_chars: int = 60_000,
    max_tokens: int = 16_000,
    include_hidden_reasoning: bool = False,
    token_counter: TokenCounter = deterministic_token_estimate,
) -> HistoryProjection:
    """Retain the newest complete conversation groups inside strict budgets."""

    max_chars = max(256, int(max_chars))
    max_tokens = max(64, int(max_tokens))
    source = list(messages)
    groups, stripped, orphans, repairs = _history_groups(
        source, include_hidden_reasoning=include_hidden_reasoning
    )
    kept: list[_HistoryGroup] = []
    for group in reversed(groups):
        candidate = [group, *kept]
        text = "\n".join(item.text for item in candidate)
        if not _fits(
            text,
            max_chars=max_chars,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ):
            break
        kept = candidate

    def build_checkpoint(selected: Sequence[_HistoryGroup]) -> tuple[dict[str, Any], str]:
        kept_indices = {item for group in selected for item in group.indices}
        omitted_values = [
            dict(message)
            for source_index, message in enumerate(source)
            if isinstance(message, Mapping) and source_index not in kept_indices
        ]
        kept_bounds = sorted(kept_indices)
        payload = {
            "source_groups": len(groups),
            "kept_groups": len(selected),
            "source_messages": len(source),
            "kept_source_messages": len(kept_indices),
            "omitted_source_messages": len(source) - len(kept_indices),
            "omitted_sha256": _digest(omitted_values),
            "stripped_reasoning_fields": stripped,
            "orphan_receipts_dropped": orphans,
            "incomplete_assistants_repaired": repairs,
            "kept_source_index_first": kept_bounds[0] if kept_bounds else None,
            "kept_source_index_last": kept_bounds[-1] if kept_bounds else None,
        }
        text = _CHECKPOINT_PREFIX + _canonical(payload)
        return {"role": "system", "content": text}, payload["omitted_sha256"]

    needs_checkpoint = (
        len(kept) < len(groups) or stripped > 0 or orphans > 0 or repairs > 0
    )
    while True:
        projected = [message for group in kept for message in group.messages]
        omitted_digest = _digest([])
        if needs_checkpoint:
            checkpoint, omitted_digest = build_checkpoint(kept)
            projected.insert(0, checkpoint)
        rendered = "\n".join(_message_text(message) for message in projected)
        if _fits(
            rendered,
            max_chars=max_chars,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ):
            break
        if kept:
            kept.pop(0)
            needs_checkpoint = True
            continue
        # This only applies to unrealistically tiny caller budgets.  Preserve a
        # deterministic marker rather than returning an over-budget projection.
        minimal = _CHECKPOINT_PREFIX + _canonical(
            {"omitted_source_messages": len(source), "omitted_sha256": _digest(source)}
        )
        while minimal and not _fits(
            _message_text({"role": "system", "content": minimal}),
            max_chars=max_chars,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ):
            minimal = minimal[:-16]
        projected = [{"role": "system", "content": minimal}]
        rendered = "\n".join(_message_text(message) for message in projected)
        omitted_digest = _digest(source)
        break

    kept_indices = {item for group in kept for item in group.indices}
    return HistoryProjection(
        messages=tuple(projected),
        source_messages=len(source),
        kept_messages=len(projected),
        omitted_messages=len(source) - len(kept_indices),
        source_groups=len(groups),
        kept_groups=len(kept),
        stripped_reasoning_fields=stripped,
        orphan_receipts=orphans,
        repaired_assistants=repairs,
        omitted_sha256=omitted_digest,
        char_cost=len(rendered),
        token_cost=token_counter(rendered),
    )


@dataclass(frozen=True)
class ActionLogProjection:
    text: str
    source_entries: int
    kept_entries: int
    omitted_entries: int
    omitted_sha256: str
    collapsed_repeats: int
    char_cost: int
    token_cost: int


def _action_key(entry: str) -> tuple[str, str]:
    actions = re.search(r"^- Actions:\s*(.*)$", entry, re.MULTILINE)
    result = re.search(
        r"^-(?: Result| Receipts):\s*(.*?)(?=\n- |\Z)", entry, re.MULTILINE | re.DOTALL
    )
    normalize = lambda value: re.sub(r"\s+", " ", value or "").strip()
    return (
        normalize(actions.group(1) if actions else ""),
        normalize(result.group(1) if result else ""),
    )


def _collapse_actions(entries: Sequence[str]) -> tuple[list[tuple[str, int]], int]:
    collapsed: list[tuple[str, int]] = []
    repeats = 0
    index = 0
    while index < len(entries):
        key = _action_key(entries[index])
        cursor = index + 1
        while cursor < len(entries) and key != ("", "") and _action_key(entries[cursor]) == key:
            cursor += 1
        count = cursor - index
        repeats += max(0, count - 1)
        text = entries[index]
        if count > 1:
            text = text.rstrip() + f"\n- HARNESS NOTE: equivalent action and result repeated {count} times."
        collapsed.append((text, count))
        index = cursor
    return collapsed, repeats


def project_action_log(
    entries: Sequence[str],
    *,
    max_chars: int = 12_000,
    max_tokens: int = 3_000,
    recent_entries: int = 6,
    token_counter: TokenCounter = deterministic_token_estimate,
) -> ActionLogProjection:
    """Build a digest-bound recent action view without mutating the durable log."""

    source = [str(entry) for entry in entries]
    if not source:
        text = "(No actions taken yet.)"
        return ActionLogProjection(text, 0, 0, 0, _digest([]), 0, len(text), token_counter(text))
    max_chars = max(256, int(max_chars))
    max_tokens = max(64, int(max_tokens))
    collapsed, repeats = _collapse_actions(source)
    full = "\n\n".join(text for text, _count in collapsed)
    if _fits(full, max_chars=max_chars, max_tokens=max_tokens, token_counter=token_counter):
        return ActionLogProjection(
            full, len(source), len(source), 0, _digest([]), repeats,
            len(full), token_counter(full),
        )

    selected = collapsed[-max(1, int(recent_entries)) :]
    while True:
        kept_count = sum(count for _text, count in selected)
        omitted_count = len(source) - kept_count
        omitted = source[:omitted_count]
        tail_key = _action_key(omitted[-1]) if omitted else ("", "")
        checkpoint = "[AEON_ACTION_CHECKPOINT]\n" + _canonical(
            {
                "source_entries": len(source),
                "kept_recent_entries": kept_count,
                "omitted_entries": omitted_count,
                "omitted_sha256": _digest(omitted),
                "collapsed_repeats": repeats,
                "last_omitted_actions": tail_key[0][:240],
                "last_omitted_result": tail_key[1][:240],
            }
        )
        text = "\n\n".join([checkpoint, *[item[0] for item in selected]])
        if _fits(
            text,
            max_chars=max_chars,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ):
            break
        if selected:
            selected.pop(0)
            continue
        while checkpoint and not _fits(
            checkpoint,
            max_chars=max_chars,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ):
            checkpoint = checkpoint[:-16]
        text = checkpoint
        break
    kept_count = sum(count for _text, count in selected)
    omitted_count = len(source) - kept_count
    return ActionLogProjection(
        text=text,
        source_entries=len(source),
        kept_entries=kept_count,
        omitted_entries=omitted_count,
        omitted_sha256=_digest(source[:omitted_count]),
        collapsed_repeats=repeats,
        char_cost=len(text),
        token_cost=token_counter(text),
    )


@dataclass(frozen=True)
class OpenFileOmission:
    path: str
    char_count: int
    sha256: str
    recency_rank: int
    reason: str


@dataclass(frozen=True)
class OpenFilesProjection:
    text: str
    selected_paths: tuple[str, ...]
    omitted: tuple[OpenFileOmission, ...]
    manifest_sha256: str
    char_cost: int
    token_cost: int


def project_open_files(
    open_files: Mapping[str, str],
    access_order: Sequence[str],
    *,
    max_chars: int = 60_000,
    max_tokens: int = 15_000,
    max_files: int | None = None,
    token_counter: TokenCounter = deterministic_token_estimate,
) -> OpenFilesProjection:
    """Select newest complete file snapshots and disclose every omission."""

    values = {str(path): str(content) for path, content in open_files.items()}
    if not values:
        text = "No files currently open."
        return OpenFilesProjection(text, (), (), _digest([]), len(text), token_counter(text))
    max_chars = max(512, int(max_chars))
    max_tokens = max(128, int(max_tokens))
    ordered: list[str] = []
    for path in reversed([str(item) for item in access_order]):
        if path in values and path not in ordered:
            ordered.append(path)
    for path in reversed(list(values)):
        if path not in ordered:
            ordered.append(path)

    selected: list[str] = []
    omissions: list[OpenFileOmission] = []
    blocks: dict[str, str] = {
        path: f"--- FILE: {path} ---\n{values[path]}\n--- END FILE: {path} ---"
        for path in ordered
    }
    for rank, path in enumerate(ordered, start=1):
        if max_files is not None and len(selected) >= max(0, int(max_files)):
            reason = "max_files"
        else:
            candidate = "\n\n".join(blocks[item] for item in [*selected, path])
            reason = "" if _fits(
                candidate,
                max_chars=max_chars,
                max_tokens=max_tokens,
                token_counter=token_counter,
            ) else "budget"
        if reason:
            omissions.append(
                OpenFileOmission(
                    path=path,
                    char_count=len(values[path]),
                    sha256=hashlib.sha256(values[path].encode("utf-8")).hexdigest(),
                    recency_rank=rank,
                    reason=reason,
                )
            )
        else:
            selected.append(path)

    manifest_digest = _digest(
        [
            {
                "path": path,
                "chars": len(values[path]),
                "sha256": hashlib.sha256(values[path].encode("utf-8")).hexdigest(),
            }
            for path in ordered
        ]
    )

    def render() -> str:
        header = (
            f"{len(selected)} complete file snapshot(s) selected by recency; "
            f"manifest_sha256={manifest_digest}."
        )
        metadata = [
            {
                "path": item.path,
                "chars": item.char_count,
                "sha256": item.sha256,
                "recency_rank": item.recency_rank,
                "reason": item.reason,
            }
            for item in omissions
        ]
        omitted_text = (
            "\nOMITTED OPEN FILES (use open_file when needed):\n" + _canonical(metadata)
            if metadata
            else ""
        )
        return "\n\n".join([header + omitted_text, *[blocks[path] for path in selected]])

    text = render()
    while selected and not _fits(
        text,
        max_chars=max_chars,
        max_tokens=max_tokens,
        token_counter=token_counter,
    ):
        path = selected.pop()
        omissions.append(
            OpenFileOmission(
                path=path,
                char_count=len(values[path]),
                sha256=hashlib.sha256(values[path].encode("utf-8")).hexdigest(),
                recency_rank=ordered.index(path) + 1,
                reason="budget",
            )
        )
        omissions.sort(key=lambda item: item.recency_rank)
        text = render()
    if not _fits(
        text,
        max_chars=max_chars,
        max_tokens=max_tokens,
        token_counter=token_counter,
    ):
        compact = (
            "No complete open file snapshot fits the context budget. "
            f"manifest_sha256={manifest_digest}; omitted_files={len(values)}."
        )
        text = compact
    return OpenFilesProjection(
        text=text,
        selected_paths=tuple(selected),
        omitted=tuple(sorted(omissions, key=lambda item: item.recency_rank)),
        manifest_sha256=manifest_digest,
        char_cost=len(text),
        token_cost=token_counter(text),
    )


__all__ = (
    "ActionLogProjection",
    "HistoryProjection",
    "OpenFileOmission",
    "OpenFilesProjection",
    "deterministic_token_estimate",
    "project_action_log",
    "project_history",
    "project_open_files",
    "strip_hidden_reasoning",
)
