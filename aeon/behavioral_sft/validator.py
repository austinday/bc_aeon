"""CPU-only schema and quality validation for Aeon's behavioral SFT corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "aeon-behavioral-sft-v1"
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = PACKAGE_ROOT / "data" / "train.jsonl"
DEFAULT_EVAL_PATH = PACKAGE_ROOT / "data" / "eval.jsonl"

CATEGORIES = frozenset({"benign", "boundary", "authorized_local", "harmful"})
EXPECTED_BY_CATEGORY = {
    "benign": "comply",
    "boundary": "cautious_comply",
    "authorized_local": "cautious_comply",
    "harmful": "refuse",
}
CATEGORY_SLUG = {
    "benign": "benign",
    "boundary": "boundary",
    "authorized_local": "authorized-local",
    "harmful": "harmful",
}
REQUIRED_TAG = {
    "benign": "ordinary_assistance",
    "boundary": "bounded_dual_use",
    "authorized_local": "authorized_local",
    "harmful": "clear_harm",
}
ALLOWED_TAGS = frozenset(
    {
        "authorized_local",
        "bounded_dual_use",
        "clear_harm",
        "cyber_abuse",
        "fraud",
        "ordinary_assistance",
        "physical_harm",
        "privacy_abuse",
        "privacy_preserving",
        "read_only_first",
        "reversible_change",
        "synthetic_only",
        "weapons",
    }
)
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "split",
        "category",
        "expected_behavior",
        "safety_tags",
        "messages",
    }
)
MINIMUM_PER_CATEGORY = {"train": 4, "eval": 2}
MINIMUM_SPLIT_ROWS = {"train": 80, "eval": 16}
MINIMUM_CORPUS_ROWS = 96

_ID_PATTERN = re.compile(
    r"^(?P<split>train|eval)-(?P<category>benign|boundary|authorized-local|harmful)-[0-9]{3}$"
)
_REFUSAL_MARKERS = (
    "i can't help",
    "i cannot help",
    "i won't help",
    "i can't assist",
    "i cannot assist",
    "i won't provide",
)
_CAUTION_MARKERS = (
    "back up",
    "backup",
    "confirm",
    "do not execute",
    "dry run",
    "dry-run",
    "localhost",
    "owned",
    "preview",
    "read only",
    "read-only",
    "reversible",
    "scope",
    "synthetic",
    "verify",
    "without executing",
)
_ABSTRACT_ONLY_PATTERNS = (
    ("code fence", re.compile(r"```")),
    ("URL", re.compile(r"https?://", re.IGNORECASE)),
    ("IP address", re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")),
    ("command-line flag", re.compile(r"(?<![A-Za-z0-9_])--[a-z][a-z0-9-]*", re.IGNORECASE)),
    (
        "shell command",
        re.compile(
            r"(?im)^\s*(?:sudo|curl|wget|bash|sh|python[0-9.]*|powershell|cmd(?:\.exe)?|nc|ncat)\b"
        ),
    ),
    (
        "system path",
        re.compile(r"(?<![A-Za-z0-9_])/(?:dev|etc|home|proc|tmp|usr|var)/"),
    ),
    ("numbered procedure", re.compile(r"(?m)^\s*[0-9]+[.)]\s+")),
)


class DatasetValidationError(ValueError):
    """Raised when one or more corpus validation checks fail."""

    def __init__(self, issues: Iterable[str]):
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


@dataclass(frozen=True)
class ValidationReport:
    """Deterministic summary of a successfully validated corpus."""

    train_path: Path
    eval_path: Path
    train_count: int
    eval_count: int
    category_counts: dict[str, dict[str, int]]
    corpus_sha256: str

    @property
    def total_count(self) -> int:
        return self.train_count + self.eval_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "train_path": str(self.train_path),
            "eval_path": str(self.eval_path),
            "train_count": self.train_count,
            "eval_count": self.eval_count,
            "total_count": self.total_count,
            "category_counts": self.category_counts,
            "corpus_sha256": self.corpus_sha256,
        }


def _plain_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        return None
    return value.strip()


def _normalized_text(value: str) -> str:
    value = value.casefold().replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']+", " ", value).strip()


def _word_count(value: str) -> int:
    return len(re.findall(r"\b\w+(?:['’]\w+)?\b", value, flags=re.UNICODE))


def _contains_refusal(value: str) -> bool:
    normalized = _normalized_text(value)
    return any(marker in normalized for marker in _REFUSAL_MARKERS)


def _read_jsonl(path: Path, *, expected_split: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    if not path.is_file():
        return rows, [f"{path}: not a regular file"]

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return rows, [f"{path}: cannot read UTF-8 JSONL: {exc}"]

    if not lines:
        return rows, [f"{path}: dataset is empty"]

    for line_number, raw in enumerate(lines, 1):
        location = f"{path}:{line_number}"
        if not raw.strip():
            issues.append(f"{location}: blank JSONL rows are not allowed")
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append(f"{location}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(row, dict):
            issues.append(f"{location}: row must be a JSON object")
            continue
        rows.append(row)
        _validate_record(row, expected_split=expected_split, location=location, issues=issues)
    return rows, issues


def _validate_record(
    row: dict[str, Any],
    *,
    expected_split: str,
    location: str,
    issues: list[str],
) -> None:
    fields = frozenset(row)
    if fields != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - fields)
        extra = sorted(fields - REQUIRED_FIELDS)
        issues.append(f"{location}: schema fields differ; missing={missing}, extra={extra}")

    if row.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"{location}: schema_version must be {SCHEMA_VERSION!r}")

    record_id = _plain_text(row.get("id"))
    match = _ID_PATTERN.fullmatch(record_id or "")
    if match is None:
        issues.append(f"{location}: id has an invalid format")

    split = row.get("split")
    if split != expected_split:
        issues.append(f"{location}: split must be {expected_split!r}")
    if match is not None and match.group("split") != split:
        issues.append(f"{location}: id prefix does not match split")

    category = row.get("category")
    if category not in CATEGORIES:
        issues.append(f"{location}: unsupported category {category!r}")
        category = None
    elif match is not None and match.group("category") != CATEGORY_SLUG[category]:
        issues.append(f"{location}: id category does not match category")

    expected_behavior = row.get("expected_behavior")
    if category is not None and expected_behavior != EXPECTED_BY_CATEGORY[category]:
        issues.append(
            f"{location}: expected_behavior must be {EXPECTED_BY_CATEGORY[category]!r} for {category}"
        )

    tags = row.get("safety_tags")
    valid_tags = (
        isinstance(tags, list)
        and bool(tags)
        and all(isinstance(tag, str) and tag for tag in tags)
    )
    if not valid_tags:
        issues.append(f"{location}: safety_tags must be a non-empty list of strings")
    else:
        if tags != sorted(set(tags)):
            issues.append(f"{location}: safety_tags must be unique and sorted")
        unknown = sorted(set(tags) - ALLOWED_TAGS)
        if unknown:
            issues.append(f"{location}: unsupported safety_tags {unknown}")
        if category is not None and REQUIRED_TAG[category] not in tags:
            issues.append(f"{location}: safety_tags must include {REQUIRED_TAG[category]!r}")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        issues.append(f"{location}: messages must contain exactly one user and one assistant turn")
        return

    expected_roles = ("user", "assistant")
    contents: list[str] = []
    for index, (message, role) in enumerate(zip(messages, expected_roles)):
        message_location = f"{location}:messages[{index}]"
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            issues.append(f"{message_location}: message must contain only role and content")
            contents.append("")
            continue
        if message.get("role") != role:
            issues.append(f"{message_location}: role must be {role!r}")
        content = _plain_text(message.get("content"))
        if content is None:
            issues.append(f"{message_location}: content must be non-empty plain text")
            contents.append("")
        else:
            if len(content) > (700 if role == "user" else 1_200):
                issues.append(f"{message_location}: content exceeds the bounded length")
            contents.append(content)

    user_text, assistant_text = contents
    if not user_text or not assistant_text or category is None:
        return

    if category == "harmful":
        if not _contains_refusal(assistant_text):
            issues.append(f"{location}: harmful example must contain a clear refusal")
        if _word_count(user_text) > 40:
            issues.append(f"{location}: harmful prompt must remain short and abstract")
        if _word_count(assistant_text) > 60:
            issues.append(f"{location}: harmful refusal must remain concise")
        for label, pattern in _ABSTRACT_ONLY_PATTERNS:
            if pattern.search(user_text) or pattern.search(assistant_text):
                issues.append(f"{location}: harmful example contains operational detail ({label})")
    else:
        if _contains_refusal(assistant_text):
            issues.append(f"{location}: non-harmful example contains an unnecessary refusal")
        if _word_count(assistant_text) < 12:
            issues.append(f"{location}: compliant response is too terse to be useful")
        if category in {"boundary", "authorized_local"}:
            normalized = _normalized_text(assistant_text)
            if not any(marker in normalized for marker in _CAUTION_MARKERS):
                issues.append(f"{location}: cautious compliance must state a concrete boundary")


def _stable_corpus_digest(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\n")
    return digest.hexdigest()


def validate_datasets(
    train_path: Path | str = DEFAULT_TRAIN_PATH,
    eval_path: Path | str = DEFAULT_EVAL_PATH,
) -> ValidationReport:
    """Validate both splits and return a deterministic report.

    Validation is intentionally implemented with the Python standard library so
    it cannot initialize a model, CUDA, a network client, or a fleet service.
    """

    train_path = Path(train_path)
    eval_path = Path(eval_path)
    train_rows, issues = _read_jsonl(train_path, expected_split="train")
    eval_rows, eval_issues = _read_jsonl(eval_path, expected_split="eval")
    issues.extend(eval_issues)

    all_rows = train_rows + eval_rows
    seen_ids: dict[str, str] = {}
    prompts_by_split: dict[str, set[str]] = {"train": set(), "eval": set()}
    counts: dict[str, Counter[str]] = {"train": Counter(), "eval": Counter()}

    for row in all_rows:
        record_id = row.get("id")
        split = row.get("split")
        category = row.get("category")
        if isinstance(record_id, str):
            if record_id in seen_ids:
                issues.append(f"duplicate id {record_id!r} in {seen_ids[record_id]} and {split}")
            else:
                seen_ids[record_id] = str(split)
        if split in counts and category in CATEGORIES:
            counts[split][category] += 1
        messages = row.get("messages")
        if (
            split in prompts_by_split
            and isinstance(messages, list)
            and messages
            and isinstance(messages[0], dict)
            and isinstance(messages[0].get("content"), str)
        ):
            normalized_prompt = _normalized_text(messages[0]["content"])
            if normalized_prompt in prompts_by_split[split]:
                issues.append(f"duplicate normalized {split} prompt in record {record_id!r}")
            prompts_by_split[split].add(normalized_prompt)

    overlap = prompts_by_split["train"] & prompts_by_split["eval"]
    if overlap:
        issues.append(f"train/eval prompt overlap detected ({len(overlap)} normalized prompts)")

    for split in ("train", "eval"):
        for category in sorted(CATEGORIES):
            minimum = MINIMUM_PER_CATEGORY[split]
            if counts[split][category] < minimum:
                issues.append(
                    f"{split} requires at least {minimum} {category} rows; found {counts[split][category]}"
                )

        split_count = sum(counts[split].values())
        if split_count < MINIMUM_SPLIT_ROWS[split]:
            issues.append(
                f"{split} requires at least {MINIMUM_SPLIT_ROWS[split]} rows; found {split_count}"
            )

    if len(all_rows) < MINIMUM_CORPUS_ROWS:
        issues.append(
            f"corpus requires at least {MINIMUM_CORPUS_ROWS} rows; found {len(all_rows)}"
        )

    train_count = sum(counts["train"].values())
    if train_count:
        low_risk_ratio = (
            counts["train"]["benign"] + counts["train"]["authorized_local"]
        ) / train_count
        boundary_ratio = counts["train"]["boundary"] / train_count
        harmful_ratio = counts["train"]["harmful"] / train_count
        if not 0.60 <= low_risk_ratio <= 0.72:
            issues.append(
                "train benign-plus-authorized ratio must be between 0.60 and 0.72; "
                f"found {low_risk_ratio:.3f}"
            )
        if not 0.15 <= boundary_ratio <= 0.28:
            issues.append(
                "train boundary ratio must be between 0.15 and 0.28; "
                f"found {boundary_ratio:.3f}"
            )
        if not 0.08 <= harmful_ratio <= 0.16:
            issues.append(
                "train harmful ratio must be between 0.08 and 0.16; "
                f"found {harmful_ratio:.3f}"
            )

    if issues:
        raise DatasetValidationError(issues)

    category_counts = {
        split: {category: counts[split][category] for category in sorted(CATEGORIES)}
        for split in ("train", "eval")
    }
    return ValidationReport(
        train_path=train_path.resolve(),
        eval_path=eval_path.resolve(),
        train_count=len(train_rows),
        eval_count=len(eval_rows),
        category_counts=category_counts,
        corpus_sha256=_stable_corpus_digest(all_rows),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    args = parser.parse_args(argv)
    try:
        report = validate_datasets(args.train, args.eval)
    except DatasetValidationError as exc:
        for issue in exc.issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
