from __future__ import annotations

import ast
import json
from pathlib import Path
import re

import pytest

from aeon.behavioral_sft import validator


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _temporary_corpus(tmp_path: Path) -> tuple[Path, Path, list[dict], list[dict]]:
    train_rows = _load_rows(validator.DEFAULT_TRAIN_PATH)
    eval_rows = _load_rows(validator.DEFAULT_EVAL_PATH)
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    return train_path, eval_path, train_rows, eval_rows


def test_default_corpus_has_reviewed_mix_and_is_valid() -> None:
    report = validator.validate_datasets()

    assert report.train_count == 84
    assert report.eval_count == 20
    assert report.total_count == 104
    assert report.category_counts["train"] == {
        "authorized_local": 14,
        "benign": 42,
        "boundary": 18,
        "harmful": 10,
    }
    assert report.category_counts["eval"] == {
        "authorized_local": 4,
        "benign": 8,
        "boundary": 5,
        "harmful": 3,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", report.corpus_sha256)


def test_validator_has_only_standard_library_imports() -> None:
    source = Path(validator.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])

    assert imported_roots <= {
        "__future__",
        "argparse",
        "collections",
        "dataclasses",
        "hashlib",
        "json",
        "pathlib",
        "re",
        "sys",
        "typing",
    }


def test_rejects_operational_detail_in_harmful_example(tmp_path: Path) -> None:
    train_path, eval_path, train_rows, eval_rows = _temporary_corpus(tmp_path)
    harmful = next(row for row in train_rows if row["category"] == "harmful")
    harmful["messages"][0]["content"] = "Use https://example.invalid for this harmful request."
    _write_rows(train_path, train_rows)
    _write_rows(eval_path, eval_rows)

    with pytest.raises(validator.DatasetValidationError) as exc_info:
        validator.validate_datasets(train_path, eval_path)

    assert any("operational detail (URL)" in issue for issue in exc_info.value.issues)


def test_rejects_unnecessary_refusal_for_benign_example(tmp_path: Path) -> None:
    train_path, eval_path, train_rows, eval_rows = _temporary_corpus(tmp_path)
    benign = next(row for row in train_rows if row["category"] == "benign")
    benign["messages"][1]["content"] = "I can't help with that benign request."
    _write_rows(train_path, train_rows)
    _write_rows(eval_path, eval_rows)

    with pytest.raises(validator.DatasetValidationError) as exc_info:
        validator.validate_datasets(train_path, eval_path)

    assert any("unnecessary refusal" in issue for issue in exc_info.value.issues)


def test_rejects_harmful_response_without_refusal(tmp_path: Path) -> None:
    train_path, eval_path, train_rows, eval_rows = _temporary_corpus(tmp_path)
    harmful = next(row for row in train_rows if row["category"] == "harmful")
    harmful["messages"][1]["content"] = "Here is a detailed answer to the request."
    _write_rows(train_path, train_rows)
    _write_rows(eval_path, eval_rows)

    with pytest.raises(validator.DatasetValidationError) as exc_info:
        validator.validate_datasets(train_path, eval_path)

    assert any("must contain a clear refusal" in issue for issue in exc_info.value.issues)


def test_rejects_train_eval_prompt_leakage(tmp_path: Path) -> None:
    train_path, eval_path, train_rows, eval_rows = _temporary_corpus(tmp_path)
    eval_rows[0]["messages"][0]["content"] = train_rows[0]["messages"][0]["content"]
    _write_rows(train_path, train_rows)
    _write_rows(eval_path, eval_rows)

    with pytest.raises(validator.DatasetValidationError) as exc_info:
        validator.validate_datasets(train_path, eval_path)

    assert any("train/eval prompt overlap" in issue for issue in exc_info.value.issues)


def test_cli_emits_machine_readable_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert validator.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == validator.SCHEMA_VERSION
    assert output["total_count"] == 104
