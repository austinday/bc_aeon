from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeon.scripts import generate_qwen38_dflash_data as data_gen


def test_task_bank_is_unique_and_bounded() -> None:
    assert len(data_gen.TASKS) == 64
    assert len(set(data_gen.TASKS)) == len(data_gen.TASKS)
    assert len(data_gen.TASK_VARIANTS) == 8
    tasks = list(data_gen._iter_tasks(512))
    assert tasks[0][0] == 0
    assert len({task for _, task in tasks}) == 512
    assert tasks[0][1].startswith(data_gen.TASKS[0])
    assert tasks[64][1].startswith(data_gen.TASKS[0])
    assert tasks[0][1] != tasks[64][1]
    with pytest.raises(ValueError):
        list(data_gen._iter_tasks(0))
    with pytest.raises(ValueError):
        list(data_gen._iter_tasks(513))


def test_load_completed_rejects_wrong_prefix_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    good = {
        "sample_id": "sample-a",
        "prefix_sha256": "a" * 64,
        "messages": [],
    }
    path.write_text(json.dumps(good) + "\n", encoding="utf-8")
    assert set(data_gen._load_completed(path, prefix_sha256="a" * 64)) == {
        "sample-a"
    }
    with pytest.raises(RuntimeError, match="another prefix"):
        data_gen._load_completed(path, prefix_sha256="b" * 64)
    path.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate sample_id"):
        data_gen._load_completed(path, prefix_sha256="a" * 64)


def test_stream_completion_collects_reasoning_content(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def iter_lines(self, decode_unicode: bool):
            assert decode_unicode
            yield 'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"c"},"finish_reason":"stop"}]}'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}'
            yield "data: [DONE]"

    monkeypatch.setattr(data_gen.requests, "post", lambda *args, **kwargs: Response())
    result = data_gen._stream_completion(
        base_url="http://127.0.0.1:1",
        model="model",
        messages=[{"role": "user", "content": "x"}],
        max_tokens=256,
        seed=1,
    )
    assert result["reasoning_content"] == "r"
    assert result["content"] == "c"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 2
