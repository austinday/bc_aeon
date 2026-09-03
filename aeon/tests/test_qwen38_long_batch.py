"""Fail-closed release gates for Qwen long context and batch throughput."""

from aeon.scripts import benchmark_qwen38_long_batch as benchmark


def _completion(content, *, prompt_tokens=120_001, reasoning="private"):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning": reasoning,
                }
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 10,
        },
    }


def test_long_gate_requires_exact_final_content_and_exact_token_receipt(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_long_messages",
        lambda _base_url, _target: ([{"role": "user", "content": "x"}], 120_001),
    )
    monkeypatch.setattr(
        benchmark,
        "_post",
        lambda *_args, **_kwargs: _completion(benchmark.NEEDLE),
    )

    result = benchmark.run_long("http://loopback", 126_000)

    assert result["exact_answer"] is True
    assert result["passed"] is True


def test_long_gate_rejects_answer_that_only_contains_the_needle(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_long_messages",
        lambda _base_url, _target: ([{"role": "user", "content": "x"}], 120_001),
    )
    monkeypatch.setattr(
        benchmark,
        "_post",
        lambda *_args, **_kwargs: _completion(f"The key is {benchmark.NEEDLE}"),
    )

    result = benchmark.run_long("http://loopback", 126_000)

    assert result["contains_answer"] is True
    assert result["exact_answer"] is False
    assert result["passed"] is False


def test_long_gate_rejects_tokenizer_and_usage_disagreement(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "_long_messages",
        lambda _base_url, _target: ([{"role": "user", "content": "x"}], 120_001),
    )
    monkeypatch.setattr(
        benchmark,
        "_post",
        lambda *_args, **_kwargs: _completion(
            benchmark.NEEDLE,
            prompt_tokens=120_000,
        ),
    )

    assert benchmark.run_long("http://loopback", 126_000)["passed"] is False


def test_batch_gate_requires_concurrency_eight_to_beat_serial():
    summary = benchmark._batch_release_summary(
        [
            {"concurrency": 1, "aggregate_decode_tps": 100.0},
            {"concurrency": 4, "aggregate_decode_tps": 180.0},
            {"concurrency": 8, "aggregate_decode_tps": 90.0},
        ]
    )

    assert summary["best_concurrency"] == 4
    assert summary["throughput_scale_vs_serial"] == 1.8
    assert summary["concurrency_8_scale_vs_serial"] == 0.9
    assert summary["passed"] is False
