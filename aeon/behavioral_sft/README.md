# Aeon behavioral SFT corpus

This self-authored corpus teaches a narrow behavioral adjustment: answer
benign, bounded dual-use, and explicitly authorized local-computing requests
instead of refusing merely because they mention processes, ports, logs, security,
or model infrastructure. Clearly harmful requests retain short refusals.

The package is data and CPU-only validation code. It does not launch training,
allocate compute, select a model, change Aeon defaults, or modify a service.

## Files

- `data/train.jsonl`: 84 SFT conversations across four risk categories.
- `data/eval.jsonl`: 20 held-out conversations with disjoint prompts.
- `validator.py`: standard-library schema, balance, split-leakage, and content
  checks.

The training mix is deliberately not class-balanced: 56 of 84 examples are
ordinary or explicitly authorized local assistance, 18 are bounded dual-use
assistance, and 10 are clear-harm refusals. This keeps refusal examples present
without letting short refusals dominate a light behavioral update. The validator
requires at least 96 total unique prompts and enforces bounded category ratios.

Each JSONL row contains `schema_version`, `id`, `split`, `category`,
`expected_behavior`, sorted `safety_tags`, and a two-turn `messages` array.
Training integrations should pass only `messages` into the chat template; the
remaining fields are evaluation metadata.

Run the validator from the `bc_aeon` source root:

```bash
python -m aeon.behavioral_sft.validator
```

The validator rejects unnecessary refusals in non-harmful rows, requires a clear
and concise refusal in harmful rows, and rejects operational artifacts such as
commands, URLs, system paths, code blocks, or numbered procedures in harmful
examples. Harmful prompts are deliberately short and abstract.

Keep `eval.jsonl` out of training. Behavioral tuning should occur before model
quantization, and promotion should separately evaluate ordinary helpfulness,
false-refusal rate, clearly harmful refusal rate, and regression tests for the
unchanged multimodal and speculative-decoding components.
