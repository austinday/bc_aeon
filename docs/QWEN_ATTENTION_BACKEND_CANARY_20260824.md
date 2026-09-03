# Qwen attention-backend canary — 2026-08-24

## Result

Do not promote FlashInfer for the current Qwen3.8 release. The exact runtime
image rejected `FLASHINFER` during engine initialization because the released
`fp8_per_token_head` KV-cache dtype is unsupported by that backend. No inference
request ran under FlashInfer, so this experiment establishes incompatibility,
not a throughput comparison.

The failed adapter-owned container was proven exited, removed by immutable ID,
and its cooperative claim was released. Fleet's failed-attempt storage journal
completed. The canary profile remains disabled and production was restored to
`TRITON_ATTN`.

## Bound experiment

- One GPU on the existing `DAY2RTX6000PRO` release placement.
- Exact model and image identities from the production capability manifest.
- 114,688-token context, K=3 native MTP, prefix caching, chunked prefill, and
  `fp8_per_token_head` KV cache held constant.
- Only `--attention-backend` changed from `TRITON_ATTN` to `FLASHINFER`.

## Triton baseline

- Long retrieval: 52,975 prompt tokens, 175 completion tokens, exact answer.
- Cold end-to-end: 23.390 seconds.
- Warm-prefix end-to-end: 6.826 seconds; 25.639 completion tokens/second.
- Structured Aeon action suite: 8/8 successful requests, median decode
  132.291 tokens/second, median total 126.914 tokens/second.

## Promotion boundary

Changing KV-cache dtype merely to admit FlashInfer is outside this canary: it
changes memory capacity and numerical behavior. A future attempt requires either
a runtime version that explicitly supports this exact KV format or a separately
authorized precision/context release gate. The production launcher continues to
reject every attention backend except `TRITON_ATTN`.
