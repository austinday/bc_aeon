"""Install reviewed bare-runtime guards and Qwen speed-lab compatibility fixes."""

from __future__ import annotations

import os

import vllm_uuid_sitecustomize  # noqa: F401
from vllm.triton_utils import tl, triton


_RELAXED_GREEDY_MARGINS = {
    "0": 0.0,
    "0.10": 0.10,
    "0.20": 0.20,
    "0.35": 0.35,
    "0.50": 0.50,
}


@triton.jit(do_not_specialize=["max_spec_len"])
def _relaxed_greedy_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    target_argmax_ptr,
    relaxed_accept_ptr,
    bonus_token_ids_ptr,
    max_spec_len,
):
    req_idx = tl.program_id(0)
    start_idx = (
        tl.zeros([], dtype=cu_num_draft_tokens_ptr.dtype.element_ty)
        if req_idx == 0
        else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(
                target_argmax_ptr + start_idx + pos
            ).to(tl.int32)
            relaxed = tl.load(relaxed_accept_ptr + start_idx + pos)
            accepted = draft_token_id >= 0 and (
                draft_token_id == target_argmax_id or relaxed
            )
            token_id = draft_token_id if accepted else target_argmax_id
            rejected = not accepted
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr
            + req_idx * (max_spec_len + 1)
            + num_draft_tokens,
            bonus_token_id,
        )


def _install_dspark_bf16_head_compatibility() -> None:
    """Keep the reviewed DSpark Markov head in checkpoint precision.

    The Qwen3.8 DSpark artifact deliberately excludes ``markov_head*`` from
    ModelOpt NVFP4 and stores both Markov matrices as BF16 [vocab, 256]
    tensors.  vLLM prefixes the serving module with ``model.``, so that
    artifact-side glob does not match ``model.markov_head...``.  Without this
    guard vLLM allocates the second matrix as a packed FP4 [vocab, 128]
    parameter and then rejects the valid BF16 checkpoint during weight load.

    Bypass quantization only for this tiny DSpark-only head.  The draft
    backbone retains its reviewed NVFP4/BF16 mixed-precision contract.
    """

    enabled = os.environ.get("AEON_DSPARK_BF16_HEADS", "0")
    if enabled not in {"0", "1"}:
        raise RuntimeError("AEON_DSPARK_BF16_HEADS must be exactly 0 or 1")
    if enabled == "0":
        return

    from vllm.model_executor.models import qwen3_dspark

    head_cls = qwen3_dspark.DSparkMarkovHead
    if getattr(head_cls, "_aeon_bf16_checkpoint_compatibility", False):
        return
    original_init = head_cls.__init__

    def init_bf16_head(
        self,
        vocab_size,
        draft_vocab_size,
        markov_rank,
        prefix,
        quant_config=None,
    ):
        return original_init(
            self,
            vocab_size,
            draft_vocab_size,
            markov_rank,
            prefix,
            quant_config=None,
        )

    head_cls.__init__ = init_bf16_head
    head_cls._aeon_bf16_checkpoint_compatibility = True


def _install_nvfp4_a16_checkpoint_compatibility() -> None:
    """Allow the reviewed W4A16 view of an existing NVFP4 checkpoint.

    The W4A16 artifact keeps the exact packed NVFP4 weights and scales from the
    W4A4 checkpoint, but deliberately removes activation quantization from its
    compressed-tensors contract.  vLLM then selects its weight-only NVFP4
    kernel and does not register ``input_global_scale`` parameters.  The source
    checkpoint still contains those now-unused calibration tensors, so bind
    their one exact suffix into vLLM's existing unexpected-weight allowlist.
    """

    enabled = os.environ.get("AEON_NVFP4_A16", "0")
    if enabled not in {"0", "1"}:
        raise RuntimeError("AEON_NVFP4_A16 must be exactly 0 or 1")
    if enabled == "0":
        return

    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

    suffix = ".input_global_scale"
    if suffix not in QuantizationConfig._ignore_unexpected_suffixes:
        QuantizationConfig._ignore_unexpected_suffixes += (suffix,)

    # AutoWeightsLoader applies the suffix allowlist for ordinary linear
    # modules. Fused MergedColumn/QKV modules consume their child stream in a
    # module-local loader first, where a missing parameter otherwise falls back
    # to the module object itself. Filter the same one suffix at that boundary.
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )

    for linear_cls in (MergedColumnParallelLinear, QKVParallelLinear):
        if getattr(linear_cls, "_aeon_nvfp4_a16_filter_installed", False):
            continue
        original_load = linear_cls.load_weights

        def load_weights(self, weights, _original=original_load):
            filtered = (
                (name, tensor)
                for name, tensor in weights
                if name != "input_global_scale" and not name.endswith(suffix)
            )
            yield from _original(self, filtered)

        linear_cls.load_weights = load_weights
        linear_cls._aeon_nvfp4_a16_filter_installed = True


def _install_quantized_dflash_context_projection() -> None:
    """Keep DFlash context-KV precomputation fast with packed linears.

    Upstream fuses dense per-layer KV weights into one BF16 GEMM. Compressed-
    tensors W4A16 linears expose packed weights instead. Dequantize only their
    K/V rows once, while weights are still in checkpoint layout, and preserve
    upstream's single fused context-KV GEMM. The draft's hot decoder linears stay
    W4A16/Marlin, and speculative verification remains exact.
    """

    import torch

    from vllm.model_executor.models import qwen3_dflash

    model_cls = qwen3_dflash.DFlashQwen3Model
    if getattr(model_cls, "_aeon_packed_context_kv_installed", False):
        return

    original_build = model_cls._build_context_kv_buffers

    def dense_kv_rows(attn, output_dtype):
        qkv = attn.qkv_proj
        weight = getattr(qkv, "weight", None)
        if (
            isinstance(weight, torch.Tensor)
            and weight.ndim == 2
            and weight.dtype.is_floating_point
        ):
            return weight[attn.q_size :].to(output_dtype)

        packed = getattr(qkv, "weight_packed", None)
        scale = getattr(qkv, "weight_scale", None)
        input_size = int(getattr(qkv, "input_size", 0))
        if (
            not isinstance(packed, torch.Tensor)
            or not isinstance(scale, torch.Tensor)
            or packed.ndim != 2
            or scale.ndim != 2
            or input_size <= 0
            or not 0 < attn.q_size < packed.shape[0]
            or scale.shape[0] != packed.shape[0]
            or input_size % scale.shape[1] != 0
        ):
            raise RuntimeError("unsupported packed DFlash QKV layout")

        packed_bits = 32 * packed.shape[1]
        if packed_bits % input_size:
            raise RuntimeError("packed DFlash QKV width is not integral")
        num_bits = packed_bits // input_size
        if num_bits not in (4, 8):
            raise RuntimeError("unsupported packed DFlash QKV precision")

        from compressed_tensors.compressors.pack_quantized.base import (
            unpack_from_int32,
        )

        # Packing is along the input dimension, so output rows are independent.
        # Slice before unpacking: Q is two thirds of this projection and is not
        # used by context-KV precomputation.
        kv_packed = packed.data[attn.q_size :]
        kv_scale = scale.data[attn.q_size :]
        kv_rows = int(kv_packed.shape[0])
        quantized = unpack_from_int32(
            kv_packed,
            num_bits,
            torch.Size((kv_rows, input_size)),
            packed_dim=1,
        )
        groups = int(kv_scale.shape[1])
        group_size = input_size // groups
        dense = (
            quantized.to(torch.float32).reshape(kv_rows, groups, group_size)
            * kv_scale.to(torch.float32).unsqueeze(-1)
        ).reshape(kv_rows, input_size)
        return dense.to(output_dtype)

    def build_context_kv_buffers(self, layers_attn, has_bias):
        dense_weights = [getattr(attn.qkv_proj, "weight", None) for attn in layers_attn]
        if all(
            isinstance(weight, torch.Tensor)
            and weight.ndim == 2
            and weight.dtype.is_floating_point
            for weight in dense_weights
        ):
            return original_build(self, layers_attn, has_bias)

        self._hidden_norm_weight = self.hidden_norm.weight.data
        output_dtype = self._hidden_norm_weight.dtype
        self._fused_kv_weight = torch.cat(
            [dense_kv_rows(attn, output_dtype) for attn in layers_attn], dim=0
        ).contiguous()
        if has_bias:
            self._fused_kv_bias = torch.cat(
                [attn.qkv_proj.bias[attn.q_size :] for attn in layers_attn], dim=0
            ).to(output_dtype)
        else:
            self._fused_kv_bias = None
        self._k_norm_weights = torch.stack(
            [attn.k_norm.weight.data for attn in layers_attn], dim=0
        ).contiguous()
        qwen3_dflash.logger.info(
            "Aeon speed patch: quantized DFlash context KV uses one fused BF16 GEMM"
        )

    model_cls._build_context_kv_buffers = build_context_kv_buffers
    model_cls._aeon_packed_context_kv_installed = True


def _install_relaxed_greedy_verification() -> None:
    """Optionally accept a draft token that is a near-tie under the target.

    Strict greedy speculative decoding accepts only the target's exact argmax.
    For explicitly labelled speed experiments, this quality-aware relaxation
    also accepts a draft token when its target logit is within a small fixed
    margin of that argmax. A 0.20 logit margin means the draft has at least
    exp(-0.20), or about 82%, of the argmax token's unnormalized probability.

    This changes greedy output and is therefore never enabled implicitly. The
    closed margin allowlist keeps every tested quality/speed trade-off explicit.
    """

    raw_margin = os.environ.get("AEON_RELAXED_GREEDY_LOGIT_MARGIN", "0")
    if raw_margin not in _RELAXED_GREEDY_MARGINS:
        raise RuntimeError(
            "AEON_RELAXED_GREEDY_LOGIT_MARGIN is outside the reviewed allowlist"
        )
    margin = _RELAXED_GREEDY_MARGINS[raw_margin]
    if margin == 0.0:
        return

    import torch

    from vllm.v1.sample import rejection_sampler as rejection

    if getattr(rejection, "_aeon_relaxed_greedy_installed", False):
        return
    original_rejection_sample = rejection.rejection_sample

    def relaxed_rejection_sample(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        synthetic_mode=False,
        synthetic_conditional_rates=None,
        use_fp64_gumbel=False,
    ):
        # Preserve upstream semantics for stochastic, mixed, and synthetic
        # requests. Aeon's measured agent profile is all-greedy.
        if synthetic_mode or not sampling_metadata.all_greedy:
            return original_rejection_sample(
                draft_token_ids,
                num_draft_tokens,
                max_spec_len,
                cu_num_draft_tokens,
                draft_probs,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
                synthetic_mode=synthetic_mode,
                synthetic_conditional_rates=synthetic_conditional_rates,
                use_fp64_gumbel=use_fp64_gumbel,
            )

        batch_size = len(num_draft_tokens)
        target_max, target_argmax = target_logits.max(dim=-1)
        safe_drafts = draft_token_ids.clamp_min(0).to(torch.int64)
        draft_target_logits = target_logits.gather(
            1, safe_drafts.unsqueeze(1)
        ).squeeze(1)
        relaxed_accept = (
            (draft_token_ids >= 0)
            & (draft_target_logits >= target_max - margin)
        ).contiguous()
        output_token_ids = torch.full(
            (batch_size, max_spec_len + 1),
            rejection.PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=target_logits.device,
        )
        _relaxed_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            relaxed_accept,
            bonus_token_ids,
            max_spec_len,
        )
        return output_token_ids

    rejection.rejection_sample = relaxed_rejection_sample
    rejection._aeon_relaxed_greedy_installed = True
    rejection.logger.warning(
        "Aeon speed experiment: relaxed greedy verification margin=%s", raw_margin
    )


def _install_dflash_feature_capture() -> None:
    """Capture exact quantized-target DFlash features for offline adaptation.

    vLLM already materializes the five target-layer outputs required by DFlash
    before every draft proposal.  In the one explicitly labelled extraction
    runtime, retain only prompt-position rows and save them under a token-content
    hash.  Normal serving and speed experiments never enter this path.
    """

    capture_dir_raw = os.environ.get("AEON_DFLASH_FEATURE_CAPTURE_DIR", "")
    if not capture_dir_raw:
        return
    if capture_dir_raw != "/features":
        raise RuntimeError("DFlash feature capture path is not reviewed")

    from array import array
    import hashlib
    import json
    from pathlib import Path
    import stat
    import sys

    import torch
    from safetensors.torch import save_file
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    if sys.byteorder != "little":
        raise RuntimeError("DFlash feature capture requires little-endian x86")
    root = Path(capture_dir_raw)
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
        or any(root.iterdir())
    ):
        raise RuntimeError("DFlash feature capture directory is unsafe or nonempty")

    model_sha256s = os.environ.get("AEON_SPEED_LAB_MODEL_SHA256S", "")
    dataset_sha256 = os.environ.get("AEON_DFLASH_FEATURE_DATASET_SHA256", "")
    if len(model_sha256s) != 64 or len(dataset_sha256) != 64:
        raise RuntimeError("DFlash feature capture identities are malformed")

    speculator_cls = DFlashSpeculator
    if getattr(speculator_cls, "_aeon_feature_capture_installed", False):
        return
    original_propose = speculator_cls.propose
    states: dict[str, dict[str, torch.Tensor | int]] = {}
    expected_layers = 5
    hidden_size = 5120

    def token_hash(token_ids: list[int]) -> str:
        values = array("I", token_ids)
        return hashlib.sha256(
            len(token_ids).to_bytes(8, "little") + values.tobytes()
        ).hexdigest()

    def atomic_json(path: Path, value: dict[str, object]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
        try:
            payload = (
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)

    def finalize(req_id: str, state: dict[str, torch.Tensor | int]) -> None:
        length = int(state["length"])
        tokens_tensor = state["tokens"]
        features = state["features"]
        filled = state["filled"]
        assert isinstance(tokens_tensor, torch.Tensor)
        assert isinstance(features, torch.Tensor)
        assert isinstance(filled, torch.Tensor)
        if not bool(filled.all().item()):
            return
        token_ids = [int(value) for value in tokens_tensor.tolist()]
        digest = token_hash(token_ids)
        feature_path = root / f"{digest}.safetensors"
        receipt_path = root / f"{digest}.json"
        if feature_path.exists() or receipt_path.exists():
            raise RuntimeError("DFlash feature capture destination already exists")
        temporary = root / f".{digest}.{os.getpid()}.safetensors.tmp"
        previous_umask = os.umask(0o077)
        try:
            save_file(
                {"hidden_states": features.contiguous()},
                str(temporary),
                metadata={
                    "schema_version": "aeon-qwen38-dflash-feature-v1",
                    "token_hash": digest,
                    "token_count": str(length),
                    "layer_ids": "6,20,34,48,62",
                    "model_sha256s": model_sha256s,
                    "dataset_sha256": dataset_sha256,
                },
            )
        finally:
            os.umask(previous_umask)
        temporary.chmod(0o600)
        os.replace(temporary, feature_path)
        atomic_json(
            receipt_path,
            {
                "schema_version": "aeon-qwen38-dflash-feature-v1",
                "token_hash": digest,
                "token_count": length,
                "layer_ids": [6, 20, 34, 48, 62],
                "hidden_size": hidden_size,
                "feature_width": expected_layers * hidden_size,
                "dtype": "bfloat16",
                "model_sha256s": model_sha256s,
                "dataset_sha256": dataset_sha256,
            },
        )
        del states[req_id]

    def capture(input_batch, aux_hidden_states, *, dummy_run: bool) -> None:
        if dummy_run or not aux_hidden_states or not input_batch.has_prefill:
            return
        if (
            len(aux_hidden_states) != expected_layers
            or any(
                value.ndim != 2 or value.shape[1] != hidden_size
                for value in aux_hidden_states
            )
        ):
            raise RuntimeError("DFlash target feature shape changed")
        for index, req_id in enumerate(input_batch.req_ids):
            length = int(input_batch.prefill_len_np[index])
            if length <= 0 or length > 10240:
                raise RuntimeError("DFlash capture prompt length is outside its bound")
            start = int(input_batch.query_start_loc_np[index])
            end = int(input_batch.query_start_loc_np[index + 1])
            positions = input_batch.positions[start:end].detach().to("cpu")
            keep = (positions >= 0) & (positions < length)
            if not bool(keep.any().item()):
                continue
            positions = positions[keep].to(torch.int64)
            token_rows = (
                input_batch.input_ids[start:end]
                .detach()
                .to("cpu", dtype=torch.int32)[keep]
            )
            feature_rows = torch.cat(
                [
                    value[start:end]
                    .detach()
                    .to("cpu", dtype=torch.bfloat16)[keep]
                    for value in aux_hidden_states
                ],
                dim=-1,
            )
            state = states.get(req_id)
            if state is None:
                state = {
                    "length": length,
                    "tokens": torch.full((length,), -1, dtype=torch.int32),
                    "features": torch.empty(
                        (length, expected_layers * hidden_size),
                        dtype=torch.bfloat16,
                    ),
                    "filled": torch.zeros((length,), dtype=torch.bool),
                }
                states[req_id] = state
            elif int(state["length"]) != length:
                raise RuntimeError("DFlash capture request length changed")
            stored_tokens = state["tokens"]
            stored_features = state["features"]
            stored_filled = state["filled"]
            assert isinstance(stored_tokens, torch.Tensor)
            assert isinstance(stored_features, torch.Tensor)
            assert isinstance(stored_filled, torch.Tensor)
            stored_tokens[positions] = token_rows
            stored_features[positions] = feature_rows
            stored_filled[positions] = True
            finalize(req_id, state)

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        capture(input_batch, aux_hidden_states, dummy_run=bool(dummy_run))
        return original_propose(
            self,
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp=num_tokens_across_dp,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
        )

    speculator_cls.propose = propose
    speculator_cls._aeon_feature_capture_installed = True
    from vllm.logger import init_logger

    init_logger(__name__).warning(
        "Aeon DFlash exact-target feature capture is enabled"
    )


_install_dspark_bf16_head_compatibility()
_install_nvfp4_a16_checkpoint_compatibility()
_install_quantized_dflash_context_projection()
_install_relaxed_greedy_verification()
_install_dflash_feature_capture()
