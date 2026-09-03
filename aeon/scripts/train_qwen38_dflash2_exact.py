#!/usr/bin/env python3
"""Warm-start and adapt DFlash2 to the exact Aeon Qwen3.8 target."""

from __future__ import annotations

import gc
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from aeon.core.dflash_adaptation import (
    ADAPTATION_MODES,
    EXPECTED_TOTAL_PARAMETERS,
    EXPECTED_TRAINABLE_PARAMETERS,
    FULL_ADAPTATION_MODE,
    sha256_file,
    trainable_parameter,
    validate_draft_config,
)
from aeon.core.dflash_dpace import dpace_weighted_loss
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.loss.dllm_loss import DLLMLossOutput
from nemo_automodel.components.speculative.dflash.core import (
    _to_full_tensor,
    compute_acceptance_stats,
)
from nemo_automodel.components.speculative.dflash.dflash2_core import (
    DFlash2StepMetrics,
    DFlash2TrainerModule,
)
from nemo_automodel.recipes.llm._spec_train_utils import (
    make_warmup_cosine_schedule,
)
from nemo_automodel.recipes.llm.train_dflash2 import TrainDFlash2Recipe
from nemo_automodel.components.speculative.dflash.registry import (
    DFLASH_DRAFT_REGISTRY,
)


logger = logging.getLogger(__name__)
_GPU_UUID_RE = re.compile(
    r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
)
_CLAIM_RE = re.compile(r"^gc-[A-Za-z0-9._:-]{8,200}$")
_RUNTIME_RE = re.compile(r"^fr-[a-f0-9]{32}$")
_TARGET_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
_STANDARD_OBJECTIVE = "dflash-decay-v1"
_DPACE_OBJECTIVE = "dpace-v1"
_TRAINING_OBJECTIVES = frozenset({_STANDARD_OBJECTIVE, _DPACE_OBJECTIVE})


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _enforce_gpu_cap() -> dict[str, Any]:
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    claim = os.environ.get("GPU_AGENT_CLAIM_ID", "")
    runtime_id = os.environ.get("AEON_DFLASH_RUNTIME_ID", "")
    raw_limit = os.environ.get("GPU_MEM_LIMIT_GB", "")
    if _GPU_UUID_RE.fullmatch(selector) is None:
        raise RuntimeError("training requires one UUID-valued CUDA selector")
    if _CLAIM_RE.fullmatch(claim) is None:
        raise RuntimeError("training requires an exact Fleet claim tag")
    if _RUNTIME_RE.fullmatch(runtime_id) is None:
        raise RuntimeError("training runtime identity is malformed")
    try:
        limit_gib = float(raw_limit)
    except ValueError as exc:
        raise RuntimeError("GPU_MEM_LIMIT_GB is malformed") from exc
    if not 64.0 <= limit_gib <= 88.0:
        raise RuntimeError("training GPU memory cap is outside the reviewed bound")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the leased training process must see exactly one GPU")
    torch.cuda.set_device(0)
    total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    limit_bytes = int(limit_gib * 1024**3)
    reserve_bytes = total_bytes - limit_bytes
    if reserve_bytes < 6 * 1024**3:
        raise RuntimeError("training cap does not preserve six GiB of GPU headroom")
    fraction = limit_bytes / total_bytes
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {
        "claim_id": claim,
        "gpu_uuid": selector,
        "runtime_id": runtime_id,
        "limit_gib": limit_gib,
        "physical_total_bytes": total_bytes,
        "allocator_fraction": fraction,
        "reserved_headroom_bytes": reserve_bytes,
    }


def _normalize_latest_checkpoint(trainer: Any) -> str:
    checkpoint_root = Path(str(trainer.checkpoint_config.checkpoint_dir))
    latest = checkpoint_root / "LATEST"
    if not latest.is_symlink():
        raise RuntimeError("final DFlash2 checkpoint pointer is absent or not a symlink")
    target = os.readlink(latest)
    if re.fullmatch(r"epoch_[0-9]+_step_[0-9]+", target) is None:
        raise RuntimeError("final DFlash2 checkpoint pointer is malformed")
    target_path = checkpoint_root / target
    metadata = target_path.lstat()
    if not target_path.is_dir() or metadata.st_uid != os.geteuid():
        raise RuntimeError("final DFlash2 checkpoint target is unsafe")
    latest.unlink()
    receipt = checkpoint_root / "LATEST.txt"
    descriptor = os.open(
        receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        os.write(descriptor, f"{target}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def _metric_receipt(trainer: Any) -> dict[str, float]:
    metrics = getattr(trainer, "_last_dflash2_metrics", None)
    if metrics is None:
        raise RuntimeError("DFlash2 training completed without final metrics")
    names = (
        "loss",
        "accuracy",
        "accept_len",
        "base_loss",
        "selector_loss",
        "base_accuracy",
        "base_accept_len",
        "candidate_recall",
    )
    return {name: float(getattr(metrics, name).detach().item()) for name in names}


class ExactDPACEDFlash2TrainerModule(DFlash2TrainerModule):
    """DFlash2 whose backbone and selector optimize accepted-prefix length."""

    def __init__(self, *args: Any, dpace_alpha: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError("D-PACE alpha must be in [0, 1]")
        self.dpace_alpha = float(dpace_alpha)

    def _base_dpace_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[DLLMLossOutput, torch.Tensor]:
        vocabulary = logits.shape[-1]
        token_nll = F.cross_entropy(
            logits.reshape(-1, vocabulary),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        loss, denominator = dpace_weighted_loss(
            token_nll,
            mask,
            alpha=self.dpace_alpha,
        )
        correct = logits.argmax(dim=-1) == targets
        correct_per_position = (
            correct.to(mask.dtype) * mask
        ).sum(dim=(0, 1))
        count_per_position = mask.sum(dim=(0, 1))
        return (
            DLLMLossOutput(
                total_loss=loss,
                dllm_loss=loss.detach().clone(),
                draft_correct_per_pos=correct_per_position,
                draft_count_per_pos=count_per_position,
            ),
            denominator,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
    ) -> DFlash2StepMetrics:
        """Run DFlash2 with D-PACE on both decode-time decision stages."""
        batch_size, sequence_length = input_ids.shape
        (
            anchor_positions,
            block_keep_mask,
            noise_embedding,
            full_position_ids,
            dflash_attention_mask,
            _,
        ) = self._prepare_block_inputs(
            input_ids,
            loss_mask,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )
        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attention_mask,
        )
        logits = _to_full_tensor(self.lm_head(output_hidden))

        block_count = anchor_positions.size(1)
        block_size = self.block_size
        _, target_ids, block_mask = self._build_block_targets(
            input_ids,
            loss_mask,
            anchor_positions,
            block_keep_mask,
            sequence_length,
            doc_remaining=doc_remaining,
        )
        pred_hidden = output_hidden.view(
            batch_size, block_count, block_size, -1
        )[:, :, 1:, :]
        pred_logits = logits.view(
            batch_size, block_count, block_size, -1
        )[:, :, 1:, :]
        pred_targets = target_ids[:, :, 1:]
        pred_mask = block_mask[:, :, 1:]

        base_loss_out, loss_denominator = self._base_dpace_loss(
            pred_logits,
            pred_targets,
            pred_mask,
        )
        scores, candidate_ids, target_index, has_target = self._selector_scores(
            pred_hidden,
            pred_logits,
            target_ids,
        )
        selector_nll = F.cross_entropy(
            scores.reshape(-1, scores.shape[-1]).float(),
            target_index.reshape(-1),
            reduction="none",
        ).view_as(pred_mask)
        # A missing true token has zero selector confidence. Keep that event in
        # the accepted-prefix confidence product, but do not train the arbitrary
        # slot-zero label used solely to keep cross-entropy tensor shapes closed.
        selector_confidence_nll = torch.where(
            has_target,
            selector_nll.detach(),
            torch.full_like(selector_nll, 100.0),
        )
        selector_loss, selector_denominator = dpace_weighted_loss(
            selector_confidence_nll + (selector_nll - selector_nll.detach()),
            pred_mask,
            alpha=self.dpace_alpha,
            supervised_mask=pred_mask * has_target.to(pred_mask.dtype),
        )
        if not torch.equal(loss_denominator, selector_denominator):
            raise RuntimeError("D-PACE backbone and selector denominators differ")
        loss = base_loss_out.total_loss + self.selector_loss_weight * selector_loss

        with torch.no_grad():
            evaluation_mask = pred_mask.bool()
            valid_tokens = pred_mask.sum()
            selected_ids = candidate_ids.gather(
                -1, scores.argmax(dim=-1, keepdim=True)
            ).squeeze(-1)
            correct_tokens = (
                (selected_ids == pred_targets) & evaluation_mask
            ).sum()
            accept_len, accept_len_sum, valid_blocks = compute_acceptance_stats(
                selected_ids,
                pred_targets,
                evaluation_mask,
            )
            base_ids = pred_logits.argmax(dim=-1)
            base_correct_tokens = (
                (base_ids == pred_targets) & evaluation_mask
            ).sum()
            base_accept_len, base_accept_len_sum, _ = compute_acceptance_stats(
                base_ids,
                pred_targets,
                evaluation_mask,
            )
            denominator = valid_tokens.clamp_min(1)
            selector_mask = pred_mask * has_target.to(pred_mask.dtype)

        return DFlash2StepMetrics(
            loss=loss,
            loss_weight=loss_denominator,
            accuracy=(correct_tokens / denominator).detach(),
            valid_tokens=valid_tokens.detach(),
            correct_tokens=correct_tokens.detach(),
            accept_len=accept_len.detach(),
            accept_len_sum=accept_len_sum.detach(),
            valid_blocks=valid_blocks.detach(),
            base_loss=base_loss_out.total_loss.detach(),
            selector_loss=selector_loss.detach(),
            base_accuracy=(base_correct_tokens / denominator).detach(),
            base_correct_tokens=base_correct_tokens.detach(),
            base_accept_len=base_accept_len.detach(),
            base_accept_len_sum=base_accept_len_sum.detach(),
            candidate_recall=(selector_mask.sum() / denominator).detach(),
        )


class ExactTargetDFlash2Recipe(TrainDFlash2Recipe):
    """DFlash2 with a hash-bound warm start and reviewed optimizer scope."""

    def _build_qwen3_draft_config(
        self,
        recipe_cfg: Any,
        *,
        target_text_config: Any,
        draft_cls: type[torch.nn.Module],
        draft_num_hidden_layers: int,
        num_target_layers: int,
        target_layer_ids: list[int],
        attention_backend: str,
    ) -> Qwen3Config:
        del target_text_config
        draft_dir = Path(str(recipe_cfg.warm_start_draft_path)).resolve(strict=True)
        config_path = draft_dir / "config.json"
        if sha256_file(config_path) != recipe_cfg.warm_start_config_sha256:
            raise RuntimeError("warm-start DFlash config hash mismatch before construction")
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        validate_draft_config(raw, label="construction source")
        expected_recipe = {
            "draft_num_attention_heads": 32,
            "draft_num_key_value_heads": 8,
            "draft_head_dim": 128,
            "draft_sliding_window": 2048,
        }
        if (
            draft_cls.__name__ != "Qwen3DFlash2DraftModel"
            or draft_num_hidden_layers != 5
            or num_target_layers != 64
            or target_layer_ids != [5, 19, 33, 47, 61]
            or attention_backend != "flex_attention"
            or self.block_size != 8
            or self.mask_token_id != 248070
            or any(recipe_cfg.get(key) != value for key, value in expected_recipe.items())
        ):
            raise RuntimeError("constructed DFlash2 recipe differs from the reviewed warm start")
        config = Qwen3Config.from_dict(raw)
        # The training class is instantiated directly, while current vLLM selects
        # its serving implementation from this exact architecture string.
        config.architectures = ["DFlash2DraftModel"]
        config._attn_implementation = attention_backend
        return config

    def _build_trainer_module(
        self,
        attention_backend: str,
        recipe_cfg: Any,
    ) -> nn.Module:
        objective = str(recipe_cfg.get("training_objective", _STANDARD_OBJECTIVE))
        if objective == _STANDARD_OBJECTIVE:
            return super()._build_trainer_module(attention_backend, recipe_cfg)
        if objective != _DPACE_OBJECTIVE:
            raise RuntimeError("the requested DFlash2 training objective is not reviewed")
        return ExactDPACEDFlash2TrainerModule(
            draft_model=self.draft_model,
            target_lm_head=self.target_model.get_output_embeddings(),
            target_embed_tokens=self.target_model.get_input_embeddings(),
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend=attention_backend,
            num_anchors=int(recipe_cfg.get("num_anchors", 512)),
            loss_decay_gamma=recipe_cfg.get("loss_decay_gamma", 7.0),
            selector_loss_weight=float(recipe_cfg.get("selector_loss_weight", 1.0)),
            sliding_window=self.draft_sliding_window,
            dpace_alpha=float(recipe_cfg.get("dpace_alpha", 0.5)),
        )

    def setup(self) -> None:
        recipe_cfg = self.cfg.recipe_args
        adaptation_mode = recipe_cfg.get("adaptation_mode")
        if adaptation_mode not in ADAPTATION_MODES:
            raise RuntimeError("the requested DFlash adaptation mode is not reviewed")
        self._adaptation_mode = str(adaptation_mode)
        training_objective = str(
            recipe_cfg.get("training_objective", _STANDARD_OBJECTIVE)
        )
        if training_objective not in _TRAINING_OBJECTIVES:
            raise RuntimeError("the requested DFlash2 training objective is not reviewed")
        self._training_objective = training_objective
        self._dpace_alpha = (
            float(recipe_cfg.get("dpace_alpha", 0.5))
            if training_objective == _DPACE_OBJECTIVE
            else None
        )
        if self.cfg.get("checkpoint.restore_from", None) is not None:
            raise RuntimeError("warm-start adaptation cannot also restore an optimizer checkpoint")
        # The pinned NeMo registry currently assigns Qwen3.5 a generic external
        # builder that ignores recipe-level draft head/window overrides. Route
        # this one exact architecture through the recipe's inline builder, which
        # above reconstructs the hash-bound published DFlash2 architecture.
        original_spec = DFLASH_DRAFT_REGISTRY.get(_TARGET_ARCHITECTURE)
        if (
            original_spec is None
            or original_spec.draft2_cls is None
            or original_spec.build_draft_config is None
        ):
            raise RuntimeError("the pinned NeMo Qwen3.5 DFlash registry changed")
        patched_spec = replace(original_spec, build_draft_config=None)
        DFLASH_DRAFT_REGISTRY[_TARGET_ARCHITECTURE] = patched_spec
        try:
            super().setup()
        finally:
            DFLASH_DRAFT_REGISTRY[_TARGET_ARCHITECTURE] = original_spec
        self._load_reviewed_warm_start()
        self._install_adaptation_optimizer()
        self._last_dflash2_metrics = None

    def _load_reviewed_warm_start(self) -> None:
        recipe_cfg = self.cfg.recipe_args
        draft_dir = Path(str(recipe_cfg.warm_start_draft_path)).resolve(strict=True)
        model_path = draft_dir / "model.safetensors"
        config_path = draft_dir / "config.json"
        if not model_path.is_file() or not config_path.is_file():
            raise RuntimeError("warm-start DFlash artifact is incomplete")
        model_sha256 = sha256_file(model_path)
        config_sha256 = sha256_file(config_path)
        if model_sha256 != recipe_cfg.warm_start_model_sha256:
            raise RuntimeError("warm-start DFlash model hash mismatch")
        if config_sha256 != recipe_cfg.warm_start_config_sha256:
            raise RuntimeError("warm-start DFlash config hash mismatch")
        warm_config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_draft_config(warm_config, label="warm-start")
        validate_draft_config(self.draft_model.config.to_dict(), label="constructed")

        state = self.draft_model.state_dict()
        with safe_open(model_path, framework="pt", device="cpu") as source:
            source_keys = set(source.keys())
            state_keys = set(state)
            if source_keys != state_keys:
                missing = sorted(state_keys - source_keys)
                unexpected = sorted(source_keys - state_keys)
                raise RuntimeError(
                    "warm-start state keys differ from the constructed draft: "
                    f"missing={missing[:8]} unexpected={unexpected[:8]}"
                )
            with torch.no_grad():
                for name in sorted(state):
                    destination = state[name]
                    shape = tuple(source.get_slice(name).get_shape())
                    if shape != tuple(destination.shape):
                        raise RuntimeError(f"warm-start tensor shape mismatch: {name}")
                    value = source.get_tensor(name)
                    if value.dtype != destination.dtype:
                        raise RuntimeError(f"warm-start tensor dtype mismatch: {name}")
                    destination.copy_(value.to(device=destination.device))
                    del value
        del state
        gc.collect()
        self._adaptation_receipt = {
            "adaptation_mode": self._adaptation_mode,
            "training_objective": self._training_objective,
            "dpace_alpha": self._dpace_alpha,
            "warm_start_model_sha256": model_sha256,
            "warm_start_config_sha256": config_sha256,
        }
        logger.info("Loaded exact reviewed DFlash2 warm start: %s", model_sha256)

    def _install_adaptation_optimizer(self) -> None:
        total_parameters = 0
        trainable_parameters = 0
        trainable_names: list[str] = []
        for name, parameter in self.draft_model.named_parameters():
            count = parameter.numel()
            total_parameters += count
            selected = (
                self._adaptation_mode == FULL_ADAPTATION_MODE
                or trainable_parameter(name)
            )
            parameter.requires_grad_(selected)
            if selected:
                trainable_parameters += count
                trainable_names.append(name)
        if total_parameters != EXPECTED_TOTAL_PARAMETERS:
            raise RuntimeError(
                f"unexpected DFlash2 parameter count: {total_parameters}"
            )
        expected_trainable = (
            EXPECTED_TOTAL_PARAMETERS
            if self._adaptation_mode == FULL_ADAPTATION_MODE
            else EXPECTED_TRAINABLE_PARAMETERS
        )
        if trainable_parameters != expected_trainable:
            raise RuntimeError(
                f"unexpected trainable DFlash2 parameter count: {trainable_parameters}"
            )

        # AdamW is lazy, so the full-parameter optimizer created by the upstream
        # setup has allocated no moments yet. Replace it before the first step.
        old_optimizer = self.optimizer
        old_scheduler = self.lr_scheduler
        self.untrack_state("optimizer", "lr_scheduler")
        opt_cfg = self.cfg.optimizer
        self.peak_lr = float(opt_cfg.lr)
        trainable = [
            parameter
            for parameter in self.trainer_module.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.peak_lr,
            betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )
        warmup_ratio = float(opt_cfg.get("warmup_ratio", 0.05))
        min_lr_ratio = float(opt_cfg.get("min_lr_ratio", 0.1))
        warmup_steps = max(1, int(warmup_ratio * self.total_optim_steps))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            make_warmup_cosine_schedule(
                warmup_steps,
                self.total_optim_steps,
                min_lr_ratio,
            ),
        )
        del old_optimizer
        del old_scheduler
        gc.collect()
        self._adaptation_receipt.update(
            {
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "trainable_tensor_count": len(trainable_names),
                "trainable_names": trainable_names,
            }
        )
        logger.info(
            "DFlash2 adaptation mode %s: %d / %d parameters (%d tensors)",
            self._adaptation_mode,
            trainable_parameters,
            total_parameters,
            len(trainable_names),
        )

    def _save_extra_state(self, path: str, epoch: int) -> None:
        super()._save_extra_state(path, epoch)
        receipt = Path(path) / "exact_target_adaptation.json"
        temporary = receipt.with_name(".exact_target_adaptation.json.tmp")
        temporary.write_text(
            json.dumps(self._adaptation_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(receipt)


def main(config_path: str | None = None) -> None:
    result_raw = os.environ.get("AEON_DFLASH_RESULT_PATH", "")
    if not result_raw.startswith("/home/aday/.local/state/fleet-compute/runs/"):
        raise RuntimeError("training result path is outside Fleet scratch")
    result_path = Path(result_raw)
    trainer = None
    cap_receipt: dict[str, Any] | None = None
    started = time.monotonic()
    try:
        cap_receipt = _enforce_gpu_cap()
        cfg = parse_args_and_load_config(config_path)
        trainer = ExactTargetDFlash2Recipe(cfg)
        trainer.setup()
        trainer.run_train_validation_loop()
        latest_checkpoint = _normalize_latest_checkpoint(trainer)
        final_metrics = _metric_receipt(trainer)
    except BaseException as exc:
        _atomic_json(
            result_path,
            {
                "schema_version": "aeon-qwen38-dflash-adaptation-result-v1",
                "success": False,
                "failure_type": type(exc).__name__,
                "failure": str(exc)[:1000],
                "gpu_cap": cap_receipt,
            },
        )
        raise
    _atomic_json(
        result_path,
        {
            "schema_version": "aeon-qwen38-dflash-adaptation-result-v1",
            "success": True,
            "global_step": int(trainer.runtime.global_step),
            "duration_seconds": time.monotonic() - started,
            "gpu_cap": cap_receipt,
            "gpu_peak": {
                "allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            },
            "latest_checkpoint": latest_checkpoint,
            "final_metrics": final_metrics,
            "adaptation": trainer._adaptation_receipt,
        },
    )


if __name__ == "__main__":
    main()
