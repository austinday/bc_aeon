"""Identity and parameter-selection helpers for exact-target DFlash adaptation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


ADAPTATION_MODE = "projection-selector-conv-norm-v1"
FULL_ADAPTATION_MODE = "all-draft-v1"
ADAPTATION_MODES = frozenset({ADAPTATION_MODE, FULL_ADAPTATION_MODE})
EXPECTED_TRAINABLE_PARAMETERS = 325_319_680
EXPECTED_TOTAL_PARAMETERS = 1_924_404_480
EXPECTED_DRAFT_CONFIG = {
    "model_type": "qwen3",
    "vocab_size": 248_320,
    "hidden_size": 5_120,
    "num_hidden_layers": 5,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "sliding_window": 2_048,
    "is_causal": False,
    "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248_070,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def critical_draft_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: config.get(key)
        for key in (
            "model_type",
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "sliding_window",
            "is_causal",
        )
    }
    dflash = config.get("dflash_config")
    result["dflash_config"] = (
        {
            key: dflash.get(key)
            for key in (
                "block_size",
                "conv_group_size",
                "conv_kernel_size",
                "mask_token_id",
                "selector_rank",
                "selector_top_k",
                "target_layer_ids",
            )
        }
        if isinstance(dflash, Mapping)
        else None
    )
    return result


def validate_draft_config(config: Mapping[str, Any], *, label: str) -> None:
    actual = critical_draft_config(config)
    if actual != EXPECTED_DRAFT_CONFIG:
        differing = {
            key: {"actual": actual.get(key), "expected": expected}
            for key, expected in EXPECTED_DRAFT_CONFIG.items()
            if actual.get(key) != expected
        }
        raise RuntimeError(
            f"{label} DFlash2 config does not match the reviewed architecture: "
            f"{differing!r}"
        )


def trainable_parameter(name: str) -> bool:
    return (
        name.startswith("fc.")
        or name.startswith("candidate_selector.")
        or ".attention_conv." in name
        or ".mlp_conv." in name
        or name == "norm.weight"
        or name.endswith("_layernorm.weight")
    )
