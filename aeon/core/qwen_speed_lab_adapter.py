"""Reviewed Fleet batch adapter for isolated Qwen3.8 speed experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import threading
from typing import Any, Mapping

from fleet_compute.adapters import AdapterLaunchError, RuntimeContext
from fleet_compute.models import (
    LaunchResult,
    ProbeResult,
    ProbeState,
    StopResult,
    StorageFinalizationResult,
    StoragePreparationResult,
)

from aeon.core.engine_closure import (
    closure_request_identity,
    load_engine_closure_receipt,
)
from aeon.core.fleet_hosts import network_address
from aeon.core.sampling import QWEN_SPEED_LAB_SAMPLING_PROFILES


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENGINE_CLOSURE_RECEIPT = load_engine_closure_receipt(
    PACKAGE_ROOT / "aeon/core/data/qwen38_v026_dev1141_engine_closure.json"
)
ENGINE_CLOSURE_IDENTITY = closure_request_identity(ENGINE_CLOSURE_RECEIPT)
HOST = "192.168.0.180"
HOSTNAME = "DAY2XRTX5000PRO-2"
HOST_CONFIGS = {
    "192.168.0.179": {
        "enable_flashinfer_autotune": True,
        "hostname": "DAY2XRTX6000-2",
        "gpu_memory_utilization": 0.42,
        "max_batched_tokens": 32768,
        "use_flashinfer_sampler": False,
    },
    HOST: {
        "enable_flashinfer_autotune": False,
        "hostname": HOSTNAME,
        "gpu_memory_utilization": 0.84,
        "max_batched_tokens": 8192,
        "use_flashinfer_sampler": True,
    },
}
REMOTE_PYTHON = (
    "/home/aday/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
LOW_PRIORITY = "/home/aday/bin/fleet-low-priority"
REMOTE_RUN_ROOT = PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
FEATURE_DATASET = Path(
    "/home/aday/.local/state/aeon-qwen38-dflash-data/"
    "ara-prefix530e-greedy-v2-256/train.jsonl"
)
FEATURE_DATASET_SHA256 = (
    "61b8e150651ecc14c47e1068ce36fc130bb56e18117b3b68e098390defea92f5"
)
FEATURE_DATASET_ROWS = 256
MODEL_ARTIFACTS = {
    "production": {
        "manifest_sha256": (
            "1a3ba1eb88d0507bdef3798a6db59830dc076199b7db7d111201f6997588220e"
        ),
        "sha256s_sha256": (
            "e7eca7ebee03c4f27482d4fe421ca1fac9f1d9986663a51fd7614361010c1237"
        ),
        "canonical_model_dir": (
            "/home/aday/.aeon/models/"
            "Qwen3.8-27B-ARA-abliterated-NVFP4-MTP"
        ),
        "file_count": 21,
        "model_dir": None,
        "nvfp4_a16": False,
        "size_bytes": 20_579_661_729,
        "stage_per_attempt": True,
    },
    "fullgdn": {
        "manifest_sha256": (
            "14c4358805448fd691d18f587c54d29bc677e0708badae179089de480d9f4ede"
        ),
        "sha256s_sha256": (
            "3e5f114afa0777e46f8d5a5ac704c7f9becde21bed3b2268ce28a0b389458471"
        ),
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/models/"
            "3e5f114afa0777e46f8d5a5ac704c7f9becde21bed3b2268ce28a0b389458471"
        ),
        "nvfp4_a16": False,
    },
    "w4a4": {
        "manifest_sha256": (
            "b9cd6f0791fe08817ec1a5e7e739ddc80230fcceb94dbcfc06fc444b94c2e624"
        ),
        "sha256s_sha256": (
            "dd5a88636198a00e02ae10df0d95d7d07987b91299d2ebf56474e6d2ef5c421b"
        ),
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/models/"
            "dd5a88636198a00e02ae10df0d95d7d07987b91299d2ebf56474e6d2ef5c421b"
        ),
        "nvfp4_a16": False,
    },
    "w4a16": {
        "manifest_sha256": (
            "ea410f6df81cc4e4a3e4ea5cd1f8a34054d7784942772f21ff40a35c2203580f"
        ),
        "sha256s_sha256": (
            "d22f7280ba4d20c4fdb94201eb0a863ddfaaeea02d2bff34960b548adbf6f284"
        ),
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/models/"
            "d22f7280ba4d20c4fdb94201eb0a863ddfaaeea02d2bff34960b548adbf6f284"
        ),
        "nvfp4_a16": True,
    },
}
IMAGE_ID = "sha256:604c2525974bf41416e76c1f34ed014a1393d55617b4c7d7fc05d6c93754d9eb"
IMAGE_SIZE_BYTES = 4_604_212_353
ENGINE_ARCHIVE_SHA256 = str(ENGINE_CLOSURE_RECEIPT["archive_sha256"])
DRAFT_ARTIFACTS = {
    "bf16": {
        "revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
        "revision_sha256": (
            "0e390b23b8c018ade89fdedcc892071b22571be473385b74e1e21183d50623f5"
        ),
        "config_sha256": (
            "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980"
        ),
        "model_sha256": (
            "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
        ),
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
        ),
    },
    "aeonv1": {
        "revision": (
            "aeon-exact-prefix-v1@"
            "3fb704651dc0de150ebbdd2b65838303bfaaf626853b6941d130d6d1f711c7c2"
        ),
        "revision_sha256": (
            "4f6735ee1c045a8b4879daef3abfcce9c91eb4c05cba638d323be8ab5e8a4acf"
        ),
        "config_sha256": (
            "817cd58af1b039e4b101373315970e43247e559e91031aeb1a7e950011e14152"
        ),
        "model_sha256": (
            "a23651e35305ff4f83a144380e201e6fcc70ed1e442058112e29e852cd639ca5"
        ),
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "a23651e35305ff4f83a144380e201e6fcc70ed1e442058112e29e852cd639ca5"
        ),
    },
    "aeonfullv1": {
        "revision": (
            "aeon-exact-prefix-full-v1@"
            "efbb5c2fd07bf65d8a6b113699c9d59c1209d5fc00fc023c2f3babfe691bb955"
        ),
        "revision_sha256": (
            "2e4b495520bd3fb800e41988da0a8163241fd0c2736618ead093ee7ce44527f1"
        ),
        "config_sha256": (
            "817cd58af1b039e4b101373315970e43247e559e91031aeb1a7e950011e14152"
        ),
        "model_sha256": (
            "cf91386d67cfce1c43f2bd312267a7381821ca63415764fbfccceff285cfcf83"
        ),
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "cf91386d67cfce1c43f2bd312267a7381821ca63415764fbfccceff285cfcf83"
        ),
    },
    "w4a16": {
        "revision": "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279",
        "revision_sha256": (
            "8b2f78143963d23aeab94d9242ce38a1654cd3fe4187857550573076f16771df"
        ),
        "config_sha256": (
            "61d6276fe8d76295232cb02d26cbb0d29c25565911f50441e779c88c9220c556"
        ),
        "model_sha256": (
            "ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7"
        ),
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7"
        ),
    },
    "w8a16": {
        "revision": "f454fa8e6a84387bf006f849584f72541cc29118",
        "revision_sha256": (
            "53fae14c440666c760f14c56386eaea14afaa626cc2196ea5bf0a39f9d0fef27"
        ),
        "config_sha256": (
            "1c63a70cd7fa0e8be7276b3597e464ab9a0e609b70611354e9fb76e76b2e42d6"
        ),
        "model_sha256": (
            "72223787fbfe7e01c4a940f80f98cf3d7a803e9b79aa4ec7bc28a461d767fa18"
        ),
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "72223787fbfe7e01c4a940f80f98cf3d7a803e9b79aa4ec7bc28a461d767fa18"
        ),
    },
    "dspark-nvfp4": {
        "revision": (
            "gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4@"
            "eba1ac5a66c74902eaa95a4000a7c5eda96d8e95"
        ),
        # SHA-256 of canonical {repo_id, revision} source identity JSON.
        "revision_sha256": (
            "89be99d1047b9999c71e7fe10be3652c00cc6896c107d580224780b05974220a"
        ),
        "config_sha256": (
            "82fd961b632c629736902d9d4fdd3258dee1080f557cf86298cac063a514a0cf"
        ),
        "hf_quant_config_sha256": (
            "cda90695e8c4a5eaed7ce7220afbc8bbe18e7624a167466ec7768c603e756a09"
        ),
        "model_sha256": (
            "212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05"
        ),
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05"
        ),
    },
}
PROFILE_ID = "aeon-qwen38-speed-lab"
PRODUCTION_K3_CANARY_PROFILE_ID = (
    "aeon-qwen38-production-k3-v026-canary-179"
)
PRODUCTION_K3_CANARY_HOST = "192.168.0.179"
PRODUCTION_K3_CANARY_VARIANT = (
    "v026-dev1141-production-nvfp4-mtp-k3-triton-fp8pthead-114688"
)
PRODUCTION_K3_CANARY_VRAM_BUDGET_GB = 48.7
WORKER_SCHEMA_VERSION = "aeon-qwen38-speed-worker-v18"
VARIANT_CONFIGS = {
    "nightly-v2-fused-gdn-int8-heads-ar-bf16kv": {
        "attention_backend": "TRITON_ATTN",
        "draft_id": "bf16",
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "auto",
        "speculative_method": "none",
        "speculative_tokens": 0,
    },
    **{
        f"nightly-v2-fused-gdn-int8-heads-mtp-k{k}-bf16kv": {
            "attention_backend": "TRITON_ATTN",
            "draft_id": "bf16",
            "kv_cache_dtype": "auto",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "speculative_method": "mtp",
            "speculative_tokens": k,
        }
        for k in range(1, 7)
    },
    "nightly-v2-fused-gdn-int8-heads-dflash2-k7-triton-bf16kv": {
        "attention_backend": "TRITON_ATTN",
        "draft_id": "bf16",
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "auto",
        "speculative_method": "dflash",
        "speculative_tokens": 7,
    },
    "nightly-v2-fused-gdn-int8-heads-dflash2-k7-flashattn-bf16kv": {
        "attention_backend": "FLASH_ATTN",
        "draft_id": "bf16",
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "auto",
        "speculative_method": "dflash",
        "speculative_tokens": 7,
    },
    "nightly-v2-fused-gdn-int8-heads-dflash2-k7-flashinfer-bf16kv": {
        "attention_backend": "FLASHINFER",
        "draft_id": "bf16",
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "auto",
        "speculative_method": "dflash",
        "speculative_tokens": 7,
    },
    "nightly-v2-fused-gdn-int8-heads-dflash2-k7-triton-bf16kv-fp16state": {
        "attention_backend": "TRITON_ATTN",
        "draft_id": "bf16",
        "kv_cache_dtype": "auto",
        "mamba_cache_dtype": "float16",
        "mamba_ssm_cache_dtype": "float16",
        "speculative_method": "dflash",
        "speculative_tokens": 7,
    },
    **{
        f"nightly-v2-fused-gdn-int8-heads-dflash2-w4a16-k{k}-triton-bf16kv": {
            "attention_backend": "TRITON_ATTN",
            "draft_id": "w4a16",
            "kv_cache_dtype": "auto",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "speculative_method": "dflash",
            "speculative_tokens": k,
        }
        for k in range(5, 8)
    },
    **{
        f"nightly-v2-fused-gdn-int8-heads-dflash2-w4a16-k{k}-triton-fp8kv": {
            "attention_backend": "TRITON_ATTN",
            "draft_id": "w4a16",
            "kv_cache_dtype": "fp8",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "speculative_method": "dflash",
            "speculative_tokens": k,
        }
        for k in range(5, 8)
    },
    **{
        f"nightly-v2-fused-gdn-int8-heads-dflash2-k{k}-{backend_name}-fp8kv": {
            "attention_backend": backend,
            "draft_id": "bf16",
            "kv_cache_dtype": "fp8",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "speculative_method": "dflash",
            "speculative_tokens": k,
        }
        for k in (5, 6, 7)
        for backend_name, backend in (
            ("triton", "TRITON_ATTN"),
            ("flashinfer", "FLASHINFER"),
        )
    },
    **{
        f"nightly-v2-fused-gdn-int8-heads-dflash2-w4a16-k{k}-flashinfer-{cache_name}": {
            "attention_backend": "FLASHINFER",
            "draft_id": "w4a16",
            "kv_cache_dtype": cache_dtype,
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "speculative_method": "dflash",
            "speculative_tokens": k,
        }
        for k in (6, 7)
        for cache_name, cache_dtype in (("bf16kv", "auto"), ("fp8kv", "fp8"))
    },
}
for _variant_config in VARIANT_CONFIGS.values():
    _variant_config["model_id"] = "w4a4"
    _variant_config["compilation_profile"] = "default"
VARIANT_CONFIGS.update(
    {
        **{
            f"nightly-v2-fused-gdn-int8-heads-dflash2-aeonv1-k{k}-{backend_name}-fp8kv": {
                "attention_backend": backend,
                "draft_id": "aeonv1",
                "kv_cache_dtype": "fp8",
                "mamba_cache_dtype": "auto",
                "mamba_ssm_cache_dtype": "auto",
                "model_id": "w4a4",
                "speculative_method": "dflash",
                "speculative_tokens": k,
            }
            for k in (5, 6, 7)
            for backend_name, backend in (
                ("triton", "TRITON_ATTN"),
                ("flashinfer", "FLASHINFER"),
            )
        },
        **{
            f"nightly-v2-fused-gdn-int8-heads-dflash2-aeonfullv1-k{k}-{backend_name}-fp8kv": {
                "attention_backend": backend,
                "draft_id": "aeonfullv1",
                "kv_cache_dtype": "fp8",
                "mamba_cache_dtype": "auto",
                "mamba_ssm_cache_dtype": "auto",
                "model_id": "w4a4",
                "speculative_method": "dflash",
                "speculative_tokens": k,
            }
            for k in (5, 6, 7)
            for backend_name, backend in (
                ("triton", "TRITON_ATTN"),
                ("flashinfer", "FLASHINFER"),
            )
        },
        "nightly-v2-fused-gdn-int8-heads-dflash2-k7-flashinfer-fp8kv-attnquant-partition": {
            "attention_backend": "FLASHINFER",
            "compilation_profile": "attnquant-partition",
            "draft_id": "bf16",
            "kv_cache_dtype": "fp8",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "model_id": "w4a4",
            "speculative_method": "dflash",
            "speculative_tokens": 7,
        },
        "nightly-v2-fused-gdn-int8-heads-dflash2-k7-flashinfer-fp8kv-attnquant-fullgraph": {
            "attention_backend": "FLASHINFER",
            "compilation_profile": "attnquant-fullgraph",
            "draft_id": "bf16",
            "kv_cache_dtype": "fp8",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "model_id": "w4a4",
            "speculative_method": "dflash",
            "speculative_tokens": 7,
        },
        **{
            f"nightly-v2-fused-gdn-int8-heads-dflash2-w8-k{k}-flashinfer-fp8kv": {
                "attention_backend": "FLASHINFER",
                "draft_id": "w8a16",
                "kv_cache_dtype": "fp8",
                "mamba_cache_dtype": "auto",
                "mamba_ssm_cache_dtype": "auto",
                "model_id": "w4a4",
                "speculative_method": "dflash",
                "speculative_tokens": k,
            }
            for k in (6, 7)
        },
        **{
            f"nightly-v2-nvfp4a16-int8-heads-dflash2-{draft_name}k{k}-{backend_name}-fp8kv": {
                "attention_backend": backend,
                "draft_id": draft_id,
                "kv_cache_dtype": "fp8",
                "mamba_cache_dtype": "auto",
                "mamba_ssm_cache_dtype": "auto",
                "model_id": "w4a16",
                "speculative_method": "dflash",
                "speculative_tokens": k,
            }
            for draft_name, draft_id in (("", "bf16"), ("w8-", "w8a16"))
            for k in (6, 7)
            for backend_name, backend in (
                ("triton", "TRITON_ATTN"),
                ("flashinfer", "FLASHINFER"),
            )
        },
        **{
            (
                "nightly-v2-fused-gdn-int8-heads-dspark-nvfp4-"
                f"k{k}{topk_name}-flashinfer-fp8kv"
            ): {
                "attention_backend": "FLASHINFER",
                "draft_id": "dspark-nvfp4",
                "dspark_draft_topk": topk,
                "enable_adaptive_verification": False,
                "kv_cache_dtype": "fp8",
                "mamba_cache_dtype": "auto",
                "mamba_ssm_cache_dtype": "auto",
                "model_id": "w4a4",
                "speculative_method": "dspark",
                "speculative_tokens": k,
            }
            for k in (5, 6, 7)
            for topk_name, topk in (
                ("", None),
                ("-topk32", 32),
                ("-topk64", 64),
                ("-topk128", 128),
                ("-topk256", 256),
            )
        },
    }
)
VARIANT_CONFIGS.update(
    {
        "nightly-v2-full-gdn-nvfp4-ar-flashinfer-fp8kv": {
            "attention_backend": "FLASHINFER",
            "draft_id": "bf16",
            "kv_cache_dtype": "fp8",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "model_id": "fullgdn",
            "speculative_method": "none",
            "speculative_tokens": 0,
        },
        **{
            f"nightly-v2-full-gdn-nvfp4-dflash2-k{k}-flashinfer-fp8kv": {
                "attention_backend": "FLASHINFER",
                "draft_id": "bf16",
                "kv_cache_dtype": "fp8",
                "mamba_cache_dtype": "auto",
                "mamba_ssm_cache_dtype": "auto",
                "model_id": "fullgdn",
                "speculative_method": "dflash",
                "speculative_tokens": k,
            }
            for k in (5, 6, 7, 8)
        },
        "nightly-v2-full-gdn-nvfp4-dflash2-k7-triton-fp8kv": {
            "attention_backend": "TRITON_ATTN",
            "draft_id": "bf16",
            "kv_cache_dtype": "fp8",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "model_id": "fullgdn",
            "speculative_method": "dflash",
            "speculative_tokens": 7,
        },
        "nightly-v2-full-gdn-nvfp4-dflash2-k7-flashinfer-fp8kv-piecewise": {
            "attention_backend": "FLASHINFER",
            "compilation_profile": "piecewise",
            "draft_id": "bf16",
            "kv_cache_dtype": "fp8",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "model_id": "fullgdn",
            "speculative_method": "dflash",
            "speculative_tokens": 7,
        },
        "nightly-v2-full-gdn-nvfp4-dflash2-k7-flashinfer-fp8kv-native-full": {
            "attention_backend": "FLASHINFER",
            "compilation_profile": "flashinfer-native-full",
            "draft_id": "bf16",
            "kv_cache_dtype": "fp8",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "model_id": "fullgdn",
            "speculative_method": "dflash",
            "speculative_tokens": 7,
        },
        "nightly-v2-full-gdn-nvfp4-dflash2-feature-capture": {
            "attention_backend": "FLASHINFER",
            "compilation_profile": "piecewise",
            "context_tokens": 12288,
            "draft_id": "bf16",
            "enable_prefix_caching": True,
            "feature_capture": True,
            "kv_cache_dtype": "fp8",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "model_id": "fullgdn",
            "speculative_method": "dflash",
            "speculative_tokens": 1,
        },
    }
)
VARIANT_CONFIGS[PRODUCTION_K3_CANARY_VARIANT] = {
    "allowed_hosts": ("192.168.0.179",),
    "attention_backend": "TRITON_ATTN",
    "compilation_profile": "default",
    "context_tokens": 114688,
    "draft_id": "bf16",
    "enable_flashinfer_autotune_override": False,
    "enable_per_request_metrics": True,
    "gpu_memory_utilization_override": 0.415,
    "kv_cache_dtype": "fp8_per_token_head",
    "mamba_cache_dtype": "auto",
    "mamba_ssm_cache_dtype": "auto",
    "max_batched_tokens_override": 32768,
    "min_physical_vram_gb": 90.0,
    "model_id": "production",
    "profile_id": PRODUCTION_K3_CANARY_PROFILE_ID,
    "speculative_method": "mtp",
    "speculative_tokens": 3,
    "use_flashinfer_sampler_override": False,
    "vram_budget_gb": PRODUCTION_K3_CANARY_VRAM_BUDGET_GB,
}
for _variant_config in VARIANT_CONFIGS.values():
    _variant_config.setdefault("allowed_hosts", tuple(HOST_CONFIGS))
    _variant_config.setdefault("compilation_profile", "default")
    _variant_config.setdefault("context_tokens", 131072)
    _variant_config.setdefault("dspark_draft_topk", None)
    _variant_config.setdefault("enable_adaptive_verification", False)
    _variant_config.setdefault("enable_flashinfer_autotune_override", None)
    _variant_config.setdefault("enable_prefix_caching", True)
    _variant_config.setdefault("enable_per_request_metrics", False)
    _variant_config.setdefault("feature_capture", False)
    _variant_config.setdefault("gpu_memory_utilization_override", None)
    _variant_config.setdefault("max_batched_tokens_override", None)
    _variant_config.setdefault("min_physical_vram_gb", 47.0)
    _variant_config.setdefault("profile_id", PROFILE_ID)
    _variant_config.setdefault("relaxed_greedy_logit_margin", "0")
    _variant_config.setdefault("use_flashinfer_sampler_override", None)
    _variant_config.setdefault("vram_budget_gb", 41.25)
VARIANT = "nightly-v2-fused-gdn-int8-heads-dflash2-w4a16-k7-triton-bf16kv"
PORT = 18033
_RUNTIME_ID_RE = re.compile(r"^fr-[a-f0-9]{32}$")
_PROCESS_IDENTITY_RE = re.compile(
    r"^aeon-speed-lab:(fr-[a-f0-9]{32}):([a-f0-9]{64}):([0-9]+)$"
)

SOURCE_FILES = (
    "aeon/__init__.py",
    "aeon/core/__init__.py",
    "aeon/core/action_schema.py",
    "aeon/core/data/qwen38_v026_dev1141_engine_closure.json",
    "aeon/core/engine_closure.py",
    "aeon/core/fleet_hosts.py",
    "aeon/core/mtp_tuning.py",
    "aeon/core/qwen_fast_service_adapter.py",
    "aeon/core/qwen_speed_lab_adapter.py",
    "aeon/core/sampling.py",
    "aeon/scripts/benchmark_qwen38_mtp.py",
    "aeon/scripts/benchmark_qwen38_speed.py",
    "aeon/scripts/build_qwen38_speed_variant.py",
    "aeon/scripts/extract_qwen38_dflash_features.py",
    "aeon/scripts/local_http_sitecustomize/sitecustomize.py",
    "aeon/scripts/qwen_speed_lab_worker.py",
    "aeon/scripts/speed_lab_sitecustomize/sitecustomize.py",
    "aeon/scripts/vllm_uuid_sitecustomize.py",
    "aeon/services/vllm/Dockerfile.speed-heads-canary",
    "aeon/services/vllm/patches/aeon-qwen38-speed-heads-a047e.patch",
)
PROMPT_SOURCE_FILES = (
    "aeon/core/prompts/core_directives.txt",
    "aeon/core/prompts/docker_directives.txt",
    "aeon/core/prompts/important_reminders.txt",
    "aeon/core/prompts/primary_agent_instructions.txt",
    "aeon/core/prompts/fleet_safety_instructions.txt",
    *tuple(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for root in (
            PACKAGE_ROOT / "aeon/core/prompts/categories",
            PACKAGE_ROOT / "aeon/core/prompts/tools",
        )
        for path in sorted(root.rglob("*.txt"))
    ),
)


class QwenSpeedLabError(RuntimeError):
    pass


class QwenSpeedLabTransportError(QwenSpeedLabError):
    """A retryable failure before the reviewed remote protocol answered."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _host_runtime_value(
    variant_config: Mapping[str, Any], host_config: Mapping[str, Any], name: str
) -> Any:
    override = variant_config[f"{name}_override"]
    return host_config[name] if override is None else override


def _source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = PACKAGE_ROOT / relative
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise QwenSpeedLabError(f"speed-lab source is mutable: {relative}")
        result[relative] = _sha256(path)
    return result


def _prompt_bundle() -> tuple[bytes, dict[str, str]]:
    parts: list[bytes] = []
    identities: dict[str, str] = {}
    for relative in PROMPT_SOURCE_FILES:
        path = PACKAGE_ROOT / relative
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_size > 2 * 1024 * 1024
        ):
            raise QwenSpeedLabError(f"prompt source is unsafe: {relative}")
        payload = path.read_bytes()
        identities[relative] = hashlib.sha256(payload).hexdigest()
        parts.append(payload.rstrip() + b"\n\n")
    bundle = b"".join(parts)
    if not 4096 <= len(bundle) <= 2 * 1024 * 1024:
        raise QwenSpeedLabError("speed-lab prompt bundle is outside its bound")
    return bundle, identities


def _ssh_base(host: str) -> list[str]:
    if host not in HOST_CONFIGS:
        raise QwenSpeedLabError("speed-lab host is not allowlisted")
    return [
        "/usr/bin/ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=6",
        f"aday@{network_address(host)}",
    ]


def _remote_command(
    host: str,
    source_root: str,
    action: str,
    request: str,
    digest: str,
    extra: str | None = None,
) -> list[str]:
    if action not in {
        "preflight",
        "spawn",
        "status",
        "stop",
        "service-spawn",
        "service-status",
        "service-stop",
        "service-cleanup",
        "settle-status",
        "mark-settled",
        "cleanup",
    }:
        raise QwenSpeedLabError("invalid speed-lab worker action")
    worker = f"{source_root}/aeon/scripts/qwen_speed_lab_worker.py"
    command = [
        "/usr/bin/env",
        "-i",
        "PATH=/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME=/home/aday",
        "LANG=C",
        "LC_ALL=C",
        f"PYTHONPATH={source_root}",
        "PYTHONDONTWRITEBYTECODE=1",
        "/usr/bin/bash",
        LOW_PRIORITY,
        REMOTE_PYTHON,
        worker,
        action,
        request,
        digest,
    ]
    if extra is not None:
        command.append(extra)
    return [*_ssh_base(host), shlex.join(command)]


def _remote_action(
    host: str,
    source_root: str,
    action: str,
    request: str,
    digest: str,
    *,
    extra: str | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    result = subprocess.run(
        _remote_command(host, source_root, action, request, digest, extra),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if len(result.stdout) > 2 * 1024 * 1024 or len(result.stderr) > 256 * 1024:
        raise QwenSpeedLabError("speed-lab worker response exceeded its bound")
    if result.returncode == 255:
        raise QwenSpeedLabTransportError("speed-lab worker transport is unavailable")
    try:
        value = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenSpeedLabTransportError(
            "speed-lab worker transport returned no valid response"
        ) from exc
    if (
        result.returncode != 0
        or not isinstance(value, dict)
        or value.get("ok") is not True
    ):
        detail = value.get("detail") if isinstance(value, dict) else None
        raise QwenSpeedLabError(
            f"speed-lab worker {action} failed: {detail or 'unknown error'}"
        )
    return value


def _remote_metrics(host: str, path: str, *, create: bool) -> tuple[str, int, int, int]:
    hostname = str(HOST_CONFIGS[host]["hostname"])
    script = """
import json, os, stat, sys
path, expected, create = sys.argv[1:]
assert os.uname().nodename == expected
if create == "1": os.makedirs(path, mode=0o700, exist_ok=True); os.chmod(path, 0o700)
try: metadata=os.lstat(path)
except FileNotFoundError:
 print(json.dumps({"state":"absent"})); raise SystemExit(0)
assert stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == os.geteuid() and not metadata.st_mode & 0o077
values=os.statvfs(path)
allocated=0
for root, directories, files in os.walk(path, topdown=True, followlinks=False):
 for name in [".", *directories, *files]:
  item=root if name == "." else os.path.join(root, name)
  item_metadata=os.lstat(item)
  assert item_metadata.st_uid == os.geteuid()
  assert stat.S_ISDIR(item_metadata.st_mode) or stat.S_ISREG(item_metadata.st_mode)
  allocated += item_metadata.st_blocks * 512
print(json.dumps({"state":"present","device":str(metadata.st_dev),"free":values.f_bavail*values.f_frsize,"inodes":values.f_favail,"allocated":allocated}, sort_keys=True))
"""
    command = [
        *_ssh_base(host),
        shlex.join(
            [
                "/usr/bin/bash",
                LOW_PRIORITY,
                REMOTE_PYTHON,
                "-c",
                script,
                path,
                hostname,
                "1" if create else "0",
            ]
        ),
    ]
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or len(result.stdout) > 4096:
        raise QwenSpeedLabError("worker storage metrics are unavailable")
    try:
        value = json.loads(result.stdout)
        if value.get("state") == "absent":
            raise FileNotFoundError(path)
        return (
            str(value["device"]),
            int(value["free"]),
            int(value["inodes"]),
            int(value["allocated"]),
        )
    except FileNotFoundError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QwenSpeedLabError("worker storage metrics are malformed") from exc


class _PreparationHeartbeat:
    def __init__(self, context: RuntimeContext) -> None:
        self.context = context
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_PreparationHeartbeat":
        self.context.heartbeat(None, "Qwen speed-lab artifact preflight")
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise QwenSpeedLabError(
                "speed-lab preflight heartbeat failed"
            ) from self.error

    def _run(self) -> None:
        while not self.stop.wait(240):
            try:
                self.context.heartbeat(
                    None, "Qwen speed-lab artifact preflight is active"
                )
            except BaseException as exc:
                self.error = exc
                return


class AeonQwenSpeedLabAdapter:
    """One allowlisted benchmark lane on the released 48 GB worker class."""

    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _payload(payload: Mapping[str, Any]) -> dict[str, int | str]:
        allowed = {
            "variant",
            "repeats",
            "quality_repeats",
            "max_tokens",
            "sampling_profile",
        }
        if not isinstance(payload, Mapping) or set(payload) - allowed:
            raise QwenSpeedLabError("speed-lab payload has unsupported fields")
        variant = payload.get("variant", VARIANT)
        repeats = payload.get("repeats", 5)
        quality_repeats = payload.get("quality_repeats", 2)
        max_tokens = payload.get("max_tokens", 512)
        sampling_profile = payload.get("sampling_profile", "aeon-greedy-medium")
        if (
            variant not in VARIANT_CONFIGS
            or not isinstance(sampling_profile, str)
            or sampling_profile not in QWEN_SPEED_LAB_SAMPLING_PROFILES
            or isinstance(repeats, bool)
            or not isinstance(repeats, int)
            or not 3 <= repeats <= 9
            or isinstance(quality_repeats, bool)
            or not isinstance(quality_repeats, int)
            or not 2 <= quality_repeats <= 5
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 256 <= max_tokens <= 2048
        ):
            raise QwenSpeedLabError("speed-lab payload is outside its reviewed bounds")
        return {
            "variant": variant,
            "repeats": repeats,
            "quality_repeats": quality_repeats,
            "max_tokens": max_tokens,
            "sampling_profile": sampling_profile,
        }

    @staticmethod
    def _profile_identity(
        context: RuntimeContext,
        sources: dict[str, str],
        prompt: bytes,
        variant_config: Mapping[str, Any],
    ) -> None:
        standard_expected = {
            "image": IMAGE_ID.removeprefix("sha256:"),
            "bare_engine_archive": ENGINE_ARCHIVE_SHA256,
            "bare_engine_closure": ENGINE_CLOSURE_IDENTITY["manifest_sha256"],
            "bare_python_executable": ENGINE_CLOSURE_IDENTITY[
                "python_executable_sha256"
            ],
            "dflash2_bf16_config": DRAFT_ARTIFACTS["bf16"]["config_sha256"],
            "dflash2_bf16_model": DRAFT_ARTIFACTS["bf16"]["model_sha256"],
            "dflash2_bf16_revision": DRAFT_ARTIFACTS["bf16"]["revision_sha256"],
            "dflash2_aeonv1_config": DRAFT_ARTIFACTS["aeonv1"]["config_sha256"],
            "dflash2_aeonv1_model": DRAFT_ARTIFACTS["aeonv1"]["model_sha256"],
            "dflash2_aeonv1_revision": DRAFT_ARTIFACTS["aeonv1"]["revision_sha256"],
            "dflash2_aeonfullv1_config": DRAFT_ARTIFACTS["aeonfullv1"]["config_sha256"],
            "dflash2_aeonfullv1_model": DRAFT_ARTIFACTS["aeonfullv1"]["model_sha256"],
            "dflash2_aeonfullv1_revision": DRAFT_ARTIFACTS["aeonfullv1"][
                "revision_sha256"
            ],
            "dflash2_w4a16_config": DRAFT_ARTIFACTS["w4a16"]["config_sha256"],
            "dflash2_w4a16_model": DRAFT_ARTIFACTS["w4a16"]["model_sha256"],
            "dflash2_w4a16_revision": DRAFT_ARTIFACTS["w4a16"]["revision_sha256"],
            "dflash2_w8a16_config": DRAFT_ARTIFACTS["w8a16"]["config_sha256"],
            "dflash2_w8a16_model": DRAFT_ARTIFACTS["w8a16"]["model_sha256"],
            "dflash2_w8a16_revision": DRAFT_ARTIFACTS["w8a16"]["revision_sha256"],
            "dspark_nvfp4_config": DRAFT_ARTIFACTS["dspark-nvfp4"]["config_sha256"],
            "dspark_nvfp4_hf_quant_config": DRAFT_ARTIFACTS["dspark-nvfp4"][
                "hf_quant_config_sha256"
            ],
            "dspark_nvfp4_model": DRAFT_ARTIFACTS["dspark-nvfp4"]["model_sha256"],
            "dspark_nvfp4_revision": DRAFT_ARTIFACTS["dspark-nvfp4"]["revision_sha256"],
            "feature_dataset": FEATURE_DATASET_SHA256,
            "model_w4a4_manifest": MODEL_ARTIFACTS["w4a4"]["manifest_sha256"],
            "model_w4a4_sha256s": MODEL_ARTIFACTS["w4a4"]["sha256s_sha256"],
            "model_w4a16_manifest": MODEL_ARTIFACTS["w4a16"]["manifest_sha256"],
            "model_w4a16_sha256s": MODEL_ARTIFACTS["w4a16"]["sha256s_sha256"],
            "model_fullgdn_manifest": MODEL_ARTIFACTS["fullgdn"][
                "manifest_sha256"
            ],
            "model_fullgdn_sha256s": MODEL_ARTIFACTS["fullgdn"][
                "sha256s_sha256"
            ],
            "source_manifest": _canonical_sha256(sources),
            "prompt_bundle": hashlib.sha256(prompt).hexdigest(),
        }
        canary_expected = {
            "bare_engine_archive": ENGINE_ARCHIVE_SHA256,
            "bare_engine_closure": ENGINE_CLOSURE_IDENTITY["manifest_sha256"],
            "bare_python_executable": ENGINE_CLOSURE_IDENTITY[
                "python_executable_sha256"
            ],
            "dflash2_bf16_config": DRAFT_ARTIFACTS["bf16"]["config_sha256"],
            "dflash2_bf16_model": DRAFT_ARTIFACTS["bf16"]["model_sha256"],
            "dflash2_bf16_revision": DRAFT_ARTIFACTS["bf16"]["revision_sha256"],
            "image": IMAGE_ID.removeprefix("sha256:"),
            "model_production_manifest": MODEL_ARTIFACTS["production"][
                "manifest_sha256"
            ],
            "model_production_sha256s": MODEL_ARTIFACTS["production"][
                "sha256s_sha256"
            ],
            "prompt_bundle": hashlib.sha256(prompt).hexdigest(),
            "runtime_variant": _canonical_sha256(
                VARIANT_CONFIGS[PRODUCTION_K3_CANARY_VARIANT]
            ),
            "source_manifest": _canonical_sha256(sources),
        }
        expected_profile_id = str(variant_config["profile_id"])
        if expected_profile_id == PROFILE_ID:
            expected = standard_expected
        elif expected_profile_id == PRODUCTION_K3_CANARY_PROFILE_ID:
            expected = canary_expected
        else:
            raise QwenSpeedLabError("speed-lab variant profile is not reviewed")
        if (
            context.profile.profile_id != expected_profile_id
            or context.profile.artifact_identity != expected
        ):
            raise QwenSpeedLabError("speed-lab profile artifact identity changed")
        lease = context.lease
        if (
            lease.host not in HOST_CONFIGS
            or lease.host not in variant_config["allowed_hosts"]
            or lease.memory_total_mib is None
            or lease.memory_total_mib
            < float(variant_config["min_physical_vram_gb"]) * 1024
            or abs(
                lease.vram_budget_gb - float(variant_config["vram_budget_gb"])
            )
            > 1e-9
            or lease.exclusive is not True
            or context.scratch_path != lease.run_dir
        ):
            raise QwenSpeedLabError("speed-lab lease differs from its reviewed profile")

    @staticmethod
    def _write_private(path: Path, payload: bytes) -> None:
        if path.exists() or path.is_symlink():
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise QwenSpeedLabError("local staging path is unsafe")
            path.unlink()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _stage_sources(
        host: str, hostname: str, scratch: str, sources: dict[str, str]
    ) -> None:
        source_root = f"{scratch}/source"
        make_script = """
import os, stat, sys
scratch, source, expected = sys.argv[1:]
assert os.uname().nodename == expected
os.makedirs(scratch, mode=0o700, exist_ok=True); os.chmod(scratch,0o700)
os.makedirs(source, mode=0o700, exist_ok=True); os.chmod(source,0o700)
for path in (scratch,source):
 m=os.lstat(path); assert stat.S_ISDIR(m.st_mode) and m.st_uid==os.geteuid() and not m.st_mode & 0o077
"""
        made = subprocess.run(
            [
                *_ssh_base(host),
                shlex.join(
                    [
                        "/usr/bin/bash",
                        LOW_PRIORITY,
                        REMOTE_PYTHON,
                        "-c",
                        make_script,
                        scratch,
                        source_root,
                        hostname,
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if made.returncode != 0:
            raise QwenSpeedLabError("worker source root could not be prepared")
        ssh_transport = " ".join(shlex.quote(item) for item in _ssh_base(host)[:-1])
        transfer = subprocess.run(
            [
                "/usr/bin/bash",
                LOW_PRIORITY,
                "/usr/bin/rsync",
                "-aR",
                "--checksum",
                "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                "--protect-args",
                "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                "-e",
                ssh_transport,
                "--",
                *sources,
                f"aday@{network_address(host)}:{source_root}/",
            ],
            cwd=PACKAGE_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if transfer.returncode != 0:
            raise QwenSpeedLabError("worker speed-lab source staging failed")

    @staticmethod
    def _stage_file(host: str, local: Path, remote: str) -> None:
        ssh_transport = " ".join(shlex.quote(item) for item in _ssh_base(host)[:-1])
        transfer = subprocess.run(
            [
                "/usr/bin/bash",
                LOW_PRIORITY,
                "/usr/bin/rsync",
                "-a",
                "--checksum",
                "--protect-args",
                "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                "-e",
                ssh_transport,
                "--",
                str(local),
                f"aday@{network_address(host)}:{remote}",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if transfer.returncode != 0:
            raise QwenSpeedLabError("worker speed-lab fixture staging failed")

    @staticmethod
    def _stage_production_model(
        context: RuntimeContext,
        host: str,
        hostname: str,
        scratch: str,
        model: Mapping[str, Any],
    ) -> str:
        source = Path(str(model["canonical_model_dir"]))
        source_metadata = source.lstat()
        files = 0
        size_bytes = 0
        if (
            not stat.S_ISDIR(source_metadata.st_mode)
            or source_metadata.st_uid != os.geteuid()
            or source_metadata.st_mode & 0o022
        ):
            raise QwenSpeedLabError("canonical production model root is unsafe")
        for path in source.rglob("*"):
            metadata = path.lstat()
            if (
                metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o022
                or not (
                    stat.S_ISDIR(metadata.st_mode)
                    or stat.S_ISREG(metadata.st_mode)
                )
            ):
                raise QwenSpeedLabError("canonical production model tree is unsafe")
            if stat.S_ISREG(metadata.st_mode):
                files += 1
                size_bytes += metadata.st_size
        if (
            files != model["file_count"]
            or size_bytes != model["size_bytes"]
            or _sha256(source / "BUILD_MANIFEST.json") != model["manifest_sha256"]
            or _sha256(source / "SHA256SUMS") != model["sha256s_sha256"]
        ):
            raise QwenSpeedLabError("canonical production model identity changed")
        target = f"{scratch}/production-model"
        make_script = """
import os, stat, sys
scratch, target, expected = sys.argv[1:]
assert os.uname().nodename == expected
assert os.path.dirname(target) == scratch
os.makedirs(target, mode=0o700, exist_ok=True); os.chmod(target,0o700)
for root, directories, files in os.walk(target, topdown=True, followlinks=False):
 for item in [root, *(os.path.join(root,name) for name in directories+files)]:
  metadata=os.lstat(item)
  assert metadata.st_uid==os.geteuid() and not stat.S_ISLNK(metadata.st_mode)
  assert stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
"""
        made = subprocess.run(
            [
                *_ssh_base(host),
                shlex.join(
                    [
                        "/usr/bin/bash",
                        LOW_PRIORITY,
                        REMOTE_PYTHON,
                        "-c",
                        make_script,
                        scratch,
                        target,
                        hostname,
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if made.returncode != 0:
            raise QwenSpeedLabError("worker production model root is unsafe")
        ssh_transport = " ".join(shlex.quote(item) for item in _ssh_base(host)[:-1])
        context.heartbeat(None, "Qwen production model staging is active")
        with _PreparationHeartbeat(context):
            transfer = subprocess.run(
                [
                    "/usr/bin/bash",
                    LOW_PRIORITY,
                    "/usr/bin/rsync",
                    "-a",
                    "--checksum",
                    "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
                    "--partial",
                    "--protect-args",
                    "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                    "-e",
                    ssh_transport,
                    "--",
                    f"{source}/",
                    f"aday@{network_address(host)}:{target}/",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        if transfer.returncode != 0:
            raise QwenSpeedLabError("worker production model staging failed")
        return target

    def prepare_storage(self, context: RuntimeContext) -> StoragePreparationResult:
        if (
            _RUNTIME_ID_RE.fullmatch(context.runtime_id) is None
            or context.job_id is None
        ):
            raise QwenSpeedLabError("speed-lab runtime/job identity is malformed")
        payload = self._payload(context.payload)
        variant_config = VARIANT_CONFIGS[str(payload["variant"])]
        draft = DRAFT_ARTIFACTS[str(variant_config["draft_id"])]
        model = MODEL_ARTIFACTS[str(variant_config["model_id"])]
        sources = _source_manifest()
        prompt, prompt_sources = _prompt_bundle()
        self._profile_identity(context, sources, prompt, variant_config)
        host = context.lease.host
        host_config = HOST_CONFIGS[host]
        hostname = str(host_config["hostname"])
        scratch = str(context.scratch_path)
        source_root = f"{scratch}/source"
        request_path = f"{scratch}/speed-lab-request.json"
        before_device, _before_free, _before_inodes, before_allocated = _remote_metrics(
            host, scratch, create=True
        )
        self._stage_sources(host, hostname, scratch, sources)
        local_prefix = context.run_dir / "system-prefix.txt"
        self._write_private(local_prefix, prompt)
        self._stage_file(host, local_prefix, f"{scratch}/system-prefix.txt")
        model_dir = str(model["model_dir"] or "")
        if model.get("stage_per_attempt") is True:
            model_dir = self._stage_production_model(
                context, host, hostname, scratch, model
            )
        if not model_dir:
            raise QwenSpeedLabError("speed-lab model path is unavailable")
        feature_capture = bool(variant_config["feature_capture"])
        feature_dataset_path: str | None = None
        feature_dataset_bytes: int | None = None
        feature_dataset_rows: int | None = None
        feature_dataset_sha256: str | None = None
        if feature_capture:
            metadata = FEATURE_DATASET.lstat()
            payload_bytes = FEATURE_DATASET.read_bytes()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
                or _sha256(FEATURE_DATASET) != FEATURE_DATASET_SHA256
                or len(payload_bytes.splitlines()) != FEATURE_DATASET_ROWS
            ):
                raise QwenSpeedLabError("feature dataset identity changed")
            feature_dataset_path = f"{scratch}/feature-train.jsonl"
            feature_dataset_bytes = metadata.st_size
            feature_dataset_rows = FEATURE_DATASET_ROWS
            feature_dataset_sha256 = FEATURE_DATASET_SHA256
            self._stage_file(host, FEATURE_DATASET, feature_dataset_path)
        request = {
            "schema_version": WORKER_SCHEMA_VERSION,
            "runtime_id": context.runtime_id,
            "job_id": context.job_id,
            "host": host,
            "hostname": hostname,
            "claim_id": context.lease.claim_id,
            "owner": context.lease.owner,
            "physical_gpu": context.lease.physical_gpu,
            "gpu_uuid": context.lease.gpu_uuid,
            "vram_budget_gb": context.lease.vram_budget_gb,
            "exclusive": context.lease.exclusive,
            "feature_dataset_bytes": feature_dataset_bytes,
            "feature_dataset_path": feature_dataset_path,
            "feature_dataset_rows": feature_dataset_rows,
            "feature_dataset_sha256": feature_dataset_sha256,
            "scratch_path": scratch,
            "source_root": source_root,
            "source_files": sources,
            "model_id": variant_config["model_id"],
            "model_dir": model_dir,
            "model_manifest_sha256": model["manifest_sha256"],
            "model_sha256s_sha256": model["sha256s_sha256"],
            "image_id": IMAGE_ID,
            "image_size_bytes": IMAGE_SIZE_BYTES,
            "engine_archive_sha256": ENGINE_ARCHIVE_SHA256,
            "engine_closure": (
                ENGINE_CLOSURE_IDENTITY
                if host == PRODUCTION_K3_CANARY_HOST
                else None
            ),
            "draft_id": variant_config["draft_id"],
            "draft_model_dir": draft["model_dir"],
            "draft_revision": draft["revision"],
            "draft_config_sha256": draft["config_sha256"],
            "draft_model_sha256": draft["model_sha256"],
            "container_name": f"aeon-speed-{context.runtime_id}",
            "port": PORT,
            "variant": payload["variant"],
            "runtime": {
                "attention_backend": variant_config["attention_backend"],
                "async_scheduling": False,
                "compilation_profile": variant_config["compilation_profile"],
                "context_tokens": variant_config["context_tokens"],
                "cuda_launch_blocking": False,
                "dspark_draft_topk": variant_config["dspark_draft_topk"],
                "enable_adaptive_verification": variant_config[
                    "enable_adaptive_verification"
                ],
                "enable_flashinfer_autotune": _host_runtime_value(
                    variant_config, host_config, "enable_flashinfer_autotune"
                ),
                "enable_prefix_caching": variant_config["enable_prefix_caching"],
                "enable_per_request_metrics": variant_config[
                    "enable_per_request_metrics"
                ],
                "feature_capture": variant_config["feature_capture"],
                "gdn_decode_kernel": "cuda",
                "gpu_memory_utilization": _host_runtime_value(
                    variant_config, host_config, "gpu_memory_utilization"
                ),
                "kv_cache_dtype": variant_config["kv_cache_dtype"],
                "local_argmax_reduction": (
                    variant_config["speculative_method"] == "mtp"
                ),
                "mamba_cache_dtype": variant_config["mamba_cache_dtype"],
                "mamba_cache_mode": "align",
                "mamba_ssm_cache_dtype": variant_config["mamba_ssm_cache_dtype"],
                "max_batched_tokens": _host_runtime_value(
                    variant_config, host_config, "max_batched_tokens"
                ),
                "max_num_seqs": 1,
                "model_runner": "v2",
                "nvfp4_a16": model["nvfp4_a16"],
                "relaxed_greedy_logit_margin": variant_config[
                    "relaxed_greedy_logit_margin"
                ],
                "speculative_method": variant_config["speculative_method"],
                "speculative_tokens": variant_config["speculative_tokens"],
                "use_flashinfer_sampler": _host_runtime_value(
                    variant_config, host_config, "use_flashinfer_sampler"
                ),
            },
            "benchmark": {
                "repeats": payload["repeats"],
                "quality_repeats": payload["quality_repeats"],
                "max_tokens": payload["max_tokens"],
                "sampling_profile": payload["sampling_profile"],
            },
        }
        request_bytes = (
            json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        local_request = context.run_dir / "speed-lab-request.json"
        self._write_private(local_request, request_bytes)
        self._stage_file(host, local_request, request_path)
        with _PreparationHeartbeat(context):
            preflight = _remote_action(
                host,
                source_root,
                "preflight",
                request_path,
                request_sha256,
                timeout=1900,
            )
        if (
            preflight.get("image_id") != IMAGE_ID
            or preflight.get("engine_archive_sha256") != ENGINE_ARCHIVE_SHA256
            or preflight.get("engine_closure")
            != (
                ENGINE_CLOSURE_IDENTITY
                if host == PRODUCTION_K3_CANARY_HOST
                else None
            )
            or preflight.get("enable_per_request_metrics")
            is not variant_config["enable_per_request_metrics"]
            or preflight.get("draft_id") != variant_config["draft_id"]
            or preflight.get("draft_revision") != draft["revision"]
            or preflight.get("draft_config_sha256") != draft["config_sha256"]
            or preflight.get("draft_model_sha256") != draft["model_sha256"]
            or preflight.get("model_manifest_sha256") != model["manifest_sha256"]
            or preflight.get("model_sha256s_sha256") != model["sha256s_sha256"]
            or preflight.get("prompt_fixture_sha256")
            != hashlib.sha256(prompt).hexdigest()
        ):
            raise QwenSpeedLabError("worker speed-lab preflight identity changed")
        filesystem_id, free_bytes, free_inodes, allocated_bytes = _remote_metrics(
            host, scratch, create=False
        )
        if filesystem_id != before_device:
            raise QwenSpeedLabError(
                "worker speed-lab filesystem changed during staging"
            )
        with self._lock:
            self._prepared[context.runtime_id] = {
                "request_sha256": request_sha256,
                "request_path": request_path,
                "source_root": source_root,
                "host": host,
                "prompt_sources": prompt_sources,
            }
        return StoragePreparationResult(
            scratch_path=context.scratch_path,
            filesystem_id=filesystem_id,
            free_bytes_after_stage=free_bytes,
            free_inodes_after_stage=free_inodes,
            staged_bytes=max(0, allocated_bytes - before_allocated),
        )

    def launch(self, context: RuntimeContext) -> LaunchResult:
        with self._lock:
            prepared = self._prepared.get(context.runtime_id)
        if prepared is None:
            raise AdapterLaunchError(
                "speed-lab preflight receipt is absent", process_absent=True
            )
        try:
            result = _remote_action(
                prepared["host"],
                prepared["source_root"],
                "spawn",
                prepared["request_path"],
                prepared["request_sha256"],
                timeout=60,
            )
            pid = result.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
                raise QwenSpeedLabError("speed-lab supervisor PID is malformed")
            context.heartbeat(pid, "Qwen speed-lab bound to exact supervisor PID")
            identity = (
                f"aeon-speed-lab:{context.runtime_id}:"
                f"{prepared['request_sha256']}:{pid}"
            )
            return LaunchResult(pid=pid, process_identity=identity)
        except BaseException as exc:
            try:
                status = _remote_action(
                    prepared["host"],
                    prepared["source_root"],
                    "status",
                    prepared["request_path"],
                    prepared["request_sha256"],
                    timeout=30,
                )
            except Exception:
                raise
            if status.get("state") == "absent":
                raise AdapterLaunchError(
                    f"speed-lab launch failed before process creation: {exc}",
                    process_absent=True,
                ) from exc
            raise

    @staticmethod
    def _runtime_identity(runtime: Mapping[str, Any]) -> tuple[str, str, int]:
        identity = str(runtime.get("process_identity") or "")
        match = _PROCESS_IDENTITY_RE.fullmatch(identity)
        if (
            match is None
            or match.group(1) != runtime.get("runtime_id")
            or int(match.group(3)) != runtime.get("pid")
            or runtime.get("host") not in HOST_CONFIGS
            or PurePosixPath(str(runtime.get("run_dir") or "")).parent
            != REMOTE_RUN_ROOT
        ):
            raise QwenSpeedLabError("speed-lab runtime identity changed")
        return match.group(1), match.group(2), int(match.group(3))

    @classmethod
    def _runtime_action(
        cls,
        runtime: Mapping[str, Any],
        action: str,
        *,
        extra: str | None = None,
        timeout: float = 120,
    ) -> dict[str, Any]:
        runtime_id, digest, _pid = cls._runtime_identity(runtime)
        scratch = str(runtime["run_dir"])
        return _remote_action(
            str(runtime["host"]),
            f"{scratch}/source",
            action,
            f"{scratch}/speed-lab-request.json",
            digest,
            extra=extra,
            timeout=timeout,
        )

    def probe(self, runtime: Mapping[str, Any]) -> ProbeResult:
        try:
            _runtime_id, _digest, pid = self._runtime_identity(runtime)
            status = self._runtime_action(runtime, "status", timeout=60)
        except QwenSpeedLabTransportError:
            raise
        except QwenSpeedLabError as exc:
            return ProbeResult(ProbeState.UNKNOWN, False, False, str(exc))
        state = status.get("state")
        if state == "running":
            if status.get("pid") != pid:
                return ProbeResult(
                    ProbeState.UNKNOWN, False, False, "speed-lab PID identity changed"
                )
            return ProbeResult(
                ProbeState.RUNNING,
                True,
                False,
                f"Qwen speed-lab is {status.get('phase') or 'running'}",
            )
        if state == "completed":
            result = status.get("result") or {}
            feature = result.get("feature_summary") or {}
            if feature:
                note = (
                    "Qwen target-feature extraction complete: "
                    f"{int(feature.get('unique_features') or 0)} documents, "
                    f"{int(feature.get('total_tokens') or 0)} tokens"
                )
                return ProbeResult(ProbeState.COMPLETED, False, True, note)
            speed = result.get("speed_summary") or {}
            note = (
                f"Qwen speed-lab complete: {float(speed.get('median_decode_tps') or 0):.2f} tok/s, "
                f"{float(speed.get('p95_warm_prefix_ttft_seconds') or 0):.3f}s p95 warm TTFT"
            )
            return ProbeResult(ProbeState.COMPLETED, False, True, note)
        if state == "failed":
            detail = str(
                (status.get("result") or {}).get("failure") or "speed-lab failed"
            )
            return ProbeResult(ProbeState.FAILED, False, True, detail[:500])
        if state == "absent":
            return ProbeResult(
                ProbeState.ABSENT, False, True, "speed-lab supervisor is absent"
            )
        return ProbeResult(
            ProbeState.UNKNOWN, False, False, "speed-lab lifecycle is ambiguous"
        )

    def stop(self, runtime: Mapping[str, Any], *, reason: str) -> StopResult:
        try:
            result = self._runtime_action(runtime, "stop", timeout=150)
        except QwenSpeedLabError as exc:
            return StopResult(False, False, str(exc))
        absent = result.get("process_absent") is True
        return StopResult(
            absent, True, reason if absent else "speed-lab is still stopping"
        )

    @staticmethod
    def _local_output_valid(path: Path) -> tuple[bool, str | None]:
        manifest = path / "MANIFEST.sha256"
        if not manifest.is_file():
            return False, None
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise QwenSpeedLabError("canonical speed-lab output directory is unsafe")
        expected = {"MANIFEST.sha256"}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"([a-f0-9]{64})  ([A-Za-z0-9_.-]{1,200})", line)
            if match is None:
                raise QwenSpeedLabError("canonical speed-lab manifest is malformed")
            candidate = path / match.group(2)
            if not candidate.is_file() or _sha256(candidate) != match.group(1):
                raise QwenSpeedLabError("canonical speed-lab output digest changed")
            expected.add(match.group(2))
        actual = {item.name for item in path.iterdir()}
        if actual != expected:
            raise QwenSpeedLabError("canonical speed-lab output file set changed")
        return True, _sha256(manifest)

    @staticmethod
    def _copy_output(host: str, remote: str, local: Path) -> None:
        local.mkdir(mode=0o700, parents=True, exist_ok=True)
        local.chmod(0o700)
        ssh_transport = " ".join(shlex.quote(item) for item in _ssh_base(host)[:-1])
        result = subprocess.run(
            [
                "/usr/bin/bash",
                LOW_PRIORITY,
                "/usr/bin/rsync",
                "-a",
                "--checksum",
                "--protect-args",
                "--rsync-path=/home/aday/bin/fleet-low-priority /usr/bin/rsync",
                "-e",
                ssh_transport,
                "--",
                f"aday@{network_address(host)}:{remote}/",
                f"{local}/",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise QwenSpeedLabError("speed-lab output settlement transfer failed")

    @staticmethod
    def _prelaunch_failure_identity(
        runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> tuple[str, str, str | None]:
        runtime_id = str(runtime.get("runtime_id") or "")
        host = str(runtime.get("host") or "")
        scratch = str(runtime.get("run_dir") or "")
        canonical = Path(str(storage.get("canonical_output_path") or ""))
        if (
            _RUNTIME_ID_RE.fullmatch(runtime_id) is None
            or host not in HOST_CONFIGS
            or PurePosixPath(scratch).parent != REMOTE_RUN_ROOT
            or PurePosixPath(scratch).name != runtime_id
            or str(storage.get("scratch_path") or "") != scratch
            or runtime.get("process_identity") is not None
            or runtime.get("pid") is not None
            or runtime.get("process_absent") not in {1, True}
            or storage.get("terminal_success") not in {0, False}
            or not str(storage.get("terminal_note") or "").startswith(
                "storage preparation failed before launch:"
            )
            or canonical.exists()
        ):
            raise QwenSpeedLabError("prelaunch failure identity changed")
        request_path = Path(scratch) / "speed-lab-request.json"
        try:
            metadata = request_path.lstat()
        except FileNotFoundError:
            # Payload validation can fail before either the local request or
            # remote scratch exists. The cleanup helper may prove only that
            # the exact remote runtime path is absent; it refuses any existing
            # directory when no request digest is available.
            return host, scratch, None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or not 1 <= metadata.st_size <= 2 * 1024 * 1024
        ):
            raise QwenSpeedLabError("prelaunch request identity is unsafe")
        payload = request_path.read_bytes()
        try:
            request = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise QwenSpeedLabError("prelaunch request is malformed") from exc
        if (
            not isinstance(request, dict)
            or request.get("runtime_id") != runtime_id
            or request.get("host") != host
            or request.get("scratch_path") != scratch
            or request.get("source_root") != f"{scratch}/source"
        ):
            raise QwenSpeedLabError("prelaunch request binding changed")
        return host, scratch, hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _cleanup_prelaunch_scratch(
        host: str, scratch: str, request_sha256: str | None, runtime_id: str
    ) -> int:
        hostname = str(HOST_CONFIGS[host]["hostname"])
        script = r"""
import hashlib, json, os, pathlib, shutil, stat, sys
scratch, expected_host, expected_name, expected_sha, runtime_id = sys.argv[1:]
if os.uname().nodename != expected_name:
    raise SystemExit("hostname changed")
root = pathlib.Path(scratch)
if root.parent != pathlib.Path("/home/aday/.local/state/fleet-compute/runs") or root.name != runtime_id:
    raise SystemExit("scratch path changed")
try:
    root_meta = root.lstat()
except FileNotFoundError:
    print(json.dumps({"state": "absent", "reclaimed_bytes": 0}, sort_keys=True))
    raise SystemExit(0)
if not stat.S_ISDIR(root_meta.st_mode) or root_meta.st_uid != os.geteuid() or root_meta.st_mode & 0o077:
    raise SystemExit("scratch root is unsafe")
request_path = root / "speed-lab-request.json"
request_meta = request_path.lstat()
payload = request_path.read_bytes()
if (not stat.S_ISREG(request_meta.st_mode) or request_meta.st_uid != os.geteuid()
        or request_meta.st_mode & 0o077 or hashlib.sha256(payload).hexdigest() != expected_sha):
    raise SystemExit("remote request identity changed")
request = json.loads(payload)
if (request.get("runtime_id") != runtime_id or request.get("host") != expected_host
        or request.get("scratch_path") != scratch or request.get("source_root") != f"{scratch}/source"):
    raise SystemExit("remote request binding changed")
for relative in (
    "worker-state.json", "container.cid", "supervisor.log", "settled.json",
    "output/result.json", "output/MANIFEST.sha256", "output/server.log",
    "output/speed.json", "output/quality.json",
):
    if (root / relative).exists() or (root / relative).is_symlink():
        raise SystemExit("runtime lifecycle evidence exists")
reclaimed = root_meta.st_size
for path in root.rglob("*"):
    meta = path.lstat()
    if meta.st_uid != os.geteuid() or stat.S_ISLNK(meta.st_mode):
        raise SystemExit("scratch contains an unsafe inode")
    if not (stat.S_ISDIR(meta.st_mode) or stat.S_ISREG(meta.st_mode)):
        raise SystemExit("scratch contains a special inode")
    reclaimed += meta.st_size
shutil.rmtree(root)
print(json.dumps({"state": "cleaned", "reclaimed_bytes": reclaimed}, sort_keys=True))
"""
        result = subprocess.run(
            [
                *_ssh_base(host),
                shlex.join(
                    [
                        "/usr/bin/bash",
                        LOW_PRIORITY,
                        REMOTE_PYTHON,
                        "-c",
                        script,
                        scratch,
                        host,
                        hostname,
                        request_sha256 or "",
                        runtime_id,
                    ]
                ),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=300,
        )
        try:
            value = json.loads(result.stdout)
            reclaimed = value["reclaimed_bytes"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise QwenSpeedLabError("prelaunch cleanup receipt is malformed") from exc
        if (
            result.returncode != 0
            or value.get("state") not in {"absent", "cleaned"}
            or isinstance(reclaimed, bool)
            or not isinstance(reclaimed, int)
            or reclaimed < 0
        ):
            raise QwenSpeedLabError("prelaunch scratch cleanup was not verified")
        return reclaimed

    def finalize_storage(
        self, runtime: Mapping[str, Any], storage: Mapping[str, Any]
    ) -> StorageFinalizationResult:
        if runtime.get("process_identity") is None:
            host, scratch, request_sha256 = self._prelaunch_failure_identity(
                runtime, storage
            )
            runtime_id = str(runtime["runtime_id"])
            reclaimed = self._cleanup_prelaunch_scratch(
                host, scratch, request_sha256, runtime_id
            )
            return StorageFinalizationResult(
                True,
                True,
                reclaimed,
                "Qwen speed-lab prelaunch scratch cleaned; no process was created",
            )
        self._runtime_identity(runtime)
        canonical = Path(str(storage["canonical_output_path"]))
        valid, local_manifest_sha = (
            self._local_output_valid(canonical) if canonical.exists() else (False, None)
        )
        scratch = str(runtime["run_dir"])
        host = str(runtime["host"])
        try:
            _remote_metrics(host, scratch, create=False)
        except FileNotFoundError:
            if valid:
                return StorageFinalizationResult(
                    True,
                    True,
                    0,
                    "speed-lab output already settled and worker scratch absent",
                )
            raise QwenSpeedLabError("worker scratch vanished before output settlement")
        status = self._runtime_action(runtime, "settle-status", timeout=120)
        manifest_sha = str(status.get("manifest_sha256") or "")
        if not re.fullmatch(r"[a-f0-9]{64}", manifest_sha):
            raise QwenSpeedLabError("worker output manifest identity is malformed")
        if not valid:
            self._copy_output(host, f"{scratch}/output", canonical)
            valid, local_manifest_sha = self._local_output_valid(canonical)
        if not valid or local_manifest_sha != manifest_sha:
            raise QwenSpeedLabError(
                "settled speed-lab output differs from worker manifest"
            )
        self._runtime_action(runtime, "mark-settled", extra=manifest_sha, timeout=60)
        cleaned = self._runtime_action(
            runtime, "cleanup", extra=manifest_sha, timeout=300
        )
        reclaimed = cleaned.get("reclaimed_bytes")
        if (
            isinstance(reclaimed, bool)
            or not isinstance(reclaimed, int)
            or reclaimed < 0
        ):
            raise QwenSpeedLabError("worker cleanup receipt is malformed")
        return StorageFinalizationResult(
            True, True, reclaimed, "Qwen speed-lab output settled on .177"
        )


def create_fleet_adapter() -> AeonQwenSpeedLabAdapter:
    return AeonQwenSpeedLabAdapter()
