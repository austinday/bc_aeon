#!/usr/bin/env python3
"""Exact worker-side supervisor for one Fleet-managed Qwen speed experiment.

The Fleet broker has already selected and leased the physical GPU.  This script
never allocates compute itself: it validates one immutable request, launches one
UUID-bound container, runs the allowlisted benchmark suite, and removes only the
container whose full identity it recorded.  It never lists or inspects unrelated
containers.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any

import requests

from aeon.core.engine_closure import (
    EngineClosureError,
    closure_request_identity,
    load_engine_closure_receipt,
    verify_engine_closure,
    verify_regular_file_identity,
)
from aeon.core.sampling import QWEN_SPEED_LAB_SAMPLING_PROFILES


SCHEMA_VERSION = "aeon-qwen38-speed-worker-v18"
EXPECTED_HOSTNAME = "DAY2XRTX5000PRO-2"
EXPECTED_FEATURE_DATASET_SHA256 = (
    "61b8e150651ecc14c47e1068ce36fc130bb56e18117b3b68e098390defea92f5"
)
EXPECTED_FEATURE_DATASET_ROWS = 256
HOST_CONFIGS = {
    "192.168.0.179": {
        "enable_flashinfer_autotune": True,
        "hostname": "DAY2XRTX6000-2",
        "gpu_memory_utilization": 0.42,
        "max_batched_tokens": 32768,
        "use_flashinfer_sampler": False,
    },
    "192.168.0.180": {
        "enable_flashinfer_autotune": False,
        "hostname": EXPECTED_HOSTNAME,
        "gpu_memory_utilization": 0.84,
        "max_batched_tokens": 8192,
        "use_flashinfer_sampler": True,
    },
}
BARE_HOST = "192.168.0.179"
ENGINE_ROOT = Path(
    "/home/aday/.aeon/runtime/qwen38/engines/"
    "604c2525974bf41416e76c1f34ed014a1393d55617b4c7d7fc05d6c93754d9eb/venv"
)
ENGINE_SITE = ENGINE_ROOT / "lib/python3.12/site-packages"
ENGINE_CLOSURE_RECEIPT = load_engine_closure_receipt(
    Path(__file__).resolve().parents[1]
    / "core/data/qwen38_v026_dev1141_engine_closure.json"
)
EXPECTED_ENGINE_CLOSURE = closure_request_identity(ENGINE_CLOSURE_RECEIPT)
ENGINE_SENTINELS = {
    "vllm/__init__.py": "fd3708e5a13abe98c566afd79a6b1987cef70c0548bb19c1218ff6a7f9d43346",
    "vllm/_C_stable_libtorch.abi3.so": "3a47324e0f242f07e0e67d4384214353b2112d5d42dc9373053926c8b395f48f",
    "vllm/model_executor/layers/logits_processor.py": "e15cd9d2827fa3690ec683d73a6095a4be8189118e9a8e124fd21d4c58930ff2",
    "vllm/model_executor/models/qwen3_5.py": "4d79b1cd10637f321e47b16a56cb977f3db216b43932d156df3520515d9b18df",
    "vllm/model_executor/models/qwen3_5_mtp.py": "f2fd77778a01e8fb2721d3798723441ac334d8215ab3ad9c811629054637646e",
    "vllm/model_executor/models/qwen3_dflash.py": "ac252548d5156d959be230b914e7460f3c903b86ac555eecf7d44c9607b4ce4e",
    "vllm/model_executor/models/qwen3_dflash2.py": "c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f",
    "vllm/v1/spec_decode/dflash.py": "bfade48ec7c8e945d78741feaf66bb26d497ba454d4635c296656dada5b0eb8a",
    "vllm/v1/spec_decode/step3p5.py": "fea9bd56e492c87e8b02a94ab15e1d32b196d5df3b4c719ca8318df7f5b6ae13",
    "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py": "1f6ff5ca9c8f38ff417aafd43bfa3116b5387bf0f7b58721acb2185781879836",
}
SCRATCH_ROOT = PurePosixPath("/home/aday/.local/state/fleet-compute/runs")
REMOTE_PYTHON = Path(
    "/home/aday/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
)
LOW_PRIORITY = Path("/home/aday/bin/fleet-low-priority")
DOCKER = Path("/home/aday/bin/docker")
BASH = Path("/usr/bin/bash")
GETFACL = Path("/usr/bin/getfacl")
SERVED_MODEL = "Qwen3.8-27B-ARA-NVFP4-MTP"
EXPECTED_IMAGE_ID = (
    "sha256:604c2525974bf41416e76c1f34ed014a1393d55617b4c7d7fc05d6c93754d9eb"
)
EXPECTED_IMAGE_SIZE_BYTES = 4_604_212_353
EXPECTED_ENGINE_ARCHIVE_SHA256 = str(ENGINE_CLOSURE_RECEIPT["archive_sha256"])
DISABLED_KERNELS = "FlashInferFP8ScaledMMLinearKernel"
EXPECTED_DRAFT_ARTIFACTS = {
    "bf16": {
        "revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
        "revision_sha256": (
            "0e390b23b8c018ade89fdedcc892071b22571be473385b74e1e21183d50623f5"
        ),
        "config_sha256": (
            "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980"
        ),
        "config_bytes": 1_239,
        "model_sha256": (
            "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
        ),
        "model_bytes": 3_848_817_896,
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
        ),
        "quantized": False,
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
        "config_bytes": 1_551,
        "model_sha256": (
            "a23651e35305ff4f83a144380e201e6fcc70ed1e442058112e29e852cd639ca5"
        ),
        "model_bytes": 3_848_818_579,
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "a23651e35305ff4f83a144380e201e6fcc70ed1e442058112e29e852cd639ca5"
        ),
        "quantized": False,
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
        "config_bytes": 1_551,
        "model_sha256": (
            "cf91386d67cfce1c43f2bd312267a7381821ca63415764fbfccceff285cfcf83"
        ),
        "model_bytes": 3_848_818_579,
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "cf91386d67cfce1c43f2bd312267a7381821ca63415764fbfccceff285cfcf83"
        ),
        "quantized": False,
    },
    "w4a16": {
        "revision": "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279",
        "revision_sha256": (
            "8b2f78143963d23aeab94d9242ce38a1654cd3fe4187857550573076f16771df"
        ),
        "config_sha256": (
            "61d6276fe8d76295232cb02d26cbb0d29c25565911f50441e779c88c9220c556"
        ),
        "config_bytes": 2_264,
        "model_sha256": (
            "ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7"
        ),
        "model_bytes": 1_280_633_960,
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7"
        ),
        "quantized": True,
    },
    "w8a16": {
        "revision": "f454fa8e6a84387bf006f849584f72541cc29118",
        "revision_sha256": (
            "53fae14c440666c760f14c56386eaea14afaa626cc2196ea5bf0a39f9d0fef27"
        ),
        "config_sha256": (
            "1c63a70cd7fa0e8be7276b3597e464ab9a0e609b70611354e9fb76e76b2e42d6"
        ),
        "config_bytes": 2_064,
        "model_sha256": (
            "72223787fbfe7e01c4a940f80f98cf3d7a803e9b79aa4ec7bc28a461d767fa18"
        ),
        "model_bytes": 2_172_742_656,
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "72223787fbfe7e01c4a940f80f98cf3d7a803e9b79aa4ec7bc28a461d767fa18"
        ),
        "quantized": True,
    },
    "dspark-nvfp4": {
        "revision": (
            "gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4@"
            "eba1ac5a66c74902eaa95a4000a7c5eda96d8e95"
        ),
        "revision_sha256": (
            "89be99d1047b9999c71e7fe10be3652c00cc6896c107d580224780b05974220a"
        ),
        "config_sha256": (
            "82fd961b632c629736902d9d4fdd3258dee1080f557cf86298cac063a514a0cf"
        ),
        "config_bytes": 2_828,
        "hf_quant_config_sha256": (
            "cda90695e8c4a5eaed7ce7220afbc8bbe18e7624a167466ec7768c603e756a09"
        ),
        "hf_quant_config_bytes": 937,
        "model_sha256": (
            "212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05"
        ),
        "model_bytes": 1_399_670_058,
        "model_dir": (
            "/home/aday/.aeon/runtime/qwen38/drafts/"
            "212fd1b8b5477536ab9e726a94d8565a2246467d044de772f6648df17d5dda05"
        ),
        "quantized": True,
    },
}
EXPECTED_MODEL_ARTIFACTS = {
    "production": {
        "manifest_sha256": (
            "1a3ba1eb88d0507bdef3798a6db59830dc076199b7db7d111201f6997588220e"
        ),
        "sha256s_sha256": (
            "e7eca7ebee03c4f27482d4fe421ca1fac9f1d9986663a51fd7614361010c1237"
        ),
        "model_dir": None,
        "nvfp4_a16": False,
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
PRODUCTION_K3_CANARY_PROFILE_ID = (
    "aeon-qwen38-production-k3-v026-canary-179"
)
PRODUCTION_K3_CANARY_VARIANT = (
    "v026-dev1141-production-nvfp4-mtp-k3-triton-fp8pthead-114688"
)
PRODUCTION_K3_CANARY_VRAM_BUDGET_GB = 48.7
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
    "allowed_hosts": (BARE_HOST,),
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
    _variant_config.setdefault("profile_id", "aeon-qwen38-speed-lab")
    _variant_config.setdefault("relaxed_greedy_logit_margin", "0")
    _variant_config.setdefault("use_flashinfer_sampler_override", None)
    _variant_config.setdefault("vram_budget_gb", 41.25)
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_SERVER_LOG_BYTES = 30 * 1024 * 1024
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_CONTAINER_ID_RE = re.compile(r"^[a-f0-9]{64}$")
_BARE_HANDLE_RE = re.compile(r"^bare-([0-9]{1,12})$")
_RUNTIME_ID_RE = re.compile(r"^fr-[a-f0-9]{32}$")
_CLAIM_RE = re.compile(r"^gc-[A-Za-z0-9._:-]{1,196}$")
_OWNER_RE = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]{8,120}$")


class SpeedLabError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_runtime_value(
    variant_config: dict[str, Any], host_config: dict[str, Any], name: str
) -> Any:
    override = variant_config[f"{name}_override"]
    return host_config[name] if override is None else override


def _private_directory(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise SpeedLabError(f"private directory identity changed: {path}")
    return path


def _bare_rpc_root(request: dict[str, Any]) -> Path:
    runtime_id = str(request["runtime_id"])
    if _RUNTIME_ID_RE.fullmatch(runtime_id) is None:
        raise SpeedLabError("bare RPC runtime identity is malformed")
    path = Path("/dev/shm") / f"aeon-vrpc-{runtime_id[3:19]}"
    if path.parent != Path("/dev/shm"):
        raise SpeedLabError("bare RPC path changed")
    return path


def _cleanup_bare_rpc(request: dict[str, Any]) -> None:
    root = _bare_rpc_root(request)
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise SpeedLabError("bare RPC root is unsafe")
    for path in root.rglob("*"):
        item = path.lstat()
        if item.st_uid != os.geteuid() or stat.S_ISLNK(item.st_mode):
            raise SpeedLabError("bare RPC root contains an unsafe inode")
        if not (
            stat.S_ISDIR(item.st_mode)
            or stat.S_ISREG(item.st_mode)
            or stat.S_ISSOCK(item.st_mode)
        ):
            raise SpeedLabError("bare RPC root contains a special inode")
    shutil.rmtree(root)


def _atomic_json(path: Path, value: Any) -> None:
    _private_directory(path.parent, create=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        payload = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    _private_directory(path.parent, create=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        os.write(descriptor, value.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _read_json(path: Path, *, maximum: int = MAX_REQUEST_BYTES) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 0 < metadata.st_size <= maximum
    ):
        raise SpeedLabError(f"private JSON identity changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeedLabError(f"private JSON is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise SpeedLabError("private JSON is not an object")
    return value


def _validate_request(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise SpeedLabError("request digest is malformed")
    request = _read_json(path)
    if _sha256(path) != expected_sha256:
        raise SpeedLabError("request bytes changed")
    required = {
        "schema_version",
        "runtime_id",
        "job_id",
        "host",
        "hostname",
        "claim_id",
        "owner",
        "physical_gpu",
        "gpu_uuid",
        "vram_budget_gb",
        "exclusive",
        "feature_dataset_bytes",
        "feature_dataset_path",
        "feature_dataset_rows",
        "feature_dataset_sha256",
        "scratch_path",
        "source_root",
        "source_files",
        "model_id",
        "model_dir",
        "model_manifest_sha256",
        "model_sha256s_sha256",
        "image_id",
        "image_size_bytes",
        "engine_archive_sha256",
        "engine_closure",
        "draft_id",
        "draft_model_dir",
        "draft_revision",
        "draft_config_sha256",
        "draft_model_sha256",
        "container_name",
        "port",
        "variant",
        "runtime",
        "benchmark",
    }
    if set(request) != required or request.get("schema_version") != SCHEMA_VERSION:
        raise SpeedLabError("request schema changed")
    runtime_id = request.get("runtime_id")
    host = request.get("host")
    host_config = HOST_CONFIGS.get(host)
    variant = request.get("variant")
    variant_config = VARIANT_CONFIGS.get(str(variant))
    model_id = request.get("model_id")
    model_config = EXPECTED_MODEL_ARTIFACTS.get(str(model_id))
    draft_id = request.get("draft_id")
    draft_config = EXPECTED_DRAFT_ARTIFACTS.get(str(draft_id))
    scratch = PurePosixPath(str(request.get("scratch_path") or ""))
    source_root = PurePosixPath(str(request.get("source_root") or ""))
    expected_model_dir = (
        str(scratch / "production-model")
        if model_config is not None and model_config.get("stage_per_attempt") is True
        else None if model_config is None else model_config["model_dir"]
    )
    if (
        not isinstance(runtime_id, str)
        or _RUNTIME_ID_RE.fullmatch(runtime_id) is None
        or scratch.parent != SCRATCH_ROOT
        or scratch.name != runtime_id
        or source_root != scratch / "source"
        or path != Path(scratch) / "speed-lab-request.json"
        or host_config is None
        or variant_config is None
        or host not in variant_config["allowed_hosts"]
        or request.get("hostname") != host_config["hostname"]
        or socket.gethostname() != host_config["hostname"]
        or not isinstance(request.get("job_id"), str)
        or not request["job_id"]
        or _CLAIM_RE.fullmatch(str(request.get("claim_id") or "")) is None
        or _OWNER_RE.fullmatch(str(request.get("owner") or "")) is None
        or isinstance(request.get("physical_gpu"), bool)
        or request.get("physical_gpu") not in {0, 1}
        or _UUID_RE.fullmatch(str(request.get("gpu_uuid") or "")) is None
        or request.get("exclusive") is not True
        or float(request.get("vram_budget_gb") or 0)
        != float(variant_config["vram_budget_gb"])
        or model_config is None
        or request.get("model_dir") != expected_model_dir
        or request.get("model_manifest_sha256") != model_config["manifest_sha256"]
        or request.get("model_sha256s_sha256") != model_config["sha256s_sha256"]
        or request.get("image_id") != EXPECTED_IMAGE_ID
        or request.get("image_size_bytes") != EXPECTED_IMAGE_SIZE_BYTES
        or request.get("engine_archive_sha256") != EXPECTED_ENGINE_ARCHIVE_SHA256
        or request.get("engine_closure")
        != (EXPECTED_ENGINE_CLOSURE if host == BARE_HOST else None)
        or draft_config is None
        or request.get("draft_model_dir") != draft_config["model_dir"]
        or request.get("draft_revision") != draft_config["revision"]
        or request.get("draft_config_sha256") != draft_config["config_sha256"]
        or request.get("draft_model_sha256") != draft_config["model_sha256"]
        or request.get("container_name") != f"aeon-speed-{runtime_id}"
        or _IMAGE_ID_RE.fullmatch(str(request.get("image_id") or "")) is None
        or not isinstance(request.get("image_size_bytes"), int)
        or not 0 < request["image_size_bytes"] <= 64 * 1024**3
        or _SHA256_RE.fullmatch(str(request.get("model_manifest_sha256") or "")) is None
        or _SHA256_RE.fullmatch(str(request.get("model_sha256s_sha256") or "")) is None
        or not isinstance(request.get("port"), int)
        or request["port"] != 18033
    ):
        raise SpeedLabError("request identity or lease contract changed")
    runtime = request.get("runtime")
    benchmark = request.get("benchmark")
    sources = request.get("source_files")
    feature_capture = bool(variant_config["feature_capture"])
    expected_feature_path = str(scratch / "feature-train.jsonl")
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != {
            "attention_backend",
            "async_scheduling",
            "compilation_profile",
            "context_tokens",
            "cuda_launch_blocking",
            "dspark_draft_topk",
            "enable_adaptive_verification",
            "enable_flashinfer_autotune",
            "enable_prefix_caching",
            "enable_per_request_metrics",
            "feature_capture",
            "gdn_decode_kernel",
            "gpu_memory_utilization",
            "kv_cache_dtype",
            "local_argmax_reduction",
            "mamba_cache_dtype",
            "mamba_cache_mode",
            "mamba_ssm_cache_dtype",
            "max_batched_tokens",
            "max_num_seqs",
            "model_runner",
            "nvfp4_a16",
            "relaxed_greedy_logit_margin",
            "speculative_method",
            "speculative_tokens",
            "use_flashinfer_sampler",
        }
        or runtime
        != {
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
            "local_argmax_reduction": variant_config["speculative_method"] == "mtp",
            "mamba_cache_dtype": variant_config["mamba_cache_dtype"],
            "mamba_cache_mode": "align",
            "mamba_ssm_cache_dtype": variant_config["mamba_ssm_cache_dtype"],
            "max_batched_tokens": _host_runtime_value(
                variant_config, host_config, "max_batched_tokens"
            ),
            "max_num_seqs": 1,
            "model_runner": "v2",
            "nvfp4_a16": model_config["nvfp4_a16"],
            "relaxed_greedy_logit_margin": variant_config[
                "relaxed_greedy_logit_margin"
            ],
            "speculative_method": variant_config["speculative_method"],
            "speculative_tokens": variant_config["speculative_tokens"],
            "use_flashinfer_sampler": _host_runtime_value(
                variant_config, host_config, "use_flashinfer_sampler"
            ),
        }
        or draft_id != variant_config["draft_id"]
        or model_id != variant_config["model_id"]
        or not isinstance(benchmark, dict)
        or set(benchmark)
        != {"max_tokens", "quality_repeats", "repeats", "sampling_profile"}
        or not 3 <= int(benchmark["repeats"]) <= 9
        or not 2 <= int(benchmark["quality_repeats"]) <= 5
        or not 256 <= int(benchmark["max_tokens"]) <= 2048
        or not isinstance(benchmark["sampling_profile"], str)
        or benchmark["sampling_profile"] not in QWEN_SPEED_LAB_SAMPLING_PROFILES
        or not isinstance(sources, dict)
        or not sources
        or (
            feature_capture
            and (
                request.get("feature_dataset_path") != expected_feature_path
                or request.get("feature_dataset_sha256")
                != EXPECTED_FEATURE_DATASET_SHA256
                or request.get("feature_dataset_rows")
                != EXPECTED_FEATURE_DATASET_ROWS
                or isinstance(request.get("feature_dataset_bytes"), bool)
                or not isinstance(request.get("feature_dataset_bytes"), int)
                or not 1 <= request["feature_dataset_bytes"] <= 32 * 1024 * 1024
            )
        )
        or (
            not feature_capture
            and any(
                request.get(key) is not None
                for key in (
                    "feature_dataset_bytes",
                    "feature_dataset_path",
                    "feature_dataset_rows",
                    "feature_dataset_sha256",
                )
            )
        )
    ):
        raise SpeedLabError("request variant configuration changed")
    for relative, digest in sources.items():
        parts = PurePosixPath(str(relative)).parts
        if (
            not isinstance(relative, str)
            or not parts
            or relative.startswith("/")
            or ".." in parts
            or _SHA256_RE.fullmatch(str(digest)) is None
        ):
            raise SpeedLabError("request source manifest is malformed")
    return request


def _paths(request: dict[str, Any]) -> dict[str, Path]:
    scratch = Path(request["scratch_path"])
    return {
        "scratch": scratch,
        "source": Path(request["source_root"]),
        "output": scratch / "output",
        "state": scratch / "worker-state.json",
        "spawn": scratch / "supervisor-spawn.json",
        "terminal": scratch / "output/result.json",
        "preflight": scratch / "output/preflight.json",
        "manifest": scratch / "output/MANIFEST.sha256",
        "settled": scratch / "settled.json",
        "cidfile": scratch / "container.cid",
        "server_log": scratch / "output/server.log",
        "supervisor_log": scratch / "supervisor.log",
        "speed": scratch / "output/speed.json",
        "quality": scratch / "output/quality.json",
        "feature_archive": scratch / "output/features.tar.zst",
        "feature_dataset": scratch / "feature-train.jsonl",
        "feature_dir": scratch / "output/features",
        "feature_index": scratch / "output/feature-index.json",
        "prefix": scratch / "system-prefix.txt",
    }


def _verify_sources(request: dict[str, Any]) -> None:
    root = _private_directory(Path(request["source_root"]))
    expected = request["source_files"]
    actual: set[str] = set()
    for path in root.rglob("*"):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
                raise SpeedLabError("source tree contains a mutable directory")
            continue
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise SpeedLabError("source tree contains an unsafe inode")
        actual.add(relative)
    if actual != set(expected):
        raise SpeedLabError("source tree file set changed")
    for relative, digest in expected.items():
        if _sha256(root / relative) != digest:
            raise SpeedLabError(f"source digest changed: {relative}")


def _docker_environment(request: dict[str, Any]) -> dict[str, str]:
    config = Path(request["scratch_path"]) / "docker-cli-empty"
    _private_directory(config, create=True)
    if any(config.iterdir()):
        raise SpeedLabError("private Docker CLI configuration is not empty")
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C",
        "LC_ALL": "C",
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "DOCKER_CONFIG": str(config),
    }


def _docker(
    request: dict[str, Any],
    *arguments: str,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BASH), str(DOCKER), *arguments],
        env=_docker_environment(request),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _verify_bare_engine(
    request: dict[str, Any], *, full_closure: bool = False
) -> dict[str, Any] | None:
    if request["host"] != BARE_HOST:
        raise SpeedLabError("bare engine is not authorized on this host")
    try:
        verify_regular_file_identity(
            REMOTE_PYTHON,
            expected_sha256=ENGINE_CLOSURE_RECEIPT["python_executable_sha256"],
            expected_bytes=ENGINE_CLOSURE_RECEIPT["python_executable_bytes"],
        )
    except EngineClosureError as exc:
        raise SpeedLabError("bare Python executable identity changed") from exc

    closure = None
    if full_closure:
        try:
            verify_engine_closure(ENGINE_ROOT, ENGINE_CLOSURE_RECEIPT)
            closure = dict(request["engine_closure"])
        except EngineClosureError as exc:
            raise SpeedLabError("bare engine full closure identity changed") from exc
    for root in (ENGINE_ROOT, ENGINE_SITE):
        metadata = root.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise SpeedLabError("bare engine root is mutable or unowned")
    for relative, expected in ENGINE_SENTINELS.items():
        path = ENGINE_SITE / relative
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
            or _sha256(path) != expected
        ):
            raise SpeedLabError(f"bare engine identity changed: {relative}")
    result = subprocess.run(
        [
            str(BASH),
            str(LOW_PRIORITY),
            str(REMOTE_PYTHON),
            "-c",
            (
                "import sys,sysconfig,torch,vllm; "
                f"assert sys.version == {ENGINE_CLOSURE_RECEIPT['python_version']!r}; "
                "assert sys.implementation.cache_tag == "
                f"{ENGINE_CLOSURE_RECEIPT['python_cache_tag']!r}; "
                "assert sysconfig.get_config_var('SOABI') == "
                f"{ENGINE_CLOSURE_RECEIPT['python_soabi']!r}; "
                "assert vllm.__version__ == "
                "'0.26.1rc1.dev1141+g0ecc28479'; "
                "assert torch.__version__ == '2.13.0+cu130'; "
                "assert torch.version.cuda == '13.0'; "
                "print('exact_bare_engine_ok')"
            ),
        ],
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/aday",
            "LANG": "C",
            "LC_ALL": "C",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(ENGINE_SITE),
            "USE_TF": "0",
            "USE_FLAX": "0",
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or result.stdout.strip() != "exact_bare_engine_ok":
        raise SpeedLabError("bare engine import identity changed")
    if request["runtime"]["enable_per_request_metrics"]:
        help_result = subprocess.run(
            [
                str(BASH),
                str(LOW_PRIORITY),
                str(REMOTE_PYTHON),
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--help",
            ],
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/home/aday",
                "LANG": "C",
                "LC_ALL": "C",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(ENGINE_SITE),
                "USE_TF": "0",
                "USE_FLAX": "0",
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if (
            help_result.returncode != 0
            or "--enable-per-request-metrics" not in help_result.stdout
        ):
            raise SpeedLabError("bare engine per-request metrics option is unavailable")
    return closure


def _inspect_image(
    request: dict[str, Any], *, full_bare_engine: bool = False
) -> dict[str, Any] | None:
    if request["host"] == BARE_HOST:
        return _verify_bare_engine(request, full_closure=full_bare_engine)
    result = _docker(request, "image", "inspect", request["image_id"], timeout=30)
    try:
        payload = json.loads(result.stdout)
        image = payload[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise SpeedLabError("speed image identity is unreadable") from exc
    if (
        result.returncode != 0
        or image.get("Id") != request["image_id"]
        or image.get("Size") != request["image_size_bytes"]
    ):
        raise SpeedLabError("speed image identity changed")
    return None


def _inspect_container(
    request: dict[str, Any], container_id: str
) -> dict[str, Any] | None:
    if _CONTAINER_ID_RE.fullmatch(container_id) is None:
        raise SpeedLabError("container ID receipt is malformed")
    result = _docker(request, "inspect", container_id, timeout=30)
    if result.returncode != 0:
        diagnostic = result.stderr.lower()
        if (
            result.returncode == 1
            and result.stdout in {"", "[]\n"}
            and "no such" in diagnostic
            and container_id in diagnostic
        ):
            daemon = _docker(
                request, "info", "--format", "{{.ServerVersion}}", timeout=20
            )
            if daemon.returncode == 0 and daemon.stdout.strip():
                return None
        raise SpeedLabError("container presence is ambiguous")
    try:
        payload = json.loads(result.stdout)
        item = payload[0]
    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        raise SpeedLabError("container identity is unreadable") from exc
    if item.get("Id") != container_id:
        raise SpeedLabError("container identity changed")
    labels = (item.get("Config") or {}).get("Labels") or {}
    environment = (item.get("Config") or {}).get("Env") or []
    if (
        labels.get("com.bc_aeon.component") != "qwen38-speed-lab"
        or labels.get("com.bc_aeon.runtime-id") != request["runtime_id"]
        or labels.get("com.bc_aeon.claim") != request["claim_id"]
        or labels.get("com.bc_aeon.variant") != request["variant"]
        or f"GPU_AGENT_CLAIM_ID={request['claim_id']}" not in environment
        or f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}" not in environment
        or f"AEON_SPEED_LAB_RUNTIME_ID={request['runtime_id']}" not in environment
        or f"AEON_SPEED_LAB_MODEL_ID={request['model_id']}" not in environment
        or (f"AEON_SPEED_LAB_MODEL_SHA256S={request['model_sha256s_sha256']}")
        not in environment
        or f"AEON_SPEED_LAB_DRAFT_SHA256={request['draft_model_sha256']}"
        not in environment
        or f"AEON_SPEED_LAB_PORT={request['port']}" not in environment
        or f"VLLM_DISABLED_KERNELS={DISABLED_KERNELS}" not in environment
        or (f"AEON_NVFP4_A16={1 if request['runtime']['nvfp4_a16'] else 0}")
        not in environment
        or (
            "AEON_RELAXED_GREEDY_LOGIT_MARGIN="
            f"{request['runtime']['relaxed_greedy_logit_margin']}"
        )
        not in environment
        or item.get("Image") != request["image_id"]
    ):
        raise SpeedLabError("container contract identity changed")
    return item


def _verify_model(request: dict[str, Any]) -> dict[str, Any]:
    root = Path(request["model_dir"])
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise SpeedLabError("model root is mutable or unowned")
    build = root / "BUILD_MANIFEST.json"
    sums = root / "SHA256SUMS"
    if (
        _sha256(build) != request["model_manifest_sha256"]
        or _sha256(sums) != request["model_sha256s_sha256"]
    ):
        raise SpeedLabError("model manifest identity changed")
    total = 0
    expected: set[str] = {"SHA256SUMS"}
    for line in sums.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"[a-f0-9]{64} [ *](.+)", line)
        if match is None:
            raise SpeedLabError("model checksum manifest is malformed")
        relative = match.group(1)
        parts = PurePosixPath(relative).parts
        if not parts or relative.startswith("/") or ".." in parts:
            raise SpeedLabError("model checksum path is unsafe")
        expected.add(relative)
    actual: set[str] = set()
    for path in root.rglob("*"):
        item = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(item.st_mode):
            if item.st_uid != os.geteuid() or item.st_mode & 0o022:
                raise SpeedLabError("model tree contains a mutable directory")
            continue
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != os.geteuid()
            or item.st_mode & 0o022
        ):
            raise SpeedLabError("model tree contains an unsafe inode")
        actual.add(relative)
        total += item.st_size
    if actual != expected or not 0 < total <= 128 * 1024**3:
        raise SpeedLabError("model artifact file set changed")
    check = subprocess.run(
        [
            str(BASH),
            str(LOW_PRIORITY),
            "/usr/bin/sha256sum",
            "--check",
            "--strict",
            "SHA256SUMS",
        ],
        cwd=root,
        env={"PATH": "/usr/bin:/bin", "HOME": "/home/aday", "LANG": "C", "LC_ALL": "C"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
    )
    if check.returncode != 0:
        raise SpeedLabError("model payload checksum verification failed")
    return {"files": len(actual), "bytes": total}


def _verify_draft_model(request: dict[str, Any]) -> dict[str, Any]:
    root = Path(request["draft_model_dir"])
    draft = EXPECTED_DRAFT_ARTIFACTS[request["draft_id"]]
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        raise SpeedLabError("draft root is mutable or unowned")
    expected = {
        "config.json": (
            request["draft_config_sha256"],
            draft["config_bytes"],
        ),
        "model.safetensors": (
            request["draft_model_sha256"],
            draft["model_bytes"],
        ),
    }
    if "hf_quant_config_sha256" in draft:
        expected["hf_quant_config.json"] = (
            draft["hf_quant_config_sha256"],
            draft["hf_quant_config_bytes"],
        )
    actual = {path.name for path in root.iterdir()}
    if actual != set(expected):
        raise SpeedLabError("draft artifact file set changed")
    total = 0
    for name, (expected_sha256, expected_bytes) in expected.items():
        path = root / name
        item = path.lstat()
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != os.geteuid()
            or item.st_nlink != 1
            or item.st_mode & 0o022
            or item.st_size != expected_bytes
            or _sha256(path) != expected_sha256
        ):
            raise SpeedLabError(f"draft identity changed: {name}")
        total += item.st_size
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeedLabError("draft config is unreadable") from exc
    common_changed = (
        config.get("hidden_size") != 5120
        or config.get("num_hidden_layers") != 5
        or config.get("num_target_layers") != 64
        or config.get("vocab_size") != 248320
        or bool(config.get("quantization_config")) != draft["quantized"]
    )
    if request["draft_id"] == "dspark-nvfp4":
        dflash = config.get("dflash_config") or {}
        quantization = config.get("quantization_config") or {}
        dspark_changed = (
            config.get("architectures") != ["Qwen3DSparkModel"]
            or config.get("block_size") != 7
            or config.get("draft_vocab_size") != 248320
            or config.get("enable_confidence_head") is not True
            or config.get("confidence_head_with_markov") is not True
            or config.get("markov_head_type") != "vanilla"
            or config.get("markov_rank") != 256
            or dflash.get("mask_token_id") != 248077
            or dflash.get("projector_type") != "dspark"
            or dflash.get("target_layer_ids") != [4, 16, 28, 40, 52]
            or quantization.get("quant_method") != "modelopt"
            or quantization.get("quant_algo") != "NVFP4"
        )
        try:
            hf_quantization = json.loads(
                (root / "hf_quant_config.json").read_text(encoding="utf-8")
            ).get("quantization", {})
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpeedLabError("DSpark quantization config is unreadable") from exc
        if (
            common_changed
            or dspark_changed
            or hf_quantization.get("quant_algo") != "NVFP4"
            or hf_quantization.get("group_size") != 16
            or hf_quantization.get("kv_cache_quant_algo") is not None
        ):
            raise SpeedLabError("DSpark draft architecture changed")
    elif (
        common_changed
        or config.get("architectures") != ["DFlash2DraftModel"]
        or (config.get("dflash_config") or {}).get("block_size") != 8
        or (config.get("dflash_config") or {}).get("mask_token_id") != 248070
    ):
        raise SpeedLabError("DFlash2 draft architecture changed")
    return {"files": len(expected), "bytes": total}


def _verify_acl(request: dict[str, Any]) -> None:
    device = Path(f"/dev/nvidia{request['physical_gpu']}")
    metadata = device.lstat()
    if not stat.S_ISCHR(metadata.st_mode):
        raise SpeedLabError("leased physical device node is unavailable")
    result = subprocess.run(
        [str(GETFACL), "-cp", "--", str(device)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0 or "user:aday:---" in result.stdout.splitlines():
        raise SpeedLabError("leased GPU is renter-blocked or ambiguous")


def _preflight(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    _private_directory(paths["scratch"])
    _private_directory(paths["output"], create=True)
    _verify_sources(request)
    prefix = paths["prefix"]
    prefix_stat = prefix.lstat()
    if (
        not stat.S_ISREG(prefix_stat.st_mode)
        or prefix_stat.st_uid != os.geteuid()
        or prefix_stat.st_mode & 0o077
        or not 4096 <= prefix_stat.st_size <= 2 * 1024 * 1024
    ):
        raise SpeedLabError("prompt fixture identity is unsafe")
    if request["runtime"]["feature_capture"]:
        dataset = paths["feature_dataset"]
        dataset_metadata = dataset.lstat()
        if (
            not stat.S_ISREG(dataset_metadata.st_mode)
            or dataset_metadata.st_uid != os.geteuid()
            or dataset_metadata.st_nlink != 1
            or dataset_metadata.st_mode & 0o077
            or dataset_metadata.st_size != request["feature_dataset_bytes"]
            or _sha256(dataset) != request["feature_dataset_sha256"]
            or sum(1 for _line in dataset.open("rb"))
            != request["feature_dataset_rows"]
        ):
            raise SpeedLabError("feature dataset identity changed on worker")
    engine_closure = _inspect_image(request, full_bare_engine=True)
    model = _verify_model(request)
    draft = _verify_draft_model(request)
    _verify_acl(request)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "image_id": request["image_id"],
        "image_size_bytes": request["image_size_bytes"],
        "engine_archive_sha256": request["engine_archive_sha256"],
        "engine_closure": engine_closure,
        "enable_per_request_metrics": request["runtime"][
            "enable_per_request_metrics"
        ],
        "draft_id": request["draft_id"],
        "draft_revision": request["draft_revision"],
        "draft_config_sha256": request["draft_config_sha256"],
        "draft_model_sha256": request["draft_model_sha256"],
        "draft_files": draft["files"],
        "draft_bytes": draft["bytes"],
        "model_manifest_sha256": request["model_manifest_sha256"],
        "model_sha256s_sha256": request["model_sha256s_sha256"],
        "model_files": model["files"],
        "model_bytes": model["bytes"],
        "prompt_fixture_sha256": _sha256(prefix),
        "prompt_fixture_bytes": prefix_stat.st_size,
        "source_files": len(request["source_files"]),
        "verified_at": time.time(),
    }
    _atomic_json(paths["preflight"], receipt)
    return {"state": "preflight_ready", **receipt}


def _process_alive(request: dict[str, Any], pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    proc = Path(f"/proc/{pid}")
    try:
        environment = (proc / "environ").read_bytes().split(b"\0")
        command = (proc / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError as exc:
        raise SpeedLabError("worker process identity is unreadable") from exc
    worker = str(Path(request["source_root"]) / "aeon/scripts/qwen_speed_lab_worker.py")
    return (
        f"GPU_AGENT_CLAIM_ID={request['claim_id']}".encode() in environment
        and f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}".encode() in environment
        and f"AEON_SPEED_LAB_RUNTIME_ID={request['runtime_id']}".encode() in environment
        and worker.encode() in command
        and b"run" in command
    )


def _service_process_alive(request: dict[str, Any], pid: int) -> bool:
    """Prove the detached production-service supervisor, not just its PID."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    proc = Path(f"/proc/{pid}")
    try:
        environment = (proc / "environ").read_bytes().split(b"\0")
        command = (proc / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return False
    except OSError as exc:
        raise SpeedLabError("service supervisor identity is unreadable") from exc
    worker = str(Path(request["source_root"]) / "aeon/scripts/qwen_speed_lab_worker.py")
    return (
        f"GPU_AGENT_CLAIM_ID={request['claim_id']}".encode() in environment
        and f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}".encode() in environment
        and f"AEON_SPEED_LAB_RUNTIME_ID={request['runtime_id']}".encode()
        in environment
        and worker.encode() in command
        and b"service-run" in command
    )


def _server_arguments(
    request: dict[str, Any], *, model_path: str, draft_model_path: str, bind_host: str
) -> list[str]:
    runtime = request["runtime"]
    speculative: dict[str, Any] | None = None
    if runtime["speculative_method"] == "mtp":
        speculative = {
            "method": "mtp",
            "num_speculative_tokens": runtime["speculative_tokens"],
            "use_local_argmax_reduction": runtime["local_argmax_reduction"],
        }
    elif runtime["speculative_method"] == "dflash":
        speculative = {
            "method": "dflash",
            "model": draft_model_path,
            "num_speculative_tokens": runtime["speculative_tokens"],
        }
    elif runtime["speculative_method"] == "dspark":
        speculative = {
            "enable_adaptive_verification": runtime["enable_adaptive_verification"],
            "method": "dspark",
            "model": draft_model_path,
            "num_speculative_tokens": runtime["speculative_tokens"],
        }
        if runtime["dspark_draft_topk"] is not None:
            speculative["dspark_draft_topk"] = runtime["dspark_draft_topk"]
    command = [
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        bind_host,
        "--port",
        str(request["port"]),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        f"{runtime['gpu_memory_utilization']:g}",
        "--attention-backend",
        runtime["attention_backend"],
        "--enable-chunked-prefill",
        "--no-async-scheduling",
        "--mamba-cache-mode",
        runtime["mamba_cache_mode"],
        "--mamba-cache-dtype",
        runtime["mamba_cache_dtype"],
        "--mamba-ssm-cache-dtype",
        runtime["mamba_ssm_cache_dtype"],
        "--max-model-len",
        str(runtime["context_tokens"]),
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--reasoning-parser",
        "qwen3",
        "--structured-outputs-config.enable_in_reasoning=False",
        "--max-num-seqs",
        str(runtime["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(runtime["max_batched_tokens"]),
        "--kv-cache-dtype",
        runtime["kv_cache_dtype"],
    ]
    command.append(
        "--enable-prefix-caching"
        if runtime["enable_prefix_caching"]
        else "--no-enable-prefix-caching"
    )
    if runtime["enable_per_request_metrics"]:
        command.append("--enable-per-request-metrics")
    command.append(
        "--enable-flashinfer-autotune"
        if runtime["enable_flashinfer_autotune"]
        else "--no-enable-flashinfer-autotune"
    )
    compilation_profile = runtime["compilation_profile"]
    if compilation_profile == "flashinfer-native-full":
        # XQA on SM120 is fast, but its mutable decode metadata is not safe in
        # the pinned runtime's FULL CUDA graph. Keep full graph capture while
        # selecting FlashInfer's native, replay-safe attention wrapper.
        command += [
            "--attention-config",
            json.dumps(
                {"use_trtllm_attention": False},
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    elif compilation_profile != "default":
        compilation_config: dict[str, Any] = {}
        if compilation_profile == "piecewise":
            # FlashInfer XQA on SM120 is not replay-safe inside FULL decode
            # graphs as of the pinned runtime. Keep attention eager while the
            # surrounding target graph remains captured. This is both a
            # long-context correctness guard and an acceptance-rate canary.
            compilation_config["cudagraph_mode"] = "PIECEWISE"
        else:
            compilation_config["pass_config"] = {
                # Qwen3.8 uses partial rotary embeddings. Keep the separate
                # QK-norm/RoPE fusion disabled: upstream issue #51049 documents
                # corrupt generation for partial-RoPE models.
                "enable_qk_norm_rope_fusion": False,
                "fuse_attn_quant": True,
            }
            if compilation_profile == "attnquant-partition":
                compilation_config["use_inductor_graph_partition"] = True
            elif compilation_profile == "attnquant-fullgraph":
                compilation_config["splitting_ops"] = []
            else:
                raise SpeedLabError("unsupported compilation profile")
        command += [
            "--compilation-config",
            json.dumps(compilation_config, sort_keys=True, separators=(",", ":")),
        ]
    if speculative is not None:
        command += [
            "--speculative-config",
            json.dumps(speculative, sort_keys=True, separators=(",", ":")),
        ]
    return command


def _container_command(request: dict[str, Any]) -> list[str]:
    paths = _paths(request)
    runtime = request["runtime"]
    scripts_root = Path(request["source_root"]) / "aeon/scripts"
    sitecustomize = scripts_root / "speed_lab_sitecustomize/sitecustomize.py"
    uuid_guard = scripts_root / "vllm_uuid_sitecustomize.py"
    if runtime["feature_capture"]:
        feature_dir = _private_directory(paths["feature_dir"], create=True)
        if any(feature_dir.iterdir()):
            raise SpeedLabError("feature capture directory is not empty")
    command = [
        str(BASH),
        str(LOW_PRIORITY),
        str(BASH),
        str(DOCKER),
        "run",
        "-d",
        "--cidfile",
        str(paths["cidfile"]),
        "--name",
        request["container_name"],
        "--hostname",
        request["container_name"],
        "--interactive=false",
        "--tty=false",
        "--network",
        "bridge",
        "--cgroupns",
        "private",
        "--runtime",
        "runc",
        "--gpus",
        f"device={request['gpu_uuid']}",
        "--shm-size",
        str(8 * 1024**3),
        "--ipc",
        "private",
        "--publish",
        f"127.0.0.1:{request['port']}:{request['port']}",
        "--oom-score-adj",
        "1000",
        "--cpu-shares",
        "2",
        "--blkio-weight",
        "10",
        "--pids-limit",
        "1024",
        "--user",
        f"{os.geteuid()}:{os.getegid()}",
        "--read-only",
        "--privileged=false",
        "--publish-all=false",
        "--init=false",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--tmpfs",
        f"/workspace/cache:rw,exec,nosuid,nodev,size={8 * 1024**3},uid={os.geteuid()},gid={os.getegid()},mode=0700",
        "--log-driver",
        "local",
        "--log-opt",
        "max-size=10m",
        "--log-opt",
        "max-file=3",
        "--restart",
        "no",
        "--label",
        "owner=aday",
        "--label",
        "com.bc_aeon.component=qwen38-speed-lab",
        "--label",
        f"com.bc_aeon.runtime-id={request['runtime_id']}",
        "--label",
        f"com.bc_aeon.claim={request['claim_id']}",
        "--label",
        f"com.bc_aeon.variant={request['variant']}",
        "--mount",
        f"type=bind,src={LOW_PRIORITY},dst=/usr/local/bin/fleet-low-priority,readonly",
        "--mount",
        f"type=bind,src={sitecustomize},dst=/workspace/aeon_runtime/sitecustomize.py,readonly",
        "--mount",
        f"type=bind,src={uuid_guard},dst=/workspace/aeon_runtime/vllm_uuid_sitecustomize.py,readonly",
        "--mount",
        f"type=bind,src={request['model_dir']},dst=/models,readonly",
        "--mount",
        f"type=bind,src={request['draft_model_dir']},dst=/draft,readonly",
    ]
    if runtime["feature_capture"]:
        command += [
            "--mount",
            f"type=bind,src={paths['feature_dir']},dst=/features",
        ]
    environment = {
        "GPU_AGENT_CLAIM_ID": request["claim_id"],
        "GPU_LEASE_OWNER": request["owner"],
        "GPU_LEASE_RUN_DIR": request["scratch_path"],
        "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
        "GPU_MEM_LIMIT_GB": f"{request['vram_budget_gb']:g}",
        "GPU_PLANNED_VRAM_GB": f"{request['vram_budget_gb']:g}",
        "GPU_RESERVE_GB": "6",
        "GPU_LEASE_EXCLUSIVE": "1",
        "AEON_SPEED_LAB_RUNTIME_ID": request["runtime_id"],
        "AEON_SPEED_LAB_MODEL_ID": request["model_id"],
        "AEON_SPEED_LAB_MODEL_SHA256S": request["model_sha256s_sha256"],
        "AEON_SPEED_LAB_DRAFT_SHA256": request["draft_model_sha256"],
        "AEON_SPEED_LAB_PORT": str(request["port"]),
        "SPT_NOENV": "1",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/workspace/aeon_runtime",
        "HOME": "/workspace/cache/home",
        "HF_HOME": "/workspace/cache/huggingface",
        "TRANSFORMERS_CACHE": "/workspace/cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/workspace/cache/home/.cache",
        "TRITON_CACHE_DIR": "/workspace/cache/triton",
        "VLLM_CACHE_ROOT": "/workspace/cache/vllm",
        "FLASHINFER_WORKSPACE_DIR": "/workspace/cache/home/.cache/flashinfer",
        "TORCHINDUCTOR_CACHE_DIR": "/workspace/cache/torchinductor",
        "TMPDIR": "/workspace/cache",
        "MAX_JOBS": "4",
        "NVCC_THREADS": "1",
        # FlashInfer's first-priority per-tensor FP8 head path lazily compiles
        # against a CUDA header set that this exact runtime image does not ship
        # consistently. Disable only that one selector candidate; vLLM then
        # chooses its prebuilt CUTLASS FP8 kernel while retaining FlashInfer's
        # NVFP4, attention, and sampling kernels.
        "VLLM_DISABLED_KERNELS": DISABLED_KERNELS,
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_LAUNCH_BLOCKING": "1" if runtime["cuda_launch_blocking"] else "0",
        "VLLM_GDN_DECODE_KERNEL": runtime["gdn_decode_kernel"],
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "MTP_DRAFT_VOCAB": "1",
        "AEON_DSPARK_BF16_HEADS": (
            "1" if runtime["speculative_method"] == "dspark" else "0"
        ),
        "AEON_NVFP4_A16": "1" if runtime["nvfp4_a16"] else "0",
        "AEON_RELAXED_GREEDY_LOGIT_MARGIN": runtime[
            "relaxed_greedy_logit_margin"
        ],
        "VLLM_USE_FLASHINFER_SAMPLER": (
            "1" if runtime["use_flashinfer_sampler"] else "0"
        ),
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
    }
    if runtime["feature_capture"]:
        environment["AEON_DFLASH_FEATURE_CAPTURE_DIR"] = "/features"
        environment["AEON_DFLASH_FEATURE_DATASET_SHA256"] = request[
            "feature_dataset_sha256"
        ]
    for key in sorted(environment):
        command += ["--env", f"{key}={environment[key]}"]
    command += [
        "--entrypoint",
        "/usr/local/bin/fleet-low-priority",
        request["image_id"],
        "python3",
        *_server_arguments(
            request,
            model_path="/models",
            draft_model_path="/draft",
            bind_host="0.0.0.0",
        ),
    ]
    return command


def _bare_environment(request: dict[str, Any]) -> dict[str, str]:
    cache = _private_directory(
        Path(request["scratch_path"]) / "bare-runtime-cache", create=True
    )
    for relative in ("home", "huggingface", "triton", "vllm", "torchinductor"):
        _private_directory(cache / relative, create=True)
    rpc_root = _private_directory(_bare_rpc_root(request), create=True)
    source_root = Path(request["source_root"])
    scripts = source_root / "aeon/scripts"
    sitecustomize = scripts / "speed_lab_sitecustomize"
    return {
        "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(cache / "home"),
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{sitecustomize}:{scripts}:{ENGINE_SITE}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "USE_TF": "0",
        "USE_FLAX": "0",
        "GPU_AGENT_CLAIM_ID": request["claim_id"],
        "GPU_LEASE_OWNER": request["owner"],
        "GPU_LEASE_RUN_DIR": request["scratch_path"],
        "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
        "GPU_MEM_LIMIT_GB": f"{request['vram_budget_gb']:g}",
        "GPU_PLANNED_VRAM_GB": f"{request['vram_budget_gb']:g}",
        "GPU_RESERVE_GB": "6",
        "GPU_LEASE_EXCLUSIVE": "1",
        "AEON_SPEED_LAB_RUNTIME_ID": request["runtime_id"],
        "AEON_SPEED_LAB_MODEL_ID": request["model_id"],
        "AEON_SPEED_LAB_MODEL_SHA256S": request["model_sha256s_sha256"],
        "AEON_SPEED_LAB_DRAFT_SHA256": request["draft_model_sha256"],
        "AEON_SPEED_LAB_PORT": str(request["port"]),
        "SPT_NOENV": "1",
        "HF_HOME": str(cache / "huggingface"),
        "TRANSFORMERS_CACHE": str(cache / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(cache / "home/.cache"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "VLLM_RPC_BASE_PATH": str(rpc_root),
        "FLASHINFER_WORKSPACE_DIR": str(cache / "home/.cache/flashinfer"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TMPDIR": str(cache),
        "MAX_JOBS": "4",
        "NVCC_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_LAUNCH_BLOCKING": (
            "1" if request["runtime"]["cuda_launch_blocking"] else "0"
        ),
        "VLLM_GDN_DECODE_KERNEL": request["runtime"]["gdn_decode_kernel"],
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "MTP_DRAFT_VOCAB": "1",
        "AEON_NVFP4_A16": ("1" if request["runtime"]["nvfp4_a16"] else "0"),
        "AEON_RELAXED_GREEDY_LOGIT_MARGIN": request["runtime"][
            "relaxed_greedy_logit_margin"
        ],
        "VLLM_USE_FLASHINFER_SAMPLER": (
            "1" if request["runtime"]["use_flashinfer_sampler"] else "0"
        ),
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
    }


def _bare_command(request: dict[str, Any]) -> list[str]:
    return [
        str(BASH),
        str(LOW_PRIORITY),
        str(REMOTE_PYTHON),
        *_server_arguments(
            request,
            model_path=request["model_dir"],
            draft_model_path=request["draft_model_dir"],
            bind_host="127.0.0.1",
        ),
    ]


def _bare_process_snapshot(pid: int) -> tuple[list[bytes], list[bytes]] | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return None
    proc = Path(f"/proc/{pid}")
    try:
        environment = (proc / "environ").read_bytes().split(b"\0")
        command = (proc / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ESRCH}:
            return None
        raise SpeedLabError("bare server identity is unreadable") from exc
    return environment, command


def _boot_id() -> str:
    try:
        value = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        )
    except OSError as exc:
        raise SpeedLabError("worker boot identity is unreadable") from exc
    if re.fullmatch(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", value) is None:
        raise SpeedLabError("worker boot identity is malformed")
    return value


def _bare_process_stat(pid: int) -> dict[str, int | str] | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return None
    try:
        payload = (Path(f"/proc/{pid}") / "stat").read_text(encoding="ascii")
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ESRCH}:
            return None
        raise SpeedLabError("bare server kernel identity is unreadable") from exc
    close = payload.rfind(")")
    if not payload.startswith(f"{pid} (") or close < len(str(pid)) + 3:
        raise SpeedLabError("bare server kernel identity is malformed")
    fields = payload[close + 2 :].split()
    if len(fields) < 20 or fields[0] in {"X", "Z"}:
        return None
    try:
        return {
            "process_parent_pid": int(fields[1]),
            "process_group_id": int(fields[2]),
            "process_session_id": int(fields[3]),
            "process_start_ticks": int(fields[19]),
            "process_boot_id": _boot_id(),
        }
    except ValueError as exc:
        raise SpeedLabError("bare server kernel identity is malformed") from exc


def _bare_environment_matches(
    request: dict[str, Any], environment: list[bytes]
) -> bool:
    return (
        f"GPU_AGENT_CLAIM_ID={request['claim_id']}".encode() in environment
        and f"CUDA_VISIBLE_DEVICES={request['gpu_uuid']}".encode() in environment
        and f"AEON_SPEED_LAB_RUNTIME_ID={request['runtime_id']}".encode() in environment
        and f"AEON_SPEED_LAB_MODEL_ID={request['model_id']}".encode() in environment
        and (f"AEON_SPEED_LAB_MODEL_SHA256S={request['model_sha256s_sha256']}").encode()
        in environment
        and (f"AEON_SPEED_LAB_DRAFT_SHA256={request['draft_model_sha256']}").encode()
        in environment
        and f"AEON_SPEED_LAB_PORT={request['port']}".encode() in environment
        and (f"AEON_NVFP4_A16={1 if request['runtime']['nvfp4_a16'] else 0}").encode()
        in environment
        and (
            "AEON_RELAXED_GREEDY_LOGIT_MARGIN="
            f"{request['runtime']['relaxed_greedy_logit_margin']}"
        ).encode()
        in environment
        and any(
            value.startswith(b"PYTHONPATH=") and str(ENGINE_SITE).encode() in value
            for value in environment
        )
    )


def _bare_process_alive(request: dict[str, Any], pid: int) -> bool:
    """Prove the same kernel process after vLLM may rewrite argv/environ."""

    state_path = _paths(request)["state"]
    if state_path.is_file():
        state = _read_json(state_path)
        identity_keys = {
            "process_parent_pid",
            "process_group_id",
            "process_session_id",
            "process_start_ticks",
            "process_boot_id",
        }
        if identity_keys <= set(state):
            current = _bare_process_stat(pid)
            return (
                current is not None
                and state.get("runtime_id") == request["runtime_id"]
                and state.get("container_pid") == pid
                and all(current[key] == state.get(key) for key in identity_keys)
            )
    # Compatibility for a process launched by an older reviewed worker before
    # kernel identity receipts existed.
    snapshot = _bare_process_snapshot(pid)
    return snapshot is not None and _bare_environment_matches(request, snapshot[0])


def _bare_spawn_identity(request: dict[str, Any], pid: int) -> bool:
    """Require the original reviewed command once, before title rewriting."""

    snapshot = _bare_process_snapshot(pid)
    if snapshot is None:
        return False
    environment, command = snapshot
    return (
        _bare_environment_matches(request, environment)
        and b"vllm.entrypoints.openai.api_server" in command
        and request["model_dir"].encode() in command
        and str(request["port"]).encode() in command
    )


def _bare_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _bare_exit_description(pid: int) -> str:
    """Reap an exited direct child and retain a useful launch diagnostic."""
    try:
        waited, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return "exit status unavailable"
    if waited == 0:
        return "process identity disappeared while its child is still running"
    return f"exit status {os.waitstatus_to_exitcode(status)}"


def _launch_bare(request: dict[str, Any], request_sha256: str) -> tuple[str, int]:
    paths = _paths(request)
    if paths["server_log"].exists() or paths["server_log"].is_symlink():
        raise SpeedLabError("bare server log receipt already exists")
    _verify_sources(request)
    _inspect_image(request)
    _verify_acl(request)
    descriptor = os.open(
        paths["server_log"],
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        process = subprocess.Popen(
            _bare_command(request),
            cwd=request["scratch_path"],
            env=_bare_environment(request),
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(descriptor)
    handle = f"bare-{process.pid}"
    state = {
        "schema_version": SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "runtime_id": request["runtime_id"],
        "worker_pid": os.getpid(),
        "phase": "bare_server_spawning",
        "container_id": handle,
        "container_pid": process.pid,
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    _atomic_json(paths["state"], state)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SpeedLabError(
                f"bare speed server exited during spawn with status {process.returncode}"
            )
        try:
            alive = _bare_spawn_identity(request, process.pid)
        except SpeedLabError as exc:
            if process.poll() is not None:
                raise SpeedLabError(
                    "bare speed server exited during spawn "
                    f"with status {process.returncode}"
                ) from exc
            raise
        if alive:
            process_identity = _bare_process_stat(process.pid)
            if (
                process_identity is None
                or process_identity["process_parent_pid"] != os.getpid()
                or process_identity["process_group_id"] != process.pid
                or process_identity["process_session_id"] != process.pid
            ):
                raise SpeedLabError("bare server kernel identity changed during spawn")
            state.update(
                phase="bare_server_starting",
                updated_at=time.time(),
                **process_identity,
            )
            _atomic_json(paths["state"], state)
            return handle, process.pid
        time.sleep(0.1)
    raise SpeedLabError("bare speed server identity did not become visible")


def _launch_container(request: dict[str, Any], request_sha256: str) -> tuple[str, int]:
    if request["host"] == BARE_HOST:
        return _launch_bare(request, request_sha256)
    paths = _paths(request)
    if paths["cidfile"].exists() or paths["cidfile"].is_symlink():
        raise SpeedLabError("container ID receipt already exists")
    _verify_sources(request)
    _inspect_image(request)
    _verify_acl(request)
    result = subprocess.run(
        _container_command(request),
        env=_docker_environment(request),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
        umask=0o077,
    )
    if result.returncode != 0:
        raise SpeedLabError("Docker refused the exact speed-lab container")
    container_id = paths["cidfile"].read_text(encoding="ascii").strip()
    if (
        _CONTAINER_ID_RE.fullmatch(container_id) is None
        or result.stdout.strip() != container_id
    ):
        raise SpeedLabError("Docker container ID receipt is inconsistent")
    item = _inspect_container(request, container_id)
    if item is None or (item.get("State") or {}).get("Running") is not True:
        raise SpeedLabError("exact speed-lab container is not running")
    pid = (item.get("State") or {}).get("Pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        raise SpeedLabError("exact speed-lab container PID is invalid")
    state = {
        "schema_version": SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "runtime_id": request["runtime_id"],
        "worker_pid": os.getpid(),
        "phase": "container_starting",
        "container_id": container_id,
        "container_pid": pid,
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    _atomic_json(paths["state"], state)
    return container_id, pid


def _capture_logs(request: dict[str, Any], container_id: str | None) -> str:
    paths = _paths(request)
    if request["host"] == BARE_HOST:
        if not paths["server_log"].is_file():
            if container_id is None:
                _atomic_text(paths["server_log"], "")
                return ""
            raise SpeedLabError("bare server log is absent")
        payload = paths["server_log"].read_bytes()
        if len(payload) > MAX_SERVER_LOG_BYTES:
            payload = payload[-MAX_SERVER_LOG_BYTES:]
        return payload.decode("utf-8", errors="replace")
    if container_id is None:
        _atomic_text(paths["server_log"], "")
        return ""
    result = _docker(request, "logs", "--timestamps", container_id, timeout=120)
    payload = (result.stdout + result.stderr).encode("utf-8", errors="replace")
    if len(payload) > MAX_SERVER_LOG_BYTES:
        payload = payload[-MAX_SERVER_LOG_BYTES:]
    text = payload.decode("utf-8", errors="replace")
    _atomic_text(paths["server_log"], text)
    return text


def _cleanup_bare(request: dict[str, Any], handle: str | None) -> bool:
    if handle is None:
        _cleanup_bare_rpc(request)
        return True
    match = _BARE_HANDLE_RE.fullmatch(handle)
    if match is None:
        raise SpeedLabError("bare server handle is malformed")
    pid = int(match.group(1))
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    if not _bare_group_exists(pid):
        _cleanup_bare_rpc(request)
        return True
    try:
        alive = _bare_process_alive(request, pid)
    except SpeedLabError:
        if not _bare_group_exists(pid):
            _cleanup_bare_rpc(request)
            return True
        raise
    if alive:
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and _bare_group_exists(pid):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            time.sleep(0.5)
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
    elif _bare_group_exists(pid):
        return False
    absent = not _bare_group_exists(pid)
    if absent:
        _cleanup_bare_rpc(request)
    return absent


def _cleanup_container(request: dict[str, Any], container_id: str | None) -> bool:
    if request["host"] == BARE_HOST:
        return _cleanup_bare(request, container_id)
    if container_id is None:
        return True
    deadline = time.monotonic() + 45
    while True:
        # _inspect_container revalidates all task-owned labels on every pass.
        # Docker stop/rm can report before containerd has published the matching
        # state transition, so reconcile exact identity instead of quarantining a
        # completed run after one transient response.
        item = _inspect_container(request, container_id)
        if item is None:
            return True
        if (item.get("State") or {}).get("Running") is True:
            _docker(request, "stop", "--time", "30", container_id, timeout=60)
        else:
            _docker(request, "rm", container_id, timeout=60)
        if time.monotonic() >= deadline:
            return _inspect_container(request, container_id) is None
        time.sleep(0.5)


def _wait_ready(request: dict[str, Any], container_id: str, stop: list[bool]) -> str:
    base = f"http://127.0.0.1:{request['port']}"
    deadline = time.monotonic() + 2100
    while time.monotonic() < deadline:
        if stop[0]:
            raise SpeedLabError("speed-lab stop requested during startup")
        if request["host"] == BARE_HOST:
            match = _BARE_HANDLE_RE.fullmatch(container_id)
            alive = False
            if match is not None:
                pid = int(match.group(1))
                try:
                    alive = _bare_process_alive(request, pid)
                except SpeedLabError as exc:
                    detail = _bare_exit_description(pid)
                    if (
                        detail
                        != "process identity disappeared while its child is still running"
                    ):
                        raise SpeedLabError(
                            f"bare speed server exited during startup: {detail}"
                        ) from exc
                    raise
            if match is None or not alive:
                detail = (
                    "malformed handle"
                    if match is None
                    else _bare_exit_description(int(match.group(1)))
                )
                raise SpeedLabError(
                    f"bare speed server exited during startup: {detail}"
                )
        else:
            item = _inspect_container(request, container_id)
            if item is None or (item.get("State") or {}).get("Running") is not True:
                raise SpeedLabError("speed-lab container exited during startup")
        try:
            response = requests.get(
                base + "/v1/models",
                timeout=(2, 10),
                allow_redirects=False,
                proxies={"http": "", "https": ""},
            )
            if response.status_code == 200 and any(
                isinstance(model, dict) and model.get("id") == SERVED_MODEL
                for model in (response.json().get("data") or [])
            ):
                return base
        except (requests.RequestException, ValueError, TypeError):
            pass
        time.sleep(5)
    raise SpeedLabError("speed-lab endpoint did not become ready in time")


def _run_child(
    command: list[str], request: dict[str, Any], stop: list[bool], timeout: float
) -> int:
    transport_guard = (
        Path(request["source_root"])
        / "aeon/scripts/local_http_sitecustomize"
    )
    guard_metadata = transport_guard.lstat()
    guard_file = transport_guard / "sitecustomize.py"
    if (
        not stat.S_ISDIR(guard_metadata.st_mode)
        or guard_metadata.st_uid != os.geteuid()
        or guard_metadata.st_mode & 0o022
        or not guard_file.is_file()
    ):
        raise SpeedLabError("benchmark local-transport guard is unavailable")
    process = subprocess.Popen(
        [str(BASH), str(LOW_PRIORITY), *command],
        cwd=request["source_root"],
        env={
            "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": "/home/aday",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONPATH": f"{transport_guard}:{request['source_root']}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "AEON_LOCAL_HTTP_PORT": str(request["port"]),
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        umask=0o077,
    )
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if stop[0]:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                raise SpeedLabError("benchmark child did not stop cleanly")
            raise SpeedLabError("speed-lab stop requested during benchmark")
        if time.monotonic() >= deadline:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                raise SpeedLabError("benchmark child exceeded timeout and did not stop")
            raise SpeedLabError("benchmark child exceeded its bounded timeout")
        time.sleep(1)
    output = process.stdout.read() if process.stdout is not None else ""
    if len(output) > 64 * 1024:
        output = output[-64 * 1024 :]
    _atomic_text(_paths(request)["output"] / f"child-{int(time.time())}.log", output)
    return int(process.returncode)


def _write_manifest(request: dict[str, Any]) -> str:
    paths = _paths(request)
    output = _private_directory(paths["output"])
    files = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.name == paths["manifest"].name:
            continue
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise SpeedLabError("output tree contains an unsafe inode")
        files.append((_sha256(path), path.name))
    if not any(name == "result.json" for _digest, name in files):
        raise SpeedLabError("terminal output has no result receipt")
    payload = "".join(f"{digest}  {name}\n" for digest, name in files)
    _atomic_text(paths["manifest"], payload)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _package_features(request: dict[str, Any]) -> dict[str, Any]:
    """Create one deterministic, transferable archive from captured tensors."""

    paths = _paths(request)
    feature_dir = _private_directory(paths["feature_dir"])
    index = _read_json(paths["feature_index"], maximum=16 * 1024 * 1024)
    if (
        index.get("schema_version") != "aeon-qwen38-dflash-feature-index-v1"
        or index.get("dataset_sha256") != request["feature_dataset_sha256"]
        or index.get("dataset_rows") != request["feature_dataset_rows"]
        or index.get("model_sha256s") != request["model_sha256s_sha256"]
        or index.get("draft_sha256") != request["draft_model_sha256"]
        or index.get("layer_ids") != [6, 20, 34, 48, 62]
        or index.get("hidden_size") != 5120
        or index.get("feature_width") != 25600
        or index.get("dtype") != "bfloat16"
        or isinstance(index.get("total_tokens"), bool)
        or not isinstance(index.get("total_tokens"), int)
        or not 1 <= index["total_tokens"] <= 3_000_000
        or isinstance(index.get("unique_features"), bool)
        or not isinstance(index.get("unique_features"), int)
        or not 1 <= index["unique_features"] <= request["feature_dataset_rows"]
    ):
        raise SpeedLabError("DFlash feature index identity changed")
    expected_names = {"index.json"}
    features = index.get("features")
    if not isinstance(features, list) or len(features) != index["unique_features"]:
        raise SpeedLabError("DFlash feature file index is malformed")
    for item in features:
        if not isinstance(item, dict):
            raise SpeedLabError("DFlash feature item is malformed")
        token_hash = str(item.get("token_hash") or "")
        if _SHA256_RE.fullmatch(token_hash) is None:
            raise SpeedLabError("DFlash feature token hash is malformed")
        feature_name = f"{token_hash}.safetensors"
        receipt_name = f"{token_hash}.json"
        if (
            item.get("feature_file") != feature_name
            or item.get("receipt_file") != receipt_name
        ):
            raise SpeedLabError("DFlash feature filename changed")
        feature_path = feature_dir / feature_name
        receipt_path = feature_dir / receipt_name
        if (
            not feature_path.is_file()
            or not receipt_path.is_file()
            or feature_path.stat().st_size != item.get("feature_bytes")
            or _sha256(feature_path) != item.get("feature_sha256")
        ):
            raise SpeedLabError("DFlash captured tensor digest changed")
        expected_names.update((feature_name, receipt_name))
    if {path.name for path in feature_dir.iterdir()} != expected_names - {"index.json"}:
        raise SpeedLabError("DFlash feature directory contains unexpected files")

    archived_index = feature_dir / "index.json"
    shutil.copyfile(paths["feature_index"], archived_index)
    archived_index.chmod(0o600)
    archive = paths["feature_archive"]
    if archive.exists() or archive.is_symlink():
        raise SpeedLabError("DFlash feature archive destination already exists")
    tar_process = subprocess.Popen(
        [
            "/usr/bin/tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--format=gnu",
            "-C",
            str(feature_dir),
            "-cf",
            "-",
            ".",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=False,
    )
    assert tar_process.stdout is not None
    zstd_process = subprocess.Popen(
        ["/usr/bin/zstd", "-q", "-T2", "-3", "-o", str(archive)],
        stdin=tar_process.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=False,
    )
    tar_process.stdout.close()
    _zstd_stdout, zstd_stderr = zstd_process.communicate(timeout=3600)
    _tar_stdout, tar_stderr = tar_process.communicate(timeout=120)
    if (
        tar_process.returncode != 0
        or zstd_process.returncode != 0
        or len(tar_stderr) > 64 * 1024
        or len(zstd_stderr) > 64 * 1024
    ):
        raise SpeedLabError("DFlash feature archive creation failed")
    archive.chmod(0o600)
    archive_sha256 = _sha256(archive)
    archive_bytes = archive.stat().st_size
    shutil.rmtree(feature_dir)
    if feature_dir.exists() or feature_dir.is_symlink():
        raise SpeedLabError("DFlash feature staging directory did not cleanly retire")
    return {
        "archive": archive.name,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "dataset_rows": index["dataset_rows"],
        "feature_bytes": index["feature_bytes"],
        "total_tokens": index["total_tokens"],
        "unique_features": index["unique_features"],
    }


def _run(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    preflight = _read_json(paths["preflight"])
    if preflight.get("request_sha256") != request_sha256:
        raise SpeedLabError("preflight receipt is absent or stale")
    stop = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        stop[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    container_id: str | None = None
    kernel_verified = False
    failure: str | None = None
    started = time.time()
    speed: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    feature_summary: dict[str, Any] | None = None
    try:
        _verify_acl(request)
        container_id, _container_pid = _launch_container(request, request_sha256)
        base = _wait_ready(request, container_id, stop)
        logs = _capture_logs(request, container_id)
        target_kernel_verified = (
            "Using V2 Model Runner" in logs and "GDN decode kernel: cuda" in logs
        )
        compilation_profile = request["runtime"]["compilation_profile"]
        attn_quant_verified = compilation_profile in {
            "default",
            "piecewise",
            "flashinfer-native-full",
        } or (
            "attn_quant" in logs
            and "'fuse_attn_quant': True" in logs
            and "'enable_qk_norm_rope_fusion': False" in logs
            and (
                compilation_profile != "attnquant-partition"
                or "'use_inductor_graph_partition': True" in logs
            )
            and (
                compilation_profile != "attnquant-fullgraph"
                or "'splitting_ops': []" in logs
            )
        )
        # DFlash2 has its own safe FULL graph even when target verification is
        # PIECEWISE, so verify the target's resolved config rather than banning
        # every generic FULL-capture log line.
        graph_mode_verified = compilation_profile != "piecewise" or (
            "'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>" in logs
        )
        attention_mode_verified = (
            compilation_profile != "flashinfer-native-full"
            or (
                "use_trtllm_attention=False" in logs
                and "decode_backend=flashinfer-native" in logs
            )
        )
        method = request["runtime"]["speculative_method"]
        speculator_verified = (
            method == "none"
            or (
                method == "mtp"
                and "Aeon speed patch: MTP drafter uses a 40960-token output head"
                in logs
            )
            or (
                method == "dflash"
                and "Capturing model for DFlash2 speculator..." in logs
                and (
                    request["draft_id"] != "w4a16"
                    or "Aeon speed patch: quantized DFlash context KV uses one fused BF16 GEMM"
                    in logs
                )
            )
            or (
                method == "dspark"
                and "Capturing model for DSpark speculator..." in logs
            )
        )
        fp8_head_verified = request["model_id"] != "fullgdn" or (
            "Selected CutlassFP8ScaledMMLinearKernel for ModelOptFp8LinearMethod"
            in logs
        )
        feature_hook_verified = not request["runtime"]["feature_capture"] or (
            "Aeon DFlash exact-target feature capture is enabled" in logs
            and (
                "'enable_prefix_caching': False" in logs
                or "enable_prefix_caching=False" in logs
            )
        )
        kernel_verified = (
            target_kernel_verified
            and speculator_verified
            and attn_quant_verified
            and graph_mode_verified
            and attention_mode_verified
            and fp8_head_verified
            and feature_hook_verified
        )
        if not kernel_verified:
            raise SpeedLabError(
                "V2 runner, fused CUDA GDN decode, requested compiler fusion, "
                "attention mode, FP8 head backend, and speculator were not all "
                "activated"
            )
        state = _read_json(paths["state"])
        state.update(phase="benchmarking", updated_at=time.time())
        _atomic_json(paths["state"], state)
        benchmark = request["benchmark"]
        if request["runtime"]["feature_capture"]:
            feature_command = [
                str(REMOTE_PYTHON),
                str(
                    Path(request["source_root"])
                    / "aeon/scripts/extract_qwen38_dflash_features.py"
                ),
                "--base-url",
                base,
                "--model",
                SERVED_MODEL,
                "--dataset",
                str(paths["feature_dataset"]),
                "--feature-dir",
                str(paths["feature_dir"]),
                "--model-sha256s",
                request["model_sha256s_sha256"],
                "--draft-sha256",
                request["draft_model_sha256"],
                "--output",
                str(paths["feature_index"]),
            ]
            feature_rc = _run_child(feature_command, request, stop, timeout=5400)
            if feature_rc != 0 or not paths["feature_index"].is_file():
                raise SpeedLabError("exact-target feature extraction failed")
            feature_summary = _package_features(request)
        else:
            speed_command = [
                str(REMOTE_PYTHON),
                str(
                    Path(request["source_root"])
                    / "aeon/scripts/benchmark_qwen38_speed.py"
                ),
                "--base-url",
                base,
                "--model",
                SERVED_MODEL,
                "--system-prefix",
                str(paths["prefix"]),
                "--repeats",
                str(benchmark["repeats"]),
                "--max-tokens",
                str(benchmark["max_tokens"]),
                "--sampling-profile",
                str(benchmark["sampling_profile"]),
                "--output",
                str(paths["speed"]),
            ]
            speed_rc = _run_child(speed_command, request, stop, timeout=5400)
            if paths["speed"].is_file():
                speed = _read_json(paths["speed"], maximum=16 * 1024 * 1024)
            if (
                speed_rc != 0
                or not speed
                or speed.get("benchmark_complete") is not True
            ):
                raise SpeedLabError("interactive speed benchmark failed")
            quality_command = [
                str(REMOTE_PYTHON),
                str(
                    Path(request["source_root"])
                    / "aeon/scripts/benchmark_qwen38_mtp.py"
                ),
                "probe",
                "--base-url",
                base,
                "--model",
                SERVED_MODEL,
                "--k",
                str(request["runtime"]["speculative_tokens"]),
                "--repeats",
                str(benchmark["quality_repeats"]),
                "--attention-backend",
                request["runtime"]["attention_backend"],
                "--kv-cache-dtype",
                request["runtime"]["kv_cache_dtype"],
                "--runtime-image-id",
                request["image_id"],
                "--sampling-profile",
                str(benchmark["sampling_profile"]),
                "--output",
                str(paths["quality"]),
            ]
            quality_rc = _run_child(quality_command, request, stop, timeout=5400)
            if paths["quality"].is_file():
                quality = _read_json(paths["quality"], maximum=32 * 1024 * 1024)
            if quality_rc != 0 or not quality or quality.get("passed") is not True:
                raise SpeedLabError("Aeon semantic quality gate failed")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"[:1000]
    finally:
        if container_id is None and paths["state"].is_file():
            try:
                launch_state = _read_json(paths["state"])
                recorded_handle = launch_state.get("container_id")
                if isinstance(recorded_handle, str) and recorded_handle:
                    container_id = recorded_handle
            except Exception:
                pass
        if container_id is not None:
            try:
                _capture_logs(request, container_id)
            except Exception as exc:
                failure = failure or f"server log capture failed: {type(exc).__name__}"
        try:
            container_absent = _cleanup_container(request, container_id)
        except Exception as exc:
            container_absent = False
            failure = (
                failure or f"container cleanup failed: {type(exc).__name__}: {exc}"
            )
        terminal_success = bool(
            failure is None
            and container_absent
            and kernel_verified
            and (
                feature_summary is not None
                if request["runtime"]["feature_capture"]
                else bool(
                    speed
                    and speed.get("benchmark_complete") is True
                    and quality
                    and quality.get("passed") is True
                )
            )
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": request["runtime_id"],
            "request_sha256": request_sha256,
            "variant": request["variant"],
            "terminal_success": terminal_success,
            "failure": failure,
            "container_absent": container_absent,
            "fused_gdn_cuda_verified": kernel_verified,
            "v2_model_runner_verified": kernel_verified,
            "started_at": started,
            "completed_at": time.time(),
            "elapsed_seconds": time.time() - started,
            "feature_summary": feature_summary,
            "speed_summary": None
            if speed is None
            else {
                "median_decode_tps": speed.get("median_decode_tps"),
                "p95_warm_prefix_ttft_seconds": speed.get(
                    "p95_warm_prefix_ttft_seconds"
                ),
                "decode_target_met": speed.get("decode_target_met"),
                "ttft_target_met": speed.get("ttft_target_met"),
            },
            "quality_summary": None
            if quality is None
            else {
                "passed": quality.get("passed"),
                "median_decode_tps": quality.get("median_decode_tps"),
                "successful_requests": quality.get("successful_requests"),
                "request_count": quality.get("request_count"),
            },
        }
        _atomic_json(paths["terminal"], result)
        try:
            _write_manifest(request)
        except Exception as exc:
            result["terminal_success"] = False
            result["failure"] = result["failure"] or (
                f"output manifest failed: {type(exc).__name__}: {exc}"
            )
            _atomic_json(paths["terminal"], result)
            _write_manifest(request)
        if paths["state"].is_file():
            state = _read_json(paths["state"])
            state.update(
                phase="completed" if result["terminal_success"] else "failed",
                updated_at=time.time(),
            )
            _atomic_json(paths["state"], state)
    return result


def _spawn(
    request: dict[str, Any], request_path: Path, request_sha256: str
) -> dict[str, Any]:
    paths = _paths(request)
    preflight = _read_json(paths["preflight"])
    if preflight.get("request_sha256") != request_sha256:
        raise SpeedLabError("preflight receipt is stale")
    if paths["terminal"].exists() or paths["state"].exists() or paths["spawn"].exists():
        raise SpeedLabError("speed-lab lifecycle receipt already exists")
    worker = Path(request["source_root"]) / "aeon/scripts/qwen_speed_lab_worker.py"
    environment = {
        "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONPATH": request["source_root"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "GPU_AGENT_CLAIM_ID": request["claim_id"],
        "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
        "GPU_MEM_LIMIT_GB": f"{request['vram_budget_gb']:g}",
        "AEON_SPEED_LAB_RUNTIME_ID": request["runtime_id"],
    }
    descriptor = os.open(
        paths["supervisor_log"],
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        process = subprocess.Popen(
            [
                str(BASH),
                str(LOW_PRIORITY),
                str(REMOTE_PYTHON),
                str(worker),
                "run",
                str(request_path),
                request_sha256,
            ],
            cwd="/home/aday",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(descriptor)
    _atomic_json(
        paths["spawn"],
        {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": request["runtime_id"],
            "request_sha256": request_sha256,
            "worker_pid": process.pid,
            "created_at": time.time(),
        },
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if _process_alive(request, process.pid):
            return {"state": "running", "pid": process.pid}
        if process.poll() is not None:
            raise SpeedLabError("speed-lab supervisor exited during spawn")
        time.sleep(0.1)
    raise SpeedLabError("speed-lab supervisor identity did not become visible")


def _service_endpoint_ready(request: dict[str, Any]) -> bool:
    base = f"http://127.0.0.1:{request['port']}"
    try:
        health = requests.get(
            base + "/health",
            timeout=(2, 10),
            allow_redirects=False,
            proxies={"http": "", "https": ""},
        )
        models = requests.get(
            base + "/v1/models",
            timeout=(2, 10),
            allow_redirects=False,
            proxies={"http": "", "https": ""},
        )
        if (
            health.status_code != 200
            or models.status_code != 200
            or len(models.content) > 256 * 1024
        ):
            return False
        value = models.json()
        return {
            item.get("id")
            for item in value.get("data", [])
            if isinstance(item, dict)
        } == {SERVED_MODEL}
    except (requests.RequestException, TypeError, ValueError):
        return False


def _service_warmup(request: dict[str, Any]) -> None:
    """Prime the exact Aeon prefix and reject a merely port-responsive server."""

    prefix = Path(request["scratch_path"]) / "system-prefix.txt"
    metadata = prefix.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or not 4096 <= metadata.st_size <= 2 * 1024 * 1024
    ):
        raise SpeedLabError("service warmup prefix identity changed")
    response = requests.post(
        f"http://127.0.0.1:{request['port']}/v1/chat/completions",
        json={
            "model": SERVED_MODEL,
            "messages": [
                {"role": "system", "content": prefix.read_text(encoding="utf-8")},
                {"role": "user", "content": "Reply with the single word READY."},
            ],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        timeout=(5, 300),
        allow_redirects=False,
        proxies={"http": "", "https": ""},
    )
    if response.status_code != 200 or len(response.content) > 4 * 1024 * 1024:
        raise SpeedLabError("service warmup request failed")
    try:
        value = response.json()
        choices = value["choices"]
        usage = value["usage"]
        completion_tokens = int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SpeedLabError("service warmup response is malformed") from exc
    if (
        value.get("model") != SERVED_MODEL
        or not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or not 1 <= completion_tokens <= 8
    ):
        raise SpeedLabError("service warmup response identity changed")


def _service_run(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    """Own one exact long-lived candidate container until Fleet requests stop."""

    paths = _paths(request)
    preflight = _read_json(paths["preflight"])
    if preflight.get("request_sha256") != request_sha256:
        raise SpeedLabError("service preflight receipt is stale")
    stop = [False]

    def request_stop(_signum, _frame):
        stop[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    container_id: str | None = None
    failure: str | None = None
    container_absent = False
    try:
        container_id, container_pid = _launch_container(request, request_sha256)
        _wait_ready(request, container_id, stop)
        _service_warmup(request)
        if not _service_endpoint_ready(request):
            raise SpeedLabError("service endpoint failed its post-warmup identity check")
        state = _read_json(paths["state"])
        state.update(
            phase="service_ready",
            container_pid=container_pid,
            ready_at=time.time(),
            updated_at=time.time(),
        )
        _atomic_json(paths["state"], state)
        failures = 0
        while not stop[0]:
            item = _inspect_container(request, container_id)
            current_pid = (item.get("State") or {}).get("Pid") if item else None
            if (
                item is None
                or (item.get("State") or {}).get("Running") is not True
                or current_pid != container_pid
            ):
                raise SpeedLabError("exact service container exited or changed identity")
            failures = 0 if _service_endpoint_ready(request) else failures + 1
            if failures >= 3:
                raise SpeedLabError("exact service endpoint failed three health checks")
            time.sleep(5)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            stop[0] = True
    finally:
        try:
            _capture_logs(request, container_id)
        except Exception as exc:
            if failure is None:
                failure = f"log capture failed: {type(exc).__name__}: {exc}"
        try:
            container_absent = _cleanup_container(request, container_id)
        except Exception as exc:
            if failure is None:
                failure = f"container cleanup failed: {type(exc).__name__}: {exc}"
        result = {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": request["runtime_id"],
            "request_sha256": request_sha256,
            "service": True,
            "terminal_success": stop[0] and failure is None and container_absent,
            "container_absent": container_absent,
            "failure": failure,
            "completed_at": time.time(),
        }
        _atomic_json(paths["terminal"], result)
        if paths["state"].is_file():
            state = _read_json(paths["state"])
            state.update(
                phase="service_stopped" if result["terminal_success"] else "service_failed",
                updated_at=time.time(),
            )
            _atomic_json(paths["state"], state)
    return {
        "state": "stopped" if container_absent else "ambiguous",
        "process_absent": container_absent,
    }


def _spawn_service(
    request: dict[str, Any], request_path: Path, request_sha256: str
) -> dict[str, Any]:
    paths = _paths(request)
    preflight = _read_json(paths["preflight"])
    if preflight.get("request_sha256") != request_sha256:
        raise SpeedLabError("service preflight receipt is stale")
    if paths["terminal"].exists() or paths["state"].exists() or paths["spawn"].exists():
        raise SpeedLabError("service lifecycle receipt already exists")
    worker = Path(request["source_root"]) / "aeon/scripts/qwen_speed_lab_worker.py"
    environment = {
        "PATH": "/home/aday/.local/bin:/home/aday/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/home/aday",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONPATH": request["source_root"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "GPU_AGENT_CLAIM_ID": request["claim_id"],
        "CUDA_VISIBLE_DEVICES": request["gpu_uuid"],
        "GPU_MEM_LIMIT_GB": f"{request['vram_budget_gb']:g}",
        "AEON_SPEED_LAB_RUNTIME_ID": request["runtime_id"],
    }
    descriptor = os.open(
        paths["supervisor_log"],
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        process = subprocess.Popen(
            [
                str(BASH),
                str(LOW_PRIORITY),
                str(REMOTE_PYTHON),
                str(worker),
                "service-run",
                str(request_path),
                request_sha256,
            ],
            cwd="/home/aday",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(descriptor)
    _atomic_json(
        paths["spawn"],
        {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": request["runtime_id"],
            "request_sha256": request_sha256,
            "worker_pid": process.pid,
            "service": True,
            "created_at": time.time(),
        },
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if _service_process_alive(request, process.pid):
            return {"state": "starting", "pid": process.pid}
        if process.poll() is not None:
            raise SpeedLabError("service supervisor exited during spawn")
        time.sleep(0.1)
    raise SpeedLabError("service supervisor identity did not become visible")


def _service_status(request: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(request)
    state = _read_json(paths["state"]) if paths["state"].is_file() else None
    spawn = _read_json(paths["spawn"]) if paths["spawn"].is_file() else None
    receipt = state or spawn
    pid = receipt.get("worker_pid") if receipt else None
    if isinstance(pid, int) and _service_process_alive(request, pid):
        phase = state.get("phase") if state else "service_supervisor_spawning"
        if phase == "service_ready":
            container_id = str(state.get("container_id") or "")
            item = _inspect_container(request, container_id)
            if (
                item is None
                or (item.get("State") or {}).get("Running") is not True
                or not _service_endpoint_ready(request)
            ):
                return {
                    "state": "unknown",
                    "pid": pid,
                    "phase": "service_identity_ambiguous",
                }
            return {"state": "ready", "pid": pid, "phase": phase}
        return {"state": "starting", "pid": pid, "phase": phase}
    if paths["terminal"].is_file():
        terminal = _read_json(paths["terminal"], maximum=4 * 1024 * 1024)
        if terminal.get("container_absent") is not True:
            return {"state": "unknown", "pid": pid, "phase": "container_ambiguous"}
        return {
            "state": "absent",
            "pid": pid,
            "phase": "service_terminal",
            "failure": terminal.get("failure"),
        }
    if state and state.get("container_id"):
        item = _inspect_container(request, str(state["container_id"]))
        if item is not None:
            return {"state": "unknown", "pid": pid, "phase": "orphaned_container"}
    return {"state": "absent", "pid": pid, "phase": "no_service_receipt"}


def _service_stop(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    state = _read_json(paths["state"]) if paths["state"].is_file() else None
    spawn = _read_json(paths["spawn"]) if paths["spawn"].is_file() else None
    receipt = state or spawn
    pid = receipt.get("worker_pid") if receipt else None
    if isinstance(pid, int) and _service_process_alive(request, pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and _service_process_alive(request, pid):
            time.sleep(0.5)
        if _service_process_alive(request, pid):
            return {"state": "ambiguous", "process_absent": False}
    container_id = str(state.get("container_id")) if state and state.get("container_id") else None
    container_absent = _cleanup_container(request, container_id)
    if container_absent and not paths["terminal"].is_file():
        _atomic_json(
            paths["terminal"],
            {
                "schema_version": SCHEMA_VERSION,
                "runtime_id": request["runtime_id"],
                "request_sha256": request_sha256,
                "service": True,
                "terminal_success": True,
                "container_absent": True,
                "failure": None,
                "completed_at": time.time(),
            },
        )
    return {
        "state": "stopped" if container_absent else "ambiguous",
        "process_absent": container_absent,
    }


def _service_cleanup(request: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(request)
    status = _service_status(request)
    if status.get("state") != "absent":
        raise SpeedLabError("service scratch is not cleanup-safe")
    root = paths["scratch"]
    metadata = root.lstat()
    if (
        root.parent != Path(SCRATCH_ROOT)
        or root.name != request["runtime_id"]
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise SpeedLabError("service cleanup root identity changed")
    reclaimed = metadata.st_blocks * 512
    for item in root.rglob("*"):
        child = item.lstat()
        if child.st_uid != os.geteuid() or stat.S_ISLNK(child.st_mode):
            raise SpeedLabError("service scratch contains an unsafe inode")
        if not (stat.S_ISDIR(child.st_mode) or stat.S_ISREG(child.st_mode)):
            raise SpeedLabError("service scratch contains a special inode")
        reclaimed += child.st_blocks * 512
    shutil.rmtree(root)
    return {"state": "cleaned", "reclaimed_bytes": reclaimed}


def _status(request: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(request)
    state = _read_json(paths["state"]) if paths["state"].is_file() else None
    spawn = _read_json(paths["spawn"]) if paths["spawn"].is_file() else None
    for receipt_value in (state, spawn):
        if receipt_value is not None and (
            receipt_value.get("runtime_id") != request["runtime_id"]
            or receipt_value.get("request_sha256")
            != _sha256(Path(request["scratch_path"]) / "speed-lab-request.json")
        ):
            raise SpeedLabError("speed-lab lifecycle receipt identity changed")
    receipt = state or spawn
    pid = receipt.get("worker_pid") if receipt else None
    if isinstance(pid, int) and _process_alive(request, pid):
        return {
            "state": "running",
            "pid": pid,
            "phase": state.get("phase") if state else "supervisor_spawning",
            "container_pid": state.get("container_pid") if state else None,
        }
    if paths["terminal"].is_file():
        result = _read_json(paths["terminal"], maximum=4 * 1024 * 1024)
        if result.get("container_absent") is not True:
            return {"state": "unknown", "pid": pid, "phase": "container_ambiguous"}
        return {
            "state": "completed"
            if result.get("terminal_success") is True
            else "failed",
            "pid": pid,
            "phase": "terminal",
            "result": result,
        }
    if state and state.get("container_id"):
        handle = str(state["container_id"])
        if request["host"] == BARE_HOST:
            match = _BARE_HANDLE_RE.fullmatch(handle)
            if match is None or _bare_group_exists(int(match.group(1))):
                return {"state": "unknown", "pid": pid, "phase": "orphaned_bare_server"}
        else:
            item = _inspect_container(request, handle)
            if item is not None:
                return {"state": "unknown", "pid": pid, "phase": "orphaned_container"}
    return {"state": "absent", "pid": pid, "phase": "no_terminal_receipt"}


def _stop(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    state = _read_json(paths["state"]) if paths["state"].is_file() else None
    spawn = _read_json(paths["spawn"]) if paths["spawn"].is_file() else None
    receipt = state or spawn
    pid = receipt.get("worker_pid") if receipt else None
    if isinstance(pid, int) and _process_alive(request, pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and _process_alive(request, pid):
            time.sleep(0.5)
        if _process_alive(request, pid):
            return {"state": "ambiguous", "process_absent": False}
    container_id = (
        str(state.get("container_id")) if state and state.get("container_id") else None
    )
    container_absent = _cleanup_container(request, container_id)
    if not paths["terminal"].is_file() and container_absent:
        result = {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": request["runtime_id"],
            "request_sha256": request_sha256,
            "variant": request["variant"],
            "terminal_success": False,
            "failure": "speed-lab stopped before terminal benchmark receipt",
            "container_absent": True,
            "fused_gdn_cuda_verified": False,
            "started_at": None,
            "completed_at": time.time(),
            "elapsed_seconds": 0,
            "feature_summary": None,
            "speed_summary": None,
            "quality_summary": None,
        }
        _atomic_json(paths["terminal"], result)
        _write_manifest(request)
    return {
        "state": "stopped" if container_absent else "ambiguous",
        "process_absent": container_absent,
    }


def _settle_status(request: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(request)
    result = _read_json(paths["terminal"], maximum=4 * 1024 * 1024)
    manifest = paths["manifest"]
    if not manifest.is_file():
        raise SpeedLabError("output manifest is absent")
    manifest_sha256 = _sha256(manifest)
    files = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  ([A-Za-z0-9_.-]{1,200})", line)
        if match is None:
            raise SpeedLabError("output manifest is malformed")
        path = paths["output"] / match.group(2)
        if _sha256(path) != match.group(1):
            raise SpeedLabError("output file digest changed")
        files.append(
            {
                "name": match.group(2),
                "sha256": match.group(1),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "state": "settle_ready",
        "manifest_sha256": manifest_sha256,
        "files": files,
        "result": result,
    }


def _mark_settled(request: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    status = _settle_status(request)
    if manifest_sha256 != status["manifest_sha256"]:
        raise SpeedLabError("settled output manifest identity changed")
    marker = {
        "schema_version": SCHEMA_VERSION,
        "runtime_id": request["runtime_id"],
        "manifest_sha256": manifest_sha256,
        "settled_at": time.time(),
    }
    _atomic_json(_paths(request)["settled"], marker)
    return {"state": "settled", "manifest_sha256": manifest_sha256}


def _cleanup(request: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    paths = _paths(request)
    marker = _read_json(paths["settled"])
    terminal = _read_json(paths["terminal"])
    state = _read_json(paths["state"]) if paths["state"].is_file() else None
    pid = state.get("worker_pid") if state else None
    if (
        marker.get("runtime_id") != request["runtime_id"]
        or marker.get("manifest_sha256") != manifest_sha256
        or terminal.get("container_absent") is not True
        or isinstance(pid, int)
        and _process_alive(request, pid)
    ):
        raise SpeedLabError("worker scratch is not safe to clean")
    scratch = _private_directory(paths["scratch"])
    total = 0
    for path in scratch.rglob("*"):
        metadata = path.lstat()
        if metadata.st_uid != os.geteuid() or stat.S_ISLNK(metadata.st_mode):
            raise SpeedLabError("worker scratch contains an unsafe inode")
        if not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise SpeedLabError("worker scratch contains a special inode")
        total += metadata.st_size
    shutil.rmtree(scratch)
    return {"state": "cleaned", "reclaimed_bytes": total}


def main() -> int:
    if len(sys.argv) not in {4, 5}:
        print(json.dumps({"ok": False, "error": "invalid_arguments"}))
        return 64
    action = sys.argv[1]
    request_path = Path(sys.argv[2])
    request_sha256 = sys.argv[3]
    extra = sys.argv[4] if len(sys.argv) == 5 else None
    if action not in {
        "preflight",
        "spawn",
        "run",
        "status",
        "stop",
        "service-spawn",
        "service-run",
        "service-status",
        "service-stop",
        "service-cleanup",
        "settle-status",
        "mark-settled",
        "cleanup",
    }:
        print(json.dumps({"ok": False, "error": "invalid_action"}))
        return 64
    try:
        request = _validate_request(request_path, request_sha256)
        if action == "preflight":
            result = _preflight(request, request_sha256)
        elif action == "spawn":
            result = _spawn(request, request_path, request_sha256)
        elif action == "run":
            result = _run(request, request_sha256)
        elif action == "status":
            result = _status(request)
        elif action == "stop":
            result = _stop(request, request_sha256)
        elif action == "service-spawn":
            result = _spawn_service(request, request_path, request_sha256)
        elif action == "service-run":
            result = _service_run(request, request_sha256)
        elif action == "service-status":
            result = _service_status(request)
        elif action == "service-stop":
            result = _service_stop(request, request_sha256)
        elif action == "service-cleanup":
            result = _service_cleanup(request)
        elif action == "settle-status":
            result = _settle_status(request)
        elif action == "mark-settled":
            result = _mark_settled(request, str(extra or ""))
        else:
            result = _cleanup(request, str(extra or ""))
    except (SpeedLabError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": type(exc).__name__, "detail": str(exc)},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
