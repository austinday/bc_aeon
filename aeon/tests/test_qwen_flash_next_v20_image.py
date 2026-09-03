import hashlib
from pathlib import Path


SERVICES = Path(__file__).resolve().parents[1] / "services" / "vllm"
DOCKERFILE = SERVICES / "Dockerfile.qwen38-flash-next-v20-bf16-ple-attestation"


def test_v20_image_uses_exact_upstream_amd64_chain_and_only_attestation_patch():
    recipe = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "FROM vllm/vllm-openai@sha256:"
        "fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
        in recipe
    )
    assert (
        'com.bc_aeon.base.amd64.digest="sha256:'
        '0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3"'
        in recipe
    )
    copy_lines = [line.strip() for line in recipe.splitlines() if line.startswith("COPY ")]
    assert copy_lines == [
        "COPY patches/qwen38-flash-next-runtime-attestation.patch /tmp/runtime-attestation.patch",
        "COPY qwen38_flash_next_attestation.py /usr/local/lib/python3.12/dist-packages/vllm/qwen38_flash_next_attestation.py",
    ]
    assert "qwen38-flash-next-radixark-ple-fp8.patch" not in recipe
    assert "apt-get" not in recipe
    assert "pip install" not in recipe


def test_v20_image_guards_patch_module_workers_and_unmodified_ple_layer():
    recipe = DOCKERFILE.read_text(encoding="utf-8")
    patch = SERVICES / "patches" / "qwen38-flash-next-runtime-attestation.patch"
    module = SERVICES / "qwen38_flash_next_attestation.py"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == (
        "81fcf77c7a83ec177ee98010d1ace082e978567f8490fadeab48d2d71044a81e"
    )
    assert hashlib.sha256(module.read_bytes()).hexdigest() == (
        "5d4ba3b47dadf99e93513b7bf4663ef7b2657db082f19fa4ac038696010baf9a"
    )
    for digest in (
        "923767cea120a027ca36683bd48b7659eadbfa4cbd4acbcf52a80cee0c0a0ec4",
        "38820bebca30c15be82eac14f641218d5d14b8c129c1df96245da18f841817b2",
        "f93a0a8c40ee3c536e60cb027f5d6e76cde0a51e62b540c82e2c15b2a72b7f5f",
        "721a15c3440e45fd7dc41c8b5a1c441c142a1520113635ec77d78bfc833746e7",
    ):
        assert digest in recipe
    ple_digest = "a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7"
    assert recipe.count(ple_digest) >= 4
    assert 'm.version("vllm") == "0.1.dev20073+g8e685d198"' in recipe


def test_v20_docs_require_offline_low_priority_build_and_no_fp8_ple_env():
    docs = (
        SERVICES / "QWEN38_FLASH_NEXT_V20_BF16_PLE_IMAGE.md"
    ).read_text(encoding="utf-8")
    assert "/home/aday/bin/fleet-low-priority" in docs
    assert "--pull=false --network=none --platform=linux/amd64" in docs
    assert "VLLM_PLE_FP8_CHECKPOINT` must remain unset" in docs
    assert "profile remains disabled" in docs
    assert (
        "Archive SHA-256: `320722f344465b162d6277e65a9a1b27e"
        "b70c9b7960259604e32da10899f4a75`"
    ) in docs
    assert (
        "linux/amd64 image manifest: `sha256:"
        "f1f8a4dbeb015d112a230406c22e00cf2003b1bb0377d789e5730afaf9a9cc51`"
    ) in docs
    assert "not authorization to fill a\nprofile, reload Fleet, or launch" in docs
