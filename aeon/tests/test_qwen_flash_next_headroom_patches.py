from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "services/sglang/patches"
CUTLASS_SCALE_PATCH = PATCH_ROOT / "qwen38-flash-next-cutlass-scale-headroom.patch"
MTP_SHARE_PATCH = PATCH_ROOT / "qwen38-flash-next-mtp-share-before-pool.patch"

CUTLASS_SCALE_PATCH_SHA256 = (
    "a6c61ef9eaa1153551506b26aca7627f7ecc98851f6cd7e7038cd6d0a25b5c6a"
)
MTP_SHARE_PATCH_SHA256 = (
    "424eb761834646089437f7e2d16694ab06f03e102f045da07f4a35aa3c83b607"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _added_lines(patch: str) -> list[str]:
    return [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def _removed_lines(patch: str) -> list[str]:
    return [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]


def test_cutlass_scale_patch_is_narrow_and_pinned() -> None:
    patch = CUTLASS_SCALE_PATCH.read_text(encoding="utf-8")
    assert _sha256(CUTLASS_SCALE_PATCH) == CUTLASS_SCALE_PATCH_SHA256
    assert patch.count("diff --git ") == 1
    assert (
        "diff --git a/python/sglang/srt/layers/quantization/modelopt_quant.py "
        "b/python/sglang/srt/layers/quantization/modelopt_quant.py"
    ) in patch

    added = _added_lines(patch)
    assert added.count("            or self.enable_flashinfer_cutlass_moe") == 2
    assert "            layer.w13_blockscale_swizzled = None" in patch
    assert "            layer.w2_blockscale_swizzled = None" in patch
    assert not any(
        "empty_cache" in line or line.lstrip().startswith("del ") for line in added
    )


def test_qwen_cutlass_scales_are_alias_compatible_and_exactly_budgeted() -> None:
    layers = 48
    experts = 512
    hidden = 2_560
    intermediate = 640
    group_size = 16

    w13_shape = (experts, 2 * intermediate, hidden // group_size)
    w2_shape = (experts, hidden, intermediate // group_size)
    # swizzle_blockscale pads M to 128 and K to 4. Qwen needs no padding, so
    # alias_or_bind_derived_param can copy the swizzle into source storage.
    assert w13_shape[1] % 128 == 0 and w13_shape[2] % 4 == 0
    assert w2_shape[1] % 128 == 0 and w2_shape[2] % 4 == 0

    fp8_bytes_per_layer = (
        w13_shape[0] * w13_shape[1] * w13_shape[2]
        + w2_shape[0] * w2_shape[1] * w2_shape[2]
    )
    assert fp8_bytes_per_layer == 157_286_400
    assert layers * fp8_bytes_per_layer == 7_549_747_200
    assert layers * fp8_bytes_per_layer / 1024**3 == 7.03125


def test_mtp_share_patch_moves_release_before_pool_sizing_with_fallback() -> None:
    patch = MTP_SHARE_PATCH.read_text(encoding="utf-8")
    assert _sha256(MTP_SHARE_PATCH) == MTP_SHARE_PATCH_SHA256
    assert patch.count("diff --git ") == 1
    assert (
        "diff --git a/python/sglang/srt/speculative/eagle_worker_v2.py "
        "b/python/sglang/srt/speculative/eagle_worker_v2.py"
    ) in patch

    added = _added_lines(patch)
    removed = _removed_lines(patch)
    init_sequence = [
        "        self.draft_runner = self.draft_worker.model_runner",
        "        self.init_token_map()",
        "        self.init_lm_head()",
        "        self._lm_head_shared = True",
        "        self._init_dsa_index_share_state()",
    ]
    positions = [patch.index(line) for line in init_sequence]
    assert positions == sorted(positions)
    assert "        if not getattr(self, \"_lm_head_shared\", False):" in added
    assert removed == ["        self.init_token_map()", "        self.init_lm_head()"]


def test_combined_headroom_budget_covers_full_bf16_mtp_and_eight_gib_gate() -> None:
    physical_bytes = 101_973_491_712
    measured_root_peak_bytes = 94_185_324_544
    eager_scale_placeholder_bytes = 7_549_747_200
    mtp_payload_bytes = 5_214_301_696
    vocab_size = 248_320
    hidden_size = 2_560
    bf16_bytes = 2
    stale_draft_embed_and_head_bytes = 2 * vocab_size * hidden_size * bf16_bytes

    assert stale_draft_embed_and_head_bytes == 2_542_796_800
    projected_steady_free = physical_bytes - (
        measured_root_peak_bytes
        - eager_scale_placeholder_bytes
        + mtp_payload_bytes
    )
    projected_late_share_free = (
        projected_steady_free - stale_draft_embed_and_head_bytes
    )
    assert projected_steady_free > 8 * 1024**3
    assert projected_late_share_free < 8 * 1024**3
    assert projected_late_share_free > 6 * 1024**3
