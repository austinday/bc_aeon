"""Immutable identities for Aeon's reviewed local video-rendering stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fleet_compute.models import ArtifactKind


VIDEO_SERVICE_ID = "aeon-video-comfyui"
VIDEO_LOCAL_PROFILE_ID = "aeon-video-comfyui-local-177"
VIDEO_WORKER_PROFILE_ID = VIDEO_SERVICE_ID
VIDEO_ADAPTER_ID = "aeon-video-comfyui-runtime-v1"

VIDEO_WORKER_HOSTNAMES = {
    "192.168.0.178": "DAY2XRTX5000",
    "192.168.0.180": "DAY2XRTX5000PRO-2",
}
VIDEO_WORKER_CACHE_ROOT = Path(
    "/home/aday/.local/state/fleet-compute/cache/aeon-video-comfyui"
)
VIDEO_WORKER_SCRATCH_ROOT = Path("/home/aday/.local/state/fleet-compute/runs")

VIDEO_LOCAL_IMAGE_ID = (
    "sha256:e87d7bcd4da3b5826e03740585ee22a5c78bf5f4468e881495375798f677ba8d"
)
VIDEO_IMAGE_ID = (
    "sha256:e87d7bcd4da3b5826e03740585ee22a5c78bf5f4468e881495375798f677ba8d"
)
VIDEO_IMAGE_CONFIG_SHA256 = (
    "75d861d5d12d2f27004d131568356d49952e24e2c500b668e771045bc20c9633"
)
VIDEO_OCI_ARCHIVE_SHA256 = (
    "7b2c2e156cdb70d8c75de1e3b6c6744e9fba6056a6dc0a9c966280a62a276091"
)
VIDEO_OCI_ARCHIVE = Path(
    "/home/aday/.local/state/fleet-compute/artifacts/aeon-video-comfyui/oci/"
    "75d861d5d12d2f27004d131568356d49952e24e2c500b668e771045bc20c9633/"
    "image.tar"
)


@dataclass(frozen=True)
class VideoArtifactSpec:
    artifact_id: str
    identity_key: str
    kind: ArtifactKind
    canonical_path: Path
    digest_sha256: str
    target_relative_path: str | None = None


VIDEO_ARTIFACTS = (
    VideoArtifactSpec(
        "video-image",
        "image",
        ArtifactKind.OCI_ARCHIVE,
        VIDEO_OCI_ARCHIVE,
        VIDEO_IMAGE_CONFIG_SHA256,
    ),
    VideoArtifactSpec(
        "video-worker-launcher",
        "worker_launcher",
        ArtifactKind.FILE,
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "start_video_comfyui_worker.sh",
        "9c91c4afa2ebcf326350a66de9d6fc70bf81b240e6cd1140c898dd08c24c0827",
    ),
    VideoArtifactSpec(
        "video-allocator-cap",
        "allocator_cap",
        ArtifactKind.FILE,
        Path(__file__).resolve().parents[1] / "scripts" / "comfyui_sitecustomize.py",
        "d42d48c48b5bf8329b3d3b63f5a815c3258bf4d9ee9e6e3002c38340543fabf8",
    ),
    VideoArtifactSpec(
        "video-h3-model",
        "h3_model",
        ArtifactKind.FILE,
        Path("/home/aday/.aeon/models/comfyui/unet/")
        / "10Eros_Max_h3_fl2va_beta2_pruned_nvfp4.safetensors",
        "712cb395746a17b46179b493d3addf9b75fbcfcbbcef8bfc19514870782be1f0",
        "unet/10Eros_Max_h3_fl2va_beta2_pruned_nvfp4.safetensors",
    ),
    VideoArtifactSpec(
        "video-h3-encoder",
        "h3_encoder",
        ArtifactKind.FILE,
        Path("/home/aday/.aeon/models/comfyui/text_encoders/")
        / "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
        "a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f",
        "text_encoders/qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
    ),
    VideoArtifactSpec(
        "video-h3-video-vae",
        "h3_video_vae",
        ArtifactKind.FILE,
        Path("/home/aday/.aeon/models/comfyui/vae/")
        / "minimax_h3_video_vae_fp16.safetensors",
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        "vae/minimax_h3_video_vae_fp16.safetensors",
    ),
    VideoArtifactSpec(
        "video-h3-audio-vae",
        "h3_audio_vae",
        ArtifactKind.FILE,
        Path("/home/aday/.aeon/models/comfyui/vae/")
        / "minimax_h3_audio_vae_fp32.safetensors",
        "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ),
    VideoArtifactSpec(
        "video-ltx-model",
        "ltx_model",
        ArtifactKind.FILE,
        Path("/home/aday/.aeon/models/comfyui/unet/") / "10Eros_v1.5-Q8_0.gguf",
        "85338c0fdf5a55e576097bd761c9afcb3f4da23a709189e3257c2dfba70b2e14",
        "unet/10Eros_v1.5-Q8_0.gguf",
    ),
    VideoArtifactSpec(
        "video-ltx-encoder",
        "ltx_encoder",
        ArtifactKind.FILE,
        Path("/home/aday/.aeon/models/comfyui/text_encoders/")
        / "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
        "ca48d724dd86b849525c3a37061829bee45075d0a70971b39715fbb4c325b8b2",
        "text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
    ),
    VideoArtifactSpec(
        "video-ltx-connectors",
        "ltx_connectors",
        ArtifactKind.FILE,
        Path("/home/aday/.aeon/models/comfyui/text_encoders/")
        / "ltx-2.3-22b-dev_embeddings_connectors.safetensors",
        "a5c5148788d8d9d5d1e650e4cbf3502a46a2f7f975ce70c59082732c8905a8ae",
        "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors",
    ),
    VideoArtifactSpec(
        "video-ltx-vae",
        "ltx_vae",
        ArtifactKind.FILE,
        Path("/home/aday/.aeon/models/comfyui/vae/")
        / "ltx-2.3-22b-dev_video_vae.safetensors",
        "8732bb70cf4343541815f45c9f90f5ff0519d679bd63483afc27bf79a08d3f4e",
        "vae/ltx-2.3-22b-dev_video_vae.safetensors",
    ),
)

VIDEO_ARTIFACTS_BY_ID = {item.artifact_id: item for item in VIDEO_ARTIFACTS}
VIDEO_PROFILE_IDENTITIES = {
    item.identity_key: item.digest_sha256 for item in VIDEO_ARTIFACTS
}
VIDEO_LOCAL_PROFILE_IDENTITIES = {
    "image": VIDEO_LOCAL_IMAGE_ID.removeprefix("sha256:"),
    "launcher": "b55df925c5e98c8e1118c68966f91542c2c82cd670011ba239a570fba2410d85",
    "allocator_cap": "d42d48c48b5bf8329b3d3b63f5a815c3258bf4d9ee9e6e3002c38340543fabf8",
    **{
        item.identity_key: item.digest_sha256
        for item in VIDEO_ARTIFACTS
        if item.target_relative_path is not None
    },
}
