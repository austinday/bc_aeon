"""Single source of truth for aeon's local model catalog.

Both setup_environment.sh (what to download, VRAM-gated) and the runtime
(`main.py` menu + adaptive deployment) read this one catalog, so a model is
described exactly once. Each entry carries the model's weight footprint and KV
cost so the deploy planner (`aeon.core.deploy_planner`) can pick a deployment
tier for whatever GPUs the current machine has.

Sizes for models not present locally are documented estimates; the launcher's
OOM-backoff and (for vLLM) gpu-memory-utilization make planning self-correcting.

CLI (consumed by setup_environment.sh):
    python -m aeon.core.model_catalog --emit-downloads <min_vram_gib>
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Download gates (multiple of one GPU's VRAM the weights must fit within).
# Normal models use 1.5x; force_split models (designed to span both GPUs) are
# allowed up to the split ceiling so future entries can target larger machines
# while remaining skipped on smaller ones.
DOWNLOAD_VRAM_MULTIPLE = 1.5
DOWNLOAD_VRAM_MULTIPLE_SPLIT = 1.9


@dataclass(frozen=True)
class Mtp:
    """MTP / speculative-draft config. llamacpp uses draft_file; vLLM draft_model."""
    draft_file: Optional[str] = None      # llamacpp: normalized assistant GGUF filename
    draft_model: Optional[str] = None     # vLLM: HF repo id of the draft
    n_max: int = 6
    method: str = "draft_model"           # vLLM --speculative-config method: 'mtp' | 'draft_model' | 'eagle'
    # Relative to ``aeon/core``. Native MTP is enabled only after this artifact
    # proves the selected K against the exact model build and runtime image.
    selection_manifest: Optional[str] = None


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    family: str
    provider: str                         # 'llamacpp' | 'vllm'
    image: str
    weights_gib: float                    # target (+ draft) on-disk/VRAM footprint
    kv_gib_per_64k: float                 # KV-cache cost per 64k tokens (one copy)
    max_ctx: int
    ports: Dict[str, int]                 # {lb, node0, node1}; single instance uses lb
    # llamacpp serving
    model_dir: Optional[str] = None       # relative to $AEON_HOME/models
    target_glob: Optional[str] = None     # glob to pick the target GGUF in model_dir
    # vLLM serving
    hf_model: Optional[str] = None
    # Relative to $AEON_HOME/models. When set, vLLM must load this validated local
    # artifact instead of fetching a repository from the Hugging Face cache.
    local_model_dir: Optional[str] = None
    served_name: Optional[str] = None
    mtp: Optional[Mtp] = None
    kv_quant: Optional[str] = None        # llamacpp -ctk/-ctv or vLLM --kv-cache-dtype
    attention_backend: Optional[str] = None  # vLLM --attention-backend; bound by MTP release gate
    # llamacpp multimodal projector + explicit chat template, paths relative to
    # model_dir (they must live inside it: the launcher mounts only model_dir).
    mmproj_file: Optional[str] = None
    chat_template_file: Optional[str] = None
    # Override the planner's global SAFETY (fraction of each GPU's VRAM treated
    # as usable, default 0.90). Only for entries whose fit has been measured by
    # hand and cannot be represented by the normal safety allowance.
    vram_safety: Optional[float] = None
    # vLLM prefill batch size (--max-num-batched-tokens). Default (small, ~2048 under
    # chunked prefill) chops a big agent prompt into many scheduler steps -> long TTFT.
    # Set high on a model with VRAM headroom so a 20-30k prompt (+ a screenshot's vision
    # tokens) prefills in ~one pass. None = leave vLLM's default.
    max_num_batched_tokens: Optional[int] = None
    # Concurrent scheduler sequences. Aeon's solo deployment is optimized for
    # one interactive control stream; additional local agents time-share it.
    # Keeping this explicit also prevents vLLM from reserving CUDA graphs for a
    # large idle concurrency default that can exceed a 48 GB lease hard cap.
    max_num_seqs: Optional[int] = None
    force_split: bool = False             # never dual-copy (e.g. always-too-big MoE)
    multimodal: bool = False              # serves vision on its chat endpoint (analyze_image can reuse it)
    # setup download
    download_dir: Optional[str] = None    # relative to $AEON_HOME/models
    download_cmd: Optional[str] = None    # hf download ... (run inside aeon_downloader)
    download_state: Optional[str] = None  # state tag for run_downloader idempotency
    extra_containers: List[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return self.name.lower().replace(".", "_").replace("/", "_")

    @property
    def supports_mtp(self) -> bool:
        return self.mtp is not None


MIN_CTX = 65536  # never deploy below 64k context

# Canonical identities for the one model permitted to receive vision payloads.
# Keep the server-facing value explicit even while it matches the display name.
QWEN38_MODEL_NAME = "Qwen3.8-27B-ARA-NVFP4-MTP"
QWEN38_SERVED_NAME = "Qwen3.8-27B-ARA-NVFP4-MTP"
VISION_MODEL_NAME = QWEN38_SERVED_NAME

# Port allocation (lb = the port the agent connects to; node0/node1 = dual-copy backends).
CATALOG: List[CatalogEntry] = [
    CatalogEntry(
        # Qwen3.8-27B, released 2026-08-14, with reproducible Heretic ARA
        # abliteration and local W4A4 NVFP4 compression. The language head,
        # recurrent convolutions, full vision tower, and all 15 native MTP tensors
        # remain BF16. A source-guarded vLLM structured-output backport fixes the
        # upstream MTP/reasoning boundary bug. The bound K=0..4 release sweep uses
        # real Aeon turn schemas plus text, vision, tool, and system-diagnosis
        # cases; it selects K=3 only when every action is semantically exact and
        # deterministic and measured decode throughput remains >=100 tok/s.
        #
        # This is intentionally local-only: it is the exact checksum-validated
        # artifact built by aeon/scripts/build_qwen38_abliterated_nvfp4.py, not a
        # mutable Hub tag. The launcher mounts it read-only at /models.
        name=QWEN38_MODEL_NAME,
        family="Qwen3.8",
        provider="vllm",
        image="aeon_vllm:latest",
        weights_gib=19.2,            # measured 19.16 GiB across four NVFP4 + one BF16 MTP shard
        # Conservative measured dynamic-FP8 cache allowance: approximately
        # 10 GiB per 64k with native MTP. Include the MTP-expanded footprint in
        # planning rather than reusing the much smaller K=0 estimate.
        kv_gib_per_64k=10.0,
        max_ctx=262144,
        # 8033-8035: 8030-8032 collide with the browser service.
        ports={"lb": 8033, "node0": 8034, "node1": 8035},
        hf_model="/models",
        local_model_dir="Qwen3.8-27B-ARA-abliterated-NVFP4-MTP",
        served_name=QWEN38_SERVED_NAME,
        # n_max is not a guessed architecture default: it must match the
        # versioned K=0..4 benchmark artifact below, which the launcher
        # revalidates against the model and Docker image on every start.
        mtp=Mtp(
            method="mtp",
            n_max=3,
            selection_manifest="data/qwen38_mtp_selection.json",
        ),
        # Per-token/head dynamic scales preserve significantly more KV accuracy
        # than uncalibrated per-tensor FP8 without giving up the FP8 footprint.
        kv_quant="fp8_per_token_head",
        attention_backend="TRITON_ATTN",
        max_num_batched_tokens=32768,
        max_num_seqs=1,
        multimodal=True,
    ),
]


def by_name(name: str) -> Optional[CatalogEntry]:
    for e in CATALOG:
        if e.name == name:
            return e
    return None


def download_multiple(entry: CatalogEntry) -> float:
    return DOWNLOAD_VRAM_MULTIPLE_SPLIT if entry.force_split else DOWNLOAD_VRAM_MULTIPLE


def fits_download(entry: CatalogEntry, min_vram_gib: float) -> bool:
    """A model is downloadable if its weights fit its download multiple x one GPU."""
    return entry.weights_gib <= download_multiple(entry) * min_vram_gib


def emit_downloads(min_vram_gib: float, lite: bool = False) -> int:
    """Print TSV lines for setup_environment.sh to act on. One line per model.

    DOWNLOAD<TAB>dir<TAB>state<TAB>cmd     -> run_downloader should fetch it
    SKIP<TAB>name<TAB>reason               -> too big / lite / nothing to fetch
    """
    for e in CATALOG:
        if not e.download_cmd:
            print(f"SKIP\t{e.name}\tno setup download (provider/other phase fetches it)")
            continue
        if not fits_download(e, min_vram_gib):
            print(f"SKIP\t{e.name}\ttoo big for {min_vram_gib:.0f} GiB GPU "
                  f"({e.weights_gib:.0f} GiB > {download_multiple(e)}x)")
            continue
        if lite and e.weights_gib > min_vram_gib:
            print(f"SKIP\t{e.name}\tlite mode (model exceeds one GPU)")
            continue
        print(f"DOWNLOAD\t{e.download_dir}\t{e.download_state}\t{e.download_cmd}")
    return 0


def _main(argv: List[str]) -> int:
    if argv and argv[0] == "--emit-downloads" and len(argv) >= 2:
        lite = "--lite" in argv[2:]
        return emit_downloads(float(argv[1]), lite=lite)
    sys.stderr.write("usage: python -m aeon.core.model_catalog --emit-downloads <min_vram_gib> [--lite]\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
