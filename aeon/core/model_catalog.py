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
# Normal models use 1.5x; force_split models (designed to span both GPUs, e.g.
# DeepSeek-V4 at ~1.6x on a 96 GB card) are allowed up to the 1.6x split ceiling
# so they download on the big machine they target but are skipped on small ones.
DOWNLOAD_VRAM_MULTIPLE = 1.5
DOWNLOAD_VRAM_MULTIPLE_SPLIT = 1.6


@dataclass(frozen=True)
class Mtp:
    """MTP / speculative-draft config. llamacpp uses draft_file; vLLM draft_model."""
    draft_file: Optional[str] = None      # llamacpp: normalized assistant GGUF filename
    draft_model: Optional[str] = None     # vLLM: HF repo id of the draft
    n_max: int = 6


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
    served_name: Optional[str] = None
    mtp: Optional[Mtp] = None
    kv_quant: Optional[str] = None        # llamacpp -ctk/-ctv (e.g. 'q4_0'); None = default f16
    force_split: bool = False             # never dual-copy (e.g. always-too-big MoE)
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

# Port allocation (lb = the port the agent connects to; node0/node1 = dual-copy backends).
CATALOG: List[CatalogEntry] = [
    CatalogEntry(
        name="Gemma-4-31B-MTP-Q8_0",
        family="Gemma-4",
        provider="llamacpp",
        image="aeon_gemma4_mtp:latest",
        weights_gib=30.7,            # 30.4 target + 0.33 normalized assistant
        kv_gib_per_64k=3.0,          # q4_0 KV, 48 layers
        max_ctx=262144,
        ports={"lb": 8013, "node0": 8014, "node1": 8015},
        model_dir="gguf_models/Gemma-4",
        target_glob="gemma-4-31b-abliterated-Q8_0.gguf",
        mtp=Mtp(draft_file="gemma-4-31B-it-assistant.aeon.Q4_K_M.gguf", n_max=6),
        kv_quant="q4_0",
        download_dir="gguf_models/Gemma-4",
        download_cmd=(
            "hf download paperscarecrow/Gemma-4-31B-it-abliterated "
            "gemma-4-31b-abliterated-Q8_0.gguf --local-dir /models && "
            "hf download AtomicChat/gemma-4-31B-it-assistant-GGUF "
            "--include '*assistant*4_*.gguf' --local-dir /models"
        ),
        download_state="gemma4-q8_0-mtp-v2",
    ),
    # NOTE: the former "Gemma-4-31B-NVFP4" vLLM entry was REMOVED because its only
    # available source (LilaRest/gemma-4-31B-it-NVFP4-turbo) is the STOCK/censored
    # model -- it was mislabeled "Abliterated: Yes". This harness keeps every model
    # uncensored; the abliterated Q8_0 MTP entry above is the canonical Gemma path
    # (best quality, fits 48 GB solo / 96 GB easily). If a fast uncensored vLLM path
    # is wanted later, self-quantize wangzhang/gemma-4-31B-it-abliterated to NVFP4.
    CatalogEntry(
        name="Qwen3.6-35B-A3B-Uncensored",
        family="Qwen3.6",
        provider="llamacpp",
        image="aeon_llamacpp:latest",
        weights_gib=21.0,            # 35B-A3B MoE Q4_K_M (estimate)
        kv_gib_per_64k=2.5,
        max_ctx=262144,
        ports={"lb": 8009, "node0": 8010, "node1": 8011},
        model_dir="vl_models/Qwen3.6-35B-A3B-GGUF",
        target_glob="*Q4_K_M*.gguf",
        # No MTP draft available for this model yet. The GGUF is fetched by the
        # vision-server phase (setup PHASE 5.7) into the same dir, so no separate
        # download here; this entry just locates it for text-chat serving.
    ),
    CatalogEntry(
        name="CyberNeurova-DeepSeek-V4-Flash",
        family="DeepSeek-V4",
        provider="llamacpp",
        image="aeon_ds4:latest",
        weights_gib=153.0,           # 284B MoE Q4KExperts (measured ~153 GiB)
        kv_gib_per_64k=1.0,          # MLA / compressed KV
        max_ctx=262144,
        ports={"lb": 8021, "node0": 8022, "node1": 8023},
        model_dir="gguf_models/CyberNeurova",
        target_glob="*.gguf",
        force_split=True,            # 153 GiB never fits one card; always span GPUs
        download_dir="gguf_models/CyberNeurova",
        download_cmd=(
            "hf download audreyt/CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF "
            "cyberneurova-DeepSeek-V4-Flash-abliterated-Q4KExperts-F16HC-F16Compressor-"
            "F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf --local-dir /models"
        ),
        download_state="cyberneurova-v4-q4k",
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
