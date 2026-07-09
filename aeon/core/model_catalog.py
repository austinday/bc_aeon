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
# allowed up to the split ceiling so they download on the big machine they
# target but are skipped on small ones. 1.9x fits the largest split model,
# Qwen3.5-397B Q3_K (~178 GiB = 1.86x a 96 GB card); on a 48 GB machine
# 1.9 x 48 = 91 GiB still correctly skips every split model.
DOWNLOAD_VRAM_MULTIPLE = 1.5
DOWNLOAD_VRAM_MULTIPLE_SPLIT = 1.9


@dataclass(frozen=True)
class Mtp:
    """MTP / speculative-draft config. llamacpp uses draft_file; vLLM draft_model."""
    draft_file: Optional[str] = None      # llamacpp: normalized assistant GGUF filename
    draft_model: Optional[str] = None     # vLLM: HF repo id of the draft
    n_max: int = 6
    method: str = "draft_model"           # vLLM --speculative-config method: 'mtp' | 'draft_model' | 'eagle'


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
    kv_quant: Optional[str] = None        # llamacpp -ctk/-ctv ('q4_0'); vLLM --kv-cache-dtype ('fp8'); None=f16
    # llamacpp multimodal projector + explicit chat template, paths relative to
    # model_dir (they must live inside it: the launcher mounts only model_dir).
    mmproj_file: Optional[str] = None
    chat_template_file: Optional[str] = None
    # Override the planner's global SAFETY (fraction of each GPU's VRAM treated
    # as usable, default 0.90). Only for entries whose fit has been measured by
    # hand — e.g. Qwen3.5-397B, whose weights alone exceed 0.90 x 2 GPUs but
    # which verifiably fits at 0.95 thanks to its tiny hybrid-attention KV.
    vram_safety: Optional[float] = None
    # vLLM prefill batch size (--max-num-batched-tokens). Default (small, ~2048 under
    # chunked prefill) chops a big agent prompt into many scheduler steps -> long TTFT.
    # Set high on a model with VRAM headroom so a 20-30k prompt (+ a screenshot's vision
    # tokens) prefills in ~one pass. None = leave vLLM's default.
    max_num_batched_tokens: Optional[int] = None
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

# Port allocation (lb = the port the agent connects to; node0/node1 = dual-copy backends).
CATALOG: List[CatalogEntry] = [
    CatalogEntry(
        # Canonical Gemma-4 path: the abliterated (uncensored) Gemma-4, self-quantized to
        # *plain* NVFP4 (4-bit weights + activations, compressed-tensors) so it runs on
        # native FP4 tensor cores on x86 Blackwell (sm_120) under vLLM 0.23. This is the
        # optimal local path (4-bit is the sweet spot here); the older Q8_0 GGUF/llama.cpp
        # entry was redundant and has been removed. Distinct from the rejected NVFP4_AWQ
        # builds. Native MTP via the official assistant draft (method "mtp"; passing
        # "draft_model" for a gemma-4 assistant silently disables MTP -- vLLM #42005).
        # vLLM fetches the weights from the hub at runtime; setup PHASE 5.6c pre-caches
        # them into ~/.cache/huggingface so first launch isn't a ~20 GB wait.
        name="Gemma-4-31B-NVFP4-MTP",
        family="Gemma-4",
        provider="vllm",
        image="aeon_vllm:latest",
        weights_gib=21.0,            # 20.4 NVFP4 target + 0.9 assistant draft
        kv_gib_per_64k=3.0,          # fp8 KV (Blackwell), ~half f16; sliding-window keeps it modest
        max_ctx=262144,
        ports={"lb": 8016, "node0": 8017, "node1": 8018},
        hf_model="aday777/gemma-4-31B-it-abliterated-NVFP4",
        served_name="Gemma-4-31B-NVFP4",
        mtp=Mtp(draft_model="google/gemma-4-31B-it-assistant", method="mtp", n_max=5),
        kv_quant="fp8",              # vLLM --kv-cache-dtype fp8 on Blackwell FP8 units
        multimodal=True,             # Gemma4ForConditionalGeneration (vision tower): serves images
                                     # on /v1/chat/completions -> analyze_image reuses it (no 2nd GPU model)
        # No download_cmd: vLLM fetches at runtime; setup PHASE 5.6c warms the HF cache.
    ),
    CatalogEntry(
        # Full-precision sibling of the NVFP4 build: the SAME abliterated Gemma-4-31B,
        # served at native BF16 (16-bit weights/activations) instead of self-quantized
        # NVFP4. Higher fidelity (no 4-bit qu/act loss -> better fine-grained grounding &
        # OCR, which is exactly what the browser agent leans on) at ~3x the VRAM. Still
        # fits ONE 96 GB Blackwell card solo on GPU0 (weights ~62 + fp8 KV), same as the
        # NVFP4 entry, and keeps native MTP via the same official assistant draft. Source
        # is huihui-ai's abliterated Gemma-4 (v2: first 5 layers left intact -> fewer
        # spurious refusals) — the same abliterated lineage the NVFP4 build was quantized
        # from. vLLM auto-detects bfloat16 from the checkpoint; the generic launcher needs
        # no quant/dtype flags. max_ctx capped at 128k because BF16 weights leave far less
        # VRAM for KV than the 21 GiB NVFP4 build (which runs 256k).
        name="Gemma-4-31B-BF16-MTP",
        family="Gemma-4",
        provider="vllm",
        image="aeon_vllm:latest",
        weights_gib=66.0,            # ~62 BF16 target + ~0.9 assistant draft, rounded up for planning headroom
        kv_gib_per_64k=3.0,          # fp8 KV (Blackwell) — same architecture as the NVFP4 build
        max_ctx=131072,              # 128k: BF16 weights leave less room for KV than NVFP4
        ports={"lb": 8024, "node0": 8025, "node1": 8026},
        hf_model="huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2",
        served_name="Gemma-4-31B-BF16",
        mtp=Mtp(draft_model="google/gemma-4-31B-it-assistant", method="mtp", n_max=5),
        kv_quant="fp8",              # vLLM --kv-cache-dtype fp8 on Blackwell FP8 units
        multimodal=True,             # Gemma4ForConditionalGeneration vision tower (same as NVFP4)
        # No download_cmd: vLLM fetches at runtime (like the NVFP4 entry). ~62 GiB first pull.
    ),
    CatalogEntry(
        # Qwen3.6-27B (dense, multimodal, Apache-2.0) — a strong AGENTIC/coding model
        # (beats Qwen3.5-397B MoE on coding benchmarks) with NATIVE Multi-Token
        # Prediction built into the weights. This is the AEON-Ultimate uncensored
        # (abliterated: 0/100 refusals) build re-quantized to vanilla-Qwen FP8 (8-bit,
        # block-128) with the MTP block included verbatim. Single-stream decode is the
        # optimization target here: native MTP (K=3, ~+90% decode TPS) + FP8 weights +
        # fp8 KV + CUDA graphs + FlashInfer. Fits one 96 GB card solo on GPU0 (~31 GiB
        # weights) with huge KV headroom (hybrid Gated-DeltaNet: 3/4 layers use linear
        # attention -> small KV), so it runs the full 256k context.
        #
        # NATIVE MTP: unlike the Gemma-4 entries (separate 'assistant' draft), the MTP
        # head is IN this checkpoint, so draft_model is None and the launcher emits a
        # model-less --speculative-config {"method":"mtp","num_speculative_tokens":3}.
        #
        # VISION: multimodal (image-text-to-text), Qwen3_5ForConditionalGeneration with
        # the vision tower kept BF16. multimodal=True so the browser loop / analyze_image
        # reuse it. LIVE-VERIFIED 2026-07-06: passes the startup vision self-test 6/6 at
        # browser resolution (Qwen3.6 OCR is resolution-sensitive — reads fine at the
        # 1920px the browser sends; an earlier low-res probe false-failed it). This is the
        # 8-bit sibling of the Huihui NVFP4 entry below.
        name="Qwen3.6-27B-FP8-MTP",
        family="Qwen3.6",
        provider="vllm",
        image="aeon_vllm:latest",
        weights_gib=32.0,            # ~31 GiB FP8 (7 shards) incl. the in-checkpoint MTP block
        kv_gib_per_64k=2.5,          # fp8 KV; hybrid linear attention keeps KV small
        max_ctx=262144,
        ports={"lb": 8027, "node0": 8028, "node1": 8029},
        hf_model="kasimat/Qwen3.6-27B-AEON-Ultimate-Uncensored-FP8-MTP",
        served_name="Qwen3.6-27B-FP8",
        mtp=Mtp(method="mtp", n_max=3),   # native in-checkpoint MTP: no separate draft model
        kv_quant="fp8",
        # TTFT fix: 32 GiB FP8 weights leave ~50 GiB free on a 96 GiB card, so prefill a
        # 20-30k agent prompt (+ screenshot vision tokens) in ~one pass instead of ~15
        # tiny chunked-prefill steps. Was the main cause of long time-to-first-token.
        max_num_batched_tokens=32768,
        multimodal=True,             # advertised multimodal; live-test vision before relying on it
        # No download_cmd here: setup PHASE 5.6d pre-caches it; vLLM also fetches at runtime.
    ),
    CatalogEntry(
        # Qwen3.6-27B, HUIHUI-lineage abliteration, NVFP4 (4-bit) + native MTP, multimodal.
        # A smaller (~20 GiB vs 32 GiB), different-lineage ALTERNATIVE to the FP8 default
        # above — NOT a vision fix for it. Earlier notes here claimed the kasimat FP8 build
        # "misreads text" (read 'RP9PCV' as 'R171'); that was a LOW-RESOLUTION probe
        # artifact, since corrected. Both builds now PASS the startup vision self-test at
        # the 1920 px the browser actually sends (FP8 re-verified 2026-07-08: read 'A7Y9AR'
        # correctly). Keep this entry as a lighter option / hedge on a different abliteration
        # (huihui-ai spare early layers), not because the default is vision-broken.
        # Architecture is Qwen3_5ForConditionalGeneration with the vision tower EXCLUDED from
        # NVFP4 (visual.* in the ignore list) — the ConditionalGeneration requirement from the
        # NVFP4 lesson is met. Native in-checkpoint MTP (method qwen3_5_mtp, K=3).
        #
        # multimodal=True is still gate-verified: the startup vision self-test
        # (aeon.core.vision_selftest) reads a probe code back before this is trusted for
        # browsing, so a regression fails LOUD rather than browsing blind.
        name="Qwen3.6-27B-Huihui-NVFP4-MTP",
        family="Qwen3.6",
        provider="vllm",
        image="aeon_vllm:latest",
        weights_gib=20.0,            # ~19.1 GiB NVFP4 (single shard) incl. the MTP head
        kv_gib_per_64k=2.5,          # fp8 KV; hybrid linear attention keeps KV small
        max_ctx=262144,
        # 8033-8035: 8030-8032 would collide with the hardcoded browser service on :8030.
        ports={"lb": 8033, "node0": 8034, "node1": 8035},
        hf_model="sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP",
        served_name="Qwen3.6-27B-Huihui-NVFP4",
        mtp=Mtp(method="qwen3_5_mtp", n_max=3),   # native in-checkpoint MTP: no separate draft
        kv_quant="fp8",
        max_num_batched_tokens=32768,  # same TTFT fix as the FP8 entry (roomy 20 GiB weights)
        multimodal=True,             # vision-preserving lineage; the boot self-test verifies it
        # No download_cmd: vLLM fetches at runtime; weights are pre-pulled into the HF cache.
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
    CatalogEntry(
        # Qwen3.5-397B-A17B (MoE: 512 experts / 10 active, ~17B active params),
        # huihui-ai ABLITERATED, Q3_K GGUF — the flagship "big brain" option. Spans
        # BOTH GPUs (weights ~176.5 GiB in 21 shards on 2 x 95.6 GiB).
        #
        # TEXT-ONLY (multimodal=False, mmproj NOT loaded): the repo ships an mmproj
        # and the base model is image-text-to-text, BUT the 176.5 GiB of weights
        # leave GPU0 only ~4 GiB free — and GPU0 also hosts the CLIP encoder + main
        # compute + KV. A browser-res image encode (mtmd_encode -> cuMemCreate) then
        # OOMs GPU0 mid-request. Confirmed live 2026-07-08: KV/prompt-processing ran
        # fine, the crash was purely the vision-encode VRAM spike. This box can't fit
        # 178 GiB weights + the vision encoder together, at ANY context (the KV delta
        # between 128k and 256k is ~0.5 GiB on GPU0; the encode spike is ~2 GiB, so
        # dropping context doesn't rescue it). So this entry serves TEXT-ONLY at the
        # full 256k; use Qwen3.6-27B for browser/vision work.
        #
        # FIT MATH (256k via q4_0 KV): hybrid attention — only 15/60 layers are
        # full attention (2 KV heads x 256 head_dim; the rest Gated-DeltaNet with
        # constant ~0.2 GiB state) — makes KV tiny: ~1.9 GiB/64k f16, ~1.0 q8_0,
        # ~0.5 q4_0. Weights are the constraint, not context: 177.4 GiB of 189.9
        # CUDA-visible leaves ~12 GiB. At q8_0, 256k KV (~4 GiB) left <1.5 GiB/GPU
        # for compute buffers + vision-encode spikes and died in real use, so this
        # used to cap at 128k. Dropping KV to q4_0 halves it: 256k q4_0 KV (~2 GiB)
        # is the SAME footprint as the old 128k q8_0 config — ~3 GiB/GPU headroom
        # that survives real use — so we serve the full 256k, GPU-resident (no CPU
        # offload; fast at every depth) for a mild long-context recall softening.
        # Needs vram_safety 0.95 because weights alone exceed the planner's default
        # 0.90 budget.
        # SPEED: fully GPU-resident (no CPU-offloaded experts), flash-attn,
        # q4_0 KV, only ~17B active params per token despite the 397B total.
        # Native GGUF arch 'qwen35moe' — supported by the llama.cpp build in
        # aeon_ds4:latest (a generic llama-server image despite the name).
        # No MTP: the repo has no draft GGUF.
        # chat_template-vl-think.jinja is the repo's fixed template (tool-calling
        # 500s with the embedded one — see the HF README); it serves text turns fine.
        name="Qwen3.5-397B-A17B-Q3K",
        family="Qwen3.5",
        provider="llamacpp",
        image="aeon_ds4:latest",
        weights_gib=177.2,           # 176.5 shards + ~0.2 recurrent state (mmproj NOT loaded)
        kv_gib_per_64k=0.5,          # q4_0; 15 full-attn layers x 2 KV heads x 256 dim
        max_ctx=262144,              # 256k: q4_0 KV keeps it GPU-resident (see fit math)
        ports={"lb": 8036, "node0": 8037, "node1": 8038},
        model_dir="gguf_models/Huihui-Qwen3.5-397B",
        target_glob="Q3_K-GGUF-00001-of-*.gguf",   # llama.cpp auto-loads the other 20 shards
        kv_quant="q4_0",
        # mmproj deliberately NOT set: vision encoder OOMs GPU0 at this weight size
        # (see header). Restoring it requires freeing GPU0 VRAM, not just re-adding it.
        chat_template_file="chat_template-vl-think.jinja",
        vram_safety=0.95,
        force_split=True,            # 178 GiB never fits one card; always span GPUs
        multimodal=False,            # text-only: vision-encode spike doesn't fit GPU0
        download_dir="gguf_models/Huihui-Qwen3.5-397B",
        # NB: --include must be REPEATED per pattern; bare filenames after it are
        # parsed as positional FILENAMES, which silently makes hf IGNORE --include.
        download_cmd=(
            "hf download huihui-ai/Huihui-Qwen3.5-397B-A17B-abliterated-GGUF "
            "--include 'Q3_K-GGUF/*' --include mmproj-model-f16.gguf "
            "--include chat_template-vl-think.jinja --local-dir /models"
        ),
        download_state="huihui-qwen35-397b-q3k",
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
