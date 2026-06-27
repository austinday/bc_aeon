"""Pick a deployment tier for a catalog model given the machine's GPUs.

Three tiers, decided by physically-grounded VRAM fit (not the fuzzy 1.5x/1.6x
download heuristics — those only gate downloads):

  dual    one full copy fits a single GPU  -> two copies (one per GPU) + router.
  split   weights+KV fit across all GPUs    -> one instance, GPU0-weighted split.
  offload weights exceed total GPU VRAM     -> one instance + CPU/RAM offload.

Always keeps context >= MIN_CTX (64k), maximizing it up to the model's max when
headroom allows. Pure function of (entry, gpus): unit-testable without a GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aeon.core.gpu import GpuInfo, min_total_vram_gib
from aeon.core.model_catalog import CatalogEntry, MIN_CTX

SAFETY = 0.90          # fraction of a GPU's VRAM usable (compute buffers, frag headroom)
MAIN_GPU_BUFFER = 1.0  # GiB reserved on GPU0 for the compute buffer in split mode
CTX_GRANULARITY = 8192 # round context down to a multiple of this


def _round_ctx(ctx: float, max_ctx: int) -> int:
    c = int(ctx // CTX_GRANULARITY) * CTX_GRANULARITY
    c = max(MIN_CTX, min(c, max_ctx))
    return c


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower())


@dataclass
class DeployPlan:
    entry_name: str
    provider: str
    image: str
    tier: str                       # 'dual' | 'split' | 'offload'
    nodes: List[Dict]               # per-instance launch params
    lb_port: int
    health_port: int                # port the agent connects to
    base_url: str
    container_name: str             # primary container (registry / refcount key)
    all_containers: List[str]       # everything to tear down
    launcher: str                   # bash launcher script name
    context_limit: int
    label: str
    mtp: bool = False
    env: Dict[str, str] = field(default_factory=dict)

    def as_env_json(self) -> str:
        return json.dumps({
            "entry_name": self.entry_name,
            "provider": self.provider,
            "image": self.image,
            "tier": self.tier,
            "nodes": self.nodes,
            "lb_port": self.lb_port,
            "health_port": self.health_port,
            "container_name": self.container_name,
            "all_containers": self.all_containers,
            "context_limit": self.context_limit,
            "mtp": self.mtp,
        })


def _tensor_split_gpu0_weighted(weight_plus_kv: float, v_usable: float, n: int) -> str:
    """GPU0-weighted layer split: pack GPU0 as full as safe, spill the rest to GPU1.."""
    # Fraction of layers GPU0 can hold (leaving its compute buffer); never <complement.
    f0_max = (v_usable - MAIN_GPU_BUFFER) / weight_plus_kv
    f0_min = 1.0 - (v_usable / weight_plus_kv)          # GPU1 must hold the remainder
    f0 = min(0.95, max(f0_min, f0_max))
    f0 = max(0.50, min(0.95, f0))                       # keep it sane / GPU0-weighted
    g0 = int(round(f0 * 100))
    return f"{g0},{100 - g0}"


def plan(entry: CatalogEntry, gpus: List[GpuInfo], mode: Optional[str] = None) -> DeployPlan:
    """Plan a deployment for `entry` on `gpus`.

    mode:
      'solo'  -> one instance on GPU0 only, MTP on, leaving GPU1 free for tools
                 (ComfyUI / vision). This is the DEFAULT for models that fit one GPU,
                 because this harness runs image/video/vision tools on GPU1.
      'dual'  -> two copies (one per GPU) + router for max LLM throughput (uses BOTH
                 GPUs, so GPU1 is NOT available for tools).
      None    -> auto: solo if it fits GPU0, else split across GPUs, else CPU offload.

    force_split models (too big for one GPU) ignore mode and use split/offload.
    """
    n = len(gpus)
    v = min_total_vram_gib(gpus)               # min per-GPU total VRAM
    v_usable = v * SAFETY
    total_usable = n * v_usable
    slug = _slug(entry.name)
    kv_per_tok = entry.kv_gib_per_64k / 65536.0
    one_copy_min = entry.weights_gib + entry.kv_gib_per_64k  # weights + KV(64k)

    fits_one_gpu = one_copy_min <= v_usable
    fits_gpus = (entry.weights_gib + entry.kv_gib_per_64k + MAIN_GPU_BUFFER) <= total_usable

    if entry.force_split:
        tier = "split" if fits_gpus else "offload"
    elif mode == "dual" and n >= 2 and fits_one_gpu:
        tier = "dual"
    elif mode == "solo" and fits_one_gpu:
        tier = "solo"
    elif fits_one_gpu:
        tier = "solo"                          # auto default: keep GPU1 free for tools
    elif fits_gpus and n >= 2:
        tier = "split"
    else:
        tier = "offload"

    launcher = ("launch_vllm_adaptive.sh" if entry.provider == "vllm"
                else "launch_llamacpp_adaptive.sh")
    lb_port = entry.ports["lb"]
    base_url = f"http://localhost:{lb_port}/v1"
    mtp = entry.supports_mtp

    if tier == "solo":
        # One instance on GPU0; agent connects to it directly (no router). GPU1 stays free.
        ctx = _round_ctx((v_usable - entry.weights_gib) / kv_per_tok, entry.max_ctx)
        node = {"role": "node", "devices": "0", "port": lb_port, "ctx": ctx,
                "tensor_split": "", "ngl": 99, "cpu_offload_gib": 0.0,
                "container": f"aeon_{slug}"}
        plan_obj = DeployPlan(
            entry_name=entry.name, provider=entry.provider, image=entry.image, tier=tier,
            nodes=[node], lb_port=lb_port, health_port=lb_port, base_url=base_url,
            container_name=f"aeon_{slug}", all_containers=[node["container"]], launcher=launcher,
            context_limit=ctx, mtp=mtp,
            label=_label(entry, "solo", n, ctx, mtp),
        )
    elif tier == "dual":
        # Largest context that fits one GPU alongside one copy of the weights.
        ctx = _round_ctx((v_usable - entry.weights_gib) / kv_per_tok, entry.max_ctx)
        node0 = {"role": "node", "devices": "0", "port": entry.ports["node0"], "ctx": ctx,
                 "tensor_split": "", "ngl": 99, "cpu_offload_gib": 0.0,
                 "container": f"aeon_{slug}_node0"}
        node1 = {"role": "node", "devices": "1", "port": entry.ports["node1"], "ctx": ctx,
                 "tensor_split": "", "ngl": 99, "cpu_offload_gib": 0.0,
                 "container": f"aeon_{slug}_node1"}
        containers = [node0["container"], node1["container"], f"aeon_{slug}_lb"]
        plan_obj = DeployPlan(
            entry_name=entry.name, provider=entry.provider, image=entry.image, tier=tier,
            nodes=[node0, node1], lb_port=lb_port, health_port=lb_port, base_url=base_url,
            container_name=f"aeon_{slug}_lb", all_containers=containers, launcher=launcher,
            context_limit=ctx, mtp=mtp,
            label=_label(entry, "dual", n, ctx, mtp),
        )
    else:
        # Single instance spanning all GPUs (split) or GPUs+RAM (offload).
        budget = total_usable - MAIN_GPU_BUFFER
        if tier == "split":
            ctx = _round_ctx((budget - entry.weights_gib) / kv_per_tok, entry.max_ctx)
            kv_total = ctx * kv_per_tok
            tsplit = _tensor_split_gpu0_weighted(entry.weights_gib + kv_total, v_usable, n)
            cpu_offload = 0.0
            ngl = 99
        else:  # offload
            ctx = MIN_CTX
            kv_total = ctx * kv_per_tok
            tsplit = _tensor_split_gpu0_weighted(min(entry.weights_gib, budget), v_usable, n)
            # Overflow that must live in system RAM.
            cpu_offload = max(0.0, entry.weights_gib + kv_total + MAIN_GPU_BUFFER - total_usable)
            # Fraction of layers that still fit on the GPUs (llamacpp -ngl proxy).
            gpu_frac = max(0.0, min(1.0, (total_usable - kv_total - MAIN_GPU_BUFFER) / entry.weights_gib))
            ngl = max(1, int(gpu_frac * 999))  # launcher clamps to model's real layer count
        node = {"role": "single", "devices": ",".join(str(i) for i in range(n)),
                "port": lb_port, "ctx": ctx, "tensor_split": tsplit, "ngl": ngl,
                "cpu_offload_gib": round(cpu_offload, 1), "container": f"aeon_{slug}"}
        containers = [node["container"]]
        plan_obj = DeployPlan(
            entry_name=entry.name, provider=entry.provider, image=entry.image, tier=tier,
            nodes=[node], lb_port=lb_port, health_port=lb_port, base_url=base_url,
            container_name=f"aeon_{slug}", all_containers=containers, launcher=launcher,
            context_limit=ctx, mtp=mtp,
            label=_label(entry, tier, n, ctx, mtp),
        )

    plan_obj.env = {
        "AEON_DEPLOY_PLAN": plan_obj.as_env_json(),
        "AEON_MODEL_DIR": entry.model_dir or "",
        "AEON_TARGET_GLOB": entry.target_glob or "",
        "AEON_HF_MODEL": entry.hf_model or "",
        "AEON_SERVED_NAME": entry.served_name or entry.name,
        "AEON_MTP_DRAFT_FILE": (entry.mtp.draft_file if entry.mtp and entry.mtp.draft_file else ""),
        "AEON_MTP_DRAFT_MODEL": (entry.mtp.draft_model if entry.mtp and entry.mtp.draft_model else ""),
        "AEON_MTP_NMAX": str(entry.mtp.n_max if entry.mtp else 0),
        "AEON_KV_QUANT": entry.kv_quant or "",
    }
    return plan_obj


def _label(entry: CatalogEntry, tier: str, n: int, ctx: int, mtp: bool) -> str:
    ctx_h = f"{ctx // 1024}k"
    mtp_h = "MTP " if mtp else ""
    prov = "vLLM" if entry.provider == "vllm" else "llama.cpp"
    desc = {
        "solo":    "GPU0 only (GPU1 free for tools)",
        "dual":    "Dual-copy both GPUs (max throughput)",
        "split":   "GPU0+GPU1 split",
        "offload": "GPU0+GPU1+CPU offload",
    }.get(tier, tier)
    return (f"{entry.name:<24} | {mtp_h}{desc} | {ctx_h} ctx | "
            f"Uncensored: Yes | Local/{prov}")
