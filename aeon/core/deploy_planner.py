"""Pick a deployment tier for a catalog model given the machine's GPUs.

Deployment tiers are decided by physically-grounded VRAM fit (not the fuzzy
1.5x/1.6x download heuristics — those only gate downloads):

  solo    one full copy on one coordinator-approved physical GPU.
  dual    one full copy fits a single GPU -> two copies (one per GPU) + router.
  split   weights+KV fit across all GPUs -> one instance spanning those GPUs.
  offload weights exceed total GPU VRAM -> one instance + CPU/RAM offload.

Qwen uses an exclusive lease. Tool workloads therefore require another safe
coordinator device or wait; no allocator-cap claim is used to imply co-location.

Always keeps context >= MIN_CTX (64k), maximizing it up to the model's max when
headroom allows. Pure function of (entry, gpus): unit-testable without a GPU.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aeon.core.gpu import GpuInfo, min_total_vram_gib
from aeon.core.model_catalog import CatalogEntry, MIN_CTX

SAFETY = 0.90          # fraction of a GPU's VRAM usable (compute buffers, frag headroom)
                       # (entries with a measured fit may override via CatalogEntry.vram_safety)
MAIN_GPU_BUFFER = 1.0  # GiB reserved on GPU0 for the compute buffer in split mode
CTX_GRANULARITY = 8192 # round context down to a multiple of this

# vLLM pre-allocates gpu_memory_utilization * the card's TOTAL VRAM, then turns all
# of the non-weight remainder into KV-cache pool. Filling 85-90% of a 96 GB card
# for a 20 GB model that needs ~10 GB of KV at 256k wastes ~50 GB per GPU on KV
# blocks a single agent never touches. For a single copy per GPU (solo / dual),
# keep two distinct budgets:
#
# * gpu_memory_utilization controls vLLM's persistent weights + KV allocation.
#   It needs only a small steady-state allowance beyond those tensors.
# * the coordinator lease's measured peak plan must additionally cover
#   transient prefill, vision, CUDA-graph, compilation, and fragmentation peaks.
#
# Folding the 12 GiB peak allowance into gpu_memory_utilization converts it into
# permanent KV blocks. The runtime then reaches its planned peak before a warmup
# GEMM can allocate, which was reproduced on a 48 GiB card at 64k context.
VLLM_ALLOCATION_HEADROOM_GIB = 3.0
KV_POOL_HEADROOM_GIB = 12.0  # historical public name; now the peak lease allowance
MAX_TOOL_VRAM_GIB = 40.0
RENTER_RESERVE_GIB = 6.0


def _fit_gpu_mem_util(entry: CatalogEntry, ctx: int, kv_per_tok: float, v_total: float) -> float:
    """gpu_memory_utilization sized to weights + KV(ctx) + headroom, as a fraction
    of one GPU's TOTAL VRAM. Capped at the model's safety ceiling and never below a
    floor that keeps the KV pool able to hold a full max_model_len sequence. The
    launcher still lets an explicit GPU_MEM_UTIL env override win."""
    if v_total <= 0:
        return SAFETY
    reserve_cap = max(0.0, (v_total - RENTER_RESERVE_GIB) / v_total)
    cap = min(entry.vram_safety or SAFETY, reserve_cap)
    target_gib = entry.weights_gib + ctx * kv_per_tok + VLLM_ALLOCATION_HEADROOM_GIB
    return round(min(cap, target_gib / v_total), 3)


def _fit_lease_budget(entry: CatalogEntry, ctx: int, kv_per_tok: float,
                      v_total: float) -> float:
    """Measured peak budget for an exclusive coordinator lease.

    Round upward to one decimal GiB so decimal serialization cannot shave the
    transient allowance, while never crossing the mandatory renter reserve.
    """
    peak = entry.weights_gib + ctx * kv_per_tok + KV_POOL_HEADROOM_GIB
    cap = max(0.0, v_total - RENTER_RESERVE_GIB)
    return min(cap, math.ceil((peak - 1e-9) * 10.0) / 10.0)


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
      'solo'  -> one instance on the first coordinator-approved physical GPU.
                 Qwen remains exclusive; tools require another device or wait.
      'dual'  -> two copies (one per GPU) + router for max LLM throughput (uses BOTH
                 GPUs, so GPU1 is NOT available for tools).
      None    -> auto: solo if it fits one GPU, else split across GPUs, else CPU offload.

    force_split models (too big for one GPU) ignore mode and use split/offload.
    """
    n = len(gpus)
    if n == 0:
        raise ValueError("cannot plan a GPU deployment without coordinator-approved GPUs")
    physical_indices = [gpu.index for gpu in gpus]
    primary_gpu = physical_indices[0]
    v = min_total_vram_gib(gpus)               # min per-GPU total VRAM
    if entry.name == "Qwen3.8-27B-ARA-NVFP4-MTP" and v < 90.0:
        raise ValueError("Qwen3.8 release requires a >=90 GiB physical GPU")
    # The percentage safety ceiling alone leaves <6 GiB on a 48 GiB card.
    # Enforce the fleet's absolute renter reserve independently of card size.
    v_usable = min(v * (entry.vram_safety or SAFETY), v - RENTER_RESERVE_GIB)
    total_usable = n * v_usable
    slug = _slug(entry.name)
    kv_per_tok = entry.kv_gib_per_64k / 65536.0
    one_copy_min = (entry.weights_gib + entry.kv_gib_per_64k
                    + KV_POOL_HEADROOM_GIB)

    fits_one_gpu = one_copy_min <= v_usable
    fits_gpus = (one_copy_min + MAIN_GPU_BUFFER) <= total_usable

    if entry.force_split:
        tier = "split" if fits_gpus else "offload"
    elif mode == "dual" and n >= 2 and fits_one_gpu:
        tier = "dual"
    elif mode == "solo" and fits_one_gpu:
        tier = "solo"
    elif fits_one_gpu:
        tier = "solo"
    elif fits_gpus and n >= 2:
        tier = "split"
    else:
        tier = "offload"

    launcher = ("launch_vllm_adaptive.sh" if entry.provider == "vllm"
                else "launch_llamacpp_adaptive.sh")
    lb_port = entry.ports["lb"]
    base_url = f"http://localhost:{lb_port}/v1"
    # ``mtp`` describes the active deployment, not merely an in-checkpoint draft
    # head: a future fail-closed release manifest may legitimately select K=0.
    mtp = bool(entry.mtp and entry.mtp.n_max > 0)
    # Fitted gpu_memory_utilization for single-copy tiers; "" (tier default) for
    # split/offload, which legitimately fill the card. Set in the solo/dual branches.
    gpu_mem_util = ""
    lease_budget_gb = ""

    if tier == "solo":
        # One instance on the first coordinator-approved physical GPU. On a
        # Qwen's exact release tuple is measured at 114688 context on the roomy
        # local card. Its exclusive lease never authorizes tool co-location.
        ctx = _round_ctx(
            (v_usable - entry.weights_gib - KV_POOL_HEADROOM_GIB) / kv_per_tok,
            entry.max_ctx,
        )
        tool_gpu_policy = "separate-preferred"
        if n == 1:
            shared_llm_budget = v - MAX_TOOL_VRAM_GIB - RENTER_RESERVE_GIB
            shared_ctx = ((shared_llm_budget - entry.weights_gib - KV_POOL_HEADROOM_GIB)
                          / kv_per_tok)
            minimum_shared = (entry.weights_gib + entry.kv_gib_per_64k
                              + KV_POOL_HEADROOM_GIB)
            if shared_llm_budget >= minimum_shared:
                ctx = min(ctx, _round_ctx(shared_ctx, entry.max_ctx))
                tool_gpu_policy = "shared-single-gpu"
            else:
                tool_gpu_policy = "insufficient-shared-capacity"
        if entry.name == "Qwen3.8-27B-ARA-NVFP4-MTP":
            ctx = 114688
            tool_gpu_policy = "exclusive-separate-required"
        gpu_mem_util = _fit_gpu_mem_util(entry, ctx, kv_per_tok, v)
        lease_budget_gb = _fit_lease_budget(entry, ctx, kv_per_tok, v)
        node = {"role": "node", "devices": str(primary_gpu), "port": lb_port, "ctx": ctx,
                "tensor_split": "", "ngl": 99, "cpu_offload_gib": 0.0,
                "container": f"aeon_{slug}"}
        plan_obj = DeployPlan(
            entry_name=entry.name, provider=entry.provider, image=entry.image, tier=tier,
            nodes=[node], lb_port=lb_port, health_port=lb_port, base_url=base_url,
            container_name=f"aeon_{slug}", all_containers=[node["container"]], launcher=launcher,
            context_limit=ctx, mtp=mtp,
            label=_label(entry, "solo", physical_indices, ctx, mtp, tool_gpu_policy),
        )
    elif tier == "dual":
        # Largest context that fits one GPU alongside one copy of the weights.
        ctx = _round_ctx(
            (v_usable - entry.weights_gib - KV_POOL_HEADROOM_GIB) / kv_per_tok,
            entry.max_ctx,
        )
        gpu_mem_util = _fit_gpu_mem_util(entry, ctx, kv_per_tok, v)
        lease_budget_gb = _fit_lease_budget(entry, ctx, kv_per_tok, v)
        tool_gpu_policy = "all-gpus-occupied"
        node0 = {"role": "node", "devices": str(physical_indices[0]), "port": entry.ports["node0"], "ctx": ctx,
                 "tensor_split": "", "ngl": 99, "cpu_offload_gib": 0.0,
                 "container": f"aeon_{slug}_node0"}
        node1 = {"role": "node", "devices": str(physical_indices[1]), "port": entry.ports["node1"], "ctx": ctx,
                 "tensor_split": "", "ngl": 99, "cpu_offload_gib": 0.0,
                 "container": f"aeon_{slug}_node1"}
        containers = [node0["container"], node1["container"], f"aeon_{slug}_lb"]
        plan_obj = DeployPlan(
            entry_name=entry.name, provider=entry.provider, image=entry.image, tier=tier,
            nodes=[node0, node1], lb_port=lb_port, health_port=lb_port, base_url=base_url,
            container_name=f"aeon_{slug}_lb", all_containers=containers, launcher=launcher,
            context_limit=ctx, mtp=mtp,
            label=_label(entry, "dual", physical_indices, ctx, mtp, tool_gpu_policy),
        )
    else:
        # Single instance spanning all GPUs (split) or GPUs+RAM (offload).
        budget = total_usable - MAIN_GPU_BUFFER
        if tier == "split":
            ctx = _round_ctx(
                (budget - entry.weights_gib - KV_POOL_HEADROOM_GIB) / kv_per_tok,
                entry.max_ctx,
            )
            kv_total = ctx * kv_per_tok
            tsplit = _tensor_split_gpu0_weighted(entry.weights_gib + kv_total, v_usable, n)
            cpu_offload = 0.0
            ngl = 99
        else:  # offload
            ctx = MIN_CTX
            kv_total = ctx * kv_per_tok
            tsplit = _tensor_split_gpu0_weighted(min(entry.weights_gib, budget), v_usable, n)
            # Overflow that must live in system RAM.
            cpu_offload = max(
                0.0,
                entry.weights_gib + kv_total + KV_POOL_HEADROOM_GIB
                + MAIN_GPU_BUFFER - total_usable,
            )
            # Fraction of layers that still fit on the GPUs (llamacpp -ngl proxy).
            gpu_frac = max(0.0, min(
                1.0,
                (total_usable - kv_total - KV_POOL_HEADROOM_GIB - MAIN_GPU_BUFFER)
                / entry.weights_gib,
            ))
            ngl = max(1, int(gpu_frac * 999))  # launcher clamps to model's real layer count
        tool_gpu_policy = "all-gpus-occupied"
        node = {"role": "single", "devices": ",".join(str(i) for i in physical_indices),
                "port": lb_port, "ctx": ctx, "tensor_split": tsplit, "ngl": ngl,
                "cpu_offload_gib": round(cpu_offload, 1), "container": f"aeon_{slug}"}
        containers = [node["container"]]
        plan_obj = DeployPlan(
            entry_name=entry.name, provider=entry.provider, image=entry.image, tier=tier,
            nodes=[node], lb_port=lb_port, health_port=lb_port, base_url=base_url,
            container_name=f"aeon_{slug}", all_containers=containers, launcher=launcher,
            context_limit=ctx, mtp=mtp,
            label=_label(entry, tier, physical_indices, ctx, mtp, tool_gpu_policy),
        )

    plan_obj.env = {
        "AEON_DEPLOY_PLAN": plan_obj.as_env_json(),
        "AEON_MODEL_DIR": entry.model_dir or "",
        "AEON_TARGET_GLOB": entry.target_glob or "",
        "AEON_HF_MODEL": entry.hf_model or "",
        "AEON_LOCAL_MODEL_DIR": entry.local_model_dir or "",
        "AEON_SERVED_NAME": entry.served_name or entry.name,
        "AEON_MTP_DRAFT_FILE": (entry.mtp.draft_file if entry.mtp and entry.mtp.draft_file else ""),
        "AEON_MTP_DRAFT_MODEL": (entry.mtp.draft_model if entry.mtp and entry.mtp.draft_model else ""),
        "AEON_MTP_METHOD": (entry.mtp.method if entry.mtp else "draft_model"),
        "AEON_MTP_NMAX": str(entry.mtp.n_max if entry.mtp else 0),
        "AEON_MTP_SELECTION_MANIFEST": (
            entry.mtp.selection_manifest
            if entry.mtp and entry.mtp.selection_manifest else ""
        ),
        "AEON_KV_QUANT": entry.kv_quant or "",
        "AEON_VLLM_ATTENTION_BACKEND": entry.attention_backend or "",
        # llamacpp vision (--mmproj) + explicit chat template, relative to model_dir.
        "AEON_MMPROJ_FILE": entry.mmproj_file or "",
        "AEON_CHAT_TEMPLATE_FILE": entry.chat_template_file or "",
        # vLLM --max-num-batched-tokens: raise prefill batch so a big agent prompt
        # prefills in ~one pass (low TTFT) instead of many chunked-prefill steps.
        "AEON_MAX_NUM_BATCHED": str(entry.max_num_batched_tokens or ""),
        "AEON_MAX_NUM_SEQS": str(entry.max_num_seqs or ""),
        # Fitted gpu_memory_utilization (solo/dual); empty for split/offload so the
        # launcher keeps the tier default. Stops vLLM claiming the whole card for a
        # KV pool a single agent never uses.
        "AEON_GPU_MEM_UTIL": str(gpu_mem_util) if gpu_mem_util else "",
        # Unlike gpu_memory_utilization, this is the measured aggregate peak
        # plan recorded by the exclusive coordinator lease. It is not a cgroup
        # or aggregate per-process hard cap.
        "AEON_LLM_VRAM_BUDGET_GB": (
            f"{lease_budget_gb:g}" if lease_budget_gb else ""
        ),
        # Sizing identity only. Placement remains coordinator-owned and the
        # returned GPU UUID replaces the planner's diagnostic physical index.
        "AEON_PLANNED_GPU_TOTAL_GB": f"{v:g}",
        "AEON_TOOL_GPU_POLICY": tool_gpu_policy,
        "AEON_MAX_TOOL_VRAM_GB": str(MAX_TOOL_VRAM_GIB),
        "AEON_RENTER_RESERVE_GB": str(RENTER_RESERVE_GIB),
    }
    return plan_obj


def _label(entry: CatalogEntry, tier: str, physical_indices: List[int], ctx: int,
           mtp: bool, tool_gpu_policy: str) -> str:
    ctx_h = f"{ctx // 1024}k"
    mtp_h = "MTP " if mtp else ""
    prov = "vLLM" if entry.provider == "vllm" else "llama.cpp"
    joined = "+".join(f"GPU{i}" for i in physical_indices)
    if tier == "solo" and tool_gpu_policy == "shared-single-gpu":
        desc = f"GPU{physical_indices[0]} shared with tools"
    elif tier == "solo" and tool_gpu_policy == "insufficient-shared-capacity":
        desc = f"GPU{physical_indices[0]} (insufficient capacity for tool co-location)"
    elif tier == "solo":
        desc = f"GPU{physical_indices[0]} (separate tool GPU preferred)"
    else:
        desc = {
            "dual": f"Dual-copy {joined} (max throughput)",
            "split": f"{joined} split",
            "offload": f"{joined}+CPU offload",
        }.get(tier, tier)
    return (f"{entry.name:<24} | {mtp_h}{desc} | {ctx_h} ctx | "
            f"Uncensored: Yes | Local/{prov}")
