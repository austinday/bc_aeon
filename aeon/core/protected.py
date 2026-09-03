"""Protected-core enforcement for self-modification.

Recursive self-modification has a specific failure mode: the cheapest way for an
agent to raise its own score is often to weaken the thing doing the scoring —
delete a hard benchmark case, loosen a success check, or neuter the rollback
machinery so a bad change can no longer be undone. Left unchecked, an optimizer
converges on gaming its own evaluator instead of getting better (Goodhart).

This module marks the constitutional safety boundary as protected: the
self-improvement scorer/rollback gates plus the Fleet admission, resource-route,
and exact-lifecycle implementations. The file-editing tools call ``guard()`` before writing
and refuse to touch a protected file unless a human sets
``AEON_ALLOW_PROTECTED_EDIT=1``. The agent stays free to improve ordinary tools,
prompts, skills, and application logic; it cannot silently dismantle the
mechanisms that constrain compute, measure a change, or revert it.
"""
import os
from pathlib import Path

OVERRIDE_ENV = "AEON_ALLOW_PROTECTED_EDIT"

# Paths (relative to the aeon source root) that constitute the self-modification
# "constitution". Each grades/guards a change or enforces resource ownership;
# letting the agent edit it unsupervised would defeat that boundary.
_PROTECTED_RELATIVE = (
    # Human/fleet policy and this guard itself.
    "AGENTS.md",
    "../AGENTS.md",
    "/home/aday/AGENTS.md",
    "/home/aday/.codex/AGENTS.md",
    "/home/aday/website_hosting/gpu_coord.py",
    "/home/aday/bin/fleet-low-priority",
    "/home/aday/NexusAgentDashboard/fleet_compute",
    "/home/aday/.local/state/fleet-compute",
    "/home/aday/.aeon",
    "../fleet_compute",
    "aeon/core/protected.py",

    # Durable self-improvement validation and rollback boundary.
    "aeon/core/checkpoint.py",
    "aeon/core/bootguard.py",
    "aeon/tools/revert.py",
    "aeon/tools/verify_modification.py",
    "aeon/selfimprove",
    "aeon/benchmarks",
    "aeon/harnesses",
    "aeon/smoke_test.py",
    "aeon/tests",
    "run_tests.sh",
    "setup_environment.sh",
    "setup.py",
    "pyproject.toml",
    "uv.lock",

    # Fleet admission, endpoint ownership, and tool-route enforcement. These
    # files collectively ensure that a tool cannot turn an ordinary command or
    # model call into an unleased GPU launch.
    "aeon/core/fleet_backend.py",
    "aeon/core/fleet_adapter.py",
    "aeon/core/gpu_queue.py",
    "aeon/core/compute_profile.py",
    "aeon/core/model_catalog.py",
    "aeon/core/process_identity.py",
    "aeon/core/qwen_capabilities.py",
    "aeon/core/qwen_runtime.py",
    "aeon/core/qwen_fleet_runtime.py",
    "aeon/core/qwen_fast_service_adapter.py",
    "aeon/core/qwen_speed_lab_adapter.py",
    "aeon/core/qwen_dflash_training_adapter.py",
    "aeon/core/qwen_full_gdn_quant_adapter.py",
    "aeon/core/qwen_artifact_cache.py",
    "aeon/core/video_artifact_cache.py",
    "aeon/core/data",
    "aeon/core/tool_resources.py",
    "aeon/core/sub_agent_environment.py",
    "aeon/core/sub_agent_state.py",
    "aeon/core/durable_agent_guard.py",
    "aeon/core/llm.py",
    "aeon/core/worker.py",
    "aeon/main.py",
    "aeon/tools/base.py",
    "aeon/tools/loader.py",
    "aeon/tools/command_fleet_guard.py",
    "aeon/tools/system.py",
    "aeon/tools/jobs.py",
    "aeon/scripts/command_service_exec.py",
    "aeon/scripts/command_service_controller.py",
    "aeon/scripts/qwen_remote_worker.py",
    "aeon/scripts/warmup_qwen38_vllm.py",
    "aeon/scripts/vllm_uuid_sitecustomize.py",
    "aeon/scripts/qwen_speed_lab_worker.py",
    "aeon/scripts/qwen_dflash_training_worker.py",
    "aeon/scripts/qwen_full_gdn_quant_worker.py",
    "aeon/scripts/train_qwen38_dflash2_exact.py",
    "aeon/scripts/build_qwen38_full_gdn_nvfp4.py",
    "aeon/scripts/benchmark_qwen38_mtp.py",
    "aeon/scripts/benchmark_qwen38_speed.py",
    "aeon/scripts/build_qwen38_speed_variant.py",
    "aeon/scripts/extract_qwen38_dflash_features.py",
    "aeon/scripts/local_http_sitecustomize",
    "aeon/scripts/speed_lab_sitecustomize",

    # Child/service adapters own independent Fleet demands and cleanup. A
    # self-modification must not weaken their identity checks or bypass the
    # broker through a media or delegated-agent tool.
    "aeon/tools/sub_agent.py",
    "aeon/scripts/sub_agent_wrapper.py",
    "aeon/core/comfy_fleet_adapter.py",
    "aeon/core/video_comfy_fleet_adapter.py",
    "aeon/core/video_comfy_release.py",
    "aeon/scripts/comfyui_sitecustomize.py",
    "aeon/scripts/start_comfyui.sh",
    "aeon/scripts/start_video_comfyui_worker.sh",
    "aeon/tools/generate_image.py",
    "aeon/tools/generate_video.py",
    "aeon/tools/vision.py",
    "aeon/tools/composite_image.py",
    "aeon/tools/file_io.py",
    "aeon/tools/analyzers",
    "aeon/tools/search.py",
    "aeon/tools/browser.py",
    "aeon/tools/external_expert.py",
    "aeon/tools/start_agent_instance.py",
    "aeon/tools/set_job_role.py",
    "aeon/tools/mcp.py",
    "aeon/remote/self_settings.py",
    "aeon/remote/mcp_capability.py",
    "aeon/remote/instances.py",
    "aeon/remote/store.py",
    "aeon/remote/security.py",
    "aeon/tools/restart.py",
    "aeon/tools/selfimprove_tool.py",
    "aeon/scripts/searxng_service.py",
    "aeon/scripts/start_searxng.sh",
    "aeon/scripts/browser_service.py",
    "aeon/scripts/start_browser.sh",
    "aeon/services/browser",
)


def _source_root() -> Path:
    try:
        from aeon.core.paths import PROJECT_ROOT
        return Path(PROJECT_ROOT)
    except Exception:
        # paths.py is at <root>/aeon/core/paths.py -> root is two parents up.
        return Path(__file__).resolve().parents[2]


def protected_paths() -> list:
    """Absolute, resolved protected paths, including any user-supplied extras
    listed one-per-line in ``<root>/.aeon_protected`` (lets a human widen the set
    without code changes; '#'-comments and blanks ignored)."""
    root = _source_root()
    rels = list(_PROTECTED_RELATIVE)
    extra = root / ".aeon_protected"
    try:
        if extra.exists():
            for line in extra.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    rels.append(line)
    except Exception:
        pass
    out = []
    for rel in rels:
        try:
            out.append((root / rel).resolve())
        except Exception:
            continue
    return out


def is_protected(abs_path: str) -> bool:
    """True if ``abs_path`` is a protected file or lives inside a protected dir."""
    try:
        target = Path(abs_path).resolve()
    except Exception:
        return False
    # Per-run receipts and command scratch are constitutional state regardless
    # of which user workspace contains them.  A PID/job ID embedded in a path is
    # not authority for a model-facing file tool to rewrite lifecycle evidence.
    if any(part in {"aeon_output", ".aeon-command-scratch"} for part in target.parts):
        return True
    for p in protected_paths():
        if target == p:
            return True
        try:
            target.relative_to(p)  # target is inside protected directory p
            return True
        except ValueError:
            continue
    return False


def override_enabled() -> bool:
    return os.environ.get(OVERRIDE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def guard(abs_path: str):
    """Return a refusal message if editing ``abs_path`` is blocked, else None."""
    if not is_protected(abs_path) or override_enabled():
        return None
    return (
        f"BLOCKED: '{abs_path}' is a PROTECTED harness guardrail (Fleet/resource admission, "
        f"exact lifecycle cleanup, benchmark/scorer, rollback, recovery, or a test gate). "
        f"Editing it would let a self-modification weaken the boundary that constrains, "
        f"measures, and reverts changes, so it is refused. If a human genuinely intends "
        f"this edit, set "
        f"{OVERRIDE_ENV}=1 in the environment and retry. Otherwise improve a NON-protected "
        f"component (tools, prompts, skills, core loop logic) instead."
    )
