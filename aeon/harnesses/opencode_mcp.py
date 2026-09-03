"""Stdio MCP bridge exposing reviewed Aeon tools to the OpenCode harness."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import signal
import stat
import sys
import threading
from pathlib import Path
from typing import Any

import anyio
import requests
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from aeon.core.llm import LLMClient
from aeon.core.model_catalog import VISION_MODEL_NAMES
from aeon.core.process_resources import cancel_process_resources
from aeon.core.tool_resources import tool_resource_policy
from aeon.core.worker import Worker
from aeon.tools.loader import load_tools_from_directory

from .opencode_completion import CompletionStateWriter, OpenCodeCompletionError
from .opencode_config import _atomic_private_bytes, _private_directory


MAX_AUTHORITY_BYTES = 40_001
MAX_IMAGE_RESULT_BYTES = 10 * 1024 * 1024
_INSTANCE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_LOCAL_HTTP = {
    "allow_redirects": False,
    "proxies": {"http": "", "https": ""},
}

# Keep the catalog task-oriented. OpenCode prefixes these with ``aeon_``.
EXPOSED_TOOLS = frozenset(
    {
        "open_file",
        "close_file",
        "write_file",
        "str_replace",
        "run_command",
        "run_command_async",
        "job_output",
        "kill_job",
        "inspect_tool_result",
        "search_web",
        "browser_navigate",
        "browser_interact",
        "browser_read",
        "browser_find",
        "browser_extract",
        "browser_switch_tab",
        "browser_close_tab",
        "browser_capture_media",
        "analyze_image",
        "generate_image",
        "edit_image",
        "composite_image",
        "generate_video",
        "fleet_batch_capabilities",
        "fleet_submit_batch_job",
        "fleet_batch_job_status",
        "spawn_sub_agent",
        "get_sub_agent_status",
        "get_sub_agent_report",
        "gather_sub_agents",
        "steer_sub_agent",
        "kill_sub_agent",
        "integrate_sub_agent_changes",
        "blackboard_post",
        "blackboard_read",
        "consult_external_expert",
        "memorize",
        "forget",
        "list_memories",
        "activate_skill",
        "deactivate_skill",
        "create_skill",
        "read_skill",
        "delete_skill",
        "remember_skill_knowledge",
        "list_skill_knowledge",
        "read_skill_knowledge",
        "search_skill_knowledge",
        "delete_skill_knowledge",
        "set_job_role",
        "start_agent_instance",
        "create_collaboration_portal",
        "send_collaborator_handoff",
        "connect_mcp_account",
        "list_mcp_credentials",
        "list_provider_credentials",
        "list_payment_addresses",
        "list_mcp_tools",
        "call_mcp_tool",
        "github_repositories",
        "github_status",
        "github_commit",
        "github_push",
        "github_verify_remote",
        "huggingface_model_search",
        "huggingface_model_info",
        "huggingface_repo_file",
        "huggingface_account",
        "huggingface_publish_model",
        "huggingface_verify_publication",
        "verify_self_modification",
    }
)


class BridgeError(RuntimeError):
    """The supervisor-to-MCP capability handoff was invalid."""


def _private_text_from_environment(name: str, maximum: int) -> str:
    value = os.environ.get(name, "")
    path = Path(value)
    if not value or not path.is_absolute():
        raise BridgeError(f"{name} is unavailable")
    try:
        _private_directory(path.parent, create=False)
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
    except (OSError, RuntimeError) as exc:
        raise BridgeError(f"{name} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or not 1 <= metadata.st_size <= maximum
        ):
            raise BridgeError(f"{name} is not an owner-private regular file")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) > maximum:
        raise BridgeError(f"{name} exceeds its size limit")
    try:
        result = payload.decode("utf-8").strip()
    except UnicodeError as exc:
        raise BridgeError(f"{name} is not UTF-8") from exc
    if not result or "\x00" in result:
        raise BridgeError(f"{name} is empty or invalid")
    return result


def _proxy_guard() -> None:
    base_url = os.environ.get("AEON_OPENCODE_PROXY_URL", "")
    token = os.environ.get("AEON_OPENCODE_PROXY_TOKEN", "")
    if not base_url.startswith("http://127.0.0.1:") or not token:
        raise BridgeError("OpenCode model capability is unavailable")
    response = requests.get(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
        **_LOCAL_HTTP,
    )
    try:
        if response.status_code != 200:
            raise BridgeError("Fleet model capability is not ready")
    finally:
        response.close()


def _browser_state_path() -> Path | None:
    value = os.environ.get("AEON_OPENCODE_BROWSER_STATE", "")
    path = Path(value)
    return path if value and path.is_absolute() else None


def _load_browser_tab() -> str | None:
    path = _browser_state_path()
    if path is None:
        return None
    descriptor: int | None = None
    try:
        _private_directory(path.parent, create=False)
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > 256
        ):
            return None
        payload = os.read(descriptor, 257)
        value = payload.decode("utf-8").strip()
        return value if value and len(value) <= 128 else None
    except (OSError, UnicodeError, RuntimeError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _save_browser_tab(worker: Worker) -> None:
    value = str(getattr(worker, "_last_browser_tab", "") or "").strip()
    path = _browser_state_path()
    if path is None or not value or len(value) > 128 or "\x00" in value:
        return
    try:
        _private_directory(path.parent, create=False)
        _atomic_private_bytes(path.parent, path.name, (value + "\n").encode("utf-8"))
    except (OSError, RuntimeError):
        return


def _build_worker() -> tuple[Worker, dict[str, Any]]:
    authority = _private_text_from_environment(
        "AEON_OPENCODE_AUTHORITY_FILE", MAX_AUTHORITY_BYTES
    )
    base_url = os.environ.get("AEON_OPENCODE_PROXY_URL", "")
    token = os.environ.get("AEON_OPENCODE_PROXY_TOKEN", "")
    logical_model = os.environ.get("AEON_OPENCODE_LOGICAL_MODEL", "")
    wire_model = os.environ.get("AEON_OPENCODE_WIRE_MODEL", "")
    if wire_model not in VISION_MODEL_NAMES:
        raise BridgeError("OpenCode wire model is not reviewed")
    model_config = {
        "provider": "vllm",
        "model": logical_model,
        "api_model": wire_model,
        "base_url": base_url,
        "api_key": token,
        "context_limit": 114688,
        "multimodal": True,
    }
    llm_client = LLMClient(model_config, before_local_request=_proxy_guard)
    worker = Worker(
        llm_client=llm_client,
        print_func=lambda *values, **kwargs: print(
            *values, file=sys.stderr, **{k: v for k, v in kwargs.items() if k != "file"}
        ),
    )
    instance_id = os.environ.get("AEON_OPENCODE_INSTANCE_ID", "")
    if not _INSTANCE_ID_RE.fullmatch(instance_id):
        raise BridgeError("OpenCode worker identity is unavailable")
    worker.instance_id = instance_id
    worker.compute_guard = _proxy_guard
    worker.model_name = logical_model
    worker.model_config = model_config
    previous_tab = _load_browser_tab()
    if previous_tab:
        worker._last_browser_tab = previous_tab
    errors: list[str] = []
    loaded = load_tools_from_directory(
        "aeon.tools",
        dependencies={"llm_client": llm_client, "worker": worker},
        errors_out=errors,
    )
    selected = [tool for tool in loaded if tool.name in EXPOSED_TOOLS]
    if errors:
        raise BridgeError("One or more reviewed Aeon tool modules failed to load")
    names = {tool.name for tool in selected}
    missing = EXPOSED_TOOLS - names
    # Optional tools are allowed to be absent only when their dependency is not
    # installed; core file/command/browser/vision tools are mandatory.
    required = {
        "open_file",
        "write_file",
        "str_replace",
        "run_command",
        "search_web",
        "browser_navigate",
        "browser_interact",
        "browser_read",
        "analyze_image",
    }
    if missing & required:
        raise BridgeError(
            "Required Aeon tools are unavailable: " + ", ".join(sorted(missing & required))
        )
    worker.register_tools(selected)
    from aeon.tools.categories import get_all_category_paths

    worker.expanded_categories.update(get_all_category_paths())
    # OpenCode starts a new local MCP process for each CLI turn. Restore the
    # reviewed Worker's bounded, owner-private state before opening the current
    # request so memory and waiting-request guards keep their legacy semantics.
    worker._maybe_load_persisted_state(authority)
    worker._begin_protocol_request(authority)
    return worker, {name: worker.tools[name] for name in sorted(worker.tools)}


def _image_content(path_value: str) -> types.ImageContent | None:
    path = Path(path_value)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
        metadata = os.fstat(descriptor)
        if (
            not path.is_absolute()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or not 1 <= metadata.st_size <= MAX_IMAGE_RESULT_BYTES
        ):
            return None
        suffix = path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix)
        if mime is None:
            return None
        chunks: list[bytes] = []
        remaining = MAX_IMAGE_RESULT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if not 1 <= len(payload) <= MAX_IMAGE_RESULT_BYTES:
            return None
        return types.ImageContent(
            type="image",
            data=base64.b64encode(payload).decode("ascii"),
            mimeType=mime,
        )
    except OSError:
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _opencode_input_schema(name: str, tool: Any) -> dict[str, Any]:
    """Adapt object unions that pinned OpenCode cannot import from MCP.

    OpenCode v1.18.27 drops the entire MCP catalog when a tool exposes a
    top-level ``oneOf``.  Aeon's ``str_replace`` uses one to describe its patch
    and literal-replacement forms.  Flatten only object unions: preserve every
    property constraint, require the fields shared by all branches, and leave
    branch selection to the existing tool implementation.  This widens schema
    parsing only; it grants no filesystem authority and malformed combinations
    still produce a typed tool error.
    """

    schema = tool.parameter_schema()
    variants = schema.get("oneOf") if isinstance(schema, dict) else None
    if variants is None:
        return schema
    if not isinstance(variants, list) or len(variants) < 2:
        raise BridgeError(f"{name} has an invalid object-union schema")
    properties: dict[str, Any] = {}
    required_sets: list[set[str]] = []
    for variant in variants:
        if (
            not isinstance(variant, dict)
            or variant.get("type") != "object"
            or not isinstance(variant.get("properties"), dict)
            or variant.get("additionalProperties") is not False
        ):
            raise BridgeError(f"{name} has an unsupported top-level union schema")
        for property_name, property_schema in variant["properties"].items():
            if property_name in properties and properties[property_name] != property_schema:
                raise BridgeError(f"{name} has conflicting union property schemas")
            properties[property_name] = property_schema
        required = variant.get("required", [])
        if not isinstance(required, list) or not all(
            isinstance(item, str) and item in variant["properties"] for item in required
        ):
            raise BridgeError(f"{name} has an invalid union requirement")
        required_sets.append(set(required))
    shared_required = set.intersection(*required_sets)
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(shared_required),
        "additionalProperties": False,
    }


def create_server() -> Server:
    worker, tools_by_name = _build_worker()
    completion = CompletionStateWriter.from_environment()
    completion.publish(worker, tool_invocations=0)
    server = Server("aeon", version="1.0.0")
    call_lock = asyncio.Lock()

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        visible = worker._active_tool_names()
        result: list[types.Tool] = []
        for name, tool in tools_by_name.items():
            if name not in visible:
                continue
            resource = tool_resource_policy(name)
            result.append(
                types.Tool(
                    name=name,
                    title=name.replace("_", " ").title(),
                    description=(
                        f"[compute-route: {resource.route.value}] "
                        + str(tool.description or "")
                    )[:12_000],
                    inputSchema=_opencode_input_schema(name, tool),
                )
            )
        return result

    @server.call_tool(validate_input=True)
    async def call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
        if name not in tools_by_name or name not in worker._active_tool_names():
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="Tool is unavailable for this request")],
                isError=True,
            )
        parameters = arguments if isinstance(arguments, dict) else {}
        async with call_lock:
            def execute() -> tuple[Any, list[str]]:
                # Tool implementations sometimes print diagnostics. Keep them
                # off stdout, which is exclusively the MCP framing channel.
                try:
                    with contextlib.redirect_stdout(sys.stderr):
                        iteration = max(1, worker.effective_iterations + 1)
                        turn = {
                            "intent": f"OpenCode requested {name}",
                            "actions": [{"tool_name": name, "parameters": parameters}],
                        }
                        receipts, _interrupted, _restart = worker._execute_protocol_actions(
                            turn,
                            iteration,
                        )
                        worker._record_protocol_tool_turn(
                            turn,
                            receipts,
                            iteration,
                        )
                    worker.effective_iterations += 1
                    images = list(worker.visual_context)
                    worker.visual_context = []
                    _save_browser_tab(worker)
                    return (receipts[0] if receipts else None), images
                finally:
                    # Memory, pending confirmation state, and exact tool receipts
                    # must survive the MCP process exiting at the end of this turn.
                    worker._persist_session_state()
                    completion.publish(
                        worker,
                        tool_invocations=worker.effective_iterations,
                    )

            try:
                receipt, image_paths = await anyio.to_thread.run_sync(execute)
            except Exception as exc:
                print(
                    f"Aeon MCP tool boundary failed: {type(exc).__name__}",
                    file=sys.stderr,
                )
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text="Aeon MCP tool boundary failed without a trusted receipt",
                        )
                    ],
                    isError=True,
                )
        if receipt is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="Tool returned no receipt")],
                isError=True,
            )
        payload = receipt.to_model_dict()
        content: list[types.TextContent | types.ImageContent] = [
            types.TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, default=str),
            )
        ]
        for image_path in image_paths[:2]:
            item = _image_content(image_path)
            if item is not None:
                content.append(item)
        return types.CallToolResult(
            content=content,
            structuredContent=payload,
            isError=not receipt.successful,
        )

    return server


async def _serve() -> None:
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
            raise_exceptions=False,
        )


def _install_termination_handlers() -> dict[int, Any]:
    """Cooperatively retire exact process-owned resources before MCP exits."""

    if threading.current_thread() is not threading.main_thread():
        return {}
    previous: dict[int, Any] = {}
    terminating = False

    def terminate(signum: int, _frame: Any) -> None:
        nonlocal terminating
        if terminating:
            raise SystemExit(128 + signum)
        terminating = True
        errors = cancel_process_resources()
        if errors:
            print(
                "Aeon MCP cooperative cleanup was not fully proven: "
                + ", ".join(errors),
                file=sys.stderr,
            )
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    return previous


def _restore_termination_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def main() -> int:
    previous = _install_termination_handlers()
    try:
        anyio.run(_serve)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except (BridgeError, OpenCodeCompletionError, ValueError, OSError) as exc:
        print(f"Aeon MCP bridge unavailable: {exc}", file=sys.stderr)
        return 1
    finally:
        _restore_termination_handlers(previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BridgeError",
    "EXPOSED_TOOLS",
    "_install_termination_handlers",
    "create_server",
    "main",
)
