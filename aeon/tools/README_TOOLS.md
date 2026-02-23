# Aeon Tools Guide

This directory contains the tools available to the Aeon agent. If you are adding or modifying a tool, adhere strictly to the following conventions:

## 1. Multi-Agent Concurrency & Registries
If your tool starts a heavy background process (like a Docker container for image generation, a specialized local LLM, or a web server), **you must implement a PID-based registry lock**.
Because multiple agents might call the tool at the exact same time:
- Use a lockfile and JSON registry (e.g., `/tmp/aeon_yourtool_registry.json`) to track the PIDs of agents currently using the tool.
- When the tool starts, register the agent's PID.
- When the tool finishes (in a `finally` block), unregister the PID.
- **Only tear down the background process if the active PID count drops to 0.** Do not forcefully kill containers if other agents are actively using them.

## 2. Tool Lifecycle
- Inherit from `BaseTool` (found in `base.py`).
- Implement the `execute` method.
- Return clean, formatted string outputs or JSON strings. Do not return raw objects.
- Wrap executions in `try/except` and use `self.format_error_message()` for graceful LLM-friendly error handling.

## 3. Tool Descriptions
Do not hardcode your tool's LLM description string in the python file. Define it in `aeon/core/prompts/` (e.g., `tool_desc_yourtool.txt`) and import it.