#!/bin/bash
# run_tests.sh - Fast, model-free verification gate for the Aeon harness.
# Runs the smoke test (imports + tool discovery + syntax) and the core unit
# tests (JSON/block parsing, truncation, token estimation). No GPU or model
# server required. Exit 0 = safe to restart_aeon.
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Keep test-fixture warnings (synthetic JSON garbage etc.) out of the real aeon.log.
export AEON_NO_FILE_LOG=1

echo "== Smoke test =="
python3 -m aeon.smoke_test && echo "[OK] smoke test"

echo "== Unit tests =="
python3 -m aeon.tests.test_agent_protocol
python3 -m aeon.tests.test_worker_protocol
python3 -m aeon.tests.test_harness_safety
python3 -m aeon.tests.test_core
python3 -m aeon.tests.test_restart_lifecycle
python3 -m aeon.tests.test_portable_cli
python3 -m aeon.tests.test_fleet_backend
python3 -m aeon.tests.test_background_jobs
python3 -m aeon.tests.test_chat_transcript
python3 -m aeon.tests.test_qwen_runtime
python3 -m aeon.tests.test_qwen_fleet_runtime
python3 -m aeon.tests.test_presence
python3 -m aeon.tests.test_remote
python3 -m aeon.tests.test_remote_providers
python3 -m aeon.tests.test_instruction_profiles
python3 -m aeon.tests.test_workspace_instructions
python3 -m aeon.tests.test_runtime_instructions
python3 -m aeon.tests.test_agent_settings
python3 -m aeon.tests.test_agent_preferences
python3 -m aeon.tests.test_project_manager
python3 -m aeon.tests.test_sub_agent_capability_boundary
python3 -m aeon.tests.test_sub_agent_termination
python3 -m aeon.tests.test_verify_modification_fleet
python3 -m aeon.tests.test_durable_agent_guard
python3 -m aeon.tests.test_start_agent_instance
python3 -m aeon.tests.test_external_expert
python3 -m aeon.tests.test_tool_resources
python3 -m aeon.tests.test_command_fleet_guard
python3 -m aeon.tests.test_local_cpu_bounds
python3 -m aeon.tests.test_local_http_transport
python3 -m aeon.tests.test_host_service_tools
python3 -m aeon.tests.test_browser_service
python3 -m aeon.tests.test_browser_media_safety
python3 -m aeon.tests.test_searxng_service
python3 -m aeon.tests.test_comfy_tool_fleet
python3 -m aeon.tests.test_generate_video
python3 -m aeon.tests.test_video_comfy_fleet_adapter
python3 -m aeon.tests.test_qwen_fast_service
python3 -m aeon.tests.test_qwen_fleet_multi_runtime
python3 -m aeon.tests.test_qwen_speed_lab
python3 -m aeon.tests.test_qwen_warmup_diagnostics

echo "All checks passed."
