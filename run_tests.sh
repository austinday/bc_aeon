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
python3 -m aeon.tests.test_core
python3 -m aeon.tests.test_portable_cli
python3 -m aeon.tests.test_fleet_backend
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
python3 -m aeon.tests.test_external_expert

echo "All checks passed."
