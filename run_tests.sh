#!/bin/bash
# run_tests.sh - Fast, model-free verification gate for the Aeon harness.
# Runs the smoke test (imports + tool discovery + syntax) and the core unit
# tests (JSON/block parsing, truncation, token estimation). No GPU or model
# server required. Exit 0 = safe to restart_aeon.
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "== Smoke test =="
python3 -m aeon.smoke_test && echo "[OK] smoke test"

echo "== Unit tests =="
python3 -m aeon.tests.test_core

echo "All checks passed."
