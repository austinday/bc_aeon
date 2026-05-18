#!/bin/bash
# Robust wrapper to run the sub-agent validation script
# Ensures PYTHONPATH is set and arguments are quoted correctly

# Set the project root
PROJECT_ROOT="/home/aday/bc_aeon"
export PYTHONPATH="$PROJECT_ROOT"

echo "--- Starting Sub-Agent Ecosystem Validation ---"

# Define arguments as variables to ensure clean quoting
AGENT_ID="val_agent"
OBJECTIVE="Verify startup and telemetry"
MODEL_CONFIG='{"model": "test-model", "provider": "local", "base_url": "http://localhost:8000/v1", "context_limit": 128000}'
WORKSPACE="test_ecosystem_val/workspace"
OUTPUT_DIR="test_ecosystem_val/agent_1"

echo "Executing: python3 aeon/scripts/sub_agent_wrapper.py --agent_id $AGENT_ID --objective \"$OBJECTIVE\" --model_config \"$MODEL_CONFIG\" --workspace \"$WORKSPACE\" --output_dir \"$OUTPUT_DIR\" --max_iterations 1 --debug"

# Run the wrapper
python3 "$PROJECT_ROOT/aeon/scripts/sub_agent_wrapper.py" \
    --agent_id "$AGENT_ID" \
    --objective "$OBJECTIVE" \
    --model_config "$MODEL_CONFIG" \
    --workspace "$WORKSPACE" \
    --output_dir "$OUTPUT_DIR" \
    --max_iterations 1 \
    --debug

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Sub-agent execution finished successfully."
else
    echo "Sub-agent execution failed with exit code $EXIT_CODE."
fi

echo "Checking for telemetry file..."
TELEMETRY_FILE="$PROJECT_ROOT/$OUTPUT_DIR/telemetry.json"
if [ -f "$TELEMETRY_FILE" ]; then
    echo "Telemetry file FOUND."
    cat "$TELEMETRY_FILE"
else
    echo "Telemetry file NOT found."
fi