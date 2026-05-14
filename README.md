# Aeon Agent

Aeon is a highly capable, autonomous LLM agent designed for complex planning and implementation tasks.

## Getting Started

### Installation
Follow the instructions in `setup_environment.sh` to prepare the environment.

### Running the Agent
It is recommended to use the provided launcher script to run the agent. This script monitors the process and automatically prints the end of the logs if the agent crashes, making troubleshooting significantly easier.

```bash
chmod +x run_aeon.sh
./run_aeon.sh --model <model_name>
```

Alternatively, you can run the agent directly via python:
```bash
python -m aeon.main --model <model_name>
```

## Project Structure
- `aeon/`: Core agent logic, worker loop, and LLM integration.
- `aeon/tools/`: Tool definitions and analyzers.
- `aeon/core/prompts/`: System prompts and directives.
- `scripts/`: Utility scripts for environment setup and service management.
- `aeon_output/`: Directory for sub-agent outputs and logs.

## Troubleshooting
If the agent crashes, check `aeon.log` in the root directory for detailed tracebacks. When using `run_aeon.sh`, the most relevant logs are printed to the console immediately after a crash.