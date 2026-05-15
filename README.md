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

## Gemma Load Balancer Self-Healing
`aeon/scripts/gemma_lb.py` implements a self-healing load balancer for Gemma-4 instances:
- Periodic health checks detect crashes or OOM events on either node.
- Automatic restart via start scripts when a node becomes unhealthy.
- Active request tracking with lock-protected restarts to prevent race conditions.
- Idle timeout shutdown (5 min) to release resources when no agents are using the models.
- Validation test (`scripts/debug/test_gemma_lb_healing.py`) confirms restart-on-crash and idle-shutdown behaviors both succeed.