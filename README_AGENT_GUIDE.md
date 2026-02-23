# Aeon Agent Codebase Guide

Welcome, future AI agent. This document outlines the core conventions of the `bc_aeon` codebase. 

## 1. Environment Setup & Downloads (CRITICAL)
**Never download models, install packages, or build Docker images at runtime.**
All environment setup, dependency resolution, and heavy downloading MUST happen inside `setup_environment.sh`. 
If you add a new tool that requires a large model (like a GGUF file) or a new system dependency, you must modify `setup_environment.sh` to download/build it ahead of time. Runtime execution should assume the environment is fully prepared.

## 2. Multi-Agent Concurrency
The Aeon system is designed to allow multiple agents to run simultaneously on the same machine. This means resources like GPU memory and background Docker containers (e.g., `aeon_brain_node`, `aeon_comfyui`) are shared.
When making architectural changes, you must ensure your code supports this concurrency. Do not arbitrarily kill containers or flush memory without checking if other agents are currently using them.

## 3. Project Structure
- `aeon/core/`: Contains the LLM clients, the worker loop, and all prompt templates.
- `aeon/tools/`: Contains the dynamic tools the agent can invoke.
- `aeon/scripts/`: Contains the Bash and Python scripts used to start background services.
- `aeon_models/`: The unified model lake (ignored by git, populated by setup scripts).