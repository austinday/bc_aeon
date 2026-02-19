# Aeon Agent

Autonomous coding and research agent powered by local and cloud LLMs.

## Architecture

- **Core** (`aeon/core/`): Worker loop, LLM client, prompt management, system info
- **Tools** (`aeon/tools/`): Dynamically loaded tool modules (file I/O, commands, search, communication)
- **ComfyUI** (`aeon/comfyui/`): Shared backend infrastructure for future generative AI tools (image, video, audio)
- **LlamaCpp** (`aeon/llamacpp/`): Dockerfile for llama.cpp server (serves large GGUF models with GPU+RAM hybrid)
- **Scripts** (`aeon/scripts/`): Container lifecycle management (brain, ComfyUI, llama.cpp, vLLM)

## Setup

```bash
bash setup_environment.sh   # Downloads models, builds Docker images (idempotent)
pip install -e .            # Install aeon package
```

## Usage

```bash
aeon                        # Interactive mode (select models from menu)
aeon --strong gemini-3-pro-preview --weak gemini-flash-latest --start "build a web scraper"
aeon --debug                # Enable LLM call logging to ~/aeon_debug_*.log
```

## Model Support

- **Local (Ollama)**: Any model pulled into the Ollama brain node (GPU 0, port 8000)
- **Local (llama.cpp)**: Qwen3.5-397B-A17B-MXFP4 GGUF (GPU 0 + RAM offload, port 8001)
- **Cloud**: Grok, Gemini (API keys in `~/grok_api_key.txt`, `~/gemini_api_key.txt`)

Strong model (planner/reasoner) and weak model (summarizer/utility) are selected independently.
