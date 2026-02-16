# ComfyUI Backend for Aeon Agent

## Architecture

This directory contains the shared ComfyUI infrastructure that powers all
generative AI tools (image, video, audio, etc.) in the Aeon agent.

The agent never interacts with ComfyUI directly. Instead:

```
Agent sees:    generate_image(prompt='a cat')    # simple, modality-specific tool
                       |
                       v
Tool layer:    GenerateImageTool                 # thin wrapper in aeon/tools/
                       |
                       v
Backend:       ComfyUIBackend                    # this directory - shared infra
                       |
                       v
Runtime:       Docker container (aeon_comfyui)   # GPU, models, ComfyUI server
```

## Directory Structure

- `backend.py`       - Shared ComfyUI container lifecycle, API client, workflow runner
- `Dockerfile`       - Docker image with ComfyUI + custom nodes pre-installed
- `profiles/`        - JSON configs for each supported model (paths, defaults, metadata)
- `workflows/`       - ComfyUI API-format workflow templates with {{PLACEHOLDER}} tokens

## How to Add a New Generative Model

Follow these steps to add support for a new model (e.g., a video generator):

### 1. Download the model in setup_environment.sh

Add a download block in the ComfyUI section of setup_environment.sh:

```bash
log_step "Downloading ComfyUI model: YourModel..."
YOUR_DIR="$COMFYUI_MODELS_DIR/your_model_subdir"
if [ -d "$YOUR_DIR" ] && [ "$(ls -A $YOUR_DIR 2>/dev/null)" ]; then
    echo "  (Already downloaded - Skipping)"
else
    mkdir -p "$YOUR_DIR"
    docker run --rm $TTY_FLAG \
        -v "$COMFYUI_MODELS_DIR:/models" \
        -e HF_HOME=/tmp/cache \
        ${HF_TOKEN_VAL:+-e HF_TOKEN="$HF_TOKEN_VAL"} \
        aeon_base:py3.10-cuda12.1 \
        bash -c "python3 -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='org/model', local_dir='/models/your_model_subdir', local_dir_use_symlinks=False)\""
fi
```

### 2. Create a model profile (profiles/your_model.json)

Define the model metadata, parameter schema, and which workflow to use:

```json
{
    "model_id": "your_model",
    "display_name": "Your Model Display Name",
    "description": "What this model does",
    "huggingface_repo": "org/model-name",
    "output_type": "video",
    "workflow_file": "your_model_api.json",
    "parameters": {
        "prompt": { "type": "string", "required": true }
    }
}
```

### 3. Create a workflow template (workflows/your_model_api.json)

The easiest way: run ComfyUI interactively, build a working workflow in the
browser, then click 'Save (API Format)'. Replace dynamic values with
{{PLACEHOLDER}} tokens that match your profile parameter names (uppercased).

To run ComfyUI interactively for workflow development:
```bash
docker run -it --gpus all -p 8188:8188 \
    -v ~/bc_aeon/aeon_models/comfyui_models:/opt/ComfyUI/models/diffusion_models \
    aeon_comfyui:latest
# Then open http://localhost:8188 in your browser
```

### 4. Create a tool wrapper (aeon/tools/generate_yourmodality.py)

This is the thin file the agent sees. Copy generate_image.py as a template:

```python
from .base import BaseTool
from ..comfyui.backend import ComfyUIBackend
from ..core.prompts import TOOL_DESC_GENERATE_VIDEO  # add to prompts/__init__.py

class GenerateVideoTool(BaseTool):
    def __init__(self):
        super().__init__(name='generate_video', description=TOOL_DESC_GENERATE_VIDEO)
        self.backend = ComfyUIBackend()

    def execute(self, prompt: str, duration: int = 4, **kwargs) -> str:
        params = {'prompt': prompt, 'duration': duration}
        return self.backend.run_model('your_model', params)
```

### 5. Add the tool description (core/prompts/tool_desc_generate_video.txt)

Write a description that tells the agent WHAT the tool does (not HOW).
Don't mention ComfyUI. Example: 'Generates a video from a text prompt.'

### 6. Register the description in core/prompts/__init__.py

Add: TOOL_DESC_GENERATE_VIDEO = _load('tool_desc_generate_video.txt')

### 7. (If needed) Install custom nodes in the Dockerfile

If the model needs ComfyUI custom nodes not already installed, add them
to the Dockerfile and rebuild with setup_environment.sh.

That's it. The tool loader discovers the new tool automatically.
