"""
ComfyUI Backend - Shared infrastructure for all generative AI tools.

This module handles:
- Docker container lifecycle (start, stop, health check)
- Model profile and workflow template loading
- Workflow parameter substitution and submission to ComfyUI API
- Result polling and output file discovery

Individual tool files (e.g., generate_image.py) are thin wrappers that
call backend.run_model(profile_id, params) and return the result.

=== FOR FUTURE LLMs MODIFYING THIS CODE ===
- Do NOT put model-specific logic here. This is generic infrastructure.
- Each model gets: a profile JSON, a workflow JSON, and a tool wrapper.
- The container mounts ~/bc_aeon/aeon_models/comfyui_models as the
  diffusion_models directory inside ComfyUI. Subdirectories there map
  to model checkpoint paths referenced in workflow JSONs.
- The container is SHARED across all generative tools. It starts on
  first use and stays running. Only clear_gpu.sh kills it.
- If a new model needs additional ComfyUI custom nodes, add them to
  aeon/comfyui/Dockerfile and rebuild via setup_environment.sh.
============================================
"""

import os
import json
import time
import random
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..core.logger import get_logger

# =============================================================================
# CONFIGURATION
# =============================================================================
COMFYUI_CONTAINER_NAME = 'aeon_comfyui'
COMFYUI_PORT = 8188
COMFYUI_API_URL = f'http://localhost:{COMFYUI_PORT}'
COMFYUI_IMAGE_NAME = 'aeon_comfyui:latest'

# Host paths - models and outputs live outside the container for persistence
MODELS_BASE = Path.home() / 'bc_aeon' / 'aeon_models' / 'comfyui_models'
VAE_HOST_DIR = MODELS_BASE / 'vae'
OUTPUT_HOST_DIR = Path.home() / 'bc_aeon' / 'comfyui_output'
PROFILES_DIR = Path(__file__).parent / 'profiles'
WORKFLOWS_DIR = Path(__file__).parent / 'workflows'


class ComfyUIBackend:
    """
    Shared backend that any generative tool can use.

    Usage from a tool:
        backend = ComfyUIBackend()
        result_str = backend.run_model('hunyuan_image', {'prompt': 'a cat', 'width': 1024})
    """

    def __init__(self):
        self.logger = get_logger()
        self._profiles_cache: Dict[str, dict] = {}
        OUTPUT_HOST_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PROFILE & WORKFLOW LOADING
    # =========================================================================

    def _load_profiles(self) -> Dict[str, dict]:
        """Load all model profiles from disk. Cached after first call."""
        if self._profiles_cache:
            return self._profiles_cache
        if not PROFILES_DIR.exists():
            return {}
        for pfile in PROFILES_DIR.glob('*.json'):
            try:
                with open(pfile, 'r') as f:
                    profile = json.load(f)
                model_id = profile.get('model_id', pfile.stem)
                self._profiles_cache[model_id] = profile
            except Exception as e:
                self.logger.warning(f'Failed to load ComfyUI profile {pfile}: {e}')
        return self._profiles_cache

    def get_profile(self, model_id: str) -> Optional[dict]:
        """Get a single model profile by ID."""
        return self._load_profiles().get(model_id)

    def list_models(self) -> str:
        """Return human-readable list of available models."""
        profiles = self._load_profiles()
        if not profiles:
            return 'No ComfyUI model profiles found. Run setup_environment.sh to install models.'
        lines = ['Available generative models (ComfyUI backend):']
        for mid, p in profiles.items():
            lines.append(f'  - {mid}: {p.get("display_name", mid)} [{p.get("output_type", "?")}]')
            lines.append(f'    {p.get("description", "")}')
        return '\n'.join(lines)

    def _load_workflow(self, workflow_file: str) -> Optional[dict]:
        """Load a workflow template JSON from the workflows directory."""
        wf_path = WORKFLOWS_DIR / workflow_file
        if not wf_path.exists():
            return None
        with open(wf_path, 'r') as f:
            return json.load(f)

    # =========================================================================
    # CONTAINER LIFECYCLE
    # =========================================================================

    def _is_container_running(self) -> bool:
        try:
            out = subprocess.check_output(
                ['docker', 'ps', '-q', '-f', f'name={COMFYUI_CONTAINER_NAME}'],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            return bool(out)
        except Exception:
            return False

    def _api_healthy(self) -> bool:
        """Check if ComfyUI API is responding."""
        try:
            req = urllib.request.Request(f'{COMFYUI_API_URL}/system_stats')
            urllib.request.urlopen(req, timeout=3)
            return True
        except Exception:
            return False

    def _start_container(self) -> str:
        """Start the ComfyUI Docker container. Returns 'OK' or error string."""
        print(f'[ComfyUI] Starting container {COMFYUI_CONTAINER_NAME}...')
        print(f'[ComfyUI] Image: {COMFYUI_IMAGE_NAME}')
        print(f'[ComfyUI] Models dir: {MODELS_BASE} (exists={MODELS_BASE.exists()})')
        print(f'[ComfyUI] Output dir: {OUTPUT_HOST_DIR}')

        # Verify image exists
        try:
            subprocess.check_output(
                ['docker', 'image', 'inspect', COMFYUI_IMAGE_NAME],
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return (f'Error: Docker image {COMFYUI_IMAGE_NAME} not found. '
                    f'Run setup_environment.sh to build it.')

        # Remove any stopped/stale container
        subprocess.run(
            ['docker', 'rm', '-f', COMFYUI_CONTAINER_NAME],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )

        MODELS_BASE.mkdir(parents=True, exist_ok=True)
        VAE_HOST_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_HOST_DIR.mkdir(parents=True, exist_ok=True)

        # === VOLUME MOUNT NOTES FOR FUTURE LLMs ===
        # - The model lake is mounted into MULTIPLE ComfyUI model subdirs so that
        #   different loader node types (CheckpointLoaderSimple, UNETLoader, etc.)
        #   can all find the files regardless of which subdir they search.
        # - output: where ComfyUI writes generated files (images, video, audio)
        # ============================================
        cmd = [
            'docker', 'run', '-d',
            '--name', COMFYUI_CONTAINER_NAME,
            '--gpus', 'device=1',
            '-p', f'{COMFYUI_PORT}:8188',
            # Mount the model lake into ALL ComfyUI model directories that loaders check
            '-v', f'{MODELS_BASE}:/opt/ComfyUI/models/checkpoints',
            '-v', f'{MODELS_BASE}:/opt/ComfyUI/models/diffusion_models',
            '-v', f'{MODELS_BASE}:/opt/ComfyUI/models/unet',
            '-v', f'{MODELS_BASE}/clip:/opt/ComfyUI/models/clip',
            '-v', f'{MODELS_BASE}/text_encoders:/opt/ComfyUI/models/text_encoders',
            '-v', f'{MODELS_BASE}/llm:/opt/ComfyUI/models/llm',
            # Mount VAE directory separately (HyVideoVAELoader looks here)
            '-v', f'{VAE_HOST_DIR}:/opt/ComfyUI/models/vae',
            # Mount output directory so host can access generated files
            '-v', f'{OUTPUT_HOST_DIR}:/opt/ComfyUI/output',
            COMFYUI_IMAGE_NAME
        ]
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            return f'Error starting ComfyUI container: {e.output}'

        return self._wait_for_api()

    def _wait_for_api(self, timeout: int = 120) -> str:
        """Block until ComfyUI API is responsive. Returns 'OK' or error."""
        print(f'[ComfyUI] Waiting for API (timeout={timeout}s)...')
        start = time.time()
        while time.time() - start < timeout:
            if self._api_healthy():
                elapsed = time.time() - start
                print(f'[ComfyUI] API healthy after {elapsed:.1f}s')
                return 'OK'
            time.sleep(2)
        # On timeout, grab container logs
        try:
            logs = subprocess.check_output(
                ['docker', 'logs', '--tail', '30', COMFYUI_CONTAINER_NAME],
                stderr=subprocess.STDOUT, text=True, timeout=5
            )
        except Exception:
            logs = '(could not fetch logs)'
        return ('Error: ComfyUI API did not respond within timeout.\n'
                f'Container logs (last 30 lines):\n{logs}')

    def _get_container_gpu_usage(self) -> dict:
        """Check GPU utilization and memory of the ComfyUI container's GPU processes.

        Returns dict with 'gpu_util_pct' and 'mem_used_gb', or None on failure.
        """
        try:
            # Get the container's PIDs
            pid_out = subprocess.check_output(
                ['docker', 'top', COMFYUI_CONTAINER_NAME, '-o', 'pid'],
                stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
            container_pids = set()
            for line in pid_out.splitlines()[1:]:  # skip header
                pid = line.strip()
                if pid.isdigit():
                    container_pids.add(pid)

            if not container_pids:
                return {'gpu_util_pct': 0.0, 'mem_used_gb': 0.0}

            # Query nvidia-smi for GPU utilization and memory
            smi_out = subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                 '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()

            total_mem_mb = 0.0
            for line in smi_out.splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and parts[0] in container_pids:
                    try:
                        total_mem_mb += float(parts[1])
                    except ValueError:
                        pass

            # Get overall GPU utilization (not per-process, but good enough)
            util_out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
            # Take max utilization across all GPUs
            gpu_util = 0.0
            for line in util_out.splitlines():
                try:
                    gpu_util = max(gpu_util, float(line.strip()))
                except ValueError:
                    pass

            return {
                'gpu_util_pct': gpu_util,
                'mem_used_gb': total_mem_mb / 1024.0
            }
        except Exception as e:
            print(f'[ComfyUI] Could not check GPU usage: {e}')
            return None

    def _wait_for_container_idle(self, timeout: int = 600, poll_interval: int = 10) -> bool:
        """Wait until the running container becomes idle. Returns True if idle, False on timeout."""
        print(f'[ComfyUI] Container is busy, waiting for it to finish (timeout={timeout}s)...')
        start = time.time()
        while time.time() - start < timeout:
            usage = self._get_container_gpu_usage()
            if usage is None:
                # Can't determine usage, assume idle
                return True
            print(f'[ComfyUI]   GPU: {usage["gpu_util_pct"]:.0f}%, VRAM: {usage["mem_used_gb"]:.1f}GB')
            if usage['gpu_util_pct'] < 5.0 and usage['mem_used_gb'] < 3.0:
                print('[ComfyUI] Container is now idle.')
                return True
            time.sleep(poll_interval)
        print('[ComfyUI] Timed out waiting for container to become idle.')
        return False

    def _ensure_running(self) -> Optional[str]:
        """Ensure container is running with correct mounts. Restarts idle containers.

        Logic:
        - If no container running: start fresh.
        - If container running + API healthy:
            - Check GPU usage. If idle (<5% util, <3GB VRAM): kill & restart
              (picks up any mount/config changes).
            - If busy (>20% util or >20GB VRAM): wait for it to finish,
              then reuse without restart.
            - If in between: treat as idle and restart.
        """
        if not self._is_container_running():
            print('[ComfyUI] No container running, starting fresh.')
            result = self._start_container()
            return None if result == 'OK' else result

        if not self._api_healthy():
            print('[ComfyUI] Container exists but API unhealthy, restarting.')
            subprocess.run(
                ['docker', 'rm', '-f', COMFYUI_CONTAINER_NAME],
                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
            )
            result = self._start_container()
            return None if result == 'OK' else result

        # Container is running and API is healthy. Check if it's busy.
        usage = self._get_container_gpu_usage()
        if usage is None:
            # Can't determine, restart to be safe
            print('[ComfyUI] Cannot determine GPU usage, restarting container.')
            subprocess.run(
                ['docker', 'rm', '-f', COMFYUI_CONTAINER_NAME],
                stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
            )
            result = self._start_container()
            return None if result == 'OK' else result

        print(f'[ComfyUI] Existing container GPU: {usage["gpu_util_pct"]:.0f}%, VRAM: {usage["mem_used_gb"]:.1f}GB')

        is_busy = usage['gpu_util_pct'] > 20.0 or usage['mem_used_gb'] > 20.0
        is_idle = usage['gpu_util_pct'] < 5.0 and usage['mem_used_gb'] < 3.0

        if is_busy:
            # Someone else is using it, wait our turn
            became_idle = self._wait_for_container_idle()
            if not became_idle:
                return ('Error: ComfyUI container is busy with another generation '
                        'and did not become idle within timeout.')
            # After it finishes, reuse it (don't restart mid-session)
            return None

        # Idle or indeterminate: restart to pick up latest mounts/config
        print('[ComfyUI] Container is idle, restarting to pick up latest config...')
        subprocess.run(
            ['docker', 'rm', '-f', COMFYUI_CONTAINER_NAME],
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )
        result = self._start_container()
        return None if result == 'OK' else result

    # =========================================================================
    # WORKFLOW EXECUTION
    # =========================================================================

    def _substitute_params(self, workflow: dict, params: Dict[str, Any]) -> dict:
        """
        Replace {{PLACEHOLDER}} tokens in workflow JSON with actual values.

        Numeric values replace the QUOTED placeholder (so ComfyUI sees a number,
        not a string). String values do plain text replacement.
        """
        wf_str = json.dumps(workflow)
        for key, value in params.items():
            placeholder = '{{' + key.upper() + '}}'
            if isinstance(value, (int, float)):
                # Replace "{{PLACEHOLDER}}" (with quotes) -> raw number
                wf_str = wf_str.replace('"' + placeholder + '"', str(value))
                # Also handle if it appears unquoted
                wf_str = wf_str.replace(placeholder, str(value))
            else:
                wf_str = wf_str.replace(placeholder, str(value))
        return json.loads(wf_str)

    def discover_nodes(self, keyword: str = '') -> dict:
        """Query ComfyUI /object_info for available node types.

        Args:
            keyword: If provided, only return nodes whose class_type contains
                     this substring (case-insensitive).

        Returns:
            Dict of node_name -> {input_schema, output_types, ...} from ComfyUI.
        """
        try:
            req = urllib.request.Request(f'{COMFYUI_API_URL}/object_info')
            resp = urllib.request.urlopen(req, timeout=15)
            all_nodes = json.loads(resp.read().decode('utf-8'))
            if keyword:
                kw = keyword.lower()
                return {k: v for k, v in all_nodes.items() if kw in k.lower()}
            return all_nodes
        except Exception as e:
            return {'_error': str(e)}

    def discover_nodes_summary(self, keywords: list = None) -> str:
        """Return a human-readable summary of available nodes matching keywords."""
        if keywords is None:
            keywords = ['hunyuan', 'HyVideo', 'gguf', 'GGUF', 'Unet', 'UNET',
                        'Diffusion', 'checkpoint', 'Checkpoint']
        lines = []
        for kw in keywords:
            nodes = self.discover_nodes(kw)
            if '_error' in nodes:
                lines.append(f'  [{kw}] Error: {nodes["_error"]}')
                continue
            if not nodes:
                continue
            for name, info in nodes.items():
                req_inputs = info.get('input', {}).get('required', {})
                opt_inputs = info.get('input', {}).get('optional', {})
                # Show input names and their accepted values (if enum/list)
                input_details = []
                for iname, ispec in req_inputs.items():
                    if isinstance(ispec, list) and len(ispec) > 0 and isinstance(ispec[0], list):
                        # Enum values - show first few
                        vals = ispec[0]
                        if len(vals) > 5:
                            val_str = str(vals[:5]) + f'... ({len(vals)} total)'
                        else:
                            val_str = str(vals)
                        input_details.append(f'{iname}={val_str}')
                    else:
                        input_details.append(iname)
                for iname in opt_inputs:
                    input_details.append(f'{iname}(opt)')
                lines.append(f'  {name}: [{", ".join(input_details)}]')
        return '\n'.join(lines) if lines else '  (no matching nodes found)'

    def _submit_prompt(self, workflow: dict) -> str:
        """POST workflow to ComfyUI /prompt endpoint. Returns prompt_id."""
        payload = json.dumps({'prompt': workflow}).encode('utf-8')
        print(f'[ComfyUI] Submitting prompt ({len(payload)} bytes)...')
        req = urllib.request.Request(
            f'{COMFYUI_API_URL}/prompt',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            print(f'[ComfyUI] HTTP {e.code} from /prompt: {body[:1000]}')
            raise RuntimeError(f'HTTP {e.code} from ComfyUI /prompt: {body[:500]}') from e
        result = json.loads(resp.read().decode('utf-8'))
        prompt_id = result.get('prompt_id', '')
        if not prompt_id:
            print(f'[ComfyUI] Empty prompt_id! Full response: {json.dumps(result)[:1000]}')
            raise RuntimeError(f'Empty prompt_id. ComfyUI response: {json.dumps(result)[:500]}')
        print(f'[ComfyUI] Got prompt_id: {prompt_id}')
        return prompt_id

    def _poll_completion(self, prompt_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Poll /history/{prompt_id} until complete, error, or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                req = urllib.request.Request(f'{COMFYUI_API_URL}/history/{prompt_id}')
                resp = urllib.request.urlopen(req, timeout=5)
                history = json.loads(resp.read().decode('utf-8'))
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get('status', {})
                    status_str = status.get('status_str', '')
                    if status.get('completed', False) or status_str == 'success':
                        return {'status': 'success', 'outputs': entry.get('outputs', {})}
                    if status_str == 'error':
                        msgs = status.get('messages', [])
                        return {'status': 'error', 'error': str(msgs)}
            except Exception:
                pass
            time.sleep(3)
        return {'status': 'timeout', 'error': f'Generation timed out after {timeout}s'}

    def _find_output_files(self, outputs: dict) -> List[str]:
        """
        Extract output file host paths from ComfyUI's outputs dict.

        ComfyUI returns output info keyed by node_id. Each node may have
        'images', 'audio', or 'video' lists with filename/subfolder entries.
        """
        files = []
        for node_id, node_output in outputs.items():
            # === FOR FUTURE LLMs ===
            # If adding a model with a new output type (e.g., 3D mesh, text),
            # check what key ComfyUI uses for that output and add handling here.
            # ========================
            for media_key in ('images', 'audio', 'video', 'gifs'):
                if media_key in node_output:
                    for item in node_output[media_key]:
                        fname = item.get('filename', '')
                        subfolder = item.get('subfolder', '')
                        if fname:
                            host_path = OUTPUT_HOST_DIR / subfolder / fname if subfolder else OUTPUT_HOST_DIR / fname
                            files.append(str(host_path))
        return files

    # =========================================================================
    # PUBLIC API - Called by tool wrappers
    # =========================================================================

    def _get_container_logs(self, tail: int = 80) -> str:
        """Fetch recent container logs for debugging."""
        try:
            out = subprocess.check_output(
                ['docker', 'logs', '--tail', str(tail), COMFYUI_CONTAINER_NAME],
                stderr=subprocess.STDOUT, text=True, timeout=5
            )
            return out
        except Exception as e:
            return f'(could not fetch container logs: {e})'

    def run_model(self, model_id: str, params: Dict[str, Any],
                  generation_timeout: int = 600) -> str:
        """
        Run a generative model end-to-end.

        This is the ONE method that tool wrappers call. It:
        1. Loads the profile and workflow for model_id
        2. Ensures the ComfyUI container is running
        3. Substitutes params into the workflow template
        4. Submits to ComfyUI API and polls for completion
        5. Returns a human-readable result string with output paths

        Args:
            model_id: Must match a profile JSON in profiles/ directory.
            params: Dict of parameter values. Keys should match the
                    {{PLACEHOLDER}} tokens in the workflow template.
            generation_timeout: Max seconds to wait for generation.

        Returns:
            Human-readable result string (success with paths, or error details).
        """
        debug_lines = []  # Collect verbose debug info throughout the run
        debug_lines.append(f'[DEBUG] run_model called: model_id={model_id}')
        debug_lines.append(f'[DEBUG] params: {json.dumps(params, default=str)}')
        debug_lines.append(f'[DEBUG] MODELS_BASE: {MODELS_BASE} (exists={MODELS_BASE.exists()})')
        debug_lines.append(f'[DEBUG] PROFILES_DIR: {PROFILES_DIR} (exists={PROFILES_DIR.exists()})')
        debug_lines.append(f'[DEBUG] WORKFLOWS_DIR: {WORKFLOWS_DIR} (exists={WORKFLOWS_DIR.exists()})')

        # Load profile
        profile = self.get_profile(model_id)
        if not profile:
            debug_lines.append(f'[DEBUG] Profile not found. Available: {list(self._load_profiles().keys())}')
            return ('\n'.join(debug_lines) + '\n'
                    f'Error: Model profile "{model_id}" not found. '
                    f'Available: {list(self._load_profiles().keys())}')
        debug_lines.append(f'[DEBUG] Profile loaded: {profile.get("display_name", model_id)}')

        # Handle seed randomization
        if params.get('seed', -1) == -1:
            params['seed'] = random.randint(0, 2**32 - 1)
        debug_lines.append(f'[DEBUG] Final seed: {params["seed"]}')

        # Ensure container is running
        debug_lines.append('[DEBUG] Ensuring ComfyUI container is running...')
        err = self._ensure_running()
        if err:
            debug_lines.append(f'[DEBUG] Container startup FAILED: {err}')
            return '\n'.join(debug_lines) + '\n' + err
        debug_lines.append('[DEBUG] Container is running and API is healthy.')

        # List model files visible inside the container for debugging
        try:
            ls_out = subprocess.check_output(
                ['docker', 'exec', COMFYUI_CONTAINER_NAME, 'find',
                 '/opt/ComfyUI/models/checkpoints', '/opt/ComfyUI/models/diffusion_models',
                 '-maxdepth', '3', '-type', 'f'],
                stderr=subprocess.STDOUT, text=True, timeout=10
            )
            debug_lines.append(f'[DEBUG] Model files visible in container:\n{ls_out.strip()}')
        except Exception as e:
            debug_lines.append(f'[DEBUG] Could not list model files in container: {e}')

        # Discover available ComfyUI nodes for debugging
        try:
            node_summary = self.discover_nodes_summary()
            debug_lines.append(f'[DEBUG] Available relevant ComfyUI nodes:\n{node_summary}')
        except Exception as e:
            debug_lines.append(f'[DEBUG] Could not discover nodes: {e}')

        # Load workflow template
        workflow_file = profile.get('workflow_file')
        if not workflow_file:
            debug_lines.append('[DEBUG] No workflow_file key in profile.')
            return '\n'.join(debug_lines) + '\n' + f'Error: No workflow_file in profile "{model_id}".'

        workflow = self._load_workflow(workflow_file)
        if workflow is None:
            debug_lines.append(f'[DEBUG] Workflow file not found: {WORKFLOWS_DIR / workflow_file}')
            return '\n'.join(debug_lines) + '\n' + f'Error: Workflow "{workflow_file}" not found in {WORKFLOWS_DIR}.'
        debug_lines.append(f'[DEBUG] Workflow loaded: {workflow_file} ({len(workflow)} nodes)')

        # Strip metadata keys (like _comment) that aren't ComfyUI nodes
        workflow = {k: v for k, v in workflow.items() if not k.startswith('_')}

        # Substitute parameters into workflow
        workflow = self._substitute_params(workflow, params)
        debug_lines.append(f'[DEBUG] Substituted workflow (submitted to ComfyUI):\n{json.dumps(workflow, indent=2)}')

        # Submit to ComfyUI
        try:
            prompt_id = self._submit_prompt(workflow)
        except Exception as e:
            debug_lines.append(f'[DEBUG] Submit FAILED with exception: {type(e).__name__}: {e}')
            debug_lines.append(f'[DEBUG] Container logs (last 40 lines):\n{self._get_container_logs(40)}')
            return ('\n'.join(debug_lines) + '\n'
                    f'Error submitting to ComfyUI: {e}\n'
                    f'The workflow likely needs adjustment for this model.')

        debug_lines.append(f'[DEBUG] Prompt submitted successfully: prompt_id={prompt_id}')
        self.logger.info(f'ComfyUI prompt submitted: {prompt_id} (model={model_id})')

        # Poll for result
        result = self._poll_completion(prompt_id, timeout=generation_timeout)
        debug_lines.append(f'[DEBUG] Poll result status: {result.get("status")}')

        if result['status'] == 'success':
            output_files = self._find_output_files(result.get('outputs', {}))
            debug_lines.append(f'[DEBUG] Raw outputs dict: {json.dumps(result.get("outputs", {}), default=str)}')
            debug_lines.append(f'[DEBUG] Discovered output files: {output_files}')
            if output_files:
                files_str = '\n'.join(output_files)
                return ('\n'.join(debug_lines) + '\n'
                        f'Generation complete ({profile.get("display_name", model_id)}).\n'
                        f'Output files:\n{files_str}')
            return ('\n'.join(debug_lines) + '\n'
                    f'Generation completed but no output files found. '
                    f'Check {OUTPUT_HOST_DIR} or: docker logs {COMFYUI_CONTAINER_NAME}')
        elif result['status'] == 'error':
            debug_lines.append(f'[DEBUG] Generation error detail: {result.get("error", "unknown")}')
            debug_lines.append(f'[DEBUG] Container logs (last 60 lines):\n{self._get_container_logs(60)}')
            return ('\n'.join(debug_lines) + '\n'
                    f'Generation failed: {result.get("error", "unknown")}\n'
                    f'The workflow may need adjustment.')
        else:
            debug_lines.append(f'[DEBUG] Timed out. Full result: {result}')
            debug_lines.append(f'[DEBUG] Container logs (last 60 lines):\n{self._get_container_logs(60)}')
            return ('\n'.join(debug_lines) + '\n'
                    f'Generation timed out after {generation_timeout}s.\n'
                    f'The model may need more time/resources.')
