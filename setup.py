from setuptools import setup, find_packages

setup(
    name="aeon",
    version="0.2.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "aeon": [
            "scripts/*.sh",
            "scripts/*.py",
            "services/*.yml",
            "core/prompts/*.txt",
            "core/prompts/categories/*.txt",
            "core/prompts/tools/*.txt",
            "core/data/*.json",
            "core/data/*.SHA256SUMS",
            "core/skills/*.txt",
            "core/skills/*/*.txt",
            "core/skills/**/*.txt",
            "remote/static/*",
            # The receipted browser image builder hashes and consumes this exact
            # source closure even when Aeon is installed from a wheel.
            "services/browser/.dockerignore",
            "services/browser/Dockerfile",
            "services/browser/entrypoint.sh",
            "services/browser/requirements.txt",
        ]
    },
    install_requires=[
        "openai>=1.12.0",
        "httpx>=0.27,<1",
        "psutil",
        "nvidia-ml-py3",
        "requests",
        "huggingface_hub",
        "prompt_toolkit>=3.0",   # multi-line/unlimited paste + line editing at every prompt
        "Pillow",                # PIL: tools/browser.py and tools/composite_image.py import it
                                 # at MODULE load, so without it the tool loader drops the
                                 # browser + image tools entirely.
        "imageio-ffmpeg>=0.6",   # pinned local ffmpeg binary for verified video extend/assembly
        "tiktoken",              # accurate token counting; the context-pressure thresholds are
                                 # tuned against cl100k. Degrades to a char heuristic if absent,
                                 # but it is a real (not optional) part of the runtime.
        "PyYAML",                # YAML syntax validation on write_file (small, common).
        "mcp>=1.26,<2",          # stable stdio bridge used by the OpenCode harness
    ],
    extras_require={
        # Optional richer file-type analysis. FileAnalyzer's handlers lazily import
        # these, so the harness runs without them (those file types degrade to a
        # generic summary). Deliberately NOT in install_requires: they are heavy and
        # niche. Dependency changes use the explicit operator upgrade workflow;
        # restart_aeon only reloads canonical source. Install with: pip install .[analysis]
        "analysis": ["numpy", "pandas", "h5py", "biopython", "PyMuPDF", "nbformat"],
        "remote": [
            "fastapi>=0.115",
            "uvicorn[standard]>=0.30",
            "argon2-cffi>=23.1",
            "psutil>=5.9",
        ],
    },
    # FastAPI/Uvicorn remain absent from the base install. They are opt-in through
    # the remote extra; browser-service dependencies still stay in its container.
    entry_points={
        "console_scripts": [
            "aeon = aeon.cli:main",
            "aeon-remote = aeon.remote.cli:main",
        ],
        "aday_fleet_compute.adapters": [
            "aeon-qwen38-runtime-v1 = aeon.core.fleet_adapter:create_fleet_adapter",
            "aeon-qwen38-fast-service-v1 = aeon.core.qwen_fast_service_adapter:create_fleet_adapter",
            "aeon-qwen38-speed-lab-v1 = aeon.core.qwen_speed_lab_adapter:create_fleet_adapter",
            "aeon-qwen38-dflash-adapt-v1 = aeon.core.qwen_dflash_training_adapter:create_fleet_adapter",
            "aeon-qwen38-full-gdn-quant-v1 = aeon.core.qwen_full_gdn_quant_adapter:create_fleet_adapter",
            "aeon-qwen38-flash-next-build-v1 = aeon.core.qwen_flash_next_build_adapter:create_fleet_adapter",
            "aeon-qwen38-flash-next-mtp-nvfp4-build-v1 = aeon.core.qwen_flash_next_mtp_quant_adapter:create_fleet_adapter",
            "aeon-qwen38-flash-next-qualification-v1 = aeon.core.qwen_flash_next_qualification_adapter:create_fleet_adapter",
            "aeon-qwen38-flash-next-vllm-canary-v1 = aeon.core.qwen_flash_next_vllm_canary_adapter:create_fleet_adapter",
            "aeon-qwen38-flash-next-vllm-canary-179-v1 = aeon.core.qwen_flash_next_vllm_remote_canary_adapter:create_fleet_adapter",
            "aeon-qwen38-flash-next-vllm-service-v1 = aeon.core.qwen_flash_next_vllm_service_adapter:create_fleet_adapter",
            "aeon-qwen38-flash-next-remote-service-v1 = aeon.core.qwen_flash_next_remote_service_adapter:create_fleet_adapter",
            "aeon-qwen38-flash-next-service-v1 = aeon.core.qwen_flash_next_service_adapter:create_fleet_adapter",
            "aeon-comfyui-runtime-v1 = aeon.core.comfy_fleet_adapter:create_fleet_adapter",
            "aeon-video-comfyui-runtime-v1 = aeon.core.video_comfy_fleet_adapter:create_fleet_adapter",
        ],
    },
    python_requires='>=3.10',
)
