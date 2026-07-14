import os
import shutil
from setuptools import setup, find_packages

# Automatically wipe stale build/cache directories before installing
for cache_dir in ['build', 'aeon.egg-info']:
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_dir)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path, ignore_errors=True)

setup(
    name="aeon",
    version="0.1.0",
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
            "core/skills/*.txt",
            "core/skills/*/*.txt",
            "core/skills/**/*.txt",
        ]
    },
    install_requires=[
        "openai>=1.12.0",
        "psutil",
        "nvidia-ml-py3",
        "requests",
        "huggingface_hub",
        "prompt_toolkit>=3.0",   # multi-line/unlimited paste + line editing at every prompt
        "Pillow",                # PIL: tools/browser.py and tools/composite_image.py import it
                                 # at MODULE load, so without it the tool loader drops the
                                 # browser + image tools entirely.
        "tiktoken",              # accurate token counting; the context-pressure thresholds are
                                 # tuned against cl100k. Degrades to a char heuristic if absent,
                                 # but it is a real (not optional) part of the runtime.
        "PyYAML",                # YAML syntax validation on write_file (small, common).
    ],
    extras_require={
        # Optional richer file-type analysis. FileAnalyzer's handlers lazily import
        # these, so the harness runs without them (those file types degrade to a
        # generic summary). Deliberately NOT in install_requires: they are heavy and
        # niche, and restart_aeon reinstalls the package on every self-modification —
        # keeping the base lean keeps restarts fast. Install with: pip install .[analysis]
        "analysis": ["numpy", "pandas", "h5py", "biopython", "PyMuPDF", "nbformat"],
    },
    # NOTE: fastapi / uvicorn / httpx / pydantic / patchright are intentionally
    # absent — they belong to the browser service and the load-balancer, which run
    # INSIDE their own Docker containers (the host never imports those modules).
    entry_points={
        "console_scripts": [
            "aeon = aeon.main:cli",
        ],
    },
    python_requires='>=3.6',
)
