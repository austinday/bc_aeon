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
            "core/skills/*.txt",
            "core/skills/*/*.txt",
            "core/skills/**/*.txt",
            "remote/static/*",
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
    },
    python_requires='>=3.10',
)
