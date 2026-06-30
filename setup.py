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
        "huggingface_hub"
    ],
    entry_points={
        "console_scripts": [
            "aeon = aeon.main:cli",
        ],
    },
    python_requires='>=3.6',
)
