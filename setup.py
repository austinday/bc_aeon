from setuptools import setup, find_packages

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
            "core/prompts/*.txt"
        ]
    },
    install_requires=[
        "openai>=1.12.0",
        "tavily-python",
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
