from setuptools import setup, find_packages

setup(
    name="aeon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.12.0",
        "tavily-python",
        "psutil",
        "nvidia-ml-py3"
    ],
    entry_points={
        "console_scripts": [
            "aeon = aeon.main:cli",
        ],
    },
    python_requires='>=3.6',
)
