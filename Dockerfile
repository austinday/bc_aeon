# 1. DEFINE BUILD ARGUMENTS
ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION_MAJOR=3
ARG PYTHON_VERSION_MINOR=10
ARG PYTORCH_CUDA_SUFFIX=cu121

# 2. START FROM THE OFFICIAL NVIDIA BASE
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Re-declare ARGs
ARG PYTHON_VERSION_MAJOR
ARG PYTHON_VERSION_MINOR
ARG PYTORCH_CUDA_SUFFIX

# 3. SET NON-INTERACTIVE MODE
ENV DEBIAN_FRONTEND=noninteractive

# 4. INSTALL SYSTEM-LEVEL DEPENDENCIES
RUN apt-get update && apt-get install -y \
    software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y \
    python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} \
    python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}-venv \
    python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}-dev \
    git curl wget vim htop unzip jq build-essential pkg-config cmake \
    libgl1-mesa-glx libglib2.0-0 libpq-dev \
    libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0 \
    nvtop \
 && rm -rf /var/lib/apt/lists/*

# 5. INSTALL GOOGLE CLOUD SDK
RUN apt-get update && apt-get install -y \
    apt-transport-https ca-certificates gnupg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
 && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
 && apt-get update && apt-get install -y google-cloud-sdk

# 6. INSTALL UV (The pip replacement)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 7. PYTHON CORE & SCIENTIFIC LIBS (Parallelized via uv)
# Note: We use --system to install into the system python environment
RUN uv pip install --system --no-cache-dir --upgrade pip setuptools wheel twine poetry black flake8 isort mypy jupyter notebook ipykernel ipywidgets cython numba pybind11 \
    numpy scipy matplotlib pandas scikit-learn tqdm pydantic beautifulsoup4 lxml polars pyarrow zarr lmdb duckdb psycopg2-binary redis pillow opencv-python-headless \
    pytest pytest-cov pytest-mock

# 8. DL & ML (Resolves wheels better to avoid build failures)
RUN uv pip install --system --no-cache-dir \
    blinker wandb deepspeed lightning[extra] \
    "transformers>=4.46.0" "accelerate>=0.26.0" qwen-vl-utils \
    datasets huggingface_hub \
    xgboost lightgbm mlflow tensorboard timm

# 9. OTHER LIBS
RUN uv pip install --system --no-cache-dir \
    biopython pysam pybedtools scikit-bio pyro-ppl pymc statsmodels sympy pysr qutip "ray[all]" "dask[complete]" dask-cuda pyspark \
    google-cloud-storage google-cloud-bigquery google-cloud-aiplatform google-cloud-pubsub gcsfs \
    seaborn plotly logomaker python-pptx WeasyPrint Jinja2 nbconvert fastapi uvicorn[standard] httpx streamlit gradio flask requests aiohttp

# 10. PYTORCH
RUN uv pip install --system --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_SUFFIX}

# 11. CLEANUP
WORKDIR /workspace
CMD ["bash"]
