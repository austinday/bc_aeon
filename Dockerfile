# 1. DEFINE BUILD ARGUMENTS
ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION_MAJOR=3
ARG PYTHON_VERSION_MINOR=10
ARG PYTORCH_CUDA_SUFFIX=cu121

# 2. START FROM THE OFFICIAL NVIDIA BASE
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Re-declare ARGs after the FROM statement
ARG PYTHON_VERSION_MAJOR
ARG PYTHON_VERSION_MINOR
ARG PYTORCH_CUDA_SUFFIX

# 3. SET NON-INTERACTIVE MODE
ENV DEBIAN_FRONTEND=noninteractive

# 4. INSTALL SYSTEM-LEVEL DEPENDENCIES (MODIFIED TO ADD PPA)
RUN apt-get update && apt-get install -y \
    # --- Add PPA for multi-python support ---
    software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y \
    # --- Python (now from PPA) ---
    python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} \
    python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}-venv \
    python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}-dev \
    # --- Basic Dev & Repo Tools ---
    git \
    curl \
    wget \
    vim \
    htop \
    unzip \
    jq \
    build-essential \
    pkg-config \
    cmake \
    # --- Bioinformatics Command-Line Tools ---
    ncbi-entrez-direct \
    ncbi-blast+ \
    # --- R Language & Core Plotting ---
    r-base \
    r-base-dev \
    r-cran-ggplot2 \
    r-cran-devtools \
    # --- Common Libs for Python Packages ---
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq-dev \
    # --- PDF/Report Generation Libs ---
    libpango-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    # --- GPU Monitoring ---
    nvtop \
 && rm -rf /var/lib/apt/lists/*

# 5. INSTALL GOOGLE CLOUD SDK (gcloud, gsutil)
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
 && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
 && apt-get update && apt-get install -y google-cloud-sdk

# 6. SET UP PYTHON & INSTALL PIP
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip 1 \
 && python --version \
 && pip --version

# 7. UPGRADE PIP & INSTALL CORE PYTHON BUILD TOOLS
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    setuptools wheel twine poetry \
    black flake8 isort mypy \
    jupyter notebook ipykernel ipywidgets \
    cython \
    numba \
    pybind11

# 8. INSTALL HEAVY SCIENTIFIC LIBRARIES
RUN pip install --no-cache-dir \
    numpy scipy matplotlib pandas scikit-learn tqdm \
    pydantic beautifulsoup4 lxml \
    polars \
    pyarrow \
    zarr \
    lmdb \
    duckdb psycopg2-binary redis \
    pillow opencv-python-headless

# 9. INSTALL TESTING FRAMEWORKS
RUN pip install --no-cache-dir \
    pytest pytest-cov pytest-mock

# 10. INSTALL DL, ML & MLOps
# --- FIX --- Force pip to take ownership of 'blinker' to avoid distutils error
RUN pip install --no-cache-dir --ignore-installed blinker
# --- End Fix ---
RUN pip install --no-cache-dir \
    wandb deepspeed lightning[extra] \
    transformers accelerate datasets huggingface_hub \
    xgboost lightgbm \
    mlflow tensorboard \
    timm

# 11. INSTALL BIOINFORMATICS
RUN pip install --no-cache-dir \
    biopython \
    pysam \
    pybedtools \
    scikit-bio

# 12. INSTALL PROBABILITY, STATS & SYMBOLIC MATH
RUN pip install --no-cache-dir \
    pyro-ppl pymc \
    statsmodels \
    sympy \
    pysr

# 13. INSTALL PHYSICS SIMULATION
RUN pip install --no-cache-dir \
    qutip

# 14. INSTALL DISTRIBUTED COMPUTING FRAMEWORKS
RUN pip install --no-cache-dir \
    "ray[all]" \
    "dask[complete]" \
    dask-cuda \
    pyspark

# 15. INSTALL GOOGLE CLOUD PYTHON LIBRARIES
RUN pip install --no-cache-dir \
    google-cloud-storage \
    google-cloud-bigquery \
    google-cloud-aiplatform \
    google-cloud-pubsub \
    gcsfs

# 16. INSTALL VISUALIZATION & REPORTING
RUN pip install --no-cache-dir \
    seaborn \
    plotly \
    logomaker \
    python-pptx \
    WeasyPrint \
    Jinja2 \
    nbconvert

# 17. INSTALL API & UI FRAMEWORKS
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] httpx \
    streamlit gradio \
    flask requests \
    aiohttp

# 18. INSTALL PYTORCH (THE BIG ONE)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_SUFFIX}

# 19. FINAL CLEANUP & CONFIG
WORKDIR /workspace
CMD ["bash"]

