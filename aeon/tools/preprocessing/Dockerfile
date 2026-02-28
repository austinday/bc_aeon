FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    Pillow \
    rembg[cpu] \
    onnxruntime

# Copy preprocessing script
COPY complete_printify_preprocess.py /app/

# Default command
CMD ["python", "complete_printify_preprocess.py"]