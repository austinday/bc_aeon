FROM python:3.11-slim

# Install system dependencies for Camoufox/Playwright
RUN apt-get update && apt-get install -y \
    xvfb \
    libgbm-dev \
    libasound2 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgtk-3-0 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic camoufox requests

ENV HOME=/tmp
WORKDIR /app

# We mount the project root to /app at runtime, so we don't COPY files here.
# Run the entrypoint script to start Xvfb and the server
CMD ["/bin/bash", "/app/aeon/services/browser/entrypoint.sh"]