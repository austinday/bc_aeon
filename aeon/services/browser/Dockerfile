FROM mcr.microsoft.com/playwright:v1.49.0-jammy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    xvfb \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app/browser_service

# Copy requirements and install
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the service code
COPY . .

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]