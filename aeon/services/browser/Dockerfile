FROM mcr.microsoft.com/playwright:v1.60.0-jammy

RUN apt-get update && apt-get install -y \
    xvfb \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY aeon/services/browser/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x aeon/services/browser/entrypoint.sh

# Use the absolute path to the entrypoint script relative to the project root
ENTRYPOINT ["/app/aeon/services/browser/entrypoint.sh"]