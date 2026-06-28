FROM mcr.microsoft.com/playwright:v1.60.0-jammy

RUN apt-get update && apt-get install -y \
    xvfb \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is aeon/services/browser/ (set by setup_environment.sh), so the
# service files are at the root of the context — reference them by bare name.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Ensure the pip-pinned Playwright's Chromium build is present (the base image
# ships browsers, but this guarantees a version match with playwright==1.60.0).
RUN python3 -m playwright install chromium

# Persistent profile mount point: a host volume is mounted here at run time so
# logins/cookies survive container restarts (see start_browser.sh).
RUN mkdir -p /profiles

COPY . .

# COPY . . placed server.py and entrypoint.sh directly in /app, which matches
# entrypoint.sh running `uvicorn server:app` from /app.
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
