FROM mcr.microsoft.com/playwright:v1.60.0-jammy

# browser_service.py validates these labels against both an immutable local
# image receipt and the exact source hash supplied by the operator build. A
# stale or differently built image can therefore never be selected by tag at
# service-launch time.
ARG AEON_BROWSER_SOURCE_SHA256
LABEL com.bc_aeon.browser.auth="required-v1" \
      com.bc_aeon.browser.source-sha256="${AEON_BROWSER_SOURCE_SHA256}"

# Install REAL Google Chrome (for channel='chrome' max-stealth browsing) plus
# Xvfb so Chrome can run HEADED (far less detectable than headless).
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y \
    xvfb openbox curl wget gnupg python3-pip ffmpeg tzdata \
 && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub \
      | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
      > /etc/apt/sources.list.d/google-chrome.list \
 && apt-get update && apt-get install -y google-chrome-stable \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is aeon/services/browser/. Only these explicit runtime sources
# are admitted by .dockerignore and copied into the image.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Patchright's patched Chromium is the fallback if real Chrome is ever absent.
RUN python3 -m patchright install chromium

# Persistent profile mount point: a host volume is mounted here at run time so
# logins/cookies survive container restarts (see start_browser.sh).
RUN mkdir -p /profiles

COPY server.py browser_util.py media_safety.py human_motion.py entrypoint.sh ./

# Capability marker checked by the launcher so an authenticated but stale image
# cannot silently run without coordinate actions/dialog policy/extraction.
LABEL com.bc_aeon.browser.api="human-v6"

# COPY . . placed server.py and entrypoint.sh directly in /app, which matches
# entrypoint.sh running `uvicorn server:app` from /app.
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
