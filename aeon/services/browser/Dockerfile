FROM mcr.microsoft.com/playwright:v1.60.0-jammy

# Install REAL Google Chrome (for channel='chrome' max-stealth browsing) plus
# Xvfb so Chrome can run HEADED (far less detectable than headless).
RUN apt-get update && apt-get install -y \
    xvfb curl wget gnupg python3-pip \
 && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub \
      | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
      > /etc/apt/sources.list.d/google-chrome.list \
 && apt-get update && apt-get install -y google-chrome-stable \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is aeon/services/browser/ (set by setup_environment.sh), so the
# service files are at the root of the context — reference them by bare name.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Patchright's patched Chromium is the fallback if real Chrome is ever absent.
RUN python3 -m patchright install chromium

# Persistent profile mount point: a host volume is mounted here at run time so
# logins/cookies survive container restarts (see start_browser.sh).
RUN mkdir -p /profiles

COPY . .

# COPY . . placed server.py and entrypoint.sh directly in /app, which matches
# entrypoint.sh running `uvicorn server:app` from /app.
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
