FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y xvfb xauth && rm -rf /var/lib/apt/lists/*
RUN pip install fastapi uvicorn camoufox
RUN camoufox fetch

COPY server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]