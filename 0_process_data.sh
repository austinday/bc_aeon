#!/bin/bash
docker build -t sentiment_processor -f Dockerfile.process .
docker run --rm -v $(pwd):/app -u $(id -u):$(id -g) sentiment_processor
