FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install AirLLM + dependencies needed for architecture detection helpers.
RUN pip install --upgrade pip && \
    pip install airllm transformers huggingface_hub pytest

# Copy this extension repo in for tests/examples.
COPY . /app

ENV PYTHONPATH=/app

CMD ["bash"]
