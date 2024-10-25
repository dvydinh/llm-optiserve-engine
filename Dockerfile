# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Base image: NVIDIA CUDA 12.1 runtime on Ubuntu 22.04
# Required for vLLM GPU kernels and AWQ GEMM operations.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-venv \
        git \
        curl && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Layer caching: install Python deps BEFORE copying application code.
# This ensures that code-only changes do not invalidate the pip cache layer.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source (respects .dockerignore)
COPY . .

# ---------------------------------------------------------------------------
# Model weights should be mounted via Docker volume at runtime:
#   docker run --gpus all -v /host/models:/app/models ...
# This keeps the image small (~15 GB savings per model).
# ---------------------------------------------------------------------------
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]
