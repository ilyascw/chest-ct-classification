# CUDA version is aligned with the official PyTorch cu128 wheels below.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    python3 \
    python3-pip \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

COPY pyproject.toml README.md requirements.txt /app/
COPY src/ /app/src/

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY models/ /app/models/

RUN mkdir -p /tmp/ct_api /opt/hf_cache && chmod 777 /tmp/ct_api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
 CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "ct_pathology.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
