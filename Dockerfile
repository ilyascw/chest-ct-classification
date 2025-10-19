FROM --platform=linux/amd64 nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" \
    FORCE_CUDA=1

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    python3.11 \
    python3.11-venv \
    python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/

# СНАЧАЛА устанавливаем PyTorch из официального репозитория
RUN pip install --no-cache-dir \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ЗАТЕМ устанавливаем остальные зависимости, ПРОПУСКАЯ PyTorch
RUN pip install --no-cache-dir \
    $(grep -v "^torch==" requirements.txt | grep -v "^torchvision==" | grep -v "^torchaudio==")

# Копируем исходники
COPY src/ /app/src/
COPY models/ /app/models/

# Установка editable packages (если есть)
RUN if [ -d /app/src/transformer_maskgit ] && [ -f /app/src/transformer_maskgit/setup.py ]; then \
        cd /app/src/transformer_maskgit && pip install --no-cache-dir -e .; \
    fi && \
    if [ -d /app/src/ct_clip ] && [ -f /app/src/ct_clip/setup.py ]; then \
        cd /app/src/ct_clip && pip install --no-cache-dir -e .; \
    fi

RUN mkdir -p /tmp/ct_api /opt/hf_cache && chmod 777 /tmp/ct_api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
 CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]