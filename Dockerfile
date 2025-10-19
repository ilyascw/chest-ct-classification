# CUDA runtime + cuDNN под RTX 30xx (CUDA 11.8)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # HuggingFace кэш внутрь образа
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    # Ускоренный backend для скачивания (можно выключить)
    HF_HUB_ENABLE_HF_TRANSFER=1

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Python
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip

WORKDIR /app

# Копируем requirements для кэша
COPY requirements.txt /app/requirements.txt

# Важно: поставить torch для CUDA 11.8
# Либо положиться на резолвер, либо явно указать индекс
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir fastapi uvicorn[standard] python-multipart requests

# Копируем исходники и модели
COPY src/ /app/src/
COPY models/ /app/models/

# Локальные editable-пакеты
RUN cd /app/src/transformer_maskgit && pip install --no-cache-dir -e . \
 && cd /app/src/ct_clip && pip install --no-cache-dir -e .

# (Опционально) Препуллить HuggingFace модель/токенайзер для офлайна
# Если интернет недоступен на проде — раскомментировать.
# RUN python3 - <<'PY'
# from transformers import BertTokenizer, BertModel
# name = "microsoft/BiomedVLP-CXR-BERT-specialized"
# BertTokenizer.from_pretrained(name)
# BertModel.from_pretrained(name)
# PY

# Папка для временных файлов API
RUN mkdir -p /tmp/ct_api

# Экспорт порта
EXPOSE 8000

# Healthcheck через curl, чтобы не тянуть requests
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
 CMD curl -fsS http://localhost:8000/health || exit 1

# Запуск API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
