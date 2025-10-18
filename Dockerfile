# Multi-stage build для оптимизации размера образа
FROM python:3.11-slim AS base

# Установка системных зависимостей (обновлённые пакеты для Debian Trixie)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование requirements для кэширования слоя
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# Копирование исходного кода
COPY src/ ./src/
COPY models/ ./models/

# Установка CT-CLIP и transformer_maskgit
RUN cd src/transformer_maskgit && pip install --no-cache-dir -e . && cd ../.. && \
    cd src/ct_clip && pip install --no-cache-dir -e . && cd ../..

# Создание директории для временных файлов
RUN mkdir -p /tmp/ct_api

# Expose порт API
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Запуск API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
