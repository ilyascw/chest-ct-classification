# 🐳 Docker Deployment Guide

<div align="center">

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Production-ready Docker setup для CT Pathology Detection System**

</div>

---

## 📋 Содержание

- [Быстрый старт](#-быстрый-старт)
- [Системные требования](#-системные-требования)
- [Загрузка образа](#-загрузка-образа-из-архива)
- [Запуск контейнера](#-запуск-контейнера)
- [Docker Compose](#-docker-compose)
- [Конфигурация](#️-конфигурация)
- [Troubleshooting](#-troubleshooting)

---

## ⚡ Быстрый старт

### Вариант 1: Из готового архива (рекомендуется)

```


# 1. Загрузить образ из tar-архива

docker load -i ct-pathology-api.tar

# 2. Запустить контейнер

docker run -d \
--name ct-pathology-api \
--gpus all \
--shm-size=8g \
-p 8000:8000 \
-v \$(pwd)/models:/app/models:ro \
-v \$(pwd)/runtime_tmp:/tmp/ct_api \
ct-pathology-api:latest

# 3. Проверить статус

curl http://localhost:8000/health

```

### Вариант 2: Docker Compose

```

docker-compose up -d

```

---

## 🖥️ Системные требования

### Hardware

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| **GPU** | NVIDIA RTX 3070 (8GB VRAM) | RTX 3090 / A100 (24GB+) |
| **RAM** | 16 GB | 32 GB |
| **Диск** | 100 GB свободного места | 150 GB+ (SSD) |
| **CPU** | 4 ядра | 8+ ядер |

### Software

- **Docker Engine**: 20.10+
- **NVIDIA Driver**: 520.61.05+ (совместимый с CUDA 11.8)
- **NVIDIA Container Toolkit**: установлен и настроен

#### Проверка NVIDIA Container Toolkit

```


# Проверка Docker + GPU

docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Если команда работает — всё готово!

```

---

## 📦 Загрузка образа из архива

```


# Распаковка и загрузка образа (может занять 5-10 минут)

docker load -i ct-pathology-api.tar.gz

# Проверка загруженного образа

docker images | grep ct-pathology-api

```

**Ожидаемый вывод:**
```

ct-pathology-api   2   abc123def456   2 hours ago   12GB

```

---

## 🚀 Запуск контейнера

### Быстрый запуск (минимальная конфигурация)

```

docker run -d \
--name ct-pathology-api \
--gpus all \
-p 8000:8000 \
ct-pathology-api:<версия_контейнера>

```

### Production-ready запуск (рекомендуется)

```


# Создайте директории для томов

mkdir -p runtime_tmp hf_cache

# Запуск с полной конфигурацией

docker run -d \
--name ct-pathology-api \
--gpus all \
--shm-size=8g \
--restart unless-stopped \
-p 8000:8000 \
-v \$(pwd)/models:/app/models:ro \
-v \$(pwd)/runtime_tmp:/tmp/ct_api \
-v \$(pwd)/hf_cache:/opt/hf_cache \
-e OMP_NUM_THREADS=4 \
-e MKL_NUM_THREADS=4 \
-e TOKENIZERS_PARALLELISM=false \
ct-pathology-api:latest

```

### Объяснение параметров

| Параметр | Назначение |
|----------|------------|
| `--gpus all` | Проброс всех GPU в контейнер (обязательно) |
| `--shm-size=8g` | Разделяемая память для обработки больших томов |
| `-p 8000:8000` | Проброс порта API |
| `-v models:/app/models:ro` | Монтирование моделей (read-only) |
| `-v runtime_tmp:/tmp/ct_api` | Временные файлы и распаковка ZIP |
| `-v hf_cache:/opt/hf_cache` | Кэш HuggingFace моделей |

---

## ⚙️ Конфигурация

### Переменные окружения

| Переменная | Значение по умолчанию | Описание |
|------------|----------------------|----------|
| `NVIDIA_VISIBLE_DEVICES` | `all` | Видимые GPU для контейнера |
| `OMP_NUM_THREADS` | `4` | Потоки OpenMP |
| `MKL_NUM_THREADS` | `4` | Потоки Intel MKL |
| `TOKENIZERS_PARALLELISM` | `false` | Отключение параллелизма токенайзера |
| `HF_HOME` | `/opt/hf_cache` | Путь к кэшу HuggingFace |

---

## 🐛 Troubleshooting

### Проблема: "could not select device driver"

**Причина:** NVIDIA Container Toolkit не установлен или не настроен.

**Решение:**
```


# Ubuntu/Debian

distribution=\$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | \
sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

```

---

### Проблема: "Torch not compiled with CUDA support"

**Причина:** Контейнер запущен без `--gpus all`.

**Решение:**
```


# Остановить контейнер

docker stop ct-pathology-api

# Удалить контейнер

docker rm ct-pathology-api

# Запустить с --gpus all

docker run -d --name ct-pathology-api --gpus all -p 8000:8000 ct-pathology-api:latest
