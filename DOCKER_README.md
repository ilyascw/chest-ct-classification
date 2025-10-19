# 🚀 Быстрый запуск CT Pathology API

## Требования
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **RAM**: 32 GB
- **Диск**: 100 GB свободного места
- **NVIDIA Container Toolkit**: установлен

---

## Шаг 1: Загрузка образа

```


# Загрузка образа (~5-10 минут)

docker load -i ct-pathology-api-amd64.tar.gz

# Проверка

docker images | grep ct-pathology-api

```

---

## Шаг 2: Запуск через Docker Compose

```


# Создание директории для временных файлов

mkdir -p runtime_tmp

# Запуск

docker-compose up -d

# Проверка логов (прогрев моделей ~1-2 минуты)

docker-compose logs -f

```

**Ожидайте сообщение:** `"Application startup complete."`

---

## Шаг 3: Проверка работы

```

curl http://localhost:8000/health

```

**Ожидаемый ответ:**
```

{
"status": "healthy",
"gpu_available": true,
"models_loaded": true
}

```

---

## Тестовый запрос

```

curl -X POST http://localhost:8000/predict \
-F "file=@study.zip" \
-o results.xlsx

```

---

## Управление

```


# Остановка

docker-compose down

# Перезапуск

docker-compose restart

# Просмотр логов

docker-compose logs -f

```

---

## Troubleshooting

### ❌ "could not select device driver"

```


# Установка NVIDIA Container Toolkit (Ubuntu/Debian)

distribution=\$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | \
sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

```

### ❌ Контейнер падает с OOM

Увеличьте `shm_size` в `docker-compose.yaml`:
```

shm_size: "16g"

```

### ❌ API не отвечает

Подождите 2 минуты для прогрева моделей. Проверьте логи:
```

docker-compose logs -f

```

---

## Технические детали

| Параметр | Значение |
|----------|---------|
| **Порт** | 8000 |
| **Время прогрева** | 60-120 сек |
| **VRAM** | ~12 GB |
| **Формат входа** | ZIP с DICOM |
| **Формат выхода** | XLSX |
| **PyTorch** | 2.1.0+cu118 |