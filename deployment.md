# 🚀 CT Pathology Detection — Инструкция по развёртыванию

## 📦 Описание решения

Docker-контейнер с FastAPI сервисом для автоматического обнаружения патологий на КТ снимках грудной клетки.

- **Образ:** `ct-pathology-api:latest`
- **Размер:** 7.33 GB (включает PyTorch, CT-CLIP модель, CatBoost)
- **Порт:** 8000
- **Требования:** Docker 20.10+, минимум 8GB RAM, рекомендуется GPU (RTX 3090 или аналог)

---

## ⚡ Быстрый старт

### 1. Загрузка образа

```


# Если образ предоставлен как .tar.gz архив

docker load < ct-pathology-api.tar.gz

# Проверка загрузки

docker images | grep ct-pathology-api

```

### 2. Запуск контейнера (CPU режим)

```

docker run -d \
--name ct-pathology-api \
-p 8000:8000 \
-v /path/to/data:/app/data:ro \
-v /path/to/results:/app/results \
ct-pathology-api:latest

```

### 3. Запуск контейнера (GPU режим)

```

docker run -d \
--name ct-pathology-api \
--gpus all \
-p 8000:8000 \
-v /path/to/data:/app/data:ro \
-v /path/to/results:/app/results \
ct-pathology-api:latest

```

### 4. Проверка работоспособности

```


# Ожидайте 30-60 секунд для инициализации моделей

docker logs -f ct-pathology-api

# Проверка health endpoint

curl http://localhost:8000/health

# Ожидаемый ответ: {"status":"healthy","pipeline_ready":true}

```

---

## 🧪 Тестирование

### Обработка ZIP архива через API

```

curl -X POST http://localhost:8000/predict \
-F "file=@/path/to/dicom_archive.zip" \
--output results.xlsx

```

### Swagger UI (интерактивная документация)

Откройте в браузере: `http://localhost:8000/docs`

Загрузите ZIP файл через веб-интерфейс и скачайте результаты.

---

## 📊 Формат входных данных

### Структура ZIP архива

```

dicom_archive.zip
├── study_001/
│   ├── slice_001.dcm
│   ├── slice_002.dcm
│   └── ...
├── study_002/
│   ├── slice_001.dcm
│   └── ...

```

**Поддерживаемые форматы:**
- DICOM серии (файлы с любым именем, распознаются по DICOM-префиксу)
- NIfTI файлы (.nii, .nii.gz)

**Требования к DICOM:**
- Наличие тегов StudyInstanceUID (0020,000D) и SeriesInstanceUID (0020,000E)
- КТ органов грудной клетки

---

## 📄 Формат выходных данных

### Excel отчёт (.xlsx)

**Лист "Results":**

| Колонка | Описание | Пример |
|---------|----------|--------|
| `path_to_study` | Путь к исследованию | `/tmp/.../study_001/` |
| `study_uid` | StudyInstanceUID | `1.2.276...863` |
| `series_uid` | SeriesInstanceUID | `1.2.276...864` |
| `probability_of_pathology` | Вероятность патологии [0, 1] | `0.734` |
| `pathology` | Бинарный прогноз (0/1) | `1` |
| `processing_status` | Статус обработки | `Success` / `Failure` |
| `time_of_processing` | Время обработки (сек) | `5.3` |
| `error_details` | Детали ошибки (если есть) | `` |

**Лист "Summary":** статистика обработки (всего/успешных/ошибок, среднее время)

**Лист "Errors":** детальные логи ошибок с traceback для проблемных исследований

---

## 🔧 Управление контейнером

### Просмотр логов

```


# Все логи

docker logs ct-pathology-api

# Логи в реальном времени

docker logs -f ct-pathology-api

# Последние 100 строк

docker logs --tail 100 ct-pathology-api

```

### Остановка и запуск

```


# Остановка

docker stop ct-pathology-api

# Запуск существующего контейнера

docker start ct-pathology-api

# Перезапуск

docker restart ct-pathology-api

```

### Удаление контейнера

```


# Остановка и удаление

docker stop ct-pathology-api
docker rm ct-pathology-api

# Принудительное удаление (если контейнер работает)

docker rm -f ct-pathology-api

```

---

## 🐛 Решение проблем

### Контейнер не запускается

```


# Проверьте логи

docker logs ct-pathology-api

# Проверьте, не занят ли порт 8000

lsof -i :8000

# или

netstat -tuln | grep 8000

```

### Health check возвращает ошибку

```


# Убедитесь, что модели инициализированы (занимает 30-60 сек)

docker logs ct-pathology-api | grep "Pipeline initialized"

# Проверьте статус pipeline

curl http://localhost:8000/info

```

### Ошибка "Out of memory" (OOM)

```


# Перезапустите контейнер с большим лимитом памяти

docker run -d \
--name ct-pathology-api \
--memory=16g \
--memory-swap=16g \
-p 8000:8000 \
ct-pathology-api:latest

```

### GPU не определяется

```


# Проверьте наличие nvidia-docker

docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Если ошибка, установите nvidia-container-toolkit:

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

```

---

## 📈 Производительность

### Рекомендуемые характеристики

**Минимальные требования (CPU):**
- CPU: 4 ядра
- RAM: 8 GB
- Диск: 10 GB (для образа)
- Время обработки: ~10-15 сек/исследование

**Рекомендуемые требования (GPU):**
- CPU: 8 ядер
- RAM: 16 GB
- GPU: NVIDIA RTX 3090 / A100 / Tesla T4
- VRAM: 8 GB
- Время обработки: ~3-5 сек/исследование

### Параллельная обработка

По умолчанию `max_workers=2`. Для изменения:

```


# Через переменную окружения

docker run -d \
--name ct-pathology-api \
-e MAX_WORKERS=4 \
-p 8000:8000 \
ct-pathology-api:latest

```

---

## 🔐 Безопасность

### Ограничения доступа

```


# Запуск только на localhost (недоступен из сети)

docker run -d \
--name ct-pathology-api \
-p 127.0.0.1:8000:8000 \
ct-pathology-api:latest

```

### Ограничения по размеру файлов

FastAPI по умолчанию ограничивает размер загружаемых файлов до 100 MB. Для изменения добавьте в `src/api/main.py`:

```

app.add_middleware(
Middleware,
max_body_size=500*1024*1024  \# 500 MB
)

```

---

## 📞 Контакты и поддержка

- **Логи ошибок:** `/tmp/ct_api/` внутри контейнера
- **API документация:** `http://localhost:8000/docs`
- **Репозиторий:** [ссылка на GitHub]
- **Email:** [ваш email]