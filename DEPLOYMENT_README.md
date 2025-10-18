# 🏥 CT Pathology Detection System — Финальная версия

## 🎯 Описание решения

Автоматизированная система обнаружения патологий на КТ снимках грудной клетки на основе CT-CLIP foundation model и CatBoost классификатора.

**Ключевые характеристики:**
- ROC AUC: **0.86** на валидационной выборке (1100+ исследований)
- Время обработки: **3-7 секунд** на исследование (GPU: Tesla T4)
- Поддержка: DICOM серии и NIfTI файлы
- API: FastAPI с автоматической документацией
- Контейнеризация: Docker образ со всеми зависимостями

---

## 📦 Состав поставки

```

├── ct-pathology-api.tar.gz         \# Docker образ (7.33 GB)
├── Dockerfile                  \# Конфигурация сборки
├── docker-compose.yml          \# Compose для быстрого запуска
├── DEPLOYMENT.md               \# Инструкция по развёртыванию
├── DOCKER_GUIDE.md             \# Справка по Docker командам
└── README_DEPLOYMENT.md        \# Этот файл

```

---

## ⚡ Быстрый запуск (30 секунд)

```


# 1. Загрузка образа

docker load < ct-pathology-api.tar.gz

# 2. Запуск контейнера

docker run -d --name ct-pathology-api -p 8000:8000 ct-pathology-api:latest

# 3. Проверка готовности (подождите 30 сек)

curl http://localhost:8000/health

# 4. Тестовая обработка

curl -X POST http://localhost:8000/predict \
-F "file=@test_data.zip" \
--output results.xlsx

```

---

## 🧪 Тестирование на примере данных

### Тестовые архивы (если предоставлены)

```


# Пример 1: норма

curl -X POST http://localhost:8000/predict \
-F "file=@norma_anon.zip" \
--output norma_results.xlsx

# Пример 2: патология

curl -X POST http://localhost:8000/predict \
-F "file=@pneumonia_anon.zip" \
--output pneumonia_results.xlsx

```

---

## 📊 Метрики качества

### Валидационная выборка (n=150)

| Метрика | Значение |
|---------|----------|
| **ROC AUC** | 0.86 |
| **Accuracy** | 0.82 |
| **Sensitivity (Recall)** | 0.88 |
| **Specificity** | 0.78 |
| **Precision** | 0.81 |

**Данные:** 1100+ КТ исследований из 4 источников (MosMedData COVID-19, рак лёгких, пневмония, норма)

---

## 🏗️ Архитектура решения

### Pipeline обработки

```

ZIP Archive
↓
Data Discovery (DICOM/NIfTI detection)
↓
Volume Loading (распаковка + валидация UIDs)
↓
Preprocessing (нормализация spacing, интенсивности)
↓
CT-CLIP Feature Extraction (512D embeddings)
↓
CatBoost Classification (pathology probability)
↓
Excel Report (StudyUID, SeriesUID, probability, status)

```

### Технологический стек

- **CT-CLIP:** foundation model для медицинских изображений (BioMedCLIP адаптация)
- **CatBoost:** градиентный бустинг для классификации на эмбеддингах
- **MONAI:** медицинская обработка изображений
- **FastAPI:** REST API с автоматической документацией
- **Docker:** контейнеризация со всеми зависимостями

---

## 🔧 Технические требования

### Минимальные (CPU)
- Docker 20.10+
- 4 CPU cores
- 8 GB RAM
- 10 GB свободного места

### Рекомендуемые (GPU)
- Docker 20.10+ с nvidia-container-toolkit
- NVIDIA GPU: RTX 3090 / A100 / Tesla T4
- 8+ GB VRAM
- 16 GB RAM
- 10 GB свободного места

---

## 📝 Формат данных

### Вход: ZIP архив

Структура:
```

archive.zip
└── study_folder/
├── slice_001.dcm
├── slice_002.dcm
└── ...

```

**Требования:**
- DICOM с тегами StudyInstanceUID (0020,000D) и SeriesInstanceUID (0020,000E)
- КТ грудной клетки
- Любые имена файлов (определяются по DICOM-префиксу)

### Выход: Excel отчёт (.xlsx)

Листы:
1. **Results:** path_to_study, study_uid, series_uid, probability_of_pathology, pathology, processing_status, time_of_processing, error_details
2. **Summary:** статистика обработки
3. **Errors:** детальные логи ошибок

---

## 🐛 Решение проблем

### Проблема: Контейнер не запускается

**Решение:**
```

docker logs ct-pathology-api

# Проверьте наличие ошибок в логах

```

### Проблема: Health check не проходит

**Решение:**
```


# Дождитесь инициализации моделей (30-60 сек)

docker logs ct-pathology-api | grep "Pipeline initialized"

```

### Проблема: Out of memory

**Решение:**
```

docker run -d --name ct-pathology-api --memory=16g -p 8000:8000 ct-pathology-api:latest

```

---