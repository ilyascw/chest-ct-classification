# 🌐 API Documentation

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Swagger](https://img.shields.io/badge/Swagger-85EA2D?style=for-the-badge&logo=swagger&logoColor=black)
![OpenAPI](https://img.shields.io/badge/OpenAPI-6BA539?style=for-the-badge&logo=openapi-initiative&logoColor=white)

**REST API для обнаружения патологий на КТ снимках грудной клетки**

[Быстрый старт](#-быстрый-старт) • [Endpoints](#-endpoints) • [Примеры](#-примеры-использования) • [Ошибки](#-обработка-ошибок)

</div>

---

## 📋 Содержание

- [Быстрый старт](#-быстрый-старт)
- [Базовый URL](#-базовый-url)
- [Endpoints](#-endpoints)
- [Форматы данных](#-форматы-данных)
- [Примеры использования](#-примеры-использования)
- [Обработка ошибок](#-обработка-ошибок)
- [Лимиты и ограничения](#️-лимиты-и-ограничения)

---

## ⚡ Быстрый старт

### 1. Запустить контейнер

```

docker run -d --name ct-api --gpus all -p 8000:8000 ct-pathology-api:latest

```

### 2. Проверить доступность

```

curl http://localhost:8000/health

```

### 3. Отправить ZIP с исследованиями

```

curl -X POST http://localhost:8000/predict \
-F "file=@studies.zip" \
-o results.xlsx

```

### 4. Открыть результаты

```


# Excel файл с 3 листами:

# - Results: детальные результаты по каждому исследованию

# - Summary: статистика обработки

# - Errors: ошибки (если были)

```

---

## 🌐 Базовый URL

```

http://localhost:8000

```

**Interactive docs (Swagger UI):**
```

http://localhost:8000/docs

```

**ReDoc:**
```

http://localhost:8000/redoc

```

---

## 📡 Endpoints

### `GET /health`

Проверка состояния сервиса.

#### Response

```

{
"status": "ok",
"models_loaded": true
}

```

#### Коды ответа

| Код | Описание |
|-----|----------|
| `200` | Сервис работает корректно |
| `503` | Сервис недоступен (модели не загружены) |

---

### `GET /info`

Информация о конфигурации сервиса.

#### Response

```

{
"version": "1.0.0",
"device": "cuda",
"models": {
"ct_clip": "/app/models/CT_LiPro_v2.pt",
"catboost": "/app/models/catboost_pathology_classifier.cbm"
},
"max_workers": 1,
"uptime_seconds": 3600
}

```

---

### `POST /predict`

**Основной эндпоинт**: обработка ZIP с КТ исследованиями.

#### Request

**Content-Type:** `multipart/form-data`

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `file` | File | ✅ | ZIP-архив с DICOM или NIfTI файлами |

#### Response

**Content-Type:** `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`

Excel файл с 3 листами:

1. **Results** — результаты по каждому исследованию
2. **Summary** — общая статистика
3. **Errors** — детали ошибок (если были)

#### Пример запроса

```

curl -X POST http://localhost:8000/predict \
-H "Content-Type: multipart/form-data" \
-F "file=@batch_studies.zip" \
-o results.xlsx

```

#### Время обработки

| Количество исследований | Примерное время |
|-------------------------|-----------------|
| 1 исследование | 5-8 секунд |
| 10 исследований | 1-2 минуты |
| 100 исследований | 10-15 минут |
| 400 исследований | ~40-50 минут |

---

## 📊 Форматы данных

### Входной ZIP-архив

Поддерживаемые структуры:

#### Вариант 1: DICOM серии

```

studies.zip
├── study_001/
│   ├── series_001/
│   │   ├── slice_001.dcm
│   │   ├── slice_002.dcm
│   │   └── ...
│   └── series_002/
│       └── ...
├── study_002/
│   └── ...

```

**Требования к DICOM:**
- Обязательные теги: `StudyInstanceUID`, `SeriesInstanceUID`
- Консистентные UIDs внутри серии
- Формат: `.dcm`, `.dicom`

#### Вариант 2: NIfTI файлы

```

studies.zip
├── study_001.nii
├── study_002.nii.gz
└── study_003.nii

```

**Поддерживаемые форматы:**
- `.nii`
- `.nii.gz` (сжатый)

---

### Выходной Excel файл

#### Лист 1: Results

| Колонка | Тип | Описание |
|---------|-----|----------|
| `path_to_study` | string | Путь к исследованию в ZIP |
| `study_uid` | string | Study Instance UID (для DICOM) |
| `series_uid` | string | Series Instance UID (для DICOM) |
| `probability_of_pathology` | float | Вероятность патологии [0.0-1.0] |
| `pathology` | string | `Pathology` или `Normal` |
| `processing_status` | string | `Success` или `Failure` |
| `time_of_processing` | float | Время обработки (секунды) |
| `error_details` | string | Описание ошибки (если есть) |

#### Лист 2: Summary

```

Total Studies: 400
Successful: 395
Failed: 5
Average Processing Time: 6.2s
Pathology Detected: 187 (47.3%)
Normal: 208 (52.7%)

```

#### Лист 3: Errors

Детальная информация об ошибках:
- Путь к файлу
- Тип ошибки
- Stack trace
- Завершённые шаги обработки

---

## 💡 Примеры использования

### Python (requests)

```

import requests

# Загрузка ZIP

with open('studies.zip', 'rb') as f:
response = requests.post(
'http://localhost:8000/predict',
files={'file': ('studies.zip', f, 'application/zip')}
)

# Сохранение результата

with open('results.xlsx', 'wb') as f:
f.write(response.content)

print(f"Status: {response.status_code}")
print(f"Saved to: results.xlsx")

```

### Python (aiohttp) — async

```

import aiohttp
import asyncio

async def process_studies(zip_path: str):
async with aiohttp.ClientSession() as session:
with open(zip_path, 'rb') as f:
data = aiohttp.FormData()
data.add_field('file', f, filename='studies.zip')

            async with session.post(
                'http://localhost:8000/predict',
                data=data
            ) as response:
                content = await response.read()
                with open('results.xlsx', 'wb') as out:
                    out.write(content)
                return response.status
    
# Запуск

status = asyncio.run(process_studies('batch.zip'))
print(f"Status: {status}")

```

### cURL — с прогресс-баром

```

curl -X POST http://localhost:8000/predict \
-F "file=@studies.zip" \
-o results.xlsx \
--progress-bar

```

### wget

```

wget --post-file=studies.zip \
--header="Content-Type: multipart/form-data" \
http://localhost:8000/predict \
-O results.xlsx

```

---

## ❌ Обработка ошибок

### HTTP коды ответа

| Код | Статус | Описание |
|-----|--------|----------|
| `200` | ✅ Success | Обработка успешна |
| `400` | ❌ Bad Request | Некорректный файл или формат |
| `422` | ❌ Validation Error | Отсутствует параметр `file` |
| `500` | ❌ Internal Error | Ошибка сервера |
| `503` | ⚠️ Service Unavailable | Модели не загружены |

### Примеры ошибок

#### 400 Bad Request

```

{
"detail": "Invalid ZIP archive: cannot extract files"
}

```

**Причина:** Повреждённый или невалидный ZIP-архив.

#### 422 Validation Error

```

{
"detail": [
{
"loc": ["body", "file"],
"msg": "field required",
"type": "value_error.missing"
}
]
}

```

**Причина:** Не передан параметр `file` в запросе.

#### 500 Internal Server Error

```

{
"detail": "CUDA out of memory",
"traceback": "RuntimeError: ..."
}

```

**Причина:** Недостаточно VRAM на GPU.

---

## ⚙️ Лимиты и ограничения

### Размер файлов

| Параметр | Лимит |
|----------|-------|
| **Максимальный размер ZIP** | Не ограничен (зависит от диска) |
| **Рекомендуемый батч** | До 100 GB |
| **Максимальный размер одного исследования** | До 2 GB |

### Производительность

| Метрика | Значение |
|---------|----------|
| **Параллельная обработка** | 1 исследование (sequential) |
| **Среднее время на исследование** | 5-8 секунд |
| **VRAM requirements** | 6-8 GB |
| **RAM requirements** | 12-16 GB |

### Форматы

| Формат | Поддержка |
|--------|-----------|
| DICOM (`.dcm`, `.dicom`) | ✅ |
| NIfTI (`.nii`, `.nii.gz`) | ✅ |
| PNG/JPEG | ❌ |
| Другие форматы | ❌ |

---

## 🔐 Валидация данных

### DICOM

Обязательные требования:
- ✅ Наличие `StudyInstanceUID`
- ✅ Наличие `SeriesInstanceUID`
- ✅ Консистентные UIDs в серии
- ✅ Валидные пиксельные данные

**При нарушении:** исследование помечается как `Failure` с детальным error_details.

### NIfTI

Требования:
- ✅ Валидный NIfTI header
- ✅ 3D volume данные
- ✅ Readable pixdim и affine

---

## 📈 Мониторинг обработки

### Просмотр логов

```


# Docker logs

docker logs -f ct-pathology-api

# Фильтр ошибок

docker logs ct-pathology-api 2>\&1 | grep ERROR

# Статистика обработки

docker logs ct-pathology-api | grep "completed in"

```

### Пример логов

```

2025-10-19 15:30:01 - INFO - Processing study: 1.2.840.113619.2.55...
2025-10-19 15:30:04 - INFO - ✅ Preprocessing completed
2025-10-19 15:30:06 - INFO - ✅ Feature extraction completed
2025-10-19 15:30:07 - INFO - ✅ Classification completed
2025-10-19 15:30:07 - INFO - Study completed in 6.2s: pathology=Normal, prob=0.123
2025-10-19 15:30:07 - DEBUG - 🧹 Memory cleanup for study 1.2.840...

```

---

## 🚨 Важные замечания

### ⚠️ GPU обязателен

Сервис **требует NVIDIA GPU** с CUDA 11.8+. CPU-режим не поддерживается из-за архитектуры CT-CLIP.

### ⚠️ Последовательная обработка

Исследования обрабатываются **последовательно** (`max_workers=1`) для:
- Стабильности памяти
- Предсказуемого времени обработки
- Избежания OOM на больших батчах

### ⚠️ Временные файлы

ZIP распаковываются в `/tmp/ct_api`. Убедитесь, что:
- Достаточно места на диске
- Директория смонтирована на быстрый диск (SSD)
- После обработки файлы автоматически удаляются

---

## 📞 Поддержка

### Swagger UI

Интерактивная документация доступна по адресу:
```

http://localhost:8000/docs

```

### Примеры запросов

Все примеры можно протестировать в Swagger UI с кнопкой **"Try it out"**.