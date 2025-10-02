# 🏥 CT Pathology Detection System

> **Автоматизированная система выявления патологий на компьютерных томограммах грудной клетки**
> 
> Интеллектуальное решение на базе CT-CLIP foundation model и CatBoost для массового скрининга медицинских изображений.

## 📋 Описание решения

CT Pathology Detection System представляет собой production-ready систему для автоматического анализа медицинских изображений (DICOM и NIfTI) с целью выявления снимков с патологиями в исследованиях КТ органов грудной клетки.

## 🏆 Производительность модели

### 📊 **Сравнение подходов (ROC AUC)**

Наша система демонстрирует значительное улучшение по сравнению с baseline подходами:

| 🔬 **Подход** | 🎯 **ROC AUC** | 📈 **Улучшение** | 💡 **Описание** |
|---------------|----------------|-------------------|------------------|
| **MedVAE3D Baseline** | `0.51` | — | Автоэнкодер для аномальных паттернов |
| **CT-CLIP Classifier** | `~0.51` | — | Прямая классификация на 18 классов |
| **CatBoost + CT-CLIP** | `0.75` | `+47%` | Наш промежуточный результат |
| **🚀 Финальная система** | `0.86` | `+69%` | **CT-CLIP + CatBoost + Optimization** |

### 🎯 **Детальные метрики производительности**

#### **🏁 TEST SET (Тестовая выборка - финальный результат)**

Accuracy: 79.50% ✅ | ROC AUC: 86.42% 🔥

Precision: 80.21% 🎯 | PR AUC: 88.33% 🔥

Recall: 84.62% 📊 | F1-Score: 82.35% ⭐

Sensitivity: 84.62% | Specificity: 72.86%

### 🎯 Назначение системы:
- **Массовый скрининг** - обработка больших объёмов медицинских данных
- **Поддержка принятия решений** - помощь врачам-рентгенологам

### 🚀 Технологический стек:
- **AI Models**: CT-CLIP (Foundation Model) + CatBoost Classifier
- **Backend**: FastAPI + Core Processing Pipeline
- **Containerization**: Docker + Docker Compose
- **Data Processing**: PyTorch, MONAI, SimpleITK, nibabel

## ✨ Основные возможности

### 📊 **Функциональные возможности:**
- ✅ Обработка ZIP архивов с DICOM сериями и NIfTI файлами
- ✅ Автоматическое выявление патологий с оценкой вероятности (0.0-1.0)
- ✅ Генерация детализированных Excel отчётов согласно требованиям ТЗ
- ✅ Локализация выявленных патологий в координатах (x,y,z)
- ✅ REST API для программной интеграции
- ✅ Web интерфейс для интерактивного использования
- ✅ Robust error handling и детальная отчётность об ошибках

### 📈 **Производительность:**
- **Время обработки**: ≤ 10 минут на одно исследование
- **Поддерживаемые форматы**: DICOM (.dcm), NIfTI (.nii, .nii.gz)
- **Выходной формат**: Excel (.xlsx) с детальной статистикой

### ⚠️ **Ограничения системы:**
- Система предназначена только для КТ исследований грудной клетки
- Требуется GPU с поддержкой CUDA для оптимальной производительности
- Система предоставляет вспомогательную информацию, окончательное решение принимает врач

## 🖥️ Системные требования

### **Рекомендуемые требования:**
- **GPU**: NVIDIA GPU с 8GB+ VRAM (Tesla T4, V100, RTX 3080+)
- **CUDA**: версия 11.8+
- **RAM**: 32GB+
- **CPU**: 8+ cores
- **Disk Space**: 100GB+ (NVMe SSD предпочтительно)

### **Сетевые требования:**
- Доступ к интернету для загрузки Docker образов
- Порты 8000 (API) и 7860 (Web UI) должны быть свободны

## 🚀 Быстрый старт (Quick Start)

```
### Запуск системы
```
# Запуск через Docker Compose (рекомендуется)

docker-compose up -d

# Проверка статуса сервисов

docker-compose ps

```

### Использование системы**

#### **REST API:**
```


# Загрузка файлов через API

curl -X POST "http://localhost:8000/api/v1/process" \
-F "files=@study1.zip" \
-F "files=@study2.zip"

# Проверка статуса обработки

curl "http://localhost:8000/api/v1/status/{task_id}"

# Скачивание результатов

curl "http://localhost:8000/api/v1/download/{task_id}" -o results.xlsx

```

### **Шаг 5: Пример результата**

После обработки вы получите Excel файл со следующей структурой:

| path_to_study | study_uid | probability_of_pathology | pathology | processing_status |
|---------------|-----------|-------------------------|-----------|-------------------|
| study_001/    | 1.2.3.4.5 | 0.8234                 | 1         | Success          |
| study_002/    | 1.2.3.4.6 | 0.1456                 | 0         | Success          |

## 📁 Структура проекта

```

ct-pathology-detection/
├── 📄 README.md                    \# Этот файл
├── 📄 docker-compose.yml           \# Оркестрация сервисов
├── 📄 Dockerfile                   \# Образ приложения
├── 📄 requirements.txt             \# Python зависимости
├── 📄 .env.example                 \# Пример переменных окружения
│
├── 📂 src/                         \# Исходный код системы
│   ├── 📄 __init__.py             \# Инициализация пакета
│   ├── 📄 preprocessing.py         \# Предобработка медицинских данных
│   ├── 📄 feature_extraction.py   \# CT-CLIP feature extraction
│   ├── 📄 model.py                \# CatBoost классификатор
│   │
│   ├── 📂 notebooks/ \# Папка с юпитер ноутбуками, в которых велась работа над проектом
│   ├── 📄 01_data_download_and_unpack.ipynb \# Загрузка, распаковка и первое знакомство с данными
│   ├── 📄 02_preprocessing_ct.ipynb \# Здесь тестируем модуль для предобработки данных
│   ├── 📄 03_medvae_baseline_anomaly_detection.ipynbpy #\ Эксперименты с моделью MedVae3d
│   ├── 📄 04_ct_clip_pipeline.py #\ Эксперименты с ct_clip - предпосылки, что при увеличении количества обучающих данных подход может сработать
│   ├── 📄 final.ipynb #\ Финальный эксперимент с ct-clip (1182 томов использовано на обучение/валидацию/тестирование)
│   ├── 📄 final&autogluon.ipynb #\ Быстрое тестирование auto ml на извлеченных эмбедингах.
│   │   
│   ├── 📂 pipeline/               \# Core processing pipeline
│   │   ├── 📄 core_pipeline.py    \# Главный обработчик
│   │   ├── 📄 data_models.py      \# Модели данных
│   │   ├── 📄 data_discovery.py   \# Обнаружение медицинских данных
│   │   └── 📄 volume_loader.py    \# Загрузка медицинских томов
│   │
│   ├── 📂 api/                    \# REST API
│   │   ├── 📄 main.py             \# FastAPI приложение
│   │   └── 📄 models.py           \# API модели данных
│   │
│   │
│   └── 📂 CTPreprocessor/         \# Дополнительные утилиты
│       └── 📄 ct_preprocessor.py  \# Robust DICOM/NIfTI loading
│
├── 📂 models/                     \# Предобученные модели
│   ├── 📄 CTLiProv2.pt           \# CT-CLIP checkpoint (~2GB)
│   └── 📄 catboost_model.cbm     \# CatBoost модель (~10MB)
│
├── 📂 docs/                       \# Документация
│   ├── 📄 deployment_guide.md     \# Руководство по развертыванию
│   ├── 📄 user_manual.md          \# Руководство пользователя
│   └── 📄 api_reference.md        \# Справочник API
│
├── 📂 tests/                      \# Тесты
    ├── 📄 test_pipeline.py        \# Тесты core pipeline
    └── 📄 test_api.py             \# Тесты API

```

## 📖 Описание основных файлов

### **🔧 Core Processing:**
- **`src/pipeline/core_pipeline.py`** - Главный класс `CTPathologyPipeline` для обработки ZIP архивов
- **`src/preprocessing.py`** - Функции предобработки медицинских изображений под CT-CLIP
- **`src/feature_extraction.py`** - CT-CLIP feature extractor с поддержкой batch processing
- **`src/model.py`** - CatBoost classifier wrapper с методами train/predict

### **🌐 API & Web:**
- **`src/api/main.py`** - FastAPI приложение с эндпоинтами `/process`, `/status`, `/download`
- **`src/web/gradio_interface.py`** - Gradio web интерфейс для интерактивного использования

### **🐳 Deployment:**
- **`docker-compose.yml`** - Оркестрация сервисов (API + Web + Redis)
- **`Dockerfile`** - Multi-stage build с оптимизацией размера образа
- **`requirements.txt`** - Точные версии всех Python зависимостей

### **📊 Models:**
- **`models/CTLiProv2.pt`** - Предобученная CT-CLIP модель (Foundation Model)
- **`models/catboost_model.cbm`** - Обученный CatBoost классификатор

## 🛟 Поддержка и документация

### **📚 Дополнительная документация:**
- [📖 Deployment Guide](docs/deployment_guide.md) - Подробное руководство по развертыванию
- [👥 User Manual](docs/user_manual.md) - Руководство пользователя с примерами
- [🔧 API Reference](docs/api_reference.md) - Полная документация API

### **🐛 Решение проблем:**

# Просмотр логов сервисов
```
docker-compose logs -f
```
# Перезапуск сервисов
```
docker-compose restart
```
# Полная переустановка
```
docker-compose down -v
docker-compose up -d --build

```
