# 🏥 CT Pathology Detection System

> **Автоматизированная система выявления патологий на компьютерных томограммах грудной клетки**
> 
> Интеллектуальное решение на базе CT-CLIP foundation model и CatBoost для массового скрининга медицинских изображений.

## 📋 Описание решения

CT Pathology Detection System представляет собой систему для автоматического анализа медицинских изображений (DICOM и NIfTI) с целью выявления снимков с патологиями в исследованиях КТ органов грудной клетки.

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

### Шаг 1: Установка зависимостей

Запускаем скрипт установки зависимостей

``` 
bash install.sh
```
**Скрипт автоматически установит:**
- CT-CLIP модуль (`src/ct_clip`)
- Transformer MaskGIT (`src/transformer_maskgit`)
- Все Python зависимости из `requirements.txt`

### Шаг 2: Скачивание моделей

Скачайте предобученные модели:
- CT_LiPro_v2.pt (~2GB) → `models/CT_LiPro_v2.pt`
- **catboost_pathology_classifier.cbm** (~10MB) → `models/catboost_pathology_classifier.cbm`

### Шаг 3: Откройте ноутбук `quick_start.ipynb`, находящийся в корне репозитория.

```
### Пример результата**

После обработки вы получите Excel файл со следующей структурой:

| path_to_study | study_uid |   series_uid   |probability_of_pathology    | pathology   | processing_status |
|---------------|-----------|----------------|----------------------------|-------------|-------------------|
| study_001/    | 1.2.3.4.5 |                |0.8234                      | 1           | Success           |
| study_002/    | 1.2.3.4.6 |                |0.1456                      | 0           | Success           |

## 📁 Структура проекта

```
```
ct-pathology-detection/
├── 📄 README.md                    \# Этот файл
├── 📄 docker-compose.yml           \# Оркестрация сервисов
├── 📄 Dockerfile                   \# Образ приложения
├── 📄 requirements.txt             \# Python зависимости
├── 📄 .env.example                 \# Пример переменных окружения
├── 📂 notebooks/ \# Папка с юпитер ноутбуками, в которых велась работа над проектом
│   ├── 📄 01_data_download_and_unpack.ipynb \# Загрузка, распаковка и первое знакомство с данными
│   ├── 📄 02_preprocessing_ct.ipynb \# Здесь тестируем модуль для предобработки данных
│   ├── 📄 03_medvae_baseline_anomaly_detection.ipynbpy #\ Эксперименты с моделью MedVae3d
│   ├── 📄 04_ct_clip_pipeline.py #\ Эксперименты с ct_clip - предпосылки, что при увеличении количества обучающих данных подход может сработать
│   ├── 📄 final.ipynb #\ Финальный эксперимент с ct-clip (1182 томов использовано на обучение/валидацию/тестирование)
|   ├── 📄 final&autogluon.ipynb #\ Быстрое тестирование auto ml на извлеченных эмбедингах.
|
|
├── 📂 src/                         \# Исходный код системы
│   ├── 📄 __init__.py             \# Инициализация пакета
│   ├── 📄 preprocessing.py         \# Предобработка медицинских данных
│   ├── 📄 feature_extraction.py   \# CT-CLIP feature extraction
│   ├── 📄 model.py                \# CatBoost классификатор
│   │
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

### **🌐 API**
- **`src/api/main.py`** - FastAPI приложение с эндпоинтами `/process`, `/status`, `/download`


```
