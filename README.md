# 🏥 CT Pathology Detection System

> **Автоматизированная система первичного скрининга КТ ОГК**
> Интеллектуальное решение на базе CT-CLIP foundation model и CatBoost для массового скрининга исследований ОГК.

## 📋 Описание решения

CT Pathology Detection System представляет собой систему для автоматического анализа КТ-исследований ОГК (форматы DICOM и NIfTI) на предмет наличия патологий. Обладает высокой производительностью: 5-8 секунд / исследование.

**Визуализация работы:**
![alt text](image.png)

## 🏆 Производительность 

### 📊 **Сравнение подходов (ROC AUC)**

Наша система демонстрирует значительное улучшение по сравнению с baseline подходами:

| 🔬 **Подход** | 🎯 **ROC AUC** | 📈 **Улучшение** | 💡 **Описание** |
|---------------|----------------|-------------------|------------------|
| **MedVAE3D with custom decoder** | `0.51` | — | Автоэнкодер для аномальных паттернов |
| **CT-CLIP Classifier** | `~0.51` | — | Прямая классификация на 18 классов |
| **CatBoost + CT-CLIP** | `0.75` | `+47%` | Наш промежуточный результат |
| **🚀 Финальная система** | `0.86` | `+69%` | **CT-CLIP + CatBoost + Optimization** |

### Доверительные интервалы по метрикам из `notebooks/final.ipynb`

#### Валидация (N=163; $n_{pos}=93$, $n_{neg}=70$)

Метрика      |  Значение  |  95% ДИ    |    
-------------|------------|----------------|
Recall       |  0.828     |  [0.739; 0.891]|
Specificity  |  0.814     |  [0.708; 0.888]|
Precision    |  0.856     |  [0.768; 0.914]|
Accuracy     |  0.822     |  [0.756; 0.873]|
ROC AUC      |  0.9055    |  [0.860; 0.951]|

#### Тест (N=161; $n_{pos}=91$, $n_{neg}=70$):

| Метрика     | Значение | 95% ДИ         |
| :---------- | -------: | :------------- |
| Recall      |    0.846 | [0.758; 0.906] |
| Specificity |    0.729 | [0.615; 0.819] |
| Precision   |    0.802 | [0.711; 0.869] |
| Accuracy    |    0.795 | [0.726; 0.850] |
| ROC AUC     |   0.8642 | [0.809; 0.920] |

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
- Требуется GPU с поддержкой CUDA
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
- Порты 8000 (API) должны быть свободны

## 🚀 Быстрый старт (Quick Start)

> Работа с docker-образом описана в [`DOCKER_README.md`](DOCKER_README.md).
Если требуется запустить API не в контейнере, см. [`API_README.md`](API_README.md).

### Чтобы работать над проектом

#### Шаг 1: Установка зависимостей

Запускаем скрипт установки зависимостей

``` 
bash install.sh
```

ИЛИ

##### Устанавливаем зависимости CT-CLIP

```
cd src/transformer_maskgit
pip install -e .
cd ../..
```

```
cd src/ct_clip
pip install -e .
cd ../..
```

##### Зависимости проекта

```
pip install -r requirements.txt
```

#### Шаг 2: Скачивание моделей

Скачайте предобученные модели:
- **CT_LiPro_v2.pt** → `models/CT_LiPro_v2.pt`
- **catboost_pathology_classifier.cbm** → `models/catboost_pathology_classifier.cbm`

#### Шаг 3: Откройте ноутбук `quick_start.ipynb`, находящийся в корне репозитория.

### Пример результата

```
После обработки вы получите Excel файл со следующей структурой:

| path_to_study | study_uid |   series_uid   |probability_of_pathology    | pathology   | processing_status |
|---------------|-----------|----------------|----------------------------|-------------|-------------------|
| study_001/    | 1.2.3.4.5 |                |0.8234                      | 1           | Success           |
| study_002/    | 1.2.3.4.6 |                |0.1456                      | 0           | Success           |
```

## 📁 Структура проекта

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
│   ├── 📄 03_medvae_baseline_anomaly_detection.ipynb #\ Эксперименты с моделью MedVae3d
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
│   │   ├── 📄 __init__.py         \# Инициализация пакета
│   │   ├── 📄 main.py             \# FastAPI приложение
│   │   
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
```

## 📖 Описание основных файлов

### **🔧 Core Processing:**
- **`src/pipeline/core_pipeline.py`** - Главный класс `CTPathologyPipeline` для обработки ZIP архивов
- **`src/preprocessing.py`** - Функции предобработки медицинских изображений под CT-CLIP
- **`src/feature_extraction.py`** - CT-CLIP feature extractor с поддержкой batch processing
- **`src/model.py`** - CatBoost classifier wrapper с методами train/predict


## Описание источников данных

1. CT-RATE

> [Ссылка](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
Взяли 814 исследований (300н/515)

2. MosMedData КТ с признаками рака легкого тип VIII

> [Ссылка](https://mosmed.ai/datasets/datasets/mosmeddata-kt-s-priznakami-raka-legkogo-tip-viii/)
Взяли 182 исследования (72н/110)

3. MosMedData НДКТ с признаками рака легкого тип I

> [Ссылка](https://mosmed.ai/datasets/datasets/mm/)
Взяли 100 исследований (50н/50)

4. MosMedData КТ с признаками коронавирусной инфекции (COVID-19) тип I

> [Ссылка](https://mosmed.ai/datasets/datasets/mosmeddata-kt-s-priznakami-koronavirusnoi-infektsii-covid-19-tip-i/)
Взяли 120 исследований (50н/70)

