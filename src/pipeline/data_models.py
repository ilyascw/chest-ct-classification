"""
Data models for Core Pipeline
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from enum import Enum
import json
import numpy as np


class DataType(Enum):
    """Тип медицинских данных"""
    DICOM = "dicom"
    NIFTI = "nifti"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Статус обработки исследования"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class StudyInfo:
    """
    Информация об обнаруженном медицинском исследовании.
    
    Attributes:
        path_to_study: Путь к директории с исследованием
        study_uid: Уникальный идентификатор исследования (для DICOM — StudyInstanceUID)
        series_uid: Уникальный идентификатор серии (для DICOM — SeriesInstanceUID)
        data_type: Тип данных (DICOM/NIFTI)
        files_count: Количество файлов в серии
        file_size_mb: Размер файлов в МБ
        metadata: Дополнительные метаданные
    """
    path_to_study: str
    study_uid: str
    series_uid: str
    data_type: DataType
    files_count: int
    file_size_mb: float
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """
        Валидация UIDs после инициализации.
        
        Для DICOM запрещена автогенерация UID — они должны быть прочитаны
        из DICOM-тегов (0020,000D) и (0020,000E).
        Для NIfTI генерируются synthetic UIDs на основе имени файла.
        """
        # Для DICOM UIDs должны быть явно заданы, автогенерация запрещена
        if self.data_type == DataType.DICOM:
            if not self.study_uid or self.study_uid == "unknown":
                raise ValueError(
                    "DICOM StudyInstanceUID (0020,000D) is missing or invalid. "
                    "Auto-generation is prohibited for DICOM data."
                )
            if not self.series_uid or self.series_uid == "unknown":
                raise ValueError(
                    "DICOM SeriesInstanceUID (0020,000E) is missing or invalid. "
                    "Auto-generation is prohibited for DICOM data."
                )
        
        # Для NIfTI генерируем UIDs на основе имени файла
        elif self.data_type == DataType.NIFTI:
            if not self.study_uid or self.study_uid == "unknown":
                file_stem = Path(self.path_to_study).stem.replace('.nii', '')
                self.study_uid = f"nifti_study_{file_stem}"
            if not self.series_uid or self.series_uid == "unknown":
                file_stem = Path(self.path_to_study).stem.replace('.nii', '')
                self.series_uid = f"nifti_series_{file_stem}"


@dataclass
class VolumeData:
    """
    Загруженный медицинский том с метаданными.
    
    Attributes:
        path: Путь к исходным данным
        volume: 3D массив с данными тома
        spacing: Размер вокселя (X, Y, Z) в мм
        origin: Координаты начала координат
        direction: Матрица направления осей
        metadata: Дополнительные метаданные (включая RescaleSlope/Intercept)
        preprocessing_info: Информация о предобработке
    """
    path: Path
    volume: np.ndarray
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: np.ndarray
    metadata: Dict[str, Any]
    preprocessing_info: Dict[str, Any]


@dataclass
class FeatureVector:
    """
    Извлечённые признаки из CT-CLIP.
    
    Attributes:
        embeddings: 512-мерный вектор эмбеддингов
        extraction_time: Время извлечения признаков (сек)
        model_info: Информация о модели
        text_prompt: Использованный текстовый промпт
    """
    embeddings: np.ndarray
    extraction_time: float
    model_info: Dict[str, str]
    text_prompt: str


@dataclass
class ClassificationResult:
    """
    Результат классификации CatBoost.
    
    Attributes:
        probability_of_pathology: Вероятность патологии [0, 1]
        pathology_prediction: Бинарный прогноз (0 = норма, 1 = патология)
        confidence_score: Уровень уверенности модели
        model_version: Версия модели CatBoost
        inference_time: Время инференса (сек)
    """
    probability_of_pathology: float
    pathology_prediction: int
    confidence_score: float
    model_version: str
    inference_time: float


@dataclass
class ProcessingResult:
    """
    Полный результат обработки одного исследования.
    
    Attributes:
        path_to_study: Путь к исследованию
        study_uid: Идентификатор исследования
        series_uid: Идентификатор серии
        probability_of_pathology: Вероятность патологии
        pathology: Бинарный прогноз (0/1)
        processing_status: Статус обработки
        time_of_processing: Время обработки (сек)
        error_details: Детали ошибки (если есть)
        processing_steps_completed: Список завершённых этапов
    """
    path_to_study: str
    study_uid: str
    series_uid: str
    probability_of_pathology: float
    pathology: int
    processing_status: str
    time_of_processing: float
    error_details: str = ""
    processing_steps_completed: List[str] = None

    def __post_init__(self) -> None:
        """Инициализация processing_steps_completed как пустого списка"""
        if self.processing_steps_completed is None:
            self.processing_steps_completed = []

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для Excel отчёта"""
        return asdict(self)


@dataclass
class PipelineConfig:
    """
    Конфигурация pipeline.
    
    Attributes:
        ct_clip_checkpoint: Путь к чекпоинту CT-CLIP
        catboost_model: Путь к модели CatBoost
        text_prompt: Текстовый промпт для CT-CLIP
        device: Устройство для вычислений (cuda/cpu/auto)
        max_workers: Количество параллельных потоков
        log_level: Уровень логирования
    """
    ct_clip_checkpoint: str
    catboost_model: str
    text_prompt: str = "chest computed tomography scan for pathology detection"
    device: str = "auto"
    max_workers: int = 4
    log_level: str = "INFO"
