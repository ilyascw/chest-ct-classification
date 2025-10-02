"""
Data models для Core Pipeline
"""
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from enum import Enum
import uuid
import json
import numpy as np

class DataType(Enum):
    """Тип медицинских данных"""
    DICOM = "dicom"
    NIFTI = "nifti"
    UNKNOWN = "unknown"

class ProcessingStatus(Enum):
    """Статус обработки"""
    PENDING = "pending"
    PROCESSING = "processing" 
    SUCCESS = "success"
    FAILURE = "failure"

@dataclass
class StudyInfo:
    """Информация об исследовании из архива"""
    path_to_study: str
    study_uid: str
    series_uid: str
    data_type: DataType
    files_count: int
    file_size_mb: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        # Ensure UIDs are strings
        if not self.study_uid or self.study_uid == 'unknown':
            self.study_uid = f"study_{uuid.uuid4().hex[:8]}"
        if not self.series_uid or self.series_uid == 'unknown':
            self.series_uid = f"series_{uuid.uuid4().hex[:8]}"

@dataclass
class VolumeData:
    """Загруженный и обработанный том"""
    path: Path
    volume: np.ndarray
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: np.ndarray
    metadata: Dict[str, Any]
    preprocessing_info: Dict[str, Any]

@dataclass
class FeatureVector:
    """Извлеченные признаки"""
    embeddings: np.ndarray
    extraction_time: float
    model_info: Dict[str, str]
    text_prompt: str

@dataclass
class ClassificationResult:
    """Результат классификации"""
    probability_of_pathology: float
    pathology_prediction: int  # 0 or 1
    confidence_score: float
    model_version: str
    inference_time: float

@dataclass
class ProcessingResult:
    """Финальный результат обработки для Excel отчёта"""
    # Обязательные поля из ТЗ
    path_to_study: str
    study_uid: str
    series_uid: str
    probability_of_pathology: float
    pathology: int
    processing_status: str  # "Success" or "Failure"
    time_of_processing: float  # seconds
    pathology_localization: str  # "x_min,x_max,y_min,y_max,z_min,z_max"
    
    # Дополнительные поля для отладки
    error_details: Optional[str] = None
    volume_shape: Optional[str] = None
    embedding_norm: Optional[float] = None
    processing_steps_completed: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для Excel"""
        result = asdict(self)
        # Убираем None значения для cleaner Excel
        return {k: v for k, v in result.items() if v is not None}

@dataclass
class PipelineConfig:
    """Конфигурация pipeline"""
    # Модели
    ctclip_checkpoint: str
    catboost_model: str
    device: str = 'auto'
    
    # Обработка
    max_workers: int = 4
    timeout_per_study: int = 600  # 10 minutes
    temp_dir: str = "/tmp/ct_processing"
    
    # CT-CLIP параметры
    text_prompt: str = "chest computed tomography scan for pathology detection"
    target_spacing: Tuple[float, float, float] = (0.75, 0.75, 1.5)
    target_shape: Tuple[int, int, int] = (480, 480, 240)
    
    # Классификация
    classification_threshold: float = 0.5
    
    # Логирование
    log_level: str = "INFO"
    save_intermediate_results: bool = True
