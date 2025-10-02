"""
Core Pipeline для CT Pathology Detection
"""
import logging
import time
import tempfile
import shutil
import uuid
import zipfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from datetime import datetime
import numpy as np

from src.preprocessing import preprocess_nifti, prepare_metadata_for_preprocessing
from src.feature_extraction import create_ct_clip_model_and_extractor
from src.model import PathologyClassifier

from .data_models import (
    StudyInfo, VolumeData, FeatureVector, ClassificationResult, 
    ProcessingResult, PipelineConfig, ProcessingStatus
)
from .data_discovery import DataDiscoveryService
from .volume_loader import VolumeLoaderService


class CTPathologyPipeline:
    """
    Production pipeline для обнаружения патологий на КТ.
    
    Flow:
    ZIP Archives → Data Discovery → Volume Loading → Preprocessing 
    → CT-CLIP Features → CatBoost Classification → Excel Report
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        self.data_discovery = DataDiscoveryService(self.logger)
        self.volume_loader = VolumeLoaderService(self.logger)
        
        self.ct_clip_model = None
        self.feature_extractor = None
        self.classifier = None
        
        self._initialize_models()
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_models(self):
        """Инициализация моделей CT-CLIP и CatBoost"""
        self.logger.info("Initializing models...")
        
        try:
            self.ct_clip_model, self.feature_extractor = create_ct_clip_model_and_extractor(
                checkpoint_path=self.config.ct_clip_checkpoint,
                device=self.config.device
            )
            self.logger.info("CT-CLIP model loaded successfully")
            
            self.classifier = PathologyClassifier()
            self.classifier.load(self.config.catboost_model)
            self.logger.info("CatBoost model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def process_zip_archives(
        self, 
        zip_paths: List[str], 
        output_excel: str
    ) -> pd.DataFrame:
        """
        Обрабатывает список ZIP архивов и создаёт Excel отчёт.
        
        Args:
            zip_paths: Список путей к ZIP архивам
            output_excel: Путь для сохранения Excel отчёта
            
        Returns:
            pd.DataFrame: Результаты обработки
        """
        self.logger.info(f"Processing {len(zip_paths)} ZIP archives")
        
        all_results = []
        
        for zip_path in zip_paths:
            try:
                results = self.process_single_zip(zip_path)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Failed to process ZIP {zip_path}: {e}")
        
        self.logger.info(f"Total results: {len(all_results)}")
        
        report_df = self.generate_excel_report(all_results, output_excel)
        
        return report_df
    
    def process_single_zip(self, zip_path: str) -> List[ProcessingResult]:
        """
        Обрабатывает один ZIP архив.
        
        Args:
            zip_path: Путь к ZIP архиву
            
        Returns:
            List[ProcessingResult]: Результаты обработки всех исследований
        """
        self.logger.info(f"Processing ZIP: {zip_path}")
        
        temp_dir = tempfile.mkdtemp(prefix="ct_pathology_")
        
        try:
            studies = self.data_discovery.discover_studies_in_zip(
                zip_path=zip_path,
                extract_dir=temp_dir
            )
            
            self.logger.info(f"Found {len(studies)} studies in ZIP")
            
            results = []
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_study = {
                    executor.submit(self.process_single_study, study): study 
                    for study in studies
                }
                
                for future in as_completed(future_to_study):
                    study = future_to_study[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Study {study.study_uid} failed: {e}")
                        error_result = self.create_failed_result(
                            study, 
                            str(e), 
                            processing_time=0.0
                        )
                        results.append(error_result)
            
            return results
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def process_single_study(self, study: StudyInfo) -> ProcessingResult:
        """
        Обрабатывает одно исследование.
        
        Args:
            study: Информация об исследовании
            
        Returns:
            ProcessingResult: Результат обработки
        """
        start_time = time.time()
        completed_steps = []
        
        try:
            self.logger.info(f"Processing study: {study.study_uid}, series: {study.series_uid}")
            
            volume_data = self.volume_loader.load_volume_from_study(study)
            completed_steps.append("volume_loading")
            
            preprocessed_volume = self.preprocess_volume(volume_data)
            completed_steps.append("preprocessing")
            
            feature_vector = self.extract_features(preprocessed_volume)
            completed_steps.append("feature_extraction")
            
            classification_result = self.classify_features(feature_vector)
            completed_steps.append("classification")
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                path_to_study=study.path_to_study,
                study_uid=study.study_uid,
                series_uid=study.series_uid,
                probability_of_pathology=classification_result.probability_of_pathology,
                pathology=classification_result.pathology_prediction,
                processing_status="Success",
                time_of_processing=processing_time,
                error_details="",
                processing_steps_completed=completed_steps
            )
            
            self.logger.info(
                f"Study {study.study_uid} completed in {processing_time:.2f}s: "
                f"pathology={result.pathology}, prob={result.probability_of_pathology:.3f}"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Study {study.study_uid} failed: {e}")
            return self.create_failed_result(
                study, 
                str(e), 
                processing_time=processing_time,
                completed_steps=completed_steps
            )
    
    def preprocess_volume(self, volume_data: VolumeData) -> np.ndarray:
        """
        Предобработка медицинского тома.
        
        Использует prepare_metadata_for_preprocessing для адаптации
        метаданных VolumeData в формат для preprocess_nifti.
        
        Args:
            volume_data: Загруженный том с метаданными
            
        Returns:
            np.ndarray: Предобработанный том
        """
        try:
            meta_row = prepare_metadata_for_preprocessing(volume_data)
            
            preprocessed = preprocess_nifti(
                nii_path=volume_data.volume,
                meta_row=meta_row,
                Volume=True
            )
            
            return preprocessed
        
        except torch.cuda.OutOfMemoryError as e:
            # Очищаем GPU память
            torch.cuda.empty_cache()

            self.logger.error(
                f"CUDA Out of Memory during preprocessing. "
                f"Volume shape: {volume_data.volume.shape}. "
                f"Try reducing batch size or processing on CPU."
            )
            raise RuntimeError(f"GPU memory overflow during preprocessing: {e}")    
        
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}")
    
    def extract_features(self, preprocessed_volume: np.ndarray) -> FeatureVector:
        """
        Извлечение признаков через CT-CLIP.

        Args:
            preprocessed_volume: Предобработанный том

        Returns:
            FeatureVector: 512-мерный вектор эмбеддингов
        """
        try:
            start_time = time.time()

            # Конвертируем numpy в torch tensor если нужно
            if isinstance(preprocessed_volume, np.ndarray):
                volume_tensor = torch.from_numpy(preprocessed_volume).float()
            else:
                volume_tensor = preprocessed_volume

            # Вызываем extract_single вместо extract_features
            embeddings = self.feature_extractor.extract_single(
                volume_tensor=volume_tensor,
                text=self.config.text_prompt,
                return_numpy=True
            )

            extraction_time = time.time() - start_time

            feature_vector = FeatureVector(
                embeddings=embeddings,
                extraction_time=extraction_time,
                model_info={
                    'model': 'CT-CLIP',
                    'checkpoint': self.config.ct_clip_checkpoint
                },
                text_prompt=self.config.text_prompt
            )

            return feature_vector
        
        except torch.cuda.OutOfMemoryError as e:
            # Очищаем GPU память
            torch.cuda.empty_cache()

            self.logger.error(
                f"CUDA Out of Memory during feature extraction. "
                f"Try reducing batch size or processing on CPU."
            )
            raise RuntimeError(f"GPU memory overflow during feature extraction: {e}")
            
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {e}")

    
    def classify_features(self, feature_vector: FeatureVector) -> ClassificationResult:
        """
        Классификация через CatBoost.
        
        Args:
            feature_vector: Вектор признаков от CT-CLIP
            
        Returns:
            ClassificationResult: Результат классификации
        """
        try:
            start_time = time.time()
            
            embeddings_reshaped = feature_vector.embeddings.reshape(1, -1)
            probabilities = self.classifier.predict_proba(embeddings_reshaped)
            
            probability_of_pathology = float(probabilities[0, 1])
            
            prediction = int(probability_of_pathology >= 0.5)
            
            confidence_score = abs(probability_of_pathology - 0.5) * 2.0
            
            inference_time = time.time() - start_time
            
            return ClassificationResult(
                probability_of_pathology=probability_of_pathology,
                pathology_prediction=prediction,
                confidence_score=confidence_score,
                model_version=self.config.catboost_model,
                inference_time=inference_time
            )
            
        except Exception as e:
            raise RuntimeError(f"Classification failed: {e}")
    
    def create_failed_result(
        self, 
        study: StudyInfo, 
        error: str, 
        processing_time: float = 0.0,
        completed_steps: Optional[List[str]] = None
    ) -> ProcessingResult:
        """
        Создаёт ProcessingResult для ошибочного исследования.
        
        Args:
            study: Информация об исследовании
            error: Описание ошибки
            processing_time: Время обработки до ошибки
            completed_steps: Список завершённых этапов
            
        Returns:
            ProcessingResult: Результат с ошибкой
        """
        full_traceback = traceback.format_exc()
        
        return ProcessingResult(
            path_to_study=study.path_to_study,
            study_uid=study.study_uid,
            series_uid=study.series_uid,
            probability_of_pathology=0.0,
            pathology=0,
            processing_status="Failure",
            time_of_processing=processing_time,
            error_details=full_traceback,
            processing_steps_completed=completed_steps or []
        )
    
    def generate_excel_report(
        self, 
        results: List[ProcessingResult], 
        output_path: str
    ) -> pd.DataFrame:
        """
        Генерирует Excel отчёт с результатами.
        
        Создаёт три листа:
        - Results: основные результаты
        - Summary: статистика обработки
        - Errors: детальные логи ошибок с traceback
        
        Args:
            results: Список результатов обработки
            output_path: Путь для сохранения Excel
            
        Returns:
            pd.DataFrame: Датафрейм с результатами
        """
        self.logger.info(f"Generating Excel report: {output_path}")
        
        results_dicts = [r.to_dict() for r in results]
        df = pd.DataFrame(results_dicts)
        
        required_columns = [
            'path_to_study',
            'study_uid',
            'series_uid',
            'probability_of_pathology',
            'pathology',
            'processing_status',
            'time_of_processing',
            'error_details'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        df = df[required_columns]
        
        success_results = [r for r in results if r.processing_status == "Success"]
        error_results = [r for r in results if r.processing_status == "Failure"]
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            summary_data = {
                'Metric': [
                    'Total Studies',
                    'Successful',
                    'Failed',
                    'Success Rate (%)',
                    'Pathologies Detected',
                    'Average Processing Time (s)',
                    'Report Generated'
                ],
                'Value': [
                    len(results),
                    len(success_results),
                    len(error_results),
                    f"{len(success_results) / len(results) * 100:.2f}" if results else "0",
                    sum(1 for r in success_results if r.pathology == 1),
                    f"{np.mean([r.time_of_processing for r in results]):.2f}" if results else "0",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            if error_results:
                error_df = pd.DataFrame([r.to_dict() for r in error_results])
                error_df = error_df.rename(columns={'error_details': 'Full Traceback'})
                error_df.to_excel(writer, sheet_name='Errors', index=False)
        
        self.logger.info(f"Excel report saved: {output_path}")
        
        return df
