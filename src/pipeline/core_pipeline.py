
"""
Главный Core Pipeline для CT Pathology Detection
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

# Импортируем ваши существующие модули
from src.preprocessing import preprocess_nifti  # Из вашего preprocessing.py
from src.feature_extraction import create_ctclip_model_and_extractor  # Из feature_extraction.py 
from src.model import PathologyClassifier  # Из model.py

# Наши новые компоненты
from .data_models import (
    StudyInfo, VolumeData, FeatureVector, ClassificationResult, 
    ProcessingResult, PipelineConfig, ProcessingStatus
)
from .data_discovery import DataDiscoveryService
from .volume_loader import VolumeLoaderService

class CTPathologyPipeline:
    """
    Главный production pipeline для обработки медицинских архивов

    Flow: ZIP Archives → Data Discovery → Volume Loading → Preprocessing → 
          CT-CLIP Features → CatBoost Classification → Excel Report
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logger()

        # Инициализируем сервисы
        self.data_discoverer = DataDiscoveryService(self.logger)
        self.volume_loader = VolumeLoaderService(self.logger)

        # Инициализируем AI модели
        self._initialize_models()

        self.logger.info("✅ CTPathologyPipeline initialized successfully")

    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger("CTPathologyPipeline")
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
        """Инициализация моделей"""

        self.logger.info("Initializing steps...")

        try:
            # 1. CT-CLIP Feature Extractor (ваш код)
            self.logger.info("Loading CT-CLIP model...")
            _, self.feature_extractor = create_ctclip_model_and_extractor(
                checkpoint_path=self.config.ctclip_checkpoint,
                device=self.config.device
            )
            self.logger.info("✅ CT-CLIP model loaded")

            # 2. CatBoost Classifier (ваш код)
            self.logger.info("Loading CatBoost classifier...")
            self.classifier = PathologyClassifier(verbose=False)
            self.classifier.load(self.config.catboost_model)
            self.logger.info("✅ CatBoost classifier loaded")

        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize models: {e}")

    def process_zip_archives(self, 
                           zip_paths: List[str],
                           output_excel_path: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Главная функция обработки ZIP архивов

        Returns:
            Tuple[str, Dict]: (путь к Excel файлу, статистика обработки)
        """

        start_time = time.time()
        self.logger.info(f"🚀 Starting processing of {len(zip_paths)} ZIP archives")

        # Создаём временную директорию
        temp_base_dir = tempfile.mkdtemp(prefix="ct_pipeline_")

        try:
            # 1. Извлечение и обнаружение исследований
            all_studies = self._discover_all_studies(zip_paths, temp_base_dir)

            if not all_studies:
                raise RuntimeError("No studies found in provided archives")

            self.logger.info(f"📊 Total studies discovered: {len(all_studies)}")

            # 2. Обработка исследований  
            processing_results = self._process_studies_sequential(all_studies)

            # 3. Генерация Excel отчёта
            excel_path = self._generate_excel_report(processing_results, output_excel_path)

            # 4. Статистика
            total_time = time.time() - start_time
            statistics = self._calculate_statistics(processing_results, total_time)

            self.logger.info(f"✅ Pipeline completed in {total_time:.2f}s")
            return excel_path, statistics

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            # Cleanup
            shutil.rmtree(temp_base_dir, ignore_errors=True)

    def _discover_all_studies(self, zip_paths: List[str], temp_base_dir: str) -> List[StudyInfo]:
        """Обнаружение всех исследований во всех архивах"""

        all_studies = []

        for i, zip_path in enumerate(zip_paths):
            self.logger.info(f"📂 Processing archive {i+1}/{len(zip_paths)}: {Path(zip_path).name}")

            # Создаём отдельную директорию для каждого архива
            extract_dir = Path(temp_base_dir) / f"archive_{i}"
            extract_dir.mkdir(exist_ok=True)

            try:
                studies = self.data_discoverer.discover_studies_in_zip(zip_path, str(extract_dir))
                all_studies.extend(studies)
                self.logger.info(f"  ✅ Found {len(studies)} studies")

            except Exception as e:
                self.logger.error(f"  ❌ Failed to process {zip_path}: {e}")
                continue

        return all_studies

    def _process_studies_sequential(self, studies: List[StudyInfo]) -> List[ProcessingResult]:
        """Sequential обработка исследований (без multiprocessing)"""

        self.logger.info(f"🔄 Sequential processing of {len(studies)} studies...")

        results = []

        from tqdm import tqdm
        for study in tqdm(studies, desc="Processing studies"):
            try:
                result = self._process_single_study(study)
                results.append(result)

                if result.processing_status == "Success":
                    self.logger.info(f"✅ {study.study_uid}: {result.probability_of_pathology:.3f} ({result.time_of_processing:.1f}s)")
                else:
                    self.logger.warning(f"⚠️ {study.study_uid}: {result.processing_status}")

            except Exception as e:
                self.logger.error(f"❌ {study.study_uid}: {e}")
                results.append(self._create_failed_result(study, str(e)))

        return results

    def _process_single_study(self, study: StudyInfo) -> ProcessingResult:
        """Обработка одного исследования"""

        start_time = time.time()
        processing_steps_completed = []

        try:
            self.logger.debug(f"🔄 Processing {study.study_uid}")

            # 1. Volume Loading
            volume_data = self.volume_loader.load_volume_from_study(study)
            processing_steps_completed.append("volume_loading")

            # 2. Preprocessing 
            preprocessed_volume = self._preprocess_volume(volume_data)
            processing_steps_completed.append("preprocessing")

            # 3. Feature Extraction
            feature_vector = self._extract_features(preprocessed_volume)
            processing_steps_completed.append("feature_extraction")

            # 4. Classification
            classification_result = self._classify_features(feature_vector)
            processing_steps_completed.append("classification")

            # 5. Результат
            processing_time = time.time() - start_time
            pathology_localization = self._generate_pathology_localization(
                volume_data.volume.shape, 
                classification_result.probability_of_pathology
            )

            result = ProcessingResult(
                path_to_study=study.path_to_study,
                study_uid=study.study_uid,
                series_uid=study.series_uid,
                probability_of_pathology=classification_result.probability_of_pathology,
                pathology=classification_result.pathology_prediction,
                processing_status="Success",
                time_of_processing=processing_time,
                pathology_localization=pathology_localization,
                volume_shape=str(volume_data.volume.shape),
                embedding_norm=float(np.linalg.norm(feature_vector.embeddings)),
                processing_steps_completed=processing_steps_completed
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ {study.study_uid} failed: {e}")
            return self._create_failed_result(study, str(e), processing_time, processing_steps_completed)

    def _create_failed_result(self, study: StudyInfo, error: str, 
                             processing_time: float = 0.0,
                             completed_steps: List[str] = None) -> ProcessingResult:
        """Создание failed result"""
        return ProcessingResult(
            path_to_study=study.path_to_study,
            study_uid=study.study_uid,
            series_uid=study.series_uid,
            probability_of_pathology=0.0,
            pathology=0,
            processing_status="Failure",
            time_of_processing=processing_time,
            pathology_localization="0,0,0,0,0,0",
            error_details=error,
            processing_steps_completed=completed_steps or []
        )

    def _preprocess_volume(self, volume_data: VolumeData) -> np.ndarray:
        """Preprocessing используя ваш preprocessing.py"""

        try:
            # Создаём mock DataFrame row для совместимости с вашим кодом
            mock_metadata = pd.Series({
                'RescaleSlope': float(volume_data.metadata.get('slope', 1.0)),
                'RescaleIntercept': float(volume_data.metadata.get('intercept', 0.0)),
                'XYSpacing': str(list(volume_data.metadata.get(spacing, "[1.0, 1.0]")[:2])),
                'ZSpacing': float(volume_data.spacing[2]) if len(volume_data.spacing) > 2 else 1.0
            })

            preprocessed = preprocess_nifti(volume_data.path, mock_metadata, Volume=True)

            if preprocessed is None:
                raise ValueError("Preprocessing returned None")

            return preprocessed

        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}")

    def _extract_features(self, volume: np.ndarray) -> FeatureVector:
        """Feature extraction используя ваш CT-CLIP"""

        start_time = time.time()

        try:
            # Используем ваш feature extractor
            embeddings = self.feature_extractor.extract_single(
                volume,
                text=self.config.text_prompt,
                return_numpy=True
            )

            if embeddings is None:
                raise ValueError("Feature extractor returned None")

            extraction_time = time.time() - start_time

            return FeatureVector(
                embeddings=embeddings,
                extraction_time=extraction_time,
                model_info={
                    'model_type': 'CT-CLIP',
                    'checkpoint': self.config.ctclip_checkpoint
                },
                text_prompt=self.config.text_prompt
            )

        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {e}")

    def _classify_features(self, feature_vector: FeatureVector) -> ClassificationResult:
        """Classification используя ваш CatBoost"""

        start_time = time.time()

        try:
            # Reshape для CatBoost (1, 512)
            embeddings_reshaped = feature_vector.embeddings.reshape(1, -1)

            # Получаем вероятности
            probabilities = self.classifier.predict_proba(embeddings_reshaped)
            probability_of_pathology = float(probabilities[0, 1])  # класс 1 = патология

            # Предсказание по threshold
            prediction = int(probability_of_pathology >= self.config.classification_threshold)

            # Confidence score (расстояние от threshold)
            confidence_score = abs(probability_of_pathology - 0.5) * 2  # 0-1 scale

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

#     def _generate_pathology_localization(self, volume_shape: tuple, probability: float) -> str:
#         """
#         Генерация локализации патологии
#         TODO: Можно улучшить через GradCAM или attention maps
#         """

#         # Пока используем заглушку - возвращаем весь том или центральную область
#         if probability > 0.7:  # High confidence - предполагаем центральную область
#             if len(volume_shape) >= 3:
#                 h, w, d = volume_shape[:3]
#                 margin_h, margin_w, margin_d = h//4, w//4, d//4

#                 x_min, x_max = margin_w, w - margin_w
#                 y_min, y_max = margin_h, h - margin_h  
#                 z_min, z_max = margin_d, d - margin_d
#             else:
#                 x_min, x_max = 0, 100
#                 y_min, y_max = 0, 100
#                 z_min, z_max = 0, 100
#         else:
#             # Low confidence или no pathology - весь том
#             if len(volume_shape) >= 3:
#                 h, w, d = volume_shape[:3]
#                 x_min, x_max = 0, w
#                 y_min, y_max = 0, h
#                 z_min, z_max = 0, d
#             else:
#                 x_min, x_max = 0, 100
#                 y_min, y_max = 0, 100
#                 z_min, z_max = 0, 100

#         return f"{x_min},{x_max},{y_min},{y_max},{z_min},{z_max}"

    def _generate_excel_report(self, 
                             results: List[ProcessingResult], 
                             output_path: Optional[str] = None) -> str:
        """Генерация Excel отчёта в формате ТЗ"""

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ct_pathology_results_{timestamp}.xlsx"

        self.logger.info(f"📊 Generating Excel report: {output_path}")

        # Конвертируем в DataFrame
        df_data = [result.to_dict() for result in results]
        df = pd.DataFrame(df_data)

        # Переупорядочиваем колонки согласно ТЗ
        required_columns = [
            'path_to_study', 'study_uid', 'series_uid', 
            'probability_of_pathology', 'pathology',
            'processing_status', 'time_of_processing', 'pathology_localization'
        ]

        # Добавляем дополнительные колонки если есть
        additional_columns = [col for col in df.columns if col not in required_columns]
        final_columns = required_columns + additional_columns

        df = df.reindex(columns=final_columns)

        # Сохраняем в Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main results
            df.to_excel(writer, sheet_name='Results', index=False)

            # Summary statistics
            summary_df = self._create_summary_dataframe(results)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Error details (если есть)
            error_results = [r for r in results if r.processing_status == "Failure"]
            if error_results:
                error_df = pd.DataFrame([r.to_dict() for r in error_results])
                error_df.to_excel(writer, sheet_name='Errors', index=False)

        self.logger.info(f"✅ Excel report saved: {output_path}")
        return output_path

    def _create_summary_dataframe(self, results: List[ProcessingResult]) -> pd.DataFrame:
        """Создание summary статистики"""

        total = len(results)
        successful = len([r for r in results if r.processing_status == "Success"])
        failed = len([r for r in results if r.processing_status == "Failure"])

        pathology_detected = len([r for r in results if r.pathology == 1])
        normal_studies = len([r for r in results if r.pathology == 0])

        processing_times = [r.time_of_processing for r in results if r.processing_status == "Success"]
        avg_processing_time = np.mean(processing_times) if processing_times else 0

        probabilities = [r.probability_of_pathology for r in results if r.processing_status == "Success"]
        avg_probability = np.mean(probabilities) if probabilities else 0

        summary_data = {
            'Metric': [
                'Total Studies',
                'Successful Processing',
                'Failed Processing',
                'Studies with Pathology',
                'Normal Studies',
                'Average Processing Time (s)',
                'Average Pathology Probability',
                'Processing Success Rate (%)',
                'Report Generated At'
            ],
            'Value': [
                total,
                successful,
                failed,
                pathology_detected,
                normal_studies,
                f"{avg_processing_time:.2f}",
                f"{avg_probability:.4f}",
                f"{(successful/total*100):.1f}" if total > 0 else "0.0",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        }

        return pd.DataFrame(summary_data)

    def _calculate_statistics(self, results: List[ProcessingResult], total_time: float) -> Dict[str, Any]:
        """Расчёт детальной статистики"""

        total = len(results)
        successful = len([r for r in results if r.processing_status == "Success"])
        failed = total - successful

        pathology_detected = len([r for r in results if r.pathology == 1])
        normal_studies = len([r for r in results if r.pathology == 0])

        processing_times = [r.time_of_processing for r in results if r.processing_status == "Success"]

        statistics = {
            'total_studies': total,
            'successful_studies': successful,
            'failed_studies': failed,
            'success_rate': successful / total if total > 0 else 0,
            'pathology_detected': pathology_detected,
            'normal_studies': normal_studies,
            'pathology_rate': pathology_detected / successful if successful > 0 else 0,
            'total_processing_time': total_time,
            'average_time_per_study': np.mean(processing_times) if processing_times else 0,
            'min_processing_time': np.min(processing_times) if processing_times else 0,
            'max_processing_time': np.max(processing_times) if processing_times else 0,
            'studies_per_minute': total / (total_time / 60) if total_time > 0 else 0
        }

        return statistics
