"""
Core Pipeline для CT Pathology Detection
"""

import gc
import logging
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ct_pathology.feature_extraction import (
    CTCLIPFeatureExtractor,
    ImageLatentsClassifier,
    create_ct_clip_model_and_extractor,
)
from ct_pathology.model import PathologyClassifier
from ct_pathology.preprocessing import (
    prepare_metadata_for_preprocessing,
    preprocess_nifti,
)

from .data_discovery import DataDiscoveryService
from .data_models import (
    ClassificationResult,
    FeatureVector,
    PipelineConfig,
    ProcessingResult,
    StudyInfo,
    VolumeData,
)
from .volume_loader import VolumeLoaderService


class CTPathologyPipeline:
    """
    Production pipeline для обнаружения патологий на КТ.

    Flow:
    ZIP Archives → Data Discovery → Volume Loading → Preprocessing
    → CT-CLIP Features → CatBoost Classification → Excel Report
    """

    def __init__(self, config: PipelineConfig):
        """
        Инициализация pipeline.

        Args:
            config: Конфигурация pipeline
        """
        if config.device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is unavailable")

        self.config = config
        self.logger = self._setup_logger()
        self.data_discovery = DataDiscoveryService(self.logger)
        self.volume_loader = VolumeLoaderService(self.logger)

        self.ct_clip_model: ImageLatentsClassifier
        self.feature_extractor: CTCLIPFeatureExtractor
        self.classifier: PathologyClassifier

        self._initialize_models()

    def _setup_logger(self) -> logging.Logger:
        """
        Настройка логгера.

        Returns:
            logging.Logger: Настроенный логгер
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_models(self) -> None:
        """
        Инициализация моделей CT-CLIP и CatBoost.

        Raises:
            Exception: При ошибках загрузки моделей
        """
        self.logger.info("Initializing models...")

        try:
            self.ct_clip_model, self.feature_extractor = create_ct_clip_model_and_extractor(
                checkpoint_path=self.config.ct_clip_checkpoint, device=self.config.device
            )
            self.logger.info("CT-CLIP model loaded successfully")

            self.classifier = PathologyClassifier()
            self.classifier.load(self.config.catboost_model)
            self.logger.info("CatBoost model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise

    def process_zip_archives(self, zip_paths: list[str], output_excel: str) -> pd.DataFrame:
        """
        Обрабатывает список ZIP архивов и создаёт Excel отчёт.

        Args:
            zip_paths: Список путей к ZIP архивам
            output_excel: Путь для сохранения Excel отчёта

        Returns:
            pd.DataFrame: Результаты обработки
        """
        if not zip_paths:
            raise ValueError("At least one ZIP archive is required")
        if not output_excel:
            raise ValueError("output_excel must not be empty")

        output_path = Path(output_excel)
        if output_path.suffix.lower() != ".xlsx":
            raise ValueError("output_excel must use the .xlsx extension")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("Processing %d ZIP archives", len(zip_paths))

        all_results: list[ProcessingResult] = []
        archive_errors: list[Exception] = []
        for zip_path in zip_paths:
            try:
                if not Path(zip_path).is_file():
                    raise FileNotFoundError(f"ZIP archive not found: {zip_path}")
                results = self.process_single_zip(zip_path)
                all_results.extend(results)
            except Exception as e:
                archive_errors.append(e)
                self.logger.error("Failed to process ZIP %s: %s", zip_path, e)

        if not all_results:
            if archive_errors:
                raise archive_errors[0]
            raise ValueError("No medical studies were discovered in the archives")

        self.logger.info("Total results: %d", len(all_results))
        report_df = self.generate_excel_report(all_results, str(output_path))

        return report_df

    def process_single_zip(self, zip_path: str) -> list[ProcessingResult]:
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
                zip_path=zip_path, extract_dir=temp_dir
            )

            self.logger.info(f"Found {len(studies)} studies in ZIP")

            results: list[ProcessingResult] = []
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_study = {
                    executor.submit(self.process_single_study, study): study for study in studies
                }

                for future in as_completed(future_to_study):
                    study = future_to_study[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Study {study.study_uid} failed: {e}")
                        error_result = self.create_failed_result(study, str(e), processing_time=0.0)
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
        completed_steps: list[str] = []

        try:
            self.logger.info(f"Processing study: {study.study_uid}, series: {study.series_uid}")

            # Volume loading
            volume_data = self.volume_loader.load_volume_from_study(study)
            completed_steps.append("volume_loading")

            # Preprocessing
            preprocessed_volume = self.preprocess_volume(volume_data)
            del volume_data  # Освобождаем память
            completed_steps.append("preprocessing")

            # Feature extraction
            feature_vector = self.extract_features(preprocessed_volume)
            del preprocessed_volume  # Освобождаем память
            completed_steps.append("feature_extraction")

            # Classification
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
                processing_steps_completed=completed_steps,
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
                study, str(e), processing_time=processing_time, completed_steps=completed_steps
            )

        finally:
            collected = gc.collect()

            # Логируем только если было что-то собрано (избегаем шума в логах)
            if collected > 0:
                self.logger.debug(
                    f"🧹 Freed {collected} unreferenced objects for study {study.study_uid}"
                )

    def preprocess_volume(self, volume_data: VolumeData) -> torch.Tensor:
        """
        Предобработка медицинского тома.

        Args:
            volume_data: Загруженный том с метаданными

        Returns:
            torch.Tensor: Предобработанный том

        Raises:
            RuntimeError: При ошибках предобработки
        """
        try:
            meta_row = prepare_metadata_for_preprocessing(volume_data)
            preprocessed = preprocess_nifti(
                nii_path=volume_data.volume, meta_row=meta_row, Volume=True
            )
            return preprocessed

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            self.logger.error(
                f"CUDA Out of Memory during preprocessing. "
                f"Volume shape: {volume_data.volume.shape}. "
                f"Try reducing batch size or processing on CPU."
            )
            raise RuntimeError(f"GPU memory overflow during preprocessing: {e}") from e

        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {e}") from e

    def extract_features(self, preprocessed_volume: torch.Tensor) -> FeatureVector:
        """
        Извлечение признаков через CT-CLIP.

        Args:
            preprocessed_volume: Предобработанный том

        Returns:
            FeatureVector: 512-мерный вектор эмбеддингов

        Raises:
            RuntimeError: При ошибках извлечения признаков
        """
        try:
            start_time = time.time()

            volume_tensor = preprocessed_volume.float()

            # КРИТИЧНО: Принудительно ставим device в cpu если CUDA недоступна
            if not torch.cuda.is_available():
                volume_tensor = volume_tensor.cpu()

            # Извлечение эмбеддингов
            embeddings = self.feature_extractor.extract_single(
                volume_tensor=volume_tensor, text=self.config.text_prompt, return_numpy=True
            )
            if not isinstance(embeddings, np.ndarray):
                raise TypeError("Expected numpy embeddings from CT-CLIP extractor")

            extraction_time = time.time() - start_time

            feature_vector = FeatureVector(
                embeddings=embeddings,
                extraction_time=extraction_time,
                model_info={"model": "CT-CLIP", "checkpoint": self.config.ct_clip_checkpoint},
                text_prompt=self.config.text_prompt,
            )

            return feature_vector

        except Exception as e:
            raise RuntimeError(f"Feature extraction failed: {e}") from e

    def classify_features(self, feature_vector: FeatureVector) -> ClassificationResult:
        """
        Классификация через CatBoost.

        Args:
            feature_vector: Вектор признаков от CT-CLIP

        Returns:
            ClassificationResult: Результат классификации

        Raises:
            RuntimeError: При ошибках классификации
        """
        try:
            start_time = time.time()

            embeddings_reshaped = feature_vector.embeddings.reshape(1, -1)
            probabilities = np.asarray(self.classifier.predict_proba(embeddings_reshaped))
            if probabilities.shape != (1, 2):
                raise ValueError(f"Classifier returned unexpected shape: {probabilities.shape}")

            probability_of_pathology = float(probabilities[0, 1])
            if not 0.0 <= probability_of_pathology <= 1.0:
                raise ValueError("Classifier probability must be between 0 and 1")
            prediction = int(probability_of_pathology >= 0.5)
            confidence_score = abs(probability_of_pathology - 0.5) * 2.0

            inference_time = time.time() - start_time

            return ClassificationResult(
                probability_of_pathology=probability_of_pathology,
                pathology_prediction=prediction,
                confidence_score=confidence_score,
                model_version=self.config.catboost_model,
                inference_time=inference_time,
            )

        except Exception as e:
            raise RuntimeError(f"Classification failed: {e}") from e

    def create_failed_result(
        self,
        study: StudyInfo,
        error: str,
        processing_time: float = 0.0,
        completed_steps: list[str] | None = None,
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
        return ProcessingResult(
            path_to_study=study.path_to_study,
            study_uid=study.study_uid,
            series_uid=study.series_uid,
            probability_of_pathology=0.0,
            pathology=0,
            processing_status="Failure",
            time_of_processing=processing_time,
            error_details=error,
            processing_steps_completed=completed_steps or [],
        )

    def generate_excel_report(
        self, results: list[ProcessingResult], output_path: str
    ) -> pd.DataFrame:
        """
        Генерирует Excel отчёт с результатами.

        Создаёт три листа:
        - Results: основные результаты
        - Summary: статистика обработки
        - Errors: детали ошибок обработки

        Args:
            results: Список результатов обработки
            output_path: Путь для сохранения Excel

        Returns:
            pd.DataFrame: Датафрейм с результатами
        """
        self.logger.info(f"Generating Excel report: {output_path}")

        # Конвертация результатов в DataFrame
        results_dicts = [r.to_dict() for r in results]
        df = pd.DataFrame(results_dicts)

        # Обязательные колонки согласно ТЗ
        required_columns = [
            "path_to_study",
            "study_uid",
            "series_uid",
            "probability_of_pathology",
            "pathology",
            "processing_status",
            "time_of_processing",
            "error_details",
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        df = df[required_columns]

        # Статистика
        success_results = [r for r in results if r.processing_status == "Success"]
        error_results = [r for r in results if r.processing_status == "Failure"]

        # Сохранение в Excel с несколькими листами
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)

            # Лист Summary
            summary_data = {
                "Metric": [
                    "Total Studies",
                    "Successful",
                    "Failed",
                    "Success Rate (%)",
                    "Pathologies Detected",
                    "Average Processing Time (s)",
                    "Report Generated",
                ],
                "Value": [
                    len(results),
                    len(success_results),
                    len(error_results),
                    f"{len(success_results) / len(results) * 100:.2f}" if results else "0",
                    sum(1 for r in success_results if r.pathology == 1),
                    f"{np.mean([r.time_of_processing for r in results]):.2f}" if results else "0",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Лист Errors (если есть ошибки)
            if error_results:
                error_df = pd.DataFrame([r.to_dict() for r in error_results])
                error_df = error_df.rename(columns={"error_details": "Error Details"})
                error_df.to_excel(writer, sheet_name="Errors", index=False)

        self.logger.info(f"Excel report saved: {output_path}")
        return df
