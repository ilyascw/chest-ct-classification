"""
Service для обнаружения медицинских исследований в ZIP архивах.
Реализует жёсткую валидацию DICOM UIDs согласно требованиям ТЗ.
"""

import logging
import zipfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import pydicom
import nibabel as nib
import os

from .data_models import StudyInfo, DataType


class DataDiscoveryService:
    """
    Сервис для обнаружения медицинских данных в ZIP архивах.
    
    Поддерживает:
    - DICOM серии с обязательной валидацией StudyInstanceUID и SeriesInstanceUID
    - NIfTI файлы
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Инициализация сервиса.
        
        Args:
            logger: Логгер для вывода сообщений
        """
        self.logger = logger or logging.getLogger(__name__)

    def discover_studies_in_zip(
    self,
    zip_path: str,
    extract_dir: str
    ) -> List[StudyInfo]:
        """
        Обнаруживает все медицинские исследования в ZIP архиве.
        
        Args:
            zip_path: Путь к ZIP архиву
            extract_dir: Директория для распаковки
        
        Returns:
            List[StudyInfo]: Список найденных исследований (без дублей)
            
        Raises:
            Exception: При ошибках распаковки или сканирования
        """
        self.logger.info(f"Discovering studies in {zip_path}")
        
        try:
            # Распаковка ZIP архива
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Используем существующий discover_inputs_robust из ct_preprocessor
            from src.CTPreprocessor.ct_preprocessor import discover_inputs_robust
            
            raw_inputs = discover_inputs_robust(Path(extract_dir), self.logger)
            
            studies = []
            seen_series = set()  # Дедупликация по (study_uid, series_uid)
            
            for raw_input in raw_inputs:
                study_info = self._convert_raw_input_to_study_info(raw_input)
                if study_info:
                    # Создаём уникальный ключ на основе study_uid и series_uid
                    series_key = (study_info.study_uid, study_info.series_uid)
                    
                    if series_key not in seen_series:
                        studies.append(study_info)
                        seen_series.add(series_key)
                        self.logger.debug(f"Added unique series: {series_key}")
                    else:
                        self.logger.warning(
                            f"Duplicate series detected and skipped: "
                            f"StudyUID={study_info.study_uid}, SeriesUID={study_info.series_uid}"
                        )
            
            self.logger.info(f"Found {len(studies)} unique studies in {zip_path}")
            return studies
            
        except Exception as e:
            self.logger.error(f"Failed to discover studies in {zip_path}: {e}")
            self.logger.debug(traceback.format_exc())
            raise


    def _convert_raw_input_to_study_info(
        self,
        raw_input: Dict[str, Any]
    ) -> Optional[StudyInfo]:
        """
        Конвертирует результат discover_inputs_robust в StudyInfo.
        
        Для DICOM:
        - Принудительно читает StudyInstanceUID (0020,000D) и SeriesInstanceUID (0020,000E)
        - При отсутствии любого из UID создаёт Failure-запись
        
        Args:
            raw_input: Словарь с информацией об обнаруженном исследовании
        
        Returns:
            StudyInfo или None при критической ошибке конвертации
        """
        try:
            input_type = raw_input.get('type', 'unknown')
            
            if input_type == 'dicom_dir':
                return self._process_dicom_input(raw_input)
            elif input_type == 'nifti':
                return self._process_nifti_input(raw_input)
            else:
                self.logger.warning(f"Unknown input type: {input_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to convert raw input: {e}")
            self.logger.debug(traceback.format_exc())
            return None

    def _process_dicom_input(self, raw_input: Dict[str, Any]) -> Optional[StudyInfo]:
        """
        Обработка DICOM input с жёсткой валидацией UIDs.
        
        Читает StudyInstanceUID (0020,000D) и SeriesInstanceUID (0020,000E)
        из первого валидного DICOM-файла. При отсутствии любого тега
        возвращает None, чтобы pipeline создал Failure-запись.
        
        Args:
            raw_input: Словарь с информацией о DICOM серии
        
        Returns:
            StudyInfo с валидными UIDs или None при отсутствии тегов
        """
        dicom_dir = raw_input.get('path', '')
        file_list = raw_input.get('file_list', [])
        
        study_uid: Optional[str] = None
        series_uid: Optional[str] = None
        metadata: Dict[str, Any] = {'input_type': 'dicom_dir'}
        
        # Поиск первого читаемого DICOM для извлечения UIDs
        for dicom_file in file_list:
            try:
                file_path = Path(dicom_dir) / dicom_file if not Path(dicom_file).is_absolute() else Path(dicom_file)
                
                self.logger.debug(f"Reading DICOM UIDs from: {file_path}")
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=False)
                
                # Извлечение обязательных DICOM тегов
                study_uid = getattr(ds, 'StudyInstanceUID', None)
                series_uid = getattr(ds, 'SeriesInstanceUID', None)
                
                if study_uid:
                    study_uid = str(study_uid).strip()
                if series_uid:
                    series_uid = str(series_uid).strip()
                
                # Дополнительные метаданные для отладки
                metadata.update({
                    'patient_id': str(getattr(ds, 'PatientID', 'N/A')),
                    'study_date': str(getattr(ds, 'StudyDate', 'N/A')),
                    'modality': str(getattr(ds, 'Modality', 'N/A')),
                    'manufacturer': str(getattr(ds, 'Manufacturer', 'N/A')),
                    'file_list': [str(f) for f in file_list]
                })
                
                break  # Нашли первый валидный DICOM
                
            except Exception as e:
                self.logger.warning(f"Failed to read DICOM {dicom_file}: {e}")
                continue
        
        # Валидация UIDs: оба тега обязательны
        if not study_uid or not series_uid:
            error_msg = []
            if not study_uid:
                error_msg.append("StudyInstanceUID (0020,000D) is missing")
            if not series_uid:
                error_msg.append("SeriesInstanceUID (0020,000E) is missing")
            
            self.logger.error(
                f"DICOM validation failed for {dicom_dir}: {'; '.join(error_msg)}. "
                f"This study will be marked as Failure."
            )
            
            # Возвращаем None, чтобы core_pipeline создал Failure-запись
            # с подробным error_details
            return None
        
        # Подсчёт размера файлов
        total_size = 0
        for dicom_file in file_list:
            file_path = Path(dicom_dir) / dicom_file if not Path(dicom_file).is_absolute() else Path(dicom_file)
            if file_path.exists():
                total_size += file_path.stat().st_size
        
        self.logger.info(
            f"DICOM study discovered: StudyUID={study_uid}, SeriesUID={series_uid}, "
            f"Files={len(file_list)}"
        )
        
        try:
            return StudyInfo(
                path_to_study=str(dicom_dir),
                study_uid=study_uid,
                series_uid=series_uid,
                data_type=DataType.DICOM,
                files_count=len(file_list),
                file_size_mb=total_size / (1024 * 1024),
                metadata=metadata
            )
        except ValueError as ve:
            # StudyInfo.__post_init__ выбросил исключение при валидации
            self.logger.error(f"StudyInfo validation failed: {ve}")
            return None

    def _process_nifti_input(self, raw_input: Dict[str, Any]) -> Optional[StudyInfo]:
        """
        Обработка NIfTI input.
        
        Для NIfTI генерируются synthetic UIDs на основе имени файла.
        
        Args:
            raw_input: Словарь с информацией о NIfTI файле
        
        Returns:
            StudyInfo с generated UIDs
        """
        nifti_path = raw_input.get('path', '')
        nifti_file = Path(nifti_path)
        
        # Извлечение метаданных из NIfTI
        metadata: Dict[str, Any] = {'input_type': 'nifti'}
        try:
            img = nib.load(str(nifti_path))
            header = img.header
            metadata.update({
                'shape': str(img.shape),
                'dtype': str(img.get_fdata().dtype),
                'pixdim': str(header.get('pixdim')),
                'qform_code': str(header.get('qform_code')),
                'sform_code': str(header.get('sform_code')),
                'descrip': str(header.get('descrip', b'').decode('utf-8', errors='ignore'))
            })
        except Exception as e:
            self.logger.warning(f"Failed to read NIfTI metadata from {nifti_path}: {e}")
        
        # UIDs на основе имени файла (генерируются в StudyInfo.__post_init__)
        file_stem = nifti_file.stem.replace('.nii', '')
        study_uid = f"nifti_study_{file_stem}"
        series_uid = f"nifti_series_{file_stem}"
        
        # Размер файла
        file_size_mb = nifti_file.stat().st_size / (1024 * 1024) if nifti_file.exists() else 0.0
        
        try:
            return StudyInfo(
                path_to_study=str(nifti_path),
                study_uid=study_uid,
                series_uid=series_uid,
                data_type=DataType.NIFTI,
                files_count=1,
                file_size_mb=file_size_mb,
                metadata=metadata
            )
        except ValueError as ve:
            self.logger.error(f"StudyInfo validation failed for NIfTI: {ve}")
            return None
