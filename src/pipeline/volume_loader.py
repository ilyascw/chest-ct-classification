"""
Service для загрузки медицинских томов из различных форматов.
Включает проверку консистентности DICOM UIDs по всей серии.
"""

import logging
import time
import traceback
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import pydicom

from .data_models import StudyInfo, VolumeData, DataType


class VolumeLoaderService:
    """
    Сервис для загрузки медицинских томов.
    
    Поддерживает:
    - DICOM серии через robust_load_dicom_volume с валидацией UIDs
    - NIfTI файлы через nibabel
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Инициализация сервиса.
        
        Args:
            logger: Логгер для вывода сообщений
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_volume_from_study(self, study_info: StudyInfo) -> VolumeData:
        """
        Загружает медицинский том на основе StudyInfo.
        
        Args:
            study_info: Информация об исследовании
        
        Returns:
            VolumeData: Загруженный том с метаданными
        
        Raises:
            ValueError: Если тип данных не поддерживается или UIDs некорректны
            RuntimeError: При ошибках загрузки
        """
        start_time = time.time()
        self.logger.debug(f"Loading volume: StudyUID={study_info.study_uid}, SeriesUID={study_info.series_uid}")
        
        try:
            if study_info.data_type == DataType.DICOM:
                volume_data = self._load_dicom_volume(study_info)
            elif study_info.data_type == DataType.NIFTI:
                volume_data = self._load_nifti_volume(study_info)
            else:
                raise ValueError(f"Unsupported data type: {study_info.data_type}")
            
            loading_time = time.time() - start_time
            self.logger.debug(f"Volume loaded in {loading_time:.2f}s: shape={volume_data.volume.shape}")
            
            return volume_data
            
        except Exception as e:
            loading_time = time.time() - start_time
            self.logger.error(f"Failed to load volume ({loading_time:.2f}s): {e}")
            self.logger.debug(traceback.format_exc())
            raise

    def _load_dicom_volume(self, study_info: StudyInfo) -> VolumeData:
        """
        Загружает DICOM том через robust_load_dicom_volume.
        
        Дополнительно проверяет консистентность StudyInstanceUID и SeriesInstanceUID
        по подвыборке файлов серии для выявления смешанных/поврежденных данных.
        
        Args:
            study_info: Информация о DICOM исследовании
        
        Returns:
            VolumeData: Загруженный DICOM том
        
        Raises:
            ValueError: При рассинхроне UIDs в серии
        """
        from src.CTPreprocessor.ct_preprocessor import robust_load_dicom_volume
        
        dicom_dir = Path(study_info.path_to_study)
        file_list = None
        
        # Извлечение списка файлов из metadata (если есть)
        if 'file_list' in study_info.metadata:
            file_list_str = study_info.metadata['file_list']
            file_list = [Path(f) for f in file_list_str]
            self.logger.debug(f"Using provided file_list with {len(file_list)} files")
        
        # Проверка консистентности UIDs по подвыборке файлов
        self._validate_dicom_uids_consistency(study_info, file_list)
        
        # Загрузка тома
        volume, metadata = robust_load_dicom_volume(
            dicom_dir=dicom_dir,
            file_list=file_list,
            logger=self.logger
        )
        
        # Извлечение spacing и других метаданных
        spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
        origin = metadata.get('origin', (0.0, 0.0, 0.0))
        direction = metadata.get('direction', np.eye(3).flatten())
        
        # Валидация spacing
        if any(s <= 0 for s in spacing):
            raise ValueError(
                f"Invalid spacing detected: {spacing}. "
                f"All spacing values must be positive."
            )
        
        if any(s > 100 for s in spacing):
            self.logger.warning(
                f"Unusually large spacing detected: {spacing} mm. "
                f"This may indicate incorrect DICOM metadata."
            )
        
        # Приведение direction к матрице 3x3
        if isinstance(direction, (list, tuple)):
            direction = np.array(direction).reshape(3, 3)
        
        volume_data = VolumeData(
            path=dicom_dir,
            volume=volume,
            spacing=spacing,
            origin=origin,
            direction=direction,
            metadata=metadata,
            preprocessing_info={}
        )
        
        return volume_data

    def _validate_dicom_uids_consistency(
        self,
        study_info: StudyInfo,
        file_list: Optional[List[Path]] = None
    ) -> None:
        """
        Проверяет консистентность StudyInstanceUID и SeriesInstanceUID по серии.
        
        Считывает UIDs из подвыборки файлов (первый, средний, последний) и
        сравнивает с ожидаемыми значениями из study_info. При рассинхроне
        выбрасывает ValueError.
        
        Args:
            study_info: Информация об исследовании с эталонными UIDs
            file_list: Список файлов серии (если None, сканирует директорию)
        
        Raises:
            ValueError: При обнаружении рассинхрона UIDs в серии
        """
        dicom_dir = Path(study_info.path_to_study)
        
        # Формирование списка файлов для проверки
        if file_list is None:
            # Сканируем директорию рекурсивно
            all_files = sorted([f for f in dicom_dir.rglob("*") if f.is_file()])
        else:
            all_files = [
                dicom_dir / f if not f.is_absolute() else f
                for f in file_list
            ]
        
        if not all_files:
            self.logger.warning(f"No files found for UID validation in {dicom_dir}")
            return
        
        # Выбираем подвыборку файлов: первый, средний, последний
        sample_indices = [0]
        if len(all_files) > 2:
            sample_indices.append(len(all_files) // 2)
        if len(all_files) > 1:
            sample_indices.append(len(all_files) - 1)
        
        sample_files = [all_files[i] for i in sample_indices]
        
        self.logger.debug(
            f"Validating DICOM UIDs consistency: checking {len(sample_files)} files "
            f"out of {len(all_files)}"
        )
        
        inconsistent_files = []
        for file_path in sample_files:
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=False)
                
                file_study_uid = str(getattr(ds, 'StudyInstanceUID', '')).strip()
                file_series_uid = str(getattr(ds, 'SeriesInstanceUID', '')).strip()
                
                # Сравнение с эталонными UIDs
                if file_study_uid != study_info.study_uid:
                    inconsistent_files.append({
                        'file': file_path.name,
                        'expected_study_uid': study_info.study_uid,
                        'actual_study_uid': file_study_uid,
                        'mismatch': 'StudyInstanceUID'
                    })
                
                if file_series_uid != study_info.series_uid:
                    inconsistent_files.append({
                        'file': file_path.name,
                        'expected_series_uid': study_info.series_uid,
                        'actual_series_uid': file_series_uid,
                        'mismatch': 'SeriesInstanceUID'
                    })
                    
            except Exception as e:
                self.logger.warning(f"Failed to validate UIDs in {file_path.name}: {e}")
                continue
        
        # При обнаружении рассинхрона выбрасываем исключение
        if inconsistent_files:
            error_details = "\n".join([
                f"  - {item['file']}: {item['mismatch']} mismatch "
                f"(expected={item.get('expected_study_uid') or item.get('expected_series_uid')}, "
                f"actual={item.get('actual_study_uid') or item.get('actual_series_uid')})"
                for item in inconsistent_files
            ])
            raise ValueError(
                f"DICOM UIDs consistency check failed for series {study_info.series_uid}:\n"
                f"{error_details}\n"
                f"This indicates mixed or corrupted DICOM series."
            )
        
        self.logger.debug("DICOM UIDs consistency validated successfully")

    def _load_nifti_volume(self, study_info: StudyInfo) -> VolumeData:
        """
        Загружает NIfTI том через nibabel.
        
        Args:
            study_info: Информация о NIfTI файле
        
        Returns:
            VolumeData: Загруженный NIfTI том
        
        Raises:
            ValueError: При некорректных метаданных (spacing и т.д.)
        """
        import nibabel as nib
        
        nifti_path = Path(study_info.path_to_study)
        
        nii_img = nib.load(str(nifti_path))
        volume = nii_img.get_fdata()
        header = nii_img.header
        
        # Извлечение spacing
        spacing = tuple(header.get_zooms()[:3])
        
        # Валидация spacing
        if any(s <= 0 for s in spacing):
            raise ValueError(
                f"Invalid spacing in NIfTI header: {spacing}. "
                f"All spacing values must be positive."
            )
        
        if any(s > 100 for s in spacing):
            self.logger.warning(
                f"Unusually large spacing in NIfTI: {spacing} mm. "
                f"This may indicate incorrect header."
            )
        
        # Извлечение affine и direction
        affine = nii_img.affine
        origin = tuple(affine[:3, 3])
        direction = affine[:3, :3] / np.array(spacing)
        
        metadata = {
            'spacing': spacing,
            'origin': origin,
            'affine': affine,
            'RescaleSlope': 1.0,
            'RescaleIntercept': 0.0,
        }
        
        volume_data = VolumeData(
            path=nifti_path,
            volume=volume,
            spacing=spacing,
            origin=origin,
            direction=direction,
            metadata=metadata,
            preprocessing_info={}
        )
        
        return volume_data
