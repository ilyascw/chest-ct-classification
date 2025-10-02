"""
Service для загрузки медицинских томов из различных форматов.
Использует существующие функции из ct_preprocessor.py
"""
import logging
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

from .data_models import StudyInfo, VolumeData, DataType


class VolumeLoaderService:
    """
    Сервис для загрузки медицинских томов.
    
    Поддерживает:
    - DICOM серии через robust_load_dicom_volume
    - NIfTI файлы через nibabel
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def load_volume_from_study(self, study_info: StudyInfo) -> VolumeData:
        """
        Загружает медицинский том на основе StudyInfo.
        
        Args:
            study_info: Информация об исследовании
            
        Returns:
            VolumeData: Загруженный том с метаданными
            
        Raises:
            ValueError: Если тип данных не поддерживается
        """
        start_time = time.time()
        self.logger.debug(f"Loading volume {study_info.study_uid}")
        
        try:
            if study_info.data_type == DataType.DICOM:
                volume_data = self._load_dicom_volume(study_info)
            elif study_info.data_type == DataType.NIFTI:
                volume_data = self._load_nifti_volume(study_info)
            else:
                raise ValueError(f"Unsupported data type: {study_info.data_type}")
            
            loading_time = time.time() - start_time
            self.logger.debug(f"Volume loaded in {loading_time:.2f}s: {volume_data.volume.shape}")
            
            return volume_data
            
        except Exception as e:
            loading_time = time.time() - start_time
            self.logger.error(f"Failed to load volume ({loading_time:.2f}s): {e}")
            raise
    
    def _load_dicom_volume(self, study_info: StudyInfo) -> VolumeData:
        """
        Загружает DICOM том через robust_load_dicom_volume.
        
        Извлекает file_list из study_info.metadata (если есть) и передаёт
        в robust_load_dicom_volume для загрузки конкретной серии.
        
        Args:
            study_info: Информация о DICOM исследовании
            
        Returns:
            VolumeData: Загруженный DICOM том
        """
        from src.CTPreprocessor.ct_preprocessor import robust_load_dicom_volume
        
        dicom_dir = Path(study_info.path_to_study)
        
        file_list = None
        if 'file_list' in study_info.metadata:
            file_list_str = study_info.metadata['file_list']
            file_list = [Path(f) for f in file_list_str]
            self.logger.debug(f"Using provided file_list with {len(file_list)} files")
        
        volume, metadata = robust_load_dicom_volume(
            dicom_dir=dicom_dir,
            file_list=file_list,
            logger=self.logger
        )
        
        spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
        origin = metadata.get('origin', (0.0, 0.0, 0.0))
        direction = metadata.get('direction', np.eye(3).flatten())

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
    
    def _load_nifti_volume(self, study_info: StudyInfo) -> VolumeData:
        """
        Загружает NIfTI том через nibabel.
        
        Args:
            study_info: Информация о NIfTI файле
            
        Returns:
            VolumeData: Загруженный NIfTI том
        """
        import nibabel as nib
        
        nifti_path = Path(study_info.path_to_study)
        
        nii_img = nib.load(str(nifti_path))
        volume = nii_img.get_fdata()
        
        header = nii_img.header
        spacing = tuple(header.get_zooms()[:3])
        
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
