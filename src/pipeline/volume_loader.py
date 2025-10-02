# src/pipeline/volume_loader.py
"""
Service для загрузки медицинских томов
Использует ваши existing functions
"""
import logging
import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

from .data_models import StudyInfo, VolumeData, DataType

class VolumeLoaderService:
    """Универсальная загрузка медицинских томов"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def load_volume_from_study(self, study_info: StudyInfo) -> VolumeData:
        """Загрузка тома на основе StudyInfo"""
        
        start_time = time.time()
        self.logger.debug(f"📂 Loading volume: {study_info.study_uid}")
        
        try:
            if study_info.data_type == DataType.DICOM:
                volume_data = self._load_dicom_volume(study_info)
            elif study_info.data_type == DataType.NIFTI:
                volume_data = self._load_nifti_volume(study_info)
            else:
                raise ValueError(f"Unsupported data type: {study_info.data_type}")
            
            loading_time = time.time() - start_time
            self.logger.debug(f"✅ Volume loaded in {loading_time:.2f}s: {volume_data.volume.shape}")
            
            return volume_data
            
        except Exception as e:
            loading_time = time.time() - start_time
            self.logger.error(f"❌ Failed to load volume ({loading_time:.2f}s): {e}")
            raise
    
    def _load_dicom_volume(self, study_info: StudyInfo) -> VolumeData:
        """Загрузка DICOM серии используя ваш robust_load_dicom_volume"""
        
        try:
            # Импортируем вашу функцию
            from src.CTPreprocessor.ct_preprocessor import robust_load_dicom_volume
            
            # Загружаем том
            volume, metadata = robust_load_dicom_volume(
                study_info.path_to_study,
                self.logger
            )
            
            # Извлекаем spacing, origin, direction из metadata
            spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
            origin = metadata.get('origin', (0.0, 0.0, 0.0))
            direction = metadata.get('direction', np.eye(3))
            
            # Дополнительная обработанная информация
            preprocessing_info = {
                'loader_method': 'robust_load_dicom_volume',
                'original_shape': str(volume.shape),
                'value_range': f"[{volume.min():.2f}, {volume.max():.2f}]",
                'data_type': str(volume.dtype)
            }
            
            return VolumeData(
                path=study_info.path_to_study,
                volume=volume,
                spacing=spacing,
                origin=origin,
                direction=direction,
                metadata=metadata,
                preprocessing_info=preprocessing_info
            )
            
        except Exception as e:
            self.logger.error(f"❌ DICOM loading failed: {e}")
            raise RuntimeError(f"DICOM loading failed: {e}")
    
    def _load_nifti_volume(self, study_info: StudyInfo) -> VolumeData:
        """Загрузка NIfTI файла"""
        
        try:
            import nibabel as nib
            
            # Загружаем NIfTI
            img = nib.load(study_info.path_to_study)
            volume = img.get_fdata().astype(np.float32)
            
            # Извлекаем пространственную информацию
            header = img.header
            affine = img.affine
            
            # Spacing из pixdim
            pixdim = header.get('pixdim')
            if pixdim is not None and len(pixdim) >= 4:
                spacing = (float(pixdim[1]), float(pixdim[2]), float(pixdim[3]))
            else:
                spacing = (1.0, 1.0, 1.0)
            
            # Origin и direction из affine matrix
            origin = tuple(affine[:3, 3].astype(float))
            direction = affine[:3, :3] / np.array(spacing)  # Normalize by spacing
            
            # Метаданные
            metadata = {
                'header': header,
                'affine': affine,
                'spacing': spacing,
                'origin': origin,
                'direction': direction,
                'units': str(header.get('xyzt_units', 'unknown')),
                'qform_code': int(header.get('qform_code', 0)),
                'sform_code': int(header.get('sform_code', 0))
            }
            
            # Preprocessing info
            preprocessing_info = {
                'loader_method': 'nibabel_load',
                'original_shape': str(volume.shape),
                'value_range': f"[{volume.min():.2f}, {volume.max():.2f}]",
                'data_type': str(volume.dtype)
            }
            
            return VolumeData(
                path=study_info.path_to_study,
                volume=volume,
                spacing=spacing,
                origin=origin,
                direction=direction,
                metadata=metadata,
                preprocessing_info=preprocessing_info
            )
            
        except Exception as e:
            self.logger.error(f"❌ NIfTI loading failed: {e}")
            raise RuntimeError(f"NIfTI loading failed: {e}")
