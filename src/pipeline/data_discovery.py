"""
Service для обнаружения медицинских исследований в ZIP архивах.
Использует существующий ct_preprocessor.py для сканирования.
"""
import logging
import zipfile
import tempfile
import shutil
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
    - DICOM серии (все серии в архиве)
    - NIfTI файлы
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
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
            List[StudyInfo]: Список найденных исследований
        """
        self.logger.info(f"Discovering studies in {zip_path}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            from src.CTPreprocessor.ct_preprocessor import discover_inputs_robust
            
            raw_inputs = discover_inputs_robust(Path(extract_dir), self.logger)
            
            studies = []
            for raw_input in raw_inputs:
                study_info = self._convert_raw_input_to_study_info(raw_input)
                if study_info:
                    studies.append(study_info)
            
            self.logger.info(f"Found {len(studies)} studies in {zip_path}")
            
            return studies
            
        except Exception as e:
            self.logger.error(f"Failed to discover studies: {e}")
            raise
    
    def _convert_raw_input_to_study_info(
        self, 
        raw_input: Dict[str, Any]
    ) -> Optional[StudyInfo]:
        """
        Конвертирует результат discover_inputs_robust в StudyInfo.
        
        Сохраняет file_list из raw_input в metadata для последующей
        передачи в volume_loader.
        
        Args:
            raw_input: Словарь с информацией об обнаруженном исследовании
            
        Returns:
            StudyInfo или None при ошибке
        """
        try:
            input_type = raw_input.get('type', 'unknown')
            
            if input_type == 'dicom_dir':
                data_type = DataType.DICOM
            elif input_type == 'nifti':
                data_type = DataType.NIFTI
            else:
                self.logger.warning(f"Unknown input type: {input_type}")
                return None
            
            metadata = {
                'input_type': input_type,
            }
            
            if 'file_list' in raw_input:
                metadata['file_list'] = raw_input['file_list']
            
            study_info = StudyInfo(
                path_to_study=raw_input.get('path', ''),
                study_uid=raw_input.get('study_uid', 'unknown'),
                series_uid=raw_input.get('series_uid', 'unknown'),
                data_type=data_type,
                files_count=raw_input.get('files_count', 0),
                file_size_mb=raw_input.get('file_size_mb', 0.0),
                metadata=metadata,
            )
            
            return study_info
            
        except Exception as e:
            self.logger.error(f"Failed to convert raw input: {e}")
            return None

    
    def _process_dicom_input(self, raw_input: Dict[str, Any]) -> StudyInfo:
        """Обработка DICOM input"""
        
        dicom_root = raw_input['root']
        filelist = raw_input.get('filelist', [])
        
        # Извлекаем метаданные из первого DICOM файла
        try:
            if filelist:
                first_dicom_path = os.path.join(dicom_root, filelist[0])
                ds = pydicom.dcmread(first_dicom_path, stop_before_pixels=True)
                
                study_uid = str(getattr(ds, 'StudyInstanceUID', 'unknown'))
                series_uid = str(getattr(ds, 'SeriesInstanceUID', 'unknown'))
                
                # Собираем метаданные
                metadata = {
                    'patient_id': str(getattr(ds, 'PatientID', 'unknown')),
                    'study_date': str(getattr(ds, 'StudyDate', 'unknown')),
                    'modality': str(getattr(ds, 'Modality', 'unknown')),
                    'manufacturer': str(getattr(ds, 'Manufacturer', 'unknown')),
                    'slice_thickness': str(getattr(ds, 'SliceThickness', 'unknown')),
                    'pixel_spacing': str(getattr(ds, 'PixelSpacing', 'unknown')),
                }
            else:
                study_uid = f"dicom_study_{uuid.uuid4().hex[:8]}"
                series_uid = f"dicom_series_{uuid.uuid4().hex[:8]}"
                metadata = {}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to read DICOM metadata: {e}")
            study_uid = f"dicom_study_{uuid.uuid4().hex[:8]}"
            series_uid = f"dicom_series_{uuid.uuid4().hex[:8]}"
            metadata = {}
        
        # Подсчитываем размер файлов
        total_size = 0
        for filename in filelist:
            filepath = os.path.join(dicom_root, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
        
        return StudyInfo(
            path_to_study=dicom_root,
            study_uid=study_uid,
            series_uid=series_uid,
            data_type=DataType.DICOM,
            files_count=len(filelist),
            file_size_mb=total_size / (1024 * 1024),
            metadata=metadata
        )
    
    def _process_nifti_input(self, raw_input: Dict[str, Any]) -> StudyInfo:
        """Обработка NIfTI input"""
        
        nifti_path = raw_input['path']
        nifti_file = Path(nifti_path)
        
        # Извлекаем метаданные из NIfTI
        try:
            img = nib.load(nifti_path)
            header = img.header
            
            metadata = {
                'shape': str(img.shape),
                'dtype': str(img.get_fdata().dtype),
                'pixdim': str(header.get('pixdim')),
                'qform_code': str(header.get('qform_code')),
                'sform_code': str(header.get('sform_code')),
                'descrip': str(header.get('descrip', b'').decode('utf-8', errors='ignore')),
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to read NIfTI metadata: {e}")
            metadata = {}
        
        # UIDs based на имени файла
        file_stem = nifti_file.stem.replace('.nii', '')  # Handle .nii.gz
        study_uid = f"nifti_{file_stem}"
        series_uid = f"series_{file_stem}"
        
        # Размер файла
        file_size_mb = os.path.getsize(nifti_path) / (1024 * 1024)
        
        return StudyInfo(
            path_to_study=nifti_path,
            study_uid=study_uid,
            series_uid=series_uid,
            data_type=DataType.NIFTI,
            files_count=1,
            file_size_mb=file_size_mb,
            metadata=metadata
        )
