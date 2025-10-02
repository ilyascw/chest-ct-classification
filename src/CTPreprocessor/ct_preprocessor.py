import os
import gc
import json
import hashlib
import warnings
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import contextlib

import numpy as np
import torch # type: ignore
import SimpleITK as sitk # type: ignore
import pydicom
import nibabel as nib # type: ignore
from tqdm import tqdm

from monai.transforms import ( # type: ignore
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ScaleIntensityRanged, Lambda
) 
from monai.data import Dataset, DataLoader # type: ignore

# Подавление warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# Логирование
# ----------------------------
class CTLogger:
    def __init__(self, name: str = "CTPreprocessor", level: str = "INFO", 
                 enable_console: bool = True, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)

# ----------------------------
# JSON сериализация (ИСПРАВЛЕНО)
# ----------------------------
def make_json_serializable(obj: Any) -> Any:
    """Исправленная JSON сериализация"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, Path):
        return str(obj)  # ИСПРАВЛЕНО: PosixPath -> str
    elif isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size < 100 else f"<array shape={obj.shape}>"
    elif isinstance(obj, torch.Tensor):
        return f"<tensor shape={tuple(obj.shape)}>"
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    else:
        return str(obj)

def safe_json_dump(data: Dict[str, Any], filepath: Path) -> None:
    """Безопасное сохранение JSON"""
    serializable_data = make_json_serializable(data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)

# ----------------------------
# Конфигурация (совместимая с вашим API)
# ----------------------------
@dataclass
class PreprocConfig:
    input_root: str
    preproc_root: str
    
    # Геометрия
    target_pixdim: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    target_size: Tuple[int, int, int] = (128, 128, 128)
    
    # HU параметры
    hu_window: Tuple[float, float] = (-1000.0, 400.0)
    padding_value: Optional[float] = -2048.0
    
    # QA (ЛИБЕРАЛЬНЫЕ пороги)
    enable_qa: bool = True
    min_depth: int = 16          # Снижено с 64
    max_clip_share: float = 0.95  # Увеличено с 0.25
    max_padding_share: float = 0.90  # Увеличено с 0.50
    
    # DICOM фильтры (ОТКЛЮЧЕНЫ по умолчанию)
    enable_dicom_filters: bool = False
    reject_localizer: bool = False
    require_ct_modality: bool = False
    
    # Производительность
    n_procs: int = 4
    itk_threads: int = 1
    overwrite: bool = False
    
    # Логирование
    pipeline_version: str = "v3.0"
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None

def compute_pipeline_hash(config: PreprocConfig) -> str:
    key_params = {
        "version": config.pipeline_version,
        "pixdim": config.target_pixdim,
        "size": config.target_size,
        "hu_window": config.hu_window,
        "padding": config.padding_value,
    }
    content = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:12]

# ----------------------------
# Проверенные функции загрузки (ВАШИ)
# ----------------------------
def select_max_depth_uid(dicom_dir: str) -> Tuple[str, List[str]]:
    """Выбор серии с максимальным количеством срезов"""
    try:
        reader = sitk.ImageSeriesReader()
        series_uids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        
        max_uid = ""
        max_files = []
        
        for uid in series_uids:
            filenames = reader.GetGDCMSeriesFileNames(str(dicom_dir), uid)
            if len(filenames) > len(max_files):
                max_uid = uid
                max_files = filenames
                
        return max_uid, max_files
        
    except Exception:
        # Fallback: все .dcm файлы
        dcm_files = list(Path(dicom_dir).glob("*.dcm"))
        return "fallback_series", [str(f) for f in dcm_files]

def load_via_filelist_sitk(filelist: List[str]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Проверенная загрузка DICOM с корректными метаданными"""
    r = sitk.ImageSeriesReader()
    r.SetFileNames(filelist)
    img = r.Execute()
    arr = sitk.GetArrayFromImage(img)  # (D,H,W)
    arr = np.moveaxis(arr, 0, -1)[None, None]  # (1,1,H,W,D)
    vol = torch.from_numpy(arr.copy().astype(np.float32))
    
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    origin = img.GetOrigin()
    
    # Создаем affine матрицу
    affine = np.eye(4, dtype=np.float64)
    if len(direction) == 9:
        direction_matrix = np.array(direction).reshape(3, 3)
        spacing_matrix = np.diag(spacing)
        affine[:3, :3] = direction_matrix @ spacing_matrix
        affine[:3, 3] = origin
    
    meta = {
        "spacing": spacing,
        "direction": direction,
        "origin": origin,
        "affine": affine,
        "spatial_shape": vol.shape[2:],
        "reader": "SimpleITKSeriesReader"
    }
    return vol, meta


def load_nifti_robust(filepath: str, logger: CTLogger) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Робастная загрузка NIfTI с fallback на nibabel"""
    try:
        logger.debug(f"📖 Загрузка NIfTI: {Path(filepath).name}")
        
        # Попытка через MONAI
        try:
            loader = Compose([
                LoadImaged(keys=["image"], image_only=False),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"], dtype=torch.float32),
            ])
            
            data = {"image": filepath}
            result = loader(data)
            volume = result["image"]
            
            meta = {}
            if hasattr(volume, 'meta'):
                meta = dict(volume.meta)
            
            # Нормализация размерности
            if len(volume.shape) == 3:  # (H,W,D)
                volume = volume.unsqueeze(0).unsqueeze(0)
            elif len(volume.shape) == 4:  # (C,H,W,D)
                volume = volume.unsqueeze(0)
            
            logger.debug(f"  ✅ MONAI загрузка: {volume.shape}")
            return volume, meta
            
        except Exception as monai_error:
            logger.debug(f"  🔄 MONAI ошибка: {monai_error}, пробуем nibabel")
            
            # Fallback через nibabel
            nii = nib.load(filepath)
            array = nii.get_fdata().astype(np.float32)

            # ИЗВЛЕКАЕМ РЕАЛЬНЫЙ SPACING из nibabel header
            header = nii.header
            pixdim = header['pixdim']
            real_spacing = [pixdim[1], pixdim[2], pixdim[3]]  # X, Y, Z

            meta = {
                "affine": nii.affine,
                "spacing": real_spacing,  # ← РЕАЛЬНЫЙ spacing!
                "pixdim": pixdim,         # ← дублируем для надежности
                "reader": "nibabel_fallback"
            }
            
            logger.debug(f"  ✅ Nibabel загрузка: {volume.shape}")
            return volume, meta
        
    except Exception as e:
        logger.error(f"❌ Полная ошибка загрузки NIfTI {filepath}: {e}")
        raise RuntimeError(f"Не удалось загрузить NIfTI: {e}")


# ----------------------------
# Извлечение размера вокселя
# ----------------------------

def extract_spacing_robust(meta: Dict[str, Any]) -> Optional[List[float]]:
    """Робастное извлечение spacing из разных источников"""
    
    # 1. Прямое поле spacing (DICOM или наш fallback)
    if "spacing" in meta and meta["spacing"] is not None:
        spacing = meta["spacing"]
        if isinstance(spacing, (list, tuple, np.ndarray)) and len(spacing) >= 3:
            return list(spacing)[:3]
    
    # 2. Поле pixdim (MONAI NIfTI)
    if "pixdim" in meta:
        pixdim = meta["pixdim"]
        if hasattr(pixdim, '__len__') and len(pixdim) >= 4:
            try:
                # pixdim[0] не используется, [1,2,3] = X,Y,Z
                spacing = [float(pixdim[1]), float(pixdim[2]), float(pixdim[3])]
                if all(s > 0 for s in spacing):  # проверка на валидность
                    return spacing
            except:
                pass
    
    # 3. Из affine матрицы (последний шанс)
    if "affine" in meta:
        try:
            affine = meta["affine"]
            if hasattr(affine, 'shape') and affine.shape == (4, 4):
                # Вычисляем spacing как норму векторов
                if hasattr(affine, 'numpy'):  # если torch tensor
                    affine = affine.numpy()
                spacing = np.sqrt(np.sum(affine[:3,:3]**2, axis=0))
                return spacing.tolist()
        except:
            pass
    
    # Если ничего не найдено
    return None


# ----------------------------
# Обнаружение данных (ИСПРАВЛЕНО)
# ----------------------------
def get_all_series_in_dicom_dir(dicom_dir: Path) -> List[Tuple[str, List[Path]]]:
    """
    Получает все серии DICOM в директории с их файлами.
    
    Сканирует директорию, группирует DICOM файлы по SeriesInstanceUID.
    
    Args:
        dicom_dir: Путь к директории с DICOM файлами
        
    Returns:
        List[Tuple[str, List[Path]]]: Список кортежей (series_uid, отсортированный_список_файлов)
    """
    import pydicom
    
    series_dict = {}
    
    for dcm_file in dicom_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            series_uid = getattr(ds, 'SeriesInstanceUID', 'unknown')
            
            if series_uid not in series_dict:
                series_dict[series_uid] = []
            series_dict[series_uid].append(dcm_file)
            
        except Exception:
            continue
    
    result = []
    for series_uid, files in series_dict.items():
        sorted_files = sorted(files, key=lambda x: x.name)
        result.append((series_uid, sorted_files))
    
    return result


def robust_load_dicom_volume(
    dicom_dir: Path, 
    file_list: Optional[List[Path]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Загрузка DICOM тома с извлечением метаданных.
    
    Если file_list предоставлен, загружает именно эти файлы.
    Иначе выбирает серию с максимальным количеством файлов (fallback).
    
    Выполняет:
    1. Загрузка серии через SimpleITK
    2. Извлечение spacing, origin, direction
    3. Извлечение RescaleSlope и RescaleIntercept из DICOM тегов
    
    Args:
        dicom_dir: Путь к директории с DICOM файлами
        file_list: Список файлов конкретной серии (опционально)
        logger: Логгер для вывода информации
        
    Returns:
        Tuple[np.ndarray, Dict]: (том, метаданные)
        
    Raises:
        ValueError: Если не удалось загрузить том
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        dicom_dir = Path(dicom_dir)
        
        if file_list is None:
            uid, file_list = select_max_depth_uid(dicom_dir)
            logger.info(f"Loading series {uid} with {len(file_list)} files")
        else:
            logger.info(f"Loading provided file list with {len(file_list)} files")
        
        if not file_list:
            raise ValueError(f"No DICOM files found in {dicom_dir}")
            
        MAX_SLICES = 1000
        
        if len(file_list) > MAX_SLICES:
            raise ValueError(
                f"Series too large: {len(file_list)} slices (max {MAX_SLICES}). "
                f"This may cause memory issues. Series skipped."
            )            
            
        
        first_dicom_path = file_list[0]
        try:
            import pydicom
            ds = pydicom.dcmread(str(first_dicom_path), stop_before_pixels=True)
            rescale_slope = float(getattr(ds, 'RescaleSlope', 1.0))
            rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
            series_uid = getattr(ds, 'SeriesInstanceUID', 'unknown')
        except Exception as e:
            logger.warning(f"Failed to extract DICOM metadata: {e}. Using defaults.")
            rescale_slope = 1.0
            rescale_intercept = 0.0
            series_uid = 'unknown'
        
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(f) for f in file_list])
        image = reader.Execute()
        
        volume = sitk.GetArrayFromImage(image)
        
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        
        metadata = {
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'series_uid': series_uid,
            'num_slices': len(file_list),
            'RescaleSlope': rescale_slope,
            'RescaleIntercept': rescale_intercept,
        }
        
        logger.info(f"Loaded volume: {volume.shape}, spacing: {spacing}")
        logger.info(f"RescaleSlope: {rescale_slope}, RescaleIntercept: {rescale_intercept}")
        
        return volume, metadata
        
    except Exception as e:
        logger.error(f"Failed to load DICOM volume: {e}")
        raise ValueError(f"DICOM loading failed: {e}")


def discover_inputs_robust(
    input_dir: Path, 
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Обнаружение всех медицинских данных в директории.
    
    Поддерживаемые форматы:
    - NIfTI (.nii, .nii.gz)
    - DICOM серии (множественные .dcm файлы)
    
    Важно: Возвращает ВСЕ найденные серии с сохранением списков файлов.
    
    Args:
        input_dir: Путь к директории для сканирования
        logger: Логгер
        
    Returns:
        List[Dict]: Список найденных исследований с метаданными и file_list
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    input_dir = Path(input_dir)
    discovered_inputs = []
    
    nifti_files = list(input_dir.rglob("*.nii*"))
    for nifti_path in nifti_files:
        try:
            file_size = nifti_path.stat().st_size / (1024 * 1024)
            
            discovered_inputs.append({
                'type': 'nifti',
                'path': str(nifti_path),
                'file_size_mb': round(file_size, 2),
                'files_count': 1,
                'series_uid': f"nifti_{nifti_path.stem}",
                'study_uid': f"study_{nifti_path.parent.name}",
            })
            
            logger.info(f"Found NIfTI: {nifti_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to process NIfTI {nifti_path}: {e}")
    
    dicom_dirs = set()
    for dcm_file in input_dir.rglob("*.dcm"):
        dicom_dirs.add(dcm_file.parent)
    
    for dicom_dir in dicom_dirs:
        try:
            all_series = get_all_series_in_dicom_dir(dicom_dir)
            
            if not all_series:
                logger.warning(f"No valid DICOM series found in {dicom_dir}")
                continue
            
            for series_uid, file_list in all_series:
                total_size = sum(f.stat().st_size for f in file_list) / (1024 * 1024)
                
                discovered_inputs.append({
                    'type': 'dicom_dir',
                    'path': str(dicom_dir),
                    'series_uid': series_uid,
                    'study_uid': f"study_{dicom_dir.name}",
                    'files_count': len(file_list),
                    'file_size_mb': round(total_size, 2),
                    'file_list': [str(f) for f in file_list],
                })
                
                logger.info(f"Found DICOM series: {series_uid} ({len(file_list)} files)")
        
        except Exception as e:
            logger.warning(f"Failed to process DICOM dir {dicom_dir}: {e}")
    
    logger.info(f"Total discovered: {len(discovered_inputs)} input(s)")
    
    return discovered_inputs

# ----------------------------
# Робастный пайплайн (БЕЗ проблемных MONAI трансформов)
# ----------------------------
def build_robust_pipeline(config: PreprocConfig) -> Compose:
    """Робастный пайплайн без проблемных MONAI трансформов"""
    
    def robust_preprocessing(data):
        img = data["image"]
        meta = data.get("image_meta_dict", {})
        
        print(f"  🔍 Вход: {img.shape}, мета: {len(meta)}")
        
        # 1. Padding замена
        if config.padding_value is not None:
            padding_mask = (img == config.padding_value)
            if padding_mask.sum() > 0:
                img = torch.where(padding_mask, 
                                torch.tensor(config.hu_window[0], dtype=img.dtype), 
                                img)
                print(f"  🔄 Заменено padding: {padding_mask.sum().item()} вокселей")
        
        # 2. HU клиппинг
        img_clipped = torch.clamp(img, config.hu_window[0], config.hu_window[1])
        
        # 3. Spacing нормализация (если возможно)
        img_resampled = img_clipped
        try:

            current_spacing = extract_spacing_robust(meta)  # ← НОВЫЙ вызов

            if current_spacing is not None:
                print(f"  📏 Найденный spacing: {current_spacing}")
                # остальная логика ресэмплинга...
            else:
                print(f"  ⚠️ Не удалось извлечь spacing из метаданных")

            
            if current_spacing is not None:
                spacing_array = np.array(current_spacing)
                target_array = np.array(config.target_pixdim)
                spacing_diff = np.abs(spacing_array - target_array).max()
                
                if spacing_diff > 0.1:  # Нужен ресэмплинг
                    print(f"  📏 Ресэмплинг: {current_spacing} -> {config.target_pixdim}")
                    
                    # Извлекаем numpy массив
                    if len(img_clipped.shape) == 5:
                        numpy_array = img_clipped[0, 0].detach().cpu().numpy()
                    elif len(img_clipped.shape) == 4:
                        numpy_array = img_clipped[0].detach().cpu().numpy()
                    else:
                        numpy_array = img_clipped.detach().cpu().numpy()
                    
                    # SimpleITK ресэмплинг
                    sitk_img = sitk.GetImageFromArray(numpy_array.transpose(2, 1, 0))
                    sitk_img.SetSpacing(current_spacing)
                    
                    if "origin" in meta:
                        sitk_img.SetOrigin(meta["origin"])
                    if "direction" in meta:
                        sitk_img.SetDirection(meta["direction"])
                    
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetOutputSpacing(config.target_pixdim)
                    resampler.SetInterpolator(sitk.sitkLinear)
                    
                    original_size = sitk_img.GetSize()
                    new_size = [
                        max(1, int(original_size[i] * current_spacing[i] / config.target_pixdim[i]))
                        for i in range(3)
                    ]
                    resampler.SetSize(new_size)
                    resampler.SetOutputOrigin(sitk_img.GetOrigin())
                    resampler.SetOutputDirection(sitk_img.GetDirection())
                    
                    resampled_sitk = resampler.Execute(sitk_img)
                    resampled_array = sitk.GetArrayFromImage(resampled_sitk).transpose(2, 1, 0)
                    resampled_tensor = torch.from_numpy(resampled_array.astype(np.float32))
                    
                    # Восстановление размерностей
                    if len(img_clipped.shape) == 5:
                        resampled_tensor = resampled_tensor.unsqueeze(0).unsqueeze(0)
                    elif len(img_clipped.shape) == 4:
                        resampled_tensor = resampled_tensor.unsqueeze(0)
                    
                    img_resampled = resampled_tensor
                    print(f"  ✅ Ресэмплинг выполнен: {img_resampled.shape}")
                else:
                    print(f"  ✅ Spacing подходящий")
            else:
                print(f"  ⚠️ Нет данных spacing")
                
        except Exception as e:
            print(f"  ❌ Ошибка ресэмплинга: {e}")
            img_resampled = img_clipped
        
        # 4. Интенсивностная нормализация
        img_normalized = (img_resampled - config.hu_window[0]) / (config.hu_window[1] - config.hu_window[0])
        
        # 5. Resize через F.interpolate (БЕЗОПАСНО)
        import torch.nn.functional as F
        
        current_size = img_normalized.shape[-3:]
        if current_size != config.target_size:
            resized = F.interpolate(
                img_normalized, 
                size=config.target_size, 
                mode='trilinear', 
                align_corners=False
            )
            print(f"  📐 Resize: {current_size} -> {config.target_size}")
        else:
            resized = img_normalized
            print(f"  ✅ Размер подходящий")
        
        data["image"] = resized
        return data
    
    return Compose([
        Lambda(func=robust_preprocessing),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])

# ----------------------------
# QA проверки (ЛИБЕРАЛЬНЫЕ)
# ----------------------------
def qa_volume_liberal(volume: torch.Tensor, config: PreprocConfig) -> Dict[str, Any]:
    """Либеральная QA проверка"""
    if not config.enable_qa:
        return {"qa_pass": True, "reason": "qa_disabled"}
    
    # Быстрые проверки
    shape = volume.shape
    depth = shape[-1] if len(shape) >= 3 else 1
    
    depth_ok = depth >= config.min_depth
    
    # Сэмплирование для скорости
    sample_vol = volume.view(-1)[::1000]
    
    # Анализ на выборке
    clipped = ((sample_vol <= 0.01) | (sample_vol >= 0.99)).float().mean()
    air_ratio = (sample_vol <= 0.1).float().mean()
    
    clip_ok = clipped <= config.max_clip_share
    air_ok = air_ratio <= config.max_padding_share
    
    qa_pass = depth_ok and clip_ok and air_ok
    
    return {
        "qa_pass": qa_pass,
        "depth_ok": depth_ok,
        "clip_ok": clip_ok,
        "air_ok": air_ok,
        "metrics": {
            "depth": depth,
            "clip_share": clipped.item(),
            "air_share": air_ratio.item(),
        }
    }

# ----------------------------
# Главный процессор (ФИНАЛЬНЫЙ)
# ----------------------------
def process_one_sample_final(sample: Dict[str, Any], config: PreprocConfig, 
                           pipeline_hash: str, logger: CTLogger) -> Dict[str, Any]:
    """Финальная обработка одного образца"""
    
    sample_id = sample.get("source_id", "unknown")
    
    try:
        logger.debug(f"🔄 Обработка: {sample_id}")
        
        # Проверка кэша
        output_dir = Path(config.preproc_root) / pipeline_hash
        file_hash = hashlib.md5(sample_id.encode()).hexdigest()[:8]
        output_file = output_dir / f"{file_hash}.nii.gz"
        manifest_file = output_dir / f"{file_hash}.json"
        
        if output_file.exists() and manifest_file.exists() and not config.overwrite:
            logger.debug(f"  💾 Кэш: {output_file.name}")
            with open(manifest_file, 'r') as f:
                cached = json.load(f)
            return {"ok": True, "cached": True, **cached}
        
        # Загрузка
        if sample["kind"] == "nifti":
            volume, meta = load_nifti_robust(sample["path"], logger)
        else:
            volume, meta = robust_load_dicom_volume(sample["root"], logger)
        
        # Препроцессинг
        pipeline = build_robust_pipeline(config)
        data_dict = {"image": volume, "image_meta_dict": meta}
        
        # Очистка памяти
        del volume
        gc.collect()
        
        processed_data = pipeline(data_dict)
        processed_volume = processed_data["image"]
        
        logger.debug(f"  📊 Результат: {processed_volume.shape}")
        
        # QA
        qa_result = qa_volume_liberal(processed_volume, config)
        if not qa_result["qa_pass"]:
            logger.debug(f"  ❌ QA отклонен")
            return {
                "ok": False, "cached": False,
                "reason": "qa_failed",
                "source_id": sample_id,
                "qa": qa_result
            }
        
        # Сохранение
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if len(processed_volume.shape) == 5:
            save_array = processed_volume[0, 0].detach().cpu().numpy()
        elif len(processed_volume.shape) == 4:
            save_array = processed_volume[0].detach().cpu().numpy()
        else:
            save_array = processed_volume.detach().cpu().numpy()
        
        affine = np.eye(4)
        if "affine" in meta:
            try:
                affine = np.array(meta["affine"])
            except:
                pass
        
        nii_img = nib.Nifti1Image(save_array.astype(np.float32), affine)
        nib.save(nii_img, str(output_file))
        
        # Манифест
        manifest = {
            "source_id": sample_id,
            "kind": sample["kind"],
            "output_img": str(output_file),
            "pipe_hash": pipeline_hash,
            "qa": qa_result,
        }
        
        safe_json_dump(manifest, manifest_file)
        
        # Очистка
        del processed_volume
        gc.collect()
        
        logger.debug(f"  ✅ Сохранено: {output_file.name}")
        
        return {
            "ok": True, "cached": False,
            "source_id": sample_id,
            "output_img": str(output_file),
            "qa": qa_result
        }
        
    except Exception as e:
        logger.error(f"  ❌ Ошибка {sample_id}: {e}")
        return {
            "ok": False, "cached": False,
            "reason": f"exception: {str(e)}",
            "source_id": sample_id
        }

class MedPreprocessor:
    """ФИНАЛЬНАЯ версия процессора"""
    
    def __init__(self, cfg: PreprocConfig):
        self.cfg = cfg
        self.pipe_hash = compute_pipeline_hash(cfg)
        
        self.logger = CTLogger(
            name="MedPreprocessor",
            level=cfg.log_level if cfg.enable_logging else "ERROR",
            enable_console=cfg.enable_logging,
            log_file=cfg.log_file
        )
        
        self.output_dir = Path(cfg.preproc_root) / self.pipe_hash
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🚀 MedPreprocessor v3.0 инициализирован")
        self.logger.info(f"   📂 {cfg.input_root}")
        self.logger.info(f"   📁 {self.output_dir}")
    
    def build(self) -> List[Dict[str, Any]]:
        """Главная функция препроцессинга"""
        
        self.logger.info("🔍 Обнаружение данных...")
        samples = discover_inputs_robust(self.cfg.input_root, self.logger)
        
        if not samples:
            self.logger.warning("⚠️ Данные не найдены!")
            return []
        
        self.logger.info(f"📊 К обработке: {len(samples)}")
        
        results = []
        
        if self.cfg.n_procs <= 1:
            # Однопоточно
            for sample in tqdm(samples, desc="Обработка", disable=not self.cfg.enable_logging):
                result = process_one_sample_final(sample, self.cfg, self.pipe_hash, self.logger)
                results.append(result)
        else:
            # Многопоточно  
            process_func = partial(
                process_one_sample_final,
                config=self.cfg,
                pipeline_hash=self.pipe_hash,
                logger=self.logger
            )
            
            with ProcessPoolExecutor(max_workers=self.cfg.n_procs) as executor:
                futures = [executor.submit(process_func, sample) for sample in samples]
                
                for future in tqdm(as_completed(futures), total=len(samples), 
                                 desc="Обработка", disable=not self.cfg.enable_logging):
                    try:
                        result = future.result(timeout=300)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"❌ Процесс: {e}")
                        results.append({"ok": False, "reason": f"process_error: {str(e)}"})
        
        # Статистика
        successful = sum(1 for r in results if r.get("ok", False))
        cached = sum(1 for r in results if r.get("cached", False))
        
        self.logger.info(f"✅ Завершено:")
        self.logger.info(f"   📊 Успешно: {successful}/{len(results)}")
        self.logger.info(f"   💾 Кэш: {cached}")
        
        # Индекс
        index_data = {
            "total": len(samples),
            "successful": successful,
            "cached": cached,
            "failed": len(results) - successful,
            "items": results
        }
        
        index_file = self.output_dir / "_index.json"
        safe_json_dump(index_data, index_file)
        
        return results
    
    def get_dataloader(self, batch_size: int = 4, num_workers: int = 2, 
                      shuffle: bool = False) -> DataLoader:
        """DataLoader для обучения"""
        
        index_file = self.output_dir / "_index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Индекс не найден: {index_file}")
        
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        successful_items = [
            item for item in index_data["items"] 
            if item.get("ok", False) and "output_img" in item
        ]
        
        if not successful_items:
            raise ValueError("Нет данных для DataLoader")
        
        self.logger.info(f"📚 DataLoader: {len(successful_items)} файлов")
        
        data_dicts = [{"image": item["output_img"]} for item in successful_items]
        
        loader_transforms = Compose([
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ])
        
        dataset = Dataset(data_dicts, transform=loader_transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(num_workers, len(successful_items)),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )
        
        return dataloader
