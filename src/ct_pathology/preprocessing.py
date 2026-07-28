"""Volume preprocessing adapted from the original CT-CLIP inference pipeline."""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ct_pathology.pipeline.data_models import VolumeData


def extract_xy_spacing(xy_string: str | float | int) -> float:
    """
    Извлекает значение X spacing из различных форматов.

    Поддерживает:
    - float/int: возвращает как есть
    - строка "[0.68, 0.68]": парсит первое значение

    Args:
        xy_string: Значение XYSpacing в любом формате

    Returns:
        float: Значение X spacing

    Raises:
        ValueError: Если не удалось распарсить значение
    """
    try:
        if isinstance(xy_string, (float, int)):
            return float(xy_string)

        if isinstance(xy_string, str):
            xy_string = xy_string.strip()
            if xy_string.startswith("["):
                return float(xy_string.strip("[]").split(",")[0])
            else:
                return float(xy_string)

        raise ValueError(f"Unsupported type: {type(xy_string)}")

    except (ValueError, TypeError, IndexError) as e:
        raise ValueError(f"Failed to parse XYSpacing '{xy_string}': {e}") from e


def prepare_metadata_for_preprocessing(volume_data: VolumeData) -> pd.Series:
    """
    Преобразует VolumeData в формат метаданных для preprocess_nifti().

    Создает pd.Series с полями:
    - RescaleSlope: коэффициент для конвертации в HU
    - RescaleIntercept: смещение для конвертации в HU
    - XYSpacing: размер пикселя по X (мм)
    - ZSpacing: размер пикселя по Z (мм)

    Args:
        volume_data: Объект VolumeData с загруженным томом и метаданными

    Returns:
        pd.Series: Метаданные в формате для preprocess_nifti()
    """
    spacing = volume_data.spacing

    rescale_slope = float(volume_data.metadata.get("RescaleSlope", 1.0))
    rescale_intercept = float(volume_data.metadata.get("RescaleIntercept", 0.0))
    xy_spacing = float(spacing[0])
    z_spacing = float(spacing[2])

    meta_row = pd.Series(
        {
            "RescaleSlope": rescale_slope,
            "RescaleIntercept": rescale_intercept,
            "XYSpacing": xy_spacing,
            "ZSpacing": z_spacing,
        }
    )

    return meta_row


def resize_array(
    array: torch.Tensor,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> NDArray[np.float32]:
    """
    Изменяет размер массива для соответствия целевому spacing.

    Оригинальная реализация из data/inference_nii.py

    Args:
        array: Тензор изображения для ресемплинга
        current_spacing: Текущий spacing (X, Y, Z) в мм
        target_spacing: Целевой spacing (X, Y, Z) в мм

    Returns:
        np.ndarray: Ресемплированный массив
    """
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]

    resized_array = (
        F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False).cpu().numpy()
    )

    return np.asarray(resized_array, dtype=np.float32)


def preprocess_nifti(
    nii_path: str | Path | NDArray[np.floating[Any]],
    meta_row: pd.Series,
    Volume: bool = False,
) -> torch.Tensor:
    """
    Предобработка NIfTI файла

    Выполняет следующие шаги:
    1. Загрузка тома (из файла или переданного Volume)
    2. Конвертация в Hounsfield Units через RescaleSlope/Intercept
    3. Клиппирование значений HU в диапазон [-1000, 1000]
    4. Ресемплинг до target spacing (0.75, 0.75, 1.5) мм
    5. Изменение размера до (480, 480, 240)
    6. Нормализация в диапазон [0, 1]

    Args:
        nii_path: Путь к .nii.gz файлу или numpy array (если Volume=True)
        meta_row: pd.Series с метаданными (RescaleSlope, RescaleIntercept, XYSpacing, ZSpacing)
        Volume: Если True, nii_path является массивом, а не путём

    Returns:
        torch.Tensor: Предобработанный том размера (1, 1, 480, 480, 240)
    """
    if not Volume:
        if isinstance(nii_path, np.ndarray):
            raise TypeError("nii_path must be a path when Volume=False")
        nii_img = nib.load(str(nii_path))
        img_data = np.asarray(nii_img.get_fdata(), dtype=np.float32)
    else:
        if not isinstance(nii_path, np.ndarray):
            raise TypeError("nii_path must be a numpy array when Volume=True")
        img_data = np.asarray(nii_path, dtype=np.float32)

    rescale_slope = float(meta_row["RescaleSlope"])
    rescale_intercept = float(meta_row["RescaleIntercept"])
    xy_spacing = extract_xy_spacing(meta_row["XYSpacing"])
    z_spacing = float(meta_row["ZSpacing"])

    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = rescale_slope * img_data + rescale_intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)

    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    img_data = resize_array(tensor, current, target)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = ((img_data) / 1000).astype(np.float32)
    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop or pad the tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before

    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before

    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1,
    )

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    return tensor
