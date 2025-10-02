"""
Preprocessing module based on original CT-CLIP data_inference_nii.py
"""

from typing import Tuple, Union, Optional, List
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def resize_array(
    array: torch.Tensor, 
    current_spacing: Tuple[float, float, float], 
    target_spacing: Tuple[float, float, float]
) -> np.ndarray:
    """
    Resize the array to match the target spacing.
    Original implementation from data_inference_nii.py
    """
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    
    resized_array = F.interpolate(
        array, 
        size=new_shape, 
        mode='trilinear', 
        align_corners=False
    ).cpu().numpy()
    
    return resized_array


def preprocess_nifti(nii_path: Union[str, Path], meta_row: pd.Series, Volume=False) -> torch.Tensor:
    """
    Preprocess NIfTI file exactly as in original data_inference_nii.py
    """
    
    if not Volume:
        
        # Load NIfTI image
        nii_img = nib.load(str(nii_path))
        img_data = nii_img.get_fdata()
    
        # Get metadata
        filename = Path(nii_path).name
        
    else:
        
        img_data = nii_path
        
    slope = float(meta_row['RescaleSlope'])
    intercept = float(meta_row['RescaleIntercept'])
    
    # Parse XYSpacing - handle both string and float formats
    xy_spacing_val = meta_row['XYSpacing']
    if isinstance(xy_spacing_val, str):
        xy_spacing = float(xy_spacing_val.strip('[]').split(',')[0])
    else:
        xy_spacing = float(xy_spacing_val)
    
    z_spacing = float(meta_row['ZSpacing'])
    
    # Target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5
    
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)
    
    # Apply rescale slope and intercept
    img_data = slope * img_data + intercept
    
    # Clip HU values
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    
    # Transpose to (D, H, W)
    img_data = img_data.transpose(2, 0, 1)
    
    # Convert to tensor and add batch + channel dimensions
    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    
    # Resize to target spacing
    img_data = resize_array(tensor, current, target)
    img_data = img_data[0, 0]  # Remove batch and channel dims
    
    # Transpose back to (H, W, D)
    img_data = np.transpose(img_data, (1, 2, 0))
    
    # Normalize to [0, 1]
    img_data = (img_data + 1000) / 2000.0
    img_data = img_data.astype(np.float32)
    
    # Convert to tensor
    tensor = torch.tensor(img_data)
    
    # Target shape for CT-CLIP
    target_shape = (480, 480, 240)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape
    
    # Calculate cropping/padding
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)
    
    # Crop tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
    
    # Calculate padding
    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before
    
    # Apply padding
    tensor = F.pad(
        tensor, 
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), 
        value=-1
    )
    
    # Final format for CT-CLIP
    tensor = tensor.permute(2, 0, 1)  # (D, H, W)
    tensor = tensor.unsqueeze(0)      # (1, D, H, W)
    
    return tensor


class CTPreprocessor:
    """
    CT preprocessing wrapper
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        if self.verbose:
            logger.info("CTPreprocessor initialized")
    
    def preprocess_file(self, file_path: Union[str, Path], metadata: pd.Series) -> torch.Tensor:
        """Preprocess single file"""
        return preprocess_nifti(file_path, metadata)
    
    def preprocess_batch(
        self, 
        file_paths: List[Union[str, Path]], 
        metadata_df: pd.DataFrame,
        show_progress: bool = True
    ) -> List[torch.Tensor]:
        """Preprocess multiple files"""
        results = []
        
        iterator = file_paths
        if show_progress:
            iterator = tqdm(file_paths, desc="Preprocessing")
        
        for file_path in iterator:
            filename = Path(file_path).name
            meta_row = metadata_df[metadata_df['VolumeName'] == filename]
            
            if meta_row.empty:
                logger.warning(f"No metadata for {filename}")
                continue
                
            try:
                tensor = self.preprocess_file(file_path, meta_row.iloc[0])
                results.append(tensor)
            except Exception as e:
                logger.error(f"Failed to preprocess {filename}: {e}")
                continue
        
        return results
