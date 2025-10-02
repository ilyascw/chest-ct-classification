"""
CT Pathology Detection Package

Production-ready system for automated pathology detection using CT-CLIP + CatBoost.
"""

__version__ = "1.0.0"
__all__ = []

# Preprocessing
try:
    from .preprocessing import resize_array, preprocess_nifti, prepare_metadata_for_preprocessing, extract_xy_spacing
    PREPROCESSING_AVAILABLE = True
    __all__.extend(['resize_array', 'preprocess_nifti', 'prepare_metadata_for_preprocessing', 'extract_xy_spacing'])
except ImportError as e:
    print(f"Preprocessing import failed: {e}")
    PREPROCESSING_AVAILABLE = False

# Feature extraction
try:
    from .feature_extraction import (
        CTCLIPFeatureExtractor, 
        ImageLatentsClassifier, 
        create_ctclip_model_and_extractor,
        create_ct_clip_model_and_extractor
    )
    FEATURE_EXTRACTION_AVAILABLE = True
    __all__.extend([
        'CTCLIPFeatureExtractor', 
        'ImageLatentsClassifier', 
        'create_ctclip_model_and_extractor',
        'create_ct_clip_model_and_extractor'
    ])
except ImportError as e:
    print(f"Feature extraction import failed: {e}")
    FEATURE_EXTRACTION_AVAILABLE = False

# Model
try:
    from .model import PathologyClassifier
    MODEL_AVAILABLE = True
    __all__.extend(['PathologyClassifier'])
except ImportError as e:
    print(f"Model import failed: {e}")
    MODEL_AVAILABLE = False


def get_available_components():
    """Возвращает информацию о доступных компонентах"""
    return {
        'preprocessing': PREPROCESSING_AVAILABLE,
        'feature_extraction': FEATURE_EXTRACTION_AVAILABLE,
        'model': MODEL_AVAILABLE,
    }
