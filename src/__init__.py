"""
CT Pathology Detection Package

Production-ready system for automated pathology detection using CT-CLIP + CatBoost.
"""

__version__ = "1.0.0"

# Import only what works
try:
    from .preprocessing import (
        resize_array,
        preprocess_nifti,
        CTPreprocessor
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Preprocessing import failed: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    from .feature_extraction import (
        CTCLIPFeatureExtractor,
        ImageLatentsClassifier,
        create_ctclip_model_and_extractor  # НОВАЯ функция!
    )
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError as e:
    print(f"Feature extraction import failed: {e}")
    FEATURE_EXTRACTION_AVAILABLE = False

try:
    from .model import (
        PathologyClassifier
    )
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Model import failed: {e}")
    MODEL_AVAILABLE = False

__all__ = []

if PREPROCESSING_AVAILABLE:
    __all__.extend(['resize_array', 'preprocess_nifti', 'CTPreprocessor'])

if FEATURE_EXTRACTION_AVAILABLE:
    __all__.extend([
        'CTCLIPFeatureExtractor', 
        'ImageLatentsClassifier',
        'create_ctclip_model_and_extractor'
    ])

if MODEL_AVAILABLE:
    __all__.extend(['PathologyClassifier'])

# Convenience function
def get_availability_status():
    """Get availability status of all modules"""
    return {
        'preprocessing': PREPROCESSING_AVAILABLE,
        'feature_extraction': FEATURE_EXTRACTION_AVAILABLE,
        'model': MODEL_AVAILABLE
    }
