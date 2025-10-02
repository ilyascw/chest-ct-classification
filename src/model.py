"""
CatBoost model wrapper for pathology classification
"""

from typing import Union, Optional, Dict, Any
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from catboost import CatBoostClassifier
import joblib

logger = logging.getLogger(__name__)

class PathologyClassifier:
    """
    CatBoost classifier wrapper
    """
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.is_fitted = False
        
        if self.verbose:
            logger.info("PathologyClassifier initialized")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default CatBoost parameters"""
        
        # Проверка доступности GPU
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False
            
        return {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': self.random_state,
            'verbose': 100 if self.verbose else False,
            'early_stopping_rounds': 100,
            'use_best_model': True,
            'od_type': 'Iter',
            'task_type': 'GPU' if gpu_available else 'CPU'
        }
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Train the model"""
        
        if params is None:
            params = self.get_default_params()
            
        self.model = CatBoostClassifier(**params)
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                plot=False
            )
        else:
            self.model.fit(X_train, y_train, plot=False)
            
        self.is_fitted = True
        
        # Return training results
        results = {
            'best_iteration': self.model.get_best_iteration() if hasattr(self.model, 'get_best_iteration') else None
        }
        
        if X_val is not None:
            val_pred = self.predict_proba(X_val)[:, 1]
            results['val_auc'] = roc_auc_score(y_val, val_pred)
            
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_pred_proba),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        filepath = Path(filepath)
        self.model.save_model(str(filepath))
        
        if self.verbose:
            logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load model"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        self.model = CatBoostClassifier()
        self.model.load_model(str(filepath))
        self.is_fitted = True
        
        if self.verbose:
            logger.info(f"Model loaded from {filepath}")
