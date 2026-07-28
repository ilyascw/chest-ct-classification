"""CatBoost model wrapper for pathology classification."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from catboost import CatBoostClassifier
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

logger = logging.getLogger(__name__)


class PathologyClassifier:
    """
    CatBoost classifier wrapper
    """

    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        self.model: CatBoostClassifier | None = None
        self.is_fitted = False

        if self.verbose:
            logger.info("PathologyClassifier initialized")

    def get_default_params(self) -> dict[str, Any]:
        """Get default CatBoost parameters"""

        # Проверка доступности GPU
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

        return {
            "iterations": 1000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": self.random_state,
            "verbose": 100 if self.verbose else False,
            "early_stopping_rounds": 100,
            "use_best_model": True,
            "od_type": "Iter",
            "task_type": "GPU" if gpu_available else "CPU",
        }

    def fit(
        self,
        X_train: NDArray[Any],
        y_train: NDArray[Any],
        X_val: NDArray[Any] | None = None,
        y_val: NDArray[Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train the model"""

        if X_train.ndim != 2:
            raise ValueError("X_train must be a two-dimensional array")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have equal lengths")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together")

        params = dict(params or self.get_default_params())
        if X_val is None:
            params.pop("early_stopping_rounds", None)
            params["use_best_model"] = False

        self.model = CatBoostClassifier(**params)

        if X_val is not None and y_val is not None:
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=False)
        else:
            self.model.fit(X_train, y_train, plot=False)

        self.is_fitted = True

        # Return training results
        results: dict[str, Any] = {
            "best_iteration": self.model.get_best_iteration()
            if hasattr(self.model, "get_best_iteration")
            else None
        }

        if X_val is not None:
            val_pred = self.predict_proba(X_val)[:, 1]
            results["val_auc"] = roc_auc_score(y_val, val_pred)

        return results

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Predict classes"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float64]:
        """Predict probabilities"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)

    def evaluate(self, X: NDArray[Any], y: NDArray[Any]) -> dict[str, Any]:
        """Evaluate model"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]

        return {
            "accuracy": accuracy_score(y, y_pred),
            "auc_roc": roc_auc_score(y, y_pred_proba),
            "classification_report": classification_report(y, y_pred, output_dict=True),
        }

    def save(self, filepath: str | Path) -> None:
        """Save model"""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted")

        filepath = Path(filepath)
        self.model.save_model(str(filepath))

        if self.verbose:
            logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str | Path) -> None:
        """Load model"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = CatBoostClassifier()
        self.model.load_model(str(filepath))
        self.is_fitted = True

        if self.verbose:
            logger.info(f"Model loaded from {filepath}")
