"""
Module xây dựng, huấn luyện và đánh giá các mô hình Machine Learning:
- Hỗ trợ Logistic Regression, Random Forest, LightGBM (tự động fallback sang HistGradientBoosting / RF).
- Đánh giá phân loại: Accuracy, Precision, Recall, F1, ROC-AUC.
- Trích xuất xác suất dự đoán (P(Up)).
- Lưu và tải mô hình nguyên khối bằng joblib.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config import MODEL_TYPE, RANDOM_STATE

logger = logging.getLogger("Model")


def get_model(model_type: str = MODEL_TYPE, random_state: int = RANDOM_STATE) -> Any:
    """
    Khởi tạo mô hình Machine Learning phù hợp với dữ liệu chuỗi thời gian tài chính,
    tập trung kiểm soát overfitting (max_depth thấp, min_samples_split hợp lý).
    """
    model_type = model_type.lower()

    if model_type == "logistic":
        logger.info("Khởi tạo mô hình Logistic Regression (L2 regularization, C=0.1)...")
        return LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
        )

    elif model_type == "rf":
        logger.info("Khởi tạo mô hình Random Forest (max_depth=5, min_samples_split=20)...")
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_type == "lgbm":
        try:
            import lightgbm as lgb
            logger.info("Khởi tạo mô hình LightGBM Classifier...")
            return lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                num_leaves=15,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbose=-1,
            )
        except ImportError:
            logger.warning(
                "Chưa cài đặt thư viện 'lightgbm'. Tự động fallback sang 'HistGradientBoostingClassifier' của scikit-learn!"
            )
            return HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                learning_rate=0.03,
                random_state=random_state,
            )

    else:
        logger.warning("Không nhận diện được model_type='%s'. Mặc định sử dụng Random Forest.", model_type)
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            random_state=random_state,
            n_jobs=-1,
        )


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Huấn luyện mô hình trên tập Train."""
    logger.info("Bắt đầu huấn luyện mô hình trên %d mẫu...", len(X_train))
    model.fit(X_train, y_train)
    logger.info("Huấn luyện hoàn tất.")
    return model


def predict_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Dự đoán xác suất thị trường tăng giá (Class 1: P(Target = 1)).
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.ravel()
    elif hasattr(model, "decision_function"):
        # Sigmoid fallback cho các model chỉ có decision_function
        dec = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-dec))
    else:
        preds = model.predict(X)
        return preds.astype(float)


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Đánh giá toàn diện các chỉ số phân loại nhị phân:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC
    """
    y_prob = predict_probabilities(model, X)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
    except Exception:
        metrics["roc_auc"] = 0.50

    return metrics


def get_feature_importance(model: Any, feature_cols: List[str]) -> pd.DataFrame:
    """Trích xuất độ quan trọng của đặc trưng (Feature Importance) để giải thích mô hình."""
    importance_values = None

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_[0])

    if importance_values is not None and len(importance_values) == len(feature_cols):
        df_imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": importance_values,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp

    return pd.DataFrame({"feature": feature_cols, "importance": [0.0] * len(feature_cols)})


def save_model_bundle(
    filepath: Path,
    model: Any,
    scaler: Any,
    feature_cols: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Lưu gói mô hình, scaler, danh sách feature và metadata bằng joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metadata": metadata or {},
    }
    joblib.dump(bundle, filepath)
    logger.info("Đã lưu model bundle vào: %s", filepath)


def load_model_bundle(filepath: Path) -> Dict[str, Any]:
    """Tải gói mô hình đã lưu."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file model tại: {filepath}")
    bundle = joblib.load(filepath)
    logger.info("Đã tải thành công model bundle từ: %s", filepath)
    return bundle
