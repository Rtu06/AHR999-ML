"""
Module phân chia tập dữ liệu và chuẩn hóa đặc trưng cho Machine Learning chuỗi thời gian.

Tuân thủ nguyên tắc chống rò rỉ dữ liệu (No Data Leakage):
1. Phân chia tuần tự theo thời gian: Train -> Validation -> Test (TUYỆT ĐỐI KHÔNG SHUFFLE).
2. Mọi phép chuẩn hóa (Fit Scaler) CHỈ được học từ tập Train, sau đó áp dụng (Transform) lên Val và Test.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from config import TRAIN_RATIO, VAL_RATIO, SCALER_TYPE

logger = logging.getLogger("Dataset")


@dataclass
class DatasetBundle:
    """Đóng gói đầy đủ dữ liệu chuỗi thời gian sau khi chia tách và chuẩn hóa."""
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Any
    feature_cols: List[str]


def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia DataFrame theo thứ tự thời gian tuyến tính thành 3 tập:
    - Train: từ 0 đến train_end
    - Validation: từ train_end đến val_end
    - Test: từ val_end đến hết chuỗi
    
    Đảm bảo shuffle=False để tránh rò rỉ thông tin tương lai về quá khứ.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy().reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    df_test = df.iloc[val_end:].copy().reset_index(drop=True)

    logger.info(
        "Phân chia tập dữ liệu theo thời gian: "
        "Train=%d mẫu (%s -> %s) | "
        "Val=%d mẫu (%s -> %s) | "
        "Test=%d mẫu (%s -> %s)",
        len(df_train), df_train["date"].iloc[0], df_train["date"].iloc[-1],
        len(df_val), df_val["date"].iloc[0], df_val["date"].iloc[-1],
        len(df_test), df_test["date"].iloc[0], df_test["date"].iloc[-1],
    )

    return df_train, df_val, df_test


def prepare_ml_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    scaler_type: str = SCALER_TYPE,
) -> DatasetBundle:
    """
    Quy trình chuẩn bị toàn diện dataset Machine Learning:
    1. Chia Train / Val / Test theo thứ tự thời gian.
    2. Khởi tạo Scaler (RobustScaler hoặc StandardScaler).
    3. Fit Scaler DUY NHẤT trên tập Train.
    4. Transform đặc trưng trên cả 3 tập Train, Val, Test.
    5. Đóng gói vào DatasetBundle.
    """
    # 1. Phân chia theo thời gian
    df_train, df_val, df_test = split_time_series(df, train_ratio=train_ratio, val_ratio=val_ratio)

    # 2. Khởi tạo Scaler
    if scaler_type.lower() == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    # 3. Fit scaler CHỈ trên tập Train (CHỐNG DATA LEAKAGE)
    scaler.fit(df_train[feature_cols])

    # 4. Transform lên các tập
    X_train = scaler.transform(df_train[feature_cols])
    X_val = scaler.transform(df_val[feature_cols])
    X_test = scaler.transform(df_test[feature_cols])

    y_train = df_train[target_col].to_numpy(dtype=int)
    y_val = df_val[target_col].to_numpy(dtype=int)
    y_test = df_test[target_col].to_numpy(dtype=int)

    return DatasetBundle(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        feature_cols=feature_cols,
    )
