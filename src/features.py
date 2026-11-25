"""
Module kỹ nghệ đặc trưng (Feature Engineering) và sinh nhãn mục tiêu (Label Generation)
cho mô hình Machine Learning dự đoán xu hướng chu kỳ Bitcoin.

Đảm bảo nguyên tắc chống rò rỉ dữ liệu (No Data Leakage):
- Mọi đặc trưng X tại ngày t chỉ dùng thông tin <= t.
- Nhãn mục tiêu y nhìn về tương lai theo horizon_days (t + horizon).
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from config import HORIZON_DAYS

logger = logging.getLogger("Features")


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Sinh tập đặc trưng tương đối, ổn định qua các chu kỳ thị trường:
    
    1. Từ AHR999 & Chi phí 200 ngày:
       - ahr999
       - ahr_zone (one-hot: is_zone_opportunity, is_zone_dca, is_zone_risk)
       - price_to_cost_200d (close / cost_200d)
       - ahr999_pct_change qua 7d, 30d, 90d, 180d, 365d
       
    2. Từ Giá & Khối lượng (OHLCV):
       - return_1d, 7d, 30d, 90d, 180d, 365d (pct_change)
       - daily_range_pct: (high - low) / close
       - volatility_14d, volatility_30d: rolling std của return_1d
       - close_to_sma_20: close / sma_20
       - close_to_sma_50: close / sma_50
       - close_to_sma_200: close / sma_200
       - rsi_14
       - vol_to_ma_7d: quote_volume / quote_volume.rolling(7).mean()
       - vol_to_ma_30d: quote_volume / quote_volume.rolling(30).mean()
       
    Trả về:
    - df: DataFrame kèm các cột feature mới.
    - feature_cols: Danh sách tên cột feature sạch phục vụ huấn luyện.
    """
    df = df.copy()

    # Đảm bảo khối lượng quote_volume (nếu không có thì dùng volume * close)
    if "quote_volume" not in df.columns or df["quote_volume"].isna().all():
        df["quote_volume"] = df["volume"] * df["close"]

    # -------------------------------------------------------------
    # Nhóm 1: AHR999 & 200 Day Cost Features
    # -------------------------------------------------------------
    df["price_to_cost_200d"] = df["close"] / df["cost_200d"]

    # Mã hóa One-hot cho ahr_zone
    df["is_zone_opportunity"] = (df["ahr_zone"] == "opportunity").astype(float)
    df["is_zone_dca"] = (df["ahr_zone"] == "dca").astype(float)
    df["is_zone_risk"] = (df["ahr_zone"] == "risk").astype(float)

    # % Thay đổi của AHR999 qua các khung thời gian
    for period, name in [(7, "7d"), (30, "30d"), (90, "90d"), (180, "180d"), (365, "365d")]:
        df[f"ahr999_change_{name}"] = df["ahr999"].pct_change(period)

    # -------------------------------------------------------------
    # Nhóm 2: Tỷ suất sinh lời & Động lượng giá (Returns)
    # -------------------------------------------------------------
    df["return_1d"] = df["close"].pct_change(1)
    for period, name in [(7, "7d"), (30, "30d"), (90, "90d"), (180, "180d"), (365, "365d")]:
        df[f"return_{name}"] = df["close"].pct_change(period)

    # Biên độ dao động ngày
    df["daily_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # Độ biến động lịch sử (Rolling standard deviation của return 1d)
    df["volatility_14d"] = df["return_1d"].rolling(14).std()
    df["volatility_30d"] = df["return_1d"].rolling(30).std()

    # Tỷ lệ so với các đường trung bình động (Moving Average Ratios)
    df["close_to_sma_20"] = df["close"] / df["sma_20"]
    df["close_to_sma_50"] = df["close"] / df["sma_50"]
    df["close_to_sma_200"] = df["close"] / df["sma_200"]

    # Khối lượng giao dịch tương đối so với trung bình 7d & 30d
    vol_ma_7 = df["quote_volume"].rolling(7).mean()
    vol_ma_30 = df["quote_volume"].rolling(30).mean()
    df["vol_to_ma_7d"] = df["quote_volume"] / vol_ma_7.replace(0.0, np.nan)
    df["vol_to_ma_30d"] = df["quote_volume"] / vol_ma_30.replace(0.0, np.nan)

    # Danh sách các cột đặc trưng sạch
    feature_cols = [
        "ahr999",
        "price_to_cost_200d",
        "is_zone_opportunity",
        "is_zone_dca",
        "is_zone_risk",
        "ahr999_change_7d",
        "ahr999_change_30d",
        "ahr999_change_90d",
        "ahr999_change_180d",
        "ahr999_change_365d",
        "return_1d",
        "return_7d",
        "return_30d",
        "return_90d",
        "return_180d",
        "return_365d",
        "daily_range_pct",
        "volatility_14d",
        "volatility_30d",
        "close_to_sma_20",
        "close_to_sma_50",
        "close_to_sma_200",
        "rsi_14",
        "vol_to_ma_7d",
        "vol_to_ma_30d",
    ]

    return df, feature_cols


def build_features_and_target(
    df: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tạo bộ đặc trưng X và nhãn mục tiêu y (Binary Classification):
    
    Target tại ngày t:
    - y = 1 nếu Close[t + horizon_days] > Close[t] (Thị trường tăng giá)
    - y = 0 nếu Close[t + horizon_days] <= Close[t] (Thị trường giảm hoặc đi ngang)
    
    Quy tắc chống rò rỉ & xử lý dữ liệu:
    - Loại bỏ các dòng đầu chuỗi thiếu giá trị X (do rolling 365 ngày).
    - Loại bỏ các dòng cuối chuỗi thiếu nhãn y (do chưa đủ horizon_days ngày để nhìn về tương lai).
    """
    df, feature_cols = create_features(df)

    # Sinh nhãn mục tiêu nhìn trước horizon_days
    future_close = df["close"].shift(-horizon_days)
    df[f"future_return_{horizon_days}d"] = (future_close - df["close"]) / df["close"]
    df["target"] = (future_close > df["close"]).astype(float)
    # Gán NaN cho các dòng cuối không đủ dữ liệu nhìn trước
    df.loc[future_close.isna(), "target"] = np.nan

    # Lưu lại kích thước ban đầu
    initial_len = len(df)

    # Loại bỏ các dòng thiếu feature X (đầu chuỗi 365d),
    # nhưng GIỮ NGUYÊN các dòng cuối chuỗi dù thiếu nhãn target y (để phục vụ backtest/live trading đến ngày mới nhất)
    df_clean = df.dropna(subset=feature_cols).copy()
    df_clean = df_clean.reset_index(drop=True)

    dropped_count = initial_len - len(df_clean)
    logger.info(
        "Kỹ nghệ đặc trưng hoàn tất: %d dòng ban đầu -> loại bỏ %d dòng thiếu feature đầu chuỗi -> %d dòng sẵn sàng.",
        initial_len,
        dropped_count,
        len(df_clean),
    )

    return df_clean, feature_cols
