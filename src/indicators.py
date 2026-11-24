"""
Module tính toán các chỉ số định lượng:
1. Chỉ số chu kỳ AHR999 (200-day Geometric Mean & Exponential Growth Line).
2. Phân loại vùng AHR999 (Opportunity, DCA, Risk).
3. Các chỉ số xu hướng / động lượng bổ trợ (SMA_20, SMA_50, SMA_200, RSI_14).
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd

from config import (
    GENESIS_DATE,
    AHR_EXP_A,
    AHR_EXP_B,
    WINDOW_COST_200D,
    AHR_ZONE_OPPORTUNITY,
    AHR_ZONE_DCA,
    SMA_TREND_WINDOW,
    SMA_50_WINDOW,
    SMA_200_WINDOW,
    RSI_WINDOW,
)

logger = logging.getLogger("Indicators")


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Tính chỉ số RSI (Relative Strength Index) theo phương pháp làm mượt hàm mũ (Wilder's Smoothing).
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing tương đương ewm với alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Tránh chia cho 0
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_ahr999_and_indicators(
    df: pd.DataFrame,
    genesis_date: str = GENESIS_DATE,
    exp_a: float = AHR_EXP_A,
    exp_b: float = AHR_EXP_B,
    window_cost: int = WINDOW_COST_200D,
    bottom_thresh: float = AHR_ZONE_OPPORTUNITY,
    dca_thresh: float = AHR_ZONE_DCA,
    sma_window: int = SMA_TREND_WINDOW,
    sma_50_window: int = SMA_50_WINDOW,
    sma_200_window: int = SMA_200_WINDOW,
    rsi_window: int = RSI_WINDOW,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Tính toán toàn diện chỉ số AHR999 và các chỉ số bổ trợ:
    
    1. cost_200d: Trung bình nhân (geometric mean) giá đóng cửa 200 ngày gần nhất:
       exp( mean( log(close) ) )
    2. days_since_genesis: Số ngày tính từ Genesis Block (2009-01-03).
    3. exp_growth: Đường tăng trưởng hàm mũ theo tuổi Bitcoin:
       10 ** (exp_a * log10(days_since_genesis) - exp_b)
    4. ahr999 = (close / cost_200d) * (close / exp_growth)
    5. ahr_zone:
       - 'opportunity' nếu ahr999 < bottom_thresh (0.36)
       - 'dca' nếu bottom_thresh <= ahr999 <= dca_thresh (0.36 - 1.00)
       - 'risk' nếu ahr999 > dca_thresh (> 1.00)
    6. sma_20, sma_50, sma_200 và rsi_14 dùng làm bộ lọc động lượng và quản trị rủi ro.
    """
    df = df.copy()

    # 1. Số ngày kể từ Genesis Block
    date_series = pd.to_datetime(df["date"])
    genesis_dt = pd.to_datetime(genesis_date)
    df["days_since_genesis"] = (date_series - genesis_dt).dt.days

    # 2. Chi phí bình quân hình học 200 ngày (200-day Geometric Mean Cost)
    # G-Mean = exp( rolling_mean( ln(Close) ) )
    log_close = np.log(df["close"])
    df["cost_200d"] = np.exp(log_close.rolling(window=window_cost, min_periods=window_cost).mean())

    # 3. Đường giá tăng trưởng hàm mũ (Exponential Growth Valuation)
    df["exp_growth"] = 10.0 ** (exp_a * np.log10(df["days_since_genesis"]) - exp_b)

    # 4. Chỉ số AHR999
    # ahr999 = (close / cost_200d) * (close / exp_growth)
    df["ahr999"] = (df["close"] / df["cost_200d"]) * (df["close"] / df["exp_growth"])

    # 5. Phân loại vùng AHR999 (ahr_zone)
    conditions = [
        df["ahr999"] < bottom_thresh,
        (df["ahr999"] >= bottom_thresh) & (df["ahr999"] <= dca_thresh),
        df["ahr999"] > dca_thresh,
    ]
    choices = ["opportunity", "dca", "risk"]
    df["ahr_zone"] = np.select(conditions, choices, default="unknown")

    # 6. Các chỉ báo kỹ thuật lọc xu hướng / động lượng
    df["sma_20"] = df["close"].rolling(window=sma_window, min_periods=sma_window).mean()
    df["sma_50"] = df["close"].rolling(window=sma_50_window, min_periods=sma_50_window).mean()
    df["sma_200"] = df["close"].rolling(window=sma_200_window, min_periods=sma_200_window).mean()
    df["rsi_14"] = calc_rsi(df["close"], period=rsi_window)

    # 7. Xử lý hàng thiếu giá trị
    initial_len = len(df)
    if drop_na:
        # Loại bỏ các dòng đầu chưa đủ 200 ngày để tính cost_200d / ahr999
        df = df.dropna(subset=["cost_200d", "ahr999"]).reset_index(drop=True)
        dropped_rows = initial_len - len(df)
        logger.info(
            "Đã loại bỏ %d dòng đầu tiên (chưa đủ %d ngày tính cost_200d). Còn lại %d dòng sẵn sàng phân tích.",
            dropped_rows,
            window_cost,
            len(df),
        )

    return df
