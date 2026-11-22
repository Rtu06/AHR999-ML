"""
Module chiến lược định lượng (Strategy Engine):
1. Chiến lược Quant v1: Quy tắc chu kỳ AHR999 kết hợp động lượng SMA20 & RSI14.
2. Chiến lược Machine Learning v2: Ra quyết định phân bổ vị thế từ xác suất dự đoán P(Up)
   kèm dải trễ (Hysteresis Band) và bộ lọc phòng vệ vĩ mô AHR999.
"""

import logging
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np

from config import (
    AHR_ZONE_OPPORTUNITY,
    AHR_ZONE_DCA,
    AHR_ZONE_EXTREME_BUBBLE,
    RSI_MOMENTUM_FILTER,
    WEIGHT_OPPORTUNITY_MOMENTUM,
    WEIGHT_OPPORTUNITY_KNIFE,
    WEIGHT_DCA_BULL,
    WEIGHT_DCA_NEUTRAL,
    WEIGHT_DCA_DEFENSIVE,
    WEIGHT_RISK_MOMENTUM,
    WEIGHT_RISK_DEFENSIVE,
    WEIGHT_RISK_EXTREME,
    PROBA_BUY_THRESHOLD,
    PROBA_EXIT_THRESHOLD,
    ML_COMBINE_AHR_FILTER,
)

logger = logging.getLogger("Strategy")


# ==============================================================================
# CHIẾN LƯỢC QUANT V1 (BASELINE DỰA TRÊN QUY TẮC)
# ==============================================================================
def evaluate_daily_signal(
    close: float,
    ahr999: float,
    ahr_zone: str,
    sma_20: float,
    rsi_14: float,
) -> Tuple[float, str, str]:
    """
    Quy tắc định lượng phân bổ tỷ trọng cho từng phiên:
    Trả về: (target_weight, signal_code, description)
    """
    # 1. VÙNG CƠ HỘI ĐÁY (Opportunity Zone: ahr999 < 0.45)
    if ahr_zone == "opportunity" or ahr999 < AHR_ZONE_OPPORTUNITY:
        if close > sma_20:
            return (
                WEIGHT_OPPORTUNITY_MOMENTUM,
                "OPPORTUNITY_FULL",
                "Đáy chu kỳ + Xác nhận hồi phục trên SMA20 -> Vị thế tối đa 100%",
            )
        else:
            return (
                WEIGHT_OPPORTUNITY_KNIFE,
                "OPPORTUNITY_KNIFE",
                "Đáy chu kỳ + Dưới SMA20 (tránh dao rơi) -> Tích lũy 80%, giữ 20% tiền mặt",
            )

    # 2. VÙNG TÍCH LŨY ĐỊNH KỲ (DCA Zone: 0.45 <= ahr999 <= 1.20)
    elif ahr_zone == "dca" or (AHR_ZONE_OPPORTUNITY <= ahr999 <= AHR_ZONE_DCA):
        if close > sma_20 and rsi_14 >= RSI_MOMENTUM_FILTER:
            return (
                WEIGHT_DCA_BULL,
                "DCA_BULL_TREND",
                "Vùng DCA + Uptrend lành mạnh (Trên SMA20 & RSI>=45) -> Tỷ trọng 75%",
            )
        elif close > sma_20:
            return (
                WEIGHT_DCA_NEUTRAL,
                "DCA_NEUTRAL",
                "Vùng DCA + Trên SMA20 nhưng động lượng trung tính -> Tỷ trọng 50%",
            )
        else:
            return (
                WEIGHT_DCA_DEFENSIVE,
                "DCA_DEFENSIVE",
                "Vùng DCA + Dưới SMA20 (điều chỉnh ngắn hạn) -> Phòng thủ 35%",
            )

    # 3. VÙNG RỦI RO / ĐỊNH GIÁ QUÁ CAO (Risk Zone: ahr999 > 1.20)
    else:
        if ahr999 >= AHR_ZONE_EXTREME_BUBBLE:
            return (
                WEIGHT_RISK_EXTREME,
                "RISK_EXTREME_BUBBLE",
                "Đỉnh hưng phấn cực đại (ahr > 2.0) -> Chốt lời toàn bộ (0% BTC, 100% USDT)",
            )
        elif close > sma_20:
            return (
                WEIGHT_RISK_MOMENTUM,
                "RISK_RUNNER",
                "Vùng rủi ro (1.2 < ahr <= 2.0) + Trên SMA20 -> Giữ 25% runner theo trend",
            )
        else:
            return (
                WEIGHT_RISK_DEFENSIVE,
                "RISK_SCALE_OUT",
                "Vùng rủi ro + Gãy SMA20 -> Hạ tỷ trọng phòng thủ xuống 10%",
            )


def generate_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sinh tín hiệu chiến lược Quant v1 quy tắc luật chuẩn:
    - target_weight: Tỷ trọng BTC mong muốn sau khi đóng nến ngày t.
    """
    df = df.copy()

    weights = []
    signals = []
    reasons = []

    for _, row in df.iterrows():
        weight, sig, desc = evaluate_daily_signal(
            close=row["close"],
            ahr999=row["ahr999"],
            ahr_zone=row.get("ahr_zone", "unknown"),
            sma_20=row["sma_20"],
            rsi_14=row["rsi_14"],
        )
        weights.append(weight)
        signals.append(sig)
        reasons.append(desc)

    df["target_weight"] = weights
    df["signal_code"] = signals
    df["signal_desc"] = reasons

    return df


# Alias hỗ trợ gọi đối chiếu rõ nghĩa
generate_quant_v1_signals = generate_strategy_signals


# ==============================================================================
# CHIẾN LƯỢC MACHINE LEARNING V2
# ==============================================================================
def generate_ml_signals(
    df: pd.DataFrame,
    proba: Union[np.ndarray, pd.Series],
    buy_threshold: float = PROBA_BUY_THRESHOLD,
    exit_threshold: float = PROBA_EXIT_THRESHOLD,
    combine_ahr_filter: bool = ML_COMBINE_AHR_FILTER,
) -> pd.DataFrame:
    """
    Sinh tín hiệu giao dịch từ xác suất dự đoán P(Up) của mô hình ML:
    
    1. Dải trễ (Hysteresis Band):
       - P(Up) >= buy_threshold: Tăng vị thế (100% BTC).
       - P(Up) <= exit_threshold: Thoát vị thế (0% BTC / 100% USDT).
       - exit_threshold < P(Up) < buy_threshold: Giữ nguyên vị thế phiên trước (chống over-trading & phí).
       
    2. Bộ lọc bảo vệ chu kỳ vĩ mô (Macro AHR Guardrails - tùy chọn):
       - ahr999 >= 2.0 (Bong bóng cực đại): Bắt buộc hạ tỷ trọng về 0% chốt lời, phòng ngừa sụp đổ.
       - ahr999 < 0.45 (Đáy định giá): Đảm bảo tỷ trọng tối thiểu 50% để không bán đáy khi thị trường nhiễu.
    """
    df = df.copy()

    if isinstance(proba, pd.Series):
        proba_values = proba.values
    else:
        proba_values = np.asarray(proba)

    if len(proba_values) != len(df):
        raise ValueError(
            f"Kích thước proba ({len(proba_values)}) không khớp với DataFrame ({len(df)})."
        )

    df["ml_proba_up"] = proba_values

    target_weights = []
    signals = []
    reasons = []

    current_weight = 0.0

    for i in range(len(df)):
        prob = proba_values[i]
        row = df.iloc[i]
        ahr = row.get("ahr999", 1.0)

        # 1. Logic phân bổ vị thế cơ sở từ xác suất ML
        if prob >= buy_threshold:
            current_weight = 1.0
            sig = "ML_BUY"
            desc = f"Xác suất tăng P={prob:.2f} >= {buy_threshold:.2f} -> Full vị thế"
        elif prob <= exit_threshold:
            current_weight = 0.0
            sig = "ML_EXIT"
            desc = f"Xác suất tăng P={prob:.2f} <= {exit_threshold:.2f} -> Về tiền mặt"
        else:
            # Nằm trong vùng trung lập -> Giữ nguyên trạng thái phiên trước
            sig = "ML_HOLD"
            desc = f"Vùng trung lập P={prob:.2f} -> Giữ nguyên tỷ trọng {current_weight*100:.0f}%"

        # 2. Áp dụng Macro Guardrails từ AHR999 nếu được bật
        if combine_ahr_filter:
            if ahr >= AHR_ZONE_EXTREME_BUBBLE:
                current_weight = 0.0
                sig = "AHR_OVERRIDE_BUBBLE"
                desc = f"AHR999={ahr:.2f} >= 2.0 (Bong bóng cực đại) -> Cưỡng chế thoát về USDT"
            elif ahr < AHR_ZONE_OPPORTUNITY and current_weight < 0.50:
                current_weight = 0.80
                sig = "AHR_OVERRIDE_BOTTOM"
                desc = f"AHR999={ahr:.2f} < 0.45 (Đáy định giá siêu rẻ) -> Tích lũy tối thiểu 80%"

        target_weights.append(current_weight)
        signals.append(sig)
        reasons.append(desc)

    df["target_weight"] = target_weights
    df["signal_code"] = signals
    df["signal_desc"] = reasons

    return df
