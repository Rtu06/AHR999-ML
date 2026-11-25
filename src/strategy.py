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
    SMA_BUFFER_PCT,
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
    HYBRID_TILT_WEIGHT,
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
    above_sma: Optional[bool] = None,
) -> Tuple[float, str, str]:
    """
    Quy tắc định lượng phân bổ tỷ trọng cho từng phiên:
    Hỗ trợ dải trễ (Hysteresis Buffer) cho đường SMA20 để chống whipsaw.
    Trả về: (target_weight, signal_code, description)
    """
    if above_sma is None:
        above_sma = close > sma_20

    # 1. VÙNG CƠ HỘI ĐÁY (Opportunity Zone: ahr999 < 0.36)
    if ahr_zone == "opportunity" or ahr999 < AHR_ZONE_OPPORTUNITY:
        if above_sma:
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

    # 2. VÙNG TÍCH LŨY ĐỊNH KỲ (DCA Zone: 0.36 <= ahr999 <= 1.00)
    elif ahr_zone == "dca" or (AHR_ZONE_OPPORTUNITY <= ahr999 <= AHR_ZONE_DCA):
        if above_sma and rsi_14 >= RSI_MOMENTUM_FILTER:
            return (
                WEIGHT_DCA_BULL,
                "DCA_BULL_TREND",
                "Vùng DCA + Uptrend lành mạnh (Trên SMA20 & RSI>=45) -> Tỷ trọng 75%",
            )
        elif above_sma:
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

    # 3. VÙNG RỦI RO / ĐỊNH GIÁ QUÁ CAO (Risk Zone: ahr999 > 1.00)
    else:
        if ahr999 >= AHR_ZONE_EXTREME_BUBBLE:
            return (
                WEIGHT_RISK_EXTREME,
                "RISK_EXTREME_BUBBLE",
                "Đỉnh hưng phấn cực đại (ahr > 1.6) -> Chốt lời toàn bộ (0% BTC, 100% USDT)",
            )
        elif above_sma:
            return (
                WEIGHT_RISK_MOMENTUM,
                "RISK_RUNNER",
                "Vùng rủi ro (1.0 < ahr <= 1.6) + Trên SMA20 -> Giữ 25% runner theo trend",
            )
        else:
            return (
                WEIGHT_RISK_DEFENSIVE,
                "RISK_SCALE_OUT",
                "Vùng rủi ro + Gãy SMA20 -> Hạ tỷ trọng phòng thủ xuống 10%",
            )


def generate_strategy_signals(
    df: pd.DataFrame,
    sma_buffer_pct: float = SMA_BUFFER_PCT,
) -> pd.DataFrame:
    """
    Sinh tín hiệu chiến lược Quant v1 quy tắc luật chuẩn có bộ đệm chống whipsaw:
    - target_weight: Tỷ trọng BTC mong muốn sau khi đóng nến ngày t.
    """
    df = df.copy()

    weights = []
    signals = []
    reasons = []

    above_sma = False

    for _, row in df.iterrows():
        close = row["close"]
        sma_20 = row["sma_20"]

        if sma_buffer_pct > 0.0:
            if close > sma_20 * (1.0 + sma_buffer_pct):
                above_sma = True
            elif close < sma_20 * (1.0 - sma_buffer_pct):
                above_sma = False
        else:
            above_sma = close > sma_20

        weight, sig, desc = evaluate_daily_signal(
            close=close,
            ahr999=row["ahr999"],
            ahr_zone=row.get("ahr_zone", "unknown"),
            sma_20=sma_20,
            rsi_14=row["rsi_14"],
            above_sma=above_sma,
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
       - ahr999 >= 1.6 (Bong bóng cực đại): Bắt buộc hạ tỷ trọng về 0% chốt lời, phòng ngừa sụp đổ.
       - ahr999 < 0.36 (Đáy định giá): Đảm bảo tỷ trọng tối thiểu 50% để không bán đáy khi thị trường nhiễu.
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
                desc = f"AHR999={ahr:.2f} >= {AHR_ZONE_EXTREME_BUBBLE:.1f} (Bong bóng cực đại) -> Cưỡng chế thoát về USDT"
            elif ahr < AHR_ZONE_OPPORTUNITY and current_weight < 0.50:
                current_weight = 0.80
                sig = "AHR_OVERRIDE_BOTTOM"
                desc = f"AHR999={ahr:.2f} < {AHR_ZONE_OPPORTUNITY:.2f} (Đáy định giá siêu rẻ) -> Tích lũy tối thiểu 80%"

        target_weights.append(current_weight)
        signals.append(sig)
        reasons.append(desc)

    df["target_weight"] = target_weights
    df["signal_code"] = signals
    df["signal_desc"] = reasons

    return df


# ==============================================================================
# CHIẾN LƯỢC LAI HYBRID V3 (QUANT BASELINE + ML CONFIDENCE TILT)
# ==============================================================================
def generate_hybrid_signals(
    df: pd.DataFrame,
    proba: Union[np.ndarray, pd.Series],
    buy_threshold: float = PROBA_BUY_THRESHOLD,
    exit_threshold: float = PROBA_EXIT_THRESHOLD,
    tilt_weight: float = HYBRID_TILT_WEIGHT,
    combine_ahr_filter: bool = ML_COMBINE_AHR_FILTER,
    sma_buffer_pct: float = SMA_BUFFER_PCT,
) -> pd.DataFrame:
    """
    Chiến lược lai Hybrid V3 kết hợp thế mạnh của 2 tầng:
    1. Lấy tỷ trọng danh mục của Quant v1 (có SMA Hysteresis Buffer) làm nền tảng (Anchor).
    2. Tinh chỉnh tỷ trọng động dựa trên mức độ tin cậy của xác suất ML P(Up):
       - P(Up) >= buy_threshold: Tăng thêm +tilt_weight (+15%) vào vị thế.
       - P(Up) <= exit_threshold: Giảm bớt -tilt_weight (-15%) khỏi vị thế.
       - Vùng trung lập: Giữ nguyên tỷ trọng nền tảng Quant v1.
    3. Giới hạn tỷ trọng trong [0.0, 1.0] và tuân thủ tuyệt đối Macro AHR Guardrails:
       - AHR >= 1.6: Ép thoát 100% USDT (Bảo toàn vốn ở đỉnh bong bóng).
       - AHR < 0.36: Ép gom tối thiểu 80% BTC (Chống bán tháo ở đáy chu kỳ).
    """
    df = df.copy()

    # Tính tỷ trọng nền tảng Quant v1
    df_quant = generate_strategy_signals(df, sma_buffer_pct=sma_buffer_pct)
    quant_weights = df_quant["target_weight"].values

    if isinstance(proba, pd.Series):
        proba_values = proba.values
    else:
        proba_values = np.asarray(proba)

    if len(proba_values) != len(df):
        raise ValueError(
            f"Kích thước proba ({len(proba_values)}) không khớp với DataFrame ({len(df)})."
        )

    df["ml_proba_up"] = proba_values
    df["quant_base_weight"] = quant_weights

    target_weights = []
    signals = []
    reasons = []

    for i in range(len(df)):
        prob = proba_values[i]
        base_w = quant_weights[i]
        row = df.iloc[i]
        ahr = row.get("ahr999", 1.0)

        # 1. Tinh chỉnh từ xác suất ML
        if prob >= buy_threshold:
            w = base_w + tilt_weight
            sig = "HYBRID_BOOST"
            desc = f"Quant={base_w*100:.0f}% + ML Boost (+{tilt_weight*100:.0f}%, P={prob:.2f}) -> {w*100:.0f}%"
        elif prob <= exit_threshold:
            w = base_w - tilt_weight
            sig = "HYBRID_TRIM"
            desc = f"Quant={base_w*100:.0f}% - ML Trim (-{tilt_weight*100:.0f}%, P={prob:.2f}) -> {w*100:.0f}%"
        else:
            w = base_w
            sig = "HYBRID_NEUTRAL"
            desc = f"Quant={base_w*100:.0f}% (ML trung tính P={prob:.2f})"

        # 2. Cắt biên [0.0, 1.0]
        w = max(0.0, min(1.0, w))

        # 3. Áp dụng Macro Guardrails
        if combine_ahr_filter:
            if ahr >= AHR_ZONE_EXTREME_BUBBLE:
                w = 0.0
                sig = "AHR_OVERRIDE_BUBBLE"
                desc = f"AHR999={ahr:.2f} >= {AHR_ZONE_EXTREME_BUBBLE:.1f} (Bong bóng cực đại) -> Cưỡng chế thoát về USDT"
            elif ahr < AHR_ZONE_OPPORTUNITY and w < 0.80:
                w = 0.80
                sig = "AHR_OVERRIDE_BOTTOM"
                desc = f"AHR999={ahr:.2f} < {AHR_ZONE_OPPORTUNITY:.2f} (Đáy định giá siêu rẻ) -> Gom tối thiểu 80%"

        target_weights.append(w)
        signals.append(sig)
        reasons.append(desc)

    df["target_weight"] = target_weights
    df["signal_code"] = signals
    df["signal_desc"] = reasons

    return df
