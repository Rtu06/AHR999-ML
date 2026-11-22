"""
Script so sánh hiệu suất định lượng trực tiếp trên tập kiểm định độc lập (Out-of-Sample Test Set):
1. ML Strategy v2: Mô hình Machine Learning dự đoán xác suất tăng giá theo chu kỳ (30 ngày).
2. Quant v1 Baseline: Chiến lược quy tắc vĩ mô AHR999 kết hợp SMA20 & RSI14.
3. Buy & Hold Benchmark: Nắm giữ 100% Bitcoin.

Đầu ra:
- In bảng so sánh định lượng 3 chiến lược chuẩn mực ra Console.
- Xuất biểu đồ trực quan hóa chuyên sâu 5 tầng ra output/ml_vs_quant_comparison.png.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    MODEL_FILE,
    DATA_ML_FEATURES_FILE,
    RAW_DATA_FILE,
    COMPARISON_CHART_FILE,
    INITIAL_CAPITAL,
    FEE_RATE,
    RISK_FREE_RATE,
    HORIZON_DAYS,
    TRAIN_RATIO,
    VAL_RATIO,
    PROBA_BUY_THRESHOLD,
    PROBA_EXIT_THRESHOLD,
    ML_COMBINE_AHR_FILTER,
    SCALER_TYPE,
)
from src.data_loader import load_btc_daily_data
from src.indicators import calculate_ahr999_and_indicators
from src.features import build_features_and_target
from src.dataset import prepare_ml_dataset
from src.model import load_model_bundle, predict_probabilities
from src.strategy import generate_quant_v1_signals, generate_ml_signals
from src.backtest import BacktestEngine, BacktestResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Compare")


def print_comparison_table(
    summary_ml: dict,
    summary_quant: dict,
    test_start: str,
    test_end: str,
    total_days: int,
    years: float,
) -> None:
    """In bảng so sánh hiệu suất 3 cột trực quan ra console."""
    sep = "=" * 94
    sub_sep = "-" * 94
    print("\n" + sep)
    print("      BÁO CÁO ĐỐI CHIẾU HIỆU SUẤT TRÊN TẬP TEST (OUT-OF-SAMPLE TEST SET)")
    print(sep)
    print(f" Khoảng thời gian Test : {test_start}  -->  {test_end} ({total_days} ngày / {years} năm)")
    print(f" Vốn ban đầu           : ${INITIAL_CAPITAL:,.2f}")
    print(f" Phí giao dịch         : {FEE_RATE * 100:.2f}% mỗi lượt tái cân bằng")
    print(sub_sep)
    print(
        f" {'CHỈ SỐ HIỆU SUẤT TÀI CHÍNH':<30} | {'ML STRATEGY (V2)':>18} | {'QUANT V1 (AHR999)':>18} | {'BUY & HOLD BTC':>18}"
    )
    print(sub_sep)
    print(
        f" {'Vốn cuối kỳ (Final Equity)':<30} | ${summary_ml['strat_final_equity']:>17,.2f} | ${summary_quant['strat_final_equity']:>17,.2f} | ${summary_ml['bnh_final_equity']:>17,.2f}"
    )
    print(
        f" {'Tổng lợi nhuận (Total Return)':<30} | {summary_ml['strat_total_return']:>17.2f}% | {summary_quant['strat_total_return']:>17.2f}% | {summary_ml['bnh_total_return']:>17.2f}%"
    )
    print(
        f" {'Lợi nhuận kép năm (CAGR)':<30} | {summary_ml['strat_cagr']:>17.2f}% | {summary_quant['strat_cagr']:>17.2f}% | {summary_ml['bnh_cagr']:>17.2f}%"
    )
    print(
        f" {'Độ biến động năm (Volatility)':<30} | {summary_ml['strat_volatility']:>17.2f}% | {summary_quant['strat_volatility']:>17.2f}% | {summary_ml['bnh_volatility']:>17.2f}%"
    )
    print(
        f" {'Sụt giảm tối đa (Max Drawdown)':<30} | {summary_ml['strat_max_dd']:>17.2f}% | {summary_quant['strat_max_dd']:>17.2f}% | {summary_ml['bnh_max_dd']:>17.2f}%"
    )
    print(
        f" {'Thời gian dưới đỉnh (DD Days)':<30} | {summary_ml['strat_max_dd_duration_days']:>13} ngày | {summary_quant['strat_max_dd_duration_days']:>13} ngày | {summary_ml['bnh_max_dd_duration_days']:>13} ngày"
    )
    print(
        f" {'Tỷ số Sharpe (rf=0%)':<30} | {summary_ml['strat_sharpe']:>18.2f} | {summary_quant['strat_sharpe']:>18.2f} | {summary_ml['bnh_sharpe']:>18.2f}"
    )
    print(
        f" {'Tỷ số Sortino':<30} | {summary_ml['strat_sortino']:>18.2f} | {summary_quant['strat_sortino']:>18.2f} | {summary_ml['bnh_sortino']:>18.2f}"
    )
    print(
        f" {'Tỷ số Calmar':<30} | {summary_ml['strat_calmar']:>18.2f} | {summary_quant['strat_calmar']:>18.2f} | {summary_ml['bnh_calmar']:>18.2f}"
    )
    print(sub_sep)
    print(
        f" {'Số lần tái cân bằng':<30} | {summary_ml['rebalance_count']:>18} | {summary_quant['rebalance_count']:>18} | {1:>18}"
    )
    print(
        f" {'Tổng phí giao dịch đã trả':<30} | ${summary_ml['total_fees_paid']:>17,.2f} | ${summary_quant['total_fees_paid']:>17,.2f} | ${INITIAL_CAPITAL * FEE_RATE:>17,.2f}"
    )
    print(
        f" {'Tỷ lệ ngày dương (Win Rate)':<30} | {summary_ml['daily_win_rate']:>17.1f}% | {summary_quant['daily_win_rate']:>17.1f}% | {'-':>18}"
    )
    print(sep + "\n")


def plot_comparison_chart(
    res_ml: BacktestResult,
    res_quant: BacktestResult,
    save_path: Path = COMPARISON_CHART_FILE,
) -> None:
    """Vẽ biểu đồ đối chiếu 5 tầng chuyên nghiệp chất lượng cao (300 DPI)."""
    df_ml = res_ml.df_equity.copy()
    df_quant = res_quant.df_equity.copy()

    df_ml["date_dt"] = pd.to_datetime(df_ml["date"])
    df_quant["date_dt"] = pd.to_datetime(df_quant["date"])

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4, 1.2, 2.5, 1.5]})

    # -------------------------------------------------------------
    # BẢNG 1: Giá BTC & Vùng AHR999
    # -------------------------------------------------------------
    ax0 = axes[0]
    ax0.plot(df_ml["date_dt"], df_ml["close"], label="BTC/USDT Close (Test Set)", color="#1e293b", linewidth=1.8)
    if "cost_200d" in df_ml.columns:
        ax0.plot(df_ml["date_dt"], df_ml["cost_200d"], label="200d Geometric Cost", color="#f59e0b", linestyle="--", linewidth=1.2)
    if "exp_growth" in df_ml.columns:
        ax0.plot(df_ml["date_dt"], df_ml["exp_growth"], label="Exp Growth Line", color="#8b5cf6", linestyle=":", linewidth=1.2)
    ax0.set_title("1. Diễn Biến Giá Bitcoin & Định Giá Chu Kỳ AHR999 (Tập Kiểm Định Test)", fontsize=13, fontweight="bold", pad=8)
    ax0.set_ylabel("Giá BTC (USD)", fontsize=10, fontweight="bold")
    ax0.legend(loc="upper left", frameon=True, fontsize=9)
    ax0.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # BẢNG 2: Xác suất dự đoán của Machine Learning P(Up)
    # -------------------------------------------------------------
    ax1 = axes[1]
    if "ml_proba_up" in df_ml.columns:
        ax1.plot(df_ml["date_dt"], df_ml["ml_proba_up"], label="P(Price Up in 30d)", color="#2563eb", linewidth=1.5)
        ax1.axhline(PROBA_BUY_THRESHOLD, color="#16a34a", linestyle="--", linewidth=1.2, label=f"Buy Threshold ({PROBA_BUY_THRESHOLD:.2f})")
        ax1.axhline(PROBA_EXIT_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.2, label=f"Exit Threshold ({PROBA_EXIT_THRESHOLD:.2f})")
        ax1.axhline(0.50, color="#94a3b8", linestyle=":", linewidth=1.0)
        ax1.fill_between(df_ml["date_dt"], df_ml["ml_proba_up"], 0.50, where=(df_ml["ml_proba_up"] >= 0.50), color="#16a34a", alpha=0.15)
        ax1.fill_between(df_ml["date_dt"], df_ml["ml_proba_up"], 0.50, where=(df_ml["ml_proba_up"] < 0.50), color="#dc2626", alpha=0.15)
    ax1.set_title("2. Tín Hiệu Dự Đoán Xác Suất Tăng Giá Từ Mô Hình ML (P(Up) & Dải Trễ Hysteresis)", fontsize=13, fontweight="bold", pad=8)
    ax1.set_ylabel("Xác suất P(Up)", fontsize=10, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # BẢNG 3: Tỷ trọng phân bổ tài sản (Target Weight)
    # -------------------------------------------------------------
    ax2 = axes[2]
    ax2.plot(df_ml["date_dt"], df_ml["actual_weight"], label="ML Strategy v2 Weight", color="#2563eb", linewidth=1.6)
    ax2.plot(df_quant["date_dt"], df_quant["actual_weight"], label="Quant v1 Rule Weight", color="#ea580c", linestyle="--", linewidth=1.4)
    ax2.set_title("3. Tỷ Trọng Phân Bổ Danh Mục Mục Tiêu (Actual Portfolio Weight: BTC vs. USDT)", fontsize=13, fontweight="bold", pad=8)
    ax2.set_ylabel("Tỷ trọng BTC", fontsize=10, fontweight="bold")
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # BẢNG 4: Đường cong vốn (Equity Curve)
    # -------------------------------------------------------------
    ax3 = axes[3]
    ax3.plot(df_ml["date_dt"], df_ml["strat_equity"], label=f"ML Strategy v2 (+{res_ml.summary['strat_total_return']:.1f}%)", color="#2563eb", linewidth=2.2)
    ax3.plot(df_quant["date_dt"], df_quant["strat_equity"], label=f"Quant v1 Baseline (+{res_quant.summary['strat_total_return']:.1f}%)", color="#ea580c", linewidth=1.8, linestyle="--")
    ax3.plot(df_ml["date_dt"], df_ml["bnh_equity"], label=f"Buy & Hold BTC (+{res_ml.summary['bnh_total_return']:.1f}%)", color="#64748b", linewidth=1.5, linestyle=":")
    ax3.set_title("4. So Sánh Đường Cong Tăng Trưởng Vốn (Equity Curve, Vốn Gốc $10,000 Đã Trừ Phí)", fontsize=13, fontweight="bold", pad=8)
    ax3.set_ylabel("Giá trị danh mục (USD)", fontsize=10, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True, fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # BẢNG 5: Mức sụt giảm từ đỉnh (Drawdown %)
    # -------------------------------------------------------------
    ax4 = axes[4]
    ax4.plot(df_ml["date_dt"], df_ml["strat_drawdown"] * 100, label=f"ML Strategy Max DD ({res_ml.summary['strat_max_dd']:.1f}%)", color="#2563eb", linewidth=1.6)
    ax4.plot(df_quant["date_dt"], df_quant["strat_drawdown"] * 100, label=f"Quant v1 Max DD ({res_quant.summary['strat_max_dd']:.1f}%)", color="#ea580c", linewidth=1.4, linestyle="--")
    ax4.plot(df_ml["date_dt"], df_ml["bnh_drawdown"] * 100, label=f"Buy & Hold Max DD ({res_ml.summary['bnh_max_dd']:.1f}%)", color="#dc2626", linewidth=1.2, linestyle=":")
    ax4.fill_between(df_ml["date_dt"], df_ml["strat_drawdown"] * 100, 0, color="#2563eb", alpha=0.1)
    ax4.set_title("5. So Sánh Mức Sụt Giảm Từ Đỉnh (Underwater Drawdown %)", fontsize=13, fontweight="bold", pad=8)
    ax4.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Thời gian", fontsize=10, fontweight="bold")
    ax4.legend(loc="lower left", frameon=True, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # Định dạng trục ngày tháng
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Đã xuất biểu đồ so sánh chi tiết ra: %s", save_path)


def run_comparison_pipeline() -> None:
    logger.info(">>> BẮT ĐẦU ĐỐI CHIẾU HIỆU SUẤT TRÊN TẬP TEST (OUT-OF-SAMPLE) <<<")

    # Kiểm tra xem file model đã tồn tại chưa
    if not MODEL_FILE.exists():
        logger.warning("Chưa tìm thấy model bundle tại: %s. Đang tự động kích hoạt train_and_eval...", MODEL_FILE)
        from train_and_eval import run_training_pipeline
        run_training_pipeline()

    # 1. Tải Model Bundle
    logger.info("Bước 1: Tải model bundle và scaler...")
    bundle_data = load_model_bundle(MODEL_FILE)
    model = bundle_data["model"]
    scaler = bundle_data["scaler"]
    feature_cols = bundle_data["feature_cols"]

    # 2. Chuẩn bị tập dữ liệu Test độc lập
    logger.info("Bước 2: Chuẩn bị dữ liệu và trích xuất tập Test độc lập...")
    if DATA_ML_FEATURES_FILE.exists():
        df_ml = pd.read_csv(DATA_ML_FEATURES_FILE)
    else:
        df_raw = load_btc_daily_data(force_download=False)
        df_ind = calculate_ahr999_and_indicators(df_raw)
        df_ml, _ = build_features_and_target(df_ind, horizon_days=HORIZON_DAYS)

    # Phân chia dữ liệu theo đúng tỷ lệ
    ds_bundle = prepare_ml_dataset(
        df=df_ml,
        feature_cols=feature_cols,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        scaler_type=SCALER_TYPE,
    )
    df_test = ds_bundle.df_test.copy()
    X_test = ds_bundle.X_test

    # 3. Dự đoán xác suất ML trên tập Test
    logger.info("Bước 3: Dự đoán xác suất tăng giá P(Up) trên tập Test...")
    y_prob_test = predict_probabilities(model, X_test)

    # 4. Sinh tín hiệu giao dịch cho 2 chiến lược trên cùng tập Test
    logger.info("Bước 4: Sinh tín hiệu phân bổ danh mục cho ML Strategy và Quant v1 Baseline...")
    df_test_ml = generate_ml_signals(
        df_test,
        proba=y_prob_test,
        buy_threshold=PROBA_BUY_THRESHOLD,
        exit_threshold=PROBA_EXIT_THRESHOLD,
        combine_ahr_filter=ML_COMBINE_AHR_FILTER,
    )
    df_test_quant = generate_quant_v1_signals(df_test)

    # 5. Khởi tạo Backtest Engine và mô phỏng giao dịch trừ phí thực tế
    logger.info("Bước 5: Chạy Backtest mô phỏng giao dịch trừ phí (%.2f%%)...", FEE_RATE * 100)
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        fee_rate=FEE_RATE,
        risk_free_rate=RISK_FREE_RATE,
    )
    res_ml = engine.run(df_test_ml)
    res_quant = engine.run(df_test_quant)

    # 6. In bảng so sánh 3 chiến lược
    test_start = df_test["date"].iloc[0]
    test_end = df_test["date"].iloc[-1]
    days = (pd.to_datetime(test_end) - pd.to_datetime(test_start)).days
    years = round(days / 365.25, 2)
    print_comparison_table(
        summary_ml=res_ml.summary,
        summary_quant=res_quant.summary,
        test_start=test_start,
        test_end=test_end,
        total_days=days,
        years=years,
    )

    # 7. Vẽ biểu đồ so sánh 5 tầng chuyên sâu
    logger.info("Bước 7: Xuất biểu đồ so sánh chuyên sâu...")
    plot_comparison_chart(res_ml, res_quant, save_path=COMPARISON_CHART_FILE)

    logger.info(">>> HOÀN TẤT ĐỐI CHIẾU THÀNH CÔNG <<<")


if __name__ == "__main__":
    run_comparison_pipeline()
