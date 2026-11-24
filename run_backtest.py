"""
Script thực thi toàn bộ pipeline định lượng BTC AHR999:
1. Tải dữ liệu nến 1d Binance API (hoặc đọc cache offline).
2. Tính toán AHR999 (200d Geometric Mean, Exp Growth) & chỉ số kỹ thuật.
3. Sinh tín hiệu và tỷ trọng phân bổ danh mục theo quy tắc định lượng.
4. Backtest mô phỏng giao dịch trừ phí thực tế.
5. Lưu dữ liệu đã xử lý vào data/processed/.
6. In bảng so sánh hiệu suất ra màn hình console.
7. Vẽ biểu đồ trực quan hóa chuyên sâu và xuất ra output/backtest_report.png.
"""

import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from config import (
    SYMBOL,
    INTERVAL,
    PROCESSED_DATA_FILE,
    REPORT_CHART_FILE,
    INITIAL_CAPITAL,
    FEE_RATE,
    RISK_FREE_RATE,
    AHR_ZONE_OPPORTUNITY,
    AHR_ZONE_DCA,
)
from src.data_loader import load_btc_daily_data
from src.indicators import calculate_ahr999_and_indicators
from src.strategy import generate_strategy_signals
from src.backtest import BacktestEngine, BacktestResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")


def print_performance_table(summary: dict) -> None:
    """In bảng so sánh hiệu suất định lượng chuẩn mực ra console."""
    sep = "=" * 82
    sub_sep = "-" * 82
    print("\n" + sep)
    print(f"       BÁO CÁO HIỆU SUẤT ĐỊNH LƯỢNG CHIẾN LƯỢC AHR999 QUANT V1 ({SYMBOL})")
    print(sep)
    print(f" Giai đoạn kiểm định : {summary['start_date']}  -->  {summary['end_date']} ({summary['total_days']} ngày / {summary['years']} năm)")
    print(f" Vốn ban đầu         : ${summary['initial_capital']:,.2f}")
    print(f" Phí giao dịch       : {FEE_RATE * 100:.2f}% mỗi lượt tái cân bằng")
    print(sub_sep)
    print(f" {'CHỈ SỐ TÀI CHÍNH':<32} | {'AHR999 STRATEGY':>20} | {'BUY & HOLD BTC':>22}")
    print(sub_sep)
    print(f" {'Vốn cuối kỳ (Final Equity)':<32} | ${summary['strat_final_equity']:>19,.2f} | ${summary['bnh_final_equity']:>21,.2f}")
    print(f" {'Tổng lợi nhuận (Total Return)':<32} | {summary['strat_total_return']:>19.2f}% | {summary['bnh_total_return']:>21.2f}%")
    print(f" {'Lợi nhuận kép năm (CAGR)':<32} | {summary['strat_cagr']:>19.2f}% | {summary['bnh_cagr']:>21.2f}%")
    print(f" {'Độ biến động năm (Volatility)':<32} | {summary['strat_volatility']:>19.2f}% | {summary['bnh_volatility']:>21.2f}%")
    print(f" {'Mức sụt giảm tối đa (Max DD)':<32} | {summary['strat_max_dd']:>19.2f}% | {summary['bnh_max_dd']:>21.2f}%")
    print(f" {'Thời gian dưới đỉnh (Max DD Days)':<32} | {summary['strat_max_dd_duration_days']:>15} ngày | {summary['bnh_max_dd_duration_days']:>17} ngày")
    print(f" {'Tỷ số Sharpe (rf=0%)':<32} | {summary['strat_sharpe']:>20.2f} | {summary['bnh_sharpe']:>22.2f}")
    print(f" {'Tỷ số Sortino':<32} | {summary['strat_sortino']:>20.2f} | {summary['bnh_sortino']:>22.2f}")
    print(f" {'Tỷ số Calmar':<32} | {summary['strat_calmar']:>20.2f} | {summary['bnh_calmar']:>22.2f}")
    print(sub_sep)
    print(f" {'Số phiên tái cân bằng':<32} | {summary['rebalance_count']:>20} | {1:>22}")
    print(f" {'Tổng phí giao dịch đã trả':<32} | ${summary['total_fees_paid']:>19,.2f} | ${summary['initial_capital'] * FEE_RATE:>21,.2f}")
    print(f" {'Tỷ lệ ngày dương (Daily Win Rate)':<32} | {summary['daily_win_rate']:>19.1f}% | {'-':>22}")
    print(sep + "\n")


def plot_backtest_report(res: BacktestResult, save_path: Path = REPORT_CHART_FILE) -> None:
    """
    Vẽ báo cáo trực quan 4 bảng chuyên nghiệp:
    1. Giá BTC vs Giá định giá hàm mũ & Chi phí 200 ngày (Geometric Mean).
    2. Chỉ số AHR999 và các vùng phân định (Opportunity, DCA, Risk).
    3. Tỷ trọng phân bổ danh mục (Asset Allocation %) theo thời gian.
    4. So sánh đường cong vốn (Equity Curve) & Drawdown giữa Chiến lược và Buy & Hold.
    """
    df = res.df_equity.copy()
    df["dt"] = pd.to_datetime(df["date"])

    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.8, 3.2]})
    plt.subplots_adjust(hspace=0.08)

    # --------------------------------------------------------------------------
    # SUBPLOT 1: Giá BTC vs 200d Geometric Cost vs Exponential Growth Curve
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(df["dt"], df["close"], label="Giá BTC (Close)", color="#1f77b4", linewidth=1.5)
    ax1.plot(df["dt"], df["cost_200d"], label="Chi phí 200d G-Mean (cost_200d)", color="#ff7f0e", linestyle="--", linewidth=1.3)
    ax1.plot(df["dt"], df["exp_growth"], label="Định giá hàm mũ (exp_growth)", color="#2ca02c", linestyle=":", linewidth=1.3)
    ax1.set_yscale("log")
    ax1.set_ylabel("Giá BTC / USD (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Mô hình Định lượng Chu kỳ Bitcoin AHR999 ({SYMBOL}) - 1D Framework", fontsize=14, fontweight="bold", pad=12)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", framealpha=0.9)

    # --------------------------------------------------------------------------
    # SUBPLOT 2: Chỉ số AHR999 & Phân vùng định giá
    # --------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(df["dt"], df["ahr999"], label="Chỉ số AHR999", color="#9467bd", linewidth=1.4)
    ax2.axhline(AHR_ZONE_OPPORTUNITY, color="green", linestyle="--", linewidth=1.2, label=f"Ngưỡng Đáy: {AHR_ZONE_OPPORTUNITY}")
    ax2.axhline(AHR_ZONE_DCA, color="orange", linestyle="--", linewidth=1.2, label=f"Ngưỡng DCA: {AHR_ZONE_DCA}")

    # Tô màu các vùng
    ax2.axhspan(0, AHR_ZONE_OPPORTUNITY, color="green", alpha=0.15, label=f"Vùng Đáy Siêu Hời (<{AHR_ZONE_OPPORTUNITY})")
    ax2.axhspan(AHR_ZONE_OPPORTUNITY, AHR_ZONE_DCA, color="gold", alpha=0.15, label=f"Vùng Tích Lũy DCA ({AHR_ZONE_OPPORTUNITY}-{AHR_ZONE_DCA:.2f})")
    ax2.axhspan(AHR_ZONE_DCA, max(df["ahr999"].max() * 1.05, 3.5), color="red", alpha=0.15, label=f"Vùng Rủi Ro / Quá Nhiệt (>{AHR_ZONE_DCA:.2f})")

    ax2.set_ylabel("Chỉ số AHR999", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, min(df["ahr999"].quantile(0.995) * 1.3, 5.0))
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", ncol=3, fontsize=9, framealpha=0.9)

    # --------------------------------------------------------------------------
    # SUBPLOT 3: Tỷ trọng phân bổ danh mục (Target Weight)
    # --------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.step(df["dt"], df["actual_weight"] * 100, where="post", color="#2ca02c", linewidth=1.2, label="Tỷ trọng BTC nắm giữ (%)")
    ax3.fill_between(df["dt"], df["actual_weight"] * 100, step="post", alpha=0.25, color="#2ca02c")
    ax3.set_ylabel("Tỷ trọng (%)", fontsize=11, fontweight="bold")
    ax3.set_ylim(-5, 105)
    ax3.set_yticks([0, 25, 50, 75, 100])
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", framealpha=0.9)

    # --------------------------------------------------------------------------
    # SUBPLOT 4: Đường cong vốn (Equity Curve) & Drawdown
    # --------------------------------------------------------------------------
    ax4 = axes[3]
    ax4.plot(df["dt"], df["strat_equity"], label=f"Chiến lược AHR999 (Final: ${res.summary['strat_final_equity']:,.0f})", color="#d62728", linewidth=1.8)
    ax4.plot(df["dt"], df["bnh_equity"], label=f"Buy & Hold BTC (Final: ${res.summary['bnh_final_equity']:,.0f})", color="#7f7f7f", linestyle="--", linewidth=1.3)
    ax4.set_ylabel("Giá trị Vốn / USD", fontsize=11, fontweight="bold")
    ax4.set_yscale("log")
    ax4.grid(True, which="both", linestyle="--", alpha=0.4)
    ax4.legend(loc="upper left", framealpha=0.9)

    # Subplot Drawdown thu nhỏ dạng twinx hoặc bảng phụ
    ax4_dd = ax4.twinx()
    ax4_dd.plot(df["dt"], df["strat_drawdown"] * 100, color="#d62728", alpha=0.25, linewidth=0.8)
    ax4_dd.plot(df["dt"], df["bnh_drawdown"] * 100, color="#7f7f7f", alpha=0.20, linestyle=":", linewidth=0.8)
    ax4_dd.fill_between(df["dt"], df["strat_drawdown"] * 100, 0, color="#d62728", alpha=0.1)
    ax4_dd.set_ylabel("Drawdown (%) [Vùng mờ]", color="gray", fontsize=10)
    ax4_dd.set_ylim(-110, 10)

    # Định dạng trục X (thời gian)
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.set_xlabel("Năm", fontsize=11, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Đã xuất biểu đồ báo cáo hiệu suất tại: %s", save_path)


def main():
    logger.info("Khởi động hệ thống Quant AHR999 BTC...")

    # BƯỚC 1: Tải dữ liệu Binance
    df_raw = load_btc_daily_data()
    logger.info("Dữ liệu thô: %d phiên từ %s đến %s", len(df_raw), df_raw["date"].iloc[0], df_raw["date"].iloc[-1])

    # BƯỚC 2: Tính toán chỉ số AHR999 và bộ lọc động lượng
    df_indicators = calculate_ahr999_and_indicators(df_raw, drop_na=True)

    # BƯỚC 3: Sinh tín hiệu và tỷ trọng phân bổ
    df_strategy = generate_strategy_signals(df_indicators)

    # BƯỚC 4: Chạy Backtest mô phỏng giao dịch
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        fee_rate=FEE_RATE,
        risk_free_rate=RISK_FREE_RATE,
    )
    result = engine.run(df_strategy)

    # BƯỚC 5: Lưu dữ liệu đã xử lý vào data/processed/
    result.df_equity.to_csv(PROCESSED_DATA_FILE, index=False)
    logger.info("Đã lưu dữ liệu định lượng vào: %s", PROCESSED_DATA_FILE)

    # BƯỚC 6: In bảng thống kê hiệu suất ra console
    print_performance_table(result.summary)

    # BƯỚC 7: Vẽ và lưu biểu đồ báo cáo trực quan
    plot_backtest_report(result, REPORT_CHART_FILE)
    logger.info("Hoàn tất quy trình định lượng thành công!")


if __name__ == "__main__":
    main()
