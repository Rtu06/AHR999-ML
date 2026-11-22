"""
Cấu hình hệ thống định lượng giao dịch BTC dựa trên chỉ số AHR999
và tầng Machine Learning dự đoán xu hướng chu kỳ vĩ mô.
File: config.py
"""

from pathlib import Path

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC & FILE
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# Đảm bảo các thư mục tồn tại
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_FILE = DATA_RAW_DIR / "btcusdt_1d.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "btcusdt_ahr999_processed.csv"
DATA_ML_FEATURES_FILE = DATA_PROCESSED_DIR / "btcusdt_ml_features.csv"
REPORT_CHART_FILE = OUTPUT_DIR / "backtest_report.png"
COMPARISON_CHART_FILE = OUTPUT_DIR / "ml_vs_quant_comparison.png"

# Đường dẫn lưu model và báo cáo ML
MODEL_FILE = MODELS_DIR / "best_model.joblib"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
METRICS_REPORT_FILE = DATA_PROCESSED_DIR / "ml_metrics_report.json"

# ==============================================================================
# CẤU HÌNH DỮ LIỆU BINANCE
# ==============================================================================
SYMBOL = "BTCUSDT"
INTERVAL = "1d"

# Danh sách API endpoints dự phòng của Binance
BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

# Ngày bắt đầu lấy dữ liệu Binance (2017-08-17 khi BTCUSDT niêm yết trên Binance)
BINANCE_START_DATE = "2017-08-17"

# ==============================================================================
# CẤU HÌNH CHỈ SỐ AHR999 & GENESIS BLOCK
# ==============================================================================
GENESIS_DATE = "2009-01-03"  # Ngày sinh Genesis Block của Bitcoin

# Tham số đường tăng trưởng hàm mũ (Exponential Growth Line):
# exp_growth = 10 ** (AHR_EXP_A * log10(days_since_genesis) - AHR_EXP_B)
AHR_EXP_A = 5.84
AHR_EXP_B = 17.01

# Chu kỳ trung bình nhân chi phí sản xuất / tích lũy
WINDOW_COST_200D = 200

# Ngưỡng phân loại vùng AHR999
AHR_ZONE_OPPORTUNITY = 0.45  # ahr999 < 0.45: Vùng đáy định giá siêu hời (Bottom / High Value)
AHR_ZONE_DCA = 1.20          # 0.45 <= ahr999 <= 1.20: Vùng tích lũy định kỳ (DCA Zone)
                             # ahr999 > 1.20: Vùng định giá cao / rủi ro quá nóng (Risk Zone)
AHR_ZONE_EXTREME_BUBBLE = 2.0  # Ngưỡng bong bóng hưng phấn cực đại (> 2.0)

# ==============================================================================
# CẤU HÌNH CHIẾN LƯỢC QUANT V1 (RULE-BASED STRATEGY)
# ==============================================================================
# Tham số kỹ thuật lọc xu hướng / động lượng để tránh bắt dao rơi
SMA_TREND_WINDOW = 20        # SMA 20 ngày xác nhận ngắn/trung hạn
SMA_50_WINDOW = 50           # SMA 50 ngày
SMA_200_WINDOW = 200         # SMA 200 ngày
RSI_WINDOW = 14              # Chu kỳ RSI 14 ngày
RSI_MOMENTUM_FILTER = 45.0   # Mức RSI phân định động lượng tích cực trong vùng DCA

# Tỷ trọng danh mục mục tiêu (Target Portfolio Weight) theo trạng thái thị trường
WEIGHT_OPPORTUNITY_MOMENTUM = 1.00   # ahr < 0.45 và Close > SMA20 (vùng đáy + có đà hồi phục)
WEIGHT_OPPORTUNITY_KNIFE = 0.80      # ahr < 0.45 và Close <= SMA20 (vùng đáy dao rơi, giữ 20% dự phòng)

WEIGHT_DCA_BULL = 0.75               # 0.45 <= ahr <= 1.20 và Close > SMA20 và RSI > 45 (uptrend lành mạnh)
WEIGHT_DCA_NEUTRAL = 0.50            # 0.45 <= ahr <= 1.20 và Close > SMA20 (đi ngang / đà yếu)
WEIGHT_DCA_DEFENSIVE = 0.35          # 0.45 <= ahr <= 1.20 và Close <= SMA20 (đang điều chỉnh)

WEIGHT_RISK_MOMENTUM = 0.25          # 1.20 < ahr <= 2.00 và Close > SMA20 (vùng nóng nhưng trend chưa gãy)
WEIGHT_RISK_DEFENSIVE = 0.10         # 1.20 < ahr <= 2.00 và Close <= SMA20 (trend suy yếu ở vùng rủi ro)
WEIGHT_RISK_EXTREME = 0.00           # ahr > 2.00: Đỉnh chu kỳ / hưng phấn tột độ -> 100% Tiền mặt (USDT)

# ==============================================================================
# CẤU HÌNH MACHINE LEARNING V2 (TẦNG DỰ ĐOÁN XÁC SUẤT XU HƯỚNG)
# ==============================================================================
# Khung thời gian nhìn trước dự đoán nhãn mục tiêu (Target Horizon):
# Hỗ trợ cấu hình: 7, 30, 90, 180, 365 ngày. Mặc định: 30 ngày (1 tháng).
HORIZON_DAYS = 30
SUPPORTED_HORIZONS = [7, 30, 90, 180, 365]

# Tỷ lệ phân chia chuỗi thời gian tuần tự (Strict Time-Series Order, NO SHUFFLE)
TRAIN_RATIO = 0.70           # 70% đầu tiên dùng để huấn luyện
VAL_RATIO = 0.15             # 15% tiếp theo dùng để validate / tinh chỉnh ngưỡng
TEST_RATIO = 0.15            # 15% cuối cùng giữ nguyên làm Out-Of-Sample kiểm định

# Thuật toán mô hình: "logistic", "rf" (Random Forest), "lgbm" (LightGBM)
# Tự động fallback sang HistGradientBoostingClassifier hoặc RandomForest nếu chưa cài LightGBM
MODEL_TYPE = "rf"

# Chuẩn hóa đặc trưng: "robust" (RobustScaler - bền vững với ngoại lai) hoặc "standard" (StandardScaler)
SCALER_TYPE = "robust"

# Ngưỡng xác suất ra quyết định vị thế (Hysteresis Banding để chống whipsaw)
PROBA_BUY_THRESHOLD = 0.55   # P(Up) >= 0.55 -> Mở vị thế / Tăng tỷ trọng
PROBA_EXIT_THRESHOLD = 0.45  # P(Up) <= 0.45 -> Thoát vị thế / Giảm tỷ trọng
                             # 0.45 < P(Up) < 0.55 -> Giữ nguyên tỷ trọng phiên trước (tránh tốn phí)

# Kết hợp bộ lọc vĩ mô AHR999 vào chiến lược ML (AHR Guardrails)
ML_COMBINE_AHR_FILTER = True

# Random seed cho tính tái lập
RANDOM_STATE = 42

# ==============================================================================
# CẤU HÌNH BACKTEST
# ==============================================================================
INITIAL_CAPITAL = 10_000.0   # Vốn ban đầu (USD/USDT)
FEE_RATE = 0.001             # Phí giao dịch Spot (0.1% = 0.001 mỗi lệnh đổi tỷ trọng)
RISK_FREE_RATE = 0.0         # Lãi suất phi rủi ro (dùng tính Sharpe ratio)
