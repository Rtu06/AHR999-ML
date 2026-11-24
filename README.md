# BTC-AHR999-ML-V2: Hệ Thống Định Lượng & Machine Learning Chu Kỳ Bitcoin

Dự án **BTC-AHR999-ML-V2** kế thừa nền tảng định lượng vĩ mô từ `btc-ahr999-quant-v1` và mở rộng thêm **tầng Machine Learning chu kỳ** nhằm dự đoán xác suất tăng giá theo khung thời gian (mặc định 30 ngày), kết hợp dải trễ (Hysteresis Banding) và bộ lọc bảo vệ vĩ mô (Macro Guardrails) để đối chiếu, nâng cấp chiến lược Quant thuần túy.

---

## 1. Kiến Trúc Hệ Thống Hai Tầng (Two-Tier Architecture)

```text
                     [Dữ Liệu Nến Ngày 1d BTCUSDT từ Binance]
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  TẦNG 1: ĐỊNH LƯỢNG CHU KỲ VĨ MÔ (Macro Cycle Quant Layer)                  │
 │  - Cost 200d (Geometric Mean): Đo lường chi phí vốn 200 ngày                │
 │  - Genesis Block (2009-01-03) & Exp Growth: Đường tăng trưởng hàm mũ        │
 │  - Chỉ số AHR999 & Phân vùng: Opportunity (< 0.36), DCA (0.36-1.0), Risk    │
 └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  TẦNG 2: MACHINE LEARNING XU HƯỚNG (Micro Machine Learning Layer)          │
 │  - 25 đặc trưng chu kỳ, lợi suất đa khung thời gian (7d - 365d), động lượng │
 │  - Chống rò rỉ dữ liệu (No Data Leakage) nghiêm ngặt                        │
 │  - Dự đoán xác suất tăng giá P(Up) sau 30 ngày                              │
 │  - Dải trễ Hysteresis: P >= 0.65 (Mua), P <= 0.45 (Bán), Giữ nguyên ở giữa │
 │  - Macro Guardrails: Chốt lời ép buộc khi AHR >= 1.6 (Bong bóng cực đại)   │
 └─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │  CÔNG CỤ KIỂM ĐỊNH & ĐỐI CHIẾU (Backtest & Benchmark Engine)                │
 │  - Mô phỏng thực thi không nhìn trước (actual_weight[t+1] = target[t])      │
 │  - Khấu trừ phí giao dịch thực tế (0.1% mỗi lần điều chỉnh tỷ trọng)       │
 │  - So sánh trực tiếp: ML Strategy v2 vs Quant v1 Baseline vs Buy & Hold     │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Cam Kết Nghiêm Ngặt Chống Rò Rỉ Dữ Liệu (No Data Leakage)

Trong bài toán tài chính chuỗi thời gian, lỗi rò rỉ thông tin tương lai (Data Leakage) là nguyên nhân hàng đầu khiến mô hình đạt điểm số ảo trên giấy nhưng thua lỗ khi thực chiến. Hệ thống tuân thủ 4 nguyên tắc bất biến:

1. **Đặc trưng chỉ nhìn về quá khứ ($X_t \le t$):**
   - Mọi đặc trưng (AHR999, SMA, RSI, biến động, lợi suất) tại phiên $t$ chỉ được tính toán từ các cây nến đã đóng $\le t$. Không dùng bất kỳ thông tin nào của tương lai.
2. **Nhãn mục tiêu nhìn trước có kiểm soát ($y_t = t + \text{horizon}$):**
   - Nhãn $y_t = 1$ nếu $\text{Close}_{t + 30} > \text{Close}_t$, ngược lại $0$.
   - Loại bỏ triệt để các phiên cuối chuỗi chưa đủ 30 ngày trong tương lai (không gán nhãn bừa bãi).
3. **Phân chia chuỗi thời gian tuần tự (Strict Time-Series Order, `shuffle=False`):**
   - Toàn bộ dữ liệu được cắt theo thứ tự thời gian tuyến tính:
     - **Train Set (70%)**: Giai đoạn đầu chuỗi để mô hình học các chu kỳ đã qua.
     - **Validation Set (15%)**: Giai đoạn tiếp theo để kiểm nghiệm và tinh chỉnh tham số.
     - **Test Set (15%)**: Giai đoạn Out-of-Sample hoàn toàn độc lập dùng để đối chiếu hiệu suất thực tế.
   - Tuyệt đối không xáo trộn ngẫu nhiên (K-Fold Shuffle) vì sẽ làm rò rỉ dữ liệu tương lai vào tập huấn luyện.
4. **Chuẩn hóa an toàn (Train-only Scaling):**
   - Bộ chuẩn hóa (`RobustScaler` / `StandardScaler`) **chỉ được `.fit()` trên tập Train**.
   - Sau đó chỉ áp dụng `.transform()` lên tập Validation và Test. Trung bình và phương sai của tập Test không bao giờ bị lộ cho mô hình khi huấn luyện.

---

## 3. Danh Mục Bộ Đặc Trưng (25 Features)

| Nhóm đặc trưng | Tên cột đặc trưng | Ý nghĩa định lượng |
| :--- | :--- | :--- |
| **AHR999 & Chi phí vốn** | `ahr999` | Định giá chu kỳ tương đối của Bitcoin |
| | `price_to_cost_200d` | Tỷ số giữa giá hiện tại và chi phí gom hàng 200 ngày |
| | `is_zone_opportunity`, `is_zone_dca`, `is_zone_risk` | Trạng thái vùng định giá chu kỳ AHR999 (One-hot) |
| | `ahr999_change_7d`, `30d`, `90d`, `180d`, `365d` | Tốc độ tăng tốc / hạ nhiệt của chỉ số AHR999 qua các chu kỳ |
| **Động lượng & Lợi suất** | `return_1d`, `7d`, `30d`, `90d`, `180d`, `365d` | Tỷ suất sinh lời đa khung thời gian |
| | `daily_range_pct` | Biên độ dao động ngày `(High - Low) / Close` |
| | `volatility_14d`, `volatility_30d` | Độ biến động lịch sử (Rolling standard deviation) |
| | `close_to_sma_20`, `close_to_sma_50`, `close_to_sma_200` | Khoảng cách giá tới các đường trung bình động ngắn, trung, dài hạn |
| | `rsi_14` | Chỉ số sức mạnh tương đối (Wilder's Smoothing) |
| **Khối lượng giao dịch** | `vol_to_ma_7d`, `vol_to_ma_30d` | Tỷ lệ khối lượng giao dịch so với trung bình 7 ngày và 30 ngày |

---

## 4. Cấu Trúc Thư Mục Dự Án

```text
btc-ahr999-ml-v2/
├── config.py                 # File cấu hình tham số AHR999, ML thresholds, tỷ lệ chia, phí
├── requirements.txt          # Danh sách thư viện Python phụ thuộc
├── train_and_eval.py         # Pipeline huấn luyện, đánh giá phân loại & lưu model bundle
├── run_compare.py            # Chạy backtest so sánh ML v2 vs Quant v1 vs Buy & Hold trên Test set
├── run_backtest.py           # Script backtest chiến lược Quant v1 truyền thống
├── README.md                 # Tài liệu hướng dẫn & giải thích hệ thống
├── data/
│   ├── raw/                  # Lưu cache nến thô 1d BTCUSDT từ Binance
│   └── processed/            # Lưu dữ liệu chỉ số AHR999, ML features & file JSON metrics
├── models/                   # Lưu trữ model bundle đã huấn luyện (.joblib)
├── output/                   # Lưu biểu đồ phân tích & so sánh (PNG 300 DPI)
└── src/
    ├── __init__.py           # Package marker
    ├── data_loader.py        # Tải nến từ Binance API (hỗ trợ mirrors, phân trang, cache)
    ├── indicators.py         # Tính Cost 200d, Exp Growth, AHR999, SMA20/50/200, RSI14
    ├── features.py           # Sinh 25 đặc trưng chu kỳ & nhãn target 30 ngày (No leakage)
    ├── dataset.py            # Phân chia chuỗi thời gian & chuẩn hóa an toàn (Train-only scaler)
    ├── model.py              # Logistic, Random Forest, LightGBM, tính metrics & joblib
    ├── strategy.py           # Bộ sinh tín hiệu Quant v1 & ML v2 (Hysteresis + AHR Guardrails)
    └── backtest.py           # Engine mô phỏng backtest trừ phí, Sharpe, Max Drawdown, CAGR
```

---

## 5. Hướng Dẫn Cài Đặt & Chạy

### Bước 1: Cài đặt thư viện phụ thuộc
Mở Terminal hoặc PowerShell tại thư mục dự án và chạy:

```bash
pip install -r requirements.txt
```

*(Lưu ý: Nếu không cài đặt `lightgbm`, hệ thống sẽ tự động fallback sang `HistGradientBoostingClassifier` hoặc `RandomForestClassifier` của `scikit-learn` mà không báo lỗi).*

---

### Bước 2: Huấn luyện mô hình Machine Learning & Đánh giá phân loại
Chạy script huấn luyện:

```bash
python train_and_eval.py
```

**Các tác vụ được thực thi tự động:**
1. Đọc dữ liệu nến Binance (offline cache hoặc tải online tự động).
2. Tính toán AHR999 và 25 đặc trưng chu kỳ.
3. Chia Train (70%), Val (15%), Test (15%) theo thời gian.
4. Fit scaler trên tập Train, chuẩn hóa các tập dữ liệu.
5. Huấn luyện mô hình Random Forest / LightGBM.
6. In bảng đánh giá hiệu năng (Accuracy, Precision, Recall, F1, ROC-AUC) trên 3 tập.
7. In bảng xếp hạng Top 10 đặc trưng quan trọng nhất.
8. Đóng gói mô hình và scaler vào `models/best_model.joblib`.
9. Lưu báo cáo JSON vào `data/processed/ml_metrics_report.json`.

---

### Bước 3: Chạy đối chiếu Out-of-Sample trên tập Test
Chạy script so sánh trực tiếp:

```bash
python run_compare.py
```

**Kết quả thu được:**
1. In bảng đối chiếu định lượng 3 chiến lược song song:
   - **ML Strategy (v2)**
   - **Quant v1 (AHR999 Baseline)**
   - **Buy & Hold BTC (Benchmark)**
2. Xuất biểu đồ trực quan hóa chuyên sâu 5 tầng ra `output/ml_vs_quant_comparison.png`.

---

## 6. Hướng Dẫn Đọc Báo Cáo & Các Chỉ Số Định Lượng

| Chỉ số | Ý nghĩa & Tiêu chuẩn đánh giá |
| :--- | :--- |
| **Total Return (%)** | Tổng tỷ suất sinh lời trên toàn bộ giai đoạn kiểm định Test. |
| **CAGR (%)** | Tỷ suất lợi nhuận kép hàng năm (Compound Annual Growth Rate). |
| **Max Drawdown (Max DD %)** | Mức giảm tối đa từ đỉnh tài khoản gần nhất. Chỉ số càng thấp, rủi ro cháy tài khoản càng nhỏ. |
| **Max DD Duration** | Số ngày tài khoản nằm dưới đỉnh (thước đo độ kiên nhẫn cần có của nhà đầu tư). |
| **Sharpe Ratio** | Tỷ lệ lợi nhuận vượt trội trên mỗi đơn vị rủi ro tổng thể. Sharpe $> 1.0$ là tốt, $> 2.0$ là xuất sắc. |
| **Sortino Ratio** | Chỉ phạt các biến động giảm tiêu cực (Downside risk). Thích hợp đánh giá tài sản có độ lệch dương lớn như BTC. |
| **Calmar Ratio** | Tỷ lệ giữa `CAGR / abs(Max DD)`. Đo lường mức độ tương xứng giữa lợi nhuận và mức sụt giảm nặng nhất. |
| **Rebalance Count & Fees** | Số lần đổi tỷ trọng và tổng phí giao dịch đã trừ (đo lường độ chịu nhiệt với chi phí trượt giá & phí sàn). |

---

## 7. Giới Hạn Của Mô Hình & Khuyến Cáo Rủi Ro

1. **Tính chất Non-Stationary của Crypto:** Thị trường tiền mã hóa biến đổi cấu trúc liên tục qua các chu kỳ Halving, sự tham gia của quỹ ETF và dòng tiền thể chế. Các quy luật trong quá khứ có thể suy giảm hiệu quả trong tương lai.
2. **Chi phí trượt giá & Phí Funding:** Backtest tính phí Spot tiêu chuẩn $0.1\%$. Trong các giai đoạn thị trường flash-crash hoặc thanh khoản mỏng, trượt giá thực tế (slippage) có thể cao hơn.
3. **Mô hình định lượng là công cụ hỗ trợ quyết định:** Tầng ML đóng vai trò bộ lọc xác suất động lượng, kết hợp cùng kỷ luật quản trị vốn và ngưỡng định giá chu kỳ AHR999 để phòng ngừa rủi ro thiên nga đen.
