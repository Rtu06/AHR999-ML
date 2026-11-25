# BTC-AHR999-ML: Hệ Thống Định Lượng & Machine Learning Chu Kỳ Bitcoin

Dự án **BTC-AHR999-ML** là hệ thống định lượng giao dịch và quản trị danh mục Bitcoin đa tầng, kết hợp giữa **nền tảng định lượng chu kỳ vĩ mô (Macro AHR999)**, **bộ đệm động lượng chống giật nhiễu (SMA Hysteresis Buffer)** và **tầng Machine Learning dự đoán xác suất xu hướng (Micro ML Layer)** kèm cơ chế dải trễ (Hysteresis Banding) và bộ lọc bảo vệ vĩ mô (Macro Guardrails).

---

## 1. Kiến Trúc Hệ Thống Ba Tầng (Three-Tier Architecture)

```text
                     [Dữ Liệu Nến Ngày 1d BTCUSDT từ Binance]
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  TẦNG 1: ĐỊNH LƯỢNG CHU KỲ VĨ MÔ & BỘ ĐỆM SMA (Macro Quant Layer)          │
  │  - Cost 200d (Geometric Mean): Đo lường chi phí vốn 200 ngày                │
  │  - Genesis Block (2009-01-03) & Exp Growth: Đường tăng trưởng hàm mũ        │
  │  - Chỉ số AHR999: Đáy (< 0.36), DCA (0.36 - 1.00), Quá nhiệt (> 1.00)       │
  │  - Bộ đệm SMA20 Buffer (1.5%): Chống nhiễu whipsaw, cắt giảm 30% phí lệnh  │
  └─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  TẦNG 2: MACHINE LEARNING XU HƯỚNG (Micro Machine Learning Layer)          │
  │  - 25 đặc trưng chu kỳ, lợi suất đa khung thời gian (7d - 365d), động lượng │
  │  - Mô hình Regularized Logistic Regression (L2, C=0.05, balanced weights)   │
  │  - Chống rò rỉ dữ liệu (No Data Leakage) & Khống chế Overfitting (AUC ~67%)│
  │  - Dải trễ Hysteresis: P >= 0.65 (Mua), P <= 0.45 (Bán), Giữ nguyên ở giữa │
  │  - Macro Guardrails: Ép chốt lời khi AHR >= 1.6, Ép gom khi AHR < 0.36     │
  └─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  TẦNG 3: CHIẾN LƯỢC LAI HYBRID V3 & ENGINE ĐỐI CHIẾU                       │
  │  - Hybrid V3: Tỷ trọng nền Quant v1 + Tinh chỉnh động ML (±15% Tilt)       │
  │  - Mô phỏng thực thi không nhìn trước (actual_weight[t+1] = target[t])      │
  │  - Khấu trừ phí giao dịch thực tế (0.1% mỗi lần điều chỉnh tỷ trọng)       │
  │  - Kiểm định toàn diện Out-of-Sample đến ngày mới nhất (không cắt 30 ngày) │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Các Chiến Lược Giao Dịch Trong Hệ Thống

### 1. Chiến Lược Quant v1 (AHR999 + SMA Buffer)
* **Quy tắc phân vùng AHR999 điều chỉnh mới:**
  * `AHR < 0.36` (Vùng đáy định giá siêu rẻ): Tỷ trọng $80\% - 100\%$ BTC.
  * `0.36 <= AHR <= 1.00` (Vùng tích lũy định kỳ DCA): Tỷ trọng $35\% - 75\%$ BTC tùy động lượng.
  * `1.00 < AHR < 1.60` (Vùng rủi ro / định giá cao): Hạ tỷ trọng phòng thủ xuống $10\% - 25\%$ BTC.
  * `AHR >= 1.60` (Đỉnh hưng phấn cực đại): Chốt lời toàn bộ ($0\%$ BTC, $100\%$ USDT).
* **Bộ đệm dải trễ SMA 20 ngày (`SMA_BUFFER_PCT = 0.015`):**
  * Chỉ xác nhận hồi phục tăng khi $\text{Close} > \text{SMA}_{20} \times 1.015$.
  * Chỉ xác nhận gãy xu hướng khi $\text{Close} < \text{SMA}_{20} \times 0.985$.
  * Giúp loại bỏ **148 lệnh giao dịch nhiễu**, tiết kiệm hơn **$800 USD phí giao dịch**, nâng Sharpe từ $0.88 \rightarrow 1.01$ và CAGR từ $24.14\% \rightarrow 27.85\%$.

### 2. Chiến Lược Machine Learning v2 (Pure ML)
* Dự đoán xác suất tăng giá $P(\text{Up})$ sau 30 ngày từ 25 đặc trưng kỹ thuật và chi phí vốn.
* **Cơ chế Dải trễ (Hysteresis Banding):**
  * $P(\text{Up}) \ge 0.65$: Mở vị thế Mua ($100\%$ BTC).
  * $P(\text{Up}) \le 0.45$: Thoát vị thế về tiền mặt ($0\%$ BTC, $100\%$ USDT).
  * $0.45 < P(\text{Up}) < 0.65$: **Giữ nguyên trạng thái hôm trước** nhằm triệt tiêu hiện tượng đảo vị thế liên tục khi thị trường đi ngang.
* **Rào chắn vĩ mô (Macro Guardrails):**
  * Nếu $AHR \ge 1.60$: Cưỡng chế thoát về $100\%$ USDT dù ML có báo xác suất tăng cao.
  * Nếu $AHR < 0.36$: Cưỡng chế nắm giữ tối thiểu $80\%$ BTC để không bán tháo trúng đáy chu kỳ.

### 3. Chiến Lược Lai Hybrid V3 (Quant Baseline Anchor + ML Dynamic Tilt)
* Lấy tỷ trọng phân bổ chu kỳ của Quant v1 làm **mỏ neo nền tảng (Anchor Weight)**.
* Sử dụng mức độ tin cậy từ xác suất ML để tinh chỉnh tỷ trọng động:
  * Khi $P(\text{Up}) \ge 0.65$: Cộng thêm $+15\%$ vào tỷ trọng vị thế (`HYBRID_TILT_WEIGHT = 0.15`).
  * Khi $P(\text{Up}) \le 0.45$: Trừ bớt $-15\%$ khỏi vị thế để phòng ngừa điều chỉnh.
  * Khi ở vùng trung lập: Giữ nguyên tỷ trọng nền tảng Quant.
* Cắt biên $[0.0, 1.0]$ và tuân thủ rào chắn vĩ mô AHR.
* **Hiệu quả**: Đạt khả năng **kiểm soát sụt giảm tài khoản tốt nhất toàn hệ thống** (Max Drawdown chỉ **-20.81%** trong khi thị trường giảm tới **-52.97%**).

---

## 3. Cam Kết Nghiêm Ngặt Chống Rò Rỉ Dữ Liệu & Thiết Kế Kiểm Định

1. **Đặc trưng chỉ nhìn về quá khứ ($X_t \le t$):** Mọi chỉ báo tại phiên $t$ chỉ dùng thông tin từ các nến đã đóng $\le t$.
2. **Nhãn mục tiêu nhìn trước có kiểm soát ($y_t = t + \text{horizon}$):**
   * Nhãn $y_t = 1$ nếu $\text{Close}_{t + 30} > \text{Close}_t$, ngược lại $0$.
   * Các ngày gần nhất (30 ngày cuối) chưa diễn ra tương lai sẽ có nhãn `NaN`.
3. **Kiểm định Out-of-Sample toàn vẹn (Không cắt bỏ 30 ngày cuối):**
   * Trong quá trình **huấn luyện (train)**, mô hình học trên các mẫu có nhãn.
   * Trong quá trình **kiểm định thực tế (test / backtest)**, hệ thống giữ nguyên vẹn toàn bộ các phiên đến ngày mới nhất (24/11/2026), sử dụng các đặc trưng $X_t$ sẵn có để mô phỏng chính xác hành vi giao dịch thực chiến.
4. **Phân chia chuỗi thời gian tuyến tính (`shuffle=False`):**
   * **Train Set (70%)** $\rightarrow$ **Validation Set (15%)** $\rightarrow$ **Test Set (15%)**.
5. **Chuẩn hóa an toàn (Train-only Scaling):**
   * Bộ `RobustScaler` **chỉ được `.fit()` trên tập Train**, sau đó chỉ `.transform()` lên Validation và Test.

---

## 4. Kết Quả Kiểm Định Thực Tế

### A. Đối đầu trực tiếp trên Out-of-Sample Test Set (27/09/2025 – 24/11/2026)
*Giai đoạn thị trường điều chỉnh sâu từ đỉnh về vùng ~59,600 USD:*

| Chỉ số hiệu suất | BUY & HOLD BTC | ML STRATEGY (V2) | QUANT V1 (BUFFERED) | **HYBRID V3 (LAI)** |
| :--- | :---: | :---: | :---: | :---: |
| **Vốn cuối kỳ (Gốc $10,000)** | $5,433.82 | $8,287.85 | $8,718.84 | **$8,803.04** ⭐ |
| **Tổng lợi nhuận (Total Return)** | **-45.66%** | -17.12% | -12.81% | **-11.97%** |
| **Lợi nhuận kép năm (CAGR)** | **-40.94%** | -14.97% | -11.16% | **-10.42%** |
| **Sụt giảm tối đa (Max Drawdown)** | **-52.97%** | -31.87% | -23.63% | **-20.81%** ⭐ |
| **Tỷ số Sharpe (rf=0%)** | -0.92 | -0.54 | -0.39 | **-0.37** |
| **Số lần tái cân bằng** | 1 | 11 | 38 | 65 |
| **Tổng phí giao dịch đã trả** | $10.00 | $92.61 | $94.09 | $144.00 |

### B. Hiệu suất toàn chu kỳ 8.7 năm (2018 – 2026) của Quant v1 có SMA Buffer
* **Tổng lợi nhuận:** **753.33%** (so với Buy & Hold 417.36%).
* **Lợi nhuận kép năm (CAGR):** **27.85%** (so với Buy & Hold 20.73%).
* **Max Drawdown:** **-35.92%** (cắt giảm hơn một nửa so với Buy & Hold -76.63%).
* **Tỷ số Sharpe:** **1.01** (so với Buy & Hold 0.33).
* **Tỷ số Sortino / Calmar:** **1.30 / 0.78**.

---

## 5. Danh Mục Bộ Đặc Trưng (25 Features)

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

## 6. Cấu Trúc Thư Mục Dự Án

```text
AHR999-ML/
├── config.py                 # File cấu hình trung tâm (AHR thresholds, SMA buffer, ML parameters)
├── requirements.txt          # Danh sách thư viện Python phụ thuộc
├── train_and_eval.py         # Pipeline huấn luyện ML, đánh giá phân loại & lưu model bundle
├── run_compare.py            # Chạy backtest so sánh 4 chiến lược (Hybrid vs ML vs Quant vs B&H)
├── run_backtest.py           # Script backtest chiến lược Quant v1 (có SMA Buffer) toàn chu kỳ 8.7 năm
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
    ├── features.py           # Sinh 25 đặc trưng chu kỳ & nhãn target 30 ngày (bảo toàn ngày mới nhất)
    ├── dataset.py            # Phân chia chuỗi thời gian & chuẩn hóa an toàn (Train-only scaler)
    ├── model.py              # Regularized Logistic Regression, Random Forest, tính metrics & joblib
    ├── strategy.py           # Bộ sinh tín hiệu Quant v1 (SMA buffer), ML v2 & Hybrid V3
    └── backtest.py           # Engine mô phỏng backtest trừ phí, Sharpe, Max Drawdown, CAGR
```

---

## 7. Hướng Dẫn Cài Đặt & Chạy

### Bước 1: Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Bước 2: Huấn luyện mô hình Machine Learning
```bash
python train_and_eval.py
```
*Tác vụ thực hiện:* Tiền xử lý dữ liệu, tạo 25 đặc trưng, huấn luyện mô hình phân loại chính quy hóa chống overfitting, in bảng metrics phân loại (Accuracy, ROC-AUC) và lưu `models/best_model.joblib`.

### Bước 3: Chạy đối chiếu 4 chiến lược trên tập Out-of-Sample
```bash
python run_compare.py
```
*Tác vụ thực hiện:* Mô phỏng và in bảng đối đầu 4 chiến lược (**Hybrid V3**, **ML Strategy v2**, **Quant v1 Buffered**, **Buy & Hold**), đồng thời xuất biểu đồ so sánh chi tiết 5 tầng ra `output/ml_vs_quant_comparison.png`.

### Bước 4: Chạy backtest toàn chu kỳ 8.7 năm cho Quant v1
```bash
python run_backtest.py
```
*Tác vụ thực hiện:* Kiểm định toàn bộ lịch sử từ 2018 đến nay và xuất biểu đồ `output/backtest_report.png`.

---

## 8. Hướng Dẫn Đọc Các Chỉ Số Hiệu Suất

| Chỉ số | Ý nghĩa & Tiêu chuẩn đánh giá |
| :--- | :--- |
| **Total Return (%)** | Tổng tỷ suất sinh lời trên toàn bộ giai đoạn kiểm định. |
| **CAGR (%)** | Tỷ suất lợi nhuận kép hàng năm (Compound Annual Growth Rate). |
| **Max Drawdown (Max DD %)** | Mức giảm tối đa từ đỉnh tài khoản gần nhất. Chỉ số càng nhỏ, rủi ro tài khoản càng thấp. |
| **Max DD Duration** | Số ngày tài khoản nằm dưới đỉnh (thước đo độ kiên nhẫn cần có của nhà đầu tư). |
| **Sharpe Ratio** | Tỷ lệ lợi nhuận vượt trội trên mỗi đơn vị rủi ro tổng thể. Sharpe $> 1.0$ là tốt. |
| **Sortino Ratio** | Chỉ phạt biến động giảm tiêu cực (Downside risk). Rất thích hợp đánh giá tài sản crypto. |
| **Calmar Ratio** | Tỷ lệ `CAGR / abs(Max DD)`. Đo lường mức độ tương xứng giữa lợi nhuận và mức sụt giảm nặng nhất. |
| **Rebalance Count & Fees** | Số lần điều chỉnh tỷ trọng và tổng phí giao dịch đã khấu trừ (phản ánh độ nhạy với chi phí sàn). |
