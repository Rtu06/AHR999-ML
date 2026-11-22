"""
Pipeline huấn luyện và đánh giá mô hình Machine Learning dự đoán xu hướng Bitcoin (v2):
1. Tải và tiền xử lý dữ liệu nến 1d Binance BTCUSDT.
2. Tính toán chỉ số AHR999, đường tăng trưởng hàm mũ và các chỉ báo kỹ thuật.
3. Kỹ nghệ đặc trưng (Features) và sinh nhãn mục tiêu nhìn trước (Target Horizon).
4. Phân chia chuỗi thời gian tuần tự Train (70%) / Val (15%) / Test (15%) - KHÔNG SHUFFLE.
5. Chuẩn hóa đặc trưng (Chỉ Fit trên Train để chống rò rỉ dữ liệu).
6. Huấn luyện mô hình Machine Learning (Logistic Regression, Random Forest, hoặc LightGBM).
7. Đánh giá đa chiều trên Train, Val, Test (Accuracy, Precision, Recall, F1, ROC-AUC).
8. Lưu mô hình vào models/best_model.joblib và xuất báo cáo metrics ra data/processed/.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

from config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    DATA_ML_FEATURES_FILE,
    MODEL_FILE,
    METRICS_REPORT_FILE,
    HORIZON_DAYS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    MODEL_TYPE,
    SCALER_TYPE,
    RANDOM_STATE,
)
from src.data_loader import load_btc_daily_data
from src.indicators import calculate_ahr999_and_indicators
from src.features import build_features_and_target
from src.dataset import prepare_ml_dataset
from src.model import (
    get_model,
    train_model,
    evaluate_model,
    get_feature_importance,
    save_model_bundle,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainAndEval")


def print_evaluation_table(train_metrics: dict, val_metrics: dict, test_metrics: dict) -> None:
    """In bảng đánh giá phân loại Machine Learning chuẩn mực."""
    sep = "=" * 74
    sub_sep = "-" * 74
    print("\n" + sep)
    print(f"       BÁO CÁO ĐÁNH GIÁ MÔ HÌNH MACHINE LEARNING (MODEL: {MODEL_TYPE.upper()})")
    print(f"       Mục tiêu: Dự đoán xác suất tăng giá sau {HORIZON_DAYS} ngày (Horizon={HORIZON_DAYS}d)")
    print(sep)
    print(f" {'METRIC PHÂN LOẠI':<24} | {'TRAIN SET':>14} | {'VAL SET':>14} | {'TEST SET':>14}")
    print(sub_sep)
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        m_name = m.upper()
        tr_val = train_metrics.get(m, 0.0) * 100
        va_val = val_metrics.get(m, 0.0) * 100
        te_val = test_metrics.get(m, 0.0) * 100
        print(f" {m_name:<24} | {tr_val:>13.2f}% | {va_val:>13.2f}% | {te_val:>13.2f}%")
    print(sep + "\n")


def print_feature_importance_table(df_imp: pd.DataFrame, top_k: int = 10) -> None:
    """In bảng xếp hạng các đặc trưng quan trọng nhất."""
    sep = "-" * 50
    print(sep)
    print(f" TOP {top_k} ĐẶC TRƯNG QUAN TRỌNG NHẤT CỦA MÔ HÌNH")
    print(sep)
    print(f" {'THỨ HẠNG':<10} | {'TÊN ĐẶC TRƯNG':<25} | {'TRỌNG SỐ':>8}")
    print(sep)
    for idx, row in df_imp.head(top_k).iterrows():
        print(f" #{idx+1:<9} | {row['feature']:<25} | {row['importance']:>8.4f}")
    print(sep + "\n")


def run_training_pipeline() -> None:
    logger.info(">>> BẮT ĐẦU PIPELINE HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH ML V2 <<<")

    # 1. Tải dữ liệu nến 1d Binance
    logger.info("Bước 1: Tải dữ liệu giá nến ngày...")
    df_raw = load_btc_daily_data(force_download=False)
    logger.info("Tải thành công: %d phiên giao dịch.", len(df_raw))

    # 2. Tính toán chỉ số AHR999 & kỹ thuật
    logger.info("Bước 2: Tính toán chỉ số AHR999, Cost 200d, Exp Growth & SMA...")
    df_indicators = calculate_ahr999_and_indicators(df_raw, drop_na=True)
    df_indicators.to_csv(PROCESSED_DATA_FILE, index=False)
    logger.info("Đã lưu dữ liệu chỉ số cơ sở vào: %s", PROCESSED_DATA_FILE)

    # 3. Kỹ nghệ đặc trưng và sinh nhãn tương lai
    logger.info("Bước 3: Tạo đặc trưng (Features) và gán nhãn mục tiêu (Horizon = %d ngày)...", HORIZON_DAYS)
    df_ml, feature_cols = build_features_and_target(df_indicators, horizon_days=HORIZON_DAYS)
    df_ml.to_csv(DATA_ML_FEATURES_FILE, index=False)
    logger.info("Đã lưu dữ liệu ML features (%d dòng, %d cột) vào: %s", len(df_ml), len(feature_cols), DATA_ML_FEATURES_FILE)

    # Tỷ lệ nhãn mục tiêu
    target_dist = df_ml["target"].value_counts(normalize=True).to_dict()
    logger.info(
        "Phân bổ nhãn mục tiêu: Up (1) = %.1f%% | Down/Neutral (0) = %.1f%%",
        target_dist.get(1, 0.0) * 100,
        target_dist.get(0, 0.0) * 100,
    )

    # 4. Phân chia tập dữ liệu và chuẩn hóa an toàn
    logger.info("Bước 4: Phân chia Train/Val/Test tuần tự và Fit Scaler trên Train...")
    bundle = prepare_ml_dataset(
        df=df_ml,
        feature_cols=feature_cols,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        scaler_type=SCALER_TYPE,
    )

    # 5. Khởi tạo và huấn luyện mô hình
    logger.info("Bước 5: Khởi tạo mô hình %s và huấn luyện...", MODEL_TYPE.upper())
    model = get_model(model_type=MODEL_TYPE, random_state=RANDOM_STATE)
    model = train_model(model, bundle.X_train, bundle.y_train)

    # 6. Đánh giá mô hình
    logger.info("Bước 6: Đánh giá hiệu năng phân loại trên các tập...")
    train_metrics = evaluate_model(model, bundle.X_train, bundle.y_train)
    val_metrics = evaluate_model(model, bundle.X_val, bundle.y_val)
    test_metrics = evaluate_model(model, bundle.X_test, bundle.y_test)

    # In kết quả ra console
    print_evaluation_table(train_metrics, val_metrics, test_metrics)

    # Trích xuất độ quan trọng đặc trưng
    df_imp = get_feature_importance(model, feature_cols)
    print_feature_importance_table(df_imp, top_k=10)

    # 7. Lưu mô hình và scaler
    logger.info("Bước 7: Đóng gói và lưu mô hình vào: %s", MODEL_FILE)
    metadata = {
        "model_type": MODEL_TYPE,
        "scaler_type": SCALER_TYPE,
        "horizon_days": HORIZON_DAYS,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "train_samples": len(bundle.df_train),
        "val_samples": len(bundle.df_val),
        "test_samples": len(bundle.df_test),
        "train_date_range": [bundle.df_train["date"].iloc[0], bundle.df_train["date"].iloc[-1]],
        "val_date_range": [bundle.df_val["date"].iloc[0], bundle.df_val["date"].iloc[-1]],
        "test_date_range": [bundle.df_test["date"].iloc[0], bundle.df_test["date"].iloc[-1]],
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
    }
    save_model_bundle(
        filepath=MODEL_FILE,
        model=model,
        scaler=bundle.scaler,
        feature_cols=feature_cols,
        metadata=metadata,
    )

    # Xuất báo cáo JSON
    report = {
        "metadata": metadata,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "top_features": df_imp.head(10).to_dict(orient="records"),
    }
    with open(METRICS_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    logger.info("Đã xuất file báo cáo metrics: %s", METRICS_REPORT_FILE)

    logger.info(">>> HOÀN TẤT PIPELINE HUẤN LUYỆN THÀNH CÔNG <<<")


if __name__ == "__main__":
    run_training_pipeline()
