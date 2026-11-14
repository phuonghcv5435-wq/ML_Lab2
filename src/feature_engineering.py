"""Leakage-aware product, customer-history, and seasonal features."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL_COLUMNS = ["product_id", "product_name", "category", "customer_id", "customer_gender", "country"]


def _key(value) -> str:
    return "__UNKNOWN__" if pd.isna(value) else str(value)


def engineer_dataframe(
    data: pd.DataFrame,
    history: dict | None = None,
    update_history: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Create features using only the history available before each row.

    For training rows the state is updated after each transaction. For test and
    inference rows, the supplied training history is read-only, so test labels
    never enter features.
    """
    frame = data.copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce", format="mixed")
    frame = frame.sort_values(["order_date"] + (["order_id"] if "order_id" in frame.columns else []))
    state = copy.deepcopy(history) if history is not None else {"products": {}, "customers": {}}
    rows = []

    for _, row in frame.iterrows():
        product_key = _key(row.get("product_id"))
        customer_key = _key(row.get("customer_id"))
        product_stats = state["products"].get(product_key, {"count": 0, "sales_sum": 0.0})
        customer_stats = state["customers"].get(customer_key, {"count": 0, "sales_sum": 0.0})
        date = row["order_date"]

        item = {}
        for column in CATEGORICAL_COLUMNS:
            if column in row.index:
                item[column] = _key(row[column])
        for column in ("unit_price", "customer_age"):
            if column in row.index:
                item[column] = row[column]

        item["product_popularity_prior"] = int(product_stats["count"])
        item["product_avg_sales_prior"] = (
            float(product_stats["sales_sum"]) / product_stats["count"] if product_stats["count"] else 0.0
        )
        item["customer_purchase_frequency"] = int(customer_stats["count"])
        item["customer_lifetime_value_prior"] = float(customer_stats["sales_sum"])
        item["customer_avg_sales_prior"] = (
            float(customer_stats["sales_sum"]) / customer_stats["count"] if customer_stats["count"] else 0.0
        )
        if pd.notna(date):
            month = int(date.month)
            day_of_year = int(date.dayofyear)
            item.update({
                "month": month,
                "quarter": int(date.quarter),
                "day_of_week": int(date.dayofweek),
                "is_weekend": int(date.dayofweek >= 5),
                "is_holiday_season": int(month in (11, 12)),
                "month_sin": float(np.sin(2 * np.pi * month / 12)),
                "month_cos": float(np.cos(2 * np.pi * month / 12)),
                "day_of_year_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
                "day_of_year_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
            })
        else:
            item.update({"month": np.nan, "quarter": np.nan, "day_of_week": np.nan,
                         "is_weekend": np.nan, "is_holiday_season": np.nan,
                         "month_sin": np.nan, "month_cos": np.nan,
                         "day_of_year_sin": np.nan, "day_of_year_cos": np.nan})
        rows.append(item)

        if update_history and "sales" in row.index and pd.notna(row["sales"]):
            sale = float(row["sales"])
            p = state["products"].setdefault(product_key, {"count": 0, "sales_sum": 0.0})
            c = state["customers"].setdefault(customer_key, {"count": 0, "sales_sum": 0.0})
            p["count"] += 1
            p["sales_sum"] += sale
            c["count"] += 1
            c["sales_sum"] += sale

    features = pd.DataFrame(rows).reset_index(drop=True)
    return features, state


def engineer_features(input_dir: str = "data/processed", output_dir: str = "data/processed") -> dict:
    base = ROOT / input_dir
    output = ROOT / output_dir
    train_path, test_path = base / "train_base.csv", base / "test_base.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run src/data_preprocessing.py before feature engineering.")

    train_base = pd.read_csv(train_path)
    test_base = pd.read_csv(test_path)
    y_train = pd.read_csv(base / "y_train.csv")["sales"]
    y_test = pd.read_csv(base / "y_test.csv")["sales"]
    train_with_target = train_base.copy()
    train_with_target["sales"] = y_train.values

    X_train, history = engineer_dataframe(train_with_target, update_history=True)
    X_test, _ = engineer_dataframe(test_base, history=history, update_history=False)

    X_train.to_csv(output / "X_train_fe.csv", index=False)
    X_test.to_csv(output / "X_test_fe.csv", index=False)
    # Retain a compatible pair of names used by a few earlier project scripts.
    X_train.to_csv(output / "X_train.csv", index=False)
    X_test.to_csv(output / "X_test.csv", index=False)
    joblib.dump(history, output / "feature_history.joblib")

    print(f"Engineered {X_train.shape[1]} features; quantity and current target are excluded.")
    print("History features use previous training transactions only.")
    return history


def build_inference_features(record: dict, history: dict) -> pd.DataFrame:
    frame = pd.DataFrame([record])
    features, _ = engineer_dataframe(frame, history=history, update_history=False)
    return features


if __name__ == "__main__":
    engineer_features()
