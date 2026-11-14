"""Data cleaning and chronological train/test split for product sales."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def clean_sales_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean source transactions and derive transaction revenue (sales).

    Invalid rows without a usable date, unit price, or quantity are removed.
    Valid target values are kept unchanged; predictor outliers are handled using
    training-only IQR bounds later in this module.
    """
    data = df.copy()
    required = {"unit_price", "quantity", "order_date"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    original_rows = len(data)
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce", format="mixed")
    for column in ("unit_price", "quantity", "customer_age"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.drop_duplicates().copy()
    data = data.dropna(subset=["order_date", "unit_price", "quantity"])
    data = data[(data["unit_price"] >= 0) & (data["quantity"] >= 0)].copy()

    if "customer_age" in data.columns:
        data.loc[~data["customer_age"].between(0, 100), "customer_age"] = float("nan")

    # The assignment's target is product sales (revenue), not unit quantity.
    data["sales"] = data["unit_price"] * data["quantity"]
    data = data.sort_values(["order_date"] + (["order_id"] if "order_id" in data.columns else [])).reset_index(drop=True)

    summary = {
        "original_rows": int(original_rows),
        "clean_rows": int(len(data)),
        "rows_removed_during_cleaning": int(original_rows - len(data)),
        "missing_values_after_cleaning": {k: int(v) for k, v in data.isna().sum().items()},
        "target_definition": "sales = unit_price * quantity",
        "date_min": data["order_date"].min().strftime("%Y-%m-%d"),
        "date_max": data["order_date"].max().strftime("%Y-%m-%d"),
    }
    return data, summary


def chronological_split(data: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on unique dates so later transactions are held out for testing."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    dates = sorted(pd.to_datetime(data["order_date"]).dropna().unique())
    if len(dates) < 5:
        raise ValueError("At least five distinct transaction dates are required")
    split_index = max(1, min(len(dates) - 1, int(len(dates) * (1 - test_size))))
    cutoff = dates[split_index]
    train = data.loc[data["order_date"] < cutoff].copy()
    test = data.loc[data["order_date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def preprocess_data(
    input_path: str = "data/raw/sales_data.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
) -> dict:
    input_file = ROOT / input_path
    output_path = ROOT / output_dir
    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file}")
    output_path.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_file)
    clean, summary = clean_sales_data(raw)
    train, test = chronological_split(clean, test_size=test_size)

    # Detect and cap predictor outliers with bounds learned from train only.
    # The target remains unchanged so reported errors reflect original revenue.
    numeric_predictors = [c for c in ["unit_price", "customer_age"] if c in train.columns]
    outlier_summary = {}
    for column in numeric_predictors:
        q1, q3 = train[column].quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        train_outliers = int(((train[column] < lower) | (train[column] > upper)).sum())
        test_outliers = int(((test[column] < lower) | (test[column] > upper)).sum())
        train[column] = train[column].clip(lower, upper)
        test[column] = test[column].clip(lower, upper)
        outlier_summary[column] = {
            "train_outliers_capped": train_outliers,
            "test_outliers_capped_using_train_bounds": test_outliers,
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        }

    # Keep the target separate from model inputs. quantity is not passed to X,
    # because sales is calculated from unit_price * quantity.
    train.drop(columns=["sales"]).to_csv(output_path / "train_base.csv", index=False)
    test.drop(columns=["sales"]).to_csv(output_path / "test_base.csv", index=False)
    train[["sales"]].to_csv(output_path / "y_train.csv", index=False)
    test[["sales"]].to_csv(output_path / "y_test.csv", index=False)
    clean.to_csv(output_path / "sales_cleaned.csv", index=False)

    summary.update({
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_max_date": train["order_date"].max().strftime("%Y-%m-%d"),
        "test_min_date": test["order_date"].min().strftime("%Y-%m-%d"),
        "outlier_review_on_predictors": outlier_summary,
        "quantity_excluded_from_features": True,
    })
    (output_path / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Prepared {len(train)} training rows and {len(test)} later-date test rows.")
    print(f"Target: sales = unit_price * quantity; quantity excluded from X.")
    print(f"Saved processed data to: {output_path}")
    return summary


if __name__ == "__main__":
    preprocess_data()
