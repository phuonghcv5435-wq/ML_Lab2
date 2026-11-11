"""Load the saved sales model and predict revenue for new transactions."""
from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import build_inference_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "best_sales_model.joblib"


def predict_sales(record: dict, model_path: str | Path = DEFAULT_MODEL_PATH) -> float:
    bundle = joblib.load(model_path)
    required = {"unit_price", "product_id", "customer_id", "order_date"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Missing input fields: {sorted(missing)}")
    features = build_inference_features(record, bundle["history"])
    # Match the exact training feature order and safely add missing optional inputs.
    features = features.reindex(columns=bundle["feature_columns"], fill_value=0)
    prediction = float(bundle["model"].predict(features)[0])
    return max(0.0, prediction)


def predict_sales_batch(records: list[dict], model_path: str | Path = DEFAULT_MODEL_PATH) -> list[float]:
    return [predict_sales(record, model_path=model_path) for record in records]
