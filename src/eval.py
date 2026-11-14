"""Re-evaluate all saved models on the chronological holdout set."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]


def evaluate_models(data_dir: str = "data/processed", model_dir: str = "models") -> pd.DataFrame:
    data_folder, models_folder = ROOT / data_dir, ROOT / model_dir
    X_test = pd.read_csv(data_folder / "X_test_fe.csv")
    y_test = pd.read_csv(data_folder / "y_test.csv")["sales"].to_numpy()
    rows = []
    for model_file in sorted(models_folder.glob("*.joblib")):
        if model_file.name in {"best_sales_model.joblib", "feature_history.joblib"}:
            continue
        model = joblib.load(model_file)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rows.append({"model": model_file.stem,
                     "MAE": mean_absolute_error(y_test, pred),
                     "MSE": mse, "RMSE": np.sqrt(mse),
                     "R2": r2_score(y_test, pred)})
    result = pd.DataFrame(rows).sort_values("MAE")
    result.to_csv(models_folder / "evaluation_metrics.csv", index=False)
    print(result.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return result


if __name__ == "__main__":
    evaluate_models()
