"""Write a concise results report using metrics produced by the executed pipeline."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def write_report() -> Path:
    models_dir = ROOT / "models"
    metrics_path = models_dir / "metrics.csv"
    summary_path = models_dir / "training_summary.json"
    data_summary_path = ROOT / "data" / "processed" / "preprocessing_summary.json"
    if not all(p.exists() for p in (metrics_path, summary_path, data_summary_path)):
        raise FileNotFoundError("Run the full pipeline before generating the report.")

    metrics = pd.read_csv(metrics_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = json.loads(data_summary_path.read_text(encoding="utf-8"))
    best = summary["best_model_test_metrics"]
    lines = [
        "# Product Sales Prediction — Results Report", "",
        "## 1. Objective", "Predict transaction sales revenue (`sales`) using product, customer, price, historical, and calendar features.",
        "", "## 2. Dataset and target", 
        f"- Clean records: {data['clean_rows']} (from {data['original_rows']} source rows).",
        f"- Chronological split: {data['train_rows']} train / {data['test_rows']} test records.",
        f"- Training dates end: {data['train_max_date']}; test dates begin: {data['test_min_date']}.",
        "- Target definition: `sales = unit_price * quantity`.",
        "- `quantity` is excluded from model features to prevent direct target leakage.",
        "", "## 3. Method", 
        "Missing numeric/categorical values are imputed inside the model pipeline; categorical features use one-hot encoding. Duplicate rows and invalid price/quantity/date rows are removed. The test set contains later dates than the training set. Historical popularity, purchase frequency, customer lifetime value, and historical average sales use only earlier training transactions. Calendar features represent month, quarter, weekday, weekend, and cyclical seasonality.",
        "", "## 4. Model comparison", "| Model | MAE | MSE | RMSE | R² | CV MAE |", "|---|---:|---:|---:|---:|---:|"
    ]
    for _, row in metrics.iterrows():
        lines.append(f"| {row['model']} | {row['MAE']:.3f} | {row['MSE']:.3f} | {row['RMSE']:.3f} | {row['R2']:.3f} | {row['CV_MAE']:.3f} |")
    lines += [
        "", "## 5. Selected model", 
        f"Selected **{summary['best_model']}** based on the lowest time-series cross-validation MAE. On the untouched chronological test set: MAE = {best['MAE']:.3f}, MSE = {best['MSE']:.3f}, RMSE = {best['RMSE']:.3f}, and R² = {best['R2']:.3f}.",
        "", "## 6. Local prediction", 
        "The selected preprocessing-plus-model pipeline and training history are saved in `models/best_sales_model.joblib`. Run `python demo.py` from the project root to load the saved model and predict revenue for the sample transaction.",
        "", "## 7. Limitations and next steps", 
        "The available dataset contains transaction prices, quantities, product/customer identifiers, demographics, country, and dates from 2022-01-01 through 2022-12-31. It does not include reviews, ratings, advertising spend, campaign records, or explicit holiday indicators. Product popularity is therefore approximated by prior transaction count; customer lifetime value is approximated by prior cumulative transaction revenue. EDA examines twelve monthly aggregates, but only one annual cycle is available, so a recurring yearly seasonal pattern cannot be validated reliably. ARIMA/Prophet-style forecasting is therefore not trained on this small one-cycle series. Recommendation systems are also not in scope because the dataset does not include rating/preference feedback.",
        "", "## 8. Reproducibility", 
        "Run `python run_pipeline.py` from the project root. The code uses a chronological holdout and `TimeSeriesSplit` for hyperparameter selection; the test set is reserved for final comparison.", ""
    ]
    report_path = ROOT / "doc" / "report.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    return report_path
