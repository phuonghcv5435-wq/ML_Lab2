"""Train, tune, compare, and persist regression models for sales revenue."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]


def build_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_transformer, numeric),
        ("categorical", categorical_transformer, categorical),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _metric_row(name: str, y_true, y_pred, cv_mae: float | None = None) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "CV_MAE": float(cv_mae) if cv_mae is not None else np.nan,
    }


def _plot_model_comparison(metrics: pd.DataFrame, figures_dir: Path) -> None:
    ordered = metrics.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["model"], ordered["MAE"])
    ax.set_title("Test-set MAE by model (lower is better)")
    ax.set_ylabel("MAE (sales units)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_mae.png", dpi=180)
    plt.close(fig)


def _plot_actual_predicted(y_true, y_pred, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.7)
    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")
    ax.set_title("Actual vs predicted sales (test set)")
    fig.tight_layout()
    fig.savefig(figures_dir / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)


def _save_feature_importance(model: Pipeline, figures_dir: Path, models_dir: Path) -> None:
    estimator = model.named_steps["model"]
    preprocess = model.named_steps["preprocess"]
    try:
        names = preprocess.get_feature_names_out()
    except Exception:
        return
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        label = "Feature importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
        label = "Absolute coefficient"
    else:
        return
    table = pd.DataFrame({"feature": names, "importance": values})
    table = table.sort_values("importance", ascending=False).head(20)
    table.to_csv(models_dir / "feature_importance.csv", index=False)
    plot_data = table.sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_data["feature"], plot_data["importance"])
    ax.set_title(f"Top 20 features — {label}")
    ax.set_xlabel(label)
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=180)
    plt.close(fig)


def train_models(data_dir: str = "data/processed", save_model_dir: str = "models") -> dict:
    data_folder = ROOT / data_dir
    models_dir = ROOT / save_model_dir
    figures_dir = ROOT / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(data_folder / "X_train_fe.csv")
    X_test = pd.read_csv(data_folder / "X_test_fe.csv")
    y_train = pd.read_csv(data_folder / "y_train.csv")["sales"].to_numpy()
    y_test = pd.read_csv(data_folder / "y_test.csv")["sales"].to_numpy()
    if "quantity" in X_train.columns or "sales" in X_train.columns:
        raise ValueError("Target leakage detected: quantity and sales must not be model features.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")

    cv = TimeSeriesSplit(n_splits=3)
    candidates = {
        "LinearRegression": (LinearRegression(), None),
        "DecisionTree": (DecisionTreeRegressor(random_state=42), {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3],
        }),
        "RandomForest": (RandomForestRegressor(random_state=42, n_jobs=-1), {
            "model__n_estimators": [100],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 3],
        }),
        "GradientBoosting": (GradientBoostingRegressor(random_state=42), {
            "model__n_estimators": [100],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
        }),
    }

    results = []
    fitted = {}
    cv_details = []
    for name, (estimator, grid) in candidates.items():
        print(f"Training {name}...")
        pipeline = build_pipeline(X_train, estimator)
        if grid:
            search = GridSearchCV(
                pipeline, grid, scoring="neg_mean_absolute_error", cv=cv,
                n_jobs=-1, refit=True, error_score="raise",
            )
            search.fit(X_train, y_train)
            fitted_model = search.best_estimator_
            cv_mae = -float(search.best_score_)
            for params, score, std in zip(
                search.cv_results_["params"],
                search.cv_results_["mean_test_score"],
                search.cv_results_["std_test_score"],
            ):
                cv_details.append({"model": name, "params": json.dumps(params, sort_keys=True),
                                   "CV_MAE": -float(score), "CV_MAE_std": float(std)})
            best_params = search.best_params_
        else:
            scores = cross_val_score(pipeline, X_train, y_train, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
            cv_mae = -float(scores.mean())
            fitted_model = pipeline.fit(X_train, y_train)
            best_params = {}

        prediction = fitted_model.predict(X_test)
        results.append(_metric_row(name, y_test, prediction, cv_mae))
        fitted[name] = fitted_model
        joblib.dump(fitted_model, models_dir / f"{name}.joblib")
        print(f"  CV MAE={cv_mae:.3f}; test MAE={results[-1]['MAE']:.3f}; params={best_params}")

    # A simple baseline checks whether the learned models improve over a constant.
    dummy = build_pipeline(X_train, DummyRegressor(strategy="mean"))
    dummy_scores = cross_val_score(dummy, X_train, y_train, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
    dummy.fit(X_train, y_train)
    dummy_prediction = dummy.predict(X_test)
    results.append(_metric_row("DummyMean", y_test, dummy_prediction, -float(dummy_scores.mean())))
    fitted["DummyMean"] = dummy
    joblib.dump(dummy, models_dir / "DummyMean.joblib")

    metrics = pd.DataFrame(results).sort_values("CV_MAE").reset_index(drop=True)
    metrics.to_csv(models_dir / "metrics.csv", index=False)
    pd.DataFrame(cv_details).to_csv(models_dir / "tuning_results.csv", index=False)

    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]
    best_prediction = best_model.predict(X_test)
    pd.DataFrame({"actual_sales": y_test, "predicted_sales": best_prediction,
                  "absolute_error": np.abs(y_test - best_prediction)}).to_csv(
        models_dir / "predictions.csv", index=False
    )

    history_path = data_folder / "feature_history.joblib"
    history = joblib.load(history_path) if history_path.exists() else {"products": {}, "customers": {}}
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "target": "sales",
        "feature_columns": X_train.columns.tolist(),
        "history": history,
        "target_definition": "unit_price * quantity",
        "quantity_is_excluded_from_features": True,
    }
    joblib.dump(bundle, models_dir / "best_sales_model.joblib")

    _plot_model_comparison(pd.DataFrame(results), figures_dir)
    _plot_actual_predicted(y_test, best_prediction, figures_dir)
    _save_feature_importance(best_model, figures_dir, models_dir)

    best_metrics = next(row for row in results if row["model"] == best_name)
    summary = {"best_model": best_name, "selection_basis": "lowest time-series CV MAE",
               "best_model_test_metrics": best_metrics,
               "models_compared": list(candidates) + ["DummyMean"]}
    (models_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nTest-set comparison:")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSelected {best_name} based on time-series cross-validation MAE.")
    print(f"Saved model bundle: {models_dir / 'best_sales_model.joblib'}")
    return summary


if __name__ == "__main__":
    train_models()
