"""Exploratory data analysis charts for the sales dataset."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def perform_eda(data_path: str = "data/raw/sales_data.csv", output_dir: str = "figures") -> dict:
    data = pd.read_csv(ROOT / data_path)
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce", format="mixed")
    data["sales"] = pd.to_numeric(data["unit_price"], errors="coerce") * pd.to_numeric(data["quantity"], errors="coerce")
    output = ROOT / output_dir
    output.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": int(len(data)), "columns": int(data.shape[1]),
        "missing_values": {k: int(v) for k, v in data.isna().sum().items()},
        "duplicate_rows": int(data.duplicated().sum()),
        "sales_mean": float(data["sales"].mean()),
        "sales_median": float(data["sales"].median()),
        "sales_min": float(data["sales"].min()),
        "sales_max": float(data["sales"].max()),
        "date_min": str(data["order_date"].min().date()),
        "date_max": str(data["order_date"].max().date()),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data["sales"].dropna(), bins=25)
    ax.set(title="Distribution of transaction sales", xlabel="Sales", ylabel="Transactions")
    fig.tight_layout(); fig.savefig(output / "eda_sales_distribution.png", dpi=180); plt.close(fig)

    for column, filename, title in [
        ("category", "eda_sales_by_category.png", "Total sales by product category"),
        ("country", "eda_sales_by_country.png", "Total sales by customer country"),
    ]:
        grouped = data.groupby(column, dropna=False)["sales"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(grouped.index.astype(str), grouped.values)
        ax.set(title=title, xlabel=column.replace("_", " ").title(), ylabel="Sales")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig(output / filename, dpi=180); plt.close(fig)

    monthly = data.dropna(subset=["order_date"]).assign(month=lambda d: d["order_date"].dt.to_period("M").astype(str))
    monthly_sales = monthly.groupby("month")["sales"].sum()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(monthly_sales.index, monthly_sales.values, marker="o")
    ax.set(title="Monthly sales trend", xlabel="Month", ylabel="Sales")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(output / "eda_monthly_sales_trend.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data["unit_price"], data["sales"], alpha=0.6)
    ax.set(title="Unit price vs transaction sales", xlabel="Unit price", ylabel="Sales")
    fig.tight_layout(); fig.savefig(output / "eda_price_vs_sales.png", dpi=180); plt.close(fig)

    pd.DataFrame([summary]).to_json(output / "eda_summary.json", orient="records", indent=2)
    print(f"EDA complete: {output}")
    print(f"Rows={summary['rows']}, duplicate rows={summary['duplicate_rows']}, missing={sum(summary['missing_values'].values())}")
    return summary


if __name__ == "__main__":
    perform_eda()
