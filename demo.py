"""Run a small end-to-end inference example with the saved best model."""
from src.inference import predict_sales


def run_demo() -> None:
    sample = {
        "product_id": 9,
        "product_name": "Tablet",
        "category": "Electronics",
        "unit_price": 650,
        "customer_id": 101,
        "customer_age": 35,
        "customer_gender": "Female",
        "country": "Germany",
        "order_date": "2023-01-15",
    }
    prediction = predict_sales(sample)
    print("Product Sales Prediction")
    print(f"Input: {sample}")
    print(f"Predicted sales revenue: ${prediction:,.2f}")


if __name__ == "__main__":
    run_demo()
