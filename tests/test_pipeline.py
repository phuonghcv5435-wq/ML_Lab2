import pandas as pd

from src.data_preprocessing import clean_sales_data, chronological_split
from src.feature_engineering import engineer_dataframe, build_inference_features


def sample_transactions():
    return pd.DataFrame([
        {"order_id": 1, "product_id": 7, "product_name": "A", "category": "Books",
         "unit_price": 10.0, "quantity": 2, "order_date": "2022-01-01",
         "customer_id": 100, "customer_age": 30, "customer_gender": "F", "country": "VN"},
        {"order_id": 2, "product_id": 7, "product_name": "A", "category": "Books",
         "unit_price": 12.0, "quantity": 3, "order_date": "2022-01-02",
         "customer_id": 100, "customer_age": 30, "customer_gender": "F", "country": "VN"},
        {"order_id": 3, "product_id": 8, "product_name": "B", "category": "Games",
         "unit_price": 20.0, "quantity": 1, "order_date": "2022-01-03",
         "customer_id": 101, "customer_age": 25, "customer_gender": "M", "country": "VN"},
        {"order_id": 4, "product_id": 9, "product_name": "C", "category": "Games",
         "unit_price": 30.0, "quantity": 2, "order_date": "2022-01-04",
         "customer_id": 102, "customer_age": 40, "customer_gender": "F", "country": "SG"},
        {"order_id": 5, "product_id": 10, "product_name": "D", "category": "Books",
         "unit_price": 15.0, "quantity": 2, "order_date": "2022-01-05",
         "customer_id": 103, "customer_age": 29, "customer_gender": "M", "country": "SG"},
    ])


def test_sales_target_is_revenue_and_invalid_rows_are_removed():
    raw = pd.concat([sample_transactions(), sample_transactions().iloc[[0]]], ignore_index=True)
    raw.loc[len(raw)] = {**sample_transactions().iloc[0].to_dict(), "order_id": 99, "unit_price": -1}
    clean, summary = clean_sales_data(raw)
    assert len(clean) == 5
    assert clean.loc[0, "sales"] == 20.0
    assert summary["target_definition"] == "sales = unit_price * quantity"


def test_split_keeps_later_dates_in_test():
    clean, _ = clean_sales_data(sample_transactions())
    train, test = chronological_split(clean, test_size=0.4)
    assert train["order_date"].max() < test["order_date"].min()
    assert len(train) + len(test) == len(clean)


def test_history_features_only_use_previous_transactions():
    clean, _ = clean_sales_data(sample_transactions())
    features, history = engineer_dataframe(clean, update_history=True)
    assert features.loc[0, "product_popularity_prior"] == 0
    assert features.loc[1, "product_popularity_prior"] == 1
    assert features.loc[1, "customer_purchase_frequency"] == 1
    assert features.loc[1, "customer_lifetime_value_prior"] == 20.0
    assert "quantity" not in features.columns
    assert "sales" not in features.columns
    assert history["products"]["7"]["count"] == 2


def test_test_and_inference_features_read_history_without_changing_it():
    train = sample_transactions().iloc[:2].copy()
    train["sales"] = train["unit_price"] * train["quantity"]
    _, history = engineer_dataframe(train, update_history=True)
    before = history["products"]["7"]["count"]
    inference = build_inference_features({
        "product_id": 7, "product_name": "A", "category": "Books",
        "unit_price": 11, "customer_id": 100, "customer_age": 30,
        "customer_gender": "F", "country": "VN", "order_date": "2022-02-01"
    }, history)
    assert inference.loc[0, "product_popularity_prior"] == 2
    assert inference.loc[0, "customer_purchase_frequency"] == 2
    assert history["products"]["7"]["count"] == before
