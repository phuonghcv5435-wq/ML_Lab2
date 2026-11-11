"""Run all project stages from raw data through evaluation and report."""
from src.data_collection import collect_data
from src.eda import perform_eda
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.training import train_models
from src.eval import evaluate_models
from src.report import write_report


def main() -> None:
    collect_data()
    perform_eda()
    preprocess_data()
    engineer_features()
    train_models()
    evaluate_models()
    write_report()
    print("\nPipeline completed. See models/, figures/, and doc/report.md.")


if __name__ == "__main__":
    main()
