"""
Main execution pipeline for Retail Sales Forecaster.
Engineered by M. Abdullah.
Runs chronological train/test split, fits candidate models, and evaluates out-of-sample metrics.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_data, add_features, train_test_split_temporal
from src.model import MachineLearningForecaster, NaiveSeasonalForecaster
from src.evaluation import compare_forecasters


def run_pipeline():
    data_path = PROJECT_ROOT / "data" / "sales.csv"
    print(f"Loading retail sales data from {data_path}...")
    df = load_data(str(data_path))
    df = add_features(df)

    # 30-day chronological holdout
    train_df, test_df = train_test_split_temporal(df, test_days=30)
    print(f"Split data: {len(train_df)} training days, {len(test_df)} holdout test days.")

    # 1. Fit Naive Seasonal Baseline (predicts lag_7)
    naive = NaiveSeasonalForecaster(lag=7)
    naive.fit(train_df)
    naive_preds = naive.predict(test_df)

    # 2. Fit ML Tree Ensemble Forecaster
    ml_model = MachineLearningForecaster(model_type="random_forest")
    ml_model.fit(train_df)
    ml_preds = ml_model.predict(test_df)

    # 3. Benchmark Comparison
    actuals = test_df["sales"].values
    results = compare_forecasters(actuals, {
        "Naive Seasonal Baseline (Lag 7)": naive_preds,
        "Random Forest Regressor (ML)": ml_preds
    })

    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE FORECASTING PERFORMANCE (LAST 30 DAYS)")
    print("=" * 60)
    print(results.to_string())
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    run_pipeline()
