"""
Streamlit Web Dashboard for Retail Sales Forecasting.
Engineered by M. Abdullah.
Interactive interface comparing Random Forest Regressor vs. Naive Seasonal Baseline on holdout data.
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from src.preprocessing import load_data, add_features, train_test_split_temporal
from src.model import MachineLearningForecaster, NaiveSeasonalForecaster, train_prophet, make_future
from src.evaluation import evaluate_model

st.set_page_config(page_title="Retail Sales Forecaster", page_icon="📈", layout="wide")

st.title("📈 Retail Sales Forecaster")
st.markdown(
    "**End-to-End Time-Series Forecasting Engine** | "
    "Benchmarking **Random Forest ML** vs. **7-Day Naive Seasonal Baseline** on holdout retail demand."
)

st.sidebar.header("⚙️ Configuration")
data_source = st.sidebar.radio("Data Source", ["Use Sample Retail Dataset (730 Days)", "Upload Custom CSV"])
test_horizon = st.sidebar.slider("Holdout Forecast Horizon (Days)", min_value=7, max_value=60, value=30)

default_csv_path = Path(__file__).resolve().parent / "data" / "sales.csv"

df_raw = None
if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Sales CSV (must contain 'date' and 'sales')", type=["csv"])
    if uploaded_file is not None:
        df_raw = load_data(uploaded_file)
else:
    if default_csv_path.exists():
        df_raw = load_data(str(default_csv_path))
    else:
        st.sidebar.error("Sample dataset not found at data/sales.csv.")

if df_raw is not None:
    st.subheader("1. Historical Sales Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations", f"{len(df_raw):,} days")
    col2.metric("Date Range", f"{df_raw['date'].min().strftime('%Y-%m-%d')} to {df_raw['date'].max().strftime('%Y-%m-%d')}")
    col3.metric("Average Daily Sales", f"${df_raw['sales'].mean():,.2f}")

    # Feature Engineering & Temporal Split
    df_featured = add_features(df_raw)
    train_df, test_df = train_test_split_temporal(df_featured, test_days=test_horizon)

    st.write(f"**Training Window:** {len(train_df)} days | **Holdout Test Window:** {len(test_df)} days (Chronological split)")

    # Train Models
    with st.spinner("Training Random Forest & Baseline Forecasters..."):
        rf_model = MachineLearningForecaster(model_type="random_forest").fit(train_df)
        naive_model = NaiveSeasonalForecaster(lag=7).fit(train_df)

        y_test = test_df["sales"].values
        rf_preds = rf_model.predict(test_df)
        naive_preds = naive_model.predict(test_df)

        rf_metrics = evaluate_model(y_test, rf_preds)
        naive_metrics = evaluate_model(y_test, naive_preds)

    st.subheader("2. Model Evaluation on Held-Out Test Window")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Random Forest WAPE", f"{rf_metrics['WAPE_pct']}%", delta=f"{rf_metrics['WAPE_pct'] - naive_metrics['WAPE_pct']:.2f}% vs Baseline", delta_color="inverse")
    m_col2.metric("Baseline WAPE", f"{naive_metrics['WAPE_pct']}%")
    m_col3.metric("Random Forest RMSE", f"${rf_metrics['RMSE']:,.2f}")
    m_col4.metric("Baseline RMSE", f"${naive_metrics['RMSE']:,.2f}")

    # Forecast Comparison Chart
    st.subheader("3. Out-of-Sample Forecast vs. Actuals")
    comparison_df = pd.DataFrame({
        "Date": test_df["date"],
        "Actual Sales": y_test,
        "Random Forest Forecast": rf_preds,
        "Naive Seasonal Baseline": naive_preds,
    }).set_index("Date")

    st.line_chart(comparison_df)

    # Optional Prophet section
    with st.expander("Explore Facebook Prophet (Optional Extended Horizon)"):
        prophet_model = train_prophet(df_raw)
        if prophet_model is not None:
            future_df = make_future(prophet_model, periods=test_horizon)
            st.line_chart(future_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
        else:
            st.info("Prophet is optional and not installed in this environment. The Random Forest autoregressive model is active above.")
else:
    st.info("👈 Please select or upload a dataset in the sidebar to begin forecasting.")
