# Retail Sales Forecaster

[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5+-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

A time series forecasting engine for retail store sales prediction. Features chronological train/test splitting (preventing lookahead leakage), multi-horizon lag and rolling window feature engineering, supervised Random Forest regressor benchmarking against seasonal persistence baselines, and comprehensive error evaluation (MAE, RMSE, MAPE, WAPE).

Maintained and enhanced by **M. Abdullah**.

---

## 📈 Methodology & ML Problem

Time series forecasting in retail operations requires predicting future customer demand while avoiding subtle lookahead data leakage. 
Common pitfalls in novice forecasting repositories include:
1. **Random Shuffling Leakage**: Using standard random cross-validation instead of temporal chronological splits.
2. **Target Leakage via Rolling Windows**: Calculating rolling statistics inclusive of the current time step.
3. **No Baseline Comparison**: Claiming high accuracy without benchmarking against a naive persistence baseline (e.g. yesterday's or last week's sales).

### Enhancements Implemented
* **Zero-Leakage Rolling Statistics**: Rolling features are shifted strictly by 1 period (`sales.shift(1).rolling(7).mean()`) so predictions only utilize past information.
* **Strict Temporal Splitting**: The holdout evaluation window is isolated chronologically at the tail of the series.
* **Benchmark Comparison**: Random Forest ML model is benchmarked against a Naive 7-Day Seasonal Persistence baseline.
* **Operational Metrics**: Evaluates MAE, RMSE, MAPE (Mean Absolute Percentage Error), and WAPE (Weighted Absolute Percentage Error).

---

## ⚡ Quickstart

### 1. Installation
```bash
cd retail-sales-forecaster
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Pipeline & Benchmark
```bash
python main.py
```
Outputs out-of-sample performance table comparing the baseline vs machine learning forecaster across the final 30 days.

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 🧪 Testing
```bash
pytest tests/ -v
```

---

## 👨‍💻 Maintainer & Attribution
- **Enhanced Implementation Maintainer**: **M. Abdullah**
- **Original Project Origin**: Derivative work based on open-source project by `torresjchristopher`.
- **Copyright**: Copyright © 2026 M. Abdullah for enhancements and additions.
