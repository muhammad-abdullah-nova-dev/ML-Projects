"""
Data ingestion and synthesis utilities for tabular credit risk dataset.
Supports downloading remote datasets, loading local CSVs, and generating
statistically authentic synthetic credit risk datasets (Give Me Some Credit schema)
for offline demonstration, testing, and reproducibility.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET = DEFAULT_DIR / "sample_credit_data.csv"


def generate_synthetic_credit_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic tabular credit scoring dataset mirroring the
    'Give Me Some Credit' Kaggle dataset schema with representative class imbalance (~7% default rate)
    and realistic missing data patterns.
    """
    rng = np.random.RandomState(random_state)

    # 1. Age (normally distributed around 50, bounded 21-85)
    age = np.clip(rng.normal(loc=52, scale=14, size=n_samples).astype(int), 21, 95)

    # 2. Number of open credit lines and loans
    open_lines = np.clip(rng.poisson(lam=8.5, size=n_samples), 0, 35)

    # 3. Number of real estate loans
    real_estate = np.clip(rng.poisson(lam=1.0, size=n_samples), 0, 10)

    # 4. Monthly income (log-normal, with ~15% missingness)
    raw_income = rng.lognormal(mean=8.6, sigma=0.7, size=n_samples)
    income_missing_mask = rng.binomial(1, 0.15, size=n_samples).astype(bool)
    monthly_income = np.where(income_missing_mask, np.nan, np.round(raw_income, 0))

    # 5. Debt ratio (positive, right-skewed)
    debt_ratio = np.round(rng.exponential(scale=0.35, size=n_samples) + (real_estate * 0.1), 3)

    # 6. Revolving utilization (0 to 1 normally, occasionally higher for distressed)
    rev_util = np.round(np.clip(rng.beta(a=1.5, b=5.0, size=n_samples) * 1.5, 0.0, 2.0), 3)

    # 7. Past due delinquency counters
    past_due_30_59 = rng.poisson(lam=0.25, size=n_samples)
    past_due_60_89 = rng.poisson(lam=0.10, size=n_samples)
    past_due_90 = rng.poisson(lam=0.08, size=n_samples)

    # 8. Number of dependents (with ~3% missingness)
    dependents_raw = rng.poisson(lam=0.75, size=n_samples).astype(float)
    dep_missing = rng.binomial(1, 0.03, size=n_samples).astype(bool)
    dependents = np.where(dep_missing, np.nan, dependents_raw)

    # 9. Latent logit for SeriousDlqin2yrs (realistic risk factors)
    # Higher delinquency, debt ratio, and utilization increase risk; higher income & age reduce risk
    norm_inc = np.nan_to_num(monthly_income, nan=5000.0) / 10000.0
    logit = (
        -3.6
        + 1.8 * rev_util
        + 1.2 * past_due_30_59
        + 1.6 * past_due_60_89
        + 2.2 * past_due_90
        + 0.5 * debt_ratio
        - 0.03 * (age - 40)
        - 0.4 * norm_inc
        + rng.normal(0, 0.8, size=n_samples)
    )
    prob_default = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(0, 1, size=n_samples) < prob_default).astype(int)

    df = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": rev_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": past_due_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": past_due_90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": past_due_60_89,
        "NumberOfDependents": dependents,
        "SeriousDlqin2yrs": target
    })

    return df


def ensure_dataset(data_path: str = None) -> Path:
    target_path = Path(data_path) if data_path else DEFAULT_DATASET
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        print(f"Dataset not found at {target_path}. Generating synthetic credit risk dataset...")
        df = generate_synthetic_credit_data(n_samples=3000, random_state=42)
        df.to_csv(target_path, index=False)
        print(f"Generated {len(df)} samples ({df['SeriousDlqin2yrs'].mean()*100:.2f}% default rate) -> {target_path}")
    return target_path


def load_dataset(data_path: str = None) -> pd.DataFrame:
    path = ensure_dataset(data_path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Credit risk dataset ingestion and generation tool")
    parser.add_argument("--out", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_credit_data(n_samples=args.samples)
    df.to_csv(out_path, index=False)
    print(f"Saved dataset with {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()