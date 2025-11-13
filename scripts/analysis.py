import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "processed" / "features" / "features.parquet"

def load_data():
    print("Loading feature table...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"{len(df):,} rows loaded.")
    return df

def basic_checks(df):
    print("Columns:", df.columns.tolist())
    print("Years present:", sorted(df['year'].unique()))
    print("Months present:", sorted(df['month'].unique()))
    print("Any missing ice values?:", df['ice_conc_mean'].isna().any())
    print("Any missing vessel counts?:", df['vessel_count'].isna().any())
    print(df[['ice_conc_mean', 'vessel_count']].describe())

def correlation_check(df):
    clean = df.dropna(subset=['ice_conc_mean'])
    corr = clean['ice_conc_mean'].corr(clean['vessel_count'])
    print("\n=== Correlation check ===")
    print(f"Pearson correlation (ice vs vessels): {corr:.3f}")

def seasonal_plot(df):
    monthly = df.groupby(['year', 'month']).agg({
        'ice_conc_mean': 'mean',
        'vessel_count': 'sum'
    }).reset_index()

    seasonal = monthly.groupby('month').agg({
        'ice_conc_mean': 'mean',
        'vessel_count': 'mean'
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(seasonal['month'], seasonal['ice_conc_mean'], label='Ice concentration', linewidth=2)
    ax1.set_ylabel("Mean ice concentration")
    ax1.set_xlabel("Month")

    ax2 = ax1.twinx()
    ax2.plot(seasonal['month'], seasonal['vessel_count'], label='Vessel activity', color='tab:red', linewidth=2)
    ax2.set_ylabel("Avg vessel detections")

    plt.title("Seasonal Arctic cycle: ice vs vessel detections")
    plt.tight_layout()
    out = PROJECT_ROOT / "processed" / "seasonal_cycle.png"
    plt.savefig(out)
    print(f"\nSeasonal plot saved to {out}")


def simple_regression(df):
    from sklearn.linear_model import LinearRegression

    clean = df.dropna(subset=['ice_conc_mean', 'vessel_count'])
    X = clean[['ice_conc_mean']].values
    y = clean['vessel_count'].values

    model = LinearRegression()
    model.fit(X, y)

    print("\n=== Simple regression: vessel_count ~ ice_conc_mean ===")
    print(f"coef: {model.coef_[0]:.4f}")
    print(f"intercept: {model.intercept_:.4f}")
    print("Interpretation: negative coefficient means vessels appear when ice is lower.")


def main():
    df = load_data()
    basic_checks(df)
    correlation_check(df)
    simple_regression(df)
    seasonal_plot(df)

if __name__ == "__main__":
    main()
