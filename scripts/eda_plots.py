#!/usr/bin/env python3
"""
eda_plots.py

Creates EDA plots for poster:
- Class imbalance for current and next-month presence
- Distributions of vessel counts and ice concentration
- Time series of total vessel activity
- Spatial map of active cells

Plots are saved to processed/analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
FEATURES_PATH = PROCESSED_DIR / "features" / "features_with_targets.parquet"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"


def ensure_output_dir():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_features():
    print(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    if "has_vessel" not in df.columns:
        df["has_vessel"] = (df["vessel_count"] > 0).astype(int)
    if "has_vessel_next" not in df.columns and "vessel_count_next" in df.columns:
        df["has_vessel_next"] = (df["vessel_count_next"] > 0).astype(int)

    return df


def plot_class_imbalance(df, col, out_name):
    counts = df[col].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar")
    plt.xlabel(col)
    plt.ylabel("Number of grid-cell months")
    plt.title(f"Class imbalance: {col}")
    plt.xticks([0, 1], ["0 (no vessel)", "1 (vessel)"], rotation=0)
    plt.tight_layout()
    out_path = ANALYSIS_DIR / out_name
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {col} class imbalance plot to {out_path}")


def plot_histogram(df, col, out_name, bins=50, xlim=None, logy=False):
    plt.figure(figsize=(6, 4))
    plt.hist(df[col].dropna(), bins=bins)
    plt.xlabel(col)
    plt.ylabel("Count")
    if logy:
        plt.yscale("log")
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    out_path = ANALYSIS_DIR / out_name
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved histogram for {col} to {out_path}")


def plot_time_series_total_vessels(df, out_name):
    # group by year, month
    if "year" not in df.columns or "month" not in df.columns:
        print("year or month not found. Skipping time series plot.")
        return

    ts = df.groupby(["year", "month"])["vessel_count"].sum().reset_index()
    # simple datetime index: assume day 1
    ts["date"] = pd.to_datetime(
        ts["year"].astype(int).astype(str) + "-" + ts["month"].astype(int).astype(str) + "-01"
    )
    ts = ts.sort_values("date")

    plt.figure(figsize=(8, 4))
    plt.plot(ts["date"], ts["vessel_count"], marker="o", linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Total vessel detections")
    plt.title("Total Arctic vessel detections per month")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = ANALYSIS_DIR / out_name
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved time series plot to {out_path}")


def plot_spatial_active_map(df, out_name, min_total_vessels=5):
    """
    Simple spatial plot of active cells colored by mean vessel_count.
    """
    group = df.groupby(["iy", "ix"]).agg(
        total_vessels=("vessel_count", "sum"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        mean_vessel=("vessel_count", "mean"),
    ).reset_index()

    active = group[group["total_vessels"] >= min_total_vessels].copy()
    if active.empty:
        print("No active cells found. Skipping spatial map.")
        return

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        active["lon"],
        active["lat"],
        c=active["mean_vessel"],
        s=5,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Mean monthly vessel count")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Active Arctic grid cells (mean vessel activity)")
    plt.tight_layout()
    out_path = ANALYSIS_DIR / out_name
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved spatial active cell map to {out_path}")


def main():
    ensure_output_dir()
    df = load_features()

    # 1. Class imbalance
    if "has_vessel" in df.columns:
        plot_class_imbalance(df, "has_vessel", "class_imbalance_has_vessel.png")
    if "has_vessel_next" in df.columns:
        plot_class_imbalance(df, "has_vessel_next", "class_imbalance_has_vessel_next.png")

    # 2. Distributions
    plot_histogram(df, "vessel_count", "hist_vessel_count.png",
                   bins=50, xlim=(0, 50), logy=True)
    if "vessel_count_next" in df.columns:
        plot_histogram(df, "vessel_count_next", "hist_vessel_count_next.png",
                       bins=50, xlim=(0, 50), logy=True)

    if "ice_conc_mean" in df.columns:
        plot_histogram(df, "ice_conc_mean", "hist_ice_conc_mean.png",
                       bins=30, xlim=(0, 1.0), logy=False)

    # 3. Time series of total vessels
    plot_time_series_total_vessels(df, "time_series_total_vessels.png")

    # 4. Spatial active cell map
    plot_spatial_active_map(df, "spatial_active_cells.png", min_total_vessels=5)


if __name__ == "__main__":
    main()