import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "processed" / "features" / "features.parquet"


def load_features():
    print(f"Loading {FEATURES_PATH} ...")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"{len(df):,} rows loaded")
    return df


def spatial_sanity_map(df, year=2018, month=8, vmax_vessels=None):

    sel = df[(df["year"] == year) & (df["month"] == month)].copy()
    if sel.empty:
        raise ValueError(f"No rows for year={year}, month={month}")

    print(f"Selected {len(sel):,} grid cells for {year}-{month:02d}")

    if sel["lon"].max() > 180:
        sel["lon_plot"] = ((sel["lon"] + 180) % 360) - 180
    else:
        sel["lon_plot"] = sel["lon"]

    ice = sel["ice_conc_mean"]
    vessels = sel["vessel_count"].fillna(0)

    if vmax_vessels is None:
        vmax_vessels = np.percentile(vessels[vessels > 0], 95) if (vessels > 0).any() else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Plot 1: ice concentration
    sc1 = axes[0].scatter(
        sel["lon_plot"],
        sel["lat"],
        c=ice,
        s=5,
        marker="s",
        edgecolors="none",
    )
    axes[0].set_title(f"Ice concentration {year}-{month:02d}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    cbar1 = fig.colorbar(sc1, ax=axes[0])
    cbar1.set_label("ice_conc_mean")

    # Plot 2: vessel detections per grid cell
    sc2 = axes[1].scatter(
        sel["lon_plot"],
        sel["lat"],
        c=vessels,
        s=5,
        marker="s",
        edgecolors="none",
        vmin=0,
        vmax=vmax_vessels,
    )
    axes[1].set_title(f"Vessel detections {year}-{month:02d}")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    cbar2 = fig.colorbar(sc2, ax=axes[1])
    cbar2.set_label("vessel_count (clipped)")

    for ax in axes:
        ax.set_aspect("equal")

    out_path = PROJECT_ROOT / "processed" / f"spatial_sanity_{year}{month:02d}.png"
    plt.savefig(out_path, dpi=200)
    print(f"Spatial sanity map saved to {out_path}")


def main():
    df = load_features()
    spatial_sanity_map(df, year=2024, month=8)


if __name__ == "__main__":
    main()
