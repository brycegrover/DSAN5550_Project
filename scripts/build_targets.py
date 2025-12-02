from pathlib import Path
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]
    feats_path = project_root / "processed" / "features" / "features.parquet"
    out_path = project_root / "processed" / "features" / "features_with_targets.parquet"

    print(f"Loading features from {feats_path}")
    df = pd.read_parquet(feats_path)

    required_cols = {
        "iy",
        "ix",
        "lat",
        "lon",
        "ice_conc_mean",
        "year",
        "month",
        "vessel_count",
        "t",
        "ice_conc_prev",
        "vessel_count_prev",
        "delta_ice",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected column(s) in features.parquet: {missing}")

    # binary vessel presence in month
    df["has_vessel"] = (df["vessel_count"] > 0).astype(int)

    # sort to match grid cell
    df = df.sort_values(["iy", "ix", "t"]).reset_index(drop=True)

    # next month vessel presence and count
    df["vessel_count_next"] = df.groupby(["iy", "ix"])["vessel_count"].shift(-1)
    df["has_vessel_next"] = (df["vessel_count_next"] > 0).astype("Int64")

    # drop rows without next month target
    before = len(df)
    df = df.dropna(subset=["vessel_count_next"]).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows without next month target")

    # data type conversions
    df["vessel_count_next"] = df["vessel_count_next"].astype(int)
    df["has_vessel_next"] = df["has_vessel_next"].astype(int)

    # save enriched features
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving enriched features to {out_path}")
    df.to_parquet(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()