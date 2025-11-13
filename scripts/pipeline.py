import os
from pathlib import Path
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ICE_DIR = PROJECT_ROOT / "ice_daily_grids"
RAW_VESSEL_DIR = PROJECT_ROOT / "vessels"

PROCESSED_DIR = PROJECT_ROOT / "processed"
ICE_MONTHLY_DIR = PROCESSED_DIR / "ice_monthly"
VESSELS_MONTHLY_DIR = PROCESSED_DIR / "vessels_monthly"
FEATURES_DIR = PROCESSED_DIR / "features"
GRID_DIR = PROCESSED_DIR / "grid"

for d in [PROCESSED_DIR, ICE_MONTHLY_DIR, VESSELS_MONTHLY_DIR, FEATURES_DIR, GRID_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ARCTIC_LAT_MIN = 50.0


def build_or_load_grid_index():

    grid_meta_csv = GRID_DIR / "grid_lookup.csv"
    grid_npz = GRID_DIR / "grid_kdtree.npz"

    if grid_meta_csv.exists() and grid_npz.exists():
        print("Loading existing grid index")
        meta = pd.read_csv(grid_meta_csv)
        lat_lon = np.load(grid_npz)["lat_lon"]
        tree = KDTree(lat_lon, metric="euclidean")
        return meta, tree

    print("Building grid index from first ice CSV")

    ice_files = sorted(RAW_ICE_DIR.glob("ice_grid_*.csv"))
    if not ice_files:
        raise FileNotFoundError(f"No ice_grid_*.csv files found in {RAW_ICE_DIR}")

    sample = pd.read_csv(ice_files[0], usecols=["iy", "ix", "lat", "lon"])
    meta = sample.drop_duplicates(subset=["iy", "ix"]).reset_index(drop=True)

    lat_lon = meta[["lat", "lon"]].to_numpy(dtype="float64")
    tree = KDTree(lat_lon, metric="euclidean")

    meta.to_csv(grid_meta_csv, index=False)
    np.savez_compressed(grid_npz, lat_lon=lat_lon)

    print(f"Grid index built with {len(meta)} cells")
    return meta, tree


def load_grid_index():
    grid_meta_csv = GRID_DIR / "grid_lookup.csv"
    grid_npz = GRID_DIR / "grid_kdtree.npz"
    if not (grid_meta_csv.exists() and grid_npz.exists()):
        raise FileNotFoundError("Grid index not found. Run build_or_load_grid_index() first.")
    meta = pd.read_csv(grid_meta_csv)
    lat_lon = np.load(grid_npz)["lat_lon"]
    tree = KDTree(lat_lon, metric="euclidean")
    return meta, tree



def process_ice_daily_to_monthly():

    ice_files = sorted(RAW_ICE_DIR.glob("ice_grid_*.csv"))
    if not ice_files:
        raise FileNotFoundError(f"No ice_grid_*.csv files found in {RAW_ICE_DIR}")

    files_by_ym = defaultdict(list)
    for fpath in ice_files:
        fname = fpath.name
        date_str = fname.split("_")[2].split(".")[0]  # 'YYYYMMDD'
        year = int(date_str[:4])
        month = int(date_str[4:6])
        files_by_ym[(year, month)].append(fpath)

    for (year, month), paths in sorted(files_by_ym.items()):
        print(f"[ICE] Year {year}, month {month:02d}: {len(paths)} daily files")

        frames = []
        for p in paths:
            df_day = pd.read_csv(
                p,
                usecols=["iy", "ix", "lat", "lon", "ice_conc_frac"]
            )
            frames.append(df_day)

        month_df = pd.concat(frames, ignore_index=True)


        agg = (
            month_df
            .groupby(["iy", "ix", "lat", "lon"], observed=True)["ice_conc_frac"]
            .mean()
            .reset_index()
        )
        agg = agg.rename(columns={"ice_conc_frac": "ice_conc_mean"})
        agg["year"] = year
        agg["month"] = month

        out_file = ICE_MONTHLY_DIR / f"ice_monthly_{year}{month:02d}.parquet"
        agg.to_parquet(out_file, index=False)
        print(f"  Wrote {out_file} with {len(agg)} grid cells")

    print("Ice monthly aggregation complete.")


def map_points_to_grid(lat, lon, tree, grid_meta, max_deg_distance=5.0):
    coords = np.column_stack([lat, lon]).astype("float64")
    dist, idx = tree.query(coords, k=1)
    dist = dist.flatten()
    idx = idx.flatten()

    valid_mask = dist <= max_deg_distance
    iy = grid_meta.loc[idx[valid_mask], "iy"].to_numpy()
    ix = grid_meta.loc[idx[valid_mask], "ix"].to_numpy()

    return valid_mask, iy, ix


def process_vessels_to_monthly(chunk_size=500_000):

    grid_meta, tree = load_grid_index()

    vessel_files = sorted(RAW_VESSEL_DIR.glob("sar_vessel_detections_*.csv"))
    if not vessel_files:
        raise FileNotFoundError(f"No sar_vessel_detections_*.csv in {RAW_VESSEL_DIR}")

    monthly_chunks = []
    usecols = ["timestamp", "lat", "lon"]

    for i, fpath in enumerate(vessel_files, start=1):
        fname = fpath.name
        print(f"[VESSEL] {i}/{len(vessel_files)}  {fname}")

        for chunk in pd.read_csv(fpath, usecols=usecols, chunksize=chunk_size):

            chunk = chunk[chunk["lat"] >= ARCTIC_LAT_MIN]
            if chunk.empty:
                continue


            chunk["timestamp"] = pd.to_datetime(
                chunk["timestamp"], utc=True, errors="coerce"
            )
            chunk = chunk.dropna(subset=["timestamp"])
            if chunk.empty:
                continue

            chunk["year"] = chunk["timestamp"].dt.year
            chunk["month"] = chunk["timestamp"].dt.month

            valid_mask, iy, ix = map_points_to_grid(
                chunk["lat"].to_numpy(),
                chunk["lon"].to_numpy(),
                tree,
                grid_meta,
                max_deg_distance=5.0,
            )
            if not valid_mask.any():
                continue

            chunk = chunk.loc[valid_mask].copy()
            chunk["iy"] = iy
            chunk["ix"] = ix

            grp = (
                chunk.groupby(["year", "month", "iy", "ix"], observed=True)
                     .size()
                     .rename("vessel_count")
                     .reset_index()
            )
            monthly_chunks.append(grp)

    if not monthly_chunks:
        print("No vessel detections survived filtering.")
        return

    all_monthly = pd.concat(monthly_chunks, ignore_index=True)
    vessels_monthly = (
        all_monthly.groupby(["year", "month", "iy", "ix"], observed=True)["vessel_count"]
                   .sum()
                   .reset_index()
    )

    out_file = VESSELS_MONTHLY_DIR / "vessels_monthly.parquet"
    vessels_monthly.to_parquet(out_file, index=False)
    print(f"Vessel monthly aggregation complete: {out_file} ({len(vessels_monthly)} rows)")


def build_feature_table():
    ice_files = sorted(ICE_MONTHLY_DIR.glob("ice_monthly_*.parquet"))
    if not ice_files:
        raise FileNotFoundError(f"No ice_monthly_*.parquet in {ICE_MONTHLY_DIR}")

    ice_list = [pd.read_parquet(f) for f in ice_files]
    ice = pd.concat(ice_list, ignore_index=True)

    vessels_path = VESSELS_MONTHLY_DIR / "vessels_monthly.parquet"
    if not vessels_path.exists():
        raise FileNotFoundError(f"{vessels_path} not found; run process_vessels_to_monthly() first.")
    vessels = pd.read_parquet(vessels_path)

    df = pd.merge(
        ice,
        vessels,
        on=["year", "month", "iy", "ix"],
        how="outer",
    )

    df["vessel_count"] = df["vessel_count"].fillna(0.0)

    df = df.sort_values(["iy", "ix", "year", "month"]).reset_index(drop=True)
    df["t"] = df["year"] * 12 + df["month"]

    df["ice_conc_prev"] = df.groupby(["iy", "ix"])["ice_conc_mean"].shift(1)
    df["vessel_count_prev"] = df.groupby(["iy", "ix"])["vessel_count"].shift(1)
    df["delta_ice"] = df["ice_conc_mean"] - df["ice_conc_prev"]

    out_file = FEATURES_DIR / "features.parquet"
    df.to_parquet(out_file, index=False)
    print(f"Feature table written: {out_file} ({len(df)} rows)")



def run_all():
    print("Step 1: Grid index")
    build_or_load_grid_index()

    print("Step 2: Ice daily -> monthly")
    process_ice_daily_to_monthly()

    print("Step 3: Vessels -> monthly grid counts")
    process_vessels_to_monthly()

    print("Step 4: Build feature table")
    build_feature_table()

    print("Pipeline complete.")


if __name__ == "__main__":
    run_all()
