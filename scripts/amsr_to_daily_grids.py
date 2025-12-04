import h5py
import numpy as np
import pandas as pd
import os
import re

DATA_DIR = "/Users/brycegrover/Desktop/DSAN/FALL_2025/DSAN5550/Project/raw_data"
OUT_DIR = "/Users/brycegrover/Desktop/DSAN/FALL_2025/DSAN5550/Project/ice_daily_grids"

os.makedirs(OUT_DIR, exist_ok=True)

NORTH_GRID_NAME = "NpPolarGrid12km"
ICE_VAR_NAME = "SI_12km_NH_ICECON_DAY"

def extract_date_str(filename: str) -> str:
    m = re.search(r"_(\d{8})\.he5$", filename)
    if not m:
        raise ValueError(f"Could not parse date from {filename}")
    return m.group(1)

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".he5"))

print(f"Found {len(files)} he5 files")

for fname in files:
    date_str = extract_date_str(fname)
    out_csv = os.path.join(OUT_DIR, f"ice_grid_{date_str}.csv")

    if os.path.exists(out_csv):
        print("Skipping existing files", out_csv)
        continue

    fpath = os.path.join(DATA_DIR, fname)
    with h5py.File(fpath, "r") as f:
        north = f["HDFEOS"]["GRIDS"][NORTH_GRID_NAME]
        ice_raw = north["Data Fields"][ICE_VAR_NAME][:]
        lat = north["lat"][:]
        lon = north["lon"][:]

    invalid = ice_raw >= 250

    ice = ice_raw.astype("float32")
    ice[invalid] = np.nan

    ice_frac = ice / 100.0

    ny, nx = ice_frac.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    df = pd.DataFrame({
        "date": date_str,
        "iy": yy.ravel(),
        "ix": xx.ravel(),
        "lat": lat.ravel(),
        "lon": lon.ravel(),
        "ice_conc_frac": ice_frac.ravel(),
    })

    df = df.dropna(subset=["ice_conc_frac"])

    df.to_csv(out_csv, index=False)
    print("Wrote", out_csv)
