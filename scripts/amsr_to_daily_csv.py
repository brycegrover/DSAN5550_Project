import h5py
import numpy as np
import pandas as pd
import os
import re

DATA_DIR = "/Users/brycegrover/Desktop/DSAN/FALL_2025/DSAN5550/Project/data"
OUTPUT_CSV = "/Users/brycegrover/Desktop/DSAN/FALL_2025/DSAN5550/Project/sea_ice_daily_summary.csv"

NORTH_GRID_NAME = "NpPolarGrid12km"
ICE_VAR_NAME = "SI_12km_NH_ICECON_DAY"

# CHAT GPT USED TO MAKE BELOW CODE
# BEGIN AI CODE

def extract_date(filename: str) -> pd.Timestamp:
    m = re.search(r"_(\d{8})\.he5$", filename)
    if not m:
        raise ValueError(f"Could not parse date from {filename}")
    return pd.to_datetime(m.group(1), format="%Y%m%d")

rows = []

files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".he5")
)

print(f"Found {len(files)} he5 files")

for fname in files:
    fpath = os.path.join(DATA_DIR, fname)
    date = extract_date(fname)

    with h5py.File(fpath, "r") as f:
        grids = f["HDFEOS"]["GRIDS"]
        north = grids[NORTH_GRID_NAME]
        data_fields = north["Data Fields"]
        ice_raw = data_fields[ICE_VAR_NAME][:]

    ice = ice_raw.astype("float32")

    ice[ice >= 110] = np.nan

    ice_frac = ice / 100.0  # convert percent to 0 to 1

    mean_conc = np.nanmean(ice_frac)
    median_conc = np.nanmedian(ice_frac)
    covered_cells = int(np.sum(~np.isnan(ice_frac)))
    total_cells = int(ice_frac.size)

    rows.append(
        {
            "date": date,
            "mean_ice_conc_frac": float(mean_conc),
            "median_ice_conc_frac": float(median_conc),
            "num_valid_cells": covered_cells,
            "total_cells": total_cells,
        }
    )

df = pd.DataFrame(rows).sort_values("date")
df.to_csv(OUTPUT_CSV, index=False)
print("Wrote:", OUTPUT_CSV)

# END AI CODE
