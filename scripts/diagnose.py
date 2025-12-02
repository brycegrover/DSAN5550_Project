import pandas as pd

df = pd.read_parquet("processed/features/features_with_targets.parquet")
print(df[["year","month"]].drop_duplicates().sort_values(["year","month"]))
print("Remaining rows:", len(df))