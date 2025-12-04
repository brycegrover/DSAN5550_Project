import h5py
import os

data_dir = "/Users/brycegrover/Desktop/DSAN/FALL_2025/DSAN5550/Project/data"
example_file = sorted(
    f for f in os.listdir(data_dir) if f.endswith(".he5"))[0]

path = os.path.join(data_dir, example_file)
print("looking at:", path)

with h5py.File(path, "r") as f:
    print("Top:", list(f.keys()))
    
    grids = f["HDFEOS"]["GRIDS"]
    print("grids:", list(grids.keys()))

    north = grids[list(grids.keys())[0]]
    print("north grids:", list(north.keys()))
    
    data_fields = north["Data Fields"]
    print("data fields:", list(data_fields.keys()))