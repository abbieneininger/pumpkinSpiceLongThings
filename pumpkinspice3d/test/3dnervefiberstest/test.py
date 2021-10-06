import tifffile
import zarr
from pathlib import Path
import math

raw_path= Path("./")
for tiff_file in raw_path.iterdir():
    if tiff_file.name.endswith(".tif"):
        name = tiff_file.name[:-4]
        name=name.replace(" ", "_")

        data = tifffile.imread(tiff_file)
        data = data / data.max()
        print(data.min(), data.max())

        zarr_container = zarr.open(f"/home/whiddonz/pumpkinSpiceLongThings/pumpkinspice3d/test/raw/{name}.zarr", "w")
        dataset = zarr_container.create_dataset("data", shape=data.shape)
        dataset[:] = data[:]

        dataset.attrs["resolution"] = (1000,156,156)