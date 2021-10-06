import tifffile
tiff = tifffile.imread("/home/whiddonz/pumpkinSpiceLongThings/pumpkinspice3d/tiff_data/3dnervefiberimages/233 t0 TA001.tif")
print (tiff.shape)
print(tiff)

import zarr
from pathlib import Path
import math

gt_path= Path("tiff_data/3dnervefibersGT")
for tiff_file in gt_path.iterdir():
    name = tiff_file.name[:-4]
    name=name.replace(" ", "_")

    data = tifffile.imread(tiff_file)
    data = data[::-1] / data.max()
    z, y, x = data.shape
    targetz = math.ceil(z * 156 / 1000)
    print(z, targetz)

    zarr_container = zarr.open(f"training/gt/{name}.zarr", "w")
    dataset = zarr_container.create_dataset("data", shape=(targetz, y, x))
    i = 0
    for z_ind in range(z):
        z_coordinate = z_ind * 156
        mod_1000 = z_coordinate % 1000
        dist_to_1000 = min(mod_1000, 1000 - mod_1000)
        if dist_to_1000 < 156/2:
            print(i, z_ind)
            dataset[i, :, :] = data[z_ind, :, :]
            i += 1

    dataset.attrs["resolution"] = (1000,156,156)