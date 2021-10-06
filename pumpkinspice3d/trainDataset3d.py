import os
from torchvision import transforms
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import random
import torch
import gunpowder as gp
from pathlib import Path
import glob

raw = gp.ArrayKey("RAW")
gt = gp.ArrayKey("GT")


training_directory = Path("pumpkinspice3d/training/raw")
gt_directory = Path("pumpkinspice3d/training/gt")
sourcestore = []

for training_zarr in training_directory.iterdir():
    trainingfilename = training_zarr.name
    
    print(trainingfilename)
    raw_source = gp.ZarrSource(f"pumpkinspice3d/training/raw/{trainingfilename}", datasets = {raw: "data"})
    
    gtfilename = trainingfilename.replace(".zarr", "-A.zarr" )
    gt_source = gp.ZarrSource(f"pumpkinspice3d/training/gt/{gtfilename}", datasets = {gt:"data"})
    sourcestore.append((raw_source, gt_source) + gp.MergeProvider() + gp.RandomLocation())

pipeline = tuple(sourcestore) + gp.RandomProvider() + gp.NoiseAugment(raw) + gp.IntensityAugment(raw, scale_min=0.9, scale_max=1.1, shift_min=-0.1, shift_max=0.1)

print(raw_source, gt_source)

pipeline += gp.Reject(gt, 0.00001)

voxel_size = gp.Coordinate((1000, 156, 156))
shape = gp.Coordinate((3,20,20))

#do this in train and validation loop

with gp.build(pipeline):
    request = gp.BatchRequest()
    request.add(raw, shape * voxel_size)
    request.add(gt, shape * voxel_size)

    batch = pipeline.request_batch(request)
    data=batch[raw].data
    gt_data =  batch [gt].data


    print(data.shape)
    print(gt_data.shape)


