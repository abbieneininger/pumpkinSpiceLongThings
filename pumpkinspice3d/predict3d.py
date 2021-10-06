import gunpowder as gp
from numpy import array
import torch
from unet_fov3d import UNet
import numpy as np

class f64tof32(gp.BatchFilter):
  def __init__(self, array):
    self.array = array

  def setup(self):
    array_spec = self.spec[self.array].copy()
    array_spec.dtype = np.float32
    self.updates(self.array, array_spec)

  def request(self, request):
    return request

  def process(self, batch, request):
    array_spec = batch[self.array].spec.copy()
    array_spec.dtype = np.float32
    output = batch[self.array].data
    output = np.array(output, dtype=np.float32)
    out_array = gp.Array(output, array_spec)

    out_batch = gp.Batch()
    out_batch[self.array] = out_array
    return out_batch


raw = gp.ArrayKey("RAW")
prediction = gp.ArrayKey("PREDICTION")
#loading weights
d_factors = [(1,2,2),(1,2,2)]
out_channels = 1
model = torch.nn.Sequential(
    #use parameters from previous model
UNet(in_channels=1,
    num_fmaps=16,
    fmap_inc_factors=3,
    downsample_factors=d_factors,
    activation='ReLU',
    padding='same',
    num_fmaps_out=16,
    constant_upsample=True
    ),
    torch.nn.Conv3d(in_channels= 16, out_channels=out_channels, kernel_size=1, padding=0, bias=True))

checkpoint = torch.load("/home/whiddonz/pumpkinSpiceLongThings/logs/checkpoint_62000")
model.load_state_dict(checkpoint)



# set model into evaluation mode
model.eval()
raw_source = gp.ZarrSource("/home/whiddonz/pumpkinSpiceLongThings/pumpkinspice3d/test/raw/1216_t0_TA001.zarr", datasets = {raw: "data"})

with gp.build(raw_source):
  roi = raw_source.spec[raw].roi
print(roi)

predict = gp.torch.Predict(
  model,
  inputs = {
    'input': raw
  },
  outputs = {
    0: prediction
  },
  array_specs={prediction: gp.ArraySpec(roi=roi)}
  )
dataset_names={prediction:"data"}
pipeline = raw_source+f64tof32(raw) +gp.Unsqueeze([raw])+ gp.Unsqueeze([raw])+ predict+gp.Squeeze([prediction])+ gp.Squeeze([prediction]) + gp.ZarrWrite(dataset_names, "/home/whiddonz/pumpkinSpiceLongThings/pumpkinspice3d/test/predictions","1216_t0_TA001.zarr")

# request matching the model input and output sizes
"""
scan_request = gp.BatchRequest()
scan_request[raw] = gp.Roi((0, 0), (64, 128))
scan_request[prediction] = gp.Roi((0, 0), (64, 128))

scan = gp.Scan(scan_request)
"""

# request for raw and prediction for the whole image
request = gp.BatchRequest()
# 3d size of your volume
voxel_size=gp.Coordinate((1000, 156, 156))
request.add(raw, roi.get_shape())
request.add(prediction, roi.get_shape())

with gp.build(pipeline):
  batch = pipeline.request_batch(request)

#imshow(batch[raw].data, None, batch[prediction].data)