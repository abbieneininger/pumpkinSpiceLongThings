from unet_fov import UNet
from utils.disc_loss import DiscriminativeLoss
import torch
import torch.nn as nn
from torchsummary import summary

prediction_type = "two_class" # same as fg/bg
#prediction_type = "affinities"
#prediction_type = "sdt"
#prediction_type = "three_class"
#prediction_type = "metric_learning"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if prediction_type == "two_class":
    out_channels = 1
    activation = nn.Sigmoid()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dtype = torch.FloatTensor
elif prediction_type == "affinities":
    out_channels = 1
    activation = nn.Sigmoid()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dtype = torch.FloatTensor
elif prediction_type == "sdt":
    out_channels = 1
    activation = nn.Sigmoid()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dtype = torch.FloatTensor
elif prediction_type == "three_class":
    out_channels = 3
    activation = nn.Sigmoid()
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    dtype = torch.FloatTensor
elif prediction_type == "metric_learning":
    out_channels = 5
    activation = None
    loss_fn = DiscriminativeLoss(device)
    dtype = torch.FloatTensor
else:
    raise RuntimeError("invalid prediction type")
torch.manual_seed(42)

d_factors = [(2,2),(2,2),(2,2),(2,2)]

net = torch.nn.Sequential(
    UNet(in_channels=1,
    num_fmaps=8,
    fmap_inc_factors=2,
    downsample_factors=d_factors,
    activation='ReLU',
    padding='same',
    num_fmaps_out=8,
    constant_upsample=True
    ),
    torch.nn.Conv2d(in_channels= 8, out_channels=out_channels, kernel_size=1, padding=0, bias=True))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
summary(net, (1, 384, 384))
