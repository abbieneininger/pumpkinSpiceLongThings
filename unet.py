from unet_fov import UNet
from utils.disc_loss import DiscriminativeLoss
import torch
import torch.nn as nn
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

out_channels = 1
activation = nn.Sigmoid()
loss_fn = torch.nn.BCEWithLogitsLoss()
dtype = torch.FloatTensor

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

device = torch.device("cuda:0")
net = net.to(device)
summary(net, (1, 512, 512))
