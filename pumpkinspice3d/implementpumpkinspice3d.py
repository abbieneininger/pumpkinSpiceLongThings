from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.tensorboard as tensorboard
from trainDataset3d import pipeline
from trainAndVal3d import train, validate, DiceCoefficient
from unet_fov3d import UNet
import torch.nn as nn
from torchsummary import summary

TRAIN_DATA_PATH = 'training/raw/'
GT_DATA_PATH = 'training/gt/'
VALIDATION_RAW_PATH = 'training/validation/images/'
VALIDATION_GT_PATH = 'training/validation/1st_manual/'

#define unet
out_channels = 1
activation = nn.Sigmoid()
loss_fn = torch.nn.BCEWithLogitsLoss()
dtype = torch.FloatTensor
d_factors = [(1,2,2),(1,2,2)]

net = torch.nn.Sequential(
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

device = torch.device("cuda:0")
net = net.to(device)
num_epochs = 1000
step = 0
tb_logger = SummaryWriter('logs/testRun100221')
optimizer = torch.optim.Adam(net.parameters())
while step < num_epochs:
  train(net, pipeline, optimizer, loss_fn, step, tb_logger, activation)
  step += 1
  #validate(net, validation_loader, loss_fn, DiceCoefficient(), tb_logger, step, activation)