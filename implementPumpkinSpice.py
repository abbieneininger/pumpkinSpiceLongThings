from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.tensorboard as tensorboard
from trainDataset import TrainDataset
from trainAndVal import train, validate, DiceCoefficient
from unet_fov import UNet
import torch.nn as nn
from torchsummary import summary

TRAIN_DATA_PATH = 'training/training/images/'
GT_DATA_PATH = 'training/training/1st_manual/'
loader = TrainDataset(TRAIN_DATA_PATH,GT_DATA_PATH)

#define unet
out_channels = 1
activation = nn.Sigmoid()
loss_fn = torch.nn.BCEWithLogitsLoss()
dtype = torch.FloatTensor
d_factors = [(2,2),(2,2),(2,2),(2,2)]

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
    torch.nn.Conv2d(in_channels= 16, out_channels=out_channels, kernel_size=1, padding=0, bias=True))

device = torch.device("cuda:0")
net = net.to(device)
num_epochs = 10
step = 0
tb_logger = SummaryWriter('logs/testRun100221')
optimizer = torch.optim.Adam(net.parameters())
while step < num_epochs:
  train(net, loader, optimizer, loss_fn, step, tb_logger)
  step += 1
  validate(net, loader, loss_fn, DiceCoefficient(), tb_logger, step)