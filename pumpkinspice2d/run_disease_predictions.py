from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.tensorboard as tensorboard
from trainDataset import TrainDataset
from trainAndVal import train, validate, DiceCoefficient
from diseasePrediction import diseaseModel
from unet_fov import UNet
import torch.nn as nn
from torchsummary import summary
from diseaseLoader import DiseaseLoader
from PIL import Image
from matplotlib import cm
import numpy as np

DISEASE_DATA_PATH = "retinopathy/raw_images"
disease_loader = DiseaseLoader(DISEASE_DATA_PATH)

checkpoint = "/home/neiningera/DLMBL/pumpkinspice2d/logs/checkPoint_102000"

# define unet
out_channels = 1
activation = nn.Sigmoid()
loss_fn = torch.nn.BCEWithLogitsLoss()
dtype = torch.FloatTensor
d_factors = [(2, 2), (2, 2), (2, 2), (2, 2)]

net = torch.nn.Sequential(
    UNet(
        in_channels=1,
        num_fmaps=16,
        fmap_inc_factors=3,
        downsample_factors=d_factors,
        activation="ReLU",
        padding="same",
        num_fmaps_out=16,
        constant_upsample=True,
    ),
    torch.nn.Conv2d(
        in_channels=16, out_channels=out_channels, kernel_size=1, padding=0, bias=True
    ),
)

def save_pred(pred, name):
    pred = torch.squeeze(pred, 0)
    pred = torch.squeeze(pred, 0)
    im = Image.fromarray(np.uint8(cm.gray(pred.cpu().numpy())*255))
    im.save(name+".png", "PNG")

net.load_state_dict(torch.load(checkpoint))
device = torch.device("cuda")
net.to(device)
net.eval()
with torch.no_grad():
    for idx, x in enumerate(disease_loader):
        x = torch.from_numpy(x)
        x = torch.unsqueeze(x, 0)
        x = x.to(device)
        prediction = net(x)
        prediction = activation(prediction)
        print(x.shape)
        print(prediction.shape)
        save_pred(prediction, f"disease_{idx}")