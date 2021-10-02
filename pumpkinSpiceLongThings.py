#pumpkinSpiceLongThings

#Import required
%matplotlib inline
%load_ext tensorboard
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
!pip install gunpowder
!pip install --upgrade pip setuptools
!pip install --upgrade imagecodecs
import random
from tqdm.notebook import tqdm
import matplotlib
import h5py
from skimage import io
from torchsummary import summary
from imgaug import augmenters as iaa

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#any PyTorch dataset class should inherit the initial torch.utils.data.Dataset
class TrainDataset():
  def __init__(self, train_dir, gt_dir):
    self.train_dir = train_dir  # the directory with all the training samples
    self.samples = os.listdir(train_dir) # list the samples
    self.gt_dir = gt_dir
    self.gtsamples = os.listdir(gt_dir)
    #  transformations to apply just to inputs
    self.inp_transforms = transforms.Compose([transforms.Grayscale(), # some of the images are RGB
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])
                                              ])

  # get the total number of samples
  def __len__(self):
    return len(self.samples)

  # fetch the training sample given its index
  def __getitem__(self, idx):
    img_path = os.path.join(self.train_dir, self.samples[idx])
    gt_path = os.path.join(self.gt_dir, self.gtsamples[idx])

    # we'll be using Pillow library for reading files 
    image = Image.open(img_path)
    image = self.inp_transforms(image)
    image = np.array(image)
    gt = Image.open(gt_path)
    gt = np.array(gt)
    gt = np.expand_dims(gt, axis=0)

    # Define transformation
    transformation = iaa.Sequential([
      iaa.Fliplr(0.5),
      iaa.Sometimes(0.7, iaa.Dropout(0.1,0.2)),
      iaa.Sometimes(0.99, iaa.ElasticTransformation(alpha=(30,200), sigma=(8,12))),
      iaa.Sometimes(0.5, iaa.Affine(rotate=(-90,90)))
    ])

    # apply transformation
    image, gt = transformation(images = image, segmentation_maps = gt)

    #apply manual crop
    xCrop = random.randrange(0, image.shape[1]-512)
    yCrop = random.randrange(0, image.shape[2]-512)
    image = image[:, xCrop:xCrop+512, yCrop:yCrop+512]
    gt = gt[:, xCrop:xCrop+512, yCrop:yCrop+512]


    #gt = np.squeeze(gt, axis=3)

    #if self.transform is not None:
    #    image, gt = self.transform([image, gt])
    image, gt = torch.tensor(image), torch.tensor(gt)
    return image, gt

  def getBatch(self,batchSize):
    batch_images = []
    batch_gt = []
    for i in range(batchSize):
        temp_image, temp_gt = self[random.randrange(0,20)]
        batch_images.append(temp_image)
        batch_gt.append(temp_gt)
    batch_images = torch.stack(batch_images, dim=0)
    batch_gt = torch.stack(batch_gt, dim=0)
    return batch_images, batch_gt

TRAIN_DATA_PATH = 'training/training/images/'
GT_DATA_PATH = 'training/training/1st_manual/'
all_train_data = TrainDataset(TRAIN_DATA_PATH,GT_DATA_PATH)

#len(all_train_data)
#dataset = []
#for i in range(10):
  #image1, gt1 = all_train_data[5]
  #fig, ax = plt.subplots(nrows=1, ncols=2)
  #ax[0].imshow(image1[0])
#  ax[1].imshow(gt1[0])

#u-net
from unet_fov import UNet
from utils.disc_loss import DiscriminativeLoss

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

# apply training for one epoch
def train(model, loader, optimizer, loss_function,
          epoch, tb_logger, log_interval=100, log_image_interval=20):

    # set the model to train mode
    model.train()
    numIter = 100
    # iterate over the batches of this epoch
    for batch_id in range(numIter):
        x, y = loader.getBatch(10)
        y = y.float()
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)
        
        # zero the gradients for this iteration
        optimizer.zero_grad()
        
        # apply model and calculate loss
        output = model(x)
        print(output.dtype,y.dtype)
        print(output.min(),output.max())
        print(y.min(),y.max())
        loss = loss_function(output,y)
        loss.backward()
        
        # backpropagate the loss and adjust the parameters
        optimizer.step()
        
        # log to console
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_id * len(x),
                  len(loader),
                  100. * batch_id / len(loader), loss.item()))

       # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=step)
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(tag='input', img_tensor=x.to('cpu'), global_step=step)
                tb_logger.add_images(tag='target', img_tensor=y.to('cpu'), global_step=step)
                tb_logger.add_images(tag='prediction', img_tensor=output.to('cpu').detach(), global_step=step);

# sorensen dice coefficient implemented in torch
# the coefficient takes values in [0, 1], where 0 is
# the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    # the dice coefficient of two sets represented as vectors a, b ca be 
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        prediction = prediction > 0.5
        intersection = np.logical_and(prediction,target).sum()
        numerator = 2*intersection
        denominator = (prediction.sum())+(target.sum())
        return numerator/denominator

# run validation after training epoch
def validate(model, loader, loss_function, metric, tb_logger, step):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0
    
    # disable gradients during validation
    with torch.no_grad():
        
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction,y)
            val_metric += metric(prediction,y).item()
    
    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)
    
    if tb_logger is not None:
        assert step is not None, "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(tag='val_metric', scalar_value=val_metric, global_step=step)
        # we always log the last validation images
        tb_logger.add_images(tag='val_input', img_tensor=x.to('cpu'), global_step=step)
        tb_logger.add_images(tag='val_target', img_tensor=y.to('cpu'), global_step=step)
        tb_logger.add_images(tag='val_prediction', img_tensor=prediction.to('cpu'), global_step=step)
        
    print('\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(val_loss, val_metric))


num_epochs = 10
step = 0
tb_logger = SummaryWriter('DLMBL/testdata/')
optimizer = torch.optim.Adam(net.parameters())
while step < num_epochs:
  train(net, all_train_data, optimizer, loss_fn, step, tb_logger)
  step += 1
  validate(net, all_train_data, loss_fn, DiceCoefficient, tb_logger, step)