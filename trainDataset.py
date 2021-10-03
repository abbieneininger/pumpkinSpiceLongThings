import os
from torchvision import transforms
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import random
import torch


class TrainDataset:
    def __init__(self, train_dir, gt_dir):
        self.train_dir = train_dir  # the directory with all the training samples
        self.samples =sorted( os.listdir(train_dir))  # list the samples
        self.gt_dir = gt_dir
        self.numsamples = len(self.samples)
        self.gtsamples = sorted(os.listdir(gt_dir))
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),  # some of the images are RGB
                transforms.ToTensor()
                #transforms.Normalize([0.5], [0.5]),
            ]
        )

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

        # IAA expects n, w, h, c for segmentation_maps
        

        # iaa.Sometimes(0.7, iaa.Dropout(0.1, 0.2)),

        # drop the segmentation_maps dimension
        

        # apply manual crop
        # xCrop = random.randrange(0, image.shape[1] - 512)
        # yCrop = random.randrange(0, image.shape[2] - 512)
        # image = image[:, xCrop : xCrop + 512, yCrop : yCrop + 512]
        # gt = gt[:, xCrop : xCrop + 512, yCrop : yCrop + 512]

        # gt = np.squeeze(gt, axis=3)

        # if self.transform is not None:
        #    image, gt = self.transform([image, gt])
        return image, gt

        # Define transformation

    def transformation(self):
        transformation = iaa.Sequential(
            [
                iaa.CropToFixedSize(256, 256)
                #iaa.Fliplr(0.5),
                #iaa.Sometimes(
                    #0.99, iaa.ElasticTransformation(alpha=(30, 200), sigma=(8, 12))
                #),
                #iaa.Sometimes(0.5, iaa.Affine(rotate=(-90, 90))),
            ]
        )
        return transformation


    def getBatch(self, batchSize):
        batch_images = []
        batch_gt = []
        for i in range(batchSize):
            temp_image, temp_gt = self[random.randrange(0, self.numsamples)]
            temp_gt = np.expand_dims(temp_gt, axis=3)
            transformation = self.transformation()
            temp_image, temp_gt = transformation(
                images=temp_image, segmentation_maps=temp_gt
            )
            temp_gt = temp_gt[:, :, :, 0]
            
            temp_image, temp_gt = torch.tensor(temp_image), torch.tensor(temp_gt)
            batch_images.append(temp_image)
            batch_gt.append(temp_gt)
        batch_images = torch.stack(batch_images, dim=0)
        batch_gt = torch.stack(batch_gt, dim=0)
        return batch_images, batch_gt
