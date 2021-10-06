import os
from torchvision import transforms
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import torch

class DiseaseLoader:
    def __init__(self, diseasePath):
        self.disease_dir = diseasePath 
        self.samples = sorted(os.listdir(diseasePath))
        self.numsamples = len(self.samples)
        #  transformations to apply just to inputs
        self.inp_transforms = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor()
            ]
        )
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = os.path.join(self.disease_dir, self.samples[idx])
        image = Image.open(img_path)
        image = self.inp_transforms(image)
        image = np.array(image)
        #transformation = iaa.CropToFixedSize(512,512)
        transformation = iaa.PadToFixedSize(width = 704, height = 704)
        image = transformation(images=image)
        return np.expand_dims(image[0], 0)
