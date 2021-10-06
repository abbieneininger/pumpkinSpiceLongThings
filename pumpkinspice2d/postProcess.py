from PIL import Image
import os
import numpy as np
import pandas as pd
from pandas import concat
import skimage
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis, skeletonize
import skan
from skan import Skeleton, summarize
from imgaug import augmenters as iaa


def importBinary(directory, samples, idx):
    img_path = os.path.join(directory, samples[idx])
    print(img_path)
    binary = Image.open(img_path)
    binary = (np.array(binary) / 255)
    binary = binary > 0.2
    if len(binary.shape) > 2:
        binary = binary[:,:,0]
    return binary

def importRawImages(directory, samples, idx):
    img_path = os.path.join(directory, samples[idx])
    print(img_path)
    image = Image.open(img_path)
    image = np.array(image)
    return image


def skeletonizeImg(binary):
    skeleton = skeletonize(binary)
    skelMed, distance = medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skelMed
    branch_data = summarize(Skeleton(skeleton))
    return skeleton, branch_data, dist_on_skel


def histograms(control_branches, disease_branches):
    # con_branch_distances = control_branches["branch-distance"]
    # dis_branch_distances = disease_branches["branch-distance"]
    # con_branch_types = control_branches["branch-type"]
    # dis_branch_types = disease_branches["branch-type"]
    fig, axes1 = plt.subplots(2, 2, figsize=(8, 8), sharex=False, sharey=False)
    #ax = axes.ravel()
    control_branches.hist(
        #title="Branch Distances Control (pixels)",
        column="branch-distance",
        bins=100,
        ax=axes1[0, 0],
        sharex=False, sharey=False
    )
    disease_branches.hist(
        #title="Branch Distances Disease (pixels)",
        column="branch-distance",
        bins=100,
        ax=axes1[1, 0],
        sharex=False, sharey=False
    )
    control_branches.hist(#title="Branch Types Control", 
        column="branch-type", bins = 4, ax=axes1[0, 1],
        sharex=False, sharey=False
        )
    
    disease_branches.hist(
        #title="Branch Types Diseased", 
        column="branch-type", bins = 4, ax=axes1[1, 1],
        sharex=False, sharey=False
    )
    plt.savefig("Final Histogram Figure")


def saveData(branch_data, idx, group):
    branch_data.to_csv(path_or_buf=f"branch{group}_{idx}.csv")


def figureMaker(image, binary, skeleton, dist_on_skel, group, idx):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    dist_on_skel *= 20
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(binary, cmap=plt.cm.gray)
    ax[1].set_title("Prediction")
    ax[1].axis("off")

    ax[2].imshow(skeleton, cmap=plt.cm.gray)
    ax[2].set_title("Skeleton")
    ax[2].axis("off")

    ax[3].imshow(dist_on_skel, cmap="magma")
    ax[3].set_title("Distance Transform")
    ax[3].axis("off")

    fig.tight_layout()
    plt.savefig(f"skeleton_images{group}_{idx}")


def main(conPATH, disPATH, conRawPATH, disRawPATH):
    control_dir = conPATH
    disease_dir = disPATH
    control_samples = sorted(os.listdir(control_dir))
    disease_samples = sorted(os.listdir(disease_dir))
    control_numsamples = len(control_samples)
    disease_numsamples = len(disease_samples)
    control_raw_dir = conRawPATH
    disease_raw_dir = disRawPATH
    control_raw_samples = sorted(os.listdir(control_raw_dir))
    disease_raw_samples = sorted(os.listdir(disease_raw_dir))
    control_branches = []
    disease_branches = []
    for i in range(control_numsamples):
        binary = importBinary(control_dir, control_samples, i)
        print(binary.shape)
        image = importRawImages(control_raw_dir, control_raw_samples, i)
        skeleton, branch_data, dist_on_skel = skeletonizeImg(binary)
        #saveData(branch_data, i, "control")
        control_branches.append(branch_data)
        figureMaker(image, binary, skeleton, dist_on_skel, "control", i)

    for j in range(disease_numsamples):
        binary = importBinary(disease_dir, disease_samples, j)
        print(binary.shape)
        image = importRawImages(disease_raw_dir, disease_raw_samples, j)
        skeleton, branch_data, dist_on_skel = skeletonizeImg(binary)
        #saveData(branch_data, j, "disease")
        disease_branches.append(branch_data)
        figureMaker(image, binary, skeleton, dist_on_skel, "disease", j)

    #control_branches = pd.concat(control_branches)
    ##disease_branches = pd.concat(disease_branches)
    #control_branches.to_csv(path_or_buf="test.csv")
    #disease_branches.to_csv(path_or_buf="disease.csv")
    #histograms(control_branches, disease_branches)


if __name__ == "__main__":
    main(
        "training/training/padded_preds",
        "retinopathy/predictions",
        "training/training/images",
        "retinopathy/raw_images",
    )
