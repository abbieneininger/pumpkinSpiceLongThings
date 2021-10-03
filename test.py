from trainDataset import TrainDataset
import matplotlib.pyplot as plt

dataset = TrainDataset('training/training/images', 'training/training/1st_manual')

for i in range(10):
    image1,gt1 = dataset[i]
    fig,ax = plt.subplots(nrows=1,ncols=2)
    ax[0].imshow(image1[0])
    ax[1].imshow(gt1[0])

    plt.savefig(f'test_{i}.png')