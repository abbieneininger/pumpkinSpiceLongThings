import torch
import numpy as np
import torch.nn as nn
from imgaug import augmenters as iaa
import matplotlib
import matplotlib.pyplot as pyplot

device = torch.device("cuda:0")


def diseaseModel(model, loader, step, activation, tb_logger):
    model.eval()
    with torch.no_grad():
        xs = []
        predictions = []
        for idx, x in enumerate(loader):
            x = torch.from_numpy(x)
            x = torch.unsqueeze(x, 0)
            x = x.to(device)
            prediction = model(x)
            prediction = activation(prediction)
            xs.append(x)
            predictions.append(prediction)
            if step % 500 == 0:
                pyplot.imshow(np.squeeze(prediction.cpu()), cmap="gray")
                pyplot.savefig("prediction" + str(step) + "_" + str(idx) + ".png")
        xs = torch.stack(xs, dim=0)
        xs = torch.squeeze(xs, dim=2)
        predictions = torch.stack(predictions, dim=0)
        predictions = torch.squeeze(predictions, dim=2)
        tb_logger.add_images(
            tag="disease_input", img_tensor=xs.to("cpu"), global_step=step
        )
        tb_logger.add_images(
            tag="disease_prediction",
            img_tensor=predictions.to("cpu").detach(),
            global_step=step,
        )
