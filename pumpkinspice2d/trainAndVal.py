import torch
import numpy as np
import torch.nn as nn
from imgaug import augmenters as iaa
import matplotlib
import matplotlib.pyplot as pyplot

device = torch.device("cuda:0")
numIter = 1000

# apply training for one epoch
def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    tb_logger,
    activation,
    log_interval=100,
    log_image_interval=20,
):

    # set the model to train mode
    model.train()
    # iterate over the batches of this epoch
    for batch_id in range(numIter):
        x, y = loader.getBatch(10)
        y = y.float() / 255
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        output = model(x)
        # print("initial output",output.min(), output.max())
        loss = loss_function(output, y)
        loss.backward()
        output = activation(output)
        # backpropagate the loss and adjust the parameters
        optimizer.step()
        step = epoch * numIter + batch_id
        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )
            torch.save(model.state_dict(), f"logs/checkPoint_{step}")
        # log to tensorboard
        
        tb_logger.add_scalar(
            tag="train_loss", scalar_value=loss.item(), global_step=step
        )
        # check if we log images in this iteration
        if step % log_image_interval == 0:
            tb_logger.add_images(tag="input", img_tensor=x.to("cpu"), global_step=step)
            tb_logger.add_images(tag="target", img_tensor=y.to("cpu"), global_step=step)
            tb_logger.add_images(
                tag="prediction",
                img_tensor=output.to("cpu").detach(),
                global_step=step,
            )


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
        intersection = torch.logical_and(prediction, target).sum()
        numerator = 2 * intersection
        denominator = (prediction.sum()) + (target.sum())
        return numerator / denominator


# run validation after training epoch
def validate(
    model, loader, loss_function, metric, tb_logger, step, activation, num_epochs
):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    finalCount = numIter * num_epochs
    val_loss = 0
    val_metric = 0
    count = 0
    valStep = numIter * step
    # disable gradients during validation
    with torch.no_grad():
        xs = []
        ys = []
        predictions = []
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            count += 1
            y = np.expand_dims(y, axis=3)
            transformation = iaa.CropToFixedSize(512, 512)
            x, y = transformation(images=x, segmentation_maps=y)
            x = torch.from_numpy(x[0])
            y = torch.from_numpy(y[0])
            x = torch.unsqueeze(x, 0)
            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)
            y = torch.unsqueeze(y, 0)
            y = torch.squeeze(y, 4)
            y = y.float() / 255
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            val_loss += loss_function(prediction, y)
            val_metric += metric(prediction, y).item()
            prediction = activation(prediction)
            if valStep == finalCount-numIter:
                pyplot.imshow(np.squeeze(prediction.cpu()), cmap = 'gray')
                pyplot.savefig("valPrediction"+str(count)+".png")
            xs.append(x)
            ys.append(y)
            predictions.append(prediction)

    xs = torch.stack(xs, dim=0)
    xs = torch.squeeze(xs, dim=2)
    ys = torch.stack(ys, dim=0)
    ys = torch.squeeze(ys, dim=2)
    predictions = torch.stack(predictions, dim=0)
    predictions = torch.squeeze(predictions, dim=2)
    tb_logger.add_images(tag="val_input", img_tensor=xs.to("cpu"), global_step=valStep)
    tb_logger.add_images(tag="val_target", img_tensor=ys.to("cpu"), global_step=valStep)
    tb_logger.add_images(
        tag="val_prediction",
        img_tensor=predictions.to("cpu"),
        global_step=valStep,
    )

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)

    tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=valStep)
    tb_logger.add_scalar(tag="val_metric", scalar_value=val_metric, global_step=valStep)

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
            val_loss, val_metric
        )
    )
