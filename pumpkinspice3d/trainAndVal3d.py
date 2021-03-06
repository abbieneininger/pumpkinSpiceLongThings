import torch
import numpy as np
import torch.nn as nn
from imgaug import augmenters as iaa
import matplotlib
import matplotlib.pyplot
import gunpowder as gp

device = torch.device("cuda:0")
numIter = 1000

# apply training for one epoch
def train(
    model,
    pipeline,
    optimizer,
    loss_function,
    epoch,
    tb_logger,
    activation,
    log_interval=100,
    log_image_interval=20,):

    # set the model to train mode
    model.train()
   
    # iterate over the batches of this epoch

    with gp.build(pipeline):
        
        for batch_id in range(numIter):

            raw = gp.ArrayKey("RAW")
            gt = gp.ArrayKey("GT")
            voxel_size = gp.Coordinate((1000, 156, 156))
            shape = gp.Coordinate((3,20,20))

            
            
            request = gp.BatchRequest()
            request.add(raw, shape * voxel_size)
            request.add(gt, shape * voxel_size)

            batch = pipeline.request_batch(request)
            x = torch.from_numpy(batch[raw].data)
            y =  torch.from_numpy(batch [gt].data)
            x=torch.unsqueeze(torch.unsqueeze(x,0),0).float()
            y=torch.unsqueeze(torch.unsqueeze(y,0),0).float()

            #y = y.float()/255
            
            x, y = x.to(device), y.to(device)

            # zero the gradients for this iteration
            optimizer.zero_grad()

            # apply model and calculate loss
            output = model(x)



            #print("initial output",output.min(), output.max())
            loss = loss_function(output, y)
            loss.backward()
            #print("loss at iteration", batch_id, loss.detach().cpu())
            output = activation(output)
            #print("output after softmax",output.min(),output.max())
            # backpropagate the loss and adjust the parameters
            optimizer.step()

            # log to console

            if batch_id % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_id * len(x),
                        numIter,
                        100.0 * batch_id / numIter,
                        loss.item(), 
                    )
                )
                torch.save(model.state_dict(), f"logs/checkpoint_{batch_id+epoch*numIter}")



            # log to tensorboard
            if tb_logger is not None:
                step = epoch * numIter + batch_id
                tb_logger.add_scalar(
                    tag="train_loss", scalar_value=loss.item(), global_step=step
                )
                # check if we log images in this iteration
                
                if step % log_image_interval == 0:
                    tb_logger.add_images(
                        tag="input", img_tensor= np.max(x.to("cpu").numpy(),axis=2), global_step=step
                    )
                    tb_logger.add_images(
                        tag="target", img_tensor= np.max(y.to("cpu").numpy() ,axis=2), global_step=step
                    )
                    tb_logger.add_images(
                        tag="prediction", img_tensor=np.max(output.to("cpu").detach().numpy(), axis=2),
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
def validate(model, loader, loss_function, metric, tb_logger, step, activation):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0
    count = 0
    # disable gradients during validation
    with torch.no_grad():

        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            count += 1
            y = np.expand_dims(y, axis=3)
            transformation = iaa.CropToFixedSize(512, 512)
            x, y = transformation(images = x, segmentation_maps = y)
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
            #matplotlib.pyplot.imsave("prediction"+str(count)+".tiff",prediction.cpu())
            val_loss += loss_function(prediction, y)
            val_metric += metric(prediction, y).item()

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)
    valStep = numIter * step
    if tb_logger is not None:
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=valStep)
        tb_logger.add_scalar(
            tag="val_metric", scalar_value=val_metric, global_step=valStep
        )
        # we always log the last validation images
        prediction = activation(prediction)
        tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=valStep)
        tb_logger.add_images(tag="val_target", img_tensor=y.to("cpu"), global_step=valStep)
        tb_logger.add_images(
            tag="val_prediction", img_tensor=prediction.to("cpu"), global_step=valStep
        )

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
            val_loss, val_metric
        )
    )