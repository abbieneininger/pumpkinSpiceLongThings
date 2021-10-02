import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
