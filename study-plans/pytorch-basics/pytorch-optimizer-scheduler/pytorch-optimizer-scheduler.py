import torch
import torch.nn as nn

def train_with_scheduler(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR, num_epochs: int) -> dict:
    """
    Returns losses and lrs as lists of Python floats in a dictionary.
    """
    losses  = []
    lrs= []
    #schedule = scheduler(optimizer)
    model.train()
    for i in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()
            num_batches +=1
            
        losses.append(epoch_loss/num_batches)
        lrs.append(scheduler.get_last_lr()[0])

        scheduler.step()
        
    return {"losses": losses, "lrs": lrs}
