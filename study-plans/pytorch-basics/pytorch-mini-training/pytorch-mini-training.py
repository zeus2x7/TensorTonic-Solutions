import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    model.train()
    losses = []
    for x, y in dataloader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        ret = loss.detach()
        losses.append(ret)
    rete = torch.mean(torch.tensor(losses))
        
    return rete.item()
