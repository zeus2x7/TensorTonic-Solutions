import torch
import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    min_val_loss = float("inf")
    tries =patience
    train_loss_list, val_loss_list  = [],[]
    for i in range(max_epochs):
        train_losses = []
        val_losses = []
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            out = model(x)
            lossb = criterion(out, y)
            lossb.backward()
            optimizer.step()
            train_losses.append(lossb.item() )
        with torch.no_grad():
            for x,y in val_loader:
                model.eval()
                out = model(x)
                lossb = criterion(out,y)
                val_losses.append(lossb.item())
        train_loss = torch.tensor(train_losses, dtype=torch.float64).mean()
        val_loss = torch.tensor(val_losses, dtype=torch.float64).mean()
        train_loss_list.append(train_loss.item())
        val_loss_list.append(val_loss.item())
        if val_loss >=min_val_loss :
            tries -=1
        else:
            tries = patience
            min_val_loss = val_loss
        if tries ==0:
            return {"train_losses": train_loss_list,  "val_losses": val_loss_list,  "stopped_epoch": i +1}
        
      
        
        

    return {"train_losses": train_loss_list,  "val_losses": val_loss_list,  "stopped_epoch": max_epochs}
        
            
            
    
