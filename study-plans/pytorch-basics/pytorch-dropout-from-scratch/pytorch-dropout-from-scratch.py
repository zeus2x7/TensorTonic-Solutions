import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p =p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """
        if not self.training or self.p ==0:
            return x 
        elif self.p ==1:
            return x * 0

        else:
            mask = torch.bernoulli(torch.full_like(x, 1-self.p))
            y = x * mask /(1-self.p)
            return y
