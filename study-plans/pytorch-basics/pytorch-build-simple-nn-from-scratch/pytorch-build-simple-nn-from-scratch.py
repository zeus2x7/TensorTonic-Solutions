import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.l1 = nn.Linear(in_features, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
