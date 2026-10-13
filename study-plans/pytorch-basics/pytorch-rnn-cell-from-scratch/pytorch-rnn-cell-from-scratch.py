import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_ih = nn.Parameter(torch.randn((hidden_size,input_size), dtype = torch.float32))
        self.b_ih = nn.Parameter(torch.randn((hidden_size),dtype= torch.float32))
        self.W_hh = nn.Parameter(torch.randn((hidden_size,hidden_size), dtype = torch.float32))
        self.b_hh = nn.Parameter(torch.randn(hidden_size,dtype = torch.float32))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 hidden-state tensor of shape (batch, hidden_size).
        """
        out = nn.functional.tanh(x@self.W_ih.t() + self.b_ih + h_prev@self.W_hh.t() +self.b_hh)
        return out.float()
