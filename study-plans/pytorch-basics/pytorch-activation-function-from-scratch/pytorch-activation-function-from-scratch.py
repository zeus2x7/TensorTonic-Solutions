import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if method == 'relu':
        act = torch.clamp(x,0 )
    if method == "sigmoid":
        act = 1/(1+torch.exp(-x))
    if method == "tanh":
        act = torch.where(
            x >= 0,
            (1 - torch.exp(-2*x)) / (1 + torch.exp(-2*x)),
            (torch.exp(2*x) - 1) / (torch.exp(2*x) + 1)
        )
    if method == "leaky_relu":
        act = torch.clamp(x, 0.01 *x)
    return act
