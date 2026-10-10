import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    x = x.to(torch.float32)
    if method == 'relu':
        return torch.clamp(x, min=0)
    elif method == "sigmoid":
        e = torch.exp(-x)
        return (1/ ( 1 + e))
    elif method == "tanh":
        e = torch.exp(-2 * x)
        return (2/ (1 + e)) - 1
    elif method == "leaky_relu":
        return torch.where(x>0, x, 0.01*x)
