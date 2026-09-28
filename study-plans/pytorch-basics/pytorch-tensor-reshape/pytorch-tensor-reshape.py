import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """
    if op == 'flatten':
        out = x.flatten()
    elif op == "squeeze":
        out = x.squeeze()
    elif op == "transpose":
        out = x.t()
    return out
        
