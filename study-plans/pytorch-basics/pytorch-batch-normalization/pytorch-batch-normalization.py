import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    mean = X.mean(dim=0, keepdim=True)
    var = X.var(dim=0, keepdim=True, unbiased=False)
    Y = gamma*(X - mean)*((var + eps)**-0.5) + beta
    return Y
