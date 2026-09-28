import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # X shape (N, D)
    mean = X.mean(dim=0, keepdim = True) # shape (1, D)
    std_div = X.std(dim =0, keepdim = True, unbiased = False) # shape (1, D) 
    y = gamma*((X-mean)/(torch.sqrt(std_div**2+ eps))) + beta
    return y