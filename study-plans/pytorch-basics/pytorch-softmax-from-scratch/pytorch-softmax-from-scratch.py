import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """
    logits = logits - torch.max(logits, dim =-1, keepdim=True).values
    return torch.exp(logits)/torch.sum(torch.exp(logits), dim =1, keepdim= True)
    pass
