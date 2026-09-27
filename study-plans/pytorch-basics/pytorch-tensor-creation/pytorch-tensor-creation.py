import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    """
    Returns a float32 tensor with the requested shape.
    """
    assert method in ["ones", "zeros" , "full"]
    if method == "ones":
        tens = torch.ones(shape, dtype = torch.float32)
    elif method == "zeros":
       tens = torch.zeros(shape, dtype =torch.float32) 
    elif method == "full":
        tens = torch.ones(shape, dtype =torch.float32  ) *value
    return tens 
