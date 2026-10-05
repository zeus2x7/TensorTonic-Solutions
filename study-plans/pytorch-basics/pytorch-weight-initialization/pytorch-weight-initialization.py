import torch
import math

def initialize_weights(fan_in: int, fan_out: int, method: str) -> torch.Tensor:
    """
    Returns a float32 weight tensor with shape (fan_out, fan_in).
    """
    if method == "xavier_uniform":
        weights = torch.zeros((fan_out, fan_in))
        rng = math.sqrt(6/(fan_in+fan_out))
        weights = weights.uniform_(-rng, rng)
        return weights
    elif method == "he_uniform":
        weights = torch.zeros((fan_out, fan_in))
        rng = math.sqrt(6/(fan_in))
        weights = weights.uniform_(-rng, rng)
        return weights
    elif method == "xavier_normal":
        weights = torch.zeros((fan_out, fan_in), dtype=torch.float32)
        std_dev = math.sqrt(2/(fan_in+fan_out))
        weights = weights.normal_(0,std_dev)
        return weights
    elif method =="he_normal":
        weights = torch.zeros((fan_out, fan_in), dtype=torch.float32)
        std_dev = math.sqrt(2/(fan_in))
        weights = weights.normal_(0,std_dev)
        return weights
    pass
