import torch

class TransformPipeline:
    def __init__(self, mean: list, std: list):
        self.mean = torch.tensor(mean).squeeze()
        self.std = torch.tensor(std).squeeze()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Returns a normalized float32 tensor with shape (C, H, W).
        """
        image = image /255.0
        
        normalizedinput = (image -  self.mean)/self.std
        transformed_input = torch.permute(normalizedinput, (2,0,1))
        return transformed_input
        
