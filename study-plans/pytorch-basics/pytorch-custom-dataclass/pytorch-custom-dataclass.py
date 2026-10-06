import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, data: list, label_col: int):
        self.data = torch.tensor(data, dtype = torch.float32)
        id_range = self.data.shape[1]
        ids = torch.arange(0,id_range)
        mask = torch.ones_like(ids, dtype= torch.bool)
        mask[label_col]=False
        x_ids = ids[mask]
        self.x = self.data[:,x_ids]
        self.y = self.data[:, label_col].unsqueeze(1)
        

    def __len__(self) -> int:
        """
        Returns the number of rows.
        """
        return self.data.shape[0]
        

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (features, label) as float32 tensors of shapes (D,) and (1,).
        """
        return (self.x[idx], self.y[idx])
        
