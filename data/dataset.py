import numpy as np
import torch
from torch.utils.data import Dataset

class SeismicSegmentationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = list(data_dir.glob("*.npz"))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        trace = data['trace']  # Shape should be (height, width)
        label = data['label']  # Ensure label is loaded correctly

        # Assuming trace and label are both of shape (height, width)
        trace = np.expand_dims(trace, axis=0)  # Shape now (1, height, width)
        label = label  # Keep label as (height, width)

        # Convert to tensor
        trace = torch.tensor(trace, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)  # For CrossEntropyLoss

        return trace, label
