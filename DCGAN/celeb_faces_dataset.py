import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


class CelebFacesDataset(Dataset):
    """CelebFaces Dataset"""
    def __init__(self, csv_file: str, root_dir: str, transform=None) -> None:
        self.landmarks = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.landmarks)

    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks.iloc[index, 0])
        image = plt.imread(img_name)

        if self.transform:
            image = self.transform(image.copy())
        return image