import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# magic number to avoid log(0)
EPSILON = 1e-8


class Generator(nn.Module):
    """
    Implementation of the CelebA-Generator.
 
    Reference:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
    https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz: int=100) -> None:
        super().__init__() 
        
        self.projection = nn.Linear(nz, 1024 * 4 * 4, bias=False)
        self.net = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, bias=False, padding=1),
            nn.Tanh()
        )
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.view(-1, 1024, 4, 4)
        return self.net(x)

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """
    Implementation of the CelebA-Discriminator.
 
    Reference:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
    https://arxiv.org/abs/1511.06434
    """ 
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, bias=False, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, bias=False, padding=1),
            nn.Flatten(start_dim=1),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GeneratorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, D_G_z: torch.Tensor) -> torch.Tensor:
        return -torch.log(D_G_z + EPSILON).mean()


class DiscriminatorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, D_x: torch.Tensor, D_G_z: torch.Tensor) -> torch.Tensor:
        return -(torch.log(D_x + EPSILON) + torch.log(1 - D_G_z + EPSILON)).mean()
    

class CelebA(Dataset):
    """
    CelebA Dataset
    
    Reference:
    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. 
    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 
    """
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