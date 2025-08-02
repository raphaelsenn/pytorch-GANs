"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-18
"""

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
    def __init__(
            self, 
            z_dim: int=100,
            channels_img: int=3,
            features_g: int=128
        ) -> None:
        super().__init__() 
        
        self.projection = nn.Linear(z_dim, features_g * 8 * 4 * 4, bias=False)
        self.features_g = features_g
        self.net = nn.Sequential(
            nn.BatchNorm2d(8 * features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=8*features_g, 
                out_channels=4*features_g, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                bias=False, 
            ),
            nn.BatchNorm2d(4*features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=4*features_g, 
                out_channels=2*features_g, 
                kernel_size=4, 
                stride=2, 
                bias=False, 
                padding=1
            ),
            nn.BatchNorm2d(2*features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=2*features_g, 
                out_channels=features_g, 
                kernel_size=4, 
                stride=2, 
                bias=False, 
                padding=1
            ),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=features_g, 
                out_channels=channels_img, 
                kernel_size=4, 
                stride=2, 
                bias=False, 
                padding=1
            ),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.view(-1, self.features_g * 8, 4, 4)
        return self.net(x)

    def _initialize_weights(self) -> None:
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
    def __init__(
            self,
            channels_img: int=3,
            features_d: int=128
            ) -> None:
        super().__init__()
        self.features_d = features_d
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=channels_img, 
                out_channels=features_d, 
                kernel_size=4, 
                stride=2, 
                bias=False, 
                padding=1
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                in_channels=features_d, 
                out_channels=2*features_d, 
                kernel_size=4, 
                stride=2, 
                bias=False, 
                padding=1
            ),
            nn.BatchNorm2d(2*features_d), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                in_channels=2*features_d, 
                out_channels=4*features_d, 
                kernel_size=4, 
                stride=2, 
                bias=False, 
                padding=1
            ),
            nn.BatchNorm2d(4*features_d), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                in_channels=4*features_d, 
                out_channels=8*features_d, 
                kernel_size=4, 
                stride=2, 
                bias=False, 
                padding=1
            ),
            nn.BatchNorm2d(8*features_d),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                in_channels=8*features_d, 
                out_channels=1, 
                kernel_size=4, 
                stride=2, 
                bias=False, 
                padding=1
            ),
            nn.Flatten(start_dim=1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def _initialize_weights(self) -> None:
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
    CelebA Dataset.
    
    Reference:
    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. 
    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 
    """
    def __init__(
            self, 
            root_dir: str, 
            csv_file: str, 
            transform=None
        ) -> None:
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