"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from GAN.maxout import Maxout

# magic number to avoid log(0)
EPSILON = 1e-8


class Generator(nn.Module):
    """
    Implementation of the conditional generator (MNIST Mirza et al., 2014).

    Reference:
    Conditional Generative Adversarial Nets, Mirza et al. 2014;
    https://arxiv.org/abs/1411.1784
    """
    def __init__(
            self,
            z_dim: int=100,
            y_dim: int=10,
            z_hdim: int=200,
            y_hdim: int=1000,
            hidden_dim: int=1200,
            out_dim: int=784
    ) -> None:
        super().__init__()

        self.fc_z = nn.Sequential(
            nn.Linear(z_dim, z_hdim),
            nn.ReLU(True)
        )

        self.fc_y = nn.Sequential(
            nn.Linear(y_dim, y_hdim),
            nn.ReLU(True),
        ) 
        
        self.out = nn.Sequential(
            nn.Linear(z_hdim + y_hdim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h_z = self.fc_z(z)
        h_y = self.fc_y(y)
        h = torch.cat([h_z, h_y], dim=1)
        return self.out(h)
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.005, 0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """
    Implementation of the conditional discriminator (MNIST, Mirza et al., 2014).

    Reference:
    Conditional Generative Adversarial Nets, Mirza et al., 2014;
    https://arxiv.org/abs/1411.1784
    """  
    def __init__(
            self,
            x_dim: int=784,
            y_dim: int=10,
            x_hdim: int=240,
            y_hdim: int=50,
            hidden_dim: int=240,
            x_pieces: int=5,
            y_pieces: int=5,
            hidden_pieces: int=4,
            x_dropout: float=0.2,
            y_dropout: float=0.2,
            hidden_dropout: float=0.5
    ) -> None:
        super().__init__()

        self.maxout_x = nn.Sequential(
            nn.Dropout(x_dropout),
            Maxout(x_dim, x_hdim, x_pieces)
        )
        
        self.maxout_y = nn.Sequential(
            nn.Dropout(y_dropout),
            Maxout(y_dim, y_hdim, y_pieces)
        )

        self.out = nn.Sequential(
            nn.Dropout(hidden_dropout),
            Maxout(x_hdim + y_hdim, num_units=hidden_dim, num_pieces=hidden_pieces),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        hx = self.maxout_x(x)
        hy = self.maxout_y(y)
        h = torch.cat([hx, hy], dim=1)
        return self.out(h)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.005, 0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GeneratorLoss(nn.Module):
    def __init__(self, maximize: bool=False) -> None:
        super().__init__() 
        self.maximize = maximize

    def forward(self, D_G_z: torch.Tensor) -> torch.Tensor:
        if self.maximize:
            return torch.log(D_G_z + EPSILON).mean()
        return torch.log(1 - D_G_z + EPSILON).mean()


class DiscriminatorLoss(nn.Module):
    def forward(self, D_x: torch.Tensor, D_G_z: torch.Tensor) -> torch.Tensor:
        return (torch.log(D_x + EPSILON) + torch.log(1 - D_G_z + EPSILON)).mean()