import torch
import torch.nn as nn

from GAN.maxout import Maxout

# magic number to avoid log(0)
EPSILON = 1e-8


class Generator(nn.Module):
    """
    Implementation of the CIFAR10-Generator of the vanilla gan paper.

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014; https://arxiv.org/abs/1406.2661
    """  
    def __init__(self, input_dim: int=100, hidden_dim: int=8000, output_dim: int=3072) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.shape[0] == 3072:
                    nn.init.uniform_(m.weight, -0.005, 0.005)
                else: nn.init.uniform_(m.weight, -0.005, 0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """
    Implementation of the CIFAR10-Discriminator of the vanilla gan paper.

    Reference:
    Generative Adversarial Networks, Goodfellow et al. 2014; https://arxiv.org/abs/1406.2661
    """   
    def __init__(self, input_dim: int=3072, hidden_dim: int=1600, output_dim:int=1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2), 
            Maxout(input_dim, hidden_dim, num_pieces=5),
            nn.Dropout(0.5),
            Maxout(hidden_dim, hidden_dim, num_pieces=5),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def initialize_weights(self) -> None:
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