import torch
import torch.nn as nn
import torch.nn.functional as F

from GAN.maxout import Maxout

# magic number to avoid log(0)
EPSILON = 1e-8


class Generator(nn.Module):
    """
    Implementation of the conditional MNIST-Generator.

    Reference:
    Conditional Generative Adversarial Nets, Mirza et al. 2014;
    https://arxiv.org/abs/1411.1784
    """
    def __init__(
            self,
            nz: int=100,
            ny: int=10,
            hz: int=200,
            hy: int=1000,
            h2: int=1200,
            nout: int=784
    ) -> None:
        super().__init__()

        self.dropout_z = nn.Dropout(0.2) 
        self.fc_z = nn.Linear(nz, hz)

        self.fc_y = nn.Linear(ny, hy)
        self.dropout_y = nn.Dropout(0.2) 
        
        self.out = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(hz + hy, h2),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(h2, nout),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h_z = F.relu(self.fc_z(z))              # [N, hz]
        h_y = F.relu(self.fc_y(y))              # [N, hy]
        h = torch.cat([h_z, h_y], dim=1)        # [N, hz + hy]
        return self.out(h)
    
    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.005, 0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    """
    Implementation of the conditional MNIST-Discriminator.

    Reference:
    Conditional Generative Adversarial Nets, Mirza et al. 2014;
    https://arxiv.org/abs/1411.1784
    """  
    def __init__(
            self,
            nx: int=784,
            ny: int=10,
            hx: int=240,
            hy: int=50,
            h2: int=240
    ) -> None:
        super().__init__()
        
        self.dropout_x = nn.Dropout(0.2) 
        self.maxout_x = Maxout(nx, num_units=hx, num_pieces=5)
        
        self.dropout_y = nn.Dropout(0.2) 
        self.maxout_y = Maxout(ny, num_units=hy, num_pieces=5)
        
        self.out = nn.Sequential(
            nn.Dropout(0.5),
            Maxout(hx + hy, num_units=h2, num_pieces=4),
            nn.Dropout(0.5),
            nn.Linear(h2, 1),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        hx = self.maxout_x(self.dropout_x(x))
        hy = self.maxout_y(self.dropout_y(y))
        h = torch.cat([hx, hy], dim=1)
        return self.out(h)

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