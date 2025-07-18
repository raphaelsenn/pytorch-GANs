import torch
import torch.nn as nn

# magic number to avoid log(0)
EPSILON = 1e-8


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        
        self.projection = nn.Linear(100, 1024 * 4 * 4, bias=False)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=2, stride=2, bias=False),
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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=5, stride=2, bias=False),
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