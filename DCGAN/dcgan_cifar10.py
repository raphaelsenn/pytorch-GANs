import torch
import torch.nn as nn

# magic number to avoid log(0)
EPSILON = 1e-8


class Generator(nn.Module):
    """
    Implementation of the CIFAR10-Generator.
 
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
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, bias=False, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, bias=False, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, bias=False, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, bias=False, padding=3),
            nn.Tanh()
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.view(-1, 1024, 4, 4)
        return self.net(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.02)



class Discriminator(nn.Module):
    """
    Implementation of the CIFAR10-Discriminator.
 
    Reference:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, Radford et al., 2016;
    https://arxiv.org/abs/1511.06434
    """  
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, bias=False, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=False, padding=2),
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=False, padding=2),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, bias=False, padding=2),
            nn.BatchNorm2d(1024), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=2, bias=False, padding=1),
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