"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-14
"""

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from GAN.gan_mnist_fc import (
    Generator,
    Discriminator,
    GeneratorLoss,
    DiscriminatorLoss
)


def initialize_device() -> torch.device:
    if torch.cuda.is_available(): 
        return torch.device('cuda')
    elif torch.backends.mps.is_available(): 
        return torch.device('mps')
    else: 
        return torch.device('cpu')


def set_seed(seed: int=42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)


def load_mnist(
        root: str='./mnist/', 
        batch_size: int=100,
        download: bool=True
    ) -> tuple[DataLoader, DataLoader]:
    """
    Loads MNIST dataset.
    """ 
    transform = transforms.Compose(
        [transforms.ToTensor(),                     # greyscale [0, 255] -> [0, 1]
        transforms.Lambda(lambda x: x.view(-1))     # shape [1, 28, 28] -> [1, 784]

    ])

    mnist = MNIST(
        root=root,
        train=True,
        download=download,
        transform=transform
    )
        
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    device = initialize_device()
    set_seed(seed=42)

    # Create generator and discriminator
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Settings/Hyperparameters
    Z_NOISE_DIM = 100 
    Z_NOISE_LOW = -1
    Z_NOISE_HIGH = 1

    EPOCHS = 100
    BATCH_SIZE = 100
    LEARNING_RATE_G = 0.002
    LEARNING_RATE_D = 0.002
    BETAS_G = (0.5, 0.99)
    BETAS_D = (0.5, 0.99)
    VERBOSE = True

    # Create the optimizer and the loss
    criterion_G = GeneratorLoss(maximize=True)
    optimizer_G = torch.optim.Adam(
        params=G.parameters(), 
        lr=LEARNING_RATE_G, 
        betas=BETAS_G, 
        maximize=True
    )
    
    criterion_D = DiscriminatorLoss()
    optimizer_D = torch.optim.Adam(
        params=D.parameters(),
        lr=LEARNING_RATE_D,
        betas=BETAS_D,
        maximize=True
    )

    # Load the dataset
    dataloader = load_mnist()

    # Start training
    if VERBOSE:
        print(f'Start training on device: {device}')

    losses_D, losses_G = [], []
    N = len(dataloader.dataset)
    for epoch in range(EPOCHS):

        total_loss_G, total_loss_D = 0.0, 0.0
        for x, _ in dataloader:
            N_batch = x.shape[0] 
            
            # x is sampled from data generating distribution x ~ p_data
            x = x.to(device)

            # z is sampled from noise prior z ~ p_noise
            z = torch.distributions.uniform.Uniform(
                low=Z_NOISE_LOW, high=Z_NOISE_HIGH
            ).sample([N_batch, Z_NOISE_DIM]).to(device)
            D_x = D(x)
            D_G_z = D(G(z))

            # update descriminator by ascending its stochastic gradient
            optimizer_D.zero_grad()
            loss_d = criterion_D(D_x, D_G_z)
            loss_d.backward()
            optimizer_D.step()
            
            # z is sampled from noise prior
            # z ~ p_noise
            z = torch.distributions.uniform.Uniform(
                low=Z_NOISE_LOW, high=Z_NOISE_HIGH
            ).sample([N_batch, Z_NOISE_DIM]).to(device)
            D_G_z = D(G(z))

            # update generator by ascending its stochastic gradient
            optimizer_G.zero_grad()
            loss_g = criterion_G(D_G_z)
            loss_g.backward()
            optimizer_G.step()

            total_loss_D += loss_d.item() * N_batch
            total_loss_G += loss_g.item() * N_batch
        losses_D.append(total_loss_D / N)
        losses_G.append(total_loss_G / N)

        if VERBOSE: 
            print(f'epoch: {epoch} loss generator: {(total_loss_G/N):.4f} loss discriminator: {(total_loss_D/N):.4f}')