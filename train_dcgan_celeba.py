"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-18
"""

import torch
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import transforms

from DCGAN.dcan import (
    Generator,
    Discriminator,
    GeneratorLoss,
    DiscriminatorLoss,
    CelebA
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


def load_celeba(
        root: str='./celeba/data/',
        csv_file: str='./celeba/landmarks.csv',
        batch_size: int=128, 
        subset_size: None | int=None
    ) -> tuple[DataLoader, DataLoader]:
    """
    Loads CelebA dataset and return as a dataloader.
    """ 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(64, 64)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    celebA = CelebA(
        root_dir=root,
        csv_file=csv_file,
        transform=transform
    )

    idx = torch.arange(subset_size)
    celebA = Subset(celebA, idx)

    dataloader = DataLoader(celebA, batch_size=batch_size, shuffle=True)
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
    IMAGE_SHAPE = (3, 64, 64)
    SUBSET_SIZE = 60000
    
    EPOCHS = 50
    BATCH_SIZE = 128
    LEARNING_RATE_G = 0.0002
    LEARNING_RATE_D = 0.0002
    BETAS_G = (0.5, 0.99)
    BETAS_D = (0.5, 0.99)
    VERBOSE = True

    # Create the optimizer and the loss
    criterion_G = GeneratorLoss()
    optimizer_G = torch.optim.Adam(params=G.parameters(), lr=LEARNING_RATE_G, betas=BETAS_G)
    criterion_D = DiscriminatorLoss()
    optimizer_D = torch.optim.Adam(params=D.parameters(), lr=LEARNING_RATE_D, betas=BETAS_D)

    # Load the dataset
    dataloader = load_celeba(batch_size=BATCH_SIZE, subset_size=SUBSET_SIZE)

    # Start training
    if VERBOSE:
        print(f'Start training on device: {device}')

    losses_D, losses_G = [], []
    N = len(dataloader.dataset)
    for epoch in range(EPOCHS):
        G.train(); D.train()
        total_loss_G, total_loss_D = 0.0, 0.0
        for x in dataloader:
            N_batch = x.shape[0]
            
            # x is sampled from data generating distribution x ~ p_data
            x = x.to(device)

            # z is sampled from noise prior z ~ p_noise
            z = torch.distributions.uniform.Uniform(
                low=Z_NOISE_LOW, high=Z_NOISE_HIGH
            ).sample([N_batch, Z_NOISE_DIM]).to(device)
            D_x = D(x)
            D_G_z = D(G(z))

            # update descriminator
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

            # update generator
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