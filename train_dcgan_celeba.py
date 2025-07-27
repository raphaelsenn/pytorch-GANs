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
    if torch.cuda.is_available(): return torch.device('cuda')
    elif torch.backends.mps.is_available(): return torch.device('mps')
    else: return torch.device('cpu')


def set_seed(seed: int=42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)


def load_celeba(root: str='./celeba/data/', batch_size: int=256, subset_size: int=60000) -> tuple[DataLoader, DataLoader]:
    """
    Loads CelebA dataset and returns it as a dataloader. 
    """ 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(64, 64)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    celeb_faces = CelebA(
        root_dir=root,
        csv_file='./celeba/landmarks.csv',
        transform=transform
    )

    idx = torch.arange(subset_size)
    celeb_faces = Subset(celeb_faces, idx)

    dataloader = DataLoader(celeb_faces, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    device = initialize_device()
    set_seed(seed=42)

    # Create generator and discriminator
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Hyperparameters
    epochs = 50
    batch_size = 128
    lr_G = 0.0002
    betas_G = (0.5, 0.99)
    lr_D = 0.0002
    betas_D = (0.5, 0.99)
    z_noise_dim = 100 
    verbose = True

    # Create the optimizer and the loss
    criterion_G = GeneratorLoss()
    optimizer_G = torch.optim.Adam(params=G.parameters(), lr=lr_G, betas=betas_G)
    criterion_D = DiscriminatorLoss()
    optimizer_D = torch.optim.Adam(params=D.parameters(), lr=lr_D, betas=betas_D)

    # Load the dataset
    IMAGE_SHAPE = (3, 64, 64)
    dataloader = load_celeba(batch_size=batch_size)

    # Create the optimizer and the loss
    criterion_G = GeneratorLoss()
    optimizer_G = torch.optim.Adam(params=G.parameters(), lr=lr_G, betas=betas_G)
    criterion_D = DiscriminatorLoss()
    optimizer_D = torch.optim.Adam(params=D.parameters(), lr=lr_D, betas=betas_D)

    losses_D, losses_G = [], []
    N = len(dataloader.dataset)
    for epoch in range(epochs):
        G.train(); D.train()
        total_loss_G, total_loss_D = 0.0, 0.0
        for x in dataloader:
            # x is sampled from data generating distribution x ~ p_data
            x = x.to(device)

            # z is sampled from noise prior z ~ p_noise
            z = torch.distributions.uniform.Uniform(low=-1, high=1).sample([x.shape[0], z_noise_dim]).to(device)
            D_x = D(x)
            D_G_z = D(G(z))

            # update descriminator by ascending its stochastic gradient
            optimizer_D.zero_grad()
            loss_d = criterion_D(D_x, D_G_z)
            loss_d.backward()
            optimizer_D.step()
            
            # z is sampled from noise prior
            # z ~ p_noise
            z = torch.distributions.uniform.Uniform(low=-1, high=1).sample([x.shape[0], z_noise_dim]).to(device)
            D_G_z = D(G(z))

            # update generator by descending its stochastic gradient
            optimizer_G.zero_grad()
            loss_g = criterion_G(D_G_z)
            loss_g.backward()
            optimizer_G.step()

            total_loss_D += loss_d.item() * x.shape[0]
            total_loss_G += loss_g.item() * z.shape[0]
        losses_D.append(total_loss_D / N)
        losses_G.append(total_loss_G / N)

        if verbose: 
            print(f'epoch: {epoch} loss generator: {(total_loss_G/N):.4f} loss discriminator: {(total_loss_D/N):.4f}')