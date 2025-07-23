import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from CGAN.cgan_mnist_fc import (
    Generator,
    Discriminator,
    GeneratorLoss,
    DiscriminatorLoss
)


def initialize_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device('cuda')
    elif torch.backends.mps.is_available(): return torch.device('mps')
    else: return torch.device('cpu')


def set_seed(seed: int=42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)


def load_mnist(root: str='./mnist/', batch_size: int=256) -> tuple[DataLoader, DataLoader]:
    """
    Loads MNIST dataset and returns it as a dataloader. 
    """ 
    transform = transforms.Compose(
        [transforms.ToTensor(),                     # greyscale [0, 255] -> [0, 1]
        transforms.Lambda(lambda x: x.view(-1))])   # shape [1, 28, 28] -> [1, 784]

    mnist_train = MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform)
        
    dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    device = initialize_device()
    set_seed(seed=42)

    # Create generator and discriminator
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Hyperparameters
    epochs = 100
    batch_size = 100
    lr_G = 0.002
    betas_G = (0.5, 0.99)
    lr_D = 0.002
    betas_D = (0.5, 0.99)
    verbose = True

    # Create the optimizer and the loss
    criterion_G = GeneratorLoss(maximize=True)
    optimizer_G = torch.optim.Adam(params=G.parameters(), lr=lr_G, betas=betas_G, maximize=True)
    criterion_D = DiscriminatorLoss()
    optimizer_D = torch.optim.Adam(params=D.parameters(), lr=lr_D, betas=betas_D, maximize=True)

    # Load the dataset
    dataloader = load_mnist()
    NUM_CLASSES = 10

    losses_D, losses_G = [], []
    N = len(dataloader.dataset)
    for epoch in range(epochs):
        G.train(); D.train()
        total_loss_G, total_loss_D = 0.0, 0.0
        for x, y in dataloader:
            # x is sampled from data generating distribution x ~ p_data
            x, y = x.to(device), F.one_hot(y.to(device), num_classes=NUM_CLASSES).float()

            # z is sampled from noise prior z ~ p_noise
            z = torch.distributions.uniform.Uniform(low=-1, high=1).sample([x.shape[0], 100]).to(device)
            D_x = D(x, y)
            D_G_z = D(G(z, y), y)

            # update descriminator by ascending its stochastic gradient
            optimizer_D.zero_grad()
            loss_d = criterion_D(D_x, D_G_z)
            loss_d.backward()
            optimizer_D.step()
            
            # z is sampled from noise prior
            # z ~ p_noise
            z = torch.distributions.uniform.Uniform(low=-1, high=1).sample([x.shape[0], 100]).to(device)
            D_G_z = D(G(z, y), y)

            # update generator by ascending its stochastic gradient
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