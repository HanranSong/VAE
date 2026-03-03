import torch
from torch import nn


class StandardGaussianPrior(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, batch_size, latent_dim, device):
        return torch.randn(batch_size, latent_dim, device=device)

    def compute_kl(self, mu, logvar, z=None):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
