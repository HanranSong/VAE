import math
import torch
from torch import nn


LOG2PI = math.log(2 * math.pi)


def gaussian_diag_logprob(z, mu, logvar):
    return -0.5 * (LOG2PI + logvar + (z - mu).pow(2) / logvar.exp())


class StandardGaussianPrior(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, batch_size, latent_dim, device):
        return torch.randn(batch_size, latent_dim, device=device)

    def log_prob(self, z):
        zeros = torch.zeros_like(z)
        logvar = torch.zeros_like(z)
        return gaussian_diag_logprob(z, zeros, logvar).sum(dim=1)

    def compute_kl(self, mu, logvar, z=None):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def build_prior(name):
    if name == "standard_gaussian":
        return StandardGaussianPrior()
    else:
        raise ValueError(f"Unknown prior: {name}")
