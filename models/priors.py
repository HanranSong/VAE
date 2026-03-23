import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Laplace, StudentT


LOG2PI = math.log(2 * math.pi)


def gaussian_diag_logprob(z, mu, logvar):
    return -0.5 * (LOG2PI + logvar + (z - mu).pow(2) / logvar.exp())


class GaussianPrior(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def sample(self, batch_size, device):
        return torch.randn(batch_size, self.latent_dim, device=device)
    
    def log_prob(self, z):
        zeros = torch.zeros_like(z)
        logvar = torch.zeros_like(z)
        return gaussian_diag_logprob(z, zeros, logvar).sum(dim=1)

    def compute_kl(self, mu, logvar, z=None):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [B]


class MoGPrior(nn.Module):
    def __init__(self, latent_dim, num_components=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components

        self.means = nn.Parameter(torch.randn(num_components, latent_dim))
        self.logvars = nn.Parameter(torch.zeros(num_components, latent_dim))
        self.weight_logits = nn.Parameter(torch.zeros(num_components))

    def sample(self, batch_size, device):
        weights = F.softmax(self.weight_logits, dim=0)
        indices = torch.multinomial(weights, batch_size, replacement=True)
        
        sampled_means = self.means[indices]
        sampled_logvars = self.logvars[indices]
        
        std = torch.exp(0.5 * sampled_logvars)
        eps = torch.randn(batch_size, self.latent_dim, device=device)
        return sampled_means + eps * std

    def log_prob(self, z):
        z_expanded = z.unsqueeze(1)
        means_expanded = self.means.unsqueeze(0)
        logvars_expanded = self.logvars.unsqueeze(0)
        
        log_p_components = gaussian_diag_logprob(z_expanded, means_expanded, logvars_expanded).sum(dim=2)
        
        log_weights = F.log_softmax(self.weight_logits, dim=0)
        log_p_z = torch.logsumexp(log_weights.unsqueeze(0) + log_p_components, dim=1)
        return log_p_z
    
    def compute_kl(self, mu, logvar, z):
        log_q_z = gaussian_diag_logprob(z, mu, logvar).sum(dim=1)  # [B]
        log_p_z = self.log_prob(z)  # [B]
        return log_q_z - log_p_z  # [B]


class LaplacePrior(nn.Module):
    def __init__(self, latent_dim, scale=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = scale

    def sample(self, batch_size, device):
        loc_t = torch.tensor(0.0, device=device)
        scale_t = torch.tensor(self.scale, device=device)
        m = Laplace(loc_t, scale_t)
        return m.sample((batch_size, self.latent_dim))

    def log_prob(self, z):
        loc_t = torch.tensor(0.0, device=z.device)
        scale_t = torch.tensor(self.scale, device=z.device)
        m = Laplace(loc_t, scale_t)
        return m.log_prob(z).sum(dim=1)

    def compute_kl(self, mu, logvar, z):
        log_q_z = gaussian_diag_logprob(z, mu, logvar).sum(dim=1)  # [B]
        log_p_z = self.log_prob(z)  # [B]
        return log_q_z - log_p_z  # [B]


class StudentTPrior(nn.Module):
    def __init__(self, latent_dim, df=3.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.df = df  # Degrees of freedom

    def sample(self, batch_size, device):
        df_t = torch.tensor(self.df, device=device)
        m = StudentT(df_t)
        return m.sample((batch_size, self.latent_dim))

    def log_prob(self, z):
        df_t = torch.tensor(self.df, device=z.device)
        m = StudentT(df_t)
        return m.log_prob(z).sum(dim=1)

    def compute_kl(self, mu, logvar, z):
        log_q_z = gaussian_diag_logprob(z, mu, logvar).sum(dim=1)  # [B]
        log_p_z = self.log_prob(z)  # [B]
        return log_q_z - log_p_z  # [B]


def build_prior(name, latent_dim=16, num_components=10, df=3.0):
    if name == "gaussian":
        return GaussianPrior(latent_dim=latent_dim)
    elif name == "mog":
        return MoGPrior(latent_dim=latent_dim, num_components=num_components)
    elif name == "laplace":
        return LaplacePrior(latent_dim=latent_dim)
    elif name == "student-t":
        return StudentTPrior(latent_dim=latent_dim, df=df)
    else:
        raise ValueError(f"Unknown prior: {name}")
