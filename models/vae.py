import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 28 -> 14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 14 -> 7

        self.fc_enc = nn.Linear(64 * 7 * 7, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc_dec1 = nn.Linear(latent_dim, 256)
        self.fc_dec2 = nn.Linear(256, 64 * 7 * 7)

        self.conv_t1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.conv_t2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)  # 14 -> 28

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.reshape(h.size(0), -1)
        h = F.relu(self.fc_enc(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec1(z))
        h = F.relu(self.fc_dec2(h))
        h = h.view(h.size(0), 64, 7, 7)
        h = F.relu(self.conv_t1(h))
        return torch.sigmoid(self.conv_t2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
