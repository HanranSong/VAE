import argparse
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--no-accel', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log-interval', type=int, default=10)
args = parser.parse_args()

use_accel = not args.no_accel and torch.accelerator.is_available()

torch.manual_seed(args.seed)


if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def train(epoch):
    model.train()
    train_loss, train_bce, train_kld = 0, 0, 0
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        optimizer.step()
    dataset_size = len(train_loader.dataset)
    return train_loss / dataset_size, train_bce / dataset_size, train_kld / dataset_size


def test(epoch):
    model.eval()
    test_loss, test_bce, test_kld = 0, 0, 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            test_bce += bce.item()
            test_kld += kld.item()
            if i == 0 and epoch % args.log_interval == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    dataset_size = len(test_loader.dataset)        
    return test_loss / dataset_size, test_bce / dataset_size, test_kld / dataset_size

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)

    pbar = tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        train_loss, train_bce, train_kld = train(epoch)
        test_loss, test_bce, test_kld = test(epoch)

        if epoch % args.log_interval == 0:
            with torch.no_grad():
                sample = torch.randn(64, 20).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28),
                        'results/sample_' + str(epoch) + '.png')
                
        pbar.set_postfix({
            "Train Loss": f"{train_loss:.2f}",
            "BCE": f"{train_bce:.2f}",
            "KLD": f"{train_kld:.2f}",
            "Test Loss": f"{test_loss:.2f}"
        })
