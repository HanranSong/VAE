from torch.nn import functional as F


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, z, prior):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    KLD = prior.compute_kl(mu, logvar, z)

    return BCE + KLD, BCE, KLD
