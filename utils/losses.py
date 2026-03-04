from torch.nn import functional as F


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, z, prior, beta=1.0):
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = prior.compute_kl(mu, logvar, z)
    loss = bce + beta * kld

    return loss, bce, kld
