from torch.nn import functional as F


def loss_function(recon_x, x, mu, logvar, z, prior, beta=1.0):
    batch_size = x.size(0)

    bce = F.binary_cross_entropy(recon_x, x, reduction="sum") / batch_size
    kld = prior.compute_kl(mu, logvar, z).mean()

    loss = bce + beta * kld
    return loss, bce, kld
