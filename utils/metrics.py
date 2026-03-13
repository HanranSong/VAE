import math
import torch
from sklearn.metrics import silhouette_score

from models.priors import gaussian_diag_logprob


def compute_latent_clustering_score(z_tensor, labels_tensor):
    z_np = z_tensor.cpu().numpy()
    labels_np = labels_tensor.cpu().numpy()
    score = silhouette_score(z_np, labels_np)
    return score


def bernoulli_log_prob_from_probs(x, probs, eps=1e-9):
    probs = probs.clamp(min=eps, max=1.0 - eps)
    log_p = x * torch.log(probs) + (1.0 - x) * torch.log(1.0 - probs)

    if log_p.dim() == 5:
        return log_p.flatten(start_dim=2).sum(dim=2)  # [K, B]
    elif log_p.dim() == 4:
        return log_p.flatten(start_dim=1).sum(dim=1)  # [B]
    else:
        raise ValueError(f"Unexpected tensor shape: {log_p.shape}")


def logmeanexp(value, dim=0):
    return torch.logsumexp(value, dim=dim) - math.log(value.size(dim))


@torch.no_grad()
def estimate_batch_log_likelihood_is(model, x, prior, num_importance_samples=5000, chunk_size=100):
    mu, logvar = model.encode(x)  # [B, D], [B, D]
    std = torch.exp(0.5 * logvar)  # [B, D]

    B, D = mu.shape
    log_weights_chunks = []

    for start in range(0, num_importance_samples, chunk_size):
        k = min(chunk_size, num_importance_samples - start)

        eps = torch.randn(k, B, D, device=x.device)
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # [k, B, D]

        z_flat = z.reshape(k * B, D)
        recon_flat = model.decode(z_flat)  # [k*B, 1, 28, 28]
        recon = recon_flat.view(k, B, *x.shape[1:])  # [k, B, 1, 28, 28]

        x_expand = x.unsqueeze(0).expand(k, -1, -1, -1, -1)  # [k, B, 1, 28, 28]

        log_p_x_given_z = bernoulli_log_prob_from_probs(x_expand, recon)  # [k, B]
        log_p_z = prior.log_prob(z_flat).view(k, B)  # [k, B]

        log_q_z_given_x = gaussian_diag_logprob(
            z,
            mu.unsqueeze(0),
            logvar.unsqueeze(0),
        ).sum(dim=2)  # [k, B]

        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x  # [k, B]
        log_weights_chunks.append(log_w)

    log_weights = torch.cat(log_weights_chunks, dim=0)  # [K, B]
    log_p_x = logmeanexp(log_weights, dim=0)  # [B]

    return log_p_x


@torch.no_grad()
def estimate_dataset_log_likelihood_is(model, data_loader, prior, device, num_importance_samples=5000, chunk_size=100):
    model.eval()

    total_log_likelihood = 0.0
    total_num_samples = 0

    for x, _ in data_loader:
        x = x.to(device)
        log_p_x = estimate_batch_log_likelihood_is(
            model=model,
            x=x,
            prior=prior,
            num_importance_samples=num_importance_samples,
            chunk_size=chunk_size,
        )

        total_log_likelihood += log_p_x.sum().item()
        total_num_samples += x.size(0)

    return total_log_likelihood / total_num_samples
