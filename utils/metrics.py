from sklearn.metrics import silhouette_score


def compute_latent_clustering_score(z_tensor, labels_tensor):
    z_np = z_tensor.cpu().numpy()
    labels_np = labels_tensor.cpu().numpy()
    
    score = silhouette_score(z_np, labels_np)
    return score
