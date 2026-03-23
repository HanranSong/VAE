import argparse
import os
import json

import torch
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from models.vae import VAE
from utils.metrics import compute_latent_clustering_score
from utils.seed import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    run_name = args.run_name
    run_dir = os.path.join("results", run_name)

    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Directory not found: {run_dir}")
    
    args_file = os.path.join(run_dir, "args.json")
    with open(args_file, "r") as f:
        run_args = json.load(f)

    set_all_seeds(run_args["seed"])
    
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Loading run: {run_dir}")

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor()),
        batch_size=512, shuffle=False
    )

    model = VAE(latent_dim=run_args["latent_dim"]).to(device)
    ckpt_path = os.path.join(run_dir, "checkpoints", "model_final.pt")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)
            
            all_z.append(z.cpu())
            all_labels.append(labels.cpu())

    z_tensor = torch.cat(all_z)  # [N, latent_dim]
    labels_tensor = torch.cat(all_labels)  # [N]
    
    print(f"Total samples: {z_tensor.shape[0]}")
    print("Computing Silhouette Score...")
    sil_score = compute_latent_clustering_score(z_tensor, labels_tensor)
    print(f"Silhouette Score: {sil_score:.4f}")
    
    metrics_path = os.path.join(run_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Silhouette Score: {sil_score:.4f}\n")
    
    print("Running t-SNE...")
    tsne = TSNE(random_state=run_args["seed"])
    z_2d = tsne.fit_transform(z_tensor.numpy())

    print("Plotting...")
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_tensor.numpy(), cmap="tab10", alpha=0.7)

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc="best")
    
    prior_name = run_args["prior"]
    plt.title(f"Latent Space t-SNE ({prior_name})", fontsize=16)
    plt.xlabel("t-SNE Dim 1", fontsize=12)
    plt.ylabel("t-SNE Dim 2", fontsize=12)

    save_path = os.path.join(run_dir, "images", "latent_space_tsne.png")
    plt.savefig(save_path) 
    plt.close()
    
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
