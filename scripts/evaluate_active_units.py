import argparse
import json
import os

import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vae import VAE
from models.priors import build_prior
from utils.seed import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    run_dir = os.path.join("results", args.run_name)
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Directory not found: {run_dir}")

    args_file = os.path.join(run_dir, "args.json")
    with open(args_file, "r") as f:
        run_args = json.load(f)
    
    set_all_seeds(run_args["seed"])

    use_accel = torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"Loading run: {run_dir} (Prior: {run_args.get('prior', 'unknown')})")

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor()),
        batch_size=512, shuffle=False
    )

    model = VAE(latent_dim=run_args["latent_dim"]).to(device)
    prior = build_prior(
        run_args["prior"], 
        latent_dim=run_args["latent_dim"],
        num_components=run_args.get("num_components", 1)
    ).to(device)

    ckpt_path = os.path.join(run_dir, "checkpoints", "model_final.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if "prior_state_dict" in checkpoint:
        prior.load_state_dict(checkpoint["prior_state_dict"])

    model.eval()
    
    all_mus = []

    print("Extracting posterior means (mu) over the test set...")
    with torch.no_grad():
        for real_imgs, _ in tqdm(test_loader):
            real_imgs = real_imgs.to(device)
            
            mu, _ = model.encode(real_imgs)
            
            all_mus.append(mu.cpu())

    all_mus = torch.cat(all_mus, dim=0)
    
    dim_variances = torch.var(all_mus, dim=0)
    
    active_mask = dim_variances > args.threshold
    active_units_count = active_mask.sum().item()
    total_units = run_args["latent_dim"]
    active_percentage = (active_units_count / total_units) * 100

    print("\nActive Units Evaluation:")
    print(f"Variance Threshold: {args.threshold}")
    print(f"Active Units:       {active_units_count} / {total_units}")
    print(f"Active Percentage:  {active_percentage:.2f}%")
    
    if args.detail:
        print("\nVariances per dimension:")
        for i, var in enumerate(dim_variances.numpy()):
            status = "ACTIVE" if var > args.threshold else "DEAD"
            print(f"Dim {i:02d}: {var:.5f} [{status}]")

    save_path = os.path.join(run_dir, "active_units.json")
    with open(save_path, "w") as f:
        json.dump({
            "run_name": args.run_name,
            "prior": run_args["prior"],
            "latent_dim": total_units,
            "threshold": args.threshold,
            "active_units_count": active_units_count,
            "active_percentage": active_percentage,
            "dim_variances": dim_variances.numpy().tolist()
        }, f, indent=4)
    
    print(f"\nSaved Active Units results to: {save_path}")


if __name__ == "__main__":
    main()
