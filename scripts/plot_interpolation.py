import argparse
import json
import os

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models.vae import VAE
from utils.seed import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=8)
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
    print(f"Generating interpolation for run: {args.run_name}")

    model = VAE(latent_dim=run_args["latent_dim"]).to(device)
    ckpt_path = os.path.join(run_dir, "checkpoints", "model_final.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor())
    
    # 0=T-shirt, 1=Trouser, 7=Sneaker, 8=Bag
    corners = {0: None, 1: None, 7: None, 8: None}
    for img, label in dataset:
        if label in corners and corners[label] is None:
            corners[label] = img.unsqueeze(0).to(device)
        if all(v is not None for v in corners.values()):
            break

    with torch.no_grad():
        mu_tl, _ = model.encode(corners[0])  # Top-Left: T-shirt
        mu_tr, _ = model.encode(corners[1])  # Top-Right: Trouser
        mu_bl, _ = model.encode(corners[7])  # Bottom-Left: Sneaker
        mu_br, _ = model.encode(corners[8])  # Bottom-Right: Bag

        z_grid = []
        for i in range(args.grid_size):
            alpha_y = i / (args.grid_size - 1)  # Vertical transition weight
            for j in range(args.grid_size):
                alpha_x = j / (args.grid_size - 1)  # Horizontal transition weight

                # Horizontal interpolation
                z_top = (1 - alpha_x) * mu_tl + alpha_x * mu_tr
                z_bottom = (1 - alpha_x) * mu_bl + alpha_x * mu_br

                # Vertical interpolation
                z_interp = (1 - alpha_y) * z_top + alpha_y * z_bottom
                z_grid.append(z_interp)

        # Decode all 64 latent vectors
        z_batch = torch.cat(z_grid, dim=0)  # [grid_size*grid_size, latent_dim]
        recon_imgs = model.decode(z_batch)  # [grid_size*grid_size, 1, 28, 28]

        # Save the grid
        save_path = os.path.join(run_dir, "images", "latent_interpolation.png")
        save_image(recon_imgs.cpu(), save_path, nrow=args.grid_size, padding=2, pad_value=1.0)
        
    print(f"Saved interpolation grid to: {save_path}")


if __name__ == "__main__":
    main()
