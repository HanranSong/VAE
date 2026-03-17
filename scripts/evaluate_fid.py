import argparse
import json
import os

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from models.vae import VAE
from models.priors import build_prior
from utils.seed import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
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
    print(f"Loading run: {run_dir}")

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor()),
        batch_size=512, shuffle=False
    )

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    model = VAE(latent_dim=run_args["latent_dim"]).to(device)
    prior = build_prior(
        run_args["prior"], 
        latent_dim=run_args["latent_dim"],
        num_components=run_args["num_components"]
    ).to(device)

    ckpt_path = os.path.join(run_dir, "checkpoints", "model_final.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    if "prior_state_dict" in checkpoint:
        prior.load_state_dict(checkpoint["prior_state_dict"])

    model.eval()
    prior.eval()

    print("Processing real images")
    for real_imgs, _ in tqdm(test_loader):
        real_imgs = real_imgs.to(device)
        real_imgs_rgb = real_imgs.repeat(1, 3, 1, 1)  # [B, 1, 28, 28] -> [B, 3, 28, 28]
        fid.update(real_imgs_rgb, real=True)

    print("Generating and processing fake images")
    num_test_samples = len(test_loader.dataset)
    num_generated = 0
    
    with torch.no_grad():
        with tqdm(total=num_test_samples) as pbar:
            while num_generated < num_test_samples:
                current_batch_size = min(512, num_test_samples - num_generated)
                
                z = prior.sample(current_batch_size, device)
                fake_imgs = model.decode(z) 
                
                fake_imgs_rgb = fake_imgs.repeat(1, 3, 1, 1)
                fid.update(fake_imgs_rgb, real=False)
                
                num_generated += current_batch_size
                pbar.update(current_batch_size)

    print("Computing FID score")
    fid_score = fid.compute().item()
    print(f"FID Score for {run_args['prior']} prior: {fid_score:.4f}")

    fid_save_path = os.path.join(run_dir, "fid_score.json")
    with open(fid_save_path, "w") as f:
        json.dump({
            "run_name": args.run_name,
            "prior": run_args["prior"],
            "fid_score": fid_score
        }, f, indent=4)
    
    print(f"Saved FID result to: {fid_save_path}")


if __name__ == "__main__":
    main()
