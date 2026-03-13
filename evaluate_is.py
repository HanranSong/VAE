import argparse
import json
import os

import torch
import torch.utils.data
from torchvision import datasets, transforms

from models.vae import VAE
from models.priors import build_prior
from utils.metrics import estimate_dataset_log_likelihood_is


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--num-importance-samples", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    run_dir = os.path.join("results", args.run_name)
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Directory not found: {run_dir}")

    args_file = os.path.join(run_dir, "args.json")
    with open(args_file, "r") as f:
        run_args = json.load(f)

    use_accel = torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Loading run: {run_dir}")

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor()),
        batch_size=512, shuffle=False)

    model = VAE(latent_dim=run_args["latent_dim"]).to(device)
    prior = build_prior(run_args["prior"]).to(device)

    ckpt_path = os.path.join(run_dir, "checkpoints", "model_final.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if "prior_state_dict" in checkpoint:
        prior.load_state_dict(checkpoint["prior_state_dict"])

    model.eval()
    prior.eval()

    avg_test_ll = estimate_dataset_log_likelihood_is(
        model=model,
        data_loader=test_loader,
        prior=prior,
        device=device,
        num_importance_samples=args.num_importance_samples,
        chunk_size=args.chunk_size
    )

    print(f"Average test marginal log-likelihood: {avg_test_ll:.4f}")

    save_path = os.path.join(run_dir, "importance_sampling_metrics.json")
    with open(save_path, "w") as f:
        json.dump(
            {
                "run_name": args.run_name,
                "prior": run_args["prior"],
                "latent_dim": run_args["latent_dim"],
                "num_importance_samples": args.num_importance_samples,
                "chunk_size": args.chunk_size,
                "average_test_marginal_log_likelihood": avg_test_ll
            },
            f,
            indent=4,
        )

    print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    main()
