import argparse
import os
import json

import torch
from torchinfo import summary

from models.vae import VAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    run_dir = os.path.join("results", args.run_name)
    args_file = os.path.join(run_dir, "args.json")
    ckpt_path = os.path.join(run_dir, "checkpoints", "model_final.pt")

    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Directory not found: {run_dir}")

    with open(args_file, "r") as f:
        run_args = json.load(f)

    use_accel = not run_args["no_accel"] and torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Loading run: {args.run_name}")

    model = VAE(latent_dim=run_args["latent_dim"]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_size = (args.batch_size, 1, 28, 28)

    s = summary(model, input_size=input_size, device=device)

    print(s)

    out_txt = os.path.join(run_dir, "model_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(str(s) + "\n")

    print(f"Saved summary to: {out_txt}")


if __name__ == "__main__":
    main()
