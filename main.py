import argparse
import os
import datetime
import json
import csv

import torch
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.vae import VAE
from models.priors import build_prior
from utils.losses import loss_function
from utils.seed import set_all_seeds


def train(model, optimizer, train_loader, prior, beta):
    model.train()
    train_loss, train_bce, train_kld = 0, 0, 0
    
    for data, _ in train_loader:
        optimizer.zero_grad()
        
        recon_batch, mu, logvar, z = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar, z, prior, beta)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.detach()
        train_bce += bce.detach()
        train_kld += kld.detach()
        
    num_batches = len(train_loader)
    return train_loss.item() / num_batches, train_bce.item() / num_batches, train_kld.item() / num_batches


def test(model, test_loader, prior, beta):
    model.eval()
    test_loss, test_bce, test_kld = 0, 0, 0
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            recon_batch, mu, logvar, z = model(data)
            
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar, z, prior, beta)
            test_loss += loss.detach()
            test_bce += bce.detach()
            test_kld += kld.detach()
            
    num_batches = len(test_loader)
    return test_loss.item() / num_batches, test_bce.item() / num_batches, test_kld.item() / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--no-accel", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--prior", type=str, default="gaussian")
    parser.add_argument("--num-components", type=int, default=10)
    parser.add_argument("--df", type=float, default=3.0)
    args = parser.parse_args()

    # ---------- Directory setup ----------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.prior}_{timestamp}"
    out_dir = os.path.join("results", run_name)
    img_dir = os.path.join(out_dir, "images")
    ckpt_dir = os.path.join(out_dir, "checkpoints")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Directory created at: {out_dir}")

    # Save parameters
    args_to_save = {"run_name": run_name, **vars(args)}
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(args_to_save, f, indent=4)

    # Initialize log file
    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_bce", "train_kld", "test_loss", "test_bce", "test_kld"])

    # ---------- Device & data ----------
    set_all_seeds(args.seed)

    use_accel = not args.no_accel and torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")
    print(f"Using device: {device}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    print("Loading dataset into GPU memory")

    train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor())

    fixed_images = []
    class_found = [False] * 10
    for img, label in test_dataset:
        if not class_found[label]:
            fixed_images.append((img, label))
            class_found[label] = True
        if all(class_found):
            break
    fixed_images.sort(key=lambda x: x[1])
    fixed_recon_tensor = torch.stack([x[0] for x in fixed_images]).to(device)  # [10, 1, 28, 28]

    train_data = train_dataset.data.unsqueeze(1).float().to(device) / 255.0
    train_targets = train_dataset.targets.to(device)

    test_data = test_dataset.data.unsqueeze(1).float().to(device) / 255.0
    test_targets = test_dataset.targets.to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_targets),
        batch_size=args.batch_size, shuffle=True, generator=g
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_targets),
        batch_size=args.batch_size, shuffle=False, generator=g
    )

    # ---------- Initialize model ----------
    model = VAE(latent_dim=args.latent_dim).to(device)
    
    prior = build_prior(
        args.prior,
        latent_dim=args.latent_dim,
        num_components=args.num_components,
        df=args.df
    ).to(device)

    if hasattr(prior, 'set_model'):
        prior.set_model(model)
    
    params_to_optimize = list(model.parameters()) + list(prior.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=args.learning_rate)

    # ---------- Training loop ----------
    pbar = tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        train_loss, train_bce, train_kld = train(model, optimizer, train_loader, prior, args.beta)
        test_loss, test_bce, test_kld = test(model, test_loader, prior, args.beta)

        # Save log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_bce, train_kld, test_loss, test_bce, test_kld])

        # Save model and sample images at intervals
        if epoch % args.log_interval == 0:
            with torch.no_grad():
                sample = prior.sample(64, device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28), os.path.join(img_dir, f"sample_{epoch}.png"))
                
                recon, _, _, _ = model(fixed_recon_tensor)
                comparison = torch.cat([fixed_recon_tensor.cpu(), recon.view(-1, 1, 28, 28).cpu()])
                save_image(comparison, os.path.join(img_dir, f"reconstruction_{epoch}.png"), nrow=10)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "prior_state_dict": prior.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt"))
        
        # Update progress bar
        pbar.set_postfix({
            "Train": f"{train_loss:.2f}",
            "BCE": f"{train_bce:.2f}",
            "KLD": f"{train_kld:.2f}",
            "Test": f"{test_loss:.2f}"
        })
    
    # ---------- Final save ----------
    torch.save({
        "model_state_dict": model.state_dict(),
        "prior_state_dict": prior.state_dict(),
    }, os.path.join(ckpt_dir, "model_final.pt"))
    print(f"Final results saved in {out_dir}")


if __name__ == "__main__":
    main()
