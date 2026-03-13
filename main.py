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
from models.priors import StandardGaussianPrior
from utils.losses import loss_function


def train(model, optimizer, train_loader, device, prior, beta):
    model.train()
    train_loss, train_bce, train_kld = 0, 0, 0
    
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar, z = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar, z, prior, beta)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_bce += bce.item()
        train_kld += kld.item()
        
    dataset_size = len(train_loader.dataset)
    return train_loss / dataset_size, train_bce / dataset_size, train_kld / dataset_size


def test(epoch, model, test_loader, device, prior, img_dir, log_interval, beta):
    model.eval()
    test_loss, test_bce, test_kld = 0, 0, 0
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar, z, prior, beta)
            test_loss += loss.item()
            test_bce += bce.item()
            test_kld += kld.item()
            
            if i == 0 and epoch % log_interval == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           os.path.join(img_dir, f"reconstruction_{epoch}.png"), nrow=n)
                           
    dataset_size = len(test_loader.dataset)        
    return test_loss / dataset_size, test_bce / dataset_size, test_kld / dataset_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--no-accel", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--prior", type=str, default="standard_gaussian")
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
    torch.manual_seed(args.seed)
    use_accel = not args.no_accel and torch.accelerator.is_available()
    device = torch.accelerator.current_accelerator() if use_accel else torch.device("cpu")
    print(f"Using device: {device}")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_accel else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("./data", train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # ---------- Initialize model ----------
    model = VAE(latent_dim=args.latent_dim).to(device)
    if args.prior == "standard_gaussian":
        prior = StandardGaussianPrior().to(device)
    else:
        raise ValueError(f"Unknown prior: {args.prior}")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ---------- Training loop ----------
    pbar = tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        train_loss, train_bce, train_kld = train(model, optimizer, train_loader, device, prior, args.beta)
        test_loss, test_bce, test_kld = test(epoch, model, test_loader, device, prior, img_dir, args.log_interval, args.beta)

        # Save log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_bce, train_kld, test_loss, test_bce, test_kld])

        # Save model and sample images at intervals
        if epoch % args.log_interval == 0:
            with torch.no_grad():
                sample = prior.sample(64, args.latent_dim, device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28), os.path.join(img_dir, f"sample_{epoch}.png"))
                
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
