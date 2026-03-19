import argparse
import os
import csv
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    run_dir = os.path.join("results", args.run_name)
    log_path = os.path.join(run_dir, "log.csv")

    epochs = []
    train_loss, train_bce, train_kld = [], [], []
    test_loss, test_bce, test_kld = [], [], []

    with open(log_path, "r") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_bce.append(float(row["train_bce"]))
            train_kld.append(float(row["train_kld"]))
            test_loss.append(float(row["test_loss"]))
            test_bce.append(float(row["test_bce"]))
            test_kld.append(float(row["test_kld"]))

    metrics = [
        ("Total Loss", train_loss, test_loss),
        ("BCE", train_bce, test_bce),
        ("KLD", train_kld, test_kld)
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, train_data, test_data) in zip(axs, metrics):
        ax.plot(epochs, train_data, label="Train")
        ax.plot(epochs, test_data, label="Test")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(save_path)
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
