# VAE with Multiple Priors

A PyTorch implementation of a Variational Autoencoder (VAE) trained on FashionMNIST with support for multiple latent prior distributions.

## Installation

This project uses **Python 3.12** and installs dependencies from a `requirements.txt` file.

### 1. Create and activate a conda environment

```bash
conda create -n myenv python=3.12 -y
conda activate myenv
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure
* `main.py`: Core training script.
* `models/`: VAE architecture and prior definitions (`gaussian`, `mog`, `laplace`, `student-t`, `vampprior`).
* `scripts/`: Tools for model evaluation and visualization.
* `utils/`: Helpers for losses, metrics, and reproducibility.

## Training

Start training by running:

```bash
python main.py
```

Outputs are saved in `results/<run_name>/`.

## Evaluation
To generate all metrics and visualizations, run the automated script:

```Bash
python -m scripts.run_all --run-name <your_run_folder_name>
```
Note: You can also run individual scripts from the `scripts/` folder as needed.

## License
MIT License.
