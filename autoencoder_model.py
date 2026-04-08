"""PyTorch autoencoder for unsupervised anomaly detection (reconstruction error)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class InsiderAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, max(latent_dim * 2, 16)),
            nn.ReLU(),
            nn.Linear(max(latent_dim * 2, 16), latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim * 2, 16)),
            nn.ReLU(),
            nn.Linear(max(latent_dim * 2, 16), n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(
    X: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    latent_dim: int = 8,
    device: str | None = None,
    seed: int = 42,
) -> tuple[InsiderAutoencoder, np.ndarray]:
    """
    Train on (presumed mostly normal) data; returns model and per-sample train MSE.
    """
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[1]
    model = InsiderAutoencoder(n_features, latent_dim=latent_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tensor_x = torch.tensor(X, dtype=torch.float32, device=dev)
    loader = DataLoader(
        TensorDataset(tensor_x), batch_size=batch_size, shuffle=True, drop_last=False
    )

    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        recon = model(tensor_x)
        mse = ((recon - tensor_x) ** 2).mean(dim=1).cpu().numpy()
    return model, mse


@torch.no_grad()
def reconstruction_errors(
    model: InsiderAutoencoder, X: np.ndarray, device: str | None = None
) -> np.ndarray:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    recon = model(x)
    return ((recon - x) ** 2).mean(dim=1).cpu().numpy()


def anomaly_scores_from_ae(
    model: InsiderAutoencoder,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    percentile: float = 95.0,
) -> tuple[np.ndarray, float]:
    """
    Threshold from train reconstruction error distribution; scores on eval are raw MSE.
    Returns (scores_eval, threshold).
    """
    train_err = reconstruction_errors(model, X_train)
    thresh = float(np.percentile(train_err, percentile))
    eval_err = reconstruction_errors(model, X_eval)
    return eval_err, thresh
