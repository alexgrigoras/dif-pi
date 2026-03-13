from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)

class _Decoder(nn.Module):
    def __init__(self, context_dim: int, latent_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )
    def forward(self, context_flat, z):
        return self.net(torch.cat([context_flat, z], dim=1))

class DGANGenerator:
    """Lightweight DGAN-style dynamic sequence generator baseline."""
    def __init__(self, context_length: int, horizon: int, hidden_dim: int = 128, latent_dim: int = 16,
                 lr: float = 1e-3, batch_size: int = 64, epochs: int = 50, device: str | None = None, seed: int = 42):
        self.context_length = int(context_length)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = _Encoder(self.context_length + self.horizon, self.hidden_dim, self.latent_dim).to(self.device)
        self.decoder = _Decoder(self.context_length, self.latent_dim, self.hidden_dim, self.horizon).to(self.device)
        self.value_mean = 0.0
        self.value_std = 1.0

    def _scale(self, arr):
        return (arr - self.value_mean) / self.value_std
    def _inverse(self, arr):
        return arr * self.value_std + self.value_mean

    def fit(self, contexts: np.ndarray, futures: np.ndarray, verbose: bool = False):
        torch.manual_seed(self.seed)
        values = np.concatenate([contexts.reshape(-1), futures.reshape(-1)])
        self.value_mean = float(np.mean(values))
        self.value_std = float(np.std(values) + 1e-6)

        X = torch.tensor(self._scale(np.asarray(contexts, dtype=np.float32)), dtype=torch.float32)
        Y = torch.tensor(self._scale(np.asarray(futures, dtype=np.float32)), dtype=torch.float32)
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        for epoch in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                enc_in = torch.cat([xb, yb], dim=1)
                mu, logvar = self.encoder(enc_in)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                pred = self.decoder(xb, z)
                recon = torch.mean((pred - yb) ** 2)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + 0.01 * kl
                opt.zero_grad()
                loss.backward()
                opt.step()
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"[DGAN] epoch={epoch+1}/{self.epochs} loss={float(loss):.4f}")
        return self

    def generate(self, context_values: np.ndarray, horizon: int | None = None) -> np.ndarray:
        horizon = int(horizon or self.horizon)
        ctx = np.asarray(context_values, dtype=np.float32).reshape(1, -1)
        if ctx.shape[1] != self.context_length:
            if ctx.shape[1] > self.context_length:
                ctx = ctx[:, -self.context_length:]
            else:
                pad = np.repeat(ctx[:, :1], self.context_length - ctx.shape[1], axis=1) if ctx.shape[1] else np.zeros((1, self.context_length))
                ctx = np.concatenate([pad, ctx], axis=1)
        x = torch.tensor(self._scale(ctx), dtype=torch.float32).to(self.device)
        z = torch.zeros((1, self.latent_dim), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            y = self.decoder(x, z).detach().cpu().numpy().reshape(-1)
        y = self._inverse(y)
        return np.maximum(y[:horizon], 0.0)
