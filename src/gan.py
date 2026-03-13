from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class _Generator(nn.Module):
    def __init__(self, context_length: int, horizon: int, noise_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_length + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )
    def forward(self, context_flat, noise):
        x = torch.cat([context_flat, noise], dim=1)
        return self.net(x)

class _Discriminator(nn.Module):
    def __init__(self, context_length: int, horizon: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_length + horizon, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    def forward(self, context_flat, future_flat):
        x = torch.cat([context_flat, future_flat], dim=1)
        return self.net(x)

class GANTimeSeriesGenerator:
    """Lightweight conditional GAN baseline for future demand generation."""
    def __init__(self, context_length: int, horizon: int, noise_dim: int = 16, hidden_dim: int = 128,
                 lr: float = 1e-3, batch_size: int = 64, epochs: int = 50, device: str | None = None, seed: int = 42):
        self.context_length = int(context_length)
        self.horizon = int(horizon)
        self.noise_dim = int(noise_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = _Generator(self.context_length, self.horizon, self.noise_dim, self.hidden_dim).to(self.device)
        self.discriminator = _Discriminator(self.context_length, self.horizon, self.hidden_dim).to(self.device)
        self.value_mean = 0.0
        self.value_std = 1.0

    def _scale(self, arr):
        return (arr - self.value_mean) / self.value_std

    def _inverse(self, arr):
        return arr * self.value_std + self.value_mean

    def fit(self, contexts: np.ndarray, futures: np.ndarray, verbose: bool = False):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        values = np.concatenate([contexts.reshape(-1), futures.reshape(-1)])
        self.value_mean = float(np.mean(values))
        self.value_std = float(np.std(values) + 1e-6)

        X = torch.tensor(self._scale(np.asarray(contexts, dtype=np.float32)), dtype=torch.float32)
        Y = torch.tensor(self._scale(np.asarray(futures, dtype=np.float32)), dtype=torch.float32)
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        bce = nn.BCELoss()

        for epoch in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                bs = xb.shape[0]
                real_lab = torch.ones(bs, 1, device=self.device)
                fake_lab = torch.zeros(bs, 1, device=self.device)

                # discriminator
                z = torch.randn(bs, self.noise_dim, device=self.device)
                fake = self.generator(xb, z).detach()
                d_real = self.discriminator(xb, yb)
                d_fake = self.discriminator(xb, fake)
                loss_d = bce(d_real, real_lab) + bce(d_fake, fake_lab)
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                # generator
                z = torch.randn(bs, self.noise_dim, device=self.device)
                gen = self.generator(xb, z)
                d_gen = self.discriminator(xb, gen)
                loss_g = bce(d_gen, real_lab) + 0.1 * torch.mean((gen - yb) ** 2)
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"[GAN] epoch={epoch+1}/{self.epochs} loss_d={float(loss_d):.4f} loss_g={float(loss_g):.4f}")
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
        x = torch.tensor(self._scale(ctx), dtype=torch.float32, device=self.device)
        z = torch.zeros((1, self.noise_dim), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            y = self.generator(x, z).detach().cpu().numpy().reshape(-1)
        y = self._inverse(y)
        y = np.maximum(y, 0.0)
        return y[:horizon]
