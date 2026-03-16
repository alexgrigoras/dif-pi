from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _TimeGANLike(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int):
        super().__init__()
        self.encoder = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.GRU(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        self.horizon = int(horizon)

    def forward(self, context, teacher_future=None):
        _, h = self.encoder(context)
        prev = context[:, -1:, :]
        outputs = []
        h_dec = h
        for t in range(self.horizon):
            out, h_dec = self.decoder(prev, h_dec)
            step = self.head(out)
            outputs.append(step)
            if self.training and teacher_future is not None:
                prev = teacher_future[:, t:t+1, :]
            else:
                prev = step
        return torch.cat(outputs, dim=1).squeeze(-1)

class TimeGANGenerator:
    """Lightweight time-aware recurrent generator benchmark."""
    def __init__(self, context_length: int, horizon: int, hidden_dim: int = 64,
                 lr: float = 1e-3, batch_size: int = 64, epochs: int = 50, device: str | None = None, seed: int = 42):
        self.context_length = int(context_length)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = _TimeGANLike(self.hidden_dim, self.horizon).to(self.device)
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

        X = torch.tensor(self._scale(np.asarray(contexts, dtype=np.float32)), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(self._scale(np.asarray(futures, dtype=np.float32)), dtype=torch.float32).unsqueeze(-1)
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for epoch in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb, teacher_future=yb)
                loss = loss_fn(pred, yb.squeeze(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"[TimeGAN] epoch={epoch+1}/{self.epochs} loss={float(loss):.4f}")
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
        x = torch.tensor(self._scale(ctx), dtype=torch.float32).unsqueeze(-1).to(self.device)
        with torch.inference_mode():
            y = self.model(x).detach().cpu().numpy().reshape(-1)
        y = self._inverse(y)
        return np.maximum(y[:horizon], 0.0)
