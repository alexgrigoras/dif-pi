from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _LSTMForecast(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )
    def forward(self, x):
        out, (h, _) = self.lstm(x)
        return self.head(h[-1])

class LSTMForecastGenerator:
    def __init__(self, context_length: int, horizon: int, hidden_dim: int = 64, num_layers: int = 1,
                 lr: float = 1e-3, batch_size: int = 64, epochs: int = 50, device: str | None = None, seed: int = 42):
        self.context_length = int(context_length)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = _LSTMForecast(self.hidden_dim, self.horizon, self.num_layers).to(self.device)
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
        Y = torch.tensor(self._scale(np.asarray(futures, dtype=np.float32)), dtype=torch.float32)
        ds = TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for epoch in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"[LSTM] epoch={epoch+1}/{self.epochs} loss={float(loss):.4f}")
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
