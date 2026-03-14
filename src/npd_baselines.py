
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import xgboost as xgb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class XGBoostNPDRegressor:
    """Recursive XGBoost baseline for customer NPD gap forecasting."""

    def __init__(
        self,
        input_length: int = 10,
        n_estimators: int = 1000,
        learning_rate: float = 0.3,
        max_depth: int = 4,
        random_state: int = 42,
    ):
        self.input_length = int(input_length)
        self.model = xgb.XGBRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            objective="reg:squarederror",
            random_state=int(random_state),
        )

    @staticmethod
    def make_windows(values: Sequence[float], input_length: int) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        X, y = [], []
        for i in range(int(input_length), len(arr)):
            X.append(arr[i - int(input_length):i])
            y.append(arr[i])
        if not X:
            return np.empty((0, int(input_length))), np.empty((0,))
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    def fit_series(self, series: Sequence[float]) -> "XGBoostNPDRegressor":
        X, y = self.make_windows(series, self.input_length)
        if len(X) == 0:
            raise ValueError("Not enough observations for XGBoostNPDRegressor.")
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.maximum(self.model.predict(X), 1.0)

    def forecast_series(self, history: Sequence[float], forecast_length: int) -> np.ndarray:
        hist = list(np.asarray(history, dtype=float).reshape(-1))
        preds = []
        for _ in range(int(forecast_length)):
            x = np.asarray(hist[-self.input_length:], dtype=float).reshape(1, -1)
            nxt = float(self.predict(x)[0])
            preds.append(max(1.0, nxt))
            hist.append(nxt)
        return np.asarray(preds, dtype=float)


class _LSTMGapModel(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


class LSTMNPDRegressor:
    """Compact PyTorch LSTM baseline for customer NPD gap forecasting."""

    def __init__(
        self,
        input_length: int = 5,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        batch_size: int = 8,
        epochs: int = 300,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.input_length = int(input_length)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _LSTMGapModel(self.hidden_dim).to(self.device)
        self.value_mean_ = 0.0
        self.value_std_ = 1.0

    @staticmethod
    def make_windows(values: Sequence[float], input_length: int) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        X, y = [], []
        for i in range(int(input_length), len(arr)):
            X.append(arr[i - int(input_length):i])
            y.append(arr[i])
        if not X:
            return np.empty((0, int(input_length))), np.empty((0,))
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.value_mean_) / self.value_std_

    def _inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.value_std_ + self.value_mean_

    def fit_series(self, series: Sequence[float], verbose: bool = False) -> "LSTMNPDRegressor":
        torch.manual_seed(self.seed)
        values = np.asarray(series, dtype=float).reshape(-1)
        if len(values) < self.input_length + 1:
            raise ValueError("Not enough observations for LSTMNPDRegressor.")

        X, y = self.make_windows(values, self.input_length)
        self.value_mean_ = float(np.mean(values))
        self.value_std_ = float(np.std(values) + 1e-6)

        Xs = torch.tensor(self._scale(X).astype(np.float32)).unsqueeze(-1)
        ys = torch.tensor(self._scale(y).astype(np.float32))
        dl = DataLoader(TensorDataset(Xs, ys), batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xs = torch.tensor(self._scale(X).astype(np.float32)).unsqueeze(-1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(Xs).cpu().numpy()
        pred = self._inverse(pred.reshape(-1))
        return np.maximum(pred, 1.0)

    def forecast_series(self, history: Sequence[float], forecast_length: int) -> np.ndarray:
        hist = list(np.asarray(history, dtype=float).reshape(-1))
        preds = []
        for _ in range(int(forecast_length)):
            x = np.asarray(hist[-self.input_length:], dtype=float).reshape(1, -1)
            nxt = float(self.predict(x)[0])
            preds.append(max(1.0, nxt))
            hist.append(nxt)
        return np.asarray(preds, dtype=float)
