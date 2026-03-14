import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Any, List, Sequence
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import set_random_seed
from tensorflow.config.experimental import enable_op_determinism
from tensorflow.keras.utils import register_keras_serializable

SEED = 42
set_random_seed(SEED)
enable_op_determinism()


@register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        position = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = position * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)
        self.d_model = d_model

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model
        })
        return config



class EpochProgressPrinter(Callback):
    """Compact training progress callback for notebook runs."""

    def __init__(self, label: str = "training", report_every: int = 1, enabled: bool = True):
        super().__init__()
        self.label = str(label)
        self.report_every = max(1, int(report_every))
        self.enabled = bool(enabled)

    def on_train_begin(self, logs=None):
        if not self.enabled:
            return
        epochs = self.params.get("epochs")
        steps = self.params.get("steps")
        print(f"[{self.label}] start | epochs={epochs} | steps_per_epoch={steps}")

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled:
            return
        if ((epoch + 1) % self.report_every) != 0:
            return
        logs = logs or {}
        metric_parts = []
        for key in ("loss", "mae", "val_loss", "val_mae", "learning_rate"):
            if key in logs and logs[key] is not None:
                try:
                    metric_parts.append(f"{key}={float(logs[key]):.4f}")
                except Exception:
                    pass
        joined = " | ".join(metric_parts) if metric_parts else "no metrics"
        print(f"[{self.label}] epoch {epoch + 1}: {joined}")

    def on_train_end(self, logs=None):
        if not self.enabled:
            return
        print(f"[{self.label}] complete")


class ScenarioGenerationTransformerForecaster:
    """Global Transformer forecaster trained on pooled windows from many SKUs.

    Key design:
    - Model weights are trained globally on a TRAIN_SKU set.
    - Scaling is applied per SKU at inference time using only that SKU's history
      (MinMaxScaler fitted on the available history window).
    - This supports the DIF-PI protocol: train once, save, then reuse in dif-pi notebook.

    Notes:
    - This is a univariate forecaster (demand-only). Price is handled separately by scenario generators.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        size_layer: int = 64,
        embedded_size: int = 64,
        output_size: int = 1,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ):
        self.sequence_length = int(sequence_length)
        self.output_size = int(output_size)
        self.model: Optional[tf.keras.Model] = None
        # Global model config (weights are shared)
        self.config = {
            "sequence_length": self.sequence_length,
            "size_layer": size_layer,
            "embedded_size": embedded_size,
            "output_size": self.output_size,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate,
        }

    def _transformer_block(self, x, size_layer, num_heads, dropout_rate):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=size_layer)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = Dense(size_layer, activation='relu')(x)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(x + ffn_output)

    def _build_model(self, size_layer, embedded_size, output_size, num_heads, dropout_rate, sequence_length):
        inputs = tf.keras.Input(shape=(sequence_length, output_size))
        x = Dense(embedded_size, activation='relu')(inputs)
        x = PositionalEncoding(d_model=embedded_size)(x)

        for _ in range(2):  # stacked Transformer blocks
            x = self._transformer_block(x, size_layer, num_heads, dropout_rate)

        x = Dense(embedded_size, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = x[:, -1, :]
        outputs = Dense(output_size)(x)
        return tf.keras.Model(inputs, outputs)

    @staticmethod
    def _make_windows(y_scaled: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(sequence_length, len(y_scaled)):
            X.append(y_scaled[i - sequence_length:i])
            y.append(y_scaled[i])
        return np.asarray(X), np.asarray(y)

    def fit_global(
        self,
        panel_df,
        sku_col: str,
        time_col: str,
        target_col: str,
        train_skus: Iterable[Any],
        batch_size: int = 64,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        per_sku_train_frac: float = 1.0,
        min_points: int = 60,
        verbose: int = 1,
        progress_label: str = "ScenarioGenerationTransformerForecaster",
        progress_report_every: int = 1,
    ) -> Dict[str, Any]:
        """Train one global model on pooled windows across multiple SKUs.

        Parameters
        ----------
        per_sku_train_frac : float
            Fraction of each SKU series to use for training windows (<=1.0).
            Set to e.g. 0.9 to drop the tail of each TRAIN_SKU (optional).
        min_points : int
            Minimum points per SKU to contribute (after truncation).
        """
        clear_session()

        train_skus = [str(s) for s in list(train_skus)]

        # Normalize SKU column dtype to string for robust matching
        panel_df = panel_df.copy()
        panel_df[sku_col] = panel_df[sku_col].astype(str)

        # Build pooled training windows
        X_all, y_all = [], []
        n_used = 0
        n_skipped = 0
        for sku in train_skus:
            s = panel_df.loc[panel_df[sku_col] == sku].sort_values(time_col)
            y = s[target_col].astype(float).values.reshape(-1, 1)
            if len(y) < max(min_points, self.sequence_length + 2):
                n_skipped += 1
                continue

            n_tr = int(np.floor(len(y) * float(per_sku_train_frac)))
            n_tr = max(n_tr, self.sequence_length + 2)
            y_tr = y[:n_tr]

            scaler = MinMaxScaler()
            y_scaled = scaler.fit_transform(y_tr)

            X_sku, y_sku = self._make_windows(y_scaled, self.sequence_length)
            if len(X_sku) == 0:
                n_skipped += 1
                continue

            X_all.append(X_sku)
            y_all.append(y_sku)
            n_used += 1

        if n_used == 0:
            raise ValueError(
                "No SKUs had enough history to train the global forecaster. "
                "This often happens when SKU ids have different dtypes (e.g., int vs str) or when min_points is too high. "
                f"Example train_sku[0:5]={train_skus[:5]} | sequence_length={self.sequence_length} | min_points={min_points}"
            )

        X_train = np.concatenate(X_all, axis=0)
        y_train = np.concatenate(y_all, axis=0)

        self.model = self._build_model(
            size_layer=self.config["size_layer"],
            embedded_size=self.config["embedded_size"],
            output_size=self.output_size,
            num_heads=self.config["num_heads"],
            dropout_rate=self.config["dropout_rate"],
            sequence_length=self.sequence_length,
        )
        self.model.compile(optimizer=Adam(learning_rate), loss=Huber())

        early_stopping = EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-5)
        progress = EpochProgressPrinter(
            label=str(progress_label),
            report_every=int(progress_report_every),
            enabled=bool(verbose),
        )

        hist = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, progress],
            verbose=verbose,
            shuffle=True
        )

        info = {
            "n_train_skus": len(train_skus),
            "n_used_skus": n_used,
            "n_skipped_skus": n_skipped,
            "n_train_windows": int(X_train.shape[0]),
            "sequence_length": self.sequence_length,
            "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
        }
        return info

    def forecast(self, history: np.ndarray, forecast_length: int) -> np.ndarray:
        """Forecast using global weights, fitting a per-series scaler on history."""
        if self.model is None:
            raise ValueError("Global model not initialized. Load or fit_global first.")

        y = np.asarray(history, dtype=float).reshape(-1, 1)
        if len(y) == 0:
            return np.zeros(int(forecast_length), dtype=float)

        # Fit scaler on this SKU history only (no leakage into future)
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y)

        window = self.sequence_length
        # pad if too short
        if y_scaled.shape[0] < window:
            pad = np.repeat(y_scaled[[0]], window - y_scaled.shape[0], axis=0)
            y_scaled = np.vstack([pad, y_scaled])

        buf = y_scaled.copy()
        preds = []
        for _ in range(int(forecast_length)):
            x = buf[-window:].reshape(1, window, self.output_size)
            pred = self.model.predict(x, verbose=0)
            preds.append(pred[0])
            buf = np.vstack([buf, pred])

        preds = np.asarray(preds).reshape(-1, self.output_size)
        return scaler.inverse_transform(preds).ravel()

    def save(self, model_dir: str) -> str:
        """Save model weights and config to a directory."""
        if self.model is None:
            raise ValueError("Nothing to save: model is None.")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "scenario_gen_transformer.keras"
        cfg_path = model_dir / "scenario_gen_transformer_config.json"
        self.model.save(model_path)
        cfg_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        return str(model_path)

    @classmethod
    def load(cls, model_dir: str) -> "ScenarioGenerationTransformerForecaster":
        model_dir = Path(model_dir)
        cfg_path = model_dir / "scenario_gen_transformer_config.json"
        model_path = model_dir / "scenario_gen_transformer.keras"
        if not cfg_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Missing saved model in {model_dir}. Expected {model_path.name} and {cfg_path.name}.")

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        inst = cls(
            sequence_length=int(cfg.get("sequence_length", 30)),
            size_layer=int(cfg.get("size_layer", 64)),
            embedded_size=int(cfg.get("embedded_size", 64)),
            output_size=int(cfg.get("output_size", 1)),
            num_heads=int(cfg.get("num_heads", 8)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        )
        inst.config = cfg
        inst.model = tf.keras.models.load_model(model_path, custom_objects={"PositionalEncoding": PositionalEncoding})
        return inst



class NPDTransformerForecaster:
    """Encoder-only Transformer for Next Purchase Day (NPD) prediction.

    Supports both classic single-step forecasting and direct multi-step forecasting.
    When ``output_horizon > 1``, the model predicts the next block of purchase gaps
    in one forward pass, which reduces recursive error accumulation.
    """

    def __init__(
        self,
        input_length: int = 12,
        output_horizon: int = 5,
        size_layer: int = 64,
        embedded_size: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        num_blocks: int = 2,
        seed: int = SEED,
    ):
        self.input_length = int(input_length)
        self.output_horizon = max(1, int(output_horizon))
        self.size_layer = int(size_layer)
        self.embedded_size = int(embedded_size)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        self.num_blocks = int(num_blocks)
        self.seed = int(seed)

        self.model: Optional[tf.keras.Model] = None
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.history_: Dict[str, Any] = {}

        self.config = {
            "input_length": self.input_length,
            "output_horizon": self.output_horizon,
            "size_layer": self.size_layer,
            "embedded_size": self.embedded_size,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "num_blocks": self.num_blocks,
            "seed": self.seed,
        }

    @property
    def is_direct_multistep(self) -> bool:
        return int(self.output_horizon) > 1

    def _transformer_block(self, x):
        attn_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(1, self.embedded_size // max(1, self.num_heads)),
            dropout=self.dropout_rate,
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = Dense(self.size_layer, activation='relu')(x)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        ffn_output = Dense(self.embedded_size, activation='relu')(ffn_output)
        return LayerNormalization(epsilon=1e-6)(x + ffn_output)

    def _build_model(self) -> tf.keras.Model:
        inputs = Input(shape=(self.input_length, 1))
        x = Dense(self.embedded_size, activation='relu')(inputs)
        x = PositionalEncoding(d_model=self.embedded_size)(x)

        for _ in range(self.num_blocks):
            x = self._transformer_block(x)

        x = Dropout(self.dropout_rate)(x)
        x = x[:, -1, :]
        x = Dense(self.embedded_size, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.output_horizon)(x)
        return Model(inputs, outputs)

    @staticmethod
    def make_windows(values: np.ndarray, input_length: int, output_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        X, y = [], []
        input_length = int(input_length)
        output_horizon = max(1, int(output_horizon))
        last_start = len(arr) - input_length - output_horizon + 1
        for start_idx in range(max(0, last_start)):
            end_x = start_idx + input_length
            end_y = end_x + output_horizon
            X.append(arr[start_idx:end_x])
            y.append(arr[end_x:end_y])
        if not X:
            return np.empty((0, input_length)), np.empty((0, output_horizon))
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        verbose: int = 0,
        progress_label: str = "NPDTransformerForecaster",
        progress_report_every: int = 1,
    ) -> Dict[str, Any]:
        clear_session()
        set_random_seed(self.seed)
        enable_op_determinism()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D with shape (n_samples, input_length); got {X.shape}")
        if X.shape[1] != self.input_length:
            raise ValueError(f"Expected input_length={self.input_length}, got X.shape[1]={X.shape[1]}")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError(f"y must be 2D with shape (n_samples, output_horizon); got {y.shape}")
        if y.shape[1] != self.output_horizon:
            raise ValueError(f"Expected output_horizon={self.output_horizon}, got y.shape[1]={y.shape[1]}")
        if len(X) == 0:
            raise ValueError("Cannot fit NPDTransformerForecaster on an empty dataset.")

        Xs = self.x_scaler.fit_transform(X).reshape(-1, self.input_length, 1)
        ys = self.y_scaler.fit_transform(y)

        validation_data = None
        if X_val is not None and y_val is not None and len(X_val):
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            Xv = self.x_scaler.transform(X_val).reshape(-1, self.input_length, 1)
            yv = self.y_scaler.transform(y_val)
            validation_data = (Xv, yv)

        self.model = self._build_model()
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=Huber(),
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        monitor_metric = 'val_loss' if validation_data is not None else 'loss'
        callbacks = [
            EarlyStopping(monitor=monitor_metric, patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=8, min_lr=1e-5),
            EpochProgressPrinter(
                label=str(progress_label),
                report_every=int(progress_report_every),
                enabled=bool(verbose),
            ),
        ]

        fit_kwargs = dict(
            x=Xs,
            y=ys,
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=int(verbose),
            shuffle=True,
            callbacks=callbacks,
        )
        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data
        else:
            fit_kwargs["validation_split"] = 0.1 if len(Xs) >= 20 else 0.0

        hist = self.model.fit(**fit_kwargs)
        self.history_ = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        return {
            "n_samples": int(len(X)),
            "input_length": int(self.input_length),
            "output_horizon": int(self.output_horizon),
            "direct_multistep": bool(self.is_direct_multistep),
            "history": self.history_,
        }

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() or load() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_length:
            raise ValueError(f"Expected input_length={self.input_length}, got X.shape[1]={X.shape[1]}")
        Xs = self.x_scaler.transform(X).reshape(-1, self.input_length, 1)
        pred_scaled = self.model.predict(Xs, verbose=verbose)
        pred = self.y_scaler.inverse_transform(np.asarray(pred_scaled, dtype=float))
        pred = np.maximum(pred, 1.0)
        if pred.shape[0] == 1:
            return pred.reshape(-1)
        return pred

    def forecast_series(self, history: np.ndarray, forecast_length: int) -> np.ndarray:
        hist = list(np.asarray(history, dtype=float).reshape(-1))
        if len(hist) < self.input_length:
            raise ValueError(
                f"Need at least {self.input_length} historical values to forecast; got {len(hist)}."
            )

        target_horizon = max(1, int(forecast_length))
        preds: List[float] = []
        while len(preds) < target_horizon:
            x = np.asarray(hist[-self.input_length:], dtype=float).reshape(1, -1)
            next_block = self.predict(x, verbose=0).reshape(-1)
            block_remaining = target_horizon - len(preds)
            block = np.maximum(next_block[:block_remaining], 1.0)
            preds.extend(block.tolist())
            hist.extend(block.tolist())
        return np.asarray(preds[:target_horizon], dtype=float)

    def fit_series(
        self,
        series: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        verbose: int = 0,
        progress_label: str = "NPDTransformerForecaster",
        progress_report_every: int = 1,
    ) -> Dict[str, Any]:
        X, y = self.make_windows(series, self.input_length, self.output_horizon)
        return self.fit(
            X=X,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            progress_label=progress_label,
            progress_report_every=progress_report_every,
        )

    def save(self, model_dir: str) -> str:
        if self.model is None:
            raise ValueError("Nothing to save: model is None.")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "npd_transformer.keras"
        cfg_path = model_dir / "npd_transformer_config.json"
        scalers_path = model_dir / "npd_transformer_scalers.npz"

        self.model.save(model_path)
        cfg_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        np.savez(
            scalers_path,
            x_mean=self.x_scaler.mean_,
            x_scale=self.x_scaler.scale_,
            y_mean=self.y_scaler.mean_,
            y_scale=self.y_scaler.scale_,
        )
        return str(model_path)

    @classmethod
    def load(cls, model_dir: str) -> "NPDTransformerForecaster":
        model_dir = Path(model_dir)
        cfg_path = model_dir / "npd_transformer_config.json"
        model_path = model_dir / "npd_transformer.keras"
        scalers_path = model_dir / "npd_transformer_scalers.npz"

        if not cfg_path.exists() or not model_path.exists() or not scalers_path.exists():
            raise FileNotFoundError(
                f"Missing saved NPD Transformer artifacts in {model_dir}. "
                "Expected npd_transformer.keras, npd_transformer_config.json, and npd_transformer_scalers.npz."
            )

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        inst = cls(
            input_length=int(cfg.get("input_length", 12)),
            output_horizon=int(cfg.get("output_horizon", 1)),
            size_layer=int(cfg.get("size_layer", 64)),
            embedded_size=int(cfg.get("embedded_size", 64)),
            num_heads=int(cfg.get("num_heads", 4)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
            num_blocks=int(cfg.get("num_blocks", 2)),
            seed=int(cfg.get("seed", SEED)),
        )
        inst.config = cfg
        inst.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"PositionalEncoding": PositionalEncoding}
        )

        scaler_payload = np.load(scalers_path)
        inst.x_scaler.mean_ = scaler_payload["x_mean"]
        inst.x_scaler.scale_ = scaler_payload["x_scale"]
        inst.x_scaler.var_ = inst.x_scaler.scale_ ** 2
        inst.x_scaler.n_features_in_ = inst.x_scaler.mean_.shape[0]

        inst.y_scaler.mean_ = scaler_payload["y_mean"]
        inst.y_scaler.scale_ = scaler_payload["y_scale"]
        inst.y_scaler.var_ = inst.y_scaler.scale_ ** 2
        inst.y_scaler.n_features_in_ = inst.y_scaler.mean_.shape[0]
        return inst


DEFAULT_NPD_FEATURE_COLS = ["recency", "frequency", "avg_gap", "std_gap", "cv_gap", "last_gap"]


def build_customer_gap_sequences(
    tx_df,
    cust_col: str = "CustomerID",
    date_col: str = "InvoiceDate",
    min_purchase_days: int = 20,
    window_size: int = 5,
    forecast_horizon: int = 5,
):
    """Build customer purchase-gap sequences and clustering descriptors."""
    import pandas as pd

    tx_local = tx_df[[cust_col, date_col]].dropna().copy()
    tx_local[date_col] = pd.to_datetime(tx_local[date_col], errors="raise")
    tx_local = tx_local.sort_values([cust_col, date_col])

    purchase_day_counts = (
        tx_local.groupby([cust_col, tx_local[date_col].dt.normalize()])
        .size()
        .reset_index(name="n_lines")
        .groupby(cust_col)
        .size()
    )

    global_last_day = tx_local[date_col].max().normalize()
    sequences: Dict[str, np.ndarray] = {}
    meta_rows: List[Dict[str, Any]] = []

    for cust, group in tx_local.groupby(cust_col):
        days = pd.Series(group[date_col].dt.normalize().unique()).sort_values().reset_index(drop=True)
        if len(days) < int(min_purchase_days):
            continue

        gaps = days.diff().dropna().dt.days.astype(float).values
        if len(gaps) < int(window_size) + int(forecast_horizon):
            continue

        recency = float((global_last_day - days.iloc[-1]).days)
        frequency = int(len(days))
        avg_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        cv_gap = float(std_gap / (avg_gap + 1e-6))
        last_gap = float(gaps[-1])

        cust_id = str(cust)
        sequences[cust_id] = gaps
        meta_rows.append(
            {
                cust_col: cust_id,
                "purchase_days": int(len(days)),
                "n_gaps": int(len(gaps)),
                "recency": recency,
                "frequency": frequency,
                "avg_gap": avg_gap,
                "std_gap": std_gap,
                "cv_gap": cv_gap,
                "last_gap": last_gap,
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    if len(meta_df):
        meta_df = meta_df.sort_values(["frequency", "purchase_days"], ascending=False).reset_index(drop=True)
    return sequences, meta_df, purchase_day_counts


def split_customers(customer_meta, cust_col: str = "CustomerID", test_frac: float = 0.20, seed: int = 42):
    eligible_customers = customer_meta[cust_col].astype(str).tolist()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(eligible_customers)
    n_test = max(1, int(round(len(eligible_customers) * float(test_frac)))) if eligible_customers else 0
    test_customers = sorted(eligible_customers[:n_test])
    train_customers = sorted(eligible_customers[n_test:])
    return train_customers, test_customers


def compute_elbow_curve(feature_matrix: np.ndarray, k_values: Sequence[int] = range(2, 9), seed: int = 42):
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for k in list(k_values):
        km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10)
        km.fit(feature_matrix)
        rows.append({"k": int(k), "sse": float(km.inertia_)})
    return pd.DataFrame(rows)


def assign_customer_clusters(
    customer_meta,
    train_customers: Sequence[str],
    test_customers: Sequence[str],
    cust_col: str = "CustomerID",
    feature_cols: Optional[Sequence[str]] = None,
    n_clusters: int = 4,
    seed: int = 42,
):
    feature_cols = list(feature_cols or DEFAULT_NPD_FEATURE_COLS)
    meta = customer_meta.copy()
    meta[cust_col] = meta[cust_col].astype(str)

    scaler = StandardScaler()
    train_feature_matrix = scaler.fit_transform(meta.set_index(cust_col).loc[list(train_customers), feature_cols])

    kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init=10)
    kmeans.fit(train_feature_matrix)

    meta["cluster"] = -1
    meta.loc[meta[cust_col].isin(train_customers), "cluster"] = kmeans.labels_

    if len(test_customers):
        test_feature_matrix = scaler.transform(meta.set_index(cust_col).loc[list(test_customers), feature_cols])
        meta.loc[meta[cust_col].isin(test_customers), "cluster"] = kmeans.predict(test_feature_matrix)

    return meta, scaler, kmeans


def load_cluster_models(cluster_root: str) -> Dict[int, "NPDTransformerForecaster"]:
    cluster_root = Path(cluster_root)
    models: Dict[int, NPDTransformerForecaster] = {}
    if not cluster_root.exists():
        return models
    for child in sorted(cluster_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("cluster_"):
            continue
        try:
            cluster_id = int(child.name.split("_")[-1])
            models[cluster_id] = NPDTransformerForecaster.load(child)
        except Exception:
            continue
    return models
