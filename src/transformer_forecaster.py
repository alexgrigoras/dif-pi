import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Any
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Input
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

        hist = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
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
