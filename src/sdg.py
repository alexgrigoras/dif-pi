"""Synthetic demand generation utilities for DIF-PI."""

import gc
import inspect
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class SyntheticGenerationExample:
    source_text: str
    target_text: str
    scale: float
    sku: str
    context_start: int
    context_end: int
    prediction_length: int


class MeanScaleUniformQuantizer:
    def __init__(
        self,
        num_bins: int = 4094,
        value_range: Tuple[float, float] = (-5.0, 5.0),
        eps: float = 1e-8,
        use_special_tokens: bool = True,
        token_prefix: str = "ts",
        token_pad_width: int = 4,
    ):
        if int(num_bins) < 2:
            raise ValueError("num_bins must be at least 2.")
        left, right = float(value_range[0]), float(value_range[1])
        if right <= left:
            raise ValueError("value_range must satisfy right > left.")
        self.num_bins = int(num_bins)
        self.value_range = (left, right)
        self.eps = float(eps)
        self.use_special_tokens = bool(use_special_tokens)
        self.token_prefix = str(token_prefix)
        self.token_pad_width = int(token_pad_width)
        self.bin_centers = np.linspace(left, right, self.num_bins)
        self.bin_edges = (self.bin_centers[:-1] + self.bin_centers[1:]) / 2.0

    def compute_scale(self, context: Sequence[float]) -> float:
        arr = np.asarray(context, dtype=float).ravel()
        if arr.size == 0:
            return 1.0
        scale = float(np.mean(np.abs(arr)))
        return scale if scale > self.eps else 1.0

    def mean_scale(self, values: Sequence[float], scale: float) -> np.ndarray:
        arr = np.asarray(values, dtype=float).ravel()
        scale = float(scale) if float(scale) > self.eps else 1.0
        return arr / scale

    def inverse_mean_scale(self, scaled_values: Sequence[float], scale: float) -> np.ndarray:
        arr = np.asarray(scaled_values, dtype=float).ravel()
        scale = float(scale) if float(scale) > self.eps else 1.0
        return arr * scale

    def quantize(self, scaled_values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(scaled_values, dtype=float).ravel()
        token_ids = np.digitize(arr, self.bin_edges, right=False) + 1
        return np.clip(token_ids.astype(int), 1, self.num_bins)

    def dequantize(self, token_ids: Sequence[int]) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=int).ravel()
        ids = np.clip(ids, 1, self.num_bins)
        return self.bin_centers[ids - 1]

    def encode(self, context: Sequence[float], values: Sequence[float]) -> Tuple[np.ndarray, float]:
        scale = self.compute_scale(context)
        scaled = self.mean_scale(values, scale)
        tokens = self.quantize(scaled)
        return tokens, scale

    def decode(self, token_ids: Sequence[int], scale: float) -> np.ndarray:
        scaled = self.dequantize(token_ids)
        return self.inverse_mean_scale(scaled, scale)

    def _symbol(self, token_id: int) -> str:
        return f"<{self.token_prefix}_{int(token_id):0{self.token_pad_width}d}>"

    def vocabulary_tokens(self) -> List[str]:
        return [self._symbol(i) for i in range(1, self.num_bins + 1)]

    def tokens_to_text(self, token_ids: Sequence[int]) -> str:
        token_ids = np.asarray(token_ids).ravel()
        if self.use_special_tokens:
            return " ".join(self._symbol(int(x)) for x in token_ids)
        return " ".join(str(int(x)) for x in token_ids)

    def text_to_tokens(self, text: str) -> List[int]:
        if text is None:
            return []

        s = str(text)
        token_ids: List[int] = []

        if self.use_special_tokens:
            pat = rf"<{re.escape(self.token_prefix)}_(\d+)>"
            token_ids.extend(int(x) for x in re.findall(pat, s))

        if not token_ids:
            token_ids.extend(int(x) for x in re.findall(r"\d+", s))

        token_ids = [min(max(1, x), self.num_bins) for x in token_ids]
        return token_ids


def _acf(values: Sequence[float], lag: int) -> float:
    x = np.asarray(values, dtype=float).ravel()
    if x.size <= lag or lag <= 0:
        return 0.0
    x1 = x[:-lag]
    x2 = x[lag:]
    if np.std(x1) < 1e-8 or np.std(x2) < 1e-8:
        return 0.0
    return float(np.corrcoef(x1, x2)[0, 1])


def compute_difpi_sku_split(
    panel_df: pd.DataFrame,
    sku_col: str = "StockCode",
    time_col: str = "timestamp",
    target_col: str = "demand",
    eligibility_mode: str = "adaptive",
    min_history_days_strict: int = 365,
    min_nonzero_days_strict: int = 60,
    min_history_days_relaxed: int = 120,
    min_nonzero_days_relaxed: int = 30,
    transformer_seq_len: int = 30,
    horizon: int = 30,
    target_eligible_fraction: float = 0.8,
    sku_holdout_enabled: bool = True,
    test_sku_frac: float = 0.20,
    split_seed: int = 42,
    case_sku_override: Optional[str] = None,
) -> Dict[str, Any]:
    use_df = panel_df.copy()
    use_df[sku_col] = use_df[sku_col].astype(str)
    use_df[time_col] = pd.to_datetime(use_df[time_col], errors="raise")

    sku_stats = []
    for sku, g in use_df.groupby(sku_col):
        g = g.sort_values(time_col)
        history_days = len(pd.Index(g[time_col]).unique())
        nonzero_days = int((g[target_col].astype(float) > 0).sum())
        sku_stats.append(
            {
                sku_col: str(sku),
                "history_days": int(history_days),
                "nonzero_days": int(nonzero_days),
            }
        )
    sku_stats = pd.DataFrame(sku_stats).sort_values(["nonzero_days", "history_days"], ascending=False)

    def _eligible(stats: pd.DataFrame, min_hist: int, min_nz: int) -> pd.DataFrame:
        return stats[(stats["history_days"] >= int(min_hist)) & (stats["nonzero_days"] >= int(min_nz))].copy()

    n_total = len(sku_stats)
    min_hist, min_nz = int(min_history_days_strict), int(min_nonzero_days_strict)

    if eligibility_mode == "strict":
        eligible = _eligible(sku_stats, min_hist, min_nz)
    elif eligibility_mode == "relaxed":
        min_hist, min_nz = int(min_history_days_relaxed), int(min_nonzero_days_relaxed)
        eligible = _eligible(sku_stats, min_hist, min_nz)
    elif eligibility_mode == "adaptive":
        steps = [
            (int(min_history_days_strict), int(min_nonzero_days_strict)),
            (270, 45),
            (180, 30),
            (120, 25),
            (90, 20),
            (60, 15),
        ]
        min_hist_floor = max(2 * int(transformer_seq_len), int(horizon) + 30)
        min_nz_floor = max(int(horizon), 10)
        chosen = None
        for h, nz in steps:
            h2 = max(int(h), int(min_hist_floor))
            nz2 = max(int(nz), int(min_nz_floor))
            cand = _eligible(sku_stats, h2, nz2)
            if len(cand) / max(n_total, 1) >= float(target_eligible_fraction):
                chosen = (h2, nz2, cand)
                break
        if chosen is None:
            h, nz = steps[-1]
            h2 = max(int(h), int(min_hist_floor))
            nz2 = max(int(nz), int(min_nz_floor))
            chosen = (h2, nz2, _eligible(sku_stats, h2, nz2))
        min_hist, min_nz, eligible = chosen
    else:
        raise ValueError(f"Unknown eligibility_mode: {eligibility_mode!r}")

    if len(eligible) == 0:
        raise ValueError("No SKU meets the eligibility thresholds.")

    eligible_skus = eligible[sku_col].astype(str).tolist()
    if sku_holdout_enabled and len(eligible_skus) >= 2:
        rng = np.random.default_rng(int(split_seed))
        shuffled = list(eligible_skus)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * float(test_sku_frac))))
        test_skus = shuffled[:n_test]
        train_skus = shuffled[n_test:]
    else:
        train_skus = eligible_skus
        test_skus = eligible_skus

    if case_sku_override not in (None, "", "None"):
        case_sku = str(case_sku_override)
    else:
        case_index = 3 if len(test_skus) > 3 else 0
        case_sku = str(test_skus[case_index])

    return {
        "sku_stats": sku_stats.reset_index(drop=True),
        "eligible": eligible.reset_index(drop=True),
        "train_skus": train_skus,
        "test_skus": test_skus,
        "case_sku": case_sku,
        "min_history_days": int(min_hist),
        "min_nonzero_days": int(min_nz),
    }


class LLMSyntheticTimeSeriesGenerator:
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        context_length: int = 140,
        prediction_length: int = 30,
        num_bins: int = 4094,
        value_range: Tuple[float, float] = (-5.0, 5.0),
        learning_rate: float = 2.0e-4,
        train_steps: int = 1500,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        max_source_length: int = 768,
        max_target_length: int = 256,
        random_state: int = 42,
        task_prefix: str = "generate synthetic retail demand future from historical context",
        seasonality_strength: float = 0.60,
        seasonal_period: int = 7,
        seasonal_fallback_strength: float = 0.25,
        zero_threshold_for_sparsity: float = 0.60,
        prefer_backend: str = "lora",
        use_special_tokens: bool = True,
        add_calendar_features: bool = True,
        warmup_ratio: float = 0.05,
        weight_decay: float = 0.01,
        privacy_reference_max_windows: int = 2000,
        privacy_min_distance_quantile: float = 0.25,
        privacy_distance_penalty: float = 4.0,
        privacy_noise_strength: float = 0.10,
        privacy_baseline_blend: float = 0.25,
        privacy_training_jitter_prob: float = 0.60,
        privacy_training_jitter_strength: float = 0.10,
        privacy_deduplicate_examples: bool = True,
        privacy_filter_enabled: bool = True,
        privacy_filter_max_retries: int = 6,
    ):
        self.model_name = str(model_name)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.learning_rate = float(learning_rate)
        self.train_steps = int(train_steps)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.batch_size = int(batch_size)
        self.gradient_accumulation_steps = int(max(1, gradient_accumulation_steps))
        self.max_source_length = int(max_source_length)
        self.max_target_length = int(max_target_length)
        self.random_state = int(random_state)
        self.task_prefix = str(task_prefix)
        self.seasonality_strength = float(seasonality_strength)
        self.seasonal_period = int(max(2, seasonal_period))
        self.seasonal_fallback_strength = float(np.clip(seasonal_fallback_strength, 0.0, 1.0))
        self.zero_threshold_for_sparsity = float(np.clip(zero_threshold_for_sparsity, 0.0, 1.0))
        self.prefer_backend = str(prefer_backend).lower()
        self.add_calendar_features = bool(add_calendar_features)
        self.warmup_ratio = float(max(0.0, warmup_ratio))
        self.weight_decay = float(max(0.0, weight_decay))
        self.privacy_reference_max_windows = int(max(128, privacy_reference_max_windows))
        self.privacy_min_distance_quantile = float(np.clip(privacy_min_distance_quantile, 0.01, 0.50))
        self.privacy_distance_penalty = float(max(0.0, privacy_distance_penalty))
        self.privacy_noise_strength = float(max(0.0, privacy_noise_strength))
        self.privacy_baseline_blend = float(np.clip(privacy_baseline_blend, 0.0, 0.75))
        self.privacy_training_jitter_prob = float(np.clip(privacy_training_jitter_prob, 0.0, 1.0))
        self.privacy_training_jitter_strength = float(max(0.0, privacy_training_jitter_strength))
        self.privacy_deduplicate_examples = bool(privacy_deduplicate_examples)
        self.privacy_filter_enabled = bool(privacy_filter_enabled)
        self.privacy_filter_max_retries = int(max(0, privacy_filter_max_retries))

        self.quantizer = MeanScaleUniformQuantizer(
            num_bins=int(num_bins),
            value_range=value_range,
            use_special_tokens=bool(use_special_tokens),
        )
        self.model: Any = None
        self.tokenizer: Any = None
        self.training_info: Dict[str, Any] = {}
        self.is_peft_model: bool = False
        self.backend_name: str = "unknown"
        self.added_special_tokens: int = 0
        self.privacy_reference_bank: Optional[np.ndarray] = None
        self.privacy_reference_stats: Dict[str, Any] = {}
        self._privacy_nbrs: Any = None

        self.base_model_id = self.model_name if self._is_hf_model_id(self.model_name) else None

        self.config: Dict[str, Any] = {
            "model_name": self.model_name,
            "base_model_id": self.base_model_id,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "num_bins": self.quantizer.num_bins,
            "value_range": list(self.quantizer.value_range),
            "learning_rate": self.learning_rate,
            "train_steps": self.train_steps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "random_state": self.random_state,
            "task_prefix": self.task_prefix,
            "seasonality_strength": self.seasonality_strength,
            "seasonal_period": self.seasonal_period,
            "seasonal_fallback_strength": self.seasonal_fallback_strength,
            "zero_threshold_for_sparsity": self.zero_threshold_for_sparsity,
            "prefer_backend": self.prefer_backend,
            "use_special_tokens": self.quantizer.use_special_tokens,
            "add_calendar_features": self.add_calendar_features,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "privacy_reference_max_windows": self.privacy_reference_max_windows,
            "privacy_min_distance_quantile": self.privacy_min_distance_quantile,
            "privacy_distance_penalty": self.privacy_distance_penalty,
            "privacy_noise_strength": self.privacy_noise_strength,
            "privacy_baseline_blend": self.privacy_baseline_blend,
            "privacy_training_jitter_prob": self.privacy_training_jitter_prob,
            "privacy_training_jitter_strength": self.privacy_training_jitter_strength,
            "privacy_deduplicate_examples": self.privacy_deduplicate_examples,
            "privacy_filter_enabled": self.privacy_filter_enabled,
            "privacy_filter_max_retries": self.privacy_filter_max_retries,
        }

    @staticmethod
    def _is_hf_model_id(value: Optional[str]) -> bool:
        if value in (None, "", "None"):
            return False
        s = str(value)
        if "\\" in s:
            s = s.replace("\\", "/")
        if s.startswith("/") or s.startswith("./") or s.startswith("../"):
            return False
        if ":" in s and not s.startswith("http"):
            return False
        if Path(s).exists():
            return False
        return "/" in s and len(s.split("/", 1)[0]) > 0 and len(s.split("/", 1)[1]) > 0

    def _resolve_base_model_id(self) -> Optional[str]:
        candidates = [
            self.config.get("base_model_id") if isinstance(getattr(self, "config", None), dict) else None,
            getattr(self, "base_model_id", None),
            getattr(self, "model_name", None),
            self.config.get("model_name") if isinstance(getattr(self, "config", None), dict) else None,
            getattr(getattr(self, "model", None), "name_or_path", None),
        ]
        for candidate in candidates:
            if self._is_hf_model_id(candidate):
                return str(candidate)
        return None

    def _calendar_special_tokens(self) -> List[str]:
        toks = []
        toks.extend([f"<dow_{i}>" for i in range(7)])
        toks.extend([f"<month_{i}>" for i in range(1, 13)])
        toks.extend([f"<zero_bin_{i}>" for i in range(6)])
        toks.extend([f"<level_bin_{i}>" for i in range(6)])
        toks.extend([f"<acf7_bin_{i}>" for i in range(6)])
        return toks

    def _register_special_tokens(self, tokenizer: Any) -> int:
        tokens = []
        if self.quantizer.use_special_tokens:
            tokens.extend(self.quantizer.vocabulary_tokens())
        if self.add_calendar_features:
            tokens.extend(self._calendar_special_tokens())
        if not tokens:
            return 0
        vocab = set(getattr(tokenizer, "get_vocab", lambda: {})().keys())
        missing = [t for t in tokens if t not in vocab]
        if missing:
            added = tokenizer.add_tokens(missing, special_tokens=True)
            return int(added or 0)
        return 0

    @staticmethod
    def _discover_target_modules(model: Any) -> List[str]:
        preferred_suffixes = ("q", "k", "v", "o", "wi", "wi_0", "wi_1", "wo")
        found = []
        for name, module in model.named_modules():
            short = name.split(".")[-1]
            if short in preferred_suffixes:
                found.append(short)
        found = sorted(set(found))
        if not found:
            return ["q", "k", "v", "o"]
        return found

    @staticmethod
    def _bin_token(value: float, edges: Sequence[float], token_prefix: str) -> str:
        idx = int(np.digitize([float(value)], np.asarray(edges, dtype=float))[0])
        idx = int(np.clip(idx, 0, len(edges)))
        return f"<{token_prefix}_{idx}>"

    def _metadata_tokens(
        self,
        context_values: Sequence[float],
        context_dates: Optional[Sequence[Any]],
        future_dates: Optional[Sequence[Any]],
    ) -> Dict[str, str]:
        if not self.add_calendar_features:
            return {}

        ctx = np.asarray(context_values, dtype=float).ravel()
        recent = ctx[-min(len(ctx), 28):] if len(ctx) else ctx
        zero_share = float(np.mean(recent == 0)) if len(recent) else 0.0
        mean_level = float(np.mean(recent)) if len(recent) else 0.0
        acf7 = float(_acf(recent, min(self.seasonal_period, max(1, len(recent) - 1)))) if len(recent) > 2 else 0.0

        if context_dates is not None and len(context_dates):
            ctx_dates = pd.to_datetime(pd.Series(list(context_dates)), errors="coerce")
            ctx_end_dow = int(ctx_dates.iloc[-1].dayofweek)
            ctx_month = int(ctx_dates.iloc[-1].month)
        else:
            ctx_end_dow = 0
            ctx_month = 1

        if future_dates is not None and len(future_dates):
            fut_dates = pd.to_datetime(pd.Series(list(future_dates)), errors="coerce")
            future_start_dow = int(fut_dates.iloc[0].dayofweek)
        else:
            future_start_dow = int((ctx_end_dow + 1) % 7)

        zero_tok = self._bin_token(zero_share, np.linspace(0.1, 0.9, 5), "zero_bin")
        level_tok = self._bin_token(np.log1p(max(mean_level, 0.0)), np.linspace(0.3, 3.0, 5), "level_bin")
        acf_tok = self._bin_token((acf7 + 1.0) / 2.0, np.linspace(0.2, 0.8, 5), "acf7_bin")

        return {
            "ctx_end_dow": f"<dow_{ctx_end_dow}>",
            "future_start_dow": f"<dow_{future_start_dow}>",
            "ctx_month": f"<month_{ctx_month}>",
            "zero_share": zero_tok,
            "level": level_tok,
            "acf7": acf_tok,
        }

    def _make_source_text(
        self,
        context_token_ids: Sequence[int],
        horizon: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        horizon = int(self.prediction_length if horizon is None else horizon)
        context_text = self.quantizer.tokens_to_text(context_token_ids)
        chunks = [self.task_prefix, f"horizon={horizon}"]
        if metadata:
            for key, value in metadata.items():
                if str(key).lower() == "sku":
                    continue
                chunks.append(f"{key}={value}")
        chunks.append(f"context: {context_text}")
        return " | ".join(chunks)

    def _make_target_text(self, future_token_ids: Sequence[int]) -> str:
        return self.quantizer.tokens_to_text(future_token_ids)

    def _decode_target_window(self, target_text: str, scale: float, horizon: Optional[int] = None) -> np.ndarray:
        horizon = int(self.prediction_length if horizon is None else horizon)
        token_ids = self.quantizer.text_to_tokens(target_text)
        if not token_ids:
            return np.zeros(horizon, dtype=float)
        token_ids = token_ids[:horizon]
        if len(token_ids) < horizon:
            token_ids = token_ids + [token_ids[-1]] * (horizon - len(token_ids))
        values = self.quantizer.decode(token_ids, scale)
        return np.maximum(0.0, np.asarray(values, dtype=float))

    def _encode_target_window_with_scale(self, values: Sequence[float], scale: float) -> str:
        arr = np.maximum(0.0, np.asarray(values, dtype=float).ravel())
        scaled = self.quantizer.mean_scale(arr, scale)
        token_ids = self.quantizer.quantize(scaled)
        return self.quantizer.tokens_to_text(token_ids)

    def _fit_privacy_index(self) -> None:
        self._privacy_nbrs = None
        bank = self.privacy_reference_bank
        if bank is None:
            return
        bank = np.asarray(bank, dtype=np.float32)
        if bank.ndim != 2 or len(bank) == 0:
            self.privacy_reference_bank = None
            return

        self.privacy_reference_bank = bank
        if len(bank) < 2:
            self.privacy_reference_stats = {
                "n_reference_windows": int(len(bank)),
                "min_distance_threshold": np.nan,
                "avg_real_nn_distance": np.nan,
                "distance_quantile": float(self.privacy_min_distance_quantile),
            }
            return

        nbrs = NearestNeighbors(n_neighbors=min(2, len(bank)), metric="euclidean")
        nbrs.fit(bank)
        dists, _ = nbrs.kneighbors(bank)
        real_nn = dists[:, 1] if dists.shape[1] > 1 else np.full(len(bank), np.nan, dtype=float)
        real_nn = real_nn[np.isfinite(real_nn)]
        threshold = float(np.quantile(real_nn, self.privacy_min_distance_quantile)) if real_nn.size else np.nan

        self._privacy_nbrs = nbrs
        self.privacy_reference_stats = {
            "n_reference_windows": int(len(bank)),
            "min_distance_threshold": threshold,
            "avg_real_nn_distance": float(np.mean(real_nn)) if real_nn.size else np.nan,
            "distance_quantile": float(self.privacy_min_distance_quantile),
        }

    def _prepare_privacy_reference_bank(self, frame: pd.DataFrame) -> Dict[str, Any]:
        if frame is None or len(frame) == 0 or "target_text" not in frame.columns or "scale" not in frame.columns:
            self.privacy_reference_bank = None
            self.privacy_reference_stats = {}
            self._privacy_nbrs = None
            return {"privacy_reference_windows": 0}

        ref_df = frame.copy()
        if self.privacy_deduplicate_examples and {"source_text", "target_text"}.issubset(ref_df.columns):
            ref_df = ref_df.drop_duplicates(subset=["source_text", "target_text"]).reset_index(drop=True)

        windows: List[np.ndarray] = []
        for row in ref_df[["target_text", "scale"]].itertuples(index=False):
            try:
                win = self._decode_target_window(row.target_text, float(row.scale))
                if win.size == self.prediction_length and np.all(np.isfinite(win)):
                    windows.append(np.asarray(win, dtype=np.float32))
            except Exception:
                continue

        if not windows:
            self.privacy_reference_bank = None
            self.privacy_reference_stats = {}
            self._privacy_nbrs = None
            return {"privacy_reference_windows": 0}

        bank = np.asarray(windows, dtype=np.float32)
        bank = np.unique(np.round(bank, 4), axis=0)

        if len(bank) > self.privacy_reference_max_windows:
            rng = np.random.default_rng(self.random_state)
            keep = rng.choice(len(bank), size=self.privacy_reference_max_windows, replace=False)
            bank = bank[np.sort(keep)]

        self.privacy_reference_bank = bank.astype(np.float32, copy=False)
        self._fit_privacy_index()
        out = dict(self.privacy_reference_stats)
        out["privacy_reference_windows"] = int(len(bank))
        return out

    def _augment_training_dataframe_for_privacy(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if train_df is None or len(train_df) == 0:
            return train_df, {"n_deduplicated": 0, "n_jittered": 0}

        out = train_df.copy()
        n_before = len(out)
        if self.privacy_deduplicate_examples and {"source_text", "target_text"}.issubset(out.columns):
            out = out.drop_duplicates(subset=["source_text", "target_text"]).reset_index(drop=True)
        n_deduplicated = int(max(0, n_before - len(out)))

        if (
            self.privacy_training_jitter_prob <= 0.0
            or self.privacy_training_jitter_strength <= 0.0
            or "target_text" not in out.columns
            or "scale" not in out.columns
        ):
            return out, {"n_deduplicated": n_deduplicated, "n_jittered": 0}

        rng = np.random.default_rng(self.random_state)
        mask = rng.random(len(out)) < float(self.privacy_training_jitter_prob)
        n_jittered = 0
        for idx in np.where(mask)[0]:
            row = out.iloc[int(idx)]
            try:
                target = self._decode_target_window(row["target_text"], float(row["scale"]))
                local_scale = float(np.std(target))
                if not np.isfinite(local_scale) or local_scale < 1e-8:
                    local_scale = float(np.mean(target[target > 0])) if np.any(target > 0) else 1.0
                noise = rng.normal(0.0, self.privacy_training_jitter_strength * max(local_scale, 1.0), size=target.shape)
                mult = 1.0 + rng.normal(0.0, self.privacy_training_jitter_strength * 0.35, size=target.shape)
                jittered = np.maximum(0.0, target * mult + noise)

                zero_share = float(np.mean(target == 0))
                if zero_share > 0.10:
                    q = float(np.quantile(jittered, min(max(zero_share, 0.0), 0.50)))
                    jittered = np.where(jittered <= q, 0.0, jittered)

                out.at[out.index[int(idx)], "target_text"] = self._encode_target_window_with_scale(jittered, float(row["scale"]))
                n_jittered += 1
            except Exception:
                continue

        return out.reset_index(drop=True), {"n_deduplicated": n_deduplicated, "n_jittered": int(n_jittered)}

    def _privacy_candidate_distance(self, candidate: Sequence[float]) -> float:
        cand = np.asarray(candidate, dtype=float).ravel()
        if self._privacy_nbrs is None or self.privacy_reference_bank is None or cand.size == 0:
            return np.nan
        if cand.size != int(self.privacy_reference_bank.shape[1]):
            return np.nan
        try:
            dist, _ = self._privacy_nbrs.kneighbors(cand.reshape(1, -1), n_neighbors=1)
            return float(dist[0, 0])
        except Exception:
            return np.nan

    def _privacy_penalty(self, candidate: Sequence[float]) -> Tuple[float, float]:
        dist = self._privacy_candidate_distance(candidate)
        threshold = float(self.privacy_reference_stats.get("min_distance_threshold", np.nan))
        if (not self.privacy_filter_enabled) or (not np.isfinite(dist)) or (not np.isfinite(threshold)) or threshold <= 1e-8:
            return 0.0, dist
        if dist >= threshold:
            return 0.0, dist
        penalty = self.privacy_distance_penalty * float((threshold - dist) / (threshold + 1e-8))
        return penalty, dist

    def _repair_candidate_for_privacy(
        self,
        candidate: Sequence[float],
        baseline: Sequence[float],
        context_values: Sequence[float],
        context_dates: Sequence[Any],
        future_dates: Sequence[Any],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        cand = np.maximum(0.0, np.asarray(candidate, dtype=float).ravel())
        best = cand.copy()
        best_dist = self._privacy_candidate_distance(best)
        threshold = float(self.privacy_reference_stats.get("min_distance_threshold", np.nan))

        if (not self.privacy_filter_enabled) or (not np.isfinite(best_dist)) or (not np.isfinite(threshold)) or best_dist >= threshold:
            return best, {"privacy_min_distance": best_dist, "privacy_threshold": threshold, "privacy_repaired": 0.0}

        rng = np.random.default_rng(self.random_state + int(np.sum(best)) + len(best))
        recent = np.asarray(context_values, dtype=float).ravel()
        recent = recent[-min(len(recent), 28):] if recent.size else best
        local_scale = float(np.std(recent))
        if not np.isfinite(local_scale) or local_scale < 1e-8:
            local_scale = float(np.mean(recent[recent > 0])) if np.any(recent > 0) else 1.0

        for _ in range(self.privacy_filter_max_retries):
            noise = rng.normal(0.0, self.privacy_noise_strength * max(local_scale, 1.0), size=best.shape)
            mult = 1.0 + rng.normal(0.0, self.privacy_noise_strength * 0.35, size=best.shape)
            trial = np.maximum(0.0, best * mult + noise)
            trial = (1.0 - self.privacy_baseline_blend) * trial + self.privacy_baseline_blend * np.asarray(baseline, dtype=float).ravel()
            trial = np.maximum(0.0, trial)
            trial = self.seasonality_aware_calibration(
                generated_future=trial,
                context_values=context_values,
                context_dates=context_dates,
                future_dates=future_dates,
                strength=min(1.0, self.seasonality_strength),
            )
            trial_dist = self._privacy_candidate_distance(trial)
            if not np.isfinite(trial_dist):
                continue
            if trial_dist > best_dist:
                best = trial
                best_dist = trial_dist
            if trial_dist >= threshold:
                break

        return best, {"privacy_min_distance": best_dist, "privacy_threshold": threshold, "privacy_repaired": 1.0}
    def make_examples_from_series(
        self,
        series: Sequence[float],
        sku: str = "series",
        stride: int = 1,
        prediction_length: Optional[int] = None,
        metadata_sequence: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[SyntheticGenerationExample]:
        values = np.asarray(series, dtype=float).ravel()
        horizon = int(self.prediction_length if prediction_length is None else prediction_length)
        if values.size < self.context_length + horizon:
            return []

        examples: List[SyntheticGenerationExample] = []
        metadata_sequence = metadata_sequence or []
        for ex_idx, start in enumerate(range(0, values.size - self.context_length - horizon + 1, int(stride))):
            context = values[start : start + self.context_length]
            future = values[start + self.context_length : start + self.context_length + horizon]
            scale = self.quantizer.compute_scale(context)
            ctx_tokens = self.quantizer.quantize(self.quantizer.mean_scale(context, scale))
            fut_tokens = self.quantizer.quantize(self.quantizer.mean_scale(future, scale))
            metadata = metadata_sequence[ex_idx] if ex_idx < len(metadata_sequence) else None
            examples.append(
                SyntheticGenerationExample(
                    source_text=self._make_source_text(ctx_tokens, horizon=horizon, metadata=metadata),
                    target_text=self._make_target_text(fut_tokens),
                    scale=float(scale),
                    sku=str(sku),
                    context_start=int(start),
                    context_end=int(start + self.context_length),
                    prediction_length=int(horizon),
                )
            )
        return examples

    def build_training_dataframe(
        self,
        panel_df: pd.DataFrame,
        sku_col: str,
        time_col: str,
        target_col: str,
        train_skus: Optional[Iterable[Any]] = None,
        per_sku_train_frac: float = 1.0,
        min_points: Optional[int] = None,
        stride: int = 1,
        include_metadata: bool = True,
        cutoff_map: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if panel_df.empty:
            raise ValueError("panel_df is empty.")
        use_df = panel_df.copy()
        use_df[sku_col] = use_df[sku_col].astype(str)
        use_df[time_col] = pd.to_datetime(use_df[time_col], errors="raise")
        use_df = use_df.sort_values([sku_col, time_col])

        if train_skus is not None:
            keep = {str(x) for x in train_skus}
            use_df = use_df[use_df[sku_col].isin(keep)].copy()

        min_points = int(min_points or (self.context_length + self.prediction_length + 1))
        rows: List[Dict[str, Any]] = []
        for sku, g in use_df.groupby(sku_col):
            g = g.sort_values(time_col).copy()

            if cutoff_map is not None and str(sku) in cutoff_map:
                cutoff = pd.to_datetime(cutoff_map[str(sku)])
                g = g[g[time_col] <= cutoff].copy()

            y = g[target_col].astype(float).values
            dates = pd.to_datetime(g[time_col], errors="coerce").tolist()
            n_train = int(math.floor(len(y) * float(per_sku_train_frac)))
            n_train = max(n_train, min_points)
            y_tr = y[: min(len(y), n_train)]
            d_tr = dates[: min(len(dates), n_train)]

            if len(y_tr) < min_points:
                continue

            metadata_sequence: List[Optional[Dict[str, Any]]] = []
            if include_metadata:
                for start in range(0, len(y_tr) - self.context_length - self.prediction_length + 1, int(stride)):
                    context = y_tr[start : start + self.context_length]
                    context_dates = d_tr[start : start + self.context_length]
                    future_dates = d_tr[start + self.context_length : start + self.context_length + self.prediction_length]
                    metadata_sequence.append(self._metadata_tokens(context, context_dates, future_dates))

            examples = self.make_examples_from_series(
                y_tr,
                sku=str(sku),
                stride=int(stride),
                prediction_length=self.prediction_length,
                metadata_sequence=metadata_sequence,
            )
            for ex in examples:
                rows.append(
                    {
                        "sku": ex.sku,
                        "context_start": ex.context_start,
                        "context_end": ex.context_end,
                        "prediction_length": ex.prediction_length,
                        "scale": ex.scale,
                        "source_text": ex.source_text,
                        "target_text": ex.target_text,
                    }
                )
        if not rows:
            raise ValueError("No SDG training examples were created.")
        return pd.DataFrame(rows)

    @staticmethod
    def _require_hf_stack() -> Dict[str, Any]:
        try:
            from datasets import Dataset
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                DataCollatorForSeq2Seq,
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
            )
        except Exception as exc:
            raise ImportError(
                "Hugging Face dependencies are not installed. Install transformers, datasets, sentencepiece, torch."
            ) from exc
        return {
            "Dataset": Dataset,
            "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
            "AutoTokenizer": AutoTokenizer,
            "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
            "Seq2SeqTrainer": Seq2SeqTrainer,
            "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
        }

    def _build_model(self) -> None:
        hf = self._require_hf_stack()
        AutoTokenizer = hf["AutoTokenizer"]
        AutoModelForSeq2SeqLM = hf["AutoModelForSeq2SeqLM"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.added_special_tokens = self._register_special_tokens(self.tokenizer)
        self.is_peft_model = False
        self.backend_name = "seq2seq"

        # Preferred route: QLoRA
        if self.prefer_backend == "qlora":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("QLoRA requires CUDA in this setup.")
                import bitsandbytes  # noqa: F401
                from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
                from transformers import BitsAndBytesConfig

                compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    quantization_config=quant_cfg,
                    device_map="auto",
                )
                if self.added_special_tokens > 0:
                    base_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
                if hasattr(base_model, "config"):
                    setattr(base_model.config, "use_cache", False)
                base_model = prepare_model_for_kbit_training(base_model)
                target_modules = self._discover_target_modules(base_model)
                peft_cfg = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    target_modules=target_modules,
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    lora_dropout=0.10,
                    bias="none",
                )
                self.model = get_peft_model(base_model, peft_cfg)
                self.is_peft_model = True
                self.backend_name = "qlora"
                return
            except Exception:
                pass

        # Second route: classic LoRA without quantization.
        try:
            from peft import LoraConfig, TaskType, get_peft_model

            base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            if self.added_special_tokens > 0:
                base_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            if hasattr(base_model, "config"):
                setattr(base_model.config, "use_cache", False)
            target_modules = self._discover_target_modules(base_model)
            peft_cfg = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules,
                task_type=TaskType.SEQ_2_SEQ_LM,
                lora_dropout=0.10,
                bias="none",
            )
            self.model = get_peft_model(base_model, peft_cfg)
            self.is_peft_model = True
            self.backend_name = "lora"
        except Exception:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            if self.added_special_tokens > 0:
                self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
            self.is_peft_model = False
            self.backend_name = "seq2seq"

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[str] = None,
        logging_steps: int = 25,
        save_steps: int = 250,
    ) -> Dict[str, Any]:
        if train_df.empty:
            raise ValueError("train_df is empty.")
        hf = self._require_hf_stack()
        Dataset = hf["Dataset"]
        DataCollatorForSeq2Seq = hf["DataCollatorForSeq2Seq"]
        Seq2SeqTrainer = hf["Seq2SeqTrainer"]
        Seq2SeqTrainingArguments = hf["Seq2SeqTrainingArguments"]

        if self.model is None or self.tokenizer is None:
            self._build_model()

        privacy_ref_info = self._prepare_privacy_reference_bank(train_df)
        train_df_use, privacy_aug_info = self._augment_training_dataframe_for_privacy(train_df)

        train_ds = Dataset.from_pandas(train_df_use[["source_text", "target_text"]].reset_index(drop=True))
        eval_ds = None
        if eval_df is not None and not eval_df.empty:
            eval_ds = Dataset.from_pandas(eval_df[["source_text", "target_text"]].reset_index(drop=True))

        tokenizer = self.tokenizer
        model = self.model

        def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
            model_inputs = tokenizer(
                batch["source_text"],
                max_length=self.max_source_length,
                truncation=True,
                padding=False,
            )
            labels = tokenizer(
                text_target=batch["target_text"],
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        remove_cols = train_ds.column_names
        train_tok = train_ds.map(_tokenize, batched=True, remove_columns=remove_cols)
        eval_tok = eval_ds.map(_tokenize, batched=True, remove_columns=eval_ds.column_names) if eval_ds is not None else None

        args_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
        trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)

        run_dir = str(Path(output_dir or "./artifacts/models/sdg_t5_qlora").resolve())
        args_kwargs = dict(
            output_dir=run_dir,
            learning_rate=self.learning_rate,
            max_steps=self.train_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            predict_with_generate=False,
            logging_steps=int(max(logging_steps, 1)),
            save_steps=int(max(save_steps, 1)),
            report_to=[],
            seed=self.random_state,
            remove_unused_columns=True,
            fp16=False,
        )

        if "gradient_accumulation_steps" in args_sig.parameters:
            args_kwargs["gradient_accumulation_steps"] = self.gradient_accumulation_steps
        if "warmup_ratio" in args_sig.parameters:
            args_kwargs["warmup_ratio"] = self.warmup_ratio
        if "weight_decay" in args_sig.parameters:
            args_kwargs["weight_decay"] = self.weight_decay
        if "lr_scheduler_type" in args_sig.parameters:
            args_kwargs["lr_scheduler_type"] = "cosine"
        if "save_total_limit" in args_sig.parameters:
            args_kwargs["save_total_limit"] = 2
        if "group_by_length" in args_sig.parameters:
            args_kwargs["group_by_length"] = True
        if "optim" in args_sig.parameters and self.backend_name == "qlora":
            args_kwargs["optim"] = "paged_adamw_8bit"
        if "gradient_checkpointing" in args_sig.parameters:
            args_kwargs["gradient_checkpointing"] = bool(self.is_peft_model)

        if "evaluation_strategy" in args_sig.parameters:
            args_kwargs["evaluation_strategy"] = "steps" if eval_tok is not None else "no"
            if eval_tok is not None:
                args_kwargs["eval_steps"] = int(max(logging_steps, 1))
        elif "eval_strategy" in args_sig.parameters:
            args_kwargs["eval_strategy"] = "steps" if eval_tok is not None else "no"
            if eval_tok is not None:
                args_kwargs["eval_steps"] = int(max(logging_steps, 1))

        if eval_tok is not None and "load_best_model_at_end" in args_sig.parameters:
            args_kwargs["load_best_model_at_end"] = True
            if "metric_for_best_model" in args_sig.parameters:
                args_kwargs["metric_for_best_model"] = "eval_loss"
            if "greater_is_better" in args_sig.parameters:
                args_kwargs["greater_is_better"] = False

        args = Seq2SeqTrainingArguments(**args_kwargs)

        trainer_kwargs = dict(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        )
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Seq2SeqTrainer(**trainer_kwargs)
        train_out = trainer.train()

        metrics = dict(getattr(train_out, "metrics", {}) or {})
        self.training_info = {
            "train_examples": int(len(train_df_use)),
            "eval_examples": int(0 if eval_df is None else len(eval_df)),
            "train_steps": int(self.train_steps),
            "learning_rate": float(self.learning_rate),
            "train_runtime": float(metrics.get("train_runtime", 0.0)),
            "train_loss": float(metrics.get("train_loss", np.nan)),
            "is_peft_model": bool(self.is_peft_model),
            "backend_name": self.backend_name,
            "added_special_tokens": int(self.added_special_tokens),
            "privacy_reference_windows": int(privacy_ref_info.get("privacy_reference_windows", 0)),
            "privacy_deduplicated_examples": int(privacy_aug_info.get("n_deduplicated", 0)),
            "privacy_jittered_examples": int(privacy_aug_info.get("n_jittered", 0)),
        }
        if self.privacy_reference_stats:
            self.training_info["privacy_reference_stats"] = dict(self.privacy_reference_stats)
        return self.training_info

    def _ensure_model_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded. Call load() or fit() first.")

    @staticmethod
    def compute_weekday_factors(
        context_values: Sequence[float],
        context_dates: Sequence[Any],
        eps: float = 1e-8,
        clip_range: Tuple[float, float] = (0.40, 2.25),
    ) -> np.ndarray:
        values = np.asarray(context_values, dtype=float).ravel()
        dates = pd.to_datetime(pd.Series(list(context_dates)), errors="coerce")
        if len(values) != len(dates) or len(values) == 0:
            return np.ones(7, dtype=float)
        overall = float(np.mean(values))
        overall = overall if abs(overall) > eps else 1.0
        df = pd.DataFrame({"date": dates, "y": values})
        dow_mean = df.groupby(df["date"].dt.dayofweek)["y"].mean()
        factors = np.ones(7, dtype=float)
        for dow in range(7):
            if dow in dow_mean.index:
                factors[dow] = float(dow_mean.loc[dow]) / overall
        factors = np.clip(factors, clip_range[0], clip_range[1])
        return factors

    @staticmethod
    def _future_dow_weights(future_dates: Sequence[Any], factors: Sequence[float]) -> np.ndarray:
        dates = pd.to_datetime(pd.Series(list(future_dates)), errors="coerce")
        dow = dates.dt.dayofweek.to_numpy()
        f = np.asarray(factors, dtype=float).ravel()
        weights = np.array([f[d] if 0 <= int(d) < len(f) else 1.0 for d in dow], dtype=float)
        weights = np.where(np.isfinite(weights), weights, 1.0)
        mean_w = float(np.mean(weights)) if len(weights) else 1.0
        return weights / (mean_w if mean_w > 1e-8 else 1.0)

    def seasonal_naive_baseline(
        self,
        context_values: Sequence[float],
        horizon: int,
        context_dates: Optional[Sequence[Any]] = None,
        future_dates: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        ctx = np.asarray(context_values, dtype=float).ravel()
        horizon = int(max(1, horizon))
        if len(ctx) == 0:
            return np.zeros(horizon, dtype=float)

        recent = ctx[-min(len(ctx), 28):]
        level = float(np.mean(recent)) if len(recent) else float(np.mean(ctx))

        if len(ctx) >= self.seasonal_period:
            seasonal = np.resize(ctx[-self.seasonal_period :], horizon).astype(float)
        else:
            seasonal = np.full(horizon, level, dtype=float)

        if context_dates is not None and future_dates is not None and len(context_dates) and len(future_dates):
            factors = self.compute_weekday_factors(ctx, context_dates)
            weights = self._future_dow_weights(future_dates, factors)
            seasonal = seasonal * weights

        seasonal_mean = float(np.mean(seasonal)) if len(seasonal) else 0.0
        if seasonal_mean > 1e-8:
            seasonal = seasonal * (level / seasonal_mean)

        baseline = 0.75 * seasonal + 0.25 * level
        return np.maximum(0.0, baseline)

    def seasonality_aware_calibration(
        self,
        generated_future: Sequence[float],
        context_values: Sequence[float],
        context_dates: Sequence[Any],
        future_dates: Sequence[Any],
        strength: float = 0.75,
    ) -> np.ndarray:
        gen = np.asarray(generated_future, dtype=float).ravel()
        if gen.size == 0:
            return gen
        strength = float(np.clip(strength, 0.0, 1.0))

        factors = self.compute_weekday_factors(context_values, context_dates)
        future_weights = self._future_dow_weights(future_dates, factors)
        calibrated = gen * ((1.0 - strength) + strength * future_weights)

        ctx = np.asarray(context_values, dtype=float).ravel()
        recent = ctx[-min(len(ctx), 28):] if len(ctx) else np.array([], dtype=float)
        recent_mean = float(np.mean(recent)) if recent.size else float(np.mean(ctx)) if len(ctx) else float(np.mean(gen))
        recent_std = float(np.std(recent)) if recent.size else float(np.std(gen))
        gen_mean = float(np.mean(calibrated))

        if abs(gen_mean) > 1e-8:
            level_blend = 0.65 * recent_mean + 0.35 * gen_mean
            calibrated = calibrated * (level_blend / gen_mean)

        gen_std = float(np.std(calibrated))
        if gen_std > 1e-8 and recent_std > 1e-8:
            target_std = 0.65 * recent_std + 0.35 * gen_std
            calibrated = (calibrated - np.mean(calibrated)) * (target_std / gen_std) + np.mean(calibrated)

        seasonal_ref = self.seasonal_naive_baseline(
            context_values=context_values,
            horizon=len(gen),
            context_dates=context_dates,
            future_dates=future_dates,
        )
        calibrated = (1.0 - self.seasonal_fallback_strength * strength) * calibrated + (
            self.seasonal_fallback_strength * strength
        ) * seasonal_ref

        zero_share = float(np.mean(ctx == 0)) if len(ctx) else 0.0
        if zero_share >= self.zero_threshold_for_sparsity:
            q = np.quantile(calibrated, min(max(zero_share, 0.0), 0.50))
            calibrated = np.where(calibrated <= q, 0.0, calibrated)

        return np.maximum(0.0, calibrated)

    def candidate_score(
        self,
        candidate: Sequence[float],
        context_values: Sequence[float],
        context_dates: Sequence[Any],
        future_dates: Sequence[Any],
    ) -> float:
        cand = np.asarray(candidate, dtype=float).ravel()
        ctx = np.asarray(context_values, dtype=float).ravel()
        if cand.size == 0:
            return 1e9

        recent = ctx[-min(len(ctx), 28):] if len(ctx) else cand
        mean_pen = abs(np.mean(cand) - np.mean(recent)) / (np.std(recent) + 1e-6)
        std_pen = abs(np.std(cand) - np.std(recent)) / (np.std(recent) + 1e-6)
        zero_pen = abs(np.mean(cand == 0) - np.mean(recent == 0))

        ctx_factors = self.compute_weekday_factors(ctx, context_dates)
        cand_factors = self.compute_weekday_factors(cand, future_dates)
        seasonal_pen = float(np.mean(np.abs(ctx_factors - cand_factors)))

        seasonal_ref = self.seasonal_naive_baseline(ctx, len(cand), context_dates, future_dates)
        ref_pen = float(np.mean(np.abs(cand - seasonal_ref)) / (np.std(recent) + 1e-6))

        acf1_pen = abs(_acf(cand, 1) - _acf(recent, 1))
        acf7_pen = abs(
            _acf(cand, min(self.seasonal_period, max(1, len(cand) - 1)))
            - _acf(recent, min(self.seasonal_period, max(1, len(recent) - 1)))
        )

        privacy_pen, _ = self._privacy_penalty(cand)
        return float(
            1.10 * mean_pen
            + 0.85 * std_pen
            + 0.60 * zero_pen
            + 0.80 * seasonal_pen
            + 0.75 * ref_pen
            + 0.60 * acf1_pen
            + 0.75 * acf7_pen
            + privacy_pen
        )

    def generate(
        self,
        context_values: Sequence[float],
        horizon: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        num_return_sequences: int = 24,
        do_sample: bool = True,
        temperature: float = 1.02,
        top_p: float = 0.92,
        top_k: int = 64,
        repetition_penalty: float = 1.08,
        context_dates: Optional[Sequence[Any]] = None,
        future_dates: Optional[Sequence[Any]] = None,
        apply_seasonal_calibration: bool = False,
    ) -> Dict[str, Any]:
        self._ensure_model_loaded()
        horizon = int(self.prediction_length if horizon is None else horizon)
        context = np.asarray(context_values, dtype=float).ravel()
        if context.size < self.context_length:
            raise ValueError(f"context length {context.size} is smaller than required {self.context_length}.")

        use_context = context[-self.context_length :]
        scale = self.quantizer.compute_scale(use_context)
        ctx_tokens = self.quantizer.quantize(self.quantizer.mean_scale(use_context, scale))

        if context_dates is None:
            context_dates = pd.date_range("2000-01-01", periods=len(use_context), freq="D")
        if future_dates is None:
            future_dates = pd.date_range(pd.to_datetime(list(context_dates))[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

        auto_metadata = self._metadata_tokens(use_context, context_dates, future_dates)
        if metadata:
            auto_metadata.update(metadata)
        source_text = self._make_source_text(ctx_tokens, horizon=horizon, metadata=auto_metadata)

        tokenizer = self.tokenizer
        model = self.model
        inputs = tokenizer(source_text, return_tensors="pt", truncation=True, max_length=self.max_source_length)
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        gen_kwargs = dict(
            max_new_tokens=min(self.max_target_length, max(24, 4 * horizon)),
            num_return_sequences=int(max(1, num_return_sequences)),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(max(1.0, repetition_penalty)),
        )
        if top_k is not None:
            gen_kwargs["top_k"] = int(max(0, top_k))
        try:
            gen_kwargs["renormalize_logits"] = True
        except Exception:
            pass

        try:
            import torch
            model.eval()
            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kwargs)
        except Exception:
            outputs = model.generate(**inputs, **gen_kwargs)

        decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        raw_candidates = []
        final_candidates = []
        scores = []
        candidate_min_distances = []
        parsed_token_counts = []
        used_fallback_flags = []
        privacy_repaired_flags = []

        baseline = self.seasonal_naive_baseline(
            context_values=use_context,
            horizon=horizon,
            context_dates=context_dates,
            future_dates=future_dates,
        )

        for txt in decoded_texts:
            token_ids = self.quantizer.text_to_tokens(txt)
            parsed_token_counts.append(int(len(token_ids)))
            used_fallback = False

            if len(token_ids) < max(3, int(round(0.60 * horizon))):
                raw = baseline.copy()
                used_fallback = True
            else:
                token_ids = token_ids[:horizon]
                if len(token_ids) < horizon:
                    token_ids = token_ids + [token_ids[-1]] * (horizon - len(token_ids))
                raw = self.quantizer.decode(token_ids, scale)
                raw = np.maximum(0.0, np.asarray(raw, dtype=float))

                if float(np.sum(raw)) <= 1e-8 or float(np.std(raw)) <= 1e-8:
                    raw = 0.50 * raw + 0.50 * baseline
                    used_fallback = True

            cal = raw.copy()
            if apply_seasonal_calibration:
                cal = self.seasonality_aware_calibration(
                    generated_future=raw,
                    context_values=use_context,
                    context_dates=context_dates,
                    future_dates=future_dates,
                    strength=self.seasonality_strength,
                )
            score = self.candidate_score(
                candidate=cal,
                context_values=use_context,
                context_dates=context_dates,
                future_dates=future_dates,
            )
            cal, privacy_meta = self._repair_candidate_for_privacy(
                candidate=cal,
                baseline=baseline,
                context_values=use_context,
                context_dates=context_dates,
                future_dates=future_dates,
            )
            score = self.candidate_score(
                candidate=cal,
                context_values=use_context,
                context_dates=context_dates,
                future_dates=future_dates,
            )
            raw_candidates.append(raw)
            final_candidates.append(cal)
            scores.append(score)
            candidate_min_distances.append(float(privacy_meta.get("privacy_min_distance", np.nan)))
            privacy_repaired_flags.append(bool(privacy_meta.get("privacy_repaired", 0.0)))
            used_fallback_flags.append(bool(used_fallback))

        best_idx = int(np.argmin(scores))
        return {
            "best_future": np.asarray(final_candidates[best_idx], dtype=float),
            "best_raw_future": np.asarray(raw_candidates[best_idx], dtype=float),
            "best_index": best_idx,
            "candidate_scores": [float(s) for s in scores],
            "candidate_futures": [np.asarray(x, dtype=float) for x in final_candidates],
            "raw_candidate_futures": [np.asarray(x, dtype=float) for x in raw_candidates],
            "candidate_min_distances": [float(x) for x in candidate_min_distances],
            "source_text": source_text,
            "decoded_texts": decoded_texts,
            "parsed_token_counts": parsed_token_counts,
            "used_fallback_flags": used_fallback_flags,
            "privacy_repaired_flags": privacy_repaired_flags,
            "fallback_share": float(np.mean(used_fallback_flags)) if used_fallback_flags else 0.0,
            "privacy_repair_share": float(np.mean(privacy_repaired_flags)) if privacy_repaired_flags else 0.0,
            "backend_name": self.backend_name,
        }

    def save(
        self,
        output_dir: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        private: bool = True,
        commit_message: str = "Upload SDG checkpoint",
    ) -> Path:
        self._ensure_model_loaded()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)

        sanitized_config = dict(self.config)
        sanitized_config["base_model_id"] = self._resolve_base_model_id()
        if not self._is_hf_model_id(sanitized_config.get("model_name")):
            sanitized_config["model_name"] = sanitized_config.get("base_model_id") or str(sanitized_config.get("model_name", ""))
        self.config = sanitized_config

        (out / "sdg_config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        (out / "training_info.json").write_text(json.dumps(self.training_info, indent=2), encoding="utf-8")
        if self.privacy_reference_bank is not None:
            np.save(out / "privacy_reference_bank.npy", np.asarray(self.privacy_reference_bank, dtype=np.float32))
        if self.privacy_reference_stats:
            (out / "privacy_reference_stats.json").write_text(json.dumps(self.privacy_reference_stats, indent=2), encoding="utf-8")
        (out / "README.md").write_text(self._build_model_card(repo_id=repo_id), encoding="utf-8")

        if push_to_hub:
            self.upload_to_hub(out, repo_id=repo_id, token=token, private=private, commit_message=commit_message)
        return out

    def _build_model_card(self, repo_id: Optional[str] = None) -> str:
        repo_name = repo_id or "sdg-t5-qlora"
        base_model_id = self._resolve_base_model_id()

        yaml_lines = [
            "---",
            "library_name: transformers",
            "pipeline_tag: time-series-forecasting",
            "tags:",
            "  - time-series",
            "  - synthetic-data",
            "  - seq2seq",
            "  - retail",
            "  - qlora",
        ]
        if base_model_id:
            yaml_lines.append(f"base_model: {base_model_id}")
        yaml_lines.append("---")
        yaml_block = "\n".join(yaml_lines)

        return f"""{yaml_block}

# {repo_name}Synthetic time-series generation checkpoint for the DIF-PI framework.

## Model summary

This checkpoint is trained as a seq2seq generator on tokenized retail demand windows. It uses a T5-style encoder-decoder backbone, QLoRA when available, extended time-series special tokens, calendar conditioning, multiple-sample generation, and a seasonality-aware calibration step at inference time.

## Intended use

The model is intended for research on synthetic retail demand generation and validation inside the DIF-PI framework. It is not intended for safety-critical or fully autonomous business decisions without human review.

## Training setup

- Base model: {base_model_id or 'not declared'}
- Context length: {self.context_length}
- Prediction length: {self.prediction_length}
- Quantization bins: {self.quantizer.num_bins}
- Backend: {self.backend_name or 'seq2seq'}
"""

    def upload_to_hub(
        self,
        local_dir: str,
        repo_id: Optional[str],
        token: Optional[str] = None,
        private: bool = True,
        commit_message: str = "Upload SDG checkpoint",
    ) -> None:
        if not repo_id:
            raise ValueError("repo_id must be provided when uploading to Hugging Face.")
        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise ImportError("huggingface_hub is required for upload_to_hub().") from exc

        local_path = Path(local_dir)
        self.config["base_model_id"] = self._resolve_base_model_id()
        if not self._is_hf_model_id(self.config.get("model_name")):
            self.config["model_name"] = self.config.get("base_model_id") or str(self.config.get("model_name", ""))
        (local_path / "sdg_config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        (local_path / "README.md").write_text(self._build_model_card(repo_id=repo_id), encoding="utf-8")

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(local_path),
            commit_message=commit_message,
            ignore_patterns=["*.ipynb_checkpoints*", "__pycache__/*"],
        )

    @staticmethod
    def _read_checkpoint_config(model_dir: Any) -> Dict[str, Any]:
        cfg_path = Path(model_dir) / "sdg_config.json"
        if not cfg_path.exists():
            return {}
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @classmethod
    def checkpoint_is_compatible(
        cls,
        model_dir: Any,
        *,
        expected_model_name: Optional[str] = None,
        expected_num_bins: Optional[int] = None,
        expected_use_special_tokens: Optional[bool] = None,
        expected_add_calendar_features: Optional[bool] = None,
        expected_context_length: Optional[int] = None,
        expected_prediction_length: Optional[int] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            return False, "missing_dir", {}
        cfg = cls._read_checkpoint_config(model_dir)
        if not cfg:
            return False, "missing_sdg_config", {}

        def _norm_model_id(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            return str(x).strip()

        ckpt_model = _norm_model_id(cfg.get("base_model_id") or cfg.get("model_name"))
        if expected_model_name is not None:
            expected_model = _norm_model_id(expected_model_name)
            if ckpt_model != expected_model:
                return False, f"base_model_mismatch: ckpt={ckpt_model} expected={expected_model}", cfg

        if expected_num_bins is not None and int(cfg.get("num_bins", expected_num_bins)) != int(expected_num_bins):
            return False, f"num_bins_mismatch: ckpt={cfg.get('num_bins')} expected={expected_num_bins}", cfg

        if expected_use_special_tokens is not None and bool(cfg.get("use_special_tokens", expected_use_special_tokens)) != bool(expected_use_special_tokens):
            return False, "use_special_tokens_mismatch", cfg

        if expected_add_calendar_features is not None and bool(cfg.get("add_calendar_features", expected_add_calendar_features)) != bool(expected_add_calendar_features):
            return False, "add_calendar_features_mismatch", cfg

        if expected_context_length is not None and int(cfg.get("context_length", expected_context_length)) != int(expected_context_length):
            return False, f"context_length_mismatch: ckpt={cfg.get('context_length')} expected={expected_context_length}", cfg

        if expected_prediction_length is not None and int(cfg.get("prediction_length", expected_prediction_length)) != int(expected_prediction_length):
            return False, f"prediction_length_mismatch: ckpt={cfg.get('prediction_length')} expected={expected_prediction_length}", cfg

        return True, "ok", cfg

    @classmethod
    def load(cls, model_dir: str) -> "LLMSyntheticTimeSeriesGenerator":
        hf = cls._require_hf_stack()
        AutoTokenizer = hf["AutoTokenizer"]
        AutoModelForSeq2SeqLM = hf["AutoModelForSeq2SeqLM"]

        model_path = Path(model_dir)
        cfg_path = model_path / "sdg_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing SDG config: {cfg_path}")

        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        obj = cls(
            model_name=str(config.get("base_model_id") or config.get("model_name") or model_path),
            context_length=int(config.get("context_length", 140)),
            prediction_length=int(config.get("prediction_length", 30)),
            num_bins=int(config.get("num_bins", 4094)),
            value_range=tuple(config.get("value_range", [-5.0, 5.0])),
            learning_rate=float(config.get("learning_rate", 2.0e-4)),
            train_steps=int(config.get("train_steps", 1500)),
            lora_rank=int(config.get("lora_rank", 32)),
            lora_alpha=int(config.get("lora_alpha", 64)),
            batch_size=int(config.get("batch_size", 2)),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
            max_source_length=int(config.get("max_source_length", 768)),
            max_target_length=int(config.get("max_target_length", 256)),
            random_state=int(config.get("random_state", 42)),
            task_prefix=str(config.get("task_prefix", "generate synthetic retail demand future from historical context")),
            seasonality_strength=float(config.get("seasonality_strength", 0.70)),
            seasonal_period=int(config.get("seasonal_period", 7)),
            seasonal_fallback_strength=float(config.get("seasonal_fallback_strength", 0.35)),
            zero_threshold_for_sparsity=float(config.get("zero_threshold_for_sparsity", 0.60)),
            prefer_backend=str(config.get("prefer_backend", "qlora")),
            use_special_tokens=bool(config.get("use_special_tokens", True)),
            add_calendar_features=bool(config.get("add_calendar_features", True)),
            warmup_ratio=float(config.get("warmup_ratio", 0.05)),
            weight_decay=float(config.get("weight_decay", 0.01)),
            privacy_reference_max_windows=int(config.get("privacy_reference_max_windows", 2000)),
            privacy_min_distance_quantile=float(config.get("privacy_min_distance_quantile", 0.10)),
            privacy_distance_penalty=float(config.get("privacy_distance_penalty", 2.0)),
            privacy_noise_strength=float(config.get("privacy_noise_strength", 0.06)),
            privacy_baseline_blend=float(config.get("privacy_baseline_blend", 0.15)),
            privacy_training_jitter_prob=float(config.get("privacy_training_jitter_prob", 0.35)),
            privacy_training_jitter_strength=float(config.get("privacy_training_jitter_strength", 0.05)),
            privacy_deduplicate_examples=bool(config.get("privacy_deduplicate_examples", True)),
            privacy_filter_enabled=bool(config.get("privacy_filter_enabled", True)),
            privacy_filter_max_retries=int(config.get("privacy_filter_max_retries", 3)),
        )
        obj.config.update(config)

        try:
            import torch
        except Exception as exc:
            raise ImportError("torch is required to load the SDG checkpoint.") from exc

        def _safe_empty_cache() -> None:
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

        obj.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            use_fast=False,
            local_files_only=True,
        )

        adapter_cfg_path = model_path / "adapter_config.json"
        if adapter_cfg_path.exists():
            try:
                from peft import PeftModel
            except Exception as exc:
                raise ImportError("peft is required to load LoRA/PEFT SDG checkpoints.") from exc

            try:
                adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
            except Exception:
                adapter_cfg = {}

            base_model_id = (
                config.get("base_model_id")
                or adapter_cfg.get("base_model_name_or_path")
                or config.get("model_name")
            )
            if not base_model_id:
                raise ValueError(f"Could not determine base model id for adapter checkpoint at {model_path}")

            _safe_empty_cache()
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(base_model_id),
                local_files_only=True,
                dtype=torch.float32,
            )

            current_vocab = int(base_model.get_input_embeddings().weight.shape[0])
            target_vocab = int(len(obj.tokenizer))
            if target_vocab != current_vocab:
                try:
                    base_model.resize_token_embeddings(target_vocab, mean_resizing=False)
                except TypeError:
                    base_model.resize_token_embeddings(target_vocab, mean_resizing=False)

            _safe_empty_cache()
            obj.model = PeftModel.from_pretrained(
                base_model,
                str(model_path),
                local_files_only=True,
            )
            obj.is_peft_model = True
            obj.backend_name = "loaded_saved_checkpoint"
            obj.config["model_name"] = str(base_model_id)
            obj.config["base_model_id"] = str(base_model_id)
        else:
            obj.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            obj.is_peft_model = False
            obj.backend_name = "loaded"


        training_info_path = model_path / "training_info.json"
        if training_info_path.exists():
            try:
                obj.training_info = json.loads(training_info_path.read_text(encoding="utf-8"))
            except Exception:
                obj.training_info = {}

        privacy_bank_path = model_path / "privacy_reference_bank.npy"
        if privacy_bank_path.exists():
            try:
                obj.privacy_reference_bank = np.load(privacy_bank_path)
            except Exception:
                obj.privacy_reference_bank = None

        privacy_stats_path = model_path / "privacy_reference_stats.json"
        if privacy_stats_path.exists():
            try:
                obj.privacy_reference_stats = json.loads(privacy_stats_path.read_text(encoding="utf-8"))
            except Exception:
                obj.privacy_reference_stats = {}

        if obj.privacy_reference_bank is not None:
            obj._fit_privacy_index()

        return obj
