"""SDG integration helpers for DIF-PI."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from src.sdg import LLMSyntheticTimeSeriesGenerator
import gc
import json
import numpy as np
import pandas as pd


# Utilities

def contiguous_runs(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            runs.append((int(start), int(i)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(mask))))
    return runs


def local_level(history: Sequence[float], end_idx: int, window: int = 14) -> float:
    lo = max(0, int(end_idx) - int(window))
    x = np.asarray(history[lo:end_idx], dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    pos = x[x > 0]
    return float(np.median(pos)) if pos.size else float(np.median(x))


def _safe_empty_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# Data preparation

def build_daily_panel_for_sdg(
    panel_df: pd.DataFrame,
    *,
    sku_col: str,
    time_col: str,
    demand_col: str,
    price_col: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    use_df = panel_df.copy()
    use_df[sku_col] = use_df[sku_col].astype(str)
    use_df[time_col] = pd.to_datetime(use_df[time_col], errors="raise")

    for sku, g in use_df.groupby(sku_col):
        g = g.sort_values(time_col).set_index(time_col)
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        gg = g.reindex(full)
        gg.index.name = time_col
        gg[sku_col] = str(sku)
        if price_col and price_col in gg.columns:
            gg[price_col] = gg[price_col].ffill().bfill()
        gg[demand_col] = gg[demand_col].fillna(0.0)
        keep_cols = [sku_col, demand_col]
        if price_col and price_col in gg.columns:
            keep_cols.append(price_col)
        rows.append(gg.reset_index()[[time_col] + keep_cols])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[sku_col, time_col, demand_col])


def prepare_missingness_masks(
    sku_df_train: pd.DataFrame,
    *,
    demand_col: str,
    explicit_col: str = "demand_missing_explicit",
    reindex_col: str = "is_reindex_gap",
    fill_reindex_gaps: bool = False,
) -> Dict[str, np.ndarray]:
    demand_real = sku_df_train[demand_col].values.astype(float)
    missing_explicit = (
        sku_df_train[explicit_col].astype(bool).values
        if explicit_col in sku_df_train.columns
        else np.zeros(len(sku_df_train), dtype=bool)
    )
    reindex_gap = (
        sku_df_train[reindex_col].astype(bool).values
        if reindex_col in sku_df_train.columns
        else np.zeros(len(sku_df_train), dtype=bool)
    )
    fill_mask = (missing_explicit | reindex_gap) if fill_reindex_gaps else missing_explicit.copy()
    return {
        "demand_real": demand_real,
        "missing_explicit_train": missing_explicit,
        "reindex_gap_train": reindex_gap,
        "sdg_fill_mask": fill_mask,
    }


# Safe saved-model loading

def _build_sdg_object_from_config(config: Dict[str, Any]) -> LLMSyntheticTimeSeriesGenerator:
    obj = LLMSyntheticTimeSeriesGenerator(
        model_name=str(config.get("base_model_id") or config.get("model_name") or "amazon/chronos-t5-small"),
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
    return obj

def _safe_load_sdg_checkpoint(model_dir: Any) -> LLMSyntheticTimeSeriesGenerator:
    model_dir = Path(model_dir)
    return LLMSyntheticTimeSeriesGenerator.load(str(model_dir))


# Model acquisition

def load_or_train_sdg_model(
    *,
    panel: pd.DataFrame,
    train_skus: Iterable[Any],
    case_sku: Any,
    decision_date: Any,
    sku_col: str,
    time_col: str,
    demand_col: str,
    price_col: Optional[str],
    model_dir: Any,
    model_name: str = "amazon/chronos-t5-small",
    context_length: int = 140,
    horizon: int = 30,
    num_bins: int = 512,
    learning_rate: float = 5e-5,
    train_steps: int = 300,
    batch_size: int = 4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    seasonality_strength: float = 0.15,
    max_train_skus: int = 64,
    train_if_missing: bool = True,
    seed: int = 42,
    stride: int = 1,
    include_metadata: bool = False,
    logging_steps: int = 25,
    save_steps: int = 250,
):
    model_dir = Path(model_dir)

    if model_dir.exists() and (model_dir / "sdg_config.json").exists():
        if not train_if_missing:
            model = _safe_load_sdg_checkpoint(model_dir)
            return model, "loaded_existing_checkpoint"

        compatible, reason, _ = LLMSyntheticTimeSeriesGenerator.checkpoint_is_compatible(
            model_dir,
            expected_model_name=model_name,
            expected_num_bins=num_bins,
            expected_context_length=context_length,
            expected_prediction_length=horizon,
        )
        if compatible:
            try:
                model = _safe_load_sdg_checkpoint(model_dir)
                return model, "loaded_existing_checkpoint"
            except Exception as exc:
                print(f"Existing SDG checkpoint at {model_dir} could not be loaded safely; retraining from scratch. Error: {exc}")
        else:
            print(f"Existing SDG checkpoint at {model_dir} is incompatible ({reason}); retraining from scratch.")

    elif not train_if_missing:
        raise FileNotFoundError(
            f"Missing SDG checkpoint at {model_dir}. "
            "Set SDG_TRAIN_IF_MISSING=True or train the SDG module separately."
        )

    model = LLMSyntheticTimeSeriesGenerator(
        model_name=model_name,
        context_length=context_length,
        prediction_length=horizon,
        num_bins=num_bins,
        learning_rate=learning_rate,
        train_steps=train_steps,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        seasonality_strength=seasonality_strength,
        random_state=seed,
    )

    sdg_skus = list(dict.fromkeys(list(train_skus)[: int(max_train_skus)] + [str(case_sku)]))
    sdg_panel = build_daily_panel_for_sdg(
        panel[panel[sku_col].astype(str).isin(set(str(x) for x in sdg_skus))].copy(),
        sku_col=sku_col,
        time_col=time_col,
        demand_col=demand_col,
        price_col=price_col,
    )

    cutoff_map = {str(case_sku): pd.to_datetime(decision_date)}
    sdg_examples = model.build_training_dataframe(
        panel_df=sdg_panel,
        sku_col=sku_col,
        time_col=time_col,
        target_col=demand_col,
        train_skus=sdg_skus,
        stride=int(stride),
        include_metadata=bool(include_metadata),
        cutoff_map=cutoff_map,
    ).reset_index(drop=True)

    if len(sdg_examples) < 20:
        raise ValueError(f"Not enough SDG training examples were created: {len(sdg_examples)}")

    split_idx = max(1, int(round(0.95 * len(sdg_examples))))
    split_idx = min(split_idx, len(sdg_examples) - 1)
    sdg_train_df = sdg_examples.iloc[:split_idx].reset_index(drop=True)
    sdg_eval_df = sdg_examples.iloc[split_idx:].reset_index(drop=True)

    print(
        f"Training SDG model from scratch on {len(sdg_train_df)} train examples "
        f"and {len(sdg_eval_df)} eval examples."
    )
    model.fit(
        train_df=sdg_train_df,
        eval_df=sdg_eval_df,
        output_dir=str(model_dir),
        logging_steps=int(logging_steps),
        save_steps=int(save_steps),
    )
    return model, "trained_in_notebook"


# Gap fill

def fill_missing_with_sdg(
    model: LLMSyntheticTimeSeriesGenerator,
    *,
    series: Sequence[float],
    fill_mask: Sequence[bool],
    dates: Sequence[Any],
    num_return_sequences: int = 32,
    temperature: float = 1.05,
    top_p: float = 0.92,
    top_k: int = 80,
    repetition_penalty: float = 1.10,
    apply_seasonal_calibration: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    # build a no-NaN working series; only true missing points are overwritten later
    original = np.asarray(series, dtype=float).copy()
    filled = original.copy()
    fill_mask = np.asarray(fill_mask, dtype=bool)
    details: List[Dict[str, Any]] = []

    finite_mask = np.isfinite(filled)
    if not np.all(finite_mask):
        s = pd.Series(filled)
        filled = s.interpolate(limit_direction="both").ffill().bfill().values.astype(float)
        filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)

    max_chunk = int(max(1, getattr(model, "prediction_length", 30)))
    base_num_return_sequences = int(max(1, num_return_sequences))

    for start, end in contiguous_runs(fill_mask):
        run_len = int(end - start)
        run_details_method = "sdg_generate_chunked" if run_len > max_chunk else "sdg_generate"

        pos = int(start)
        while pos < int(end):
            chunk_end = int(min(end, pos + max_chunk))
            chunk_len = int(chunk_end - pos)

            ctx_end = int(pos)
            ctx_start = max(0, ctx_end - int(model.context_length))
            context_len = int(ctx_end - ctx_start)

            if context_len < int(model.context_length):
                fallback_value = local_level(filled, ctx_end, window=14)
                filled[pos:chunk_end] = fallback_value
                details.append(
                    {
                        "start": int(pos),
                        "end": int(chunk_end),
                        "length": int(chunk_len),
                        "run_start": int(start),
                        "run_end": int(end),
                        "run_length": int(run_len),
                        "method": "local_median_fallback",
                        "context_len": int(context_len),
                    }
                )
                pos = chunk_end
                continue

            context = np.asarray(filled[ctx_start:ctx_end], dtype=float)
            context_dates = pd.to_datetime(dates[ctx_start:ctx_end])
            future_dates = pd.date_range(pd.to_datetime(dates[pos]), periods=chunk_len, freq="D")

            effective_num_return_sequences = base_num_return_sequences
            if run_len > max_chunk:
                effective_num_return_sequences = min(base_num_return_sequences, 6)

            gen = model.generate(
                context_values=context,
                horizon=chunk_len,
                num_return_sequences=int(effective_num_return_sequences),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                context_dates=context_dates,
                future_dates=future_dates,
                apply_seasonal_calibration=bool(apply_seasonal_calibration),
            )

            pred = np.asarray(gen.get("best_future", gen.get("best_raw_future")), dtype=float)[:chunk_len]
            pred = np.maximum(0.0, pred)

            local = local_level(filled, ctx_end, window=14)
            if np.isfinite(local) and local > 0:
                pred = 0.85 * pred + 0.15 * local

            filled[pos:chunk_end] = pred
            details.append(
                {
                    "start": int(pos),
                    "end": int(chunk_end),
                    "length": int(chunk_len),
                    "run_start": int(start),
                    "run_end": int(end),
                    "run_length": int(run_len),
                    "method": run_details_method,
                    "context_len": int(context_len),
                    "candidate_best_score": float(min(gen.get("candidate_scores", [np.nan]))),
                    "num_return_sequences_used": int(effective_num_return_sequences),
                    "used_fallback_share": float(np.mean(gen.get("used_fallback_flags", [False]))),
                    "avg_parsed_token_count": float(np.mean(gen.get("parsed_token_counts", [0]))),
                }
            )
            pos = chunk_end

    # preserve original non-missing observations exactly
    result = original.copy()
    result[fill_mask] = filled[fill_mask]
    return result, pd.DataFrame(details)


# Top-level integration

def run_sdg_gapfill_for_case(
    *,
    sku_df_train: pd.DataFrame,
    panel: pd.DataFrame,
    sku_col: str,
    time_col: str,
    demand_col: str,
    price_col: Optional[str],
    case_sku: Any,
    train_skus: Iterable[Any],
    decision_date: Any,
    use_sdg_gapfill: bool = True,
    fill_reindex_gaps: bool = False,
    model_dir: Any = "./artifacts/models/sdg_chronos_lora_dunnhumby",
    model_name: str = "amazon/chronos-t5-small",
    context_length: int = 140,
    horizon: int = 30,
    num_bins: int = 512,
    learning_rate: float = 5e-5,
    train_steps: int = 300,
    batch_size: int = 4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    seasonality_strength: float = 0.15,
    num_return_sequences: int = 24,
    max_train_skus: int = 64,
    train_if_missing: bool = True,
    seed: int = 42,
    stride: int = 1,
    include_metadata: bool = False,
    logging_steps: int = 25,
    save_steps: int = 250,
    temperature: float = 1.02,
    top_p: float = 0.92,
    top_k: int = 80,
    repetition_penalty: float = 1.10,
    apply_seasonal_calibration: bool = False,
) -> Dict[str, Any]:
    time_train = sku_df_train[time_col].values
    masks = prepare_missingness_masks(
        sku_df_train,
        demand_col=demand_col,
        fill_reindex_gaps=fill_reindex_gaps,
    )
    demand_real = masks["demand_real"]
    fill_mask = masks["sdg_fill_mask"]
    demand_gapfilled = demand_real.copy()
    status = "not_used"
    details = pd.DataFrame()

    if use_sdg_gapfill and np.any(fill_mask):
        model, status = load_or_train_sdg_model(
            panel=panel,
            train_skus=train_skus,
            case_sku=case_sku,
            decision_date=decision_date,
            sku_col=sku_col,
            time_col=time_col,
            demand_col=demand_col,
            price_col=price_col,
            model_dir=model_dir,
            model_name=model_name,
            context_length=context_length,
            horizon=horizon,
            num_bins=num_bins,
            learning_rate=learning_rate,
            train_steps=train_steps,
            batch_size=batch_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            seasonality_strength=seasonality_strength,
            max_train_skus=max_train_skus,
            train_if_missing=train_if_missing,
            seed=seed,
            stride=stride,
            include_metadata=include_metadata,
            logging_steps=logging_steps,
            save_steps=save_steps,
        )
        demand_gapfilled, details = fill_missing_with_sdg(
            model,
            series=demand_real,
            fill_mask=fill_mask,
            dates=time_train,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            apply_seasonal_calibration=apply_seasonal_calibration,
        )
    else:
        status = "not_needed" if not np.any(fill_mask) else "disabled"

    return {
        "demand_real": demand_real,
        "time_train": time_train,
        "missing_explicit_train": masks["missing_explicit_train"],
        "reindex_gap_train": masks["reindex_gap_train"],
        "sdg_fill_mask": fill_mask,
        "demand_gapfilled": demand_gapfilled,
        "sdg_status": status,
        "sdg_details": details,
    }
