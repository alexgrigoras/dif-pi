
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def set_random_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_daily_series(df: pd.DataFrame, sku: str, sku_col: str, time_col: str, price_col: str, demand_col: str) -> pd.DataFrame:
    g = df[df[sku_col].astype(str) == str(sku)].copy()
    g = g.sort_values(time_col).set_index(time_col)
    full = pd.date_range(g.index.min(), g.index.max(), freq='D')
    g = g.reindex(full)
    g.index.name = time_col
    if price_col in g.columns:
        g[price_col] = g[price_col].ffill().bfill()
    g[demand_col] = pd.to_numeric(g[demand_col], errors='coerce').fillna(0.0)
    g[sku_col] = str(sku)
    return g.reset_index()


def build_train_windows(
    panel_df: pd.DataFrame,
    skus: Iterable[str],
    sku_col: str,
    time_col: str,
    price_col: str,
    demand_col: str,
    context_length: int,
    horizon: int,
    stride: int = 7,
    max_windows_per_sku: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X, Y, rows = [], [], []
    for sku in skus:
        sku_df = make_daily_series(panel_df, str(sku), sku_col, time_col, price_col, demand_col)
        series = sku_df[demand_col].astype(float).values
        if len(series) < context_length + horizon:
            continue
        local = 0
        for end in range(context_length, len(series) - horizon + 1, stride):
            start = end - context_length
            future_end = end + horizon
            X.append(series[start:end])
            Y.append(series[end:future_end])
            rows.append({
                'sku': str(sku),
                'context_start': int(start),
                'context_end': int(end),
                'future_end': int(future_end),
            })
            local += 1
            if max_windows_per_sku is not None and local >= int(max_windows_per_sku):
                break
    if not X:
        return np.empty((0, context_length)), np.empty((0, horizon)), pd.DataFrame()
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float), pd.DataFrame(rows)


def build_eval_cases(
    panel_df: pd.DataFrame,
    skus: Iterable[str],
    sku_col: str,
    time_col: str,
    price_col: str,
    demand_col: str,
    context_length: int,
    horizon: int,
) -> List[Dict]:
    cases: List[Dict] = []
    for sku in skus:
        sku_df = make_daily_series(panel_df, str(sku), sku_col, time_col, price_col, demand_col)
        decision_date = sku_df[time_col].max() - pd.Timedelta(days=horizon)
        train_df = sku_df[sku_df[time_col] <= decision_date].copy()
        test_df = sku_df[sku_df[time_col] > decision_date].copy()
        if len(train_df) < context_length or len(test_df) < horizon:
            continue
        context_values = train_df[demand_col].astype(float).values[-context_length:]
        future_values = test_df[demand_col].astype(float).values[:horizon]
        context_dates = train_df[time_col].iloc[-context_length:].tolist()
        future_dates = test_df[time_col].iloc[:horizon].tolist()
        cases.append({
            'sku': str(sku),
            'context_values': np.asarray(context_values, dtype=float),
            'future_values': np.asarray(future_values, dtype=float),
            'context_dates': context_dates,
            'future_dates': future_dates,
        })
    return cases


def lag_acf(values: np.ndarray, lag: int) -> float:
    x = np.asarray(values, dtype=float).ravel()
    if len(x) <= lag or lag <= 0:
        return 0.0
    x1 = x[:-lag]
    x2 = x[lag:]
    if np.std(x1) < 1e-8 or np.std(x2) < 1e-8:
        return 0.0
    return float(np.corrcoef(x1, x2)[0, 1])


def summarize_windows(windows: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(windows, dtype=float).ravel()
    return {
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'median': float(np.median(flat)),
        'zero_share': float(np.mean(flat == 0)),
        'lag1_acf': float(lag_acf(flat, 1)),
        'lag7_acf': float(lag_acf(flat, 7)),
    }


def js_distance_from_hist(real_vals: np.ndarray, syn_vals: np.ndarray, bins: int = 30) -> float:
    real_vals = np.asarray(real_vals, dtype=float)
    syn_vals = np.asarray(syn_vals, dtype=float)
    lo = min(float(np.min(real_vals)), float(np.min(syn_vals)))
    hi = max(float(np.max(real_vals)), float(np.max(syn_vals)))
    if hi <= lo:
        return 0.0
    h1, edges = np.histogram(real_vals, bins=bins, range=(lo, hi), density=True)
    h2, _ = np.histogram(syn_vals, bins=edges, density=True)
    h1 = (h1 + 1e-12) / (h1 + 1e-12).sum()
    h2 = (h2 + 1e-12) / (h2 + 1e-12).sum()
    return float(jensenshannon(h1, h2))


def tstr_trts_metrics(real_windows: np.ndarray, syn_windows: np.ndarray) -> Dict:
    real_windows = np.asarray(real_windows, dtype=float)
    syn_windows = np.asarray(syn_windows, dtype=float)
    if len(real_windows) < 4 or len(syn_windows) < 4:
        return {'tstr': {'mae': np.nan, 'rmse': np.nan}, 'trts': {'mae': np.nan, 'rmse': np.nan}}
    X_real = real_windows[:, :-1]
    y_real = real_windows[:, -1]
    X_syn = syn_windows[:, :-1]
    y_syn = syn_windows[:, -1]
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_real, y_real, test_size=0.3, random_state=42)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_syn, y_syn, test_size=0.3, random_state=42)

    rf_tstr = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_tstr.fit(Xs_tr, ys_tr)
    pred_tstr = rf_tstr.predict(Xr_te)

    rf_trts = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_trts.fit(Xr_tr, yr_tr)
    pred_trts = rf_trts.predict(Xs_te)

    return {
        'tstr': {
            'mae': float(mean_absolute_error(yr_te, pred_tstr)),
            'rmse': float(np.sqrt(mean_squared_error(yr_te, pred_tstr))),
        },
        'trts': {
            'mae': float(mean_absolute_error(ys_te, pred_trts)),
            'rmse': float(np.sqrt(mean_squared_error(ys_te, pred_trts))),
        }
    }


def detection_auc(real_windows: np.ndarray, syn_windows: np.ndarray) -> float:
    X = np.vstack([real_windows, syn_windows])
    y = np.array([1] * len(real_windows) + [0] * len(syn_windows))
    if len(np.unique(y)) < 2 or len(X) < 10:
        return np.nan
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, proba))


def privacy_proxy(real_windows: np.ndarray, syn_windows: np.ndarray) -> Dict[str, float]:
    real = np.asarray(real_windows, dtype=float)
    syn = np.asarray(syn_windows, dtype=float)

    def pairwise_min_mean(A: np.ndarray, B: np.ndarray, exclude_self: bool = False) -> float:
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2) ** 0.5
        if exclude_self and A.shape == B.shape:
            np.fill_diagonal(d, np.inf)
        return float(np.mean(np.min(d, axis=1)))

    syn_to_real = pairwise_min_mean(syn, real, exclude_self=False)
    real_to_real = pairwise_min_mean(real, real, exclude_self=True)
    ratio = syn_to_real / real_to_real if real_to_real > 1e-8 else np.nan
    return {
        'syn_to_real_mean_min_dist': syn_to_real,
        'real_to_real_mean_min_dist': real_to_real,
        'nn_distance_ratio': ratio,
    }


def windowed_time_series(series: np.ndarray, window_size: int, step_size: int = 1) -> List[np.ndarray]:
    series = np.asarray(series, dtype=float).ravel()
    return [series[i:i+window_size] for i in range(0, len(series) - window_size + 1, step_size)]


def average_cosine_similarity(series1: np.ndarray, series2: np.ndarray, window_size: int = 3, step_size: int = 1) -> float:
    windows1 = windowed_time_series(series1, window_size, step_size)
    windows2 = windowed_time_series(series2, window_size, step_size)
    min_windows = min(len(windows1), len(windows2))
    if min_windows == 0:
        return np.nan
    sims = []
    for w1, w2 in zip(windows1[:min_windows], windows2[:min_windows]):
        sims.append(float(cosine_similarity(np.asarray(w1).reshape(1, -1), np.asarray(w2).reshape(1, -1))[0, 0]))
    return float(np.mean(sims))


def average_jensen_shannon_distance(series1: np.ndarray, series2: np.ndarray, window_size: int = 3, step_size: int = 1) -> float:
    windows1 = windowed_time_series(series1, window_size, step_size)
    windows2 = windowed_time_series(series2, window_size, step_size)
    min_windows = min(len(windows1), len(windows2))
    if min_windows == 0:
        return np.nan
    dists = []
    for w1, w2 in zip(windows1[:min_windows], windows2[:min_windows]):
        sum1 = np.sum(w1)
        sum2 = np.sum(w2)
        if sum1 == 0 and sum2 == 0:
            dists.append(0.0)
            continue
        if sum1 == 0 or sum2 == 0:
            dists.append(1.0)
            continue
        p1 = np.asarray(w1, dtype=float) / float(sum1)
        p2 = np.asarray(w2, dtype=float) / float(sum2)
        d = float(jensenshannon(p1, p2))
        dists.append(1.0 if np.isnan(d) else d)
    return float(np.mean(dists))


def windows_to_feature_dataframe(windows: np.ndarray, prefix: str = 't') -> pd.DataFrame:
    arr = np.asarray(windows, dtype=float)
    cols = [f'{prefix}_{i:03d}' for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def evaluate_generator_on_cases(model, eval_cases: List[Dict], model_name: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    real_windows, syn_windows, rows = [], [], []
    case_predictions = {}
    for case in eval_cases:
        context = case['context_values']
        future = case['future_values']
        pred = model.generate(context_values=context, horizon=len(future))
        pred = np.asarray(pred, dtype=float).ravel()
        if len(pred) != len(future):
            if len(pred) < len(future):
                pad = np.repeat(pred[-1] if len(pred) else 0.0, len(future) - len(pred))
                pred = np.concatenate([pred, pad])
            else:
                pred = pred[:len(future)]
        real_windows.append(future)
        syn_windows.append(pred)
        case_predictions[str(case['sku'])] = pred
        rows.append({
            'model': model_name,
            'sku': str(case['sku']),
            'mae': float(mean_absolute_error(future, pred)),
            'rmse': float(np.sqrt(mean_squared_error(future, pred))),
        })
    if not rows:
        return pd.DataFrame(), np.empty((0, 0)), np.empty((0, 0)), {'case_predictions': case_predictions}
    real_arr = np.asarray(real_windows, dtype=float)
    syn_arr = np.asarray(syn_windows, dtype=float)
    metrics = {
        'real_summary': summarize_windows(real_arr),
        'synthetic_summary': summarize_windows(syn_arr),
        'wasserstein_distance': float(wasserstein_distance(real_arr.ravel(), syn_arr.ravel())),
        'avg_cosine_similarity': float(np.mean([cosine_similarity(r.reshape(1, -1), s.reshape(1, -1))[0, 0] for r, s in zip(real_arr, syn_arr)])),
        'avg_jensen_shannon_distance': float(np.mean([js_distance_from_hist(r, s) for r, s in zip(real_arr, syn_arr)])),
        'pearson_corr': float(np.corrcoef(real_arr.ravel(), syn_arr.ravel())[0, 1]) if real_arr.size and syn_arr.size else np.nan,
    }
    metrics.update(tstr_trts_metrics(real_arr, syn_arr))
    metrics['detection_auc'] = detection_auc(real_arr, syn_arr)
    metrics['privacy_proxy'] = privacy_proxy(real_arr, syn_arr)
    metrics['case_predictions'] = case_predictions
    return pd.DataFrame(rows), real_arr, syn_arr, metrics


def comparison_summary_table(result_dict: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for name, payload in result_dict.items():
        m = payload.get('metrics', {})
        rows.append({
            'model': name,
            'mean': m.get('synthetic_summary', {}).get('mean', np.nan),
            'std': m.get('synthetic_summary', {}).get('std', np.nan),
            'zero_share': m.get('synthetic_summary', {}).get('zero_share', np.nan),
            'wasserstein_distance': m.get('wasserstein_distance', np.nan),
            'avg_cosine_similarity': m.get('avg_cosine_similarity', np.nan),
            'avg_jensen_shannon_distance': m.get('avg_jensen_shannon_distance', np.nan),
            'pearson_corr': m.get('pearson_corr', np.nan),
            'tstr_mae': m.get('tstr', {}).get('mae', np.nan),
            'trts_mae': m.get('trts', {}).get('mae', np.nan),
            'detection_auc': m.get('detection_auc', np.nan),
            'nn_distance_ratio': m.get('privacy_proxy', {}).get('nn_distance_ratio', np.nan),
            'avg_rmse': payload.get('per_sku', pd.DataFrame()).get('rmse', pd.Series(dtype=float)).mean(),
        })
    return pd.DataFrame(rows).sort_values('avg_rmse').reset_index(drop=True)
