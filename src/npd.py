import numpy as np
import pandas as pd


class NPDModel:
    """Next Purchase Day (NPD) utilities.

    This class provides:
    - supervised dataset construction for next inter-purchase gap prediction
    - a simple purchase intention index over a future horizon
    - an automatic mapping from NPD MAE (days) to timing influence alpha

    Notes
    -----
    This module is intentionally model-agnostic: any regressor that
    implements `.fit(X, y)` and `.predict(X)` can be plugged in.
    """

    def __init__(
        self,
        k: int = 5,
        min_events: int = 7,
        horizon: int = 30,
        good: float = 2.0,
        ok: float = 5.0,
        alpha_good: float = 0.30,
        alpha_ok: float = 0.15,
    ):
        self.k = int(k)
        self.min_events = int(min_events)
        self.horizon = int(horizon)
        self.good = float(good)
        self.ok = float(ok)
        self.alpha_good = float(alpha_good)
        self.alpha_ok = float(alpha_ok)

    def make_supervised(
        self,
        tx: pd.DataFrame,
        cust_col: str,
        date_col: str,
        k: int | None = None,
        min_events: int | None = None,
    ):
        """Build supervised samples for Next Purchase Day (NPD).

        For each customer, compute inter-purchase gaps (in days). Use the last k gaps to predict the next gap.
        Returns:
            X: shape (n_samples, k)
            y: shape (n_samples,)
        """
        k = self.k if k is None else int(k)
        min_events = self.min_events if min_events is None else int(min_events)

        t = tx[[cust_col, date_col]].dropna().sort_values([cust_col, date_col])
        X, y = [], []
        for _, g in t.groupby(cust_col):
            d = g[date_col].values
            if len(d) < min_events:
                continue
            gaps = np.diff(d).astype('timedelta64[D]').astype(int)
            for i in range(k, len(gaps)):
                X.append(gaps[i-k:i])
                y.append(gaps[i])
        if not X:
            raise ValueError('Not enough customer events for NPD. Consider lowering k/min_events.')
        return np.asarray(X, float), np.asarray(y, float)

    def intention_index(
        self,
        tx: pd.DataFrame,
        model,
        cust_col: str,
        date_col: str,
        k: int | None = None,
        horizon: int | None = None,
    ):
        """Compute a simple purchase intention index over the next `horizon` days.

        For each customer with enough history, predict the next inter-purchase gap (days),
        map it to a date, and count how many customers are predicted to purchase on each day.

        This version supports both:
        - classic one-step regressors returning shape (n_samples,)
        - direct multi-step forecasters returning shape (n_samples, H)

        For multi-step models, the first predicted step is used as the next-gap estimate.
        """
        k = self.k if k is None else int(k)
        horizon = self.horizon if horizon is None else int(horizon)

        # If the loaded model exposes its required input length, prefer that.
        model_k = int(getattr(model, "input_length", k))

        t = tx[[cust_col, date_col]].dropna().sort_values([cust_col, date_col])
        t0 = t[date_col].max().normalize()
        days = pd.date_range(t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=horizon), freq="D")
        counts = pd.Series(0.0, index=days)

        for _, g in t.groupby(cust_col):
            d = g[date_col].values
            if len(d) < model_k + 1:
                continue

            gaps = np.diff(d).astype("timedelta64[D]").astype(int)
            if len(gaps) < model_k:
                continue

            x = gaps[-model_k:].reshape(1, -1)

            pred = np.asarray(model.predict(x), dtype=float).reshape(-1)
            if pred.size == 0:
                continue

            # For direct multi-step models, use the first step as the next-gap estimate
            gap_pred = float(np.clip(pred[0], 1.0, horizon))

            pred_date = (t0 + pd.Timedelta(days=int(round(gap_pred)))).normalize()
            if pred_date in counts.index:
                counts.loc[pred_date] += 1.0

        idx = counts / counts.max() if counts.max() > 0 else counts
        return t0, idx

    @staticmethod
    def alpha_from_mae(
        mae_days,
        good: float = 2.0,
        ok: float = 5.0,
        alpha_good: float = 0.30,
        alpha_ok: float = 0.15,
    ) -> float:
        """Map NPD MAE (days) -> timing influence alpha.
        - mae <= good: strong timing weight (alpha_good)
        - good < mae <= ok: mild timing weight (alpha_ok)
        - mae > ok: disable timing influence (0.0)
        """
        try:
            mae = float(mae_days)
        except Exception:
            return 0.0
        if mae <= good:
            return float(alpha_good)
        if mae <= ok:
            return float(alpha_ok)
        return 0.0