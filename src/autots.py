from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class _FallbackOrder:
    p: int
    d: int
    q: int
    aic: float


class AutoTSNPDRegressor:
    """ARIMA / AutoTS-style baseline for NPD gap forecasting.

    If the external `autots` package is available, it is used with a constrained
    ARIMA-only search. For short histories or any AutoTS fit failure, the class
    falls back automatically to a compact SARIMAX grid search and keeps the same
    public API expected by the validation notebook.
    """

    def __init__(
        self,
        forecast_length: int = 5,
        max_p: int = 3,
        max_d: int = 1,
        max_q: int = 3,
        seasonal: bool = False,
        verbose: int = 0,
        min_history_for_autots: Optional[int] = None,
    ):
        self.forecast_length = int(forecast_length)
        self.max_p = int(max_p)
        self.max_d = int(max_d)
        self.max_q = int(max_q)
        self.seasonal = bool(seasonal)
        self.verbose = int(verbose)
        self.min_history_for_autots = (
            int(min_history_for_autots)
            if min_history_for_autots is not None
            else max(12, self.forecast_length * 3)
        )

        self.backend_ = "sarimax_grid"
        self.model_ = None
        self.best_order_: Optional[tuple[int, int, int]] = None
        self.series_: Optional[np.ndarray] = None

        try:
            from autots import AutoTS  # type: ignore
            self._AutoTS = AutoTS
            self.backend_ = "autots"
        except Exception:
            self._AutoTS = None
            self.backend_ = "sarimax_grid"

    def _fit_fallback(self, series: np.ndarray) -> None:
        y = pd.Series(np.asarray(series, dtype=float).reshape(-1))
        best: Optional[_FallbackOrder] = None
        best_res = None

        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and d == 0 and q == 0:
                        continue
                    try:
                        res = SARIMAX(
                            y,
                            order=(p, d, q),
                            trend="c",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                        aic = float(res.aic)
                        if best is None or aic < best.aic:
                            best = _FallbackOrder(p=p, d=d, q=q, aic=aic)
                            best_res = res
                    except Exception:
                        continue

        if best_res is None:
            self.best_order_ = (0, 0, 0)
            self.model_ = None
        else:
            self.best_order_ = (best.p, best.d, best.q)
            self.model_ = best_res

    def fit_series(self, series: Sequence[float]) -> "AutoTSNPDRegressor":
        values = np.asarray(series, dtype=float).reshape(-1)
        if len(values) < 8:
            raise ValueError("AutoTSNPDRegressor requires at least 8 observations in the history series.")
        self.series_ = values.copy()

        can_try_autots = (
            self.backend_ == "autots"
            and self._AutoTS is not None
            and len(values) >= self.min_history_for_autots
        )

        if can_try_autots:
            df = pd.DataFrame({
                "date": pd.date_range("2000-01-01", periods=len(values), freq="D"),
                "value": values,
            })
            try:
                model = self._AutoTS(
                    forecast_length=self.forecast_length,
                    frequency="D",
                    model_list=["ARIMA"],
                    ensemble=None,
                    max_generations=1,
                    num_validations=0,
                    verbose=self.verbose,
                )
                model = model.fit(df=df, date_col="date", value_col="value", id_col=None)
                self.model_ = model
                self.best_order_ = None
                self.backend_ = "autots"
                return self
            except Exception as exc:
                if self.verbose:
                    print(
                        "[AutoTSNPDRegressor] AutoTS fit failed; "
                        f"falling back to SARIMAX grid. Reason: {exc}"
                    )

        self.backend_ = "sarimax_grid"
        self._fit_fallback(values)
        return self

    def forecast_series(self, history: Sequence[float] | None = None, forecast_length: Optional[int] = None) -> np.ndarray:
        horizon = int(self.forecast_length if forecast_length is None else forecast_length)
        values = self.series_ if history is None else np.asarray(history, dtype=float).reshape(-1)
        if values is None or len(values) == 0:
            raise ValueError("No history available. Fit the model first or pass an explicit history series.")

        if self.backend_ == "autots" and self.model_ is not None:
            pred = self.model_.predict().forecast.values.reshape(-1)
            pred = pred[:horizon]
        elif self.model_ is not None:
            pred = np.asarray(self.model_.forecast(steps=horizon), dtype=float).reshape(-1)
        else:
            pred = np.repeat(float(np.mean(values)), horizon)

        return np.maximum(pred, 1.0)
