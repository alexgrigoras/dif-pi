import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score


class TBWISAGenerator:
    """Scenario generation using structural causal inference.

    Notes for DIF-PI usage:
    - The default behavior is the original log-log LinearRegression elasticity.
    - The notebook may optionally enable a controlled, robust, zero-safe elasticity estimator
      (Huber + trend/seasonality/lag + event-based filtering) via `configure_controlled_elasticity()`.
    - Scenario generation remains unchanged: `generate_scenarios()` calls `calculate_structural_elasticity()`,
      so enabling the controlled estimator automatically affects scenario generation without monkey-patching.
    """

    def __init__(self):
        # If set via `configure_controlled_elasticity()`, the controlled estimator will be used.
        self._controlled_elasticity_cfg = None
        self._last_elasticity_meta = {}


    # Utility helpers

    @staticmethod
    def revenue(price, demand):
        """Revenue helper used in DIF-PI."""
        return np.asarray(price) * np.asarray(demand)

    @staticmethod
    def best_window(rev, min_len=7, max_len=30, length_penalty=0.0):
        """Find the best window on a 1D revenue array (same logic as DIF-PI notebook)."""
        rev = np.asarray(rev, float)
        best = None
        for L in range(int(min_len), int(min(max_len, len(rev))) + 1):
            for s in range(0, len(rev) - L + 1):
                avg = float(np.mean(rev[s:s+L]))
                score = avg - float(length_penalty) * L
                if (best is None) or (score > best['score']):
                    best = {'start': s, 'end': s+L, 'len': L, 'avg': avg, 'score': score}
        return best


    # Elasticity estimation

    def configure_controlled_elasticity(
        self,
        seasonal_periods=(7.0, 365.25),
        eps=1e-9,
        prior_eps=0.05,
        use_event=True,
        price_change_pct_thresh=0.01,
        elast_max_abs=None,
        huber_epsilon=1.35,
        huber_alpha=1e-4,
        min_fit_rows=30,
    ):
        """Enable the controlled, robust, zero-safe elasticity estimator.

        This matches the DIF-PI notebook logic:
        - zero-safe log fit (demand>0 and lag(demand)>0)
        - robust regression (Huber)
        - controls (trend z-score + Fourier seasonality + lag log-demand)
        - optional event-based fitting on price-change days
        - non-positive prior: beta <= -prior_eps
        - optional magnitude cap (beta >= -elast_max_abs if provided)
        """
        self._controlled_elasticity_cfg = dict(
            seasonal_periods=tuple(seasonal_periods),
            eps=float(eps),
            prior_eps=float(prior_eps),
            use_event=bool(use_event),
            price_change_pct_thresh=float(price_change_pct_thresh),
            elast_max_abs=None if elast_max_abs is None else float(elast_max_abs),
            huber_epsilon=float(huber_epsilon),
            huber_alpha=float(huber_alpha),
            min_fit_rows=int(min_fit_rows),
        )
        return self._controlled_elasticity_cfg

    def _controlled_elasticity(self, price_s, demand_s, **kwargs):
        """Internal implementation of controlled elasticity (ported from DIF-PI notebook)."""
        seasonal_periods = kwargs.get('seasonal_periods', (7.0, 365.25))
        eps = kwargs.get('eps', 1e-9)
        prior_eps = kwargs.get('prior_eps', 0.05)
        use_event = kwargs.get('use_event', True)
        price_change_pct_thresh = kwargs.get('price_change_pct_thresh', 0.01)
        elast_max_abs = kwargs.get('elast_max_abs', None)
        huber_epsilon = kwargs.get('huber_epsilon', 1.35)
        huber_alpha = kwargs.get('huber_alpha', 1e-4)
        min_fit_rows = kwargs.get('min_fit_rows', 30)

        p = np.asarray(price_s, float)
        d = np.asarray(demand_s, float)
        n = int(min(len(p), len(d)))
        if n < 5:
            beta_raw = 0.0
            beta = -abs(float(prior_eps))
            intercept = float(np.log(np.clip(np.mean(d) if n else 1.0, eps, None)))
            resid = np.zeros(n, dtype=float)
            meta = {'beta_raw': beta_raw, 'beta_final': beta, 'intercept': intercept, 'r2': np.nan, 'n_fit': 0, 'clipped': False,
                    'event_used': False, 'n_fit_event': None, 'price_change_pct_thresh': float(price_change_pct_thresh)}
            return beta, intercept, resid, meta

        # Build supervised rows (t>=1) with a valid lag and strictly positive demand (zero-safe)
        idx = np.arange(1, n)
        mask_base = (p[idx] > 0) & (d[idx] > 0) & (d[idx-1] > 0)

        # Optional: event-based fitting (focus on periods where price actually changes)
        event_used = False
        n_fit_event = None
        if bool(use_event):
            thr = float(np.log1p(float(price_change_pct_thresh)))
            dlogp = np.abs(np.log(np.clip(p[idx], eps, None)) - np.log(np.clip(p[idx-1], eps, None)))
            mask_event = mask_base & (dlogp >= thr)
            n_fit_event = int(mask_event.sum())
            if n_fit_event >= int(min_fit_rows):
                mask = mask_event
                event_used = True
            else:
                mask = mask_base
        else:
            mask = mask_base

        if int(mask.sum()) < int(min_fit_rows):
            # Not enough usable rows -> return a conservative tiny negative elasticity
            beta_raw = 0.0
            beta = -abs(float(prior_eps))
            intercept = float(np.log(np.clip(np.mean(d[d>0]) if np.any(d>0) else 1.0, eps, None)))
            x_all = np.log(np.clip(p[:n], eps, None))
            y_all = np.log(np.clip(d[:n], eps, None))
            resid = (y_all - (intercept + beta * x_all)).astype(float)
            resid[0] = resid[1] if n > 1 else 0.0
            meta = {'beta_raw': beta_raw, 'beta_final': beta, 'intercept': intercept, 'r2': np.nan, 'n_fit': int(mask.sum()), 'clipped': False, 'fallback': 'min_fit_rows',
                    'event_used': bool(event_used), 'n_fit_event': n_fit_event, 'price_change_pct_thresh': float(price_change_pct_thresh)}
            return beta, intercept, resid, meta

        # Prepare regressors on masked rows
        p_t = p[idx][mask]
        d_t = d[idx][mask]
        d_lag = d[idx-1][mask]
        t = np.arange(n, dtype=float)
        t_t = t[idx][mask]

        y_t = np.log(d_t)
        x_price = np.log(p_t)

        # trend (z-scored)
        trend = (t_t - t_t.mean()) / (t_t.std() if t_t.std() > 0 else 1.0)

        cols = {'log_price': x_price, 'trend_z': trend}

        # seasonality via Fourier terms on index
        for per in seasonal_periods:
            cols[f'sin_{per}'] = np.sin(2.0 * np.pi * t_t / per)
            cols[f'cos_{per}'] = np.cos(2.0 * np.pi * t_t / per)

        cols['lag_log_demand'] = np.log(d_lag)

        X_df = pd.DataFrame(cols)

        reg = HuberRegressor(epsilon=float(huber_epsilon), alpha=float(huber_alpha))
        reg.fit(X_df, y_t)

        beta_raw = float(reg.coef_[0])  # log_price coefficient
        intercept = float(reg.intercept_)

        y_hat = reg.predict(X_df)
        r2 = float(r2_score(y_t, y_hat))

        # Non-positive prior: beta <= -prior_eps
        beta = float(beta_raw)
        if beta >= 0:
            beta = -abs(float(prior_eps))

        clipped = False
        if elast_max_abs is not None:
            cap = -abs(float(elast_max_abs))
            if beta < cap:
                beta = cap
                clipped = True

        # Residuals used by TBWISA internals: keep length n
        x_all = np.log(np.clip(p[:n], eps, None))
        y_all = np.log(np.clip(d[:n], eps, None))
        resid = (y_all - (intercept + beta * x_all)).astype(float)
        resid[0] = resid[1] if n > 1 else 0.0

        meta = {
            'beta_raw': beta_raw,
            'beta_final': beta,
            'intercept': intercept,
            'r2': r2,
            'n_fit': int(mask.sum()),
            'clipped': clipped,
            'huber_epsilon': float(huber_epsilon),
            'huber_alpha': float(huber_alpha),
            'seasonal_periods': list(seasonal_periods),
            'event_used': bool(event_used),
            'n_fit_event': n_fit_event,
            'price_change_pct_thresh': float(price_change_pct_thresh),
        }
        return beta, intercept, resid, meta

    def calculate_structural_elasticity(self, input, output):
        """Structural causal elasticity calculation.

        If `configure_controlled_elasticity()` was called, uses the controlled estimator.
        Otherwise uses the original log-log LinearRegression estimator.
        """
        if self._controlled_elasticity_cfg is not None:
            beta, intercept, resid, meta = self._controlled_elasticity(
                np.asarray(input, float),
                np.asarray(output, float),
                **self._controlled_elasticity_cfg
            )
            self._last_elasticity_meta = meta
            return beta, intercept, np.asarray(resid, float).flatten()

        regression = LinearRegression()
        x = np.log(np.asarray(input, float)).reshape(-1, 1)
        y = np.log(np.asarray(output, float)).reshape(-1, 1)
        regression.fit(x, y)
        elasticity = float(regression.coef_[0][0])
        intercept = float(regression.intercept_[0])
        residuals = (y - regression.predict(x)).flatten()
        self._last_elasticity_meta = {'beta_raw': elasticity, 'beta_final': elasticity, 'intercept': intercept, 'r2': np.nan, 'n_fit': int(len(x)), 'clipped': False}
        return elasticity, intercept, residuals

    def non_linear_elasticity(self, change, base_elasticity):
        """Non-linear elasticity adjustment"""
        return base_elasticity * (1 + 0.5 * change**2)

    def apply_randomness(self, demand, change, randomness_factor=0.01, cap=0.05, seed=42):
        np.random.seed(seed)
        randomness = np.random.normal(loc=0, scale=randomness_factor * abs(change), size=len(demand))
        capped_randomness = np.clip(randomness, -cap, cap)
        stochastic_demand = demand * (1 + capped_randomness)
        return stochastic_demand

    def generate_scenarios(self, data, input_col, output_col, price_change_percentages):
        """Scenario generation using structural causal inference."""
        elasticity, intercept, residuals = self.calculate_structural_elasticity(data[input_col], data[output_col])
        scenarios_input = {}
        scenarios_output = {}
        for change in price_change_percentages:
            adjusted_input = data[input_col] * (1 + change / 100)
            adjusted_elasticity = self.non_linear_elasticity(change / 100, elasticity)
            counterfactual_output = np.exp(intercept + adjusted_elasticity * np.log(adjusted_input) + residuals)
            counterfactual_output = self.apply_randomness(counterfactual_output, change, randomness_factor=0.1, cap=0.1)
            scenarios_input[f'{input_col} change {change}%'] = adjusted_input
            scenarios_output[f'{input_col} change {change}%'] = counterfactual_output
        return scenarios_input, scenarios_output

    def calculate_score_with_demand(self, revenue_window, demand_window=None, penalty_factor=0.0, demand_weight=0.0):
        """
        Compute score: revenue minus length penalty, plus (optional) reward for low demand.
        Notes:- If `demand_window` is None or `demand_weight=0`, this reduces to the revenue-only scoring used in DIF-PI.
        """
        rev_window = np.asarray(revenue_window, float)
        avg_revenue = float(np.mean(rev_window))
        window_size = int(len(rev_window))
        score = avg_revenue - float(penalty_factor) * window_size

        if (demand_window is None) or (float(demand_weight) == 0.0):
            return score

        dem_window = np.asarray(demand_window, float)
        max_demand = float(np.max(dem_window)) if float(np.max(dem_window)) > 0 else 1.0
        avg_demand_norm = float(np.mean(dem_window)) / max_demand
        demand_boost = float(demand_weight) * (1.0 - avg_demand_norm)
        return score + demand_boost

    def find_optimal_window_with_demand(self, revenue, demand=None, penalty_factor=0.0, demand_weight=0.0,
                                        min_window_size=7, max_window_size=30):
        """Find the best window on revenue (optionally demand-weighted).

        Supported input styles:

        1) Array mode (used by DIF-PI):
           - `revenue`: 1D array-like
           - `demand`: optional 1D array-like (same length)
           Returns: {'start','end','len','avg','score'} where `end` is exclusive.

        2) Scenario-dict mode (backward compatible with older TBWISA usage):
           - `revenue`: dict mapping scenario -> 1D array-like
           - `demand`:  dict mapping scenario -> 1D array-like (optional)
           Returns: {'scenario','start','end','score'} where `end` is inclusive (legacy behavior).
        """
        # --- Scenario-dict mode (legacy) ---
        if isinstance(revenue, dict):
            best = {"scenario": None, "start": None, "end": None, "score": -np.inf}
            for scenario, rev_series in revenue.items():
                if scenario in ("Baseline", "Actuals"):
                    continue
                rev_series = np.asarray(rev_series, float)
                dem_series = None
                if isinstance(demand, dict) and (scenario in demand):
                    dem_series = np.asarray(demand[scenario], float)

                scenario_best_score = -np.inf
                scenario_best_range = (None, None)

                for window_size in range(int(min_window_size), int(min(len(rev_series), max_window_size)) + 1):
                    for start in range(0, len(rev_series) - window_size + 1):
                        rev_window = rev_series[start:start + window_size]
                        dem_window = dem_series[start:start + window_size] if dem_series is not None else None
                        score = self.calculate_score_with_demand(rev_window, dem_window, penalty_factor, demand_weight)

                        if score > scenario_best_score:
                            scenario_best_score = score
                            # legacy: end index inclusive
                            scenario_best_range = (start, start + window_size - 1)

                if scenario_best_score > best["score"]:
                    best.update({
                        "scenario": scenario,
                        "start": scenario_best_range[0],
                        "end": scenario_best_range[1],
                        "score": float(scenario_best_score)
                    })
            return best

        # --- Array mode (DIF-PI) ---
        rev = np.asarray(revenue, float)
        dem = None if demand is None else np.asarray(demand, float)

        best = None
        for L in range(int(min_window_size), int(min(max_window_size, len(rev))) + 1):
            for s in range(0, len(rev) - L + 1):
                rev_window = rev[s:s+L]
                dem_window = dem[s:s+L] if dem is not None else None
                score = float(self.calculate_score_with_demand(rev_window, dem_window, penalty_factor, demand_weight))
                avg = float(np.mean(rev_window))
                if (best is None) or (score > best["score"]):
                    best = {"start": int(s), "end": int(s+L), "len": int(L), "avg": avg, "score": score}
        return best
