"""
strategy.base
-------------

Abstract base class and shared utilities for hedge-ratio strategies.

Design principles:
- A strategy ONLY computes hedge ratio h_t.
- It must NOT:
    * round contracts
    * apply transaction costs
    * apply roll logic
- It must NOT use future data (no leakage).

Interface contract:

Preferred (fast) interface:
    h_path = strategy.build_h_path(dS, dF, dates, scenario_meta)

    Returns a NumPy array of hedge ratios for the entire trade window.
    This is the recommended method for performance.

Legacy interface:
    h_t = strategy.get_h(t_index, history_df, scenario_meta)

    Still supported for compatibility but slower and less efficient.

Expected columns inside history_df (at minimum):
- spot  : spot price level
- fut   : futures price level
- dS    : spot price difference (optional)
- dF    : futures price difference (optional)

Strategies are free to compute returns internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Sequence

import numpy as np
import pandas as pd

ScenarioMeta = Dict[str, Any]


def _meta_get(meta: ScenarioMeta, key: str, default: Any = None) -> Any:
    """Safe meta getter (avoids KeyError)."""
    return meta.get(key, default)


class HedgeStrategy(ABC):
    """Abstract base class for hedge-ratio generators."""

    name: str = "BaseStrategy"

    def __init__(self, **kwargs: Any):
        # store arbitrary config params
        self.params = kwargs

    def reset(self) -> None:
        """Reset internal state (if any) before a new scenario starts."""
        pass

    def build_h_path(
        self,
        dS: np.ndarray,
        dF: np.ndarray,
        dates: Optional[Sequence[pd.Timestamp]] = None,
        scenario_meta: Optional[ScenarioMeta] = None,
        spot: Optional[np.ndarray] = None,
        fut: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return hedge ratios for the full trade window (preferred fast API).

        Parameters
        ----------
        dS, dF
            Arrays of spot/futures *price differences* aligned to the trade window.
            Length must match the window length.
        dates
            Optional dates aligned to the window.
        scenario_meta
            Scenario-level metadata.
        spot, fut
            Optional price level arrays aligned to the window.

        Notes
        -----
        - Default implementation falls back to calling legacy `get_h` sequentially.
        - Child classes should override this to avoid pandas/history slicing.
        """
        scenario_meta = scenario_meta or {}
        dS = np.asarray(dS, dtype=float)
        dF = np.asarray(dF, dtype=float)
        n = len(dS)
        if n != len(dF):
            raise ValueError("dS and dF must have the same length")

        # Fallback: emulate history_df incrementally (slow; for compatibility only)
        h = np.zeros(n, dtype=float)
        if spot is None or fut is None:
            # Create minimal history using cumulative sum of diffs if levels absent
            spot_series = pd.Series(dS).cumsum()
            fut_series = pd.Series(dF).cumsum()
        else:
            spot_series = pd.Series(np.asarray(spot, dtype=float))
            fut_series = pd.Series(np.asarray(fut, dtype=float))

        hist_full = pd.DataFrame({
            "spot": spot_series,
            "fut": fut_series,
            "dS": pd.Series(dS),
            "dF": pd.Series(dF),
        })
        if dates is not None:
            hist_full.index = pd.DatetimeIndex(pd.to_datetime(list(dates)))

        # Ensure strategy state is clean per scenario
        self.reset()
        for t in range(n):
            hist = hist_full.iloc[: t + 1]
            h[t] = float(self.get_h(t, hist, scenario_meta))
        return h

    @abstractmethod
    def get_h(
        self,
        t_index: int,
        history_df: pd.DataFrame,
        scenario_meta: Dict[str, Any],
    ) -> float:
        """Legacy (compatibility) API. Prefer overriding build_h_path().

        Return hedge ratio h_t.

        Must only use information available in `history_df`.
        Must return a float (can be negative or >1).
        """
        raise NotImplementedError

    # ------------------------------------------------------------
    # Shared helper utilities (optional use by child classes)
    # ------------------------------------------------------------

    @staticmethod
    def simple_returns(x: Any) -> np.ndarray:
        """Compute simple returns r_t = P_t / P_{t-1} - 1.

        Returns a NumPy array (first element is NaN).
        """
        if isinstance(x, pd.Series):
            arr = x.to_numpy(dtype=float)
        else:
            arr = np.asarray(x, dtype=float)
        out = np.empty_like(arr, dtype=float)
        out[:] = np.nan
        out[1:] = (arr[1:] / arr[:-1]) - 1.0
        return out

    @staticmethod
    def safe_ols_beta(x: np.ndarray, y: np.ndarray, intercept: bool = False) -> float:
        """Compute OLS slope (beta) safely.

        If intercept=False:
            y = beta * x
        If intercept=True:
            y = alpha + beta * x

        Returns beta. If insufficient data or zero variance, returns 0.0.
        """

        if len(x) < 2:
            return 0.0

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            return 0.0

        if not intercept:
            # beta = sum(x*y) / sum(x^2)
            denom = np.sum(x ** 2)
            if denom == 0:
                return 0.0
            return float(np.sum(x * y) / denom)
        else:
            # beta from linear regression with intercept
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            denom = np.sum((x - x_mean) ** 2)
            if denom == 0:
                return 0.0
            return float(np.sum((x - x_mean) * (y - y_mean)) / denom)

    @staticmethod
    def min_variance_hedge_ratio(r_s: np.ndarray, r_f: np.ndarray) -> float:
        """Classic MVHR formula using covariance / variance.

        h = Cov(r_s, r_f) / Var(r_f)

        Returns 0.0 if variance is zero or insufficient data.
        """
        if len(r_s) < 2 or len(r_f) < 2:
            return 0.0

        r_s = np.asarray(r_s, dtype=float)
        r_f = np.asarray(r_f, dtype=float)

        mask = np.isfinite(r_s) & np.isfinite(r_f)
        r_s = r_s[mask]
        r_f = r_f[mask]

        if len(r_s) < 2:
            return 0.0

        var_f = np.var(r_f, ddof=1)
        if var_f == 0:
            return 0.0

        # sample covariance (ddof=1) without allocating a full 2x2 matrix
        rs = r_s - np.mean(r_s)
        rf = r_f - np.mean(r_f)
        cov = float(np.sum(rs * rf) / (len(r_s) - 1))
        return float(cov / var_f)

    @staticmethod
    def clip_h(h: Any, h_min: float = -2.0, h_max: float = 2.0) -> Any:
        """Clip hedge ratio(s) to a safe range."""
        return np.clip(h, h_min, h_max)