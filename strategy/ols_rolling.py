"""strategy.ols_rolling

Rolling OLS hedge ratio strategy.

Key properties:
- Uses SIMPLE returns: r_t = P_t / P_{t-1} - 1
- No leakage: only uses data passed in history_df
- Supports configurable rolling window (30/60/120/252)
- Direct hedge (WTI) → no intercept (default)
- Cross hedge (Brent/OPEC) → with intercept (default)

IMPORTANT:
This strategy ONLY computes h_t.
Rounding, hedge direction, transaction costs and roll handling
are done inside the simulator.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .base import HedgeStrategy


class OLSRollingStrategy(HedgeStrategy):
    """Rolling OLS hedge ratio estimator."""

    name = "OLSRolling"

    def __init__(
        self,
        window: int = 120,
        intercept: Optional[bool] = None,
        min_obs: int = 30,
        exposure_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            window=window,
            intercept=intercept,
            min_obs=min_obs,
            exposure_id=exposure_id,
            **kwargs,
        )
        self.window = int(window)
        self.user_intercept = intercept
        self.min_obs = int(min_obs)
        self.exposure_id = exposure_id

    # ------------------------------------------------------------
    # Default intercept rule
    # ------------------------------------------------------------

    @staticmethod
    def _default_intercept(exposure_id: str) -> bool:
        e = str(exposure_id).upper()
        if e in ("WTI_SPOT", "WTI"):
            return False  # direct hedge
        if e in ("BRENT_SPOT", "BRENT", "OPEC_BASKET", "OPEC"):
            return True   # cross hedge
        return True

    @staticmethod
    def _rolling_beta(
        x: np.ndarray,
        y: np.ndarray,
        window: int,
        min_obs: int,
        intercept: bool,
    ) -> np.ndarray:
        """Vectorized rolling OLS slope (beta) for y ~ a + b x.

        Parameters
        ----------
        x, y : np.ndarray
            Arrays of returns aligned to the trade window.
        window : int
            Lookback window length in *observations* (index-based window).
        min_obs : int
            Minimum non-NaN observations required to produce a beta.
        intercept : bool
            If True, uses demeaned covariance/variance (OLS with intercept).
            If False, uses beta = sum(x*y)/sum(x^2) (no intercept).

        Returns
        -------
        np.ndarray
            Beta array with NaN where insufficient data.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        out = np.full(n, np.nan, dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            return out

        xv = np.where(valid, x, 0.0)
        yv = np.where(valid, y, 0.0)

        cN = np.cumsum(valid.astype(int))
        cX = np.cumsum(xv)
        cY = np.cumsum(yv)
        cX2 = np.cumsum(xv * xv)
        cXY = np.cumsum(xv * yv)

        # Helper to slice cumulative sums on [s..t]
        def win_sum(c: np.ndarray, s: int, t: int) -> float:
            return float(c[t] - (c[s - 1] if s > 0 else 0.0))

        for t in range(n):
            s = max(0, t - window + 1) if window and window > 0 else 0
            nn = int(win_sum(cN, s, t))
            if nn < max(2 if intercept else 1, min_obs):
                continue

            sumx = win_sum(cX, s, t)
            sumy = win_sum(cY, s, t)
            sumx2 = win_sum(cX2, s, t)
            sumxy = win_sum(cXY, s, t)

            if intercept:
                # cov/var with intercept; (n-1) cancels in slope
                denom = (sumx2 - (sumx * sumx) / nn)
                if denom == 0.0:
                    continue
                numer = (sumxy - (sumx * sumy) / nn)
                out[t] = numer / denom
            else:
                if sumx2 == 0.0:
                    continue
                out[t] = sumxy / sumx2

        return out

    # ------------------------------------------------------------
    # Main hedge-ratio computation
    # ------------------------------------------------------------

    def build_h_path(
        self,
        dS: np.ndarray,
        dF: np.ndarray,
        dates=None,
        scenario_meta: Optional[Dict[str, Any]] = None,
        spot: Optional[np.ndarray] = None,
        fut: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        scenario_meta = scenario_meta or {}

        # Resolve exposure id
        exp = (
            self.exposure_id
            or scenario_meta.get("exposure_id")
            or scenario_meta.get("exposure")
            or ""
        )

        intercept = (
            self._default_intercept(str(exp))
            if self.user_intercept is None
            else bool(self.user_intercept)
        )

        # We prefer price levels if provided; otherwise reconstruct levels from diffs (relative scale is fine for returns)
        if spot is None:
            spot_lvl = np.cumsum(np.asarray(dS, dtype=float))
        else:
            spot_lvl = np.asarray(spot, dtype=float)

        if fut is None:
            fut_lvl = np.cumsum(np.asarray(dF, dtype=float))
        else:
            fut_lvl = np.asarray(fut, dtype=float)

        r_s = self.simple_returns(spot_lvl)
        r_f = self.simple_returns(fut_lvl)

        betas = self._rolling_beta(
            x=r_f,
            y=r_s,
            window=int(self.window) if self.window is not None else 0,
            min_obs=int(self.min_obs),
            intercept=bool(intercept),
        )

        # Replace NaNs with 0.0 for early period / insufficient data
        h = np.where(np.isfinite(betas), betas, 0.0)
        # Optional safety clip
        h = self.clip_h(h)
        return h

    def get_h(
        self,
        t_index: int,
        history_df: pd.DataFrame,
        scenario_meta: Dict[str, Any],
    ) -> float:
        # Legacy API expects history_df with columns spot/fut.
        if "spot" not in history_df.columns or "fut" not in history_df.columns:
            return 0.0

        spot = history_df["spot"].astype(float).to_numpy()
        fut = history_df["fut"].astype(float).to_numpy()

        # Build path on the available history and return last element
        # (This remains slower than using build_h_path from the simulator, but avoids per-step DataFrame allocations.)
        dS = np.diff(spot, prepend=np.nan)
        dF = np.diff(fut, prepend=np.nan)
        h_path = self.build_h_path(dS=dS, dF=dF, dates=None, scenario_meta=scenario_meta, spot=spot, fut=fut)
        return float(h_path[-1]) if len(h_path) else 0.0