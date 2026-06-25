"""strategy.ols_static

OLS/MVHR static hedge ratio strategy.

Purpose:
- Estimate a single hedge ratio at trade start (t=0) using historical data.
- Hold that hedge ratio constant for the entire trade.

Return definition:
- Uses SIMPLE returns:
    r_t = P_t / P_{t-1} - 1

OLS variants:
- Direct hedge (WTI_SPOT vs CL): default is NO intercept.
- Cross hedge (BRENT_SPOT, OPEC_BASKET vs CL): default is WITH intercept.

You can override intercept via constructor parameter `intercept`.

Important:
- Strategy returns h_t only.
- Simulator handles hedge direction, rounding, costs, and roll.

Leakage:
- Uses ONLY the history available up to the current day inside the trade window.
  In static mode, simulator calls this once at t=0.

Implementation choice:
- We estimate beta using OLS slope. This is equivalent to MVHR under standard assumptions.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .base import HedgeStrategy


class OLSStaticStrategy(HedgeStrategy):
    """Static OLS hedge ratio (beta) estimated once at t=0."""

    name = "OLSStatic"

    def __init__(
        self,
        intercept: Optional[bool] = None,
        min_obs: int = 60,
        exposure_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create static OLS strategy.

        Args:
            intercept: If None, choose based on exposure (direct vs cross).
            min_obs: Minimum observations needed to estimate beta; otherwise return 0.
            exposure_id: Optional exposure id (WTI_SPOT/BRENT_SPOT/OPEC_BASKET). If None, inferred from scenario_meta.
        """
        super().__init__(intercept=intercept, min_obs=min_obs, exposure_id=exposure_id, **kwargs)
        self.user_intercept = intercept
        self.min_obs = int(min_obs)
        self.exposure_id = exposure_id
        self._cached_h: Optional[float] = None

    def reset(self) -> None:
        self._cached_h = None

    @staticmethod
    def _default_intercept(exposure_id: str) -> bool:
        """Default intercept rule.

        - WTI_SPOT (direct hedge) => False
        - Brent/OPEC (cross hedge) => True
        """
        e = str(exposure_id).upper()
        if e == "WTI_SPOT" or e == "WTI":
            return False
        # Cross-hedges
        if e in ("BRENT_SPOT", "BRENT", "OPEC_BASKET", "OPEC"):
            return True
        # Conservative default: include intercept
        return True

    def build_h_path(
        self,
        dS: np.ndarray,
        dF: np.ndarray,
        dates=None,
        scenario_meta: Optional[Dict[str, Any]] = None,
        spot: Optional[np.ndarray] = None,
        fut: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Preferred fast API: compute a single static hedge ratio and repeat it over the window.

        Notes
        -----
        - Uses only the data provided via `spot`/`fut` (or reconstructed levels from diffs).
        - To estimate using pre-trade history, the simulator should pass that history as `spot`/`fut`.
        """
        scenario_meta = scenario_meta or {}
        n = len(dS)

        # Determine exposure id
        exp = self.exposure_id or scenario_meta.get("exposure_id") or scenario_meta.get("exposure") or ""

        # Determine intercept usage
        if self.user_intercept is None:
            intercept = self._default_intercept(str(exp))
        else:
            intercept = bool(self.user_intercept)

        # Resolve price levels
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

        mask = np.isfinite(r_s) & np.isfinite(r_f)
        if np.sum(mask) < self.min_obs:
            beta = 0.0
        else:
            beta = float(self.safe_ols_beta(r_f[mask], r_s[mask], intercept=intercept))

        self._cached_h = float(beta)
        h = np.full(n, float(beta), dtype=float)
        h = self.clip_h(h)
        return h

    def get_h(self, t_index: int, history_df: pd.DataFrame, scenario_meta: Dict[str, Any]) -> float:
        # cache result for the scenario
        if self._cached_h is not None:
            return float(self._cached_h)

        # Determine exposure id
        exp = self.exposure_id or scenario_meta.get("exposure_id") or scenario_meta.get("exposure") or ""

        # Determine intercept usage
        if self.user_intercept is None:
            intercept = self._default_intercept(str(exp))
        else:
            intercept = bool(self.user_intercept)

        # Need spot/fut columns
        if "spot" not in history_df.columns or "fut" not in history_df.columns:
            self._cached_h = 0.0
            return 0.0

        spot = pd.to_numeric(history_df["spot"], errors="coerce").to_numpy(dtype=float)
        fut = pd.to_numeric(history_df["fut"], errors="coerce").to_numpy(dtype=float)

        r_s = self.simple_returns(spot)
        r_f = self.simple_returns(fut)

        mask = np.isfinite(r_s) & np.isfinite(r_f)
        if np.sum(mask) < self.min_obs:
            self._cached_h = 0.0
            return 0.0

        beta = float(self.safe_ols_beta(r_f[mask], r_s[mask], intercept=intercept))

        # cache
        self._cached_h = float(beta)
        return float(beta)