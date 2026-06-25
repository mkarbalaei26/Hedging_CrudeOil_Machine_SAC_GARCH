"""strategy/ccc_garch.py

CCC-GARCH (proxy) hedge strategy.

This strategy uses the precomputed columns produced by BaseGARCH.py:
  - <PREFIX>_h_ccc_proxy_<W>
where PREFIX ∈ {WTI, BRENT, OPEC} and W ∈ {30, 60, 120, 252}.

Economic meaning:
  h_t = (sigma_spot_t / sigma_fut_t) * rho_t

Implementation notes:
- Works in both static and dynamic modes:
  * static: uses a single h (first available within the trade window)
  * dynamic: uses daily h_t (with an internal ffill within the trade window)
- Robust to missing h values: forward-fills within the trade window; if still missing,
  falls back to `fallback_h` (default 0.0).
- Optional clipping of h to avoid extreme leverage due to noisy correlation.

This file intentionally does NOT fit any GARCH model. It only consumes the precomputed
features stored in MasterData.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from .base import HedgeStrategy


def _prefix_from_exposure_id(exposure_id: str) -> str:
    """Map simulator exposure_id to feature prefix used in MasterData."""
    e = (exposure_id or "").upper()
    if "WTI" in e:
        return "WTI"
    if "BRENT" in e:
        return "BRENT"
    if "OPEC" in e:
        return "OPEC"
    raise ValueError(f"Cannot map exposure_id='{exposure_id}' to one of {{WTI,BRENT,OPEC}}.")


@dataclass
class CCCGarchProxyStrategy(HedgeStrategy):
    """CCC-GARCH proxy strategy based on precomputed h_ccc_proxy columns."""

    exposure_id: str
    corr_window: int = 120  # one of 30/60/120/252
    h_col: Optional[str] = None  # optional override from factory/config
    clip_abs_h: Optional[float] = 3.0
    fallback_h: float = 0.0
    name: str = "CCCGarchProxyStrategy"

    def __post_init__(self) -> None:
        if self.corr_window not in (30, 60, 120, 252):
            raise ValueError("corr_window must be one of {30, 60, 120, 252}.")
        if self.clip_abs_h is not None and self.clip_abs_h <= 0:
            raise ValueError("clip_abs_h must be positive or None.")

        self.prefix = _prefix_from_exposure_id(self.exposure_id)

        # Allow the strategy factory to pass an explicit hedge-ratio column name.
        # If provided, we do NOT overwrite it; otherwise derive from exposure + window.
        if self.h_col is None or str(self.h_col).strip() == "":
            self.h_col = f"{self.prefix}_h_ccc_proxy_{self.corr_window}"
        else:
            self.h_col = str(self.h_col)

    def build_h_path(
        self,
        dS: np.ndarray,
        dF: np.ndarray,
        dates=None,
        scenario_meta: Optional[Dict[str, Any]] = None,
        spot: Optional[np.ndarray] = None,
        fut: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Preferred fast API: compute hedge ratios over the full trade window.

        This strategy consumes precomputed hedge ratio columns from MasterData.
        The simulator is expected to pass the window's h-column values via `scenario_meta[self.h_col]` as a NumPy array.
        """
        scenario_meta = scenario_meta or {}
        arr = scenario_meta.get(self.h_col)
        n = len(dS)
        if isinstance(arr, np.ndarray):
            h = np.asarray(arr, dtype=float)
            if len(h) != n:
                # best effort align: truncate or pad with NaN
                if len(h) > n:
                    h = h[:n]
                else:
                    h = np.pad(h, (0, n - len(h)), constant_values=np.nan)
        else:
            # If not provided, fallback to constant
            h = np.full(n, np.nan, dtype=float)

        # Forward-fill within the trade window
        if n > 0:
            mask = np.isfinite(h)
            if mask.any():
                # ffill: carry last seen finite value forward
                last = np.nan
                for i in range(n):
                    if np.isfinite(h[i]):
                        last = h[i]
                    else:
                        h[i] = last
            # clip + fallback
            h = np.where(np.isfinite(h), h, float(self.fallback_h))
            if self.clip_abs_h is not None:
                h = np.clip(h, -float(self.clip_abs_h), float(self.clip_abs_h))
        return h.astype(float)

    # ------------------------
    # Integration helpers
    # ------------------------

    def required_feature_cols(self) -> List[str]:
        """Columns that WindowEngine should include for this strategy."""
        return [self.h_col]

    def reset(self) -> None:
        """Simulator compatibility: no internal state to reset."""
        return None

    def get_h(self, t: int, hist: pd.DataFrame, scenario_row: dict) -> float:
        """Simulator API: return hedge ratio h_t at step t.

        `hist` is the window dataframe (pre-trade history + trade history up to current t).
        We take the latest available value of the precomputed hedge ratio column.
        If it is NaN, we forward-fill within `hist` and then take the last value.
        If still missing, fall back to `fallback_h`.
        """
        if self.h_col not in hist.columns:
            raise ValueError(
                f"Required feature column '{self.h_col}' not found in hist. "
                f"Available: {list(hist.columns)}"
            )
        # Use last known non-NaN in the column up to current t (cheap scan backwards)
        col = hist[self.h_col]
        val = np.nan
        # iterate backwards until a finite value is found
        for v in reversed(col.to_numpy()):
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                val = fv
                break
        return self._sanitize_h(val)

    # ------------------------
    # Hedge ratio retrieval
    # ------------------------

    def _sanitize_h(self, h: float) -> float:
        if np.isnan(h):
            return float(self.fallback_h)
        if self.clip_abs_h is not None:
            h = float(np.clip(h, -self.clip_abs_h, self.clip_abs_h))
        return float(h)

    def _ffill_series(self, s: pd.Series) -> pd.Series:
        # Keep dtype numeric; forward-fill within trade window only.
        s2 = pd.to_numeric(s, errors="coerce").copy()
        return s2.ffill()

    def get_h_static(self, trade_window: pd.DataFrame) -> float:
        """Return one fixed hedge ratio for the whole trade.

        Policy: pick the first non-NaN h within the trade window; if none, fallback.
        """
        if self.h_col not in trade_window.columns:
            raise ValueError(
                f"Required column '{self.h_col}' not found in trade_window. "
                f"Available: {list(trade_window.columns)}"
            )

        s = self._ffill_series(trade_window[self.h_col])
        # first value after ffill (if the window starts with NaN and later has value,
        # ffill won't fill backward; so pick first non-NaN directly)
        first_valid = s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan
        return self._sanitize_h(float(first_valid) if not pd.isna(first_valid) else np.nan)

    def get_h_path_dynamic(self, trade_window: pd.DataFrame) -> pd.Series:
        """Return daily hedge ratio series aligned to trade_window index."""
        if self.h_col not in trade_window.columns:
            raise ValueError(
                f"Required column '{self.h_col}' not found in trade_window. "
                f"Available: {list(trade_window.columns)}"
            )
        s = self._ffill_series(trade_window[self.h_col])
        if s.isna().all():
            # all missing -> constant fallback
            return pd.Series(self.fallback_h, index=trade_window.index, dtype=float)
        if self.clip_abs_h is not None:
            s = s.clip(lower=-self.clip_abs_h, upper=self.clip_abs_h)
        # remaining NaNs (leading) -> fallback
        s = s.fillna(self.fallback_h)
        return s.astype(float)


# Backward-compatible alias (in case make_strategy expects this name)
CCCGarchStrategy = CCCGarchProxyStrategy