"""strategy/dcc_garch.py

DCC-GARCH (practical) hedge strategy.

Why this implementation?
------------------------
A fully estimated multivariate DCC-GARCH can be heavy and library-dependent.
For this project we need a reproducible, leakage-safe, and fast baseline that:
- uses the GARCH conditional volatilities (precomputed in MasterData)
- updates correlations dynamically via the DCC(1,1) recursion
- uses Numba for the DCC recursion when available (falls back to pure NumPy)

We therefore implement an explicit DCC(1,1) correlation update on standardized
returns z_t = r_t / sigma_t. The parameters (a,b) can be:
- fixed (default a=0.01, b=0.98) -> very stable and common in practice
- or grid-selected on the estimation window (optional)

Hedge ratio:
------------
Given spot volatility sigma_s,t, futures volatility sigma_f,t, and dynamic
correlation rho_t:
    h_t = (sigma_s,t / sigma_f,t) * rho_t

Compatibility:
--------------
Implements the simulator strategy API:
- build_h_path(...) (preferred)
- get_h(...) (legacy)

Required columns in hist:
- Spot price column (WTI/Brent/OPEC)
- CL1 futures price column
- CL1_sigma_garch
- <PREFIX>_sigma_spot_garch  (PREFIX in {WTI,BRENT,OPEC})

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore
    _HAS_NUMBA = False

from .base import HedgeStrategy


def _prefix_from_exposure_id(exposure_id: str) -> str:
    e = (exposure_id or "").upper()
    if "WTI" in e:
        return "WTI"
    if "BRENT" in e:
        return "BRENT"
    if "OPEC" in e:
        return "OPEC"
    raise ValueError(f"Cannot map exposure_id='{exposure_id}' to one of {{WTI,BRENT,OPEC}}.")


def _spot_col_from_prefix(prefix: str) -> str:
    # MasterData naming
    if prefix == "WTI":
        return "WTI"
    if prefix == "BRENT":
        return "Brent"
    if prefix == "OPEC":
        return "OPEC"
    raise ValueError(prefix)


def _simple_returns_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    out[:] = np.nan
    out[1:] = (x[1:] / x[:-1]) - 1.0
    return out


def _dcc_rho_from_z(z1: np.ndarray, z2: np.ndarray, a: float, b: float) -> float:
    """Compute last-step DCC correlation using recursion on standardized residuals.

    Uses:
      Q_t = (1-a-b) S + a z_{t-1} z_{t-1}' + b Q_{t-1}
      R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

    Returns rho_T (correlation at final time step).
    """
    if len(z1) != len(z2):
        raise ValueError("z1 and z2 must have same length")
    T = len(z1)
    if T < 5:
        return np.nan

    Z = np.column_stack([z1, z2])
    # unconditional covariance of z
    S = np.cov(Z.T)
    if not np.isfinite(S).all():
        return np.nan

    Q = S.copy()
    # recursion uses lagged z; start from t=1 to T-1 using z[t-1]
    for t in range(1, T):
        z_prev = Z[t - 1 : t, :].T  # 2x1
        Q = (1.0 - a - b) * S + a * (z_prev @ z_prev.T) + b * Q

    d = np.sqrt(np.diag(Q))
    if np.any(d <= 0) or not np.isfinite(d).all():
        return np.nan
    Dinv = np.diag(1.0 / d)
    R = Dinv @ Q @ Dinv
    rho = float(R[0, 1])
    # numerical clipping
    return float(np.clip(rho, -0.999, 0.999))


def _cov2_from_xy(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fast covariance elements for two series.

    Returns (s11, s22, s12) using population moments (divide by n), which is fine here.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= 1:
        return (np.nan, np.nan, np.nan)
    mx = float(np.mean(x))
    my = float(np.mean(y))
    dx = x - mx
    dy = y - my
    s11 = float(np.mean(dx * dx))
    s22 = float(np.mean(dy * dy))
    s12 = float(np.mean(dx * dy))
    return (s11, s22, s12)


if _HAS_NUMBA:

    @njit(cache=True)
    def _dcc_rho_path_numba(
        z1: np.ndarray,
        z2: np.ndarray,
        a: float,
        b: float,
        window: int,
    ) -> np.ndarray:
        """Numba-accelerated DCC(1,1) rho path (scalar 2x2 recursion)."""
        T = z1.shape[0]
        rho = np.empty(T, dtype=np.float64)
        for i in range(T):
            rho[i] = np.nan
        if T < 5:
            return rho

        # valid mask
        valid = np.empty(T, dtype=np.bool_)
        nvalid = 0
        for i in range(T):
            ok = (z1[i] == z1[i]) and (z2[i] == z2[i])  # not NaN
            valid[i] = ok
            if ok:
                nvalid += 1
        if nvalid < 5:
            return rho

        # Choose initial indices for S (first `window` valid, else all valid)
        use_w = window if (window > 0) else 0
        # Collect means over chosen set
        mx = 0.0
        my = 0.0
        cnt = 0
        for i in range(T):
            if not valid[i]:
                continue
            mx += z1[i]
            my += z2[i]
            cnt += 1
            if use_w > 0 and cnt >= use_w:
                break
        if cnt < 2:
            return rho
        mx /= cnt
        my /= cnt

        # Cov elements
        s11 = 0.0
        s22 = 0.0
        s12 = 0.0
        cnt2 = 0
        for i in range(T):
            if not valid[i]:
                continue
            dx = z1[i] - mx
            dy = z2[i] - my
            s11 += dx * dx
            s22 += dy * dy
            s12 += dx * dy
            cnt2 += 1
            if use_w > 0 and cnt2 >= use_w:
                break
        if cnt2 < 2:
            return rho
        s11 /= cnt2
        s22 /= cnt2
        s12 /= cnt2

        # Initialize Q = S
        q11 = s11
        q22 = s22
        q12 = s12

        # Recursion: rho[t] depends on z[t-1]
        for t in range(1, T):
            j = t - 1
            if valid[j]:
                z1p = z1[j]
                z2p = z2[j]
                outer11 = z1p * z1p
                outer22 = z2p * z2p
                outer12 = z1p * z2p

                q11 = (1.0 - a - b) * s11 + a * outer11 + b * q11
                q22 = (1.0 - a - b) * s22 + a * outer22 + b * q22
                q12 = (1.0 - a - b) * s12 + a * outer12 + b * q12

            # correlation
            if q11 > 0.0 and q22 > 0.0:
                denom = (q11 * q22) ** 0.5
                if denom > 0.0:
                    r = q12 / denom
                    if r > 0.999:
                        r = 0.999
                    elif r < -0.999:
                        r = -0.999
                    rho[t] = r

        return rho


def _dcc_rho_path_from_z(
    z1: np.ndarray,
    z2: np.ndarray,
    a: float,
    b: float,
    window: Optional[int] = None,
) -> np.ndarray:
    """Compute a rho_t path using a single-pass DCC(1,1) recursion.

    Notes
    -----
    - Leakage-safe within the provided arrays: rho[t] depends on z up to t-1.
    - Uses a fixed S estimated from the first available valid window (or all valid if shorter).
    - Skips updates when z at t-1 is invalid; keeps previous Q.

    Complexity: O(T) time, O(1) state.
    """
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    if len(z1) != len(z2):
        raise ValueError("z1 and z2 must have same length")

    T = len(z1)
    rho = np.full(T, np.nan, dtype=float)
    if T < 5:
        return rho

    valid = np.isfinite(z1) & np.isfinite(z2)
    idx = np.where(valid)[0]
    if len(idx) < 5:
        return rho

    w = int(window) if (window is not None) else 0
    if _HAS_NUMBA:
        # Numba path (fast)
        return _dcc_rho_path_numba(z1.astype(np.float64), z2.astype(np.float64), float(a), float(b), w)

    # Pure NumPy scalar recursion (still fast, avoids np.cov and 2x2 matrix ops)
    if w > 0 and len(idx) > w:
        idx0 = idx[:w]
    else:
        idx0 = idx

    s11, s22, s12 = _cov2_from_xy(z1[idx0], z2[idx0])
    if not (np.isfinite(s11) and np.isfinite(s22) and np.isfinite(s12)):
        return rho

    q11, q22, q12 = float(s11), float(s22), float(s12)

    for t in range(1, T):
        j = t - 1
        if valid[j]:
            z1p = float(z1[j])
            z2p = float(z2[j])
            outer11 = z1p * z1p
            outer22 = z2p * z2p
            outer12 = z1p * z2p
            q11 = (1.0 - a - b) * s11 + a * outer11 + b * q11
            q22 = (1.0 - a - b) * s22 + a * outer22 + b * q22
            q12 = (1.0 - a - b) * s12 + a * outer12 + b * q12

        if q11 > 0 and q22 > 0:
            denom = (q11 * q22) ** 0.5
            if denom > 0 and np.isfinite(denom):
                r = q12 / denom
                rho[t] = float(np.clip(r, -0.999, 0.999))

    return rho


def _grid_select_ab(z1: np.ndarray, z2: np.ndarray, grid_a: List[float], grid_b: List[float]) -> Tuple[float, float]:
    """Very small, deterministic grid search for (a,b).

    Objective: maximize average log(1 - rho_t^2) proxy (stability) by penalizing extreme rho.
    This is NOT full MLE, but gives a robust (a,b) for our baseline.
    """
    best = (0.01, 0.98)
    best_score = -np.inf

    for a in grid_a:
        for b in grid_b:
            if a <= 0 or b <= 0 or (a + b) >= 0.999:
                continue
            rho = _dcc_rho_from_z(z1, z2, a, b)
            if not np.isfinite(rho):
                continue
            # prefer rho away from +/-1
            score = -np.log(1.0 - min(0.999, rho * rho))
            # we want smaller score (less extreme), so invert
            score = -score
            if score > best_score:
                best_score = score
                best = (a, b)

    return best


@dataclass
class DCCGarchStrategy(HedgeStrategy):
    """DCC-GARCH dynamic hedge ratio using precomputed GARCH sigmas and DCC recursion."""

    exposure_id: str

    # estimation window length in days for computing z and rho
    window: int = 252

    # DCC(1,1) parameters (if grid_search=False)
    a: float = 0.01
    b: float = 0.98

    # optionally choose (a,b) by deterministic grid on each trade (or periodically)
    grid_search: bool = False
    grid_a: Tuple[float, ...] = (0.005, 0.01, 0.02)
    grid_b: Tuple[float, ...] = (0.95, 0.97, 0.98)

    # refit frequency for grid_search (days). If None, refit every day.
    refit_every: Optional[int] = 20

    # practical controls
    clip_abs_h: Optional[float] = 3.0
    fallback_h: float = 0.0

    name: str = "DCCGarchStrategy"

    def __post_init__(self) -> None:
        if self.window < 30:
            raise ValueError("window must be >= 30")
        self.prefix = _prefix_from_exposure_id(self.exposure_id)
        # WindowEngine provides standardized columns: `spot` and `fut`.
        # Keep the raw MasterData spot name only for reference.
        self._spot_col_raw = _spot_col_from_prefix(self.prefix)
        self.sigma_s_col = f"{self.prefix}_sigma_spot_garch"
        self.sigma_f_col = "CL1_sigma_garch"
        # WindowEngine provides `fut` as the held futures mark-to-market series.
        self._fut_col_raw = "CL1"

        # internal cache
        self._last_fit_t: Optional[int] = None
        self._ab: Tuple[float, float] = (float(self.a), float(self.b))

    def required_feature_cols(self) -> List[str]:
        return [self.sigma_s_col, self.sigma_f_col]

    def reset(self) -> None:
        self._last_fit_t = None
        self._ab = (float(self.a), float(self.b))

    def _sanitize_h(self, h: float) -> float:
        if np.isnan(h) or not np.isfinite(h):
            return float(self.fallback_h)
        if self.clip_abs_h is not None:
            h = float(np.clip(h, -self.clip_abs_h, self.clip_abs_h))
        return float(h)

    def _maybe_refit_ab(self, t: int, z1: np.ndarray, z2: np.ndarray) -> None:
        if not self.grid_search:
            return
        if self.refit_every is None:
            do = True
        else:
            do = (self._last_fit_t is None) or ((t - self._last_fit_t) >= self.refit_every)
        if not do:
            return

        a, b = _grid_select_ab(z1, z2, list(self.grid_a), list(self.grid_b))
        self._ab = (float(a), float(b))
        self._last_fit_t = int(t)

    def build_h_path(
        self,
        dS: np.ndarray,
        dF: np.ndarray,
        dates=None,
        scenario_meta: Optional[Dict[str, Any]] = None,
        spot: Optional[np.ndarray] = None,
        fut: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Preferred fast API: compute h_t path over the full trade window.

        Leakage-safe: at each t, uses only data up to t (within the provided window).
        """
        scenario_meta = scenario_meta or {}

        # We need price levels for returns; if not provided, reconstruct relative levels from diffs.
        if spot is None:
            spot_lvl = np.cumsum(np.asarray(dS, dtype=float))
        else:
            spot_lvl = np.asarray(spot, dtype=float)

        if fut is None:
            fut_lvl = np.cumsum(np.asarray(dF, dtype=float))
        else:
            fut_lvl = np.asarray(fut, dtype=float)

        # Get sigma arrays from scenario_meta (simulator should pass these arrays).
        sig_s = scenario_meta.get(self.sigma_s_col)
        sig_f = scenario_meta.get(self.sigma_f_col)

        sigma_s_arr = np.asarray(sig_s, dtype=float) if isinstance(sig_s, np.ndarray) else None
        sigma_f_arr = np.asarray(sig_f, dtype=float) if isinstance(sig_f, np.ndarray) else None

        n = len(spot_lvl)
        h = np.full(n, float(self.fallback_h), dtype=float)

        # If sigmas are not provided as arrays, we cannot proceed in this fast path.
        if sigma_s_arr is None or sigma_f_arr is None:
            return h

        r_s = _simple_returns_np(spot_lvl)
        r_f = _simple_returns_np(fut_lvl)

        # Standardized residuals
        valid = (
            np.isfinite(r_s)
            & np.isfinite(r_f)
            & np.isfinite(sigma_s_arr)
            & np.isfinite(sigma_f_arr)
            & (sigma_s_arr > 0)
            & (sigma_f_arr > 0)
        )

        z1 = np.full(n, np.nan, dtype=float)
        z2 = np.full(n, np.nan, dtype=float)
        z1[valid] = r_s[valid] / sigma_s_arr[valid]
        z2[valid] = r_f[valid] / sigma_f_arr[valid]

        # Reset strategy internal state
        self.reset()

        # If grid_search enabled, we occasionally refit (a,b) on trailing window.
        # Still O(T) for the core recursion; refit cost is extra and can be expensive.
        a, b = self._ab

        rho_path = _dcc_rho_path_from_z(z1, z2, a=float(a), b=float(b), window=int(self.window))

        if self.grid_search:
            # Periodically refit (a,b) using the most recent valid tail and recompute rho forward.
            # This is deterministic but can be slower; use larger refit_every for performance.
            for t in range(1, n):
                if self.refit_every is None:
                    do = True
                else:
                    do = (t == 1) or ((t % int(self.refit_every)) == 0)
                if not do:
                    continue
                # trailing valid window up to t
                tail_valid = np.isfinite(z1[: t + 1]) & np.isfinite(z2[: t + 1])
                idx = np.where(tail_valid)[0]
                if len(idx) < max(30, self.window // 2):
                    continue
                if len(idx) > self.window:
                    idx = idx[-self.window :]
                a2, b2 = _grid_select_ab(z1[idx], z2[idx], list(self.grid_a), list(self.grid_b))
                self._ab = (float(a2), float(b2))
                # recompute rho path from scratch with new params (still O(T))
                rho_path = _dcc_rho_path_from_z(z1, z2, a=float(a2), b=float(b2), window=int(self.window))

        # Build hedge ratio path (vectorized)
        mask = np.isfinite(rho_path) & valid
        if np.any(mask):
            h[mask] = (sigma_s_arr[mask] / sigma_f_arr[mask]) * rho_path[mask]

        # Apply clipping and fallback
        if self.clip_abs_h is not None:
            h = np.clip(h, -float(self.clip_abs_h), float(self.clip_abs_h))

        # Any invalid entries -> fallback
        h[~np.isfinite(h)] = float(self.fallback_h)
        h[~mask] = float(self.fallback_h)

        return h

    def get_h(self, t: int, hist: pd.DataFrame, scenario_row: dict) -> float:
        """Return hedge ratio h_t at step t.

        We compute rho_t from standardized returns on the trailing `window` observations
        ending at the current time (using only past information within hist).
        """
        # WindowEngine provides `spot` and `fut` columns; fall back to raw names if needed.
        spot_col = "spot" if "spot" in hist.columns else self._spot_col_raw
        fut_col = "fut" if "fut" in hist.columns else self._fut_col_raw

        for c in (spot_col, fut_col, self.sigma_s_col, self.sigma_f_col):
            if c not in hist.columns:
                raise ValueError(
                    f"DCC-GARCH requires column '{c}' in hist. Available: {list(hist.columns)}"
                )

        spot = pd.to_numeric(hist[spot_col], errors="coerce").to_numpy(dtype=float)
        fut = pd.to_numeric(hist[fut_col], errors="coerce").to_numpy(dtype=float)
        sigma_s = pd.to_numeric(hist[self.sigma_s_col], errors="coerce").to_numpy(dtype=float)
        sigma_f = pd.to_numeric(hist[self.sigma_f_col], errors="coerce").to_numpy(dtype=float)

        r_s = _simple_returns_np(spot)
        r_f = _simple_returns_np(fut)

        valid = np.isfinite(r_s) & np.isfinite(r_f) & np.isfinite(sigma_s) & np.isfinite(sigma_f) & (sigma_s > 0) & (sigma_f > 0)
        if np.sum(valid) < max(30, self.window // 2):
            return self._sanitize_h(np.nan)

        idx = np.where(valid)[0]
        if len(idx) > self.window:
            idx = idx[-self.window :]

        z1 = r_s[idx] / sigma_s[idx]
        z2 = r_f[idx] / sigma_f[idx]

        # Optional parameter update
        self._maybe_refit_ab(t, z1, z2)
        a, b = self._ab

        rho = _dcc_rho_from_z(z1, z2, a=a, b=b)

        sig_s_last = float(sigma_s[idx[-1]])
        sig_f_last = float(sigma_f[idx[-1]])
        if sig_f_last <= 0 or not np.isfinite(sig_s_last) or not np.isfinite(sig_f_last):
            return self._sanitize_h(np.nan)

        h = (sig_s_last / sig_f_last) * float(rho)
        return self._sanitize_h(h)