"""Precompute roll-aware futures PnL series + feature matrices for fast RL training.

RL training is bottlenecked by per-step Pandas operations. This module converts the
project's "Universe" (Date + spot + CL1/CL2 + selected features) into pure NumPy
arrays and precomputes the heavy/slow components:

- spot and dS
- roll-aware futures mark and per-contract daily PnL (pnl_1c), plus roll_flag
- tradable flag derived from RAW (pre-ffill) CL1 settlement updates (from DataAdapter)
- feature_matrix (float32)

The output is saved as a compressed .npz file per exposure.

Design notes:
- No look-ahead: all features must be causal by construction/config.
- Conservative defaults: if a flag is missing, fall back to safe assumptions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import numpy as np

# Ensure repository root is importable when running as a script (python rl/precompute.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dates_to_int(dates_np: np.ndarray) -> np.ndarray:
    """Store dates as int64 days since epoch (stable in npz)."""
    if np.issubdtype(dates_np.dtype, np.datetime64):
        d = dates_np.astype("datetime64[D]")
        return d.astype("int64")
    return np.array([np.datetime64(x, "D").astype("int64") for x in dates_np], dtype="int64")


def _stack_features(features: Dict[str, np.ndarray], n: int) -> Tuple[np.ndarray, List[str]]:
    """Stack feature dict into (n, k) float32 matrix with deterministic column order."""
    if not features:
        return np.zeros((n, 0), dtype=np.float32), []

    names = sorted(list(features.keys()))
    cols: List[np.ndarray] = []
    for k in names:
        v = np.asarray(features[k])
        if v.ndim != 1:
            v = v.reshape(-1)
        if len(v) != n:
            raise ValueError(f"Feature '{k}' length mismatch: {len(v)} != {n}")
        v = v.astype(np.float32, copy=False)
        v = np.where(np.isfinite(v), v, np.nan).astype(np.float32, copy=False)
        cols.append(v)

    X = np.column_stack(cols).astype(np.float32, copy=False)
    return X, names


def _safe_log_returns(x: np.ndarray) -> np.ndarray:
    """Causal log return helper; first value is zero and invalid moves are zeroed."""
    x = np.asarray(x, dtype=np.float64)
    r = np.zeros_like(x, dtype=np.float64)
    good = np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (x[1:] > 0.0) & (x[:-1] > 0.0)
    r[1:][good] = np.log(x[1:][good] / x[:-1][good])
    r[~np.isfinite(r)] = 0.0
    return r



def _shift1(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Lag by one step to avoid look-ahead in feature t."""
    y = np.empty_like(np.asarray(x, dtype=np.float64))
    y[0] = fill
    y[1:] = x[:-1]
    return y


def _rolling_sum_excluding_current(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sum using observations t-window ... t-1 for feature at t."""
    x = np.asarray(x, dtype=np.float64)
    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")
    x_clean = np.where(np.isfinite(x), x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(x_clean)])
    out = np.zeros_like(x_clean, dtype=np.float64)
    for t in range(len(x_clean)):
        lo = max(0, t - window)
        hi = t
        out[t] = cs[hi] - cs[lo]
    return out


def _rolling_mean_excluding_current(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean using observations t-window ... t-1 for feature at t."""
    x = np.asarray(x, dtype=np.float64)
    window = int(window)
    if window <= 0:
        raise ValueError("window must be positive")
    finite = np.isfinite(x)
    x_clean = np.where(finite, x, 0.0)
    cs_x = np.concatenate([[0.0], np.cumsum(x_clean)])
    cs_n = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
    out = np.zeros_like(x_clean, dtype=np.float64)
    for t in range(len(x_clean)):
        lo = max(0, t - window)
        hi = t
        count = cs_n[hi] - cs_n[lo]
        if count > 0:
            out[t] = (cs_x[hi] - cs_x[lo]) / count
        else:
            out[t] = 0.0
    return out


def _safe_ratio_minus_one(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Return num / den - 1 with non-finite and zero-denominator values set to zero."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.zeros_like(num, dtype=np.float64)
    good = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[good] = (num[good] / den[good]) - 1.0
    out[~np.isfinite(out)] = 0.0
    return out


def _ewma_variance(r: np.ndarray, lam: float = 0.94, min_var: float = 1e-10) -> np.ndarray:
    """Fast causal variance proxy used as fallback when heavy GARCH estimation is unavailable."""
    r = np.asarray(r, dtype=np.float64)
    out = np.empty_like(r, dtype=np.float64)
    finite = r[np.isfinite(r)]
    init = float(np.nanvar(finite[: min(len(finite), 252)])) if len(finite) else min_var
    init = max(init, min_var)
    out[0] = init
    for i in range(1, len(r)):
        prev_r2 = float(r[i - 1] ** 2) if np.isfinite(r[i - 1]) else 0.0
        out[i] = max((lam * out[i - 1]) + ((1.0 - lam) * prev_r2), min_var)
    return out


def _garch11_variance_proxy(r: np.ndarray, omega: float = 1e-8, alpha: float = 0.06, beta: float = 0.92) -> np.ndarray:
    """Causal GARCH(1,1)-style recursive variance proxy.

    This is not an MLE refit. It is deliberately fast and stable for precompute/runs.
    For thesis finalization, it can be replaced by rl/garch_features.py with rolling MLE.
    """
    r = np.asarray(r, dtype=np.float64)
    out = np.empty_like(r, dtype=np.float64)
    finite = r[np.isfinite(r)]
    init = float(np.nanvar(finite[: min(len(finite), 252)])) if len(finite) else 1e-8
    out[0] = max(init, 1e-10)
    for i in range(1, len(r)):
        prev_r2 = float(r[i - 1] ** 2) if np.isfinite(r[i - 1]) else 0.0
        out[i] = max(omega + alpha * prev_r2 + beta * out[i - 1], 1e-10)
    return out


def _igarch_variance_proxy(r: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    """Causal IGARCH-style recursive variance proxy: beta = 1 - alpha."""
    return _garch11_variance_proxy(r, omega=0.0, alpha=alpha, beta=1.0 - alpha)


def _egarch_variance_proxy(r: np.ndarray, omega: float = -0.05, alpha: float = 0.10, gamma: float = -0.05, beta: float = 0.94) -> np.ndarray:
    """Causal EGARCH-style recursive variance proxy with leverage term."""
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    out = np.empty(n, dtype=np.float64)
    finite = r[np.isfinite(r)]
    init = float(np.nanvar(finite[: min(len(finite), 252)])) if len(finite) else 1e-8
    logh = np.log(max(init, 1e-10))
    out[0] = np.exp(logh)
    expected_abs_z = np.sqrt(2.0 / np.pi)
    for i in range(1, n):
        prev_h = max(out[i - 1], 1e-10)
        z = float(r[i - 1] / np.sqrt(prev_h)) if np.isfinite(r[i - 1]) else 0.0
        z = float(np.clip(z, -10.0, 10.0))
        logh = omega + beta * logh + alpha * (abs(z) - expected_abs_z) + gamma * z
        logh = float(np.clip(logh, -30.0, 5.0))
        out[i] = max(np.exp(logh), 1e-10)
    return out


def _dcc_correlation_proxy(rs: np.ndarray, rf: np.ndarray, lam: float = 0.97) -> Tuple[np.ndarray, np.ndarray]:
    """Fast causal DCC-style EWMA covariance/correlation proxy.

    Uses only information up to t-1 for feature at t.
    """
    rs = np.asarray(rs, dtype=np.float64)
    rf = np.asarray(rf, dtype=np.float64)
    n = len(rs)
    cov = np.zeros(n, dtype=np.float64)
    corr = np.zeros(n, dtype=np.float64)
    vs = _ewma_variance(rs, lam=lam)
    vf = _ewma_variance(rf, lam=lam)
    c = 0.0
    for i in range(1, n):
        prev_prod = float(rs[i - 1] * rf[i - 1]) if np.isfinite(rs[i - 1]) and np.isfinite(rf[i - 1]) else 0.0
        c = lam * c + (1.0 - lam) * prev_prod
        cov[i] = c
        denom = np.sqrt(max(vs[i], 1e-10) * max(vf[i], 1e-10))
        corr[i] = float(np.clip(c / denom, -0.999, 0.999)) if denom > 0 else 0.0
    return cov, corr


def _build_builtin_garch_feature_dict(
    *,
    dates_int: np.ndarray,
    spot: np.ndarray,
    futures_mark: np.ndarray,
    futures_pnl_1c: np.ndarray,
    tradable: np.ndarray,
    exposure_id: str,
    config_path: Optional[str] = None,
    models: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Built-in fallback GARCH feature generator.

    This keeps precompute runnable even before a dedicated rl/garch_features.py exists.
    It creates causal volatility/covariance features using stable recursive proxies.
    """
    del dates_int, futures_pnl_1c, exposure_id, config_path  # reserved for external implementation compatibility
    models_set = set(models or ["garch", "igarch", "egarch", "dcc"])

    rs = _safe_log_returns(spot)
    rf = _safe_log_returns(futures_mark)
    tradable_bool = np.asarray(tradable, dtype=np.int8) > 0
    rs = np.where(tradable_bool, rs, 0.0)
    rf = np.where(tradable_bool, rf, 0.0)

    feats: Dict[str, np.ndarray] = {}

    basis = np.asarray(spot, dtype=np.float64) - np.asarray(futures_mark, dtype=np.float64)
    basis_ret = np.zeros_like(basis, dtype=np.float64)
    basis_ret[1:] = np.diff(basis)
    basis_ret[~np.isfinite(basis_ret)] = 0.0
    feats["basis_lag1"] = _shift1(np.where(np.isfinite(basis), basis, 0.0))
    feats["basis_change_lag1"] = _shift1(basis_ret)

    # Causal trend / momentum features. These are deliberately simple and fast.
    # Feature at t uses only information available up to t-1, so they can be used
    # by the end-of-day decision for the next hedge position without look-ahead.
    for w in (3, 5, 10, 20):
        feats[f"spot_mom_{w}d_lag1"] = _rolling_sum_excluding_current(rs, w)
        feats[f"fut_mom_{w}d_lag1"] = _rolling_sum_excluding_current(rf, w)

    spot_ma_5 = _rolling_mean_excluding_current(spot, 5)
    spot_ma_10 = _rolling_mean_excluding_current(spot, 10)
    spot_ma_20 = _rolling_mean_excluding_current(spot, 20)
    fut_ma_5 = _rolling_mean_excluding_current(futures_mark, 5)
    fut_ma_10 = _rolling_mean_excluding_current(futures_mark, 10)
    fut_ma_20 = _rolling_mean_excluding_current(futures_mark, 20)

    feats["spot_ma_5_20_gap_lag1"] = _safe_ratio_minus_one(spot_ma_5, spot_ma_20)
    feats["spot_ma_10_20_gap_lag1"] = _safe_ratio_minus_one(spot_ma_10, spot_ma_20)
    feats["fut_ma_5_20_gap_lag1"] = _safe_ratio_minus_one(fut_ma_5, fut_ma_20)
    feats["fut_ma_10_20_gap_lag1"] = _safe_ratio_minus_one(fut_ma_10, fut_ma_20)

    # Recent spot/futures co-movement and basis trend proxies.
    feats["basis_mom_5d_lag1"] = _rolling_sum_excluding_current(basis_ret, 5)
    feats["basis_mom_10d_lag1"] = _rolling_sum_excluding_current(basis_ret, 10)
    feats["spot_minus_fut_mom_5d_lag1"] = feats["spot_mom_5d_lag1"] - feats["fut_mom_5d_lag1"]
    feats["spot_minus_fut_mom_10d_lag1"] = feats["spot_mom_10d_lag1"] - feats["fut_mom_10d_lag1"]

    if "garch" in models_set:
        hs = _garch11_variance_proxy(rs)
        hf = _garch11_variance_proxy(rf)
        feats["garch_spot_vol_lag1"] = np.sqrt(hs)
        feats["garch_fut_vol_lag1"] = np.sqrt(hf)
        feats["garch_vol_spread_lag1"] = np.sqrt(hs) - np.sqrt(hf)

    if "igarch" in models_set:
        hs_i = _igarch_variance_proxy(rs)
        hf_i = _igarch_variance_proxy(rf)
        feats["igarch_spot_vol_lag1"] = np.sqrt(hs_i)
        feats["igarch_fut_vol_lag1"] = np.sqrt(hf_i)

    if "egarch" in models_set:
        hs_e = _egarch_variance_proxy(rs)
        hf_e = _egarch_variance_proxy(rf)
        feats["egarch_spot_vol_lag1"] = np.sqrt(hs_e)
        feats["egarch_fut_vol_lag1"] = np.sqrt(hf_e)
        feats["egarch_vol_spread_lag1"] = np.sqrt(hs_e) - np.sqrt(hf_e)

    if "dcc" in models_set:
        cov, corr = _dcc_correlation_proxy(rs, rf)
        hf_for_hr = _garch11_variance_proxy(rf)
        feats["dcc_cov_sf_lag1"] = cov
        feats["dcc_corr_sf_lag1"] = corr
        feats["dcc_hr_lag1"] = np.where(hf_for_hr > 1e-10, cov / hf_for_hr, 0.0)

    basis_var = _ewma_variance(basis_ret, lam=0.94)
    feats["basis_vol_lag1"] = np.sqrt(basis_var)

    clean: Dict[str, np.ndarray] = {}
    n = len(spot)
    for name, arr in feats.items():
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if len(arr) != n:
            raise ValueError(f"Built-in GARCH feature '{name}' length mismatch: {len(arr)} != {n}")
        arr = np.where(np.isfinite(arr), arr, 0.0)
        clean[name] = arr.astype(np.float32)
    return clean


@dataclass(frozen=True)
class PrecomputeResult:
    exposure_id: str
    dates_int: np.ndarray
    spot: np.ndarray
    dS: np.ndarray
    cl1: np.ndarray
    cl2: Optional[np.ndarray]
    f_mark: np.ndarray
    pnl_1c: np.ndarray
    roll_flag: np.ndarray
    tradable: np.ndarray
    feature_matrix: np.ndarray
    feature_names: List[str]
    meta: Dict


def precompute_exposure(
    *,
    data_path: str,
    config_path: str,
    exposure_id: str,
    include_features: bool = True,
    feature_role: str = "ai_feature_only",
    include_garch_features: bool = False,
    garch_config_path: Optional[str] = None,
    garch_models: Optional[List[str]] = None,
) -> PrecomputeResult:
    """Precompute arrays for a single exposure."""

    from data_adapter import DataAdapter
    from price_engine import PriceEngineCL

    da = DataAdapter(data_path, config_path)

    # Fast arrays (already masked to safe tradable days for this exposure)
    u = da.get_universe_arrays_fast(
        exposure_id,
        include_features=include_features,
        feature_role=feature_role,
    )

    dates = u["dates"]
    spot = np.asarray(u["spot"], dtype=np.float64)
    cl1 = np.asarray(u["cl1"], dtype=np.float64)
    cl2 = u.get("cl2", None)
    if cl2 is not None:
        cl2 = np.asarray(cl2, dtype=np.float64)

    # dS (first element 0)
    dS = np.empty_like(spot)
    dS[0] = 0.0
    dS[1:] = spot[1:] - spot[:-1]

    # tradable flag
    tradable = u.get("tradable", None)
    if tradable is None:
        tradable = np.isfinite(cl1).astype(np.int8)
    else:
        tradable = np.asarray(tradable, dtype=np.int8)

    # Build roll-aware futures series using PriceEngineCL on the full universe df (not array-masked)
    df_u = da.get_universe(
        exposure_id,
        include_features=False,
        feature_role="both",
    )

    # Column names (provided by DataAdapter)
    cl1_col = u.get("cl1_col", "CL1")
    cl2_col = u.get("cl2_col", None)

    try:
        pe = PriceEngineCL(df_u, date_col=da.date_col, cl1_col=cl1_col, cl2_col=cl2_col)
    except TypeError:
        pe = PriceEngineCL(df_u, cl1_col=cl1_col, cl2_col=cl2_col)

    s_roll = pe.get_series(mode="roll")

    # Align engine outputs to masked universe dates
    dates_int = _dates_to_int(np.asarray(dates))
    eng_dates_int = _dates_to_int(np.asarray(s_roll[da.date_col].to_numpy()))

    f_mark_all = np.asarray(s_roll["F_mark"].to_numpy(), dtype=np.float64)
    pnl_1c_all = np.asarray(s_roll["pnl_1c"].to_numpy(), dtype=np.float64)
    roll_flag_all = np.asarray(s_roll["roll_flag"].to_numpy(), dtype=np.int8)

    pos = np.searchsorted(eng_dates_int, dates_int)
    if np.any(pos < 0) or np.any(pos >= len(eng_dates_int)):
        raise RuntimeError("Date alignment failed: universe dates out of engine range")
    if not np.all(eng_dates_int[pos] == dates_int):
        m = {int(d): i for i, d in enumerate(eng_dates_int)}
        idx = np.array([m.get(int(d), -1) for d in dates_int], dtype=int)
        if np.any(idx < 0):
            raise RuntimeError("Date alignment failed: some universe dates not found in engine")
        pos = idx

    f_mark = f_mark_all[pos]
    pnl_1c = pnl_1c_all[pos]
    roll_flag = roll_flag_all[pos]

    # ------------------------
    # Sanity: pnl_1c is a *daily* per-contract PnL series.
    # The first element (t=0) has no prior day by construction, so it can be NaN/inf.
    # For RL stability and to avoid NaN rewards, we conservatively set non-finite pnl_1c to 0
    # and mark those days as non-tradable (no reliable settlement move).
    # ------------------------
    bad_pnl = ~np.isfinite(pnl_1c)
    if bad_pnl.any():
        pnl_1c = pnl_1c.copy()
        pnl_1c[bad_pnl] = 0.0
        tradable = tradable.copy()
        tradable[bad_pnl] = 0

    # Enforce t=0 as safe baseline (common source of NaN due to diff)
    if len(pnl_1c) > 0:
        pnl_1c = pnl_1c.copy()
        pnl_1c[0] = 0.0
        tradable = tradable.copy()
        tradable[0] = 0

    assert np.isfinite(pnl_1c).all(), "pnl_1c still contains non-finite values after sanitation"

    # Feature matrix
    features = dict(u.get("features", {}) if include_features else {})

    # Optional econometric volatility/correlation features.
    # Important: these features must be generated causally by the helper module
    # (train/estimate only on information available up to t-1 when producing a t feature).
    # The implementation is intentionally kept outside this fast precompute file so that
    # heavy GARCH estimation code is isolated and can be unit-tested separately.
    if include_garch_features:
        try:
            from rl.garch_features import build_garch_feature_dict
        except ImportError:
            build_garch_feature_dict = _build_builtin_garch_feature_dict

        garch_features = build_garch_feature_dict(
            dates_int=dates_int,
            spot=spot,
            futures_mark=f_mark,
            futures_pnl_1c=pnl_1c,
            tradable=tradable,
            exposure_id=exposure_id,
            config_path=garch_config_path,
            models=garch_models,
        )
        overlap = sorted(set(features).intersection(garch_features))
        if overlap:
            raise ValueError(f"GARCH feature name collision: {overlap}")
        features.update(garch_features)

    X, feat_names = _stack_features(features, n=len(dates_int))

    meta = {
        "exposure_id": exposure_id,
        "n": int(len(dates_int)),
        "k": int(X.shape[1]),
        "feature_role": feature_role,
        "include_features": bool(include_features),
        "include_garch_features": bool(include_garch_features),
        "garch_config_path": garch_config_path,
        "garch_models": list(garch_models) if garch_models is not None else None,
        "cl1_col": cl1_col,
        "cl2_col": cl2_col,
    }

    return PrecomputeResult(
        exposure_id=exposure_id,
        dates_int=dates_int,
        spot=spot,
        dS=dS,
        cl1=cl1,
        cl2=cl2,
        f_mark=f_mark,
        pnl_1c=pnl_1c,
        roll_flag=roll_flag,
        tradable=tradable,
        feature_matrix=X,
        feature_names=feat_names,
        meta=meta,
    )


def save_npz(res: PrecomputeResult, out_path: str, overwrite: bool = False) -> str:
    outp = Path(out_path)
    _ensure_dir(outp.parent)
    if outp.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {outp}")

    payload: Dict[str, np.ndarray] = {
        "dates_int": np.asarray(res.dates_int, dtype=np.int64),
        "spot": np.asarray(res.spot, dtype=np.float64),
        "dS": np.asarray(res.dS, dtype=np.float64),
        "cl1": np.asarray(res.cl1, dtype=np.float64),
        "f_mark": np.asarray(res.f_mark, dtype=np.float64),
        "pnl_1c": np.asarray(res.pnl_1c, dtype=np.float64),
        "roll_flag": np.asarray(res.roll_flag, dtype=np.int8),
        "tradable": np.asarray(res.tradable, dtype=np.int8),
        "feature_matrix": np.asarray(res.feature_matrix, dtype=np.float32),
        "feature_names": np.asarray(res.feature_names, dtype=object),
        "meta_json": np.asarray(json.dumps(res.meta, ensure_ascii=False), dtype=object),
    }
    if res.cl2 is not None:
        payload["cl2"] = np.asarray(res.cl2, dtype=np.float64)

    np.savez_compressed(outp, **payload)
    return str(outp)


def load_npz(path: str) -> PrecomputeResult:
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta_json"].item()))
        feature_names = [str(x) for x in z["feature_names"].tolist()]
        cl2 = z["cl2"] if "cl2" in z.files else None
        return PrecomputeResult(
            exposure_id=str(meta.get("exposure_id")),
            dates_int=np.asarray(z["dates_int"], dtype=np.int64),
            spot=np.asarray(z["spot"], dtype=np.float64),
            dS=np.asarray(z["dS"], dtype=np.float64),
            cl1=np.asarray(z["cl1"], dtype=np.float64),
            cl2=(np.asarray(cl2, dtype=np.float64) if cl2 is not None else None),
            f_mark=np.asarray(z["f_mark"], dtype=np.float64),
            pnl_1c=np.asarray(z["pnl_1c"], dtype=np.float64),
            roll_flag=np.asarray(z["roll_flag"], dtype=np.int8),
            tradable=np.asarray(z["tradable"], dtype=np.int8),
            feature_matrix=np.asarray(z["feature_matrix"], dtype=np.float32),
            feature_names=feature_names,
            meta=meta,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute RL arrays per exposure")
    p.add_argument("--data", required=True, help="Path to MasterData.csv/parquet")
    p.add_argument("--config", required=True, help="Path to config.yaml")
    p.add_argument("--exposure", required=True, help="Exposure id, e.g., WTI/BRENT/OPEC")
    p.add_argument("--out_dir", default="rl_cache", help="Output directory")
    p.add_argument("--feature_role", default="ai_feature_only", help="Feature role filter")
    p.add_argument("--no_features", action="store_true", help="Disable feature matrix")
    p.add_argument("--garch_features", action="store_true", help="Append causal GARCH/IGARCH/EGARCH/DCC features to feature_matrix")
    p.add_argument("--garch_config", default=None, help="Optional YAML/JSON config for GARCH feature generation")
    p.add_argument(
        "--garch_models",
        default="garch,igarch,egarch,dcc",
        help="Comma-separated GARCH feature models to include, e.g. garch,igarch,egarch,dcc",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing cache")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    garch_models = [m.strip().lower() for m in str(args.garch_models).split(",") if m.strip()]

    res = precompute_exposure(
        data_path=args.data,
        config_path=args.config,
        exposure_id=args.exposure,
        include_features=not args.no_features,
        feature_role=args.feature_role,
        include_garch_features=bool(args.garch_features),
        garch_config_path=args.garch_config,
        garch_models=garch_models,
    )

    out_path = Path(args.out_dir) / f"precompute_{args.exposure}.npz"
    saved = save_npz(res, str(out_path), overwrite=bool(args.overwrite))
    print(f"[precompute] saved: {saved}")
    print(f"[precompute] n={res.meta['n']}  k={res.meta['k']}")


if __name__ == "__main__":
    main()
