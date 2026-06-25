"""BaseGARCH.py

هدف
----
افزودن مجموعه‌ای کوچک و تمیز از ویژگی‌های مبتنی بر نوسان (GARCH) و همبستگی به فایل MasterData.

اصل ساده‌سازی (مهم)
-------------------
در این نسخه، برای پرهیز از پیچیدگی CL1/CL2 و رول‌اور، همه محاسبات ریسک/نوسان بر مبنای قرارداد آتی ماه نزدیک (CL1) انجام می‌شود.
با توجه به همبستگی بسیار بالای CL1 و CL2 در بیشتر دوره‌ها، این تقریب برای «حس‌گیری از بازار» و تولید فیچرهای پایدار قابل دفاع است.
رول‌اور و سری پرپچوال در موتور شبیه‌سازی/PNL جداگانه مدیریت می‌شود.

چه چیزی محاسبه می‌شود؟
-----------------------
برای هر exposure (WTI, Brent, OPEC) در برابر CL1:
- CL1_sigma_garch              : نوسان شرطی بازده CL1 با GARCH(1,1)
- X_sigma_spot_garch           : نوسان شرطی بازده Spot با GARCH(1,1)
- X_rho_{W}                    : همبستگی رولینگ W روزه بازده Spot و بازده CL1، برای W∈{30,60,120,252}
- X_h_ccc_proxy_{W}            : نسبت پوشش ریسک proxy متناظر با همان W، برای W∈{30,60,120,252}

نکته کنترلی مهم
---------------
برای جلوگیری از پر شدن ستون‌ها در روزهای فاقد داده‌ی Spot، ستون‌های وابسته به Spot با ماسک اعتبار (Spot و بازده‌ها) NaN می‌شوند.

خروجی
------
- results/MasterData_enriched_garch_clean.parquet
- results/MasterData_enriched_garch_clean.csv

اجرا
----
python BaseGARCH.py

وابستگی‌ها
-----------
arch, pandas, numpy
"""

from __future__ import annotations
import os

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# Prefer Parquet master in project root; fallback to CSV if needed
MASTER_PATH = "MasterData.parquet"
MASTER_FALLBACK_CSV = "MasterData.csv"

# Write outputs to project root by default (user moved MasterData to root)
OUT_DIR = "."
OUT_PARQUET = "MasterData_enriched_garch_clean.parquet"
OUT_CSV = "MasterData_enriched_garch_clean.csv"

# FeaturePack NPZ directory (fast loading)
FEATURE_DIR = "features"

# Map logical exposure names to actual MasterData columns
EXPOSURES = {
    "WTI": "WTI",
    "BRENT": "Brent",
    "OPEC": "OPEC",
}

FUT_COL = "CL1"  # simplified: use only CL1 for risk/feature computations

RHO_WINS = [30, 60, 120, 252]
MIN_GARCH_OBS = 750


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def simple_returns(series: pd.Series) -> pd.Series:
    """Simple returns without implicit padding/ffill."""
    return series.pct_change(fill_method=None)


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """EWMA (RiskMetrics-style) volatility fallback.

    Vectorized implementation:
    v_t = lam * v_{t-1} + (1-lam) * r_t^2
    sigma_t = sqrt(v_t)

    Notes
    -----
    - Uses exponential weighting on squared returns.
    - Preserves NaNs at timestamps where returns are NaN.
    """
    r = returns.astype(float)
    if r.dropna().empty:
        return pd.Series(index=r.index, dtype=float)

    alpha = 1.0 - float(lam)
    v = r.pow(2).ewm(alpha=alpha, adjust=False).mean()
    sigma = np.sqrt(v)
    sigma[r.isna()] = np.nan
    return sigma


def fit_garch_vol(returns: pd.Series) -> pd.Series:
    """Fit GARCH(1,1) and return conditional volatility.

    Stability rules:
    - Try dist='normal'
    - If it fails, try dist='t'
    - If it fails again, fallback to EWMA(0.94)
    """
    r = returns.dropna().astype(float)
    if len(r) < MIN_GARCH_OBS:
        return pd.Series(index=returns.index, dtype=float)

    r_scaled = r * 100.0

    for dist in ("normal", "t"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                am = arch_model(r_scaled, vol="GARCH", p=1, q=1, dist=dist, rescale=False)
                res = am.fit(disp="off", show_warning=False)
                vol = (res.conditional_volatility / 100.0)
                out = pd.Series(np.nan, index=returns.index, dtype=float)
                # conditional_volatility aligns with the fitted (non-NaN) return index
                out.loc[vol.index] = vol
                return out
            except Exception:
                pass

    # fallback
    return ewma_vol(returns, lam=0.94)


def _require_cols(df: pd.DataFrame, cols: list[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {ctx}: {missing}. Available: {list(df.columns)}")


def _load_master(path: str) -> pd.DataFrame:
    p = str(path)
    if os.path.exists(p) and p.lower().endswith(".parquet"):
        df = pd.read_parquet(p)
    elif os.path.exists(p) and p.lower().endswith(".csv"):
        df = pd.read_csv(p, parse_dates=["Date"], low_memory=False)
    else:
        # fallback to csv name if parquet missing
        if os.path.exists(MASTER_FALLBACK_CSV):
            df = pd.read_csv(MASTER_FALLBACK_CSV, parse_dates=["Date"], low_memory=False)
        else:
            raise FileNotFoundError(f"MasterData not found: {p} (or {MASTER_FALLBACK_CSV})")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:
    print("[BaseGARCH] Loading MasterData...")
    df = _load_master(MASTER_PATH)

    # Simple cache: if enriched parquet exists and is aligned to the same last date, skip recompute
    try:
        if os.path.exists(OUT_PARQUET):
            df_prev = pd.read_parquet(OUT_PARQUET)
            if "Date" in df_prev.columns:
                last_in = pd.to_datetime(df["Date"]).max()
                last_out = pd.to_datetime(df_prev["Date"]).max()
                if pd.notna(last_in) and pd.notna(last_out) and pd.Timestamp(last_in) == pd.Timestamp(last_out):
                    # Ensure core columns exist
                    core_cols = ["Date", FUT_COL, f"{FUT_COL}_sigma_garch"]
                    if all(c in df_prev.columns for c in core_cols):
                        print("[BaseGARCH] Enriched output already up-to-date. Skipping recompute.")
                        return
    except Exception:
        # If cache check fails, proceed with recompute
        pass

    _require_cols(df, ["Date", FUT_COL], ctx="MasterData")

    # Futures (CL1)
    fut = df[FUT_COL].astype(float)
    r_f = simple_returns(fut)

    print(f"[BaseGARCH] Fitting GARCH(1,1) for futures '{FUT_COL}'...")
    sigma_f = fit_garch_vol(r_f)
    df[f"{FUT_COL}_sigma_garch"] = sigma_f
    sigma_f2 = sigma_f ** 2

    # Exposure-specific features
    for name, spot_col in EXPOSURES.items():
        _require_cols(df, [spot_col], ctx="MasterData")

        print(f"[BaseGARCH] Computing features for {name} (spot='{spot_col}') vs {FUT_COL}...")

        spot = df[spot_col].astype(float)
        r_s = simple_returns(spot)

        print(f"[BaseGARCH]   - fitting GARCH(1,1) for {name} spot returns...")
        sigma_s = fit_garch_vol(r_s)

        # Store spot volatility
        df[f"{name}_sigma_spot_garch"] = sigma_s

        # Rolling correlations + CCC-proxy hedge ratios for multiple windows
        for W in RHO_WINS:
            min_p = max(20, W // 2)
            rho = r_s.rolling(W, min_periods=min_p).corr(r_f)
            cov = sigma_s * sigma_f * rho
            h_ccc = cov / sigma_f2

            df[f"{name}_rho_{W}"] = rho
            df[f"{name}_h_ccc_proxy_{W}"] = h_ccc

        # Validity mask (minimal): if spot price itself is missing, these features are undefined.
        valid_spot = spot.notna()
        mask_cols = [f"{name}_sigma_spot_garch"] + [
            f"{name}_rho_{W}" for W in RHO_WINS
        ] + [
            f"{name}_h_ccc_proxy_{W}" for W in RHO_WINS
        ]
        df.loc[~valid_spot, mask_cols] = np.nan

        # Write compact FeaturePack for fast loaders (arrays only)
        try:
            out_npz = Path(FEATURE_DIR) / f"features_garch_{name}.npz"
            np.savez_compressed(
                str(out_npz),
                Date=pd.to_datetime(df["Date"]).dt.normalize().to_numpy(dtype="datetime64[ns]"),
                CL1_sigma_garch=pd.to_numeric(df[f"{FUT_COL}_sigma_garch"], errors="coerce").to_numpy(dtype=float),
                spot_sigma_garch=pd.to_numeric(df[f"{name}_sigma_spot_garch"], errors="coerce").to_numpy(dtype=float),
                **{f"rho_{W}": pd.to_numeric(df[f"{name}_rho_{W}"], errors="coerce").to_numpy(dtype=float) for W in RHO_WINS},
                **{f"h_ccc_proxy_{W}": pd.to_numeric(df[f"{name}_h_ccc_proxy_{W}"], errors="coerce").to_numpy(dtype=float) for W in RHO_WINS},
            )
            print(f"[BaseGARCH]   - wrote FeaturePack: {out_npz}")
        except Exception as e:
            print(f"[BaseGARCH]   - FeaturePack write failed for {name}: {e}")

    # Save enriched master
    Path(OUT_DIR).mkdir(exist_ok=True)
    Path(FEATURE_DIR).mkdir(exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print(f"[BaseGARCH] Saved: {OUT_PARQUET}")
    print(f"[BaseGARCH] Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()