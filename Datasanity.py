# -*- coding: utf-8 -*-
"""
Step 0 — Data Audit + Data Dictionary Generator
Inputs:
  - data/MasterData.csv   (or a custom path via CLI)
Outputs:
  - reports/data_qc_report.md
  - reports/data_dictionary.md
  - reports/figures/*.png   (optional, controlled by flags)

Design principles:
  - Avoid over-processing feature-only columns (no returns/ACF/PACF unless selected).
  - Produce thesis/paper-grade QC tables + standard plots for core series.
  - Reproducible: everything derived from config + deterministic rules.

Run examples:
  python step0_data_audit.py --input "/mnt/data/MasterData.csv" --outdir "reports" --make_plots 1
  python step0_data_audit.py --input "data/MasterData.csv" --outdir "reports" --make_plots 0
"""

import os
import re
import json
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# Optional stats/plots
from scipy import stats
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.stattools import acf, pacf
    STATS_MODELS_OK = True
except Exception:
    STATS_MODELS_OK = False


# -----------------------------
# Config (edit as needed)
# -----------------------------
CORE_PRICE_COLS = [
    "WTI", "Brent", "OPEC", "CL1", "CL2"
]

# Columns that are "prices/levels" (eligible for returns + ACF/PACF)
# Keep this conservative; add more only if you're sure they are level series.
LEVEL_COLS_EXTRA = [
    "VIX", "OVX", "DXY", "TNX", "DTB3", "TB3M"  # examples (edit to match your columns)
]

# For ACF/PACF we usually analyze RETURNS (stationary-ish) rather than levels.
# We'll do returns only for SELECTED series to avoid pointless work.
DEFAULT_ACF_PACF_SERIES = ["WTI", "Brent", "OPEC", "CL1", "CL2"]

# Feature-only columns (we will compute basic QC + missingness + descriptive stats only)
# Leave empty to auto-infer (everything not in level cols) as feature-only.
FEATURE_ONLY_COLS_MANUAL = []


# Thresholds for "paper-grade" QC flags
MISSING_WARN = 0.05      # >5% missing => warning
MISSING_HIGH = 0.20      # >20% missing => high missing
OUTLIER_Z_WARN = 6.0     # |z|>6 on returns => potential outlier day (core series)
NEGATIVE_EVENT_COLS = ["WTI", "CL1", "CL2", "Brent", "OPEC"]  # track negative price events
ZERO_EVENT_COLS = ["WTI", "CL1", "CL2", "Brent", "OPEC"]      # track zero price events

# -----------------------------
# Column metadata (thesis-ready)
# -----------------------------
# Fill what we know; anything missing stays blank and can be completed later.
# role_in_model:
#   - "baseline"        => used in classical hedges / baselines
#   - "ai_feature"      => only used as AI feature (not required for baselines)
#   - "both"            => used in both
COLUMN_METADATA = {
    # --- Core spot / index series ---
    "WTI": {
        "economic_definition": "WTI crude oil spot price",
        "unit": "USD/bbl",
        "source": "EIA (Cushing, OK WTI Spot) or equivalent provider used to build MasterData",
        "native_frequency": "Daily (trading days)",
        "alignment": "Daily calendar; forward-fill for missing dates (no trading on non-settlement days)",
        "role_in_model": "both",
    },
    "Brent": {
        "economic_definition": "Brent crude oil spot price",
        "unit": "USD/bbl",
        "source": "EIA / ICE / equivalent provider used to build MasterData",
        "native_frequency": "Daily (trading days)",
        "alignment": "Daily calendar; forward-fill for missing dates (no trading on non-settlement days)",
        "role_in_model": "both",
    },
    "OPEC": {
        "economic_definition": "OPEC Reference Basket price/index",
        "unit": "USD/bbl (index/spot series)",
        "source": "OPEC (OPEC Reference Basket) or equivalent provider used to build MasterData",
        "native_frequency": "Daily (as available) / published series",
        "alignment": "Daily calendar; forward-fill within OPEC universe only",
        "role_in_model": "both",
    },

    # --- Futures (hedging instrument) ---
    "CL1": {
        "economic_definition": "NYMEX WTI Light Sweet Crude Oil Futures — front-month settlement",
        "unit": "USD/bbl",
        "source": "CME/NYMEX settlements (via data vendor used to build MasterData)",
        "native_frequency": "Daily (trading days)",
        "alignment": "Daily calendar; forward-fill only for alignment (no trading on non-settlement days)",
        "role_in_model": "baseline",
    },
    "CL2": {
        "economic_definition": "NYMEX WTI Light Sweet Crude Oil Futures — second month settlement",
        "unit": "USD/bbl",
        "source": "CME/NYMEX settlements (via data vendor used to build MasterData)",
        "native_frequency": "Daily (trading days)",
        "alignment": "Daily calendar; forward-fill only for alignment (no trading on non-settlement days)",
        "role_in_model": "ai_feature",
    },

    # --- Volatility indices ---
    "VIX": {
        "economic_definition": "CBOE Volatility Index",
        "unit": "Index level (annualized implied vol, percent units)",
        "source": "CBOE / FRED (VIXCLS) or equivalent provider used to build MasterData",
        "native_frequency": "Daily (trading days)",
        "alignment": "Daily calendar; forward-fill",
        "role_in_model": "ai_feature",
    },
    "OVX": {
        "economic_definition": "CBOE Crude Oil Volatility Index",
        "unit": "Index level (annualized implied vol, percent units)",
        "source": "CBOE (OVX) or equivalent provider used to build MasterData",
        "native_frequency": "Daily (trading days)",
        "alignment": "Daily calendar; forward-fill within availability window",
        "role_in_model": "ai_feature",
    },

    # --- FX / rates / macro (examples; edit to match your columns) ---
    "DXY": {
        "economic_definition": "US Dollar Index",
        "unit": "Index level",
        "source": "ICE / FRED / equivalent provider used to build MasterData",
        "native_frequency": "Daily (trading days)",
        "alignment": "Daily calendar; forward-fill",
        "role_in_model": "ai_feature",
    },
    "DTB3": {
        "economic_definition": "3-Month Treasury Bill: Secondary Market Rate",
        "unit": "Percent per annum",
        "source": "FRED (DTB3)",
        "native_frequency": "Daily",
        "alignment": "Daily calendar; forward-fill",
        "role_in_model": "ai_feature",
    },
    "TB3M": {
        "economic_definition": "3-Month Treasury Bill (monthly series, if present)",
        "unit": "Percent per annum",
        "source": "FRED (TB3MS) or equivalent",
        "native_frequency": "Monthly",
        "alignment": "Forward-fill to daily",
        "role_in_model": "ai_feature",
    },
    "TNX": {
        "economic_definition": "10Y Treasury yield proxy (if present)",
        "unit": "Percent per annum",
        "source": "Market data vendor / FRED equivalent",
        "native_frequency": "Daily (trading days)",
        "alignment": "Daily calendar; forward-fill",
        "role_in_model": "ai_feature",
    },

    # --- Policy uncertainty (examples) ---
    "GEPUCURRENT": {
        "economic_definition": "Global Economic Policy Uncertainty index (current)",
        "unit": "Index level",
        "source": "policyuncertainty.com / data vendor used to build MasterData",
        "native_frequency": "Monthly/weekly (varies by release)",
        "alignment": "Forward-fill to daily",
        "role_in_model": "ai_feature",
    },
    "EPUTRADE": {
        "economic_definition": "Trade Policy Uncertainty index",
        "unit": "Index level",
        "source": "policyuncertainty.com / data vendor used to build MasterData",
        "native_frequency": "Monthly/weekly (varies by release)",
        "alignment": "Forward-fill to daily",
        "role_in_model": "ai_feature",
    },

    # --- Sector equity (example) ---
    "SP500_ENERGY": {
        "economic_definition": "S&P 500 Energy sector index/return proxy (if present)",
        "unit": "Index level",
        "source": "S&P / data vendor used to build MasterData",
        "native_frequency": "Daily (trading days)",
        "alignment": "Daily calendar; forward-fill",
        "role_in_model": "ai_feature",
    },

    # --- World uncertainty index (example) ---
    "WTUI_global_gdpw": {
        "economic_definition": "World Trade Uncertainty Index (GDP-weighted, if present)",
        "unit": "Index level",
        "source": "Ahir, Bloom & Furceri (WTUI) / data vendor used to build MasterData",
        "native_frequency": "Quarterly (typical)",
        "alignment": "Forward-fill to daily",
        "role_in_model": "ai_feature",
    },
}


def get_column_metadata(col: str) -> dict:
    """Return thesis-ready metadata fields for a column (blank if unknown)."""
    base = {
        "economic_definition": "",
        "unit": "",
        "source": "",
        "native_frequency": "",
        "alignment": "",
        "role_in_model": "",
    }
    meta = COLUMN_METADATA.get(col, {})
    base.update(meta)
    return base


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def is_business_day(dt: pd.Timestamp) -> bool:
    # Monday=0 ... Sunday=6
    return dt.weekday() < 5

def md_table(df: pd.DataFrame, index: bool = True) -> str:
    # GitHub-flavored markdown table
    if df is None or df.empty:
        return "_(خالی)_\n"
    return df.to_markdown(index=index)

def describe_series(s: pd.Series) -> dict:
    # Robust descriptive stats (levels)
    s_clean = s.dropna().astype(float)
    if len(s_clean) == 0:
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "q01": np.nan, "q05": np.nan,
            "q50": np.nan, "q95": np.nan, "q99": np.nan, "max": np.nan,
            "skew": np.nan, "kurtosis_excess": np.nan
        }
    q = s_clean.quantile([0.01, 0.05, 0.50, 0.95, 0.99]).to_dict()
    return {
        "n": int(len(s_clean)),
        "mean": float(s_clean.mean()),
        "std": float(s_clean.std(ddof=1)) if len(s_clean) > 1 else np.nan,
        "min": float(s_clean.min()),
        "q01": float(q.get(0.01, np.nan)),
        "q05": float(q.get(0.05, np.nan)),
        "q50": float(q.get(0.50, np.nan)),
        "q95": float(q.get(0.95, np.nan)),
        "q99": float(q.get(0.99, np.nan)),
        "max": float(s_clean.max()),
        "skew": float(stats.skew(s_clean, bias=False)) if len(s_clean) > 2 else np.nan,
        "kurtosis_excess": float(stats.kurtosis(s_clean, fisher=True, bias=False)) if len(s_clean) > 3 else np.nan,
    }


def compute_simple_return(price: pd.Series) -> pd.Series:
    """Simple return: (P_t/P_{t-1}) - 1. Works with negative prices."""
    p = price.astype(float)
    return p / p.shift(1) - 1.0


def compute_log_return(price: pd.Series) -> pd.Series:
    """Log return: log(P_t/P_{t-1}). Only defined when both prices are > 0."""
    p = price.astype(float)
    ok = (p > 0) & (p.shift(1) > 0)
    lr = pd.Series(np.nan, index=p.index)
    lr.loc[ok] = np.log(p.loc[ok] / p.shift(1).loc[ok])
    return lr


def compute_hybrid_return(price: pd.Series) -> pd.DataFrame:
    """
    Hybrid return used to keep log-returns as default, but avoid breaking on non-positive prices.
    Returns a DataFrame with:
      - ret_log: log return when defined
      - ret_simple: simple return (always defined except missing)
      - ret_used: ret_log where defined else ret_simple
      - used_simple_flag: True when ret_simple used because log not defined
    """
    ret_simple = compute_simple_return(price)
    ret_log = compute_log_return(price)
    used_simple = ret_log.isna() & ret_simple.notna()
    ret_used = ret_log.copy()
    ret_used.loc[used_simple] = ret_simple.loc[used_simple]
    return pd.DataFrame({
        "ret_log": ret_log,
        "ret_simple": ret_simple,
        "ret_used": ret_used,
        "used_simple_flag": used_simple.astype(int),
    })

def rolling_non_trading_days(df: pd.DataFrame, date_col: str, price_cols: list) -> dict:
    """
    Defines "trading day" as a date where at least one of the key futures settlements exists.
    This avoids claiming trading/adjustment on weekends/holidays.
    """
    d = {}
    dates = df[date_col]
    # Core definition: day is tradable if a settlement UPDATE is observed.
    # This avoids treating forward-filled holidays/weekends as tradable.
    if "CL1" in df.columns:
        cl1 = df["CL1"].astype(float)
        tradable = cl1.notna() & (cl1 != cl1.shift(1))
    else:
        # fallback: any key price changed
        changed = pd.DataFrame({c: (df[c].astype(float) != df[c].astype(float).shift(1)) for c in price_cols if c in df.columns})
        tradable = df[price_cols].notna().any(axis=1) & changed.any(axis=1)

    total = len(df)
    non_trading = (~tradable).sum()
    weekend = dates.dt.weekday >= 5
    non_trading_weekend = ((~tradable) & weekend).sum()
    non_trading_weekday = non_trading - non_trading_weekend

    d["total_days"] = int(total)
    d["tradable_days"] = int(tradable.sum())
    d["non_tradable_days"] = int(non_trading)
    d["non_tradable_weekend"] = int(non_trading_weekend)
    d["non_tradable_weekday"] = int(non_trading_weekday)
    d["non_tradable_pct"] = float(non_trading / total) if total else np.nan
    return d

def acf_pacf_summary(x: pd.Series, nlags: int = 30) -> pd.DataFrame:
    """
    Returns a dataframe of ACF/PACF values for lags 1..nlags.
    Use on returns (recommended).
    """
    if not STATS_MODELS_OK:
        raise RuntimeError("statsmodels is not available; install statsmodels to compute ACF/PACF.")

    x = x.dropna().astype(float)
    if len(x) < (nlags + 5):
        return pd.DataFrame({"lag": [], "acf": [], "pacf": []})

    a = acf(x, nlags=nlags, fft=True)
    p = pacf(x, nlags=nlags, method="ywm")
    # exclude lag 0
    lags = np.arange(1, nlags + 1)
    return pd.DataFrame({
        "lag": lags,
        "acf": a[1:],
        "pacf": p[1:]
    })

def plot_timeseries(df: pd.DataFrame, date_col: str, col: str, outpath: str, title: str = None):
    plt.figure()
    plt.plot(df[date_col], df[col])
    plt.xlabel("Date")
    plt.ylabel(col)
    plt.title(title or f"Time Series: {col}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_missingness_heatmap(df: pd.DataFrame, outpath: str, max_cols: int = 35):
    """
    Simple missingness image (0=present, 1=missing). Paper-grade enough for appendix.
    """
    cols = df.columns.tolist()
    # keep it readable
    if len(cols) > max_cols:
        cols = cols[:max_cols]

    m = df[cols].isna().astype(int).T.values  # rows=cols
    plt.figure(figsize=(12, max(4, len(cols) * 0.25)))
    plt.imshow(m, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(cols)), cols, fontsize=8)
    plt.xticks([])
    plt.title("Missingness Heatmap (1=missing)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_return_hist(ret: pd.Series, outpath: str, title: str):
    r = ret.dropna().astype(float)
    plt.figure()
    plt.hist(r, bins=80)
    plt.title(title)
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_acf_pacf_bars(acf_pacf_df: pd.DataFrame, outpath: str, title: str):
    if acf_pacf_df.empty:
        return
    lags = acf_pacf_df["lag"].values
    plt.figure(figsize=(10, 4))
    plt.bar(lags - 0.2, acf_pacf_df["acf"].values, width=0.4, label="ACF")
    plt.bar(lags + 0.2, acf_pacf_df["pacf"].values, width=0.4, label="PACF")
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -----------------------------
# Main audit pipeline
# -----------------------------
def run_audit(input_path: str, outdir: str, make_plots: bool, acf_pacf_series: list, nlags: int):
    ensure_dir(outdir)
    figdir = os.path.join(outdir, "figures")
    ensure_dir(figdir)

    # Read CSV
    df = pd.read_csv(input_path)

    # Find date column (robust)
    date_candidates = [c for c in df.columns if c.lower() in ["date", "dt", "time", "timestamp"]]
    if not date_candidates:
        # fallback: first column named like date
        date_candidates = [c for c in df.columns if re.search(r"date", c, re.IGNORECASE)]
    if not date_candidates:
        raise ValueError("No date column found. Expected a column named 'Date' or similar.")

    date_col = date_candidates[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # Coerce numeric columns (except date)
    for c in df.columns:
        if c == date_col:
            continue
        # Keep as numeric when possible, else NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Determine column roles
    level_cols = [c for c in (CORE_PRICE_COLS + LEVEL_COLS_EXTRA) if c in df.columns]
    all_cols = [c for c in df.columns if c != date_col]
    if FEATURE_ONLY_COLS_MANUAL:
        feature_only_cols = [c for c in FEATURE_ONLY_COLS_MANUAL if c in df.columns]
    else:
        feature_only_cols = [c for c in all_cols if c not in level_cols]

    # -----------------------
    # Table 1: Column coverage (start/end, missing)
    # -----------------------
    coverage_rows = []
    for c in all_cols:
        s = df[c]
        non_na = s.dropna()
        start = df.loc[s.notna(), date_col].min() if s.notna().any() else pd.NaT
        end = df.loc[s.notna(), date_col].max() if s.notna().any() else pd.NaT
        missing_pct = float(s.isna().mean())
        coverage_rows.append({
            "column": c,
            "role": "LEVEL" if c in level_cols else "FEATURE",
            "start_date": str(start.date()) if pd.notna(start) else "",
            "end_date": str(end.date()) if pd.notna(end) else "",
            "n_obs": int(non_na.shape[0]),
            "missing_pct": missing_pct,
            "missing_flag": ("HIGH" if missing_pct >= MISSING_HIGH else ("WARN" if missing_pct >= MISSING_WARN else "OK"))
        })
    coverage_df = pd.DataFrame(coverage_rows).sort_values(["role", "missing_pct"], ascending=[True, False])

    # -----------------------
    # Table 2: Descriptive stats (levels) — all columns, but keep it readable
    # -----------------------
    desc_rows = []
    for c in all_cols:
        d = describe_series(df[c])
        d["column"] = c
        d["role"] = "LEVEL" if c in level_cols else "FEATURE"
        desc_rows.append(d)
    desc_df = pd.DataFrame(desc_rows).set_index("column")
    # reorder
    desc_df = desc_df[["role", "n", "mean", "std", "min", "q01", "q05", "q50", "q95", "q99", "max", "skew", "kurtosis_excess"]]
    # sort for readability: core first, then others
    ordering = {c: i for i, c in enumerate(CORE_PRICE_COLS)}
    desc_df["_ord"] = [ordering.get(c, 999) for c in desc_df.index]
    desc_df = desc_df.sort_values(["_ord", "role"]).drop(columns=["_ord"])

    # -----------------------
    # Table 3: Non-trading day diagnostics (based on CL1 availability / key prices)
    # -----------------------
    non_trading = rolling_non_trading_days(df, date_col, [c for c in CORE_PRICE_COLS if c in df.columns])
    non_trading_df = pd.DataFrame([non_trading])

    # -----------------------
    # Table 4: Negative/Zero price events (core)
    # -----------------------
    neg_events = []
    zero_events = []
    for c in NEGATIVE_EVENT_COLS:
        if c in df.columns:
            hits = df.loc[df[c].notna() & (df[c] < 0), [date_col, c]]
            for _, row in hits.iterrows():
                neg_events.append({"column": c, "date": str(row[date_col].date()), "value": float(row[c])})
    for c in ZERO_EVENT_COLS:
        if c in df.columns:
            hits = df.loc[df[c].notna() & (df[c] == 0), [date_col, c]]
            for _, row in hits.iterrows():
                zero_events.append({"column": c, "date": str(row[date_col].date()), "value": float(row[c])})
    neg_df = pd.DataFrame(neg_events).sort_values(["date", "column"]) if neg_events else pd.DataFrame(columns=["column", "date", "value"])
    zero_df = pd.DataFrame(zero_events).sort_values(["date", "column"]) if zero_events else pd.DataFrame(columns=["column", "date", "value"])

    # -----------------------
    # Returns + outlier checks for core only
    # -----------------------
    ret_stats_rows = []
    outlier_rows = []
    returns = {}  # per-series DataFrame with ret_log/ret_simple/ret_used/used_simple_flag
    for c in [x for x in CORE_PRICE_COLS if x in df.columns]:
        r = compute_hybrid_return(df[c])
        returns[c] = r

        # Use ret_log as the primary reporting metric, but stats/outliers on ret_used
        rc = r["ret_log"].dropna()
        ru = r["ret_used"].dropna()
        used_simple_cnt = int(r["used_simple_flag"].sum())

        if len(ru) > 0:
            ret_stats_rows.append({
                "series": c,
                "n_ret_used": int(len(ru)),
                "n_ret_log": int(len(rc)),
                "n_used_simple": used_simple_cnt,
                "mean_used": float(ru.mean()),
                "std_used": float(ru.std(ddof=1)) if len(ru) > 1 else np.nan,
                "min_used": float(ru.min()),
                "q01_used": float(ru.quantile(0.01)),
                "q05_used": float(ru.quantile(0.05)),
                "q50_used": float(ru.quantile(0.50)),
                "q95_used": float(ru.quantile(0.95)),
                "q99_used": float(ru.quantile(0.99)),
                "max_used": float(ru.max()),
                "skew_used": float(stats.skew(ru, bias=False)) if len(ru) > 2 else np.nan,
                "kurtosis_excess_used": float(stats.kurtosis(ru, fisher=True, bias=False)) if len(ru) > 3 else np.nan,
            })

            # outlier detection on ret_used by z-score
            z = (ru - ru.mean()) / (ru.std(ddof=1) if ru.std(ddof=1) and ru.std(ddof=1) > 0 else np.nan)
            if np.isfinite(z).any():
                out_idx = z.index[np.abs(z.values) >= OUTLIER_Z_WARN]
                for i in out_idx[:200]:
                    outlier_rows.append({
                        "series": c,
                        "date": str(df.loc[i, date_col].date()),
                        "return_used": float(ru.loc[i]),
                        "zscore": float(z.loc[i]),
                        "used_simple_flag": int(r.loc[i, "used_simple_flag"]),
                    })

    ret_stats_df = pd.DataFrame(ret_stats_rows).set_index("series") if ret_stats_rows else pd.DataFrame()
    outliers_df = pd.DataFrame(outlier_rows).sort_values(["series", "date"]) if outlier_rows else pd.DataFrame(columns=["series","date","return_used","zscore","used_simple_flag"])

    # -----------------------
    # ACF/PACF for selected series (returns) — optional + only if statsmodels installed
    # -----------------------
    acf_pacf_outputs = {}
    if STATS_MODELS_OK:
        for c in acf_pacf_series:
            if c in returns:
                ap = acf_pacf_summary(returns[c]["ret_used"], nlags=nlags)
                acf_pacf_outputs[c] = ap
    else:
        # we'll note in report
        pass

    # -----------------------
    # Plots (optional, only for core series to keep it paper-grade not bloated)
    # -----------------------
    plot_paths = []
    if make_plots:
        # Missingness heatmap (all cols)
        p = os.path.join(figdir, "missingness_heatmap.png")
        plot_missingness_heatmap(df.drop(columns=[date_col]), p, max_cols=35)
        plot_paths.append(p)

        # Core series time series plots
        for c in [x for x in CORE_PRICE_COLS if x in df.columns]:
            p = os.path.join(figdir, f"ts_{c}.png")
            plot_timeseries(df, date_col, c, p, title=f"{c} (Level)")
            plot_paths.append(p)

            # returns histogram
            p = os.path.join(figdir, f"hist_ret_{c}.png")
            plot_return_hist(returns[c]["ret_used"], p, title=f"{c} — ret_used (hybrid) Returns Histogram")
            plot_paths.append(p)

            # acf/pacf bars
            if STATS_MODELS_OK and c in acf_pacf_outputs and not acf_pacf_outputs[c].empty:
                p = os.path.join(figdir, f"acf_pacf_ret_{c}.png")
                plot_acf_pacf_bars(acf_pacf_outputs[c], p, title=f"{c} — ACF/PACF of ret_used (hybrid) Returns")
                plot_paths.append(p)

    # -----------------------
    # Export thesis-ready CSV tables
    # -----------------------
    tables_dir = os.path.join(outdir, "tables")
    ensure_dir(tables_dir)

    # 1) Coverage / missingness
    coverage_out = coverage_df.copy()
    coverage_out.to_csv(os.path.join(tables_dir, "01_coverage_missingness.csv"), index=False)

    # 2) Descriptive stats (levels)
    desc_out = desc_df.reset_index().rename(columns={"index": "column"})
    desc_out.to_csv(os.path.join(tables_dir, "02_descriptive_levels.csv"), index=False)

    # 3) Non-trading diagnostics
    non_trading_df.to_csv(os.path.join(tables_dir, "03_non_trading_diagnostics.csv"), index=False)

    # 4) Negative / zero events
    neg_df.to_csv(os.path.join(tables_dir, "04_negative_price_events.csv"), index=False)
    zero_df.to_csv(os.path.join(tables_dir, "05_zero_price_events.csv"), index=False)

    # 5) Returns (core) stats + outliers
    if not ret_stats_df.empty:
        ret_stats_df.reset_index().to_csv(os.path.join(tables_dir, "06_core_returns_stats_hybrid.csv"), index=False)
    outliers_df.to_csv(os.path.join(tables_dir, "07_core_return_outliers.csv"), index=False)

    # 6) ACF/PACF outputs (per series)
    if STATS_MODELS_OK and acf_pacf_outputs:
        for c, apdf in acf_pacf_outputs.items():
            apdf.to_csv(os.path.join(tables_dir, f"08_acf_pacf_ret_used_{c}.csv"), index=False)

    # 7) Feature availability windows (start/end)
    availability = coverage_df[["column", "role", "start_date", "end_date", "missing_pct", "missing_flag"]].copy()
    availability.to_csv(os.path.join(tables_dir, "09_feature_availability_windows.csv"), index=False)

    # -----------------------
    # Write data_dictionary.md (column-by-column)
    # -----------------------
    dd_lines = []
    dd_lines.append("# Data Dictionary (MasterData.csv)\n")
    dd_lines.append(f"- Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    dd_lines.append(f"- Input file: `{input_path}`\n")
    dd_lines.append("\n## Column Definitions (auto-generated)\n")
    dd_lines.append("> **توجه:** این فایل «دیکشنری اولیه» است. تعریف اقتصادی/منبع داده هر ستون باید توسط شما تکمیل شود.\n\n")

    for _, row in coverage_df.iterrows():
        c = row["column"]
        meta = get_column_metadata(c)
        tpl_role = "Level/Price" if row["role"] == "LEVEL" else "Feature/Exogenous"

        dd_lines.append(f"### `{c}`\n")
        dd_lines.append(f"- Role (data type): **{tpl_role}**\n")
        dd_lines.append(f"- Role (model use): `{meta['role_in_model']}`\n")
        dd_lines.append(f"- Economic definition: {meta['economic_definition']}\n")
        dd_lines.append(f"- Unit: `{meta['unit']}`\n")
        dd_lines.append(f"- Source: `{meta['source']}`\n")
        dd_lines.append(f"- Native frequency: `{meta['native_frequency']}`\n")
        dd_lines.append(f"- Alignment method: `{meta['alignment']}`\n")
        dd_lines.append(f"- Coverage: {row['start_date']} → {row['end_date']}  \n")
        dd_lines.append(f"- Missing: {row['missing_pct']:.2%} (flag: {row['missing_flag']})  \n")
        dd_lines.append(f"- Notes: \n\n")

    dd_path = os.path.join(outdir, "data_dictionary.md")
    with open(dd_path, "w", encoding="utf-8") as f:
        f.write("\n".join(dd_lines))

    # -----------------------
    # Write data_qc_report.md (paper-grade QC)
    # -----------------------
    qc_lines = []
    qc_lines.append("# Data QC Report — MasterData.csv\n")
    qc_lines.append(f"- Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    qc_lines.append(f"- Input file: `{input_path}`\n")
    qc_lines.append(f"- Rows: **{len(df):,}** | Columns (excluding date): **{len(all_cols):,}**\n")
    qc_lines.append(f"- Date range: **{df[date_col].min().date()} → {df[date_col].max().date()}**\n")

    qc_lines.append("\n## 1) Coverage & Missingness (per column)\n")
    qc_lines.append(md_table(coverage_df[["column","role","start_date","end_date","n_obs","missing_pct","missing_flag"]], index=False))
    qc_lines.append("\n**QC KPI:** ستون‌های با `missing_flag=HIGH` نیازمند تصمیم صریح هستند: حذف/جایگزینی/صرفاً فیچر.  \n")

    qc_lines.append("\n## 2) Descriptive Statistics (Levels)\n")
    qc_lines.append("> برای ستون‌های «فیچر» این آمار صرفاً جهت QC است و الزاماً برای مدل‌های پایه بازده/ACF محاسبه نمی‌شود.\n\n")
    # Keep table manageable: show core + top 15 most-missing features
    core_subset = [c for c in CORE_PRICE_COLS if c in desc_df.index]
    features_sorted = coverage_df.query("role=='FEATURE'").sort_values("missing_pct", ascending=False)["column"].tolist()
    show_cols = core_subset + features_sorted[:15]
    qc_lines.append(md_table(desc_df.loc[[c for c in show_cols if c in desc_df.index]], index=True))
    qc_lines.append("\n**QC KPI:** کشیدگی/چولگی بالا برای سطح قیمت‌ها طبیعی است؛ اما برای بازده‌ها باید با outlier و بحران‌ها کنترل شود.\n")

    qc_lines.append("\n## 3) Non-trading / Non-settlement Day Diagnostics\n")
    qc_lines.append("> تعریف روز قابل‌معامله: روزی که تسویه‌ی `CL1` موجود است (یا در نبود آن، حداقل یکی از سری‌های قیمت کلیدی موجود باشد).\n\n")
    qc_lines.append(md_table(non_trading_df, index=False))
    qc_lines.append("\n**تفسیر عملی:** اگر روز «غیرقابل معامله» باشد، در شبیه‌سازی **اجازه‌ی adjustment روزانه** نباید داده شود؛ یا باید صراحتاً گفت که قیمت forward-fill شده و معامله مجازی انجام شده است.\n")

    qc_lines.append("\n## 4) Negative Price Events (Core)\n")
    qc_lines.append(md_table(neg_df, index=False))
    qc_lines.append("\n## 5) Zero Price Events (Core)\n")
    qc_lines.append(md_table(zero_df, index=False))

    qc_lines.append("\n## 6) Returns QC (Core only) — Simple Returns\n")
    qc_lines.append("> بازده اصلی **لگاریتمی** است (log-return). برای روزهایی که قیمت غیرمثبت است، log-return تعریف‌پذیر نیست؛ بنابراین QC از بازده **hybrid** استفاده می‌کند: در حالت عادی log-return و در روزهای غیرقابل تعریف، simple return. تعداد روزهای استفاده از simple در جدول آمار بازده گزارش می‌شود.\n\n")
    if not ret_stats_df.empty:
        qc_lines.append(md_table(ret_stats_df, index=True))
    else:
        qc_lines.append("_(No core return stats computed)_\n")

    qc_lines.append("\n## 7) Return Outliers (Core) — Z-score threshold\n")
    qc_lines.append(f"> آستانه: |z| ≥ {OUTLIER_Z_WARN}. این‌ها الزاماً خطا نیستند؛ ممکن است شوک‌های واقعی بازار باشند.\n\n")
    qc_lines.append(md_table(outliers_df.head(80), index=False))
    if len(outliers_df) > 80:
        qc_lines.append(f"\n_(Showing first 80 of {len(outliers_df)} outlier rows to keep report readable.)_\n")

    qc_lines.append("\n## 8) ACF/PACF (Returns) — Selected Series\n")
    if STATS_MODELS_OK and acf_pacf_outputs:
        qc_lines.append(f"> nlags={nlags}. این بخش فقط برای سری‌های منتخب محاسبه شده تا گزارش حجیم نشود.\n\n")
        for c, apdf in acf_pacf_outputs.items():
            qc_lines.append(f"### {c}\n")
            qc_lines.append(md_table(apdf.head(nlags), index=False))
            qc_lines.append("\n")
    else:
        qc_lines.append("- `statsmodels` در محیط موجود نیست یا محاسبه غیرفعال است. برای ACF/PACF: `pip install statsmodels`\n")

    if make_plots:
        qc_lines.append("\n## 9) Figures\n")
        qc_lines.append("> مسیر فایل‌ها: `reports/figures/`\n\n")
        qc_lines.append("- Missingness heatmap: `figures/missingness_heatmap.png`\n")
        for c in [x for x in CORE_PRICE_COLS if x in df.columns]:
            qc_lines.append(f"- {c} level: `figures/ts_{c}.png`\n")
            qc_lines.append(f"- {c} ret_used (hybrid) histogram: `figures/hist_ret_{c}.png`\n")
            if STATS_MODELS_OK:
                qc_lines.append(f"- {c} ret_used (hybrid) ACF/PACF: `figures/acf_pacf_ret_{c}.png`\n")

    qc_path = os.path.join(outdir, "data_qc_report.md")
    with open(qc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(qc_lines))

    # Save a machine-readable summary (handy for pipelines)
    summary = {
        "input": input_path,
        "generated_utc": datetime.utcnow().isoformat(),
        "n_rows": int(len(df)),
        "n_cols_excl_date": int(len(all_cols)),
        "date_col": date_col,
        "date_min": str(df[date_col].min().date()),
        "date_max": str(df[date_col].max().date()),
        "non_trading": non_trading,
        "core_cols_found": [c for c in CORE_PRICE_COLS if c in df.columns],
        "plots_made": bool(make_plots),
        "statsmodels_ok": bool(STATS_MODELS_OK),
        "tables_dir": os.path.join(outdir, "tables"),
    }
    with open(os.path.join(outdir, "data_qc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("DONE.")
    print(f"QC report: {qc_path}")
    print(f"Data dictionary: {dd_path}")
    if make_plots:
        print(f"Figures: {figdir}")
    print(f"Summary JSON: {os.path.join(outdir, 'data_qc_summary.json')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="MasterData.csv", help="Path to MasterData.csv")
    p.add_argument("--outdir", type=str, default="reports", help="Output directory")
    p.add_argument("--make_plots", type=int, default=1, help="1 to generate figures, 0 otherwise")
    p.add_argument("--acf_pacf_series", type=str, default=",".join(DEFAULT_ACF_PACF_SERIES),
                   help="Comma-separated list of series names for ACF/PACF on returns")
    p.add_argument("--nlags", type=int, default=30, help="Number of lags for ACF/PACF")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    series = [s.strip() for s in args.acf_pacf_series.split(",") if s.strip()]
    run_audit(
        input_path=args.input,
        outdir=args.outdir,
        make_plots=bool(args.make_plots),
        acf_pacf_series=series,
        nlags=int(args.nlags)
    )