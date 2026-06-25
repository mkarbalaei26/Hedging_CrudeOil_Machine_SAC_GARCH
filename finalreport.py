

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finalreport.py

گزارش نهایی مقایسه RL-PPO با مدل‌های کلاسیک هجینگ.

این اسکریپت عمداً از فایل‌های خام `results/BASELINE_BATCH/**/hedge_summary_*.parquet`
می‌خواند، نه از فایل‌های merged قبلی؛ چون فایل‌های خام اطلاعات دقیق window/mode/roll
را در نام فایل حفظ کرده‌اند.

کارهای اصلی:
1) خواندن خروجی خام baselineها و استخراج window/mode/roll از نام فایل.
2) خواندن خروجی RL و استانداردسازی ستون‌های آن.
3) اتصال metadata سناریوها با scenario_id از پوشه scenarios، مخصوصاً:
   oracle_series, oracle_pool, oracle_freq, label=BEST/WORST.
4) تولید جداول تمیز در CSV و Excel.
5) تولید نمودارهای علمی با seaborn برای مقایسه RL با baselineها.

نمونه اجرا:
python finalreport.py \
  --asset OPEC_BASKET \
  --baseline_glob "results/BASELINE_BATCH/**/hedge_summary_*.parquet" \
  --rl_path "rl_runs/FINAL_OPEC_config351/results_all_windows.parquet" \
  --scenario_root scenarios \
  --out_dir reports/final_opec_rl_vs_baseline_scientific \
  --strict_intersection
"""

from __future__ import annotations

import argparse
import glob
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

RL_STRATEGY_NAME = "SAC"

STRATEGY_ORDER = [
    "NoHedge",
    "Naive",
    "OLSStatic",
    "OLSRolling",
    "CCCGarchProxyStrategy",
    "DCCGarchStrategy",
    RL_STRATEGY_NAME,
]

PRETTY_STRATEGY = {
    "NoHedge": "No Hedge",
    "Naive": "Naive Hedge",
    "OLSStatic": "Static OLS",
    "OLSRolling": "Rolling OLS",
    "CCCGarchProxyStrategy": "CCC-GARCH",
    "DCCGarchStrategy": "DCC-GARCH",
    RL_STRATEGY_NAME: "SAC-LPM",
}


PALETTE = {
    "NoHedge": "#7f7f7f",
    "Naive": "#4C78A8",
    "OLSStatic": "#72B7B2",
    "OLSRolling": "#54A24B",
    "CCCGarchProxyStrategy": "#F58518",
    "DCCGarchStrategy": "#B279A2",
    RL_STRATEGY_NAME: "#D62728",
}

# Helper: pretty_palette_for
from typing import Optional, Sequence, Dict
def pretty_palette_for(values: Optional[Sequence[object]] = None) -> Dict[str, str]:
    """Return a seaborn palette keyed by pretty strategy names.

    Seaborn validates palette dictionary keys against the plotted category labels.
    Our canonical PALETTE is keyed by raw strategy names, while plot columns often
    use `strategy_pretty`; this helper prevents missing-key errors.
    """
    pal = {PRETTY_STRATEGY.get(k, k): v for k, v in PALETTE.items()}
    if values is not None:
        for v in values:
            pal.setdefault(str(v), "#999999")
    return pal


# -----------------------------------------------------------------------------
# Filename metadata inference
# -----------------------------------------------------------------------------

WINDOW_RE_LIST = [
    re.compile(r"__w(?P<w>\d+|NA)__", re.IGNORECASE),
    re.compile(r"(?:^|[|_\-])w(?P<w>\d{2,4})(?:[|_\-]|$)", re.IGNORECASE),
    re.compile(r"(?:window|win)[_=\-]?(?P<w>\d{2,4})", re.IGNORECASE),
]
MODE_RE_LIST = [
    re.compile(r"__(?P<mode>dynamic|static)__", re.IGNORECASE),
    re.compile(r"_(?P<mode>dyn|dynamic|static)_", re.IGNORECASE),
]
ROLL_RE_LIST = [
    re.compile(r"__(?P<roll>roll|noroll|no_roll|no-roll)__", re.IGNORECASE),
    re.compile(r"_(?P<roll>roll|noroll|no_roll|no-roll)_", re.IGNORECASE),
]


def infer_window_from_text(x: object) -> Optional[int]:
    s = str(x)
    for rx in WINDOW_RE_LIST:
        m = rx.search(s)
        if m:
            val = m.group("w")
            if str(val).upper() == "NA":
                return None
            try:
                return int(val)
            except Exception:
                return None
    return None


def infer_mode_from_text(x: object) -> str:
    s = str(x)
    for rx in MODE_RE_LIST:
        m = rx.search(s)
        if m:
            val = m.group("mode").lower()
            return "dynamic" if val in {"dyn", "dynamic"} else "static"
    return "unknown"


def infer_roll_from_text(x: object) -> str:
    s = str(x)
    for rx in ROLL_RE_LIST:
        m = rx.search(s)
        if m:
            val = m.group("roll").lower().replace("_", "-")
            return "noroll" if val in {"noroll", "no-roll"} else "roll"
    return "unknown"


def infer_kind_from_text(x: object) -> str:
    s = str(x).lower()
    if "oracle_universe" in s:
        return "oracle_universe"
    if "oracle_all" in s:
        return "oracle_all"
    if "baseline" in s or "scenarios_baseline" in s:
        return "baseline"
    if "company" in s or "scenarios_company" in s:
        return "company"
    return "unknown"


def normalize_strategy_name(x: object) -> str:
    s = str(x)
    sl = s.lower().replace(" ", "_")
    if sl in {"rl", "rl_ppo", "ppo", "ppo_rl", "sac", "rl_sac", "sac_lpm", "rl_sac_lpm"}:
        return RL_STRATEGY_NAME
    if sl in {"nohedge", "no_hedge"}:
        return "NoHedge"
    if sl in {"naive", "naivehedge", "naive_hedge"}:
        return "Naive"
    if sl in {"olsstatic", "ols_static", "static_ols"}:
        return "OLSStatic"
    if sl in {"olsrolling", "ols_roll", "rolling_ols"}:
        return "OLSRolling"
    if "ccc" in sl and "garch" in sl:
        return "CCCGarchProxyStrategy"
    if "dcc" in sl and "garch" in sl:
        return "DCCGarchStrategy"
    return s


def pretty_strategy(x: object) -> str:
    return PRETTY_STRATEGY.get(str(x), str(x))


# -----------------------------------------------------------------------------
# Scenario metadata
# -----------------------------------------------------------------------------

SCENARIO_FILES = {
    "baseline": ["baseline.parquet", "scenarios_baseline.parquet"],
    "company": ["companies.parquet", "company.parquet", "scenarios_company.parquet"],
    "oracle_universe": ["oracle_universe.parquet"],
    "oracle_all": ["oracle_all.parquet"],
}


def parse_oracle_series(value: object) -> Tuple[str, str, str]:
    if value is None or pd.isna(value):
        return "unknown", "unknown", "unknown"
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return "unknown", "unknown", "unknown"
    parts = re.split(r"[_\-\s]+", s.upper())

    pool = "unknown"
    if "EXTREME" in parts:
        pool = "EXTREME"
    elif "FEASIBLE" in parts:
        pool = "FEASIBLE"

    freq = "unknown"
    for cand in ["DAILY", "WEEKLY", "MONTHLY", "QUARTERLY"]:
        if cand in parts:
            freq = cand
            break

    label = "unknown"
    if "BEST" in parts:
        label = "BEST"
    elif "WORST" in parts:
        label = "WORST"

    return pool, freq, label


def scenario_kind_from_oracle(row: pd.Series) -> str:
    kind_raw = str(row.get("scenario_kind", "unknown"))
    kind = kind_raw.strip().lower().replace("-", "_").replace(" ", "_")
    oracle_series = str(row.get("oracle_series", "")).upper()
    oracle_pool = str(row.get("oracle_pool", "")).upper()
    tag = str(row.get("tag", "")).upper()
    label = str(row.get("label", "")).upper()
    text = "__".join([kind.upper(), oracle_series, oracle_pool, tag, label])

    if "EXTREME" in text:
        return "oracle_extreme"
    if "FEASIBLE" in text:
        return "oracle_feasible"
    if kind.startswith("oracle_all"):
        return "oracle_all"
    return kind_raw


def canonicalize_oracle_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    for col in ["scenario_kind", "oracle_series", "oracle_pool", "label", "tag"]:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown")

    combined = (
        out["scenario_kind"].astype(str)
        + "__" + out["oracle_series"].astype(str)
        + "__" + out["oracle_pool"].astype(str)
        + "__" + out["label"].astype(str)
        + "__" + out["tag"].astype(str)
    ).str.upper().str.replace("-", "_", regex=False).str.replace(" ", "_", regex=False)

    label_clean = out["label"].astype(str).str.upper()
    missing_label = label_clean.isin(["", "NAN", "NONE", "<NA>", "UNKNOWN"])
    out.loc[missing_label & combined.str.contains("BEST", na=False), "label"] = "BEST"
    out.loc[missing_label & combined.str.contains("WORST", na=False), "label"] = "WORST"

    pool_clean = out["oracle_pool"].astype(str).str.upper()
    missing_pool = pool_clean.isin(["", "NAN", "NONE", "<NA>", "UNKNOWN"])
    out.loc[missing_pool & combined.str.contains("FEASIBLE", na=False), "oracle_pool"] = "FEASIBLE"
    out.loc[missing_pool & combined.str.contains("EXTREME", na=False), "oracle_pool"] = "EXTREME"

    out["scenario_kind"] = out.apply(scenario_kind_from_oracle, axis=1)
    return out


def expand_globs(patterns: Sequence[str]) -> List[Path]:
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted({Path(f) for f in files if str(f).lower().endswith((".parquet", ".csv"))})


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def resolve_rl_paths(paths):
    from pathlib import Path
    resolved = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            candidates = [
                p / "results_all_windows.parquet",
                p / "results_all_windows.csv",
                p / "results.parquet",
                p / "results.csv",
            ]
            found = [c for c in candidates if c.exists()]
            if found:
                resolved.append(found[0])
            else:
                matches = list(p.glob("**/results_all_windows.parquet"))
                matches += list(p.glob("**/results_all_windows.csv"))
                if matches:
                    resolved.append(matches[0])
                else:
                    print(f"[WARN] no RL results file in {p}")
        else:
            resolved.append(p)
    return resolved


# --------------------------------------------------------------------------
# Asset filtering helpers
# --------------------------------------------------------------------------

def normalize_assets_arg(assets: Optional[Sequence[str]]) -> Optional[List[str]]:
    if assets is None:
        return None
    out = [str(a).strip() for a in assets if str(a).strip()]
    return out or None


def filter_to_assets(df: pd.DataFrame, assets: Optional[Sequence[str]], *, stage: str = "") -> pd.DataFrame:
    """Hard asset filter used after every metadata operation.

    This prevents accidental cross-asset leakage when raw files or scenario metadata
    contain overlapping scenario_id values or when a previous merged layer is mixed.
    """
    assets_norm = normalize_assets_arg(assets)
    if not assets_norm or df.empty:
        return df
    if "exposure_id" not in df.columns:
        raise ValueError(f"Cannot filter to assets={assets_norm}: missing exposure_id at stage={stage}")
    before = len(df)
    out = df[df["exposure_id"].astype(str).isin(set(assets_norm))].copy()
    after = len(out)
    if before != after:
        print(f"[finalreport] asset filter at {stage}: {before:,} -> {after:,} rows; assets={assets_norm}")
    bad_assets = sorted(set(out["exposure_id"].astype(str).unique()) - set(assets_norm))
    if bad_assets:
        raise ValueError(f"Asset leakage after filter at {stage}: unexpected assets={bad_assets}; expected={assets_norm}")
    return out


def path_looks_like_asset(path: Path, assets: Optional[Sequence[str]]) -> bool:
    assets_norm = normalize_assets_arg(assets)
    if not assets_norm:
        return True
    s = str(path).upper()
    return any(asset.upper() in s for asset in assets_norm)


def load_scenario_metadata(scenario_root: Path, assets: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not scenario_root.exists():
        raise FileNotFoundError(f"Scenario root not found: {scenario_root}")

    assets = normalize_assets_arg(assets)

    # --- caching ---
    cache_asset_key = "ALL" if not assets else "__".join(sorted(assets))
    cache_path = scenario_root / f".scenario_metadata_cache__{cache_asset_key}.parquet"

    if cache_path.exists():
        try:
            print(f"[finalreport] loading metadata cache: {cache_path}")
            return pd.read_parquet(cache_path)
        except Exception:
            print("[WARN] cache broken, rebuilding")

    asset_dirs = [p for p in scenario_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if assets:
        allowed = set(assets)
        asset_dirs = [p for p in asset_dirs if p.name in allowed]

    wanted_cols = [
        "scenario_id", "scenario_record_id", "exposure_id", "scenario_kind",
        "start_date", "end_date", "horizon_days", "horizon_days_target", "horizon_days_realized",
        "volume_bbl", "oracle_series", "oracle_pool", "oracle_freq", "label", "tag",
        "oracle_candidate", "company_id", "company_size",
    ]

    frames: List[pd.DataFrame] = []
    for asset_dir in asset_dirs:
        asset = asset_dir.name
        for kind, filenames in SCENARIO_FILES.items():
            for filename in filenames:
                p = asset_dir / filename
                if not p.exists():
                    continue
                df = pd.read_parquet(p)
                if "scenario_id" not in df.columns:
                    continue
                keep = [c for c in wanted_cols if c in df.columns]
                df = df[keep].copy()
                if "exposure_id" not in df.columns:
                    df["exposure_id"] = asset
                if "scenario_kind" not in df.columns:
                    df["scenario_kind"] = kind
                df["scenario_file"] = str(p)
                df["scenario_id"] = df["scenario_id"].astype(str)
                df["exposure_id"] = df["exposure_id"].fillna(asset).astype(str)
                df["scenario_kind"] = df["scenario_kind"].fillna(kind).astype(str)
                frames.append(df)
                break

    if not frames:
        return pd.DataFrame()

    meta = pd.concat(frames, ignore_index=True)

    for col in ["start_date", "end_date"]:
        if col in meta.columns:
            meta[col] = pd.to_datetime(meta[col], errors="coerce")

    if "oracle_series" not in meta.columns:
        meta["oracle_series"] = pd.NA

    parsed = meta["oracle_series"].map(parse_oracle_series)
    parsed_pool = pd.Series([x[0] for x in parsed], index=meta.index)
    parsed_freq = pd.Series([x[1] for x in parsed], index=meta.index)
    parsed_label = pd.Series([x[2] for x in parsed], index=meta.index)

    for col, values in [("oracle_pool", parsed_pool), ("oracle_freq", parsed_freq), ("label", parsed_label)]:
        if col not in meta.columns:
            meta[col] = values
        else:
            cur = meta[col]
            miss = cur.isna() | cur.astype(str).str.lower().isin(["", "nan", "none", "<na>", "unknown"])
            meta.loc[miss, col] = values[miss]

    meta = canonicalize_oracle_fields(meta)
    meta = meta.drop_duplicates(subset=["scenario_id", "exposure_id"], keep="first")

    # save cache
    try:
        meta.to_parquet(cache_path, index=False)
        print(f"[finalreport] saved metadata cache: {cache_path}")
    except Exception as e:
        print(f"[WARN] cannot save cache: {e}")

    return meta



def attach_scenario_metadata(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if df.empty or meta.empty or "scenario_id" not in df.columns:
        return df

    out = df.copy()
    out["scenario_id"] = out["scenario_id"].astype(str)
    if "exposure_id" not in out.columns:
        out["exposure_id"] = pd.NA
    out["exposure_id"] = out["exposure_id"].astype(str)

    meta_cols = [
        "scenario_id", "exposure_id", "scenario_record_id", "start_date", "end_date",
        "horizon_days", "horizon_days_target", "horizon_days_realized", "volume_bbl",
        "oracle_series", "oracle_pool", "oracle_freq", "label", "tag",
        "oracle_candidate", "company_id", "company_size", "scenario_file",
    ]
    keep = [c for c in meta_cols if c in meta.columns]
    m = meta[keep].copy()

    merged = out.merge(m, on=["scenario_id", "exposure_id"], how="left", suffixes=("", "_meta"))

    for col in [c for c in meta_cols if c not in {"scenario_id", "exposure_id"}]:
        meta_col = f"{col}_meta"
        if meta_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[meta_col]
        else:
            cur = merged[col]
            miss = cur.isna()
            if cur.dtype == object or str(cur.dtype).startswith("string"):
                miss = miss | cur.astype(str).str.lower().isin(["", "nan", "none", "<na>", "unknown"])
            merged.loc[miss, col] = merged.loc[miss, meta_col]
        merged = merged.drop(columns=[meta_col])

    if "oracle_series" not in merged.columns:
        merged["oracle_series"] = "unknown"

    parsed = merged["oracle_series"].map(parse_oracle_series)
    for col, pos in [("oracle_pool", 0), ("oracle_freq", 1), ("label", 2)]:
        values = pd.Series([x[pos] for x in parsed], index=merged.index)
        if col not in merged.columns:
            merged[col] = values
        else:
            cur = merged[col]
            miss = cur.isna() | cur.astype(str).str.lower().isin(["", "nan", "none", "<na>", "unknown"])
            merged.loc[miss, col] = values[miss]

    merged = canonicalize_oracle_fields(merged)
    return merged


# -----------------------------------------------------------------------------
# Reading model outputs
# -----------------------------------------------------------------------------


def ensure_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "strategy" not in out.columns:
        out["strategy"] = "unknown"
    out["strategy"] = out["strategy"].map(normalize_strategy_name)

    if "scenario_kind" in out.columns:
        out = canonicalize_oracle_fields(out)

    if "scenario_id" in out.columns:
        out["scenario_id"] = out["scenario_id"].astype(str)

    for col in ["start_date", "end_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in [
        "spot_pnl_total", "fut_pnl_total", "cost_trade_total", "cost_roll_total", "net_pnl_total",
        "turnover_contracts", "turnover_h", "trade_contracts", "roll_contracts", "max_abs_contracts", "mdd_equity",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Only synthesize net PnL from decomposition if at least one decomposition leg exists.
    # This prevents RL rows from becoming fake zero-PnL rows when RL exports net PnL
    # under a different column name.
    has_decomposition = any(
        col in df.columns and out[col].notna().any()
        for col in ["spot_pnl_total", "fut_pnl_total", "cost_trade_total", "cost_roll_total"]
    )
    if out["net_pnl_total"].isna().all() and has_decomposition:
        out["net_pnl_total"] = (
            out["spot_pnl_total"].fillna(0)
            + out["fut_pnl_total"].fillna(0)
            - out["cost_trade_total"].fillna(0)
            - out["cost_roll_total"].fillna(0)
        )

    out["gross_pnl_total"] = out["spot_pnl_total"].fillna(0) + out["fut_pnl_total"].fillna(0)
    out["total_cost"] = out["cost_trade_total"].fillna(0) + out["cost_roll_total"].fillna(0)
    return out


def read_baseline_raw(files: Sequence[Path], assets: Optional[Sequence[str]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in files:
        if not path_looks_like_asset(p, assets):
            continue
        try:
            df = read_table(p)
        except Exception as exc:
            print(f"[WARN] cannot read baseline file {p}: {exc}")
            continue
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = str(p)

        file_kind = infer_kind_from_text(p)
        file_window = infer_window_from_text(p)
        file_mode = infer_mode_from_text(p)
        file_roll = infer_roll_from_text(p)

        if "scenario_kind" not in df.columns:
            df["scenario_kind"] = file_kind
        else:
            bad = df["scenario_kind"].isna() | df["scenario_kind"].astype(str).str.lower().isin(["", "nan", "none", "unknown"])
            df.loc[bad, "scenario_kind"] = file_kind

        if "window" not in df.columns:
            df["window"] = file_window
        else:
            df["window"] = pd.to_numeric(df["window"], errors="coerce")
            df["window"] = df["window"].fillna(file_window)

        if "mode" not in df.columns:
            df["mode"] = file_mode
        else:
            bad = df["mode"].isna() | df["mode"].astype(str).str.lower().isin(["", "nan", "none", "unknown"])
            df.loc[bad, "mode"] = file_mode

        if "roll" not in df.columns:
            df["roll"] = file_roll
        else:
            bad = df["roll"].isna() | df["roll"].astype(str).str.lower().isin(["", "nan", "none", "unknown"])
            df.loc[bad, "roll"] = file_roll

        if file_mode != "unknown":
            df["dynamic"] = 1 if file_mode == "dynamic" else 0
        elif "dynamic" not in df.columns:
            df["dynamic"] = 0

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = ensure_common_columns(out)

    out = filter_to_assets(out, assets, stage="baseline_raw")

    dedup_cols = [c for c in ["scenario_id", "exposure_id", "scenario_kind", "strategy", "window", "mode", "roll"] if c in out.columns]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="last")
    return out


def ingest_rl_results(paths: Sequence[Path], assets: Optional[Sequence[str]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            d = read_table(p)
        except Exception as exc:
            print(f"[WARN] cannot read RL file {p}: {exc}")
            continue
        if d.empty:
            continue
        d = d.copy()
        d["source_file"] = str(p)

        if "exposure_id" not in d.columns:
            sp = str(p).upper()
            if "OPEC" in sp:
                d["exposure_id"] = "OPEC_BASKET"
            elif "BRENT" in sp:
                d["exposure_id"] = "BRENT_SPOT"
            elif "WTI" in sp:
                d["exposure_id"] = "WTI_SPOT"
            else:
                d["exposure_id"] = "unknown"

        if "scenario_kind" not in d.columns:
            if "dataset" in d.columns:
                d["scenario_kind"] = d["dataset"].astype(str)
            else:
                d["scenario_kind"] = "oracle_universe"

        d["scenario_kind"] = d["scenario_kind"].astype(str).replace({
            "test:univ": "oracle_universe",
            "val:univ": "oracle_universe",
            "train:univ": "oracle_universe",
            "univ": "oracle_universe",
            "universe": "oracle_universe",
            "test:all": "oracle_all",
            "all": "oracle_all",
            "test:base": "baseline",
            "base": "baseline",
        })

        column_candidates = {
            "net_pnl_total": [
                "net_pnl_total", "pnl_net_sum", "episode_net_pnl", "episode_pnl",
                "net_pnl", "pnl_net", "pnl", "pnl_total", "total_pnl",
                "final_net_pnl", "net_profit", "episode_profit", "profit",
                "equity_pnl", "final_pnl", "total_net_pnl", "test_pnl",
                "realized_pnl", "portfolio_pnl"
            ],
            "cost_trade_total": ["cost_trade_total", "cost_sum", "trade_cost", "trade_cost_total", "cost_trade", "transaction_cost", "total_cost"],
            "cost_roll_total": ["cost_roll_total", "roll_cost", "roll_cost_total", "cost_roll"],
            "turnover_contracts": ["turnover_contracts", "turnover_contract", "turnover", "contracts_turnover", "contract_turnover"],
            "turnover_h": ["turnover_h", "hedge_turnover", "h_turnover"],
            "max_abs_contracts": ["max_abs_contracts", "max_contracts", "max_abs_position"],
            "mdd_equity": ["mdd_equity", "mdd", "max_drawdown", "episode_mdd", "maximum_drawdown"],
        }
        for target, candidates in column_candidates.items():
            if target not in d.columns:
                for cand in candidates:
                    if cand in d.columns:
                        d[target] = d[cand]
                        break

        d["strategy"] = d["strategy"].map(normalize_strategy_name) if "strategy" in d.columns else RL_STRATEGY_NAME
        d["strategy"] = d["strategy"].fillna(RL_STRATEGY_NAME)
        if "window" not in d.columns:
            d["window"] = d["source_file"].map(infer_window_from_text)
        if "mode" not in d.columns:
            d["mode"] = "dynamic"
        if "roll" not in d.columns:
            d["roll"] = "roll"
        if "dynamic" not in d.columns:
            d["dynamic"] = 1
        if "spot_pnl_total" not in d.columns:
            d["spot_pnl_total"] = np.nan
        if "fut_pnl_total" not in d.columns:
            d["fut_pnl_total"] = np.nan

        frames.append(d)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = ensure_common_columns(out)
    out = filter_to_assets(out, assets, stage="rl_raw")

    dedup_cols = [c for c in ["scenario_id", "exposure_id", "scenario_kind", "strategy", "window", "mode", "roll"] if c in out.columns]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="last")
    return out


# -----------------------------------------------------------------------------
# Alignment
# -----------------------------------------------------------------------------


def strict_intersection_with_rl(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only scenario_ids that are present in RL for the same exposure.

    This is intentionally NOT grouped by scenario_kind/oracle_series/label. In this
    project, oracle_all is a subset of oracle_universe and scenario_id can overlap
    across scenario files. Grouping by report-time buckets can wrongly remove
    oracle_universe rows. For a fair RL-vs-baseline comparison, the safest
    intersection key is exposure_id + scenario_id.
    """
    if df.empty or RL_STRATEGY_NAME not in set(df["strategy"].astype(str)):
        return df
    if "scenario_id" not in df.columns or "exposure_id" not in df.columns:
        raise ValueError("strict_intersection requires scenario_id and exposure_id columns")

    rl = df[df["strategy"].astype(str).eq(RL_STRATEGY_NAME)].copy()
    if rl.empty:
        return df.iloc[0:0].copy()

    key = rl[["exposure_id", "scenario_id"]].dropna().copy()
    key["exposure_id"] = key["exposure_id"].astype(str)
    key["scenario_id"] = key["scenario_id"].astype(str)
    key = key.drop_duplicates()
    key["_keep_common_with_rl"] = 1

    out = df.copy()
    out["exposure_id"] = out["exposure_id"].astype(str)
    out["scenario_id"] = out["scenario_id"].astype(str)
    out = out.merge(key, on=["exposure_id", "scenario_id"], how="inner")
    out = out.drop(columns=["_keep_common_with_rl"], errors="ignore")

    # Avoid exact duplicate rows after the merge.
    dedup_cols = [
        c for c in [
            "scenario_id", "exposure_id", "scenario_kind", "oracle_series", "label",
            "strategy", "window", "mode", "roll",
        ] if c in out.columns
    ]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols, keep="last")
    return out


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def safe_var(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    return float(np.var(x, ddof=1))


def value_at_risk(x: pd.Series, alpha: float = 0.95) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, 1 - alpha))


def expected_shortfall(x: pd.Series, alpha: float = 0.95) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    q = np.quantile(x, 1 - alpha)
    tail = x[x <= q]
    return float(tail.mean()) if len(tail) else np.nan


def downside_deviation(x: pd.Series, tau: float = 0.0) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    downside = np.minimum(x - tau, 0.0)
    return float(np.sqrt(np.mean(downside ** 2)))


# -----------------------------------------------------------------------------
# Lower Partial Moments (LPM) metrics
# -----------------------------------------------------------------------------

def lower_partial_moment(x: pd.Series, tau: float = 0.0, order: int = 2) -> float:
    """Lower Partial Moment around threshold tau.

    LPM_p(tau) = E[max(tau - X, 0)^p]. For PnL data with tau=0, this measures
    expected downside loss magnitude. order=2 is the usual semi-variance-style
    downside risk metric.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    downside = np.maximum(float(tau) - x, 0.0)
    return float(np.mean(downside ** int(order)))


def lower_partial_moment_root(x: pd.Series, tau: float = 0.0, order: int = 2) -> float:
    """Root-LPM in original currency units, useful for readable plots."""
    val = lower_partial_moment(x, tau=tau, order=order)
    if np.isnan(val):
        return np.nan
    return float(val ** (1.0 / int(order)))


def omega_ratio(x: pd.Series, tau: float = 0.0) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    gains = np.maximum(x - tau, 0.0).sum()
    losses = np.maximum(tau - x, 0.0).sum()
    return float(gains / losses) if losses > 0 else np.inf


def tail_ratio(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    q95 = np.quantile(x, 0.95)
    q05 = np.quantile(x, 0.05)
    return float(abs(q95 / q05)) if q05 != 0 else np.nan


def sortino_ratio(x: pd.Series, tau: float = 0.0) -> float:
    mean = pd.to_numeric(x, errors="coerce").mean()
    dd = downside_deviation(x, tau)
    return float((mean - tau) / dd) if dd and not np.isnan(dd) and dd > 0 else np.nan


def fit_distribution_summary(x: pd.Series) -> Dict[str, float | str]:
    x = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    if len(x) < 30 or stats is None:
        return {"dist_best_fit": "NA", "dist_aic_normal": np.nan, "dist_aic_student_t": np.nan, "t_df": np.nan}

    arr = x.to_numpy()
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    sigma = max(sigma, 1e-12)
    ll_norm = float(np.sum(stats.norm.logpdf(arr, loc=mu, scale=sigma)))
    aic_norm = 2 * 2 - 2 * ll_norm

    try:
        df_t, loc_t, scale_t = stats.t.fit(arr)
        scale_t = max(scale_t, 1e-12)
        ll_t = float(np.sum(stats.t.logpdf(arr, df_t, loc=loc_t, scale=scale_t)))
        aic_t = 2 * 3 - 2 * ll_t
        best = "Student-t" if aic_t < aic_norm else "Normal"
    except Exception:
        df_t, aic_t, best = np.nan, np.nan, "Normal"

    return {
        "dist_best_fit": best,
        "dist_aic_normal": float(aic_norm),
        "dist_aic_student_t": float(aic_t) if not np.isnan(aic_t) else np.nan,
        "t_df": float(df_t) if not np.isnan(df_t) else np.nan,
    }


def aggregate_metrics(df: pd.DataFrame, alpha: float = 0.95, tau: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d = canonicalize_oracle_fields(d)

    cat_cols = [
        "exposure_id", "scenario_kind", "oracle_series", "oracle_pool", "oracle_freq", "label",
        "strategy", "window", "mode", "roll",
    ]
    for c in cat_cols:
        if c not in d.columns:
            d[c] = "unknown"
        d[c] = d[c].fillna("unknown").astype(str)

    dedup_cols = [
        "scenario_id", "exposure_id", "scenario_kind", "oracle_series", "label",
        "strategy", "window", "mode", "roll",
    ]
    dedup_cols = [c for c in dedup_cols if c in d.columns]
    d = d.drop_duplicates(subset=dedup_cols, keep="last")

    group_cols = ["exposure_id", "scenario_kind", "oracle_series", "oracle_pool", "oracle_freq", "label", "strategy", "window", "mode", "roll"]
    group_cols_agg = ["exposure_id", "scenario_kind", "oracle_series", "oracle_pool", "oracle_freq", "label", "strategy"]

    def _agg(g: pd.DataFrame) -> Dict[str, float]:
        x = pd.to_numeric(g["net_pnl_total"], errors="coerce")
        gross = pd.to_numeric(g["gross_pnl_total"], errors="coerce")
        spot = pd.to_numeric(g.get("spot_pnl_total", pd.Series(index=g.index, dtype=float)), errors="coerce")
        mdd = pd.to_numeric(g.get("mdd_equity", pd.Series(index=g.index, dtype=float)), errors="coerce")
        out: Dict[str, float] = {
            "n_scenarios": int(g["scenario_id"].nunique()) if "scenario_id" in g.columns else int(len(g)),
            "n_rows": int(len(g)),
            "mean_net": float(x.mean()),
            "median_net": float(x.median()),
            "std_net": float(x.std(ddof=1)),
            "cv": float(abs(x.std(ddof=1) / x.mean())) if x.mean() not in [0, np.nan] and not np.isnan(x.mean()) else np.nan,
            "min_net": float(x.min()),
            "max_net": float(x.max()),
            "var_net": safe_var(x),
            "var_gross": safe_var(gross),
            "var_spot": safe_var(spot),
            "var95": value_at_risk(x, alpha),
            "es95": expected_shortfall(x, alpha),
            "downside_dev": downside_deviation(x, tau),
            "lpm1_tau0": lower_partial_moment(x, tau=tau, order=1),
            "lpm2_tau0": lower_partial_moment(x, tau=tau, order=2),
            "root_lpm2_tau0": lower_partial_moment_root(x, tau=tau, order=2),
            "omega": omega_ratio(x, tau),
            "tail_ratio": tail_ratio(x),
            "sortino_tau0": sortino_ratio(x, tau),
            "prob_profit": float((x > 0).mean()),
            "mdd_equity_mean": float(mdd.mean()),
            "mdd_equity_min": float(mdd.min()),
            "turnover_contracts_mean": float(pd.to_numeric(g.get("turnover_contracts", pd.Series(index=g.index)), errors="coerce").mean()),
            "turnover_h_mean": float(pd.to_numeric(g.get("turnover_h", pd.Series(index=g.index)), errors="coerce").mean()),
            "trade_contracts_mean": float(pd.to_numeric(g.get("trade_contracts", pd.Series(index=g.index)), errors="coerce").mean()),
            "roll_contracts_mean": float(pd.to_numeric(g.get("roll_contracts", pd.Series(index=g.index)), errors="coerce").mean()),
            "max_abs_contracts_mean": float(pd.to_numeric(g.get("max_abs_contracts", pd.Series(index=g.index)), errors="coerce").mean()),
            "mean_cost_trade": float(pd.to_numeric(g.get("cost_trade_total", pd.Series(index=g.index)), errors="coerce").mean()),
            "mean_cost_roll": float(pd.to_numeric(g.get("cost_roll_total", pd.Series(index=g.index)), errors="coerce").mean()),
            "mean_total_cost": float(pd.to_numeric(g.get("total_cost", pd.Series(index=g.index)), errors="coerce").mean()),
            "mean_spot": float(spot.mean()),
            "mean_gross": float(gross.mean()),
        }
        out["skew"] = float(x.skew()) if len(x.dropna()) >= 3 else np.nan
        out["kurtosis_excess"] = float(x.kurtosis()) if len(x.dropna()) >= 4 else np.nan
        return out

    metrics = d.groupby(group_cols, dropna=False).apply(lambda g: pd.Series(_agg(g))).reset_index()
    metrics_agg = d.groupby(group_cols_agg, dropna=False).apply(lambda g: pd.Series(_agg(g))).reset_index()

    # Distribution fit only for oracle_universe.
    dist_rows: List[Dict[str, object]] = []
    d_ou = d[d["scenario_kind"] == "oracle_universe"].copy()
    for keys, g in d_ou.groupby(group_cols_agg, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols_agg, keys))
        row.update(fit_distribution_summary(g["net_pnl_total"]))
        dist_rows.append(row)
    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        metrics_agg = metrics_agg.merge(dist_df, on=group_cols_agg, how="left")

    def _add_he(m: pd.DataFrame, granular: bool) -> pd.DataFrame:
        out = m.copy()
        id_cols = group_cols if granular else group_cols_agg
        bench_cols = [c for c in id_cols if c != "strategy"]

        no = out[out["strategy"].astype(str).str.lower().isin({"nohedge", "no_hedge", "no hedge"})]
        no = no[bench_cols + ["var_net", "var_gross", "var_spot", "mean_net"]].rename(columns={
            "var_net": "var_unhedged_net",
            "var_gross": "var_unhedged_gross",
            "var_spot": "var_unhedged_spot",
            "mean_net": "mean_nohedge_net",
        })
        out = out.merge(no, on=bench_cols, how="left")

        # Fallback benchmark at scenario metadata level.
        fallback_cols = ["exposure_id", "scenario_kind", "oracle_series", "oracle_pool", "oracle_freq", "label"]
        fallback_cols = [c for c in fallback_cols if c in out.columns]
        no_fb = out[out["strategy"].astype(str).str.lower().isin({"nohedge", "no_hedge", "no hedge"})]
        no_fb = (
            no_fb.groupby(fallback_cols, dropna=False)
            .agg(
                var_unhedged_net_fb=("var_net", "mean"),
                var_unhedged_gross_fb=("var_gross", "mean"),
                var_unhedged_spot_fb=("var_spot", "mean"),
                mean_nohedge_net_fb=("mean_net", "mean"),
            )
            .reset_index()
        )
        out = out.merge(no_fb, on=fallback_cols, how="left")
        for c, fb in [
            ("var_unhedged_net", "var_unhedged_net_fb"),
            ("var_unhedged_gross", "var_unhedged_gross_fb"),
            ("var_unhedged_spot", "var_unhedged_spot_fb"),
            ("mean_nohedge_net", "mean_nohedge_net_fb"),
        ]:
            out[c] = out[c].fillna(out[fb])

        out["he_net_vs_nohedge"] = np.where(out["var_unhedged_net"] > 0, 1 - out["var_net"] / out["var_unhedged_net"], np.nan)
        out["he_gross_vs_nohedge"] = np.where(out["var_unhedged_gross"] > 0, 1 - out["var_gross"] / out["var_unhedged_gross"], np.nan)
        out["he_spot_vs_nohedge"] = np.where(out["var_unhedged_spot"] > 0, 1 - out["var_spot"] / out["var_unhedged_spot"], np.nan)
        out = out.drop(columns=[c for c in out.columns if c.endswith("_fb")], errors="ignore")
        return out

    metrics = _add_he(metrics, granular=True)
    metrics_agg = _add_he(metrics_agg, granular=False)

    rank_cols = ["exposure_id", "scenario_kind", "oracle_series", "label"]
    for frame in [metrics, metrics_agg]:
        frame["rank_by_he"] = frame.groupby(rank_cols, dropna=False)["he_net_vs_nohedge"].rank(ascending=False, method="dense")
        frame["rank_by_es"] = frame.groupby(rank_cols, dropna=False)["es95"].rank(ascending=False, method="dense")
        frame["rank_by_mdd"] = frame.groupby(rank_cols, dropna=False)["mdd_equity_mean"].rank(ascending=False, method="dense")

    return metrics, metrics_agg


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def strategy_order(df: pd.DataFrame) -> List[str]:
    vals = list(pd.unique(df["strategy"].astype(str)))
    ordered = [s for s in STRATEGY_ORDER if s in vals]
    ordered.extend([s for s in vals if s not in ordered])
    return ordered


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def prepare_plot_df(metrics: pd.DataFrame, asset: str) -> pd.DataFrame:
    d = metrics[metrics["exposure_id"] == asset].copy()
    d = canonicalize_oracle_fields(d)
    d = filter_dynamic_roll_for_plots(d)
    if d.empty:
        return d
    d["strategy_pretty"] = d["strategy"].map(pretty_strategy)
    d["bucket"] = d["scenario_kind"].astype(str)
    if "label" in d.columns:
        mask = d["label"].astype(str).isin(["BEST", "WORST"])
        d.loc[mask, "bucket"] = d.loc[mask, "bucket"] + " / " + d.loc[mask, "label"].astype(str)
    return d

# Helper: filter to dynamic+roll rows for plots
def filter_dynamic_roll_for_plots(d: pd.DataFrame) -> pd.DataFrame:
    """Keep only dynamic + roll rows for plots when these columns exist.

    RL_PPO is effectively dynamic and roll-enabled. To avoid unfair visual
    comparisons against cheaper/static/noroll variants, thesis plots should compare
    RL with baseline variants under the same dynamic+roll convention. Tables remain
    unchanged; this filter is intentionally plot-only.
    """
    out = d.copy()
    if "mode" in out.columns:
        out = out[out["mode"].astype(str).str.lower().eq("dynamic")].copy()
    if "roll" in out.columns:
        out = out[out["roll"].astype(str).str.lower().isin(["roll", "rolling"])].copy()
    return out


# -----------------------------------------------------------------------------
# Scenario plotting buckets (thesis-safe)
# -----------------------------------------------------------------------------
from typing import List, Tuple
def scenario_plot_buckets(d: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """Return thesis-safe plotting buckets.

    baseline and oracle_universe are plotted as one bucket each. oracle_extreme and
    oracle_feasible are split into BEST/WORST buckets because those labels are
    economically different scenario designs and should not be visually pooled.
    """
    buckets: List[Tuple[str, pd.DataFrame]] = []
    if d.empty or "scenario_kind" not in d.columns:
        return buckets

    for scen in sorted(d["scenario_kind"].dropna().astype(str).unique().tolist()):
        ds = d[d["scenario_kind"].astype(str) == scen].copy()
        if ds.empty:
            continue
        if scen in {"oracle_extreme", "oracle_feasible"} and "label" in ds.columns:
            for label in ["BEST", "WORST"]:
                dl = ds[ds["label"].astype(str).str.upper() == label].copy()
                if not dl.empty:
                    buckets.append((f"{scen}_{label}", dl))
            other = ds[~ds["label"].astype(str).str.upper().isin(["BEST", "WORST"])].copy()
            if not other.empty:
                buckets.append((f"{scen}_UNKNOWN", other))
        else:
            buckets.append((scen, ds))
    return buckets


# -----------------------------------------------------------------------------
# Collapse strategies to one row per scenario bucket for plotting
# -----------------------------------------------------------------------------

def collapse_strategy_rows_for_plot(ds: pd.DataFrame) -> pd.DataFrame:
    """Collapse plot data to exactly one row per strategy.

    The report tables may contain one row per window/oracle_series/mode/roll. For
    thesis plots, however, each strategy should appear once inside each scenario
    bucket. This function prevents seaborn from drawing error bars or multiple
    scatter points for SAC or any baseline strategy.
    """
    if ds.empty:
        return ds
    d = ds.copy()
    if "strategy" not in d.columns:
        return d
    if "strategy_pretty" not in d.columns:
        d["strategy_pretty"] = d["strategy"].map(pretty_strategy)
    if "n_scenarios" not in d.columns:
        d["n_scenarios"] = 1
    d["n_scenarios"] = pd.to_numeric(d["n_scenarios"], errors="coerce").fillna(1).clip(lower=1)

    rows: List[Dict[str, object]] = []
    for strategy, g in d.groupby("strategy", dropna=False):
        g = g.copy()
        w = pd.to_numeric(g["n_scenarios"], errors="coerce").fillna(1).clip(lower=1)
        n_total = float(w.sum())
        row: Dict[str, object] = {
            "strategy": strategy,
            "strategy_pretty": pretty_strategy(strategy),
            "n_scenarios": int(n_total),
        }

        # Preserve categorical identifiers for titles/debugging.
        for col in ["exposure_id", "scenario_kind", "oracle_series", "oracle_pool", "oracle_freq", "label", "mode", "roll"]:
            if col in g.columns:
                vals = sorted(set(g[col].dropna().astype(str).tolist()))
                row[col] = vals[0] if len(vals) == 1 else "ALL"

        # Weighted mean for all scalar numeric plotting metrics.
        numeric_cols = [
            "mean_net", "median_net", "std_net", "cv", "min_net", "max_net", "var_net", "var_gross", "var_spot",
            "var95", "es95", "downside_dev", "lpm1_tau0", "lpm2_tau0", "root_lpm2_tau0", "omega", "tail_ratio", "sortino_tau0", "prob_profit",
            "mdd_equity_mean", "mdd_equity_min", "turnover_contracts_mean", "turnover_h_mean",
            "trade_contracts_mean", "roll_contracts_mean", "max_abs_contracts_mean", "mean_cost_trade",
            "mean_cost_roll", "mean_total_cost", "mean_spot", "mean_gross", "he_net_vs_nohedge",
            "he_gross_vs_nohedge", "he_spot_vs_nohedge",
        ]
        for col in numeric_cols:
            if col not in g.columns:
                continue
            x = pd.to_numeric(g[col], errors="coerce")
            valid = x.notna()
            if valid.any():
                row[col] = float(np.average(x[valid], weights=w[valid]))
            else:
                row[col] = np.nan

        # Better pooled standard deviation when mean/std/n are available.
        if {"mean_net", "std_net", "n_scenarios"}.issubset(g.columns):
            means = pd.to_numeric(g["mean_net"], errors="coerce")
            stds = pd.to_numeric(g["std_net"], errors="coerce")
            ns = pd.to_numeric(g["n_scenarios"], errors="coerce").fillna(1).clip(lower=1)
            valid = means.notna() & stds.notna() & ns.notna()
            if valid.any() and ns[valid].sum() > 1:
                pooled_mean = float(np.average(means[valid], weights=ns[valid]))
                ss_within = ((ns[valid] - 1) * (stds[valid] ** 2)).sum()
                ss_between = (ns[valid] * ((means[valid] - pooled_mean) ** 2)).sum()
                row["mean_net"] = pooled_mean
                row["std_net"] = float(np.sqrt((ss_within + ss_between) / max(ns[valid].sum() - 1, 1)))

        rows.append(row)

    out = pd.DataFrame(rows)
    out["strategy"] = out["strategy"].map(normalize_strategy_name)
    out["strategy_pretty"] = out["strategy"].map(pretty_strategy)
    return out


def plot_bar_metric(metrics: pd.DataFrame, asset: str, out_dir: Path, metric: str, title: str, ylabel: str) -> None:
    d = prepare_plot_df(metrics, asset)
    if d.empty or metric not in d.columns:
        return

    # Plot by scenario/label bucket. oracle_extreme and oracle_feasible are split
    # into BEST/WORST because pooling them hides the experiment design.
    for scen, ds in scenario_plot_buckets(d):
        ds = collapse_strategy_rows_for_plot(ds)
        if ds.empty or metric not in ds.columns:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        order = [pretty_strategy(s) for s in STRATEGY_ORDER if pretty_strategy(s) in set(ds["strategy_pretty"])]
        order += [s for s in ds["strategy_pretty"].dropna().unique().tolist() if s not in order]
        sns.barplot(
            data=ds,
            x="strategy_pretty",
            y=metric,
            order=order,
            hue="strategy_pretty",
            hue_order=order,
            palette=pretty_palette_for(order),
            legend=False,
            errorbar=None,
            ax=ax
        )

        # --- annotate bars with values (scientific style)
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt=lambda x: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.2f}",
                padding=3,
                fontsize=8,
                rotation=0
            )

        ax.set_title(f"{title} | Scenario: {scen}")
        ax.set_xlabel("Strategies")
        ax.set_ylabel(f"{ylabel} (USD)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=25)

        save_fig(fig, out_dir / f"bar_{metric}_{asset}_{scen}.png")


def plot_hedging_effectiveness(metrics: pd.DataFrame, asset: str, out_dir: Path) -> None:
    """Plot hedge effectiveness as percentage, excluding NoHedge.

    HE is defined as variance reduction versus NoHedge. NoHedge is excluded because
    its HE is mechanically zero and only compresses the useful visual scale.
    """
    d = prepare_plot_df(metrics, asset)
    if d.empty or "he_net_vs_nohedge" not in d.columns:
        return

    d = d[~d["strategy"].astype(str).str.lower().isin({"nohedge", "no_hedge", "no hedge"})].copy()
    if d.empty:
        return
    d["he_percent"] = pd.to_numeric(d["he_net_vs_nohedge"], errors="coerce") * 100.0

    for scen, ds in scenario_plot_buckets(d):
        ds = collapse_strategy_rows_for_plot(ds)
        ds = ds.copy()
        if "he_net_vs_nohedge" in ds.columns:
            ds["he_percent"] = pd.to_numeric(ds["he_net_vs_nohedge"], errors="coerce") * 100.0
        ds = ds.dropna(subset=["he_percent"])
        if ds.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        order = [pretty_strategy(s) for s in STRATEGY_ORDER if pretty_strategy(s) in set(ds["strategy_pretty"])]
        order += [s for s in ds["strategy_pretty"].dropna().unique().tolist() if s not in order]
        sns.barplot(
            data=ds,
            x="strategy_pretty",
            y="he_percent",
            order=order,
            hue="strategy_pretty",
            hue_order=order,
            palette=pretty_palette_for(order),
            legend=False,
            errorbar=None,
            ax=ax,
        )
        # --- annotate HE bars as percentages
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt=lambda x: f"{x:.1f}%",
                padding=3,
                fontsize=8,
                rotation=0
            )
        ax.set_title(f"Hedging Effectiveness (Variance Reduction) | {asset} | {scen}")
        ax.set_xlabel("Strategies")
        ax.set_ylabel("Hedging Effectiveness (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=25)
        save_fig(fig, out_dir / f"bar_hedging_effectiveness_percent_{asset}_{scen}.png")


def plot_risk_return(metrics: pd.DataFrame, asset: str, out_dir: Path) -> None:
    d = prepare_plot_df(metrics, asset)
    if d.empty:
        return

    for scen, ds in scenario_plot_buckets(d):
        ds = collapse_strategy_rows_for_plot(ds)
        ds = ds.dropna(subset=["std_net", "mean_net"])
        if ds.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=ds,
            x="std_net",
            y="mean_net",
            hue="strategy_pretty",
            palette=pretty_palette_for(ds["strategy_pretty"].dropna().unique()),
            s=120,
            ax=ax
        )
        ax.set_title(f"Risk-Return Tradeoff | {asset} | {scen}")
        ax.set_xlabel("Risk (Std of Net PnL, USD)")
        ax.set_ylabel("Mean Net PnL (USD)")
        ax.axhline(0, color="black", linewidth=0.8)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("Strategies")

        save_fig(fig, out_dir / f"scatter_risk_return_{asset}_{scen}.png")


def plot_tail_mdd(metrics: pd.DataFrame, asset: str, out_dir: Path) -> None:
    d = prepare_plot_df(metrics, asset)
    if d.empty:
        return

    for scen, ds in scenario_plot_buckets(d):
        ds = collapse_strategy_rows_for_plot(ds)
        ds = ds.dropna(subset=["es95", "mdd_equity_mean"])
        if ds.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=ds,
            x="es95",
            y="mdd_equity_mean",
            hue="strategy_pretty",
            palette=pretty_palette_for(ds["strategy_pretty"].dropna().unique()),
            s=120,
            ax=ax
        )
        ax.set_title(f"Tail Risk vs Drawdown | {asset} | {scen}")
        ax.set_xlabel("Expected Shortfall 95% (USD)")
        ax.set_ylabel("Mean Maximum Drawdown (USD)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("Strategies")

        save_fig(fig, out_dir / f"scatter_tail_mdd_{asset}_{scen}.png")


def plot_oracle_universe_distributions(df: pd.DataFrame, asset: str, out_dir: Path, max_points_per_strategy: int = 25000) -> None:
    d = df[df["exposure_id"] == asset].copy()
    d = filter_dynamic_roll_for_plots(d)
    if d.empty:
        return

    d["strategy_pretty"] = d["strategy"].map(pretty_strategy)

    # Distribution plots are generated per economically coherent scenario bucket:
    # oracle_universe, oracle_extreme_BEST, oracle_extreme_WORST,
    # oracle_feasible_BEST, oracle_feasible_WORST. This avoids visually pooling
    # BEST and WORST oracle designs.
    for scen, ds in scenario_plot_buckets(d):
        if scen == "baseline":
            continue
        ds = ds.copy()
        if ds.empty:
            continue

        pal = pretty_palette_for(ds["strategy_pretty"].dropna().unique())

        # Robust x-axis range: use 0.5%..99.5% quantiles so a few extreme outliers
        # do not flatten the central distribution. The full tails are still reported
        # in VaR/ES tables.
        x = pd.to_numeric(ds["net_pnl_total"], errors="coerce").dropna()
        if x.empty:
            continue
        lo = float(x.quantile(0.005))
        hi = float(x.quantile(0.995))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo = float(x.min())
            hi = float(x.max())
        if lo == hi:
            lo -= 1.0
            hi += 1.0

        # --- Histogram: high-resolution binned frequency/density.
        # bins=360 gives much finer clusters than the previous 120-bin view.
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.histplot(
            data=ds,
            x="net_pnl_total",
            hue="strategy_pretty",
            bins=360,
            binrange=(lo, hi),
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            linewidth=1.4,
            palette=pal,
            ax=ax,
        )
        ax.set_title(f"Overlaid Net PnL Histogram | {asset} | {scen}")
        ax.set_xlabel("Net PnL (USD; 360 bins, 0.5%–99.5% display range)")
        ax.set_ylabel("Density")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.1)
        ax.set_xlim(lo, hi)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("Strategies")
        save_fig(fig, out_dir / f"hist_overlay_{asset}_{scen}.png")

        # --- KDE: smoothed shape view, wider aspect ratio to reveal tail/skew differences.
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.kdeplot(
            data=ds,
            x="net_pnl_total",
            hue="strategy_pretty",
            common_norm=False,
            fill=False,
            linewidth=2.0,
            bw_adjust=0.65,
            clip=(lo, hi),
            palette=pal,
            ax=ax,
        )
        ax.set_title(f"KDE Shape Comparison: Tail and Skew | {asset} | {scen}")
        ax.set_xlabel("Net PnL (USD; clipped to 0.5%–99.5% for visual readability)")
        ax.set_ylabel("Density")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.1)
        ax.set_xlim(lo, hi)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("Strategies")
        save_fig(fig, out_dir / f"kde_overlay_{asset}_{scen}.png")

        # --- ECDF: wider figure + display-range zoom to make strategy differences visible.
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.ecdfplot(
            data=ds,
            x="net_pnl_total",
            hue="strategy_pretty",
            palette=pal,
            linewidth=2.0,
            ax=ax,
        )
        ax.set_title(f"ECDF Comparison: Tail Risk and Profit Probability | {asset} | {scen}")
        ax.set_xlabel("Net PnL (USD; zoomed to 0.5%–99.5% display range)")
        ax.set_ylabel("Cumulative Probability")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.1)
        ax.set_xlim(lo, hi)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("Strategies")
        save_fig(fig, out_dir / f"ecdf_overlay_{asset}_{scen}.png")


# -----------------------------------------------------------------------------
# New: VaR/ES curves and cumulative LPM plots
# -----------------------------------------------------------------------------

def _alpha_grid_for_tail_curves() -> np.ndarray:
    """Dense confidence grid from 80% to 99.9%, with more detail near the tail."""
    base = np.linspace(0.80, 0.99, 20)
    tail = np.array([0.991, 0.9925, 0.995, 0.9975, 0.999])
    return np.unique(np.concatenate([base, tail]))


def _tail_curve_rows(ds: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build VaR/ES curves for one scenario bucket from raw scenario-level PnL rows."""
    rows: List[Dict[str, object]] = []
    for strategy, g in ds.groupby("strategy", dropna=False):
        x = pd.to_numeric(g["net_pnl_total"], errors="coerce").dropna()
        if len(x) < 5:
            continue
        for alpha in _alpha_grid_for_tail_curves():
            if metric == "var":
                val = value_at_risk(x, float(alpha))
            elif metric == "es":
                val = expected_shortfall(x, float(alpha))
            else:
                raise ValueError(metric)
            rows.append({
                "strategy": normalize_strategy_name(strategy),
                "strategy_pretty": pretty_strategy(normalize_strategy_name(strategy)),
                "alpha": float(alpha),
                "metric_value": float(val) if pd.notna(val) else np.nan,
                "n_scenarios": int(len(x)),
            })
    return pd.DataFrame(rows)


def plot_var_es_curves(df: pd.DataFrame, asset: str, out_dir: Path, metric: str) -> None:
    """Plot VaR or ES/CVaR across confidence levels 80%..99.9% from raw rows."""
    d = df[df["exposure_id"] == asset].copy()
    d = canonicalize_oracle_fields(d)
    d = filter_dynamic_roll_for_plots(d)
    if d.empty or "net_pnl_total" not in d.columns:
        return
    d["strategy_pretty"] = d["strategy"].map(pretty_strategy)

    title_name = "VaR" if metric == "var" else "ES / CVaR"
    y_name = "VaR of Net PnL (USD)" if metric == "var" else "Expected Shortfall / CVaR of Net PnL (USD)"
    file_prefix = "curve_var" if metric == "var" else "curve_es_cvar"

    for scen, ds in scenario_plot_buckets(d):
        curve = _tail_curve_rows(ds, metric=metric)
        curve = curve.dropna(subset=["metric_value"])
        if curve.empty:
            continue
        fig, ax = plt.subplots(figsize=(13, 7))
        order = [pretty_strategy(s) for s in STRATEGY_ORDER if pretty_strategy(s) in set(curve["strategy_pretty"])]
        order += [s for s in curve["strategy_pretty"].dropna().unique().tolist() if s not in order]
        sns.lineplot(
            data=curve,
            x="alpha",
            y="metric_value",
            hue="strategy_pretty",
            hue_order=order,
            palette=pretty_palette_for(order),
            marker="o",
            linewidth=2.0,
            ax=ax,
        )
        ax.set_title(f"{title_name} Curve by Confidence Level | {asset} | {scen}")
        ax.set_xlabel("Confidence level α")
        ax.set_ylabel(y_name)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks([0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999])
        ax.set_xticklabels(["80%", "85%", "90%", "95%", "97.5%", "99%", "99.5%", "99.9%"])
        ax.tick_params(axis="x", rotation=25)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("Strategies")
        save_fig(fig, out_dir / f"{file_prefix}_{asset}_{scen}.png")


def plot_cumulative_lpm(df: pd.DataFrame, asset: str, out_dir: Path, tau: float = 0.0, order: int = 2) -> None:
    """Plot cumulative episode-level LPM contribution through time.

    This is not the RL reward step-by-step LPM; it is the cumulative downside
    contribution across evaluated scenarios ordered by scenario end/start date.
    It is useful for seeing when a strategy accumulates downside losses during the
    walk-forward test history.
    """
    d = df[df["exposure_id"] == asset].copy()
    d = canonicalize_oracle_fields(d)
    d = filter_dynamic_roll_for_plots(d)
    if d.empty or "net_pnl_total" not in d.columns:
        return

    if "end_date" in d.columns:
        d["plot_date"] = pd.to_datetime(d["end_date"], errors="coerce")
    elif "start_date" in d.columns:
        d["plot_date"] = pd.to_datetime(d["start_date"], errors="coerce")
    else:
        d["plot_date"] = pd.NaT

    if d["plot_date"].isna().all():
        if "window" in d.columns:
            d["plot_date"] = pd.to_datetime(pd.to_numeric(d["window"], errors="coerce"), unit="D", origin="2000-01-01")
        else:
            d["plot_date"] = pd.to_datetime(np.arange(len(d)), unit="D", origin="2000-01-01")

    d["downside_contrib"] = np.maximum(float(tau) - pd.to_numeric(d["net_pnl_total"], errors="coerce"), 0.0) ** int(order)
    d["strategy_pretty"] = d["strategy"].map(pretty_strategy)

    for scen, ds in scenario_plot_buckets(d):
        rows: List[pd.DataFrame] = []
        for strategy, g in ds.groupby("strategy", dropna=False):
            g = g.sort_values("plot_date").copy()
            g["cum_lpm"] = g["downside_contrib"].cumsum()
            g["cum_root_lpm"] = g["cum_lpm"] ** (1.0 / int(order))
            g["strategy_pretty"] = pretty_strategy(normalize_strategy_name(strategy))
            rows.append(g[["plot_date", "cum_lpm", "cum_root_lpm", "strategy_pretty"]])
        if not rows:
            continue
        curve = pd.concat(rows, ignore_index=True).dropna(subset=["plot_date"])
        if curve.empty:
            continue

        fig, ax = plt.subplots(figsize=(15, 7))
        order_names = [pretty_strategy(s) for s in STRATEGY_ORDER if pretty_strategy(s) in set(curve["strategy_pretty"])]
        order_names += [s for s in curve["strategy_pretty"].dropna().unique().tolist() if s not in order_names]
        sns.lineplot(
            data=curve,
            x="plot_date",
            y="cum_root_lpm",
            hue="strategy_pretty",
            hue_order=order_names,
            palette=pretty_palette_for(order_names),
            linewidth=2.0,
            ax=ax,
        )
        ax.set_title(f"Cumulative Root-LPM{order} Through Time | {asset} | {scen}")
        ax.set_xlabel("Scenario end date")
        ax.set_ylabel(f"Cumulative Root-LPM{order} (USD units)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=25)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title("Strategies")
        save_fig(fig, out_dir / f"curve_cumulative_root_lpm{order}_{asset}_{scen}.png")


def plot_all(
    metrics: pd.DataFrame,
    metrics_agg: pd.DataFrame,
    df: pd.DataFrame,
    assets: Sequence[str],
    out_dir: Path,
    *,
    include_cumulative_plots: bool = True,
) -> None:
    plot_dir = out_dir / "plots"
    # Thesis figures use metrics_agg and then collapse any remaining oracle/window
    # rows to one row per strategy per scenario bucket. This avoids misleading
    # seaborn error bars and multiple scatter points for the same strategy.
    plot_metrics = metrics_agg.copy()
    for asset in assets:
        plot_bar_metric(plot_metrics, asset, plot_dir, "mean_net", f"Mean net PnL by model and scenario: {asset}", "Mean net PnL")
        plot_hedging_effectiveness(plot_metrics, asset, plot_dir)
        plot_bar_metric(plot_metrics, asset, plot_dir, "es95", f"Expected Shortfall 95% by model and scenario: {asset}", "ES 95% net PnL")
        plot_bar_metric(plot_metrics, asset, plot_dir, "downside_dev", f"Downside deviation by model and scenario: {asset}", "Downside deviation")
        plot_bar_metric(plot_metrics, asset, plot_dir, "root_lpm2_tau0", f"Root-LPM2 downside risk by model and scenario: {asset}", "Root-LPM2")
        plot_bar_metric(plot_metrics, asset, plot_dir, "prob_profit", f"Probability of positive net PnL: {asset}", "P(Net PnL > 0)")
        plot_bar_metric(plot_metrics, asset, plot_dir, "mean_total_cost", f"Mean transaction + roll cost: {asset}", "Mean total cost")
        plot_risk_return(plot_metrics, asset, plot_dir)
        plot_tail_mdd(plot_metrics, asset, plot_dir)
        plot_var_es_curves(df, asset, plot_dir, metric="var")
        plot_var_es_curves(df, asset, plot_dir, metric="es")
        if include_cumulative_plots:
            plot_cumulative_lpm(df, asset, plot_dir, tau=0.0, order=2)
        plot_oracle_universe_distributions(df, asset, plot_dir)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def write_excel(path: Path, metrics: pd.DataFrame, metrics_agg: pd.DataFrame, df: pd.DataFrame, asset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = metrics[metrics["exposure_id"] == asset].copy()
    a = metrics_agg[metrics_agg["exposure_id"] == asset].copy()
    raw_sample = df[df["exposure_id"] == asset].head(5000).copy()
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        a.to_excel(writer, sheet_name="summary_agg", index=False)
        m.to_excel(writer, sheet_name="summary_by_window", index=False)
        raw_sample.to_excel(writer, sheet_name="raw_sample", index=False)
        if "label" in a.columns:
            a[a["label"].astype(str).isin(["BEST", "WORST"])].to_excel(writer, sheet_name="oracle_best_worst", index=False)
        a.sort_values(["scenario_kind", "label", "he_net_vs_nohedge"], ascending=[True, True, False]).to_excel(writer, sheet_name="ranking_by_HE", index=False)
        a.sort_values(["scenario_kind", "label", "es95"], ascending=[True, True, False]).to_excel(writer, sheet_name="ranking_by_ES", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        pct_fmt = workbook.add_format({"num_format": "0.00%"})
        num_fmt = workbook.add_format({"num_format": "#,##0"})
        float_fmt = workbook.add_format({"num_format": "#,##0.000"})
        for sheet, frame in {
            "summary_agg": a,
            "summary_by_window": m,
            "raw_sample": raw_sample,
            "oracle_best_worst": a[a["label"].astype(str).isin(["BEST", "WORST"])] if "label" in a.columns else pd.DataFrame(),
            "ranking_by_HE": a,
            "ranking_by_ES": a,
        }.items():
            if sheet not in writer.sheets:
                continue
            ws = writer.sheets[sheet]
            for i, col in enumerate(frame.columns):
                ws.write(0, i, col, header_fmt)
                width = max(12, min(35, len(str(col)) + 3))
                fmt = None
                if col in {"he_net_vs_nohedge", "he_gross_vs_nohedge", "prob_profit"}:
                    fmt = pct_fmt
                elif col in {"omega", "tail_ratio", "sortino_tau0", "skew", "kurtosis_excess"}:
                    fmt = float_fmt
                elif col.startswith("mean_") or col in {"var95", "es95", "downside_dev", "median_net", "min_net", "max_net"}:
                    fmt = num_fmt
                ws.set_column(i, i, width, fmt)
            ws.freeze_panes(1, 0)
            if len(frame.columns) > 0:
                ws.autofilter(0, 0, max(len(frame), 1), len(frame.columns) - 1)


def write_markdown(path: Path, metrics_agg: pd.DataFrame, assets: Sequence[str]) -> None:
    lines: List[str] = [
        "# Final RL vs Baseline Hedging Report",
        "",
        "This report compares SAC-LPM against classical hedging strategies under transaction costs, rollover, and scenario-level physical trading simulation.",
        "",
        "## Methodological notes",
        "- Baseline outputs are read from raw hedge_summary files to preserve window/mode/roll metadata.",
        "- RL rows are joined to scenario metadata using scenario_id.",
        "- oracle_all is split into oracle_extreme and oracle_feasible when oracle_series indicates EXTREME/FEASIBLE.",
        "- BEST/WORST labels are retained from oracle_series.",
        "- Hedging effectiveness is computed as variance reduction versus NoHedge in comparable scenario buckets.",
        "",
    ]
    for asset in assets:
        d = metrics_agg[metrics_agg["exposure_id"] == asset].copy()
        lines.append(f"## {asset}")
        if d.empty:
            lines.append("No rows.\n")
            continue
        cols = ["scenario_kind", "oracle_series", "label", "strategy", "n_scenarios", "mean_net", "he_net_vs_nohedge", "var95", "es95", "prob_profit", "mdd_equity_mean"]
        cols = [c for c in cols if c in d.columns]
        compact = d.sort_values(["scenario_kind", "label", "he_net_vs_nohedge"], ascending=[True, True, False])[cols].head(40)
        lines.append(compact.to_markdown(index=False))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Thesis-grade RL vs baseline report with seaborn plots.")
    ap.add_argument("--baseline_glob", nargs="+", default=["results/BASELINE_BATCH/**/hedge_summary_*.parquet"], help="Raw baseline hedge_summary file globs.")
    ap.add_argument("--rl_path", nargs="*", default=[], help="One or more RL result parquet/csv files.")
    ap.add_argument("--scenario_root", default="scenarios", help="Scenario root folder.")
    ap.add_argument("--out_dir", default="reports/final_rl_vs_baseline", help="Output directory.")
    ap.add_argument("--asset", nargs="*", default=None, help="Optional assets, e.g. OPEC_BASKET WTI_SPOT BRENT_SPOT.")
    ap.add_argument("--alpha", type=float, default=0.95, help="VaR/ES confidence level.")
    ap.add_argument("--tau", type=float, default=0.0, help="Downside threshold.")
    ap.add_argument("--strict_intersection", action="store_true", help="Restrict all model rows to exposure_id + scenario_id pairs that exist in RL output.")
    ap.add_argument("--no_plots", action="store_true", help="Skip seaborn plot generation.")
    ap.add_argument(
        "--no_cumulative_plots",
        action="store_true",
        help="Skip cumulative/time-series plots such as cumulative LPM. Other plots are still generated.",
    )
    ap.add_argument("--max_files", type=int, default=0, help="Debug: limit number of baseline files.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = args.asset if args.asset else None
    assets = normalize_assets_arg(assets)

    print("[finalreport] loading scenario metadata")
    meta = load_scenario_metadata(Path(args.scenario_root), assets=assets)
    print(f"[finalreport] scenario metadata rows={len(meta):,}")

    baseline_files = expand_globs(args.baseline_glob)
    if args.max_files and args.max_files > 0:
        baseline_files = baseline_files[: args.max_files]
    print(f"[finalreport] baseline raw files={len(baseline_files):,}")
    baseline = read_baseline_raw(baseline_files, assets=assets)
    baseline = attach_scenario_metadata(baseline, meta)
    baseline = filter_to_assets(baseline, assets, stage="baseline_after_metadata")
    print(f"[finalreport] baseline rows={len(baseline):,}")

    rl = pd.DataFrame()
    if args.rl_path:
        rl_paths = resolve_rl_paths(args.rl_path)
        print(f"[finalreport] resolved RL paths: {[str(p) for p in rl_paths]}")
        rl = ingest_rl_results(rl_paths, assets=assets)
        rl = attach_scenario_metadata(rl, meta)
        rl = filter_to_assets(rl, assets, stage="rl_after_metadata")
        print(f"[finalreport] RL rows={len(rl):,}")

    frames = [baseline]
    if not rl.empty:
        frames.append(rl)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = attach_scenario_metadata(df, meta)
    df = filter_to_assets(df, assets, stage="combined_after_metadata")

    if args.strict_intersection and not rl.empty:
        before = len(df)
        df = strict_intersection_with_rl(df)
        df = filter_to_assets(df, assets, stage="after_strict_intersection")
        print(f"[finalreport] strict intersection rows: {before:,} -> {len(df):,}")

    df["strategy"] = df["strategy"].map(normalize_strategy_name)
    df["strategy_pretty"] = df["strategy"].map(pretty_strategy)
    for col in ["mode", "roll", "window", "oracle_series", "oracle_pool", "oracle_freq", "label"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    # Validate final asset set
    if assets:
        actual_assets = sorted(df["exposure_id"].dropna().astype(str).unique().tolist())
        expected_assets = sorted(assets)
        if actual_assets != expected_assets:
            raise ValueError(f"Final report asset mismatch: actual={actual_assets}, expected={expected_assets}")

    audit_path = out_dir / "cleaned_model_scenario_rows.parquet"
    df.to_parquet(audit_path, index=False)
    print(f"[finalreport] wrote audit rows: {audit_path}")

    print("[finalreport] computing metrics")
    metrics, metrics_agg = aggregate_metrics(df, alpha=args.alpha, tau=args.tau)

    sort_cols = ["exposure_id", "scenario_kind", "oracle_series", "label", "strategy", "window", "mode", "roll"]
    metrics = metrics.sort_values([c for c in sort_cols if c in metrics.columns]).reset_index(drop=True)
    metrics_agg = metrics_agg.sort_values([c for c in sort_cols if c in metrics_agg.columns]).reset_index(drop=True)

    metrics_path = out_dir / "metrics_all_assets.csv"
    metrics_agg_path = out_dir / "metrics_all_assets_agg.csv"
    metrics.to_csv(metrics_path, index=False)
    metrics_agg.to_csv(metrics_agg_path, index=False)
    print(f"[finalreport] wrote {metrics_path}")
    print(f"[finalreport] wrote {metrics_agg_path}")

    assets_out = list(assets) if assets else sorted(df["exposure_id"].dropna().astype(str).unique().tolist())
    for asset in assets_out:
        write_excel(out_dir / asset / f"report_{asset}.xlsx", metrics, metrics_agg, df, asset)
        print(f"[finalreport] wrote Excel for {asset}")

    write_markdown(out_dir / "report_summary.md", metrics_agg, assets_out)
    print("[finalreport] wrote report_summary.md")

    if not args.no_plots:
        print("[finalreport] generating seaborn plots")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_all(
                metrics,
                metrics_agg,
                df,
                assets_out,
                out_dir,
                include_cumulative_plots=not args.no_cumulative_plots,
            )
        print(f"[finalreport] plots written to {out_dir / 'plots'}")

    print("[finalreport] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())