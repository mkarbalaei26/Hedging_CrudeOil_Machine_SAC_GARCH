

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Baseline_report.py

Generate thesis-grade Excel reports (multi-sheet + charts) from merged hedge summary outputs.

Expected input: one merged Parquet (or many parquet files) containing at least:
  - exposure_id (asset)
  - strategy
  - mode_roll (e.g. 'roll' / 'no-roll' / 'noroll')
  - dynamic (bool/int)
  - spot_pnl_total
  - fut_pnl_total
  - cost_trade_total
  - cost_roll_total
  - net_pnl_total
  - turnover_contracts / turnover_h / max_abs_contracts (if present)
  - mdd_equity (if present)

If window/mode/roll/kind columns are missing, they are inferred from a source/path column or from the parquet filename.

Outputs:
  results/<REPORT_OUT>/<ASSET>/report_<ASSET>.xlsx

Usage examples:
  python Baseline_report.py --inputs results/BATCH_ALL/*/*/hedge_summary_*.parquet --out_dir reports/final_excel
  python Baseline_report.py --inputs results/BATCH_ALL/_merged/merged_all.parquet --out_dir reports/final_excel

Notes:
  - Charts are produced with matplotlib (saved as PNG) and embedded into Excel via xlsxwriter.
  - This script is designed to be robust to your evolving pipeline (new strategies like LSTM/Transformer).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from results_schema import validate_and_cast_results, print_schema_report

# Safe import of pyarrow for optional batch streaming merge
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore

# matplotlib is used only for producing images; Excel embedding uses xlsxwriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Helpers / parsing
# -----------------------------

#
# Window can appear in several filename/tag styles across the pipeline:
#   __w120__
#   _w120_
#   |w120|
#   ...w120...
WINDOW_RE_LIST = [
    re.compile(r"__w(?P<w>\d+|NA)__", re.IGNORECASE),
    re.compile(r"(?:^|[\|_\-])w(?P<w>\d{2,4})(?:[\|_\-]|$)", re.IGNORECASE),
    re.compile(r"(?:window|win)[_=\-]?(?P<w>\d{2,4})", re.IGNORECASE),
]
MODE_RE = re.compile(r"__(?P<mode>dynamic|static)__", re.IGNORECASE)
ROLL_RE = re.compile(r"__(?P<roll>roll|noroll|no_roll|no-roll)__", re.IGNORECASE)
KIND_RE = re.compile(r"(scenarios_baseline|oracle_all|oracle_universe|scenarios_company)", re.IGNORECASE)


def ts_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.()-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s



def infer_kind_from_any(s: str) -> str:
    s_low = str(s).lower().strip()

    # --- RL reference labels (common) ---
    # train/val/test may be encoded as: test:univ, val:univ, test:all, test:base, etc.
    # Also allow short tokens like 'univ'/'universe' and 'base'.
    # We intentionally check these early so they don't fall into the generic 'baseline'/'company' keyword matches.
    if any(tok in s_low for tok in ["test:univ", "val:univ", "univ", "universe", "oracle_universe"]):
        return "oracle_universe"
    if any(tok in s_low for tok in ["test:all", "oracle_all", "oracle all"]):
        return "oracle_all"
    if any(tok in s_low for tok in ["test:base", "baseline", "scenarios_baseline", "__baseline__", "__scenarios_baseline__", "base_"]):
        return "baseline"

    # Strong signals from the orchestrated hedge_summary filenames
    if "__oracle_universe__" in s_low or "_oracle_universe_" in s_low:
        return "oracle_universe"
    if "__oracle_all__" in s_low or "_oracle_all_" in s_low:
        return "oracle_all"
    if "__scenarios_baseline__" in s_low:
        return "baseline"
    if "__scenarios_company__" in s_low:
        return "company"

    # Folder-based fallback (FINAL_MERGED/<ASSET>/<KIND>/...)
    parts = [p.lower() for p in Path(str(s)).parts]
    if "oracle_universe" in parts:
        return "oracle_universe"
    if "oracle_all" in parts:
        return "oracle_all"
    if "baseline" in parts or "scenarios_baseline" in parts:
        return "baseline"
    if "company" in parts or "scenarios_company" in parts:
        return "company"

    # Weak fallback keywords
    if "company" in s_low:
        return "company"

    return "unknown"


# --- Helper: infer oracle subkind from any part ---
from typing import Optional
def infer_oracle_subkind_from_any(*parts: object) -> Optional[str]:
    """Try to split oracle_all into oracle_feasible / oracle_extreme.

    We look for keywords in scenario_id / scenario_file / source path.
    Returns None if not inferable.
    """
    blob = " | ".join([str(p) for p in parts if p is not None]).lower()
    if not blob:
        return None
    if "extreme" in blob:
        return "oracle_extreme"
    if "feasible" in blob:
        return "oracle_feasible"
    return None


# --- Scenario metadata cache (for scenario_id -> oracle_series, and for filling missing scenario_id) ---
_SCENARIO_CACHE: Dict[str, pd.DataFrame] = {}


def _try_read_parquet_cols(path: str, cols: List[str]) -> Optional[pd.DataFrame]:
    try:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        keep = [c for c in cols if c in df.columns]
        if not keep:
            return None
        return df[keep].copy()
    except Exception:
        return None



def _resolve_oracle_all_path(asset: str) -> Optional[str]:
    # Common locations/names in this repo
    candidates = [
        f"scenarios/{asset}/Oracle_all.parquet",
        f"scenarios/{asset}/oracle_all.parquet",
        f"scenarios/{asset}/ORACLE_ALL.parquet",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


# --- Scenario path resolver for various kinds
def _resolve_scenario_path(asset: str, kind: str) -> Optional[str]:
    """Resolve canonical scenario parquet path for a given asset+kind in this repo."""
    kind = str(kind).lower()
    candidates: List[str] = []
    if kind in {"oracle_all", "oracle_extreme", "oracle_feasible", "oracle_other"}:
        # stored as oracle_all parquet
        candidates = [
            f"scenarios/{asset}/oracle_all.parquet",
            f"scenarios/{asset}/Oracle_all.parquet",
            f"scenarios/{asset}/ORACLE_ALL.parquet",
        ]
    elif kind in {"oracle_universe", "universe"}:
        candidates = [
            f"scenarios/{asset}/oracle_universe.parquet",
            f"scenarios/{asset}/Oracle_universe.parquet",
            f"scenarios/{asset}/ORACLE_UNIVERSE.parquet",
        ]
    elif kind in {"baseline"}:
        candidates = [
            f"scenarios/{asset}/scenarios_baseline.parquet",
            f"scenarios/{asset}/baseline.parquet",
        ]
    elif kind in {"company"}:
        candidates = [
            f"scenarios/{asset}/scenarios_company.parquet",
            f"scenarios/{asset}/company.parquet",
        ]

    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _load_scenario_df(path: str) -> Optional[pd.DataFrame]:
    """Load scenario parquet with caching. Returns df or None."""
    if not path:
        return None
    key = str(Path(path))
    if key in _SCENARIO_CACHE:
        return _SCENARIO_CACHE[key]
    df = _try_read_parquet_cols(key, [
        "scenario_id", "scenario_record_id",
        "exposure_id", "scenario_kind",
        "start_date", "end_date",
        "horizon_days", "horizon_days_target", "horizon_days_realized", "volume_bbl",
        "oracle_series", "oracle_pool", "oracle_freq", "label", "tag",
        "oracle_candidate", "company_id", "company_size",
    ])
    if df is None:
        return None
    # normalize dtypes
    if "scenario_id" in df.columns:
        df["scenario_id"] = df["scenario_id"].astype(str)
    for dc in ["start_date", "end_date"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
    _SCENARIO_CACHE[key] = df
    return df



def _oracle_kind_from_oracle_series(x: object) -> Optional[str]:
    """Map oracle_series values to reporting scenario_kind buckets."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    # user example: EXTREME_DAILY_BEST
    if "extreme" in s:
        return "oracle_extreme"
    if "feasible" in s:
        return "oracle_feasible"
    return "oracle_other"


def _parse_oracle_series_fields(x: object) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse strings like 'EXTREME_DAILY_BEST' -> (pool='EXTREME', freq='DAILY', label='BEST')."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None, None, None
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None, None, None
    s = s.upper()
    parts = [p for p in s.split("_") if p]
    if not parts:
        return None, None, None
    pool = parts[0] if len(parts) >= 1 else None
    freq = parts[1] if len(parts) >= 2 else None
    label = parts[-1] if parts[-1] in {"BEST", "WORST"} else None
    return pool, freq, label


def _fill_missing_scenario_id_from_scenario_file(df: pd.DataFrame) -> pd.DataFrame:
    """If some rows have missing scenario_id, try to fill using scenario_file and (start_date,end_date,...) keys."""
    if "scenario_id" not in df.columns:
        df["scenario_id"] = pd.NA

    if "scenario_file" not in df.columns:
        return df

    miss = df["scenario_id"].isna()
    if not miss.any():
        return df

    # Build mapping per scenario_file
    df = df.copy()
    sf = df.loc[miss, "scenario_file"].astype(str)
    unique_files = [p for p in sf.unique().tolist() if p and p != "nan"]

    for path in unique_files:
        scen = _load_scenario_df(path)
        if scen is None:
            continue
        if "scenario_id" not in scen.columns:
            continue

        # Build a robust key; prefer exact (start,end,horizon,volume) when available
        key_cols = [c for c in ["start_date", "end_date", "horizon_days", "volume_bbl"] if c in scen.columns]
        if not key_cols:
            continue

        scen_k = scen.copy()
        for dc in ["start_date", "end_date"]:
            if dc in scen_k.columns:
                scen_k[dc] = pd.to_datetime(scen_k[dc], errors="coerce")

        # Create dict mapping tuple -> scenario_id
        scen_k = scen_k.dropna(subset=["scenario_id"])
        m: Dict[Tuple[object, ...], str] = {}
        for _, r in scen_k.iterrows():
            k = tuple(r[c] for c in key_cols)
            if k not in m:
                m[k] = str(r["scenario_id"])

        # Apply to matching rows
        sel = miss & (df["scenario_file"].astype(str) == path)
        if not sel.any():
            continue

        # Normalize row keys
        for dc in ["start_date", "end_date"]:
            if dc in df.columns:
                df.loc[sel, dc] = pd.to_datetime(df.loc[sel, dc], errors="coerce")

        def _lookup_row(row: pd.Series) -> object:
            k = tuple(row.get(c, pd.NA) for c in key_cols)
            return m.get(k, pd.NA)

        df.loc[sel, "scenario_id"] = df.loc[sel].apply(_lookup_row, axis=1)

    return df


def _apply_oracle_series_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split oracle_all into oracle_extreme/oracle_feasible.

    Priority:
      1) If hedge_summary already contains `oracle_series`, split directly from it.
      2) Else fall back to reading oracle_all scenario parquet and mapping via scenario_id.
    """
    if "scenario_kind" not in df.columns:
        return df
    is_oracle_all = df["scenario_kind"].astype(str).eq("oracle_all")
    if not is_oracle_all.any():
        return df

    df = df.copy()

    # Fast path: oracle_series already present in results
    if "oracle_series" in df.columns and df.loc[is_oracle_all, "oracle_series"].notna().any():
        k = df.loc[is_oracle_all, "oracle_series"].map(_oracle_kind_from_oracle_series)
        df.loc[is_oracle_all & (k == "oracle_extreme"), "scenario_kind"] = "oracle_extreme"
        df.loc[is_oracle_all & (k == "oracle_feasible"), "scenario_kind"] = "oracle_feasible"
        df.loc[is_oracle_all & (k == "oracle_other"), "scenario_kind"] = "oracle_other"
        return df

    # Prefer scenario_file if present and readable; otherwise fall back to scenarios/<asset>/Oracle_all.parquet
    if "scenario_id" not in df.columns:
        df["scenario_id"] = pd.NA

    # Build per-asset map: scenario_id -> oracle_kind
    for asset in df.loc[is_oracle_all, "exposure_id"].astype(str).unique().tolist():
        # Choose a path
        path: Optional[str] = None
        if "scenario_file" in df.columns:
            # take first existing scenario_file for this asset+oracle_all
            cand = df.loc[is_oracle_all & (df["exposure_id"].astype(str) == asset), "scenario_file"].astype(str)
            for p in cand.unique().tolist():
                if p and p != "nan" and Path(p).exists():
                    path = p
                    break
        if path is None:
            path = _resolve_oracle_all_path(asset)

        scen = _load_scenario_df(path) if path else None
        if scen is None or "scenario_id" not in scen.columns or "oracle_series" not in scen.columns:
            continue

        tmp = scen[["scenario_id", "oracle_series"]].dropna(subset=["scenario_id"]).copy()
        tmp["oracle_kind"] = tmp["oracle_series"].map(_oracle_kind_from_oracle_series)
        m = dict(zip(tmp["scenario_id"].astype(str), tmp["oracle_kind"]))

        sel = is_oracle_all & (df["exposure_id"].astype(str) == asset)
        sids = df.loc[sel, "scenario_id"].astype(str)
        kinds = sids.map(lambda x: m.get(x, None))
        df.loc[sel & (kinds == "oracle_extreme"), "scenario_kind"] = "oracle_extreme"
        df.loc[sel & (kinds == "oracle_feasible"), "scenario_kind"] = "oracle_feasible"
        df.loc[sel & (kinds == "oracle_other"), "scenario_kind"] = "oracle_other"

    return df
def _fill_missing_dates_from_scenario_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """If start_date/end_date are missing, try to fill from scenario parquet using scenario_id.

    Priority:
      1) Use scenario_file column if present and exists.
      2) Else resolve canonical path by (exposure_id, scenario_kind).
    """
    if "scenario_id" not in df.columns:
        return df

    needs_start = ("start_date" not in df.columns) or df["start_date"].isna().any()
    needs_end = ("end_date" not in df.columns) or df["end_date"].isna().any()
    if not (needs_start or needs_end):
        return df

    d = df.copy()
    if "start_date" not in d.columns:
        d["start_date"] = pd.NaT
    if "end_date" not in d.columns:
        d["end_date"] = pd.NaT

    # Only attempt for rows with scenario_id
    d["scenario_id"] = d["scenario_id"].astype(str)

    # Iterate per asset+kind bucket (avoids loading many parquets)
    for (asset, kind), gidx in d.groupby(["exposure_id", "scenario_kind"], dropna=False).groups.items():
        asset = str(asset)
        kind = str(kind)

        path: Optional[str] = None
        if "scenario_file" in d.columns:
            cand = d.loc[gidx, "scenario_file"].astype(str)
            for p in cand.unique().tolist():
                if p and p != "nan" and Path(p).exists():
                    path = p
                    break

        if path is None:
            path = _resolve_scenario_path(asset, kind)

        scen = _load_scenario_df(path) if path else None
        if scen is None or "scenario_id" not in scen.columns:
            continue

        # Build mapping scenario_id -> dates
        scen2 = scen[["scenario_id"] + [c for c in ["start_date", "end_date"] if c in scen.columns]].copy()
        scen2["scenario_id"] = scen2["scenario_id"].astype(str)
        m_start = dict(zip(scen2["scenario_id"], scen2.get("start_date", pd.Series(index=scen2.index, dtype="datetime64[ns]")).tolist()))
        m_end = dict(zip(scen2["scenario_id"], scen2.get("end_date", pd.Series(index=scen2.index, dtype="datetime64[ns]")).tolist()))

        sel = d.index.isin(gidx)
        miss_start = sel & d["start_date"].isna()
        miss_end = sel & d["end_date"].isna()
        if miss_start.any():
            d.loc[miss_start, "start_date"] = d.loc[miss_start, "scenario_id"].map(lambda x: m_start.get(str(x), pd.NaT))
        if miss_end.any():
            d.loc[miss_end, "end_date"] = d.loc[miss_end, "scenario_id"].map(lambda x: m_end.get(str(x), pd.NaT))

    return d



def _fill_scenario_metadata_from_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Fill scenario-level metadata from canonical scenario parquet files using scenario_id.

    This repairs RL rows whose outputs usually omit scenario metadata such as
    oracle_series / oracle_pool / oracle_freq / label / volume_bbl.
    """
    if "scenario_id" not in df.columns or "exposure_id" not in df.columns or "scenario_kind" not in df.columns:
        return df

    d = df.copy()
    d["scenario_id"] = d["scenario_id"].astype(str)
    d["exposure_id"] = d["exposure_id"].astype(str)
    d["scenario_kind"] = d["scenario_kind"].fillna("unknown").astype(str)

    meta_cols = [
        "scenario_record_id",
        "start_date", "end_date",
        "horizon_days", "horizon_days_target", "horizon_days_realized", "volume_bbl",
        "oracle_series", "oracle_pool", "oracle_freq", "label", "tag",
        "oracle_candidate", "company_id", "company_size",
    ]

    for c in meta_cols:
        if c not in d.columns:
            d[c] = pd.NA

    for (asset, kind), gidx in d.groupby(["exposure_id", "scenario_kind"], dropna=False).groups.items():
        asset = str(asset)
        kind = str(kind)
        lookup_kind = "oracle_all" if kind in {"oracle_extreme", "oracle_feasible", "oracle_other"} else kind

        path = None
        if "scenario_file" in d.columns:
            cand = d.loc[gidx, "scenario_file"].astype(str)
            for pp in cand.unique().tolist():
                if pp and pp != "nan" and Path(pp).exists():
                    path = pp
                    break

        if path is None:
            path = _resolve_scenario_path(asset, lookup_kind)

        scen = _load_scenario_df(path) if path else None
        if scen is None or "scenario_id" not in scen.columns:
            continue

        scen = scen.copy()
        scen["scenario_id"] = scen["scenario_id"].astype(str)
        keep = ["scenario_id"] + [c for c in meta_cols if c in scen.columns]
        scen = scen[keep].drop_duplicates(subset=["scenario_id"], keep="first")

        left_idx = list(gidx)
        left = d.loc[left_idx, ["scenario_id"]].copy()
        left["_row_index"] = left_idx
        joined = left.merge(scen, on="scenario_id", how="left").set_index("_row_index")

        for c in meta_cols:
            if c not in joined.columns:
                continue

            vals = joined[c]

            if c in {"start_date", "end_date"}:
                vals = pd.to_datetime(vals, errors="coerce")
                cur = pd.to_datetime(d.loc[left_idx, c], errors="coerce")
                miss = cur.isna() & vals.notna()
            else:
                cur = d.loc[left_idx, c]
                miss = cur.isna() & vals.notna()
                if cur.dtype == object or str(cur.dtype).startswith("string"):
                    miss = (miss | cur.astype(str).str.lower().isin(["", "nan", "none", "<na>"])) & vals.notna()

            if miss.any():
                d.loc[pd.Index(left_idx)[miss.to_numpy()], c] = vals.loc[miss].to_numpy()

    # Always derive oracle_pool/oracle_freq/label from oracle_series when available.
    if "oracle_series" in d.columns:
        parsed = d["oracle_series"].map(_parse_oracle_series_fields)
        for col_name, pos in [("oracle_pool", 0), ("oracle_freq", 1), ("label", 2)]:
            vals = pd.Series([t[pos] for t in parsed], index=d.index)
            if col_name not in d.columns:
                d[col_name] = vals
            else:
                cur = d[col_name]
                miss = cur.isna() | cur.astype(str).str.lower().isin(["", "nan", "none", "<na>"])
                d.loc[miss & vals.notna(), col_name] = vals[miss & vals.notna()]

    return d


def infer_window_from_any(s: str) -> Optional[int]:
    s = str(s)
    for rx in WINDOW_RE_LIST:
        m = rx.search(s)
        if not m:
            continue
        w = m.group("w")
        if isinstance(w, str) and w.upper() == "NA":
            return None
        try:
            return int(w)
        except Exception:
            continue
    return None


def infer_mode_from_any(s: str, dynamic_val: Optional[object] = None) -> Optional[str]:
    # Prefer explicit dynamic column if it exists
    if dynamic_val is not None:
        try:
            if bool(dynamic_val):
                return "dynamic"
            return "static"
        except Exception:
            pass
    m = MODE_RE.search(s)
    if m:
        return m.group("mode").lower()
    return None


def normalize_roll_flag(x: str) -> str:
    x = (x or "").strip().lower().replace("_", "-")

    # numeric / boolean encodings
    if x in {"1", "true", "t", "yes", "y"}:
        return "roll"
    if x in {"0", "false", "f", "no", "n"}:
        return "noroll"

    if x == "roll":
        return "roll"
    if x in {"noroll", "no-roll", "no_roll", "no-roll"}:
        return "noroll"
    return "unknown"


def infer_roll_from_any(s: str, mode_roll_val: Optional[object] = None) -> Optional[str]:
    """Infer roll flag (roll/noroll) from either an explicit column or the filename tokens."""
    # 1) Prefer explicit column if it exists and is usable
    if mode_roll_val is not None:
        try:
            # common numeric encodings: 1=roll, 0=noroll
            if isinstance(mode_roll_val, (int, float, np.integer, np.floating)):
                if not pd.isna(mode_roll_val):
                    iv = int(mode_roll_val)
                    if iv == 1:
                        return "roll"
                    if iv == 0:
                        return "noroll"
            r = normalize_roll_flag(str(mode_roll_val))
            if r != "unknown":
                return r
        except Exception:
            pass

    # 2) Fall back to parsing from filename/path tokens
    m = ROLL_RE.search(str(s))
    if m:
        return normalize_roll_flag(m.group("roll"))

    return None


def infer_asset_from_path(p: str) -> Optional[str]:
    # Try to catch .../WTI_SPOT/... etc
    parts = Path(p).parts
    for part in parts:
        if part in {"WTI_SPOT", "BRENT_SPOT", "OPEC_BASKET"}:
            return part
    return None


def safe_float(x: object) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


# -----------------------------
# Metrics
# -----------------------------

@dataclass
class MetricSpec:
    alpha: float = 0.95
    tau: float = 0.0


def var_es(x: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Return (VaR_alpha, ES_alpha) as positive numbers for loss tail."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    q = np.quantile(x, 1.0 - alpha)
    var_a = -q
    tail = x[x <= q]
    if tail.size == 0:
        es_a = np.nan
    else:
        es_a = -float(np.mean(tail))
    return float(var_a), float(es_a)


def lpm(x: np.ndarray, tau: float, order: int) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    d = np.maximum(0.0, tau - x)
    return float(np.mean(d ** order))


def omega_ratio(x: np.ndarray, tau: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    gains = np.maximum(0.0, x - tau)
    losses = np.maximum(0.0, tau - x)
    denom = float(np.mean(losses))
    if denom <= 0:
        return np.nan
    return float(np.mean(gains) / denom)


def tail_ratio(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    q95 = np.quantile(x, 0.95)
    q05 = np.quantile(x, 0.05)
    if q05 == 0:
        return np.nan
    return float(abs(q95) / abs(q05))


def hedging_effectiveness(var_hedged: float, var_unhedged: float) -> float:
    if not np.isfinite(var_hedged) or not np.isfinite(var_unhedged) or var_unhedged <= 0:
        return np.nan
    return float(1.0 - (var_hedged / var_unhedged))


# -----------------------------
# Reporting logic
# -----------------------------

def read_inputs(inputs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    files: List[str] = []
    for pat in inputs:
        expanded = glob.glob(pat, recursive=True)
        if expanded:
            files.extend(expanded)
        else:
            # treat as literal
            if Path(pat).exists():
                files.append(pat)

    files = sorted({str(Path(f)) for f in files})
    if not files:
        raise FileNotFoundError("No input files matched. Provide --inputs with a parquet path or a glob.")

    dfs: List[pd.DataFrame] = []
    for f in files:
        if f.lower().endswith(".parquet"):
            df = pd.read_parquet(f)
        elif f.lower().endswith(".csv"):
            df = pd.read_csv(f)
        else:
            raise ValueError(f"Unsupported input: {f}")
        # Preserve original batch filename if it already exists (e.g., from FINAL_MERGED builder)
        if "source_file" not in df.columns:
            df["source_file"] = f
        df["_source_file"] = f
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out, files


def enrich_meta(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer full on-disk path when it contains orchestrator tokens; otherwise fall back to source_file.
    sig_path = df.get("_source_file", pd.Series(["" for _ in range(len(df))])).astype(str).fillna("")

    # Prefer full on-disk path when it contains orchestrator tokens; otherwise fall back to source_file.
    sig = sig_path
    path_rich = sig_path.str.contains(
        r"__oracle_|__scenarios_|__w\d+__|__dynamic__|__static__|__roll__|__noroll__",
        regex=True,
        case=False,
    )

    if "source_file" in df.columns:
        sf = df["source_file"].astype(str).fillna("")
        sf_rich = sf.str.contains(
            r"__oracle_|__scenarios_|__w\d+__|__dynamic__|__static__|__roll__|__noroll__",
            regex=True,
            case=False,
        )
        # if the full path is not rich but the stored source_file is, use source_file.
        sig = sig.where(path_rich, sf.where(sf_rich, sig))

    sig = sig.fillna("")
    # Asset
    if "exposure_id" not in df.columns:
        if "exposure" in df.columns:
            df = df.rename(columns={"exposure": "exposure_id"})
        else:
            df["exposure_id"] = sig.map(lambda p: infer_asset_from_path(p) or "UNKNOWN")

    # Strategy
    if "strategy" not in df.columns:
        raise ValueError("Input missing required column: strategy")

    # Scenario kind (baseline/oracle_all/oracle_universe/company)
    if "scenario_kind" not in df.columns:
        if "scenario_file" in df.columns:
            df["scenario_kind"] = df["scenario_file"].astype(str).map(infer_kind_from_any)
        else:
            df["scenario_kind"] = sig.map(infer_kind_from_any)
    # Normalize scenario_kind dtype early to avoid pandas merging object vs float64 when NaN exists.
    # Keep NaN as a real category for reporting comparability.
    df["scenario_kind"] = df["scenario_kind"].fillna("unknown").astype(str)
    # If scenario_kind was missing in upstream outputs or became 'unknown', try to recover from filenames.
    unk = df["scenario_kind"].astype(str).eq("unknown")
    if unk.any():
        df.loc[unk, "scenario_kind"] = sig[unk].map(infer_kind_from_any).fillna("unknown").astype(str)

    # Try to fill scenario_id for pipelines (e.g., LSTM_SIM) that may omit it, using scenario_file metadata.
    df = _fill_missing_scenario_id_from_scenario_file(df)

    # --- Split oracle_all into feasible vs extreme using oracle_series from scenario parquet ---
    df = _apply_oracle_series_split(df)

    # If scenario window dates or other scenario-level metadata are missing in hedge_summary,
    # fill them from scenario parquet. This repairs RL rows and restores oracle_series/BEST/WORST labels.
    df = _fill_missing_dates_from_scenario_parquet(df)
    df = _fill_scenario_metadata_from_parquet(df)

    # Window
    # NOTE: window/mode/roll are inferred from `sig` which is chosen to preserve orchestrator tokens.
    if "window" not in df.columns:
        # If another window-like column exists, reuse it.
        for alt in ["lookback", "ols_window", "strategy_window", "h_window", "corr_window", "garch_window"]:
            if alt in df.columns:
                df["window"] = pd.to_numeric(df[alt], errors="coerce")
                break
        else:
            df["window"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df["window"] = pd.to_numeric(df["window"], errors="coerce")

    # Backfill missing windows from filename/path tokens
    w_inf = sig.map(infer_window_from_any)
    df.loc[df["window"].isna() & w_inf.notna(), "window"] = w_inf[df["window"].isna() & w_inf.notna()].astype(int)

    # Final fallbacks: try other text fields (run_key/strategy)
    if df["window"].isna().any():
        if "run_key" in df.columns:
            w2 = df["run_key"].astype(str).map(infer_window_from_any)
            df.loc[df["window"].isna() & w2.notna(), "window"] = w2[df["window"].isna() & w2.notna()].astype(int)
        if "strategy" in df.columns:
            w3 = df["strategy"].astype(str).map(infer_window_from_any)
            df.loc[df["window"].isna() & w3.notna(), "window"] = w3[df["window"].isna() & w3.notna()].astype(int)

    df["window"] = pd.to_numeric(df["window"], errors="coerce").astype("Int64")
    

    # Mode (static/dynamic)
    if "mode" not in df.columns:
        df["mode"] = pd.NA

    mode_norm = df["mode"].astype(str).str.lower().replace({"nan": "", "none": ""})
    bad_mode = mode_norm.isna() | mode_norm.eq("") | mode_norm.eq("unknown")

    dyn_col = df["dynamic"] if "dynamic" in df.columns else None
    if dyn_col is not None:
        m_inf = pd.Series([infer_mode_from_any(s, d) for s, d in zip(sig, dyn_col)], index=df.index)
    else:
        m_inf = sig.map(lambda s: infer_mode_from_any(s))

    df.loc[bad_mode & m_inf.notna(), "mode"] = m_inf[bad_mode & m_inf.notna()]
    df["mode"] = df["mode"].fillna("unknown").astype(str)

    # Roll flag (roll/noroll)
    if "roll" not in df.columns:
        df["roll"] = pd.NA

    roll_norm = df["roll"].astype(str).str.lower().replace({"nan": "", "none": ""})
    bad_roll = roll_norm.isna() | roll_norm.eq("") | roll_norm.eq("unknown")

    mr = df["mode_roll"] if "mode_roll" in df.columns else None
    if mr is not None:
        r_inf = pd.Series([infer_roll_from_any(s, m) for s, m in zip(sig, mr)], index=df.index)
    else:
        r_inf = sig.map(lambda s: infer_roll_from_any(s))

    df.loc[bad_roll & r_inf.notna(), "roll"] = r_inf[bad_roll & r_inf.notna()]
    df["roll"] = df["roll"].fillna("unknown").astype(str).map(normalize_roll_flag)

    # normalize
    df["roll"] = df["roll"].astype(str).map(normalize_roll_flag)

    # A canonical run key (stable even if tags differ)
    df["run_key"] = (
        df["exposure_id"].astype(str)
        + "|" + df["scenario_kind"].astype(str)
        + "|" + df["strategy"].astype(str)
        + "|w" + df["window"].astype("Int64").astype(str).replace("<NA>", "NA")
        + "|" + df["mode"].astype(str)
        + "|" + df["roll"].astype(str)
    )

    return df


# ---------------------------------------------------------
# Optional: build FINAL_MERGED parquet files from BATCH outputs
# ---------------------------------------------------------
from typing import Dict
def build_final_merged_from_batch(
    batch_globs: List[str],
    merged_root: Path,
    final_tag: str = "final_nocomp",
    skip_company: bool = True,
) -> List[Path]:
    """Build FINAL_MERGED style outputs from BATCH_ALL hedge_summary parquet files.

    Writes files like:
      <merged_root>/<ASSET>/<KIND>/hedge_summary_<ASSET>_<KIND>_<final_tag>.parquet

    This is intentionally streaming (ParquetWriter) to avoid loading all BATCH results into RAM.
    It also ensures `window/mode/roll/scenario_kind` are present by enriching from the source file path.
    """

    if pa is None or pq is None:
        raise RuntimeError(
            "pyarrow is required for --build_from_batch streaming merge. Install with: pip install pyarrow"
        )

    # collect batch files
    files: List[str] = []
    for pat in batch_globs:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted({str(Path(f)) for f in files if str(f).lower().endswith(".parquet")})
    if not files:
        raise FileNotFoundError("No BATCH parquet files matched. Check --batch_glob")

    merged_root.mkdir(parents=True, exist_ok=True)

    writers: Dict[Tuple[str, str], pq.ParquetWriter] = {}
    out_paths: Dict[Tuple[str, str], Path] = {}

    def _get_out_path(asset: str, kind: str) -> Path:
        d = merged_root / asset / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"hedge_summary_{asset}_{kind}_{final_tag}.parquet"

    written_paths: List[Path] = []

    try:
        for i, f in enumerate(files, 1):
            df = pd.read_parquet(f)
            # Keep original batch path for downstream inference when merged filenames lose detail
            df["source_file"] = f
            df["_source_file"] = f

            # enrich meta (adds window/mode/roll/scenario_kind)
            df = enrich_meta(df)
            # Force stable dtypes so ParquetWriter schema is consistent across files
            df["window"] = df["window"].astype("Int64")
            for c in [
                "scenario_kind", "mode", "roll", "run_key", "strategy", "exposure_id", "scenario_id",
                "scenario_record_id", "scenario_file",
                "oracle_series", "oracle_pool", "oracle_freq", "label", "tag",
                "source_file", "_source_file",
                "start_date", "end_date",
            ]:
                if c in df.columns:
                    # dates kept as datetime; others as strings
                    if c in {"start_date", "end_date"}:
                        df[c] = pd.to_datetime(df[c], errors="coerce")
                    else:
                        df[c] = df[c].astype(str)

            # --- Canonical encodings to avoid schema drift across pipelines ---
            # dynamic: store as int64 {0,1} (some files use bool, some use ints)
            if "mode" in df.columns:
                df["dynamic"] = (df["mode"].astype(str).str.lower() == "dynamic").astype("int64")
            elif "dynamic" in df.columns:
                df["dynamic"] = df["dynamic"].astype(bool).astype("int64")
            else:
                df["dynamic"] = 0

            # mode_roll: store as canonical string {'roll','noroll'} for all inputs
            # (some files encode 0/1; LSTM_SIM uses strings)
            if "roll" in df.columns:
                df["mode_roll"] = df["roll"].astype(str)
            elif "mode_roll" in df.columns:
                df["mode_roll"] = df["mode_roll"].astype(str).map(normalize_roll_flag)
            else:
                df["mode_roll"] = "unknown"

            df["mode_roll"] = df["mode_roll"].astype(str)

            # skip company if requested
            if skip_company:
                df = df[df["scenario_kind"].astype(str) != "company"].copy()
                if df.empty:
                    continue

            # write per (asset, kind)
            for (asset, kind), g in df.groupby(["exposure_id", "scenario_kind"], dropna=False):
                asset = str(asset)
                kind = str(kind)
                if skip_company and kind == "company":
                    continue

                key = (asset, kind)
                out_p = out_paths.get(key)
                if out_p is None:
                    out_p = _get_out_path(asset, kind)
                    out_paths[key] = out_p
                    written_paths.append(out_p)

                table = pa.Table.from_pandas(g, preserve_index=False)

                w = writers.get(key)
                if w is None:
                    # overwrite existing file (fresh rebuild)
                    if out_p.exists():
                        out_p.unlink()

                    # Ensure the initial schema is stable (especially window)
                    schema = table.schema
                    # If window somehow ended up as null, coerce it to int64
                    if "window" in schema.names and pa.types.is_null(schema.field("window").type):
                        table = table.set_column(
                            schema.get_field_index("window"),
                            "window",
                            pa.array([None] * table.num_rows, type=pa.int64()),
                        )
                        schema = table.schema

                    w = pq.ParquetWriter(str(out_p), schema, compression="snappy")
                    writers[key] = w
                else:
                    # --- Align columns strictly to writer schema ---
                    expected_cols = w.schema.names
                    g2 = g.copy()

                    # Add missing columns
                    for col in expected_cols:
                        if col not in g2.columns:
                            g2[col] = np.nan

                    # Drop extra columns
                    extra_cols = [c for c in g2.columns if c not in expected_cols]
                    if extra_cols:
                        g2 = g2.drop(columns=extra_cols)

                    # Reorder columns
                    g2 = g2[expected_cols]

                    table = pa.Table.from_pandas(g2, preserve_index=False)

                    # Force cast to writer schema to avoid dtype mismatches
                    try:
                        table = table.cast(w.schema, safe=False)
                    except Exception:
                        # As last fallback, rebuild strictly using writer schema
                        table = pa.Table.from_pandas(g2, schema=w.schema, preserve_index=False)

                    w.write_table(table)

            if i % 50 == 0:
                print(f"[Baseline_report] build_from_batch progress: {i}/{len(files)}")

    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass

    return sorted(set(written_paths))


def _restrict_to_common_scenarios_for_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Force apples-to-apples scenario coverage across strategies.

    For each (exposure_id, scenario_kind) bucket we keep only scenario_id values present for ALL
    strategies that have non-empty scenario coverage in that bucket.
    """
    if "scenario_id" not in df.columns:
        return df

    keep_parts: List[pd.DataFrame] = []
    for (asset, kind), g in df.groupby(["exposure_id", "scenario_kind"], dropna=False):
        if g["strategy"].nunique(dropna=False) <= 1:
            keep_parts.append(g)
            continue

        sets: List[set] = []
        strategies_used: List[str] = []
        for strat, gs in g.groupby("strategy", dropna=False):
            sids = set(gs["scenario_id"].dropna().astype(str).unique().tolist())
            if len(sids) == 0:
                continue
            sets.append(sids)
            strategies_used.append(str(strat))

        if not sets:
            keep_parts.append(g)
            continue

        inter = set.intersection(*sets)
        if len(inter) == 0:
            print(
                f"[Baseline_report][WARN] common-scenario intersection empty for asset={asset} kind={kind} "
                f"strategies={strategies_used}. Keeping original rows."
            )
            keep_parts.append(g)
            continue

        keep_parts.append(g[g["scenario_id"].astype(str).isin(inter)].copy())

    return pd.concat(keep_parts, ignore_index=True) if keep_parts else df


def ensure_report_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the report-required PnL/cost columns exist.

    This report was originally designed for hedge_simulator outputs which have:
      spot_pnl_total, fut_pnl_total, cost_trade_total, cost_roll_total, net_pnl_total

    RL outputs may instead provide aggregated columns like:
      pnl_net_sum, cost_sum, reward_sum, etc.

    This function maps common RL column names into the report schema, and creates
    missing columns (spot/fut) as NaN if they are unavailable.

    We only *require* net_pnl_total to be present after mapping.
    """
    d = df.copy()

    # --- net PnL ---
    if "net_pnl_total" not in d.columns:
        for cand in [
            "pnl_net_sum",
            "pnl_net_total",
            "pnl_net",
            "net_pnl",
            "net_pnl_sum",
            "pnl_sum",
        ]:
            if cand in d.columns:
                d["net_pnl_total"] = pd.to_numeric(d[cand], errors="coerce")
                break

    # --- costs ---
    if "cost_trade_total" not in d.columns:
        for cand in ["cost_sum", "cost_total", "cost", "cost_trade"]:
            if cand in d.columns:
                d["cost_trade_total"] = pd.to_numeric(d[cand], errors="coerce")
                break
    if "cost_roll_total" not in d.columns:
        # If RL provides a separate roll cost, use it, else 0.
        if "cost_roll_total" in d.columns:
            pass
        elif "cost_roll" in d.columns:
            d["cost_roll_total"] = pd.to_numeric(d["cost_roll"], errors="coerce")
        else:
            d["cost_roll_total"] = 0.0

    # --- spot/futures decomposition ---
    # For RL episode summaries we often do not have spot/fut split; keep NaN.
    if "spot_pnl_total" not in d.columns:
        d["spot_pnl_total"] = np.nan
    if "fut_pnl_total" not in d.columns:
        d["fut_pnl_total"] = np.nan

    # Basic type normalization
    for c in ["spot_pnl_total", "fut_pnl_total", "cost_trade_total", "cost_roll_total", "net_pnl_total"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "net_pnl_total" not in d.columns:
        raise ValueError(
            "Input missing required net PnL column. Provide net_pnl_total or one of: "
            "pnl_net_sum / pnl_net_total / pnl_net / net_pnl / net_pnl_sum / pnl_sum"
        )

    return d


def compute_group_metrics(df: pd.DataFrame, spec: MetricSpec) -> pd.DataFrame:
    # Make the function robust to RL-only inputs (which may not have spot/fut split)
    df = ensure_report_schema(df)

    # Derived series
    df = df.copy()
    # Defensive dtype normalization: avoid merge errors when scenario_kind has mixed NaN/str.
    for c in ["exposure_id", "scenario_kind", "strategy", "mode", "roll"]:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype(str)
    df["gross_hedged"] = df["spot_pnl_total"].astype(float) + df["fut_pnl_total"].astype(float)
    # If spot/fut are not available (e.g., RL-only), gross_hedged will be all-NaN.
    # Keep it as NaN; HE on gross will naturally be NaN. Net-based metrics still work.
    df["total_cost"] = df["cost_trade_total"].astype(float) + df["cost_roll_total"].astype(float)

    # Granular grouping (legacy): keeps window/mode/roll separation
    group_cols = ["exposure_id", "scenario_kind", "strategy", "window", "mode", "roll"]

    # Aggregated grouping: one row per strategy per scenario_kind
    group_cols_simple = ["exposure_id", "scenario_kind", "strategy"]

    # Need unhedged variance within same asset/kind/mode/roll to compute HE
    base = df[df["strategy"].astype(str).str.lower().isin(["nohedge", "no_hedge", "no-hedge"])].copy()
    if base.empty:
        # still compute metrics but HE will be NaN
        base_var = pd.DataFrame(columns=["exposure_id", "scenario_kind", "mode", "roll", "var_unhedged_gross", "var_unhedged_spot", "var_unhedged_net"])  # type: ignore
    else:
        base_var = (
            base.groupby(["exposure_id", "scenario_kind", "mode", "roll"], dropna=False)
            .agg(
                var_unhedged_gross=("gross_hedged", "var"),
                var_unhedged_spot=("spot_pnl_total", "var"),
                var_unhedged_net=("net_pnl_total", "var"),
            )
            .reset_index()
        )

    def _agg(g: pd.DataFrame) -> Dict[str, object]:
        x_net = g["net_pnl_total"].to_numpy(dtype=float)
        x_gross = g["gross_hedged"].to_numpy(dtype=float)
        x_spot = g["spot_pnl_total"].to_numpy(dtype=float)

        mean_net = float(np.nanmean(x_net)) if np.isfinite(x_net).any() else np.nan
        std_net = float(np.nanstd(x_net, ddof=1)) if np.isfinite(x_net).sum() > 1 else np.nan
        cv = (std_net / abs(mean_net)) if (np.isfinite(std_net) and np.isfinite(mean_net) and abs(mean_net) > 1e-12) else np.nan

        var95, es95 = var_es(x_net, spec.alpha)
        lpm1 = lpm(x_net, spec.tau, 1)
        lpm2 = lpm(x_net, spec.tau, 2)
        dd = float(np.sqrt(lpm2)) if np.isfinite(lpm2) else np.nan
        omega = omega_ratio(x_net, spec.tau)
        tr = tail_ratio(x_net)

        skew = float(pd.Series(x_net).dropna().skew()) if np.isfinite(x_net).sum() > 2 else np.nan
        kurt = float(pd.Series(x_net).dropna().kurt()) if np.isfinite(x_net).sum() > 3 else np.nan

        out: Dict[str, object] = {
            "n_rows": int(len(g)),
            "n_scenarios": int(g["scenario_id"].nunique()) if "scenario_id" in g.columns else int(len(g)),
            "mean_net": mean_net,
            "median_net": float(np.nanmedian(x_net)) if np.isfinite(x_net).any() else np.nan,
            "std_net": std_net,
            "cv": cv,
            "min_net": float(np.nanmin(x_net)) if np.isfinite(x_net).any() else np.nan,
            "max_net": float(np.nanmax(x_net)) if np.isfinite(x_net).any() else np.nan,
            "var95": var95,
            "es95": es95,
            "lpm1": lpm1,
            "lpm2": lpm2,
            "downside_dev": dd,
            "omega": omega,
            "tail_ratio": tr,
            "skew": skew,
            "kurtosis_excess": kurt,
            "mean_spot": float(np.nanmean(x_spot)) if np.isfinite(x_spot).any() else np.nan,
            "mean_fut": float(np.nanmean(g["fut_pnl_total"].to_numpy(dtype=float))) if np.isfinite(g["fut_pnl_total"].to_numpy(dtype=float)).any() else np.nan,
            "mean_cost_trade": float(np.nanmean(g["cost_trade_total"].to_numpy(dtype=float))) if np.isfinite(g["cost_trade_total"].to_numpy(dtype=float)).any() else np.nan,
            "mean_cost_roll": float(np.nanmean(g["cost_roll_total"].to_numpy(dtype=float))) if np.isfinite(g["cost_roll_total"].to_numpy(dtype=float)).any() else np.nan,
            "mean_total_cost": float(np.nanmean(g["total_cost"].to_numpy(dtype=float))) if np.isfinite(g["total_cost"].to_numpy(dtype=float)).any() else np.nan,
            "mean_gross": float(np.nanmean(x_gross)) if np.isfinite(x_gross).any() else np.nan,
            "var_net": float(np.nanvar(x_net, ddof=1)) if np.isfinite(x_net).sum() > 1 else np.nan,
            "var_gross": float(np.nanvar(x_gross, ddof=1)) if np.isfinite(x_gross).sum() > 1 else np.nan,
            "var_spot": float(np.nanvar(x_spot, ddof=1)) if np.isfinite(x_spot).sum() > 1 else np.nan,
        }

        # drawdown proxies
        if "mdd_equity" in g.columns:
            try:
                out["mdd_equity_mean"] = float(np.nanmean(g["mdd_equity"].to_numpy(dtype=float)))
                out["mdd_equity_min"] = float(np.nanmin(g["mdd_equity"].to_numpy(dtype=float)))
            except Exception:
                out["mdd_equity_mean"] = np.nan
                out["mdd_equity_min"] = np.nan

        # turnover
        for tc in ["turnover_contracts", "turnover_h", "max_abs_contracts", "trade_contracts", "roll_contracts"]:
            if tc in g.columns:
                try:
                    out[f"{tc}_mean"] = float(np.nanmean(g[tc].to_numpy(dtype=float)))
                except Exception:
                    out[f"{tc}_mean"] = np.nan

        return out

    metrics = df.groupby(group_cols, dropna=False).apply(_agg).apply(pd.Series).reset_index()

    # --- Aggregated metrics (strategy-level only; no window/mode/roll split) ---
    # IMPORTANT: ensure comparable scenario coverage across strategies (esp. LSTM vs baselines).
    df_for_agg = _restrict_to_common_scenarios_for_agg(df)

    metrics_agg = (
        df_for_agg.groupby(group_cols_simple, dropna=False)
        .apply(_agg)
        .apply(pd.Series)
        .reset_index()
    )

    # For aggregated HE, compute unhedged variance using the same scenario coverage as df_for_agg
    base_agg = df_for_agg[df_for_agg["strategy"].astype(str).str.lower().isin(["nohedge", "no_hedge", "no-hedge"])].copy()
    if base_agg.empty:
        base_var_agg = pd.DataFrame(
            columns=["exposure_id", "scenario_kind", "var_unhedged_gross", "var_unhedged_spot", "var_unhedged_net"]
        )
    else:
        base_agg["gross_hedged"] = base_agg["spot_pnl_total"].astype(float) + base_agg["fut_pnl_total"].astype(float)
        base_var_agg = (
            base_agg.groupby(["exposure_id", "scenario_kind"], dropna=False)
            .agg(
                var_unhedged_gross=("gross_hedged", "var"),
                var_unhedged_spot=("spot_pnl_total", "var"),
                var_unhedged_net=("net_pnl_total", "var"),
            )
            .reset_index()
        )

    metrics_agg = metrics_agg.merge(base_var_agg, on=["exposure_id", "scenario_kind"], how="left")

    metrics_agg["he_gross_vs_nohedge"] = [
        hedging_effectiveness(vh, vu)
        for vh, vu in zip(
            metrics_agg["var_gross"].to_numpy(dtype=float),
            metrics_agg["var_unhedged_gross"].to_numpy(dtype=float),
        )
    ]
    metrics_agg["he_net_vs_nohedge"] = [
        hedging_effectiveness(vh, vu)
        for vh, vu in zip(
            metrics_agg["var_net"].to_numpy(dtype=float),
            metrics_agg["var_unhedged_net"].to_numpy(dtype=float),
        )
    ]
    metrics_agg["he_spot_vs_nohedge"] = [
        hedging_effectiveness(vh, vu)
        for vh, vu in zip(
            metrics_agg["var_spot"].to_numpy(dtype=float),
            metrics_agg["var_unhedged_spot"].to_numpy(dtype=float),
        )
    ]

    metrics_agg["sortino_tau0"] = np.where(
        metrics_agg["downside_dev"].to_numpy(dtype=float) > 0,
        metrics_agg["mean_net"].to_numpy(dtype=float) / metrics_agg["downside_dev"].to_numpy(dtype=float),
        np.nan,
    )

    # Merge in unhedged vars
    metrics = metrics.merge(base_var, on=["exposure_id", "scenario_kind", "mode", "roll"], how="left")

    metrics["he_gross_vs_nohedge"] = [
        hedging_effectiveness(vh, vu)
        for vh, vu in zip(metrics["var_gross"].to_numpy(dtype=float), metrics["var_unhedged_gross"].to_numpy(dtype=float))
    ]
    metrics["he_net_vs_nohedge"] = [
        hedging_effectiveness(vh, vu)
        for vh, vu in zip(metrics["var_net"].to_numpy(dtype=float), metrics["var_unhedged_net"].to_numpy(dtype=float))
    ]
    metrics["he_spot_vs_nohedge"] = [
        hedging_effectiveness(vh, vu)
        for vh, vu in zip(metrics["var_spot"].to_numpy(dtype=float), metrics["var_unhedged_spot"].to_numpy(dtype=float))
    ]

    # Sortino-like / Sharpe-like (use tau=0 baseline)
    metrics["sortino_tau0"] = np.where(
        metrics["downside_dev"].to_numpy(dtype=float) > 0,
        metrics["mean_net"].to_numpy(dtype=float) / metrics["downside_dev"].to_numpy(dtype=float),
        np.nan,
    )

    # Basic ranking score (tunable): maximize HE_net, minimize ES, minimize MDD, maximize mean_net
    # This is not a "final truth"; it is for quick sanity-check ordering.
    # Normalize within each asset+kind.
    def _z(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True)
        if not np.isfinite(sd) or sd == 0:
            return x * np.nan
        return (x - mu) / sd

    metrics["score_quick"] = 0.0
    for (asset, kind), idx in metrics.groupby(["exposure_id", "scenario_kind"], dropna=False).groups.items():
        sub = metrics.loc[idx]
        # higher is better
        z_mean = _z(sub["mean_net"])
        z_he = _z(sub["he_net_vs_nohedge"])
        z_sort = _z(sub["sortino_tau0"])
        # lower is better -> negate
        z_es = -_z(sub["es95"])
        z_mdd = -_z(sub.get("mdd_equity_mean", pd.Series(index=sub.index, dtype=float)))
        score = 0.35 * z_he + 0.25 * z_es + 0.25 * z_mdd + 0.15 * z_mean
        # fallback if mdd missing
        if "mdd_equity_mean" not in sub.columns:
            score = 0.45 * z_he + 0.30 * z_es + 0.25 * z_mean
        metrics.loc[idx, "score_quick"] = score.to_numpy()

    # Quick score for aggregated table too (same recipe, normalized within asset+kind)
    metrics_agg["score_quick"] = 0.0
    for (asset, kind), idx in metrics_agg.groupby(["exposure_id", "scenario_kind"], dropna=False).groups.items():
        sub = metrics_agg.loc[idx]
        z_mean = _z(sub["mean_net"])
        z_he = _z(sub["he_net_vs_nohedge"])
        z_es = -_z(sub["es95"])
        z_mdd = -_z(sub.get("mdd_equity_mean", pd.Series(index=sub.index, dtype=float)))
        score = 0.35 * z_he + 0.25 * z_es + 0.25 * z_mdd + 0.15 * z_mean
        if "mdd_equity_mean" not in sub.columns:
            score = 0.45 * z_he + 0.30 * z_es + 0.25 * z_mean
        metrics_agg.loc[idx, "score_quick"] = score.to_numpy()

    return metrics, metrics_agg



# -----------------------------
# RL ingestion and RL reference alignment helper
# -----------------------------


# --- RL ingestion helper ---
def ingest_rl_results(rl_df: pd.DataFrame) -> pd.DataFrame:
    """Convert RL aggregated outputs to the baseline-report schema.

    The RL pipeline may output episode-level aggregates with columns like:
      - pnl_net_sum / pnl_net / net_pnl_total
      - cost_sum / cost_total / cost
      - scenario_id, exposure_id
      - dataset (baseline/oracle_universe/oracle_all)

    We map those into the columns expected by this report so RL shows up as a strategy.
    We intentionally set mode='dynamic' and roll='roll' (locked design), and window=<NA>.
    """
    if rl_df is None or rl_df.empty:
        return pd.DataFrame()

    d = rl_df.copy()

    # Required identifiers
    if "exposure_id" not in d.columns:
        raise ValueError("RL reference missing exposure_id")
    if "scenario_id" not in d.columns:
        raise ValueError("RL reference missing scenario_id")

    d["exposure_id"] = d["exposure_id"].astype(str)
    d["scenario_id"] = d["scenario_id"].astype(str)

    # Map dataset -> scenario_kind if possible
    if "scenario_kind" not in d.columns:
        if "dataset" in d.columns:
            d["scenario_kind"] = d["dataset"].astype(str).map(infer_kind_from_any)
        else:
            # fallback
            d["scenario_kind"] = "unknown"
    # Normalize for consistency with baseline rows
    d["scenario_kind"] = d["scenario_kind"].fillna("unknown").astype(str)

    # Strategy name
    if "strategy" not in d.columns:
        # If algo exists, incorporate it
        if "algo" in d.columns:
            d["strategy"] = "RL_" + d["algo"].astype(str)
        else:
            d["strategy"] = "RL"

    # PnL mapping
    if "net_pnl_total" in d.columns:
        net = d["net_pnl_total"].astype(float)
    elif "pnl_net_sum" in d.columns:
        net = d["pnl_net_sum"].astype(float)
    elif "pnl_net" in d.columns:
        net = d["pnl_net"].astype(float)
    else:
        raise ValueError("RL reference missing net PnL column (net_pnl_total / pnl_net_sum / pnl_net)")

    # Cost mapping
    if "cost_trade_total" in d.columns and "cost_roll_total" in d.columns:
        c_trade = d["cost_trade_total"].astype(float)
        c_roll = d["cost_roll_total"].astype(float)
    elif "cost_sum" in d.columns:
        c_trade = d["cost_sum"].astype(float)
        c_roll = 0.0
    elif "cost_total" in d.columns:
        c_trade = d["cost_total"].astype(float)
        c_roll = 0.0
    elif "cost" in d.columns:
        c_trade = d["cost"].astype(float)
        c_roll = 0.0
    else:
        # Allow missing; set to 0 but warn
        print("[Baseline_report][WARN] RL reference missing cost columns; assuming 0.")
        c_trade = 0.0
        c_roll = 0.0

    out = pd.DataFrame({
        "exposure_id": d["exposure_id"].astype(str),
        "scenario_kind": d["scenario_kind"].astype(str),
        "strategy": d["strategy"].astype(str),
        "scenario_id": d["scenario_id"].astype(str),
        "scenario_record_id": d["scenario_record_id"].astype(str) if "scenario_record_id" in d.columns else pd.NA,
        # Core totals expected by report
        "spot_pnl_total": np.nan,  # RL pipeline may not export this; keep NaN
        "fut_pnl_total": np.nan,
        "cost_trade_total": c_trade,
        "cost_roll_total": c_roll,
        "net_pnl_total": net,
        # Force locked settings
        "mode": "dynamic",
        "roll": "roll",
        "mode_roll": "roll",
        "dynamic": 1,
        "window": pd.Series([pd.NA] * len(d), dtype="Int64"),
    })

    # window_id is a label (often WF_train...); keep it as string.
    if "window_id" in d.columns:
        out["window_id"] = d["window_id"].astype(str).fillna("rl_full")
    elif "window_label" in d.columns:
        out["window_id"] = d["window_label"].astype(str).fillna("rl_full")
    else:
        out["window_id"] = "rl_full"

    out["window_id"] = out["window_id"].astype(str)

    # Optional passthroughs if present
    for col in ["start_date", "end_date", "horizon_days", "horizon_days_target", "horizon_days_realized", "volume_bbl", "oracle_series", "oracle_pool", "oracle_freq", "label", "tag", "dataset", "split", "window_id", "window_label"]:
        if col in d.columns and col not in out.columns:
            out[col] = d[col]

    out["_source_file"] = "__RL__"

    return out


def align_with_rl_reference(df: pd.DataFrame, rl_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict rows to the exact scenario coverage present in RL reference.

    Matching keys (in priority order):
      1) (exposure_id, scenario_kind, scenario_id, scenario_record_id) if record id exists in BOTH
      2) else (exposure_id, scenario_kind, scenario_id)
    """
    if rl_df is None or rl_df.empty:
        return df

    if "scenario_id" not in df.columns or "scenario_id" not in rl_df.columns:
        print("[Baseline_report][WARN] scenario_id missing; cannot align with RL reference.")
        return df

    df2 = df.copy()
    rl2 = rl_df.copy()

    # Ensure scenario_kind exists on RL side
    if "scenario_kind" not in rl2.columns:
        if "dataset" in rl2.columns:
            rl2["scenario_kind"] = rl2["dataset"].astype(str).map(infer_kind_from_any)
        else:
            rl2["scenario_kind"] = "unknown"

    if "scenario_kind" not in df2.columns:
        df2["scenario_kind"] = "unknown"

    for col in ["exposure_id", "scenario_kind", "scenario_id"]:
        df2[col] = df2[col].astype(str)
        rl2[col] = rl2[col].astype(str)

    # Prefer oracle_all disambiguation when possible
    use_record = ("scenario_record_id" in rl2.columns) and ("scenario_record_id" in df2.columns)
    if use_record:
        df2["scenario_record_id"] = df2["scenario_record_id"].astype(str)
        rl2["scenario_record_id"] = rl2["scenario_record_id"].astype(str)
        allowed = rl2[["exposure_id", "scenario_kind", "scenario_id", "scenario_record_id"]].dropna().drop_duplicates()
        merged = df2.merge(
            allowed,
            on=["exposure_id", "scenario_kind", "scenario_id", "scenario_record_id"],
            how="inner"
        )
        print(f"[Baseline_report] RL-aligned rows (with record_id): before={len(df2)} after={len(merged)}")
        return merged

    # Build allowed keys. If RL kinds are unknown or missing, fall back to (exposure_id, scenario_id).
    allowed_kind = rl2[["exposure_id", "scenario_kind", "scenario_id"]].dropna().drop_duplicates()

    # If RL scenario_kind is unusable (all unknown), align only by (exposure_id, scenario_id)
    rl_kinds = set(allowed_kind["scenario_kind"].astype(str).unique().tolist()) if not allowed_kind.empty else set()
    if (not allowed_kind.empty) and (rl_kinds.issubset({"unknown"}) or rl_kinds == {"unknown"}):
        allowed2 = rl2[["exposure_id", "scenario_id"]].dropna().drop_duplicates()
        merged = df2.merge(allowed2, on=["exposure_id", "scenario_id"], how="inner")
        print(f"[Baseline_report] RL-aligned rows (fallback no-kind): before={len(df2)} after={len(merged)}")
        return merged

    # Normal path: align on (asset, kind, scenario_id)
    if allowed_kind.empty:
        print("[Baseline_report][WARN] RL reference has no valid (asset, kind, scenario_id) keys.")
        return df2

    merged = df2.merge(allowed_kind, on=["exposure_id", "scenario_kind", "scenario_id"], how="inner")
    print(f"[Baseline_report] RL-aligned rows: before={len(df2)} after={len(merged)}")
    return merged

def write_df_sheet(writer: pd.ExcelWriter, df: pd.DataFrame, sheet: str, freeze: Tuple[int, int] = (1, 0)) -> None:
    df.to_excel(writer, sheet_name=sheet, index=False)
    ws = writer.sheets[sheet]
    # freeze header row
    ws.freeze_panes(*freeze)
    # basic column widths
    for i, col in enumerate(df.columns):
        width = min(45, max(10, int(len(str(col)) * 1.2)))
        ws.set_column(i, i, width)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out_png: Path, rotate: int = 30) -> None:
    if df.empty:
        return
    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(111)
    ax.bar(df[x].astype(str), df[y].astype(float))
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", labelrotation=rotate)
    ax.grid(True, axis="y", alpha=0.25)
    save_plot(out_png)


def plot_cost_pie(df_row: pd.Series, title: str, out_png: Path) -> None:
    trade = safe_float(df_row.get("mean_cost_trade", np.nan))
    roll = safe_float(df_row.get("mean_cost_roll", np.nan))
    other = 0.0
    vals = [trade, roll]
    labels = ["trade_cost", "roll_cost"]
    # clean non-finite
    vals2, labels2 = [], []
    for v, l in zip(vals, labels):
        if np.isfinite(v) and abs(v) > 0:
            vals2.append(abs(v))
            labels2.append(l)
    if not vals2:
        return
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    ax.pie(vals2, labels=labels2, autopct="%1.1f%%")
    ax.set_title(title)
    save_plot(out_png)


def embed_images(ws, images: List[Tuple[str, str, float]]) -> None:
    """images: list of (cell, png_path, scale)"""
    for cell, img_path, scale in images:
        if Path(img_path).exists():
            ws.insert_image(cell, img_path, {"x_scale": scale, "y_scale": scale})


def build_excel_for_asset(
    asset: str,
    df_asset: pd.DataFrame,
    metrics: pd.DataFrame,
    metrics_agg: pd.DataFrame,
    out_dir: Path,
    spec: MetricSpec,
    temp_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / f"report_{asset}.xlsx"

    # A few "main" comparison slices for charts
    # 1) Primary: dynamic+roll, window in {120, 252}
    main_dyn_roll = metrics[
        (metrics["exposure_id"] == asset)
        & (metrics["mode"] == "dynamic")
        & (metrics["roll"] == "roll")
        & (metrics["window"].isin([120, 252]) | metrics["window"].isna())
    ].copy()

    # 2) Roll vs noroll (dynamic), window=120 preferred
    roll_vs = metrics[
        (metrics["exposure_id"] == asset)
        & (metrics["mode"] == "dynamic")
        & (metrics["window"].fillna(-1).astype(int).isin([120, -1]))
        & (metrics["roll"].isin(["roll", "noroll"]))
    ].copy()

    # 3) Static vs dynamic (roll), window=120 preferred
    mode_vs = metrics[
        (metrics["exposure_id"] == asset)
        & (metrics["roll"] == "roll")
        & (metrics["window"].fillna(-1).astype(int).isin([120, -1]))
        & (metrics["mode"].isin(["static", "dynamic"]))
    ].copy()

    # Rankings
    ranks = metrics[metrics["exposure_id"] == asset].copy()
    ranks = ranks.sort_values(["scenario_kind", "score_quick"], ascending=[True, False])

    # Create Excel
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        wb = writer.book
        # Formats
        fmt_title = wb.add_format({"bold": True, "font_size": 14})
        fmt_note = wb.add_format({"font_color": "#666666", "font_size": 10})
        fmt_pct = wb.add_format({"num_format": "0.00%"})
        fmt_num = wb.add_format({"num_format": "0.00"})
        fmt_money = wb.add_format({"num_format": "#,##0"})

        # Sheet: README
        sheet = "README"
        ws = wb.add_worksheet(sheet)
        writer.sheets[sheet] = ws
        ws.write(0, 0, f"Report for {asset}", fmt_title)
        ws.write(2, 0, f"Generated: {ts_utc()}")
        ws.write(3, 0, f"VaR/ES alpha: {spec.alpha}")
        ws.write(4, 0, f"Omega/Downside threshold tau: {spec.tau}")
        ws.write(6, 0, "Key columns used: net_pnl_total, spot_pnl_total, fut_pnl_total, cost_trade_total, cost_roll_total")
        ws.write(7, 0, "Hedging Effectiveness (HE): 1 - Var(hedged)/Var(NoHedge) within the same asset+scenario_kind+mode+roll")
        ws.write(9, 0, "Tip: add LSTM/Transformer later -> just add new strategy name in outputs; this report will pick it up automatically.", fmt_note)

        # Sheet: METRICS
        m = ranks.copy()
        # nicer ordering
        cols_first = [
            "exposure_id", "scenario_kind", "strategy", "window", "mode", "roll",
            "n_scenarios", "mean_net", "std_net", "cv", "var95", "es95",
            "he_net_vs_nohedge", "he_gross_vs_nohedge", "downside_dev", "omega", "tail_ratio",
            "mdd_equity_mean", "turnover_contracts_mean", "max_abs_contracts_mean",
            "mean_cost_trade", "mean_cost_roll", "mean_total_cost",
            "score_quick",
        ]
        for c in cols_first:
            if c not in m.columns:
                m[c] = np.nan
        m = m[cols_first + [c for c in m.columns if c not in cols_first]]
        write_df_sheet(writer, m, "METRICS")

        # Sheet: METRICS_AGG (one row per strategy per scenario_kind; no window/mode/roll split)
        m_agg = metrics_agg[metrics_agg["exposure_id"] == asset].copy()
        # Prefer a similar column ordering if columns exist
        cols_first_agg = [
            "exposure_id", "scenario_kind", "strategy",
            "n_scenarios", "mean_net", "std_net", "cv", "var95", "es95",
            "he_net_vs_nohedge", "he_gross_vs_nohedge", "downside_dev", "omega", "tail_ratio",
            "mdd_equity_mean", "turnover_contracts_mean", "max_abs_contracts_mean",
            "mean_cost_trade", "mean_cost_roll", "mean_total_cost",
            "score_quick",
        ]
        for c in cols_first_agg:
            if c not in m_agg.columns:
                m_agg[c] = np.nan
        m_agg = m_agg[cols_first_agg + [c for c in m_agg.columns if c not in cols_first_agg]]
        write_df_sheet(writer, m_agg, "METRICS_AGG")

        # Sheet: RANKING (top per kind)
        top_rows = []
        for kind in ["baseline", "oracle_feasible", "oracle_extreme", "oracle_universe", "company", "unknown"]:
            sub = ranks[ranks["scenario_kind"] == kind]
            if sub.empty:
                continue
            top_rows.append(sub.head(15))
        top = pd.concat(top_rows, ignore_index=True) if top_rows else ranks.head(20)
        write_df_sheet(writer, top, "TOP15_PER_KIND")

        # Sheet: CHARTS
        ws_c = wb.add_worksheet("CHARTS")
        writer.sheets["CHARTS"] = ws_c
        ws_c.write(0, 0, f"Charts for {asset}", fmt_title)

        images: List[Tuple[str, str, float]] = []
        row_cursor = 2

        def _chart_block(title: str, pngs: List[Path]) -> None:
            nonlocal row_cursor
            ws_c.write(row_cursor, 0, title, wb.add_format({"bold": True, "font_size": 12}))
            row_cursor += 1
            col = 0
            for p in pngs:
                cell = xl_rowcol_to_cell(row_cursor, col)
                images.append((cell, str(p), 0.9))
                col += 8
            row_cursor += 16

        # Need xlsxwriter utility
        from xlsxwriter.utility import xl_rowcol_to_cell

        # Build bar charts per scenario_kind focusing on the most meaningful comparisons
        chart_pngs: List[Path] = []
        for kind in ["baseline", "oracle_feasible", "oracle_extreme", "oracle_universe", "company"]:
            sub = main_dyn_roll[main_dyn_roll["scenario_kind"] == kind].copy()
            if sub.empty:
                continue
            # pick window 120 first, else any
            if (sub["window"] == 120).any():
                sub = sub[(sub["window"] == 120) | (sub["window"].isna())]
            sub = sub.sort_values("score_quick", ascending=False)
            # bar: HE net
            p1 = temp_dir / f"{asset}_{kind}_he_net.png"
            plot_bar(
                sub.assign(lbl=sub["strategy"].astype(str) + "|w" +
                           sub["window"].astype("Int64").astype(str).replace("<NA>", "NA")),
                "lbl",
                "he_net_vs_nohedge",
                f"{kind}: HE(net) vs NoHedge (dyn+roll)",
                p1
            )
            # bar: ES
            p2 = temp_dir / f"{asset}_{kind}_es95.png"
            plot_bar(
                sub.assign(lbl=sub["strategy"].astype(str) + "|w" +
                           sub["window"].astype("Int64").astype(str).replace("<NA>", "NA")),
                "lbl",
                "es95",
                f"{kind}: ES@{int(spec.alpha*100)}% (net PnL)",
                p2
            )
            # bar: mean net
            p3 = temp_dir / f"{asset}_{kind}_mean_net.png"
            plot_bar(
                sub.assign(lbl=sub["strategy"].astype(str) + "|w" +
                           sub["window"].astype("Int64").astype(str).replace("<NA>", "NA")),
                "lbl",
                "mean_net",
                f"{kind}: Mean net PnL",
                p3
            )
            chart_pngs.extend([p1, p2, p3])

            # pie: cost breakdown for best run
            best = sub.head(1)
            if not best.empty:
                p4 = temp_dir / f"{asset}_{kind}_cost_pie.png"
                plot_cost_pie(best.iloc[0], f"{kind}: cost share (best by score)", p4)
                chart_pngs.append(p4)

            _chart_block(f"{kind} (dyn+roll, main windows)", [p1, p2])
            _chart_block(f"{kind} (more)", [p3, p4] if Path(p4).exists() else [p3])

        # Roll vs noroll: HE and ES (baseline + oracle_feasible + oracle_extreme)
        for kind in ["baseline", "oracle_feasible", "oracle_extreme"]:
            sub = roll_vs[roll_vs["scenario_kind"] == kind].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["strategy", "roll"])
            sub["lbl"] = sub["strategy"].astype(str) + "|" + sub["roll"].astype(str)
            p1 = temp_dir / f"{asset}_{kind}_roll_vs_he.png"
            plot_bar(sub, "lbl", "he_net_vs_nohedge", f"{kind}: HE(net) roll vs noroll (dyn, w120 pref)", p1)
            p2 = temp_dir / f"{asset}_{kind}_roll_vs_es.png"
            plot_bar(sub, "lbl", "es95", f"{kind}: ES roll vs noroll (dyn, w120 pref)", p2)
            _chart_block(f"{kind}: roll vs noroll", [p1, p2])

        # Mode static vs dynamic (roll)
        for kind in ["baseline", "oracle_feasible", "oracle_extreme"]:
            sub = mode_vs[mode_vs["scenario_kind"] == kind].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["strategy", "mode"])
            sub["lbl"] = sub["strategy"].astype(str) + "|" + sub["mode"].astype(str)
            p1 = temp_dir / f"{asset}_{kind}_mode_vs_he.png"
            plot_bar(sub, "lbl", "he_net_vs_nohedge", f"{kind}: HE(net) static vs dynamic (roll, w120 pref)", p1)
            p2 = temp_dir / f"{asset}_{kind}_mode_vs_es.png"
            plot_bar(sub, "lbl", "es95", f"{kind}: ES static vs dynamic (roll, w120 pref)", p2)
            _chart_block(f"{kind}: static vs dynamic", [p1, p2])

        embed_images(ws_c, images)

        # Sheet: RAW_SAMPLES (small slice)
        sample = df_asset.head(5000).copy()
        write_df_sheet(writer, sample, "RAW_SAMPLE")

        # A nicer numeric format on METRICS sheet
        ws_m = writer.sheets["METRICS"]
        # set percent formats on HE columns if present
        for col_name in ["he_net_vs_nohedge", "he_gross_vs_nohedge", "he_spot_vs_nohedge"]:
            if col_name in m.columns:
                idx = m.columns.get_loc(col_name)
                ws_m.set_column(idx, idx, 14, fmt_pct)

        # common numeric formatting columns
        for col_name in ["mean_net", "std_net", "var95", "es95", "mean_total_cost", "mean_cost_trade", "mean_cost_roll", "mean_spot", "mean_fut"]:
            if col_name in m.columns:
                idx = m.columns.get_loc(col_name)
                ws_m.set_column(idx, idx, 14, fmt_money)

    return out_xlsx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more parquet/csv paths or globs. Example: results/BATCH_ALL/*/*/hedge_summary_*.parquet",
    )
    ap.add_argument("--out_dir", default="reports/final_excel", help="Output directory for generated Excel reports")
    ap.add_argument("--alpha", type=float, default=0.95, help="Confidence level for VaR/ES (e.g., 0.95)")
    ap.add_argument("--tau", type=float, default=0.0, help="Threshold tau for Omega/LPM/Downside deviation")
    ap.add_argument("--assets", nargs="*", default=None, help="Optional subset of assets (e.g., WTI_SPOT BRENT_SPOT OPEC_BASKET)")
    ap.add_argument(
        "--only_opec",
        action="store_true",
        help="If set, restrict processing strictly to OPEC_BASKET for faster runs.",
    )
    ap.add_argument(
        "--build_from_batch",
        action="store_true",
        help="If set, first build FINAL_MERGED parquet files from BATCH hedge_summary outputs (streaming) and then run the Excel report on them.",
    )
    ap.add_argument(
        "--batch_glob",
        nargs="+",
        default=[
            "results/BATCH_ALL/**/hedge_summary_*.parquet",
            "results/BASELINE_BATCH/**/hedge_summary_*.parquet",
        ],
        help=(
            "One or more recursive globs for RAW batch hedge_summary parquet files. "
            "These raw files preserve filename tokens such as window/mode/roll and are preferred "
            "for rebuilding a clean merged layer before reporting."
        ),
    )
    ap.add_argument(
        "--merged_root",
        default="results/FINAL_MERGED",
        help="Where to write merged FINAL_MERGED outputs when --build_from_batch is used.",
    )
    ap.add_argument(
        "--final_tag",
        default="final_nocomp",
        help="Suffix tag used in merged filenames when --build_from_batch is used.",
    )
    ap.add_argument(
        "--include_company",
        action="store_true",
        help="If set with --build_from_batch, company files will also be built. (Default: company is skipped.)",
    )
    ap.add_argument(
        "--include_lstm_sim",
        action="store_true",
        help="If set with --build_from_batch, also merge LSTM simulation hedge_summary outputs from --lstm_sim_root into FINAL_MERGED (so LSTM strategies appear alongside baselines).",
    )
    ap.add_argument(
        "--lstm_sim_root",
        default="results/LSTM_SIM",
        help="Root folder for LSTM simulation outputs (expects **/hedge_summary_*.parquet under it).",
    )
    ap.add_argument(
        "--rl_reference",
        default=None,
        help="Optional: path to RL aggregated parquet/csv. If provided, baseline strategies will be restricted to the same (asset, scenario_id) coverage as RL before computing metrics.",
    )
    ap.add_argument(
        "--include_rl",
        action="store_true",
        help="If set (and --rl_reference is provided), append RL episode results as a strategy so it is directly compared in the report.",
    )
    ap.add_argument(
        "--show_progress",
        action="store_true",
        help="If set, show simple progress information while processing assets.",
    )
    ap.add_argument(
    "--strict_intersection",
    action="store_true",
    help="If set, restrict ALL rows to common scenario_id intersection across strategies within each (asset, kind).",
    )
    ap.add_argument(
    "--align_to_rl_reference",
    action="store_true",
    help=(
        "If set (and --rl_reference is provided), restrict rows to RL scenario coverage "
        "before computing metrics. If NOT set, keep full baseline coverage (RL can still be appended)."
    ),
    )
    ap.add_argument(
        "--auto_rebuild_from_batch_if_meta_missing",
        action="store_true",
        help=(
            "If set, and the provided inputs appear to be merged files with poor metadata quality "
            "(e.g. window missing, mode/roll=unknown), automatically rebuild a clean merged layer "
            "from RAW batch files matched by --batch_glob, then continue reporting from that rebuilt layer."
        ),
    )
    return ap.parse_args()


def _expand_globs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return sorted({str(Path(f)) for f in files if str(f).lower().endswith((".parquet", ".csv"))})


def _meta_quality_is_poor(df: pd.DataFrame) -> bool:
    """Heuristic: merged inputs are poor if window is mostly missing and mode/roll are mostly unknown.

    We use this to decide whether to rebuild a clean merged layer from RAW batch outputs whose
    filenames still carry the orchestrator tokens (__w30__dynamic__noroll__ etc.).
    """
    if df is None or df.empty:
        return False

    # window quality
    if "window" not in df.columns:
        window_poor = True
    else:
        w = pd.to_numeric(df["window"], errors="coerce")
        window_poor = (w.notna().mean() < 0.10)

    # mode quality
    if "mode" not in df.columns:
        mode_poor = True
    else:
        m = df["mode"].astype(str).str.lower()
        mode_poor = (m.isin(["unknown", "", "nan", "none"]).mean() > 0.80)

    # roll quality
    if "roll" not in df.columns:
        roll_poor = True
    else:
        r = df["roll"].astype(str).str.lower()
        roll_poor = (r.isin(["unknown", "", "nan", "none"]).mean() > 0.80)

    return bool(window_poor and mode_poor and roll_poor)


def main() -> int:
    args = parse_args()
    spec = MetricSpec(alpha=float(args.alpha), tau=float(args.tau))

    print(f"[Baseline_report] start {ts_utc()}")

    # Optional: build merged FINAL_MERGED files from raw BATCH outputs (and ensure window column exists)
    if getattr(args, "build_from_batch", False):
        merged_root = Path(getattr(args, "merged_root", "results/FINAL_MERGED"))
        skip_company = not bool(getattr(args, "include_company", False))
        batch_globs = list(getattr(args, "batch_glob", ["results/BATCH_ALL/**/hedge_summary_*.parquet"]))

        # Optionally include LSTM simulation outputs (already in hedge_summary_*.parquet form)
        if bool(getattr(args, "include_lstm_sim", False)):
            lstm_root = str(getattr(args, "lstm_sim_root", "results/LSTM_SIM")).rstrip("/")
            batch_globs.append(f"{lstm_root}/**/hedge_summary_*.parquet")

        built = build_final_merged_from_batch(
            batch_globs=batch_globs,
            merged_root=merged_root,
            final_tag=str(getattr(args, "final_tag", "final_nocomp")),
            skip_company=skip_company,
        )
        print(f"[Baseline_report] build_from_batch wrote files={len(built)} root={merged_root}")

        # Override inputs to point at the freshly built merged files
        args.inputs = [str(merged_root / "**" / f"hedge_summary_*{args.final_tag}*.parquet")]

    df, files = read_inputs(args.inputs)

    # Optional auto-rebuild path: if the provided inputs are merged files that lost fine-grained
    # metadata (window/mode/roll), rebuild a clean merged layer from RAW batch outputs first.
    if bool(getattr(args, "auto_rebuild_from_batch_if_meta_missing", False)):
        probe = enrich_meta(df.copy())
        if _meta_quality_is_poor(probe):
            raw_batch_files = _expand_globs(list(getattr(args, "batch_glob", [])))
            if raw_batch_files:
                merged_root = Path(getattr(args, "merged_root", "results/FINAL_MERGED"))
                skip_company = not bool(getattr(args, "include_company", False))
                batch_globs = list(getattr(args, "batch_glob", []))

                if bool(getattr(args, "include_lstm_sim", False)):
                    lstm_root = str(getattr(args, "lstm_sim_root", "results/LSTM_SIM")).rstrip("/")
                    batch_globs.append(f"{lstm_root}/**/hedge_summary_*.parquet")

                built = build_final_merged_from_batch(
                    batch_globs=batch_globs,
                    merged_root=merged_root,
                    final_tag=str(getattr(args, "final_tag", "final_nocomp")),
                    skip_company=skip_company,
                )
                print(
                    f"[Baseline_report] auto-rebuild triggered due to poor merged metadata; "
                    f"rebuilt files={len(built)} root={merged_root}"
                )

                args.inputs = [str(merged_root / "**" / f"hedge_summary_*{args.final_tag}*.parquet")]
                df, files = read_inputs(args.inputs)
            else:
                print("[Baseline_report][WARN] auto-rebuild requested but no RAW batch files matched --batch_glob.")

    # RL reference: optionally append RL rows (as a strategy) and/or align coverage
    rl_df = None
    if getattr(args, "rl_reference", None):
        if Path(args.rl_reference).exists():
            if str(args.rl_reference).lower().endswith(".parquet"):
                rl_df = pd.read_parquet(args.rl_reference)
            else:
                rl_df = pd.read_csv(args.rl_reference)
            print(f"[Baseline_report] loaded RL reference rows={len(rl_df)}")

            if bool(getattr(args, "include_rl", False)):
                rl_rows = ingest_rl_results(rl_df)
                if not rl_rows.empty:
                    # Append RL as its own strategy (dynamic+roll)
                    df = pd.concat([df, rl_rows], ignore_index=True)
                    print(f"[Baseline_report] appended RL rows={len(rl_rows)} total_rows={len(df)}")

            # Align everything (including RL) to RL coverage for apples-to-apples
            if bool(getattr(args, "align_to_rl_reference", False)):
                df = align_with_rl_reference(df, rl_df)
            else:
                print("[Baseline_report] align_to_rl_reference disabled; keeping full baseline coverage (RL appended if requested).")

            # Re-enrich after RL append/alignment so RL rows inherit scenario metadata
            # such as oracle_series and BEST/WORST labels from scenarios/<ASSET>/*.parquet.
            df = enrich_meta(df)
        else:
            print(f"[Baseline_report][WARN] RL reference not found: {args.rl_reference}")

    print(f"[Baseline_report] loaded files={len(files)} rows={len(df)}")

    df = enrich_meta(df)
    # window_id is a label (e.g., 'full', 'WF_train...'); keep it as string.
    if "window_id" in df.columns:
        df["window_id"] = df["window_id"].fillna("full").astype(str)
    df["scenario_kind"] = df["scenario_kind"].fillna("unknown").astype(str)

    # (4) Schema enforcement — two-step:
    #   Step A (soft): normalize common names/dtypes early to avoid merge/join surprises.
    #   Step B (after ensure_report_schema): enforce again so RL-mapped columns also follow contract.
    try:
        df, rep = validate_and_cast_results(df, strict=False, allow_empty=False)
        print_schema_report(rep)
    except Exception as e:
        print(f"[Baseline_report][WARN] results_schema validation failed (continuing): {e}")

    # Ensure core report schema exists even for RL-only inputs
    try:
        df = ensure_report_schema(df)
    except Exception as e:
        # Defer hard failure to compute_group_metrics, but log here for clarity.
        print(f"[Baseline_report][WARN] schema normalization note: {e}")

    # Second pass schema normalization (after mapping RL fields into report schema)
    try:
        df, _ = validate_and_cast_results(df, strict=False, allow_empty=False)
    except Exception as e:
        print(f"[Baseline_report][WARN] post-normalization schema check failed (continuing): {e}")

    # (7) Optional strict intersection at RAW row level (granular fairness)
    # If enabled, we restrict df itself (not only aggregated table) to scenario_id intersection
    # within each (exposure_id, scenario_kind) across strategies.
    if bool(getattr(args, "strict_intersection", False)):
        before = len(df)
        df = _restrict_to_common_scenarios_for_agg(df)
        print(f"[Baseline_report] strict_intersection enabled; rows: {before} -> {len(df)}")

    if getattr(args, "only_opec", False):
        df = df[df["exposure_id"].astype(str) == "OPEC_BASKET"].copy()
        print(f"[Baseline_report] restricted to OPEC_BASKET rows={len(df)}")

    if "scenario_kind" in df.columns:
        vc = df["scenario_kind"].value_counts(dropna=False).to_dict()
        print(f"[Baseline_report] scenario_kind counts: {vc}")

    # filter assets if requested
    if args.assets:
        wanted = set(args.assets)
        df = df[df["exposure_id"].astype(str).isin(wanted)].copy()
        print(f"[Baseline_report] filtered assets={sorted(wanted)} rows={len(df)}")

    metrics, metrics_agg = compute_group_metrics(df, spec)

    print(f"[Baseline_report] computed metrics rows={len(metrics)}")
    print(f"[Baseline_report] strategies found: {sorted(metrics['strategy'].unique())}")
    kinds = metrics["scenario_kind"].fillna("unknown").astype(str).unique().tolist()
    kinds_sorted = sorted(set(kinds), key=lambda x: str(x))
    print(f"[Baseline_report] scenario_kinds found: {kinds_sorted}")

    if metrics.empty:
        print("[Baseline_report][ERROR] No metrics computed. Check input alignment or filters.")
        return 1

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="report_pngs_"))
    try:
        out_paths: List[Path] = []
        assets_list = list(df["exposure_id"].dropna().unique())
        total_assets = len(assets_list)

        for idx, (asset, df_asset) in enumerate(df.groupby("exposure_id", dropna=False), 1):
            if getattr(args, "show_progress", False):
                print(f"[Baseline_report] processing asset {idx}/{total_assets}: {asset}")
            asset = str(asset)
            if asset == "UNKNOWN":
                continue
            out_dir = out_root / asset
            p = build_excel_for_asset(asset, df_asset, metrics, metrics_agg, out_dir, spec, temp_dir)
            out_paths.append(p)
            print(f"[Baseline_report] wrote {p}")

        # write a global metrics file as CSV for quick checks
        global_csv = out_root / "metrics_all_assets.csv"
        metrics.to_csv(global_csv, index=False)
        print(f"[Baseline_report] wrote {global_csv}")
        global_csv_agg = out_root / "metrics_all_assets_agg.csv"
        metrics_agg.to_csv(global_csv_agg, index=False)
        print(f"[Baseline_report] wrote {global_csv_agg}")

        print(f"[Baseline_report] done {ts_utc()}")
        return 0
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())