"""results_schema.py

A lightweight schema contract for all generated hedging results.

Why this exists:
- Prevent silent breakage when different strategies write slightly different column names/types.
- Provide a single, reproducible contract so Baseline_report and RL outputs can be compared fairly.

Design principles:
- Enforce a SMALL set of required columns.
- Normalize dtypes deterministically.
- Keep extra columns (do not drop) to preserve traceability.

This module is intentionally dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Canonical column names
# ----------------------------

# Columns required for any per-episode / per-scenario result row.
REQUIRED_RESULT_COLS: Tuple[str, ...] = (
    "exposure_id",
    "scenario_id",
    "scenario_kind",
    "strategy",
    "split",  # train/val/test (or NA for baselines)
    "window_id",  # walk-forward window label/id (e.g., WF_train..., full)
    # economics
    "net_pnl_total",
    "spot_pnl_total",
    "fut_pnl_total",
    "cost_trade_total",
    "cost_roll_total",
    "turnover_contracts",
    "mdd_equity",
)

# Optional but strongly recommended for oracle_all disambiguation.
OPTIONAL_KEY_COLS: Tuple[str, ...] = (
    "scenario_record_id",  # unique row id inside oracle_all (preferred)
    "scenario_id_with_label",  # alternative composite id
    "tag",  # ORACLE_BEST / ORACLE_WORST / FEASIBLE ...
)


# ----------------------------
# Dtype targets
# ----------------------------

DTYPE_TARGETS: Dict[str, str] = {
    # ids
    "exposure_id": "string",
    "scenario_id": "string",
    "scenario_kind": "string",
    "strategy": "string",
    "split": "string",
    "window_id": "string",
    "scenario_record_id": "string",
    "scenario_id_with_label": "string",
    "tag": "string",
    # economics
    "net_pnl_total": "float64",
    "spot_pnl_total": "float64",
    "fut_pnl_total": "float64",
    "cost_trade_total": "float64",
    "cost_roll_total": "float64",
    "turnover_contracts": "float64",
    "mdd_equity": "float64",
}


# ----------------------------
# Aliases (backward compatibility)
# ----------------------------

# If upstream code writes old column names, map them to the canonical ones.
ALIASES: Dict[str, str] = {
    # ids
    "exposure": "exposure_id",
    "scenario": "scenario_id",
    "kind": "scenario_kind",
    "scenario_type": "scenario_kind",
    "model": "strategy",
    "wf_window": "window_id",
    # economics
    "pnl_net_total": "net_pnl_total",
    "pnl_total": "net_pnl_total",
    "spot_pnl": "spot_pnl_total",
    "fut_pnl": "fut_pnl_total",
    "cost_trade": "cost_trade_total",
    "cost_roll": "cost_roll_total",
    "turnover": "turnover_contracts",
    "mdd": "mdd_equity",
}


@dataclass(frozen=True)
class SchemaReport:
    ok: bool
    missing_required: List[str]
    renamed: Dict[str, str]
    casted: List[str]
    notes: List[str]


def _rename_aliases(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rename_map: Dict[str, str] = {}
    for c in list(df.columns):
        if c in ALIASES and ALIASES[c] not in df.columns:
            rename_map[c] = ALIASES[c]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, rename_map


def _ensure_required(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    missing = [c for c in required if c not in df.columns]
    return missing


def _safe_to_float(s: pd.Series) -> pd.Series:
    # robust conversion: strings -> numeric; keep NaN where impossible
    return pd.to_numeric(s, errors="coerce")


def validate_and_cast_results(
    df: pd.DataFrame,
    *,
    require_cols: Iterable[str] = REQUIRED_RESULT_COLS,
    allow_empty: bool = False,
    strict: bool = True,
) -> Tuple[pd.DataFrame, SchemaReport]:
    """Validate a results DataFrame and normalize its schema.

    Parameters
    ----------
    df:
        Input results.
    require_cols:
        Required canonical columns.
    allow_empty:
        If False, empty df is treated as failure.
    strict:
        If True, missing required columns raises ValueError.

    Returns
    -------
    (df2, report)
    """

    notes: List[str] = []
    if df is None:
        raise ValueError("df is None")

    if (not allow_empty) and len(df) == 0:
        msg = "results df is empty"
        if strict:
            raise ValueError(msg)
        return df.copy(), SchemaReport(False, list(require_cols), {}, [], [msg])

    df2, renamed = _rename_aliases(df.copy())

    missing = _ensure_required(df2, require_cols)
    if missing:
        msg = f"Missing required columns: {missing}"
        if strict:
            raise ValueError(msg)
        return df2, SchemaReport(False, missing, renamed, [], [msg])

    casted: List[str] = []

    # Fill defaults for common fields
    if "window_id" not in df2.columns:
        df2["window_id"] = "full"
        casted.append("window_id(default)")
    else:
        # Ensure missing window_id becomes a stable label
        df2["window_id"] = df2["window_id"].astype("string").fillna("full")
        casted.append("window_id")

    if "split" not in df2.columns:
        df2["split"] = "NA"
        casted.append("split(default)")

    # Cast known columns
    for col, dtype in DTYPE_TARGETS.items():
        if col not in df2.columns:
            continue
        try:
            if dtype == "float64":
                df2[col] = _safe_to_float(df2[col]).astype("float64")
            else:
                df2[col] = df2[col].astype(dtype)
            casted.append(col)
        except Exception as e:
            notes.append(f"cast_failed:{col}:{e}")
            if strict:
                raise

    # Canonicalize strings (strip)
    for c in ["exposure_id", "scenario_id", "scenario_kind", "strategy", "split", "window_id"]:
        if c in df2.columns:
            df2[c] = df2[c].astype("string").str.strip()

    # Sanity checks
    # 1) net pnl must approximately equal spot + fut - costs if those are provided
    econ_cols = ["net_pnl_total", "spot_pnl_total", "fut_pnl_total", "cost_trade_total", "cost_roll_total"]
    if all(c in df2.columns for c in econ_cols):
        recon = df2["spot_pnl_total"] + df2["fut_pnl_total"] - (df2["cost_trade_total"] + df2["cost_roll_total"])
        diff = (df2["net_pnl_total"] - recon).abs()
        # tolerate tiny numeric drift; flag big drift
        bad = diff > 1e-6
        if bad.any():
            notes.append(f"econ_identity_violations={int(bad.sum())}")

    # 2) oracle_all disambiguation recommendation
    if ("oracle_all" in set(df2["scenario_kind"].dropna().astype(str))) and (
        ("scenario_record_id" not in df2.columns) and ("scenario_id_with_label" not in df2.columns)
    ):
        notes.append("oracle_all_missing_disambiguator: recommend adding scenario_record_id or scenario_id_with_label")

    rep = SchemaReport(True, [], renamed, casted, notes)
    return df2, rep


def schema_signature(df: pd.DataFrame) -> Dict[str, str]:
    """Return a simple {col: dtype} signature for debugging reproducibility."""
    return {c: str(df[c].dtype) for c in df.columns}


def print_schema_report(rep: SchemaReport) -> None:
    """Human-friendly summary."""
    status = "OK" if rep.ok else "FAIL"
    print(f"[schema] {status}")
    if rep.renamed:
        print(f"  renamed: {rep.renamed}")
    if rep.missing_required:
        print(f"  missing_required: {rep.missing_required}")
    if rep.casted:
        print(f"  casted: {sorted(set(rep.casted))}")
    if rep.notes:
        print("  notes:")
        for n in rep.notes:
            print(f"   - {n}")