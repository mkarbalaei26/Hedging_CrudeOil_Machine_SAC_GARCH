"""LSTM input builder for the crude-oil hedging project.

This script builds three separate, LSTM-ready feature packs (WTI/Brent/OPEC) from MasterData.

Design choices (reproducible):
- Work at DAILY frequency using the provided MasterData (already aligned).
- For correlation screening we used MODE=returns; therefore, here we transform price-like columns
  into log-returns *in-place while keeping the original column names* to match feature-selector outputs.
- Volatility/ratio columns (sigma_*, rho_*, h_*) are kept in levels.
- Sparse columns are kept (user requirement). We add per-column missing indicators (isnan_*).
- Output is written as Parquet per exposure:
    out_dir/WTI_SPOT_lstm_features.parquet
    out_dir/BRENT_SPOT_lstm_features.parquet
    out_dir/OPEC_BASKET_lstm_features.parquet
- Parquet output uses snappy compression and Date is written as datetime64[ms] for DuckDB compatibility.
- Also writes a __duckdb.parquet variant with sanitized column names and conservative Parquet encoding for problematic viewers.

Optional:
- If you pass selected-features CSV (from Feature_selector), we will subset to those features
  (plus mandatory columns). If omitted, we keep a sane default set.

IMPORTANT:
- Teacher label is DCC-120 hedge ratio. Column names can differ; we search a set of candidates.

Run examples:
  python LSTMInput.py --master MasterData_enriched_garch_clean.parquet --out_dir data/lstm

  python LSTMInput.py --master MasterData.parquet --out_dir data/lstm \
      --selected_wti results/feat_sel_wti/selected_features_returns_spearman.csv \
      --selected_brent results/feat_sel_brent/selected_features_returns_spearman.csv \
      --selected_opec results/feat_sel_opec/selected_features_returns_spearman.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Additional imports for ultra-compatible Parquet output
import re
import pyarrow as pa
import pyarrow.parquet as pq


# ------------------------------
# Column universe (from user)
# ------------------------------
ALL_COLUMNS = [
    "Date",
    "DTB3",
    "CL1",
    "CL2",
    "CL3",
    "CL4",
    "OPEC",
    "WTI",
    "Brent",
    "DXY",
    "SP500_ENERGY",
    "VIX",
    "OVX",
    "EUI_eq_w",
    "GPRD",
    "GPRD_ACT",
    "GPRD_THREAT",
    "GPRD_MA30",
    "GPRD_MA7",
    "USEPUINDXD",
    "TB3MS",
    "OPU_index",
    "GEPUCURRENT",
    "EPUTRADE",
    "WUI_Global_(simple_average)",
    "WUI_Global_(GDP_weighted_average)",
    "WUI_Advanced_economies",
    "WUI_Emerging_economies",
    "WUI_Low-income_economies",
    "WUI_Africa",
    "WUI_Asia_and_the_Pacific",
    "WUI_Europe",
    "WUI_Middle_East_and_Central_Asia",
    "WUI_Western_Hemisphere",
    "WTUI_global_gdpw",
    "CL1_sigma_garch",
    "WTI_sigma_spot_garch",
    "WTI_rho_30",
    "WTI_h_ccc_proxy_30",
    "WTI_rho_60",
    "WTI_h_ccc_proxy_60",
    "WTI_rho_120",
    "WTI_h_ccc_proxy_120",
    "WTI_rho_252",
    "WTI_h_ccc_proxy_252",
    "BRENT_sigma_spot_garch",
    "BRENT_rho_30",
    "BRENT_h_ccc_proxy_30",
    "BRENT_rho_60",
    "BRENT_h_ccc_proxy_60",
    "BRENT_rho_120",
    "BRENT_h_ccc_proxy_120",
    "BRENT_rho_252",
    "BRENT_h_ccc_proxy_252",
    "OPEC_sigma_spot_garch",
    "OPEC_rho_30",
    "OPEC_h_ccc_proxy_30",
    "OPEC_rho_60",
    "OPEC_h_ccc_proxy_60",
    "OPEC_rho_120",
    "OPEC_h_ccc_proxy_120",
    "OPEC_rho_252",
    "OPEC_h_ccc_proxy_252",
]


@dataclass(frozen=True)
class ExposureSpec:
    exposure_id: str
    spot_col: str
    teacher_prefix: str  # used to find label candidates


EXPOSURES: Dict[str, ExposureSpec] = {
    "WTI_SPOT": ExposureSpec("WTI_SPOT", "WTI", "WTI"),
    "BRENT_SPOT": ExposureSpec("BRENT_SPOT", "Brent", "BRENT"),
    "OPEC_BASKET": ExposureSpec("OPEC_BASKET", "OPEC", "OPEC"),
}


# Columns we consider "price-like" and convert to log-returns (in-place)
PRICE_LIKE_BASE = {
    "CL1",
    "CL2",
    "CL3",
    "CL4",
    "WTI",
    "Brent",
    "OPEC",
    "DXY",
    "SP500_ENERGY",
    "VIX",
    "OVX",
    "DTB3",
    "TB3MS",
    "OPU_index",
    "GEPUCURRENT",
    "EPUTRADE",
    "USEPUINDXD",
    "EUI_eq_w",
    "GPRD",
    "GPRD_ACT",
    "GPRD_THREAT",
    "GPRD_MA30",
    "GPRD_MA7",
    "WUI_Global_(simple_average)",
    "WUI_Global_(GDP_weighted_average)",
    "WUI_Advanced_economies",
    "WUI_Emerging_economies",
    "WUI_Low-income_economies",
    "WUI_Africa",
    "WUI_Asia_and_the_Pacific",
    "WUI_Europe",
    "WUI_Middle_East_and_Central_Asia",
    "WUI_Western_Hemisphere",
    "WTUI_global_gdpw",
}


def _read_df(path: str) -> pd.DataFrame:
    p = str(path)
    if p.lower().endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, low_memory=False)
    if "Date" not in df.columns:
        raise ValueError("Input master must include 'Date' column")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def _find_teacher_col(df: pd.DataFrame, prefix: str, window: int = 120) -> str:
    """Find the DCC-120 hedge ratio label column.

    We try multiple candidates because naming may vary across implementations.
    """
    candidates = [
        f"{prefix}_h_dcc_{window}",
        f"{prefix}_h_dcc_proxy_{window}",
        f"{prefix}_h_dcc_{window}_proxy",
        f"{prefix}_h_dccgarch_{window}",
        # fallback to CCC proxy if user forgot to compute DCC (not preferred)
        f"{prefix}_h_ccc_proxy_{window}",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find teacher label for prefix={prefix}. Tried: {candidates}. "
        f"Available cols sample: {list(df.columns)[:25]}"
    )


def _log_return_inplace(df: pd.DataFrame, col: str) -> None:
    """Replace a column with its log-return, keeping the same column name."""
    x = pd.to_numeric(df[col], errors="coerce")
    # log return: ln(P_t/P_{t-1})
    # If values can be <=0 (rare for indices; oil went negative historically),
    # we use signed log(1+pct) fallback for non-positive.
    prev = x.shift(1)
    out = np.full(len(df), np.nan, dtype=float)

    pos = (x > 0) & (prev > 0)
    pos_arr = pos.to_numpy(dtype=bool)
    # use numpy arrays consistently
    x_arr = x.to_numpy(dtype=float)
    prev_arr = prev.to_numpy(dtype=float)

    # primary: log-return for strictly-positive pairs
    with np.errstate(divide="ignore", invalid="ignore"):
        out[pos_arr] = np.log(x_arr[pos_arr] / prev_arr[pos_arr])

    # fallback for non-positive values (handles negative oil prices)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_arr = (x_arr - prev_arr) / prev_arr

    # when prev==0 -> inf/NaN; keep NaN
    safe = np.isfinite(pct_arr)
    fb_mask = (~pos_arr) & safe
    if np.any(fb_mask):
        out[fb_mask] = np.sign(pct_arr[fb_mask]) * np.log1p(np.abs(pct_arr[fb_mask]))

    df[col] = out


def _make_spreads_and_basis(df: pd.DataFrame) -> None:
    """Add economically meaningful, low-dimensional features (optional but recommended)."""
    # Term structure spreads (levels), then convert to returns-like by differencing
    for a, b, name in [("CL1", "CL2", "spread_CL1_CL2"), ("CL1", "CL4", "spread_CL1_CL4")]:
        if a in df.columns and b in df.columns:
            pa = pd.to_numeric(df[a], errors="coerce")
            pb = pd.to_numeric(df[b], errors="coerce")
            sp = pa - pb
            df[name] = sp
            # change of spread
            df[f"d_{name}"] = sp.diff()

    # Basis (levels) for each exposure vs CL1; keep both level and change
    if "CL1" in df.columns:
        cl1 = pd.to_numeric(df["CL1"], errors="coerce")
        for spot_col in ["WTI", "Brent", "OPEC"]:
            if spot_col in df.columns:
                s = pd.to_numeric(df[spot_col], errors="coerce")
                bs = s - cl1
                df[f"basis_{spot_col}"] = bs
                df[f"d_basis_{spot_col}"] = bs.diff()


def _read_selected_features(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p)
    if "feature" not in df.columns:
        raise ValueError("Selected-features file must have a 'feature' column")
    feats = [str(x).strip() for x in df["feature"].tolist() if isinstance(x, str) and str(x).strip()]
    # de-dup while preserving order
    feats = list(dict.fromkeys(feats))
    return feats



def _add_missing_indicators(df: pd.DataFrame, cols: List[str]) -> List[str]:
    added = []
    for c in cols:
        if c == "Date":
            continue
        ind = f"isnan_{c}"
        if ind in df.columns:
            continue
        df[ind] = df[c].isna().astype(np.int32)
        added.append(ind)
    return added


# --- Ultra-compatible Parquet helpers ---
def _sanitize_column_names(cols: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Sanitize column names for maximum SQL/viewer compatibility.

    Rules:
      - keep 'Date' as-is
      - replace any non [A-Za-z0-9_] with '_'
      - collapse multiple '_' and strip
      - ensure uniqueness by suffixing _2, _3, ...
    """
    mapping: Dict[str, str] = {}
    used: Dict[str, int] = {}
    out: List[str] = []

    for c in cols:
        if c == "Date":
            new = "Date"
        else:
            new = re.sub(r"[^0-9A-Za-z_]", "_", str(c))
            new = re.sub(r"_+", "_", new).strip("_")
            if not new:
                new = "col"

        base = new
        if base in used:
            used[base] += 1
            new = f"{base}_{used[base]}"
        else:
            used[base] = 1

        mapping[c] = new
        out.append(new)

    return out, mapping


def _write_duckdb_compatible_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a very compatible Parquet file for DuckDB-based viewers."""
    # Coerce Date to ms timestamp
    ddf = df.copy()
    ddf["Date"] = pd.to_datetime(ddf["Date"], errors="coerce").astype("datetime64[ms]")

    # Convert pandas -> arrow with safe timestamp coercion
    table = pa.Table.from_pandas(ddf, preserve_index=False)

    pq.write_table(
        table,
        where=str(path),
        compression=None,
        use_dictionary=False,
        coerce_timestamps="ms",
        allow_truncated_timestamps=True,
        data_page_version="1.0",
    )


def build_exposure_pack(
    master: pd.DataFrame,
    spec: ExposureSpec,
    out_path: Path,
    *,
    selected_features: Optional[List[str]],
    make_spreads: bool,
    keep_sparse: bool,
) -> None:
    df = master.copy(deep=False)

    # Ensure required base columns
    required = ["Date", "CL1", "CL2", "CL4", spec.spot_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {spec.exposure_id}: {missing}")

    # Add engineered features BEFORE return-transform (so spreads/basis can be differenced)
    if make_spreads:
        _make_spreads_and_basis(df)

    # Teacher label
    teacher_col = _find_teacher_col(df, spec.teacher_prefix, window=120)
    df = df[["Date"] + [c for c in df.columns if c != "Date"]]  # stable order

    # Convert price-like columns into log-returns in-place to match feature selector (mode=returns)
    for c in PRICE_LIKE_BASE:
        if c in df.columns:
            _log_return_inplace(df, c)

    # For engineered spreads/basis we already created d_* columns; keep as is.
    # (They are not in PRICE_LIKE_BASE, so they won't be overwritten.)

    # Decide final column set
    mandatory = ["Date", teacher_col]

    if selected_features is None:
        # Default: keep broad but not insane set
        keep = [
            # core market
            "CL1",
            "CL2",
            "CL4",
            spec.spot_col,
            "DXY",
            "VIX",
            "OVX",
            # volatility signals
            "CL1_sigma_garch",
            f"{spec.teacher_prefix}_sigma_spot_garch" if f"{spec.teacher_prefix}_sigma_spot_garch" in df.columns else None,
            # proxies
            f"{spec.teacher_prefix}_h_ccc_proxy_60" if f"{spec.teacher_prefix}_h_ccc_proxy_60" in df.columns else None,
            f"{spec.teacher_prefix}_rho_60" if f"{spec.teacher_prefix}_rho_60" in df.columns else None,
            # macro risk
            "GPRD_MA30",
            "USEPUINDXD",
            "WUI_Global_(GDP_weighted_average)",
        ]
        keep = [c for c in keep if c is not None and c in df.columns]
    else:
        # Subset to feature-selector output (names match because we kept same column names)
        keep = [c for c in selected_features if c in df.columns]

        # If user selected features are too strict and drop the spot itself, force-keep basics
        for c in [spec.spot_col, "CL1", "CL2", "CL4", "DXY", "VIX"]:
            if c in df.columns and c not in keep:
                keep.append(c)

        # Add engineered spreads/basis if present (they reduce multicollinearity vs multiple CL levels)
        for c in [
            "d_spread_CL1_CL2",
            "d_spread_CL1_CL4",
            f"d_basis_{spec.spot_col}",
        ]:
            if c in df.columns and c not in keep:
                keep.append(c)

    # Keep sparse columns if explicitly requested: do not drop due to NaNs here.
    final_cols = mandatory + keep
    final_cols = list(dict.fromkeys([c for c in final_cols if c in df.columns]))

    out_df = df[final_cols].copy()

    # Rename label consistently
    out_df = out_df.rename(columns={teacher_col: "y_h_dcc_120"})

    # Optional: keep sparse columns with masks
    if keep_sparse:
        added_masks = _add_missing_indicators(out_df, [c for c in out_df.columns if c not in ("Date", "y_h_dcc_120")])
        # Keep masks at the end
        # (Already appended by assignment)
        _ = added_masks

    # Write outputs
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure Date is ms-resolution timestamp (some readers choke on ns)
    out_df["Date"] = pd.to_datetime(out_df["Date"], errors="coerce").astype("datetime64[ms]")

    # Coerce any accidental object columns to numeric where possible (keeps NaN)
    for c in out_df.columns:
        if c == "Date":
            continue
        if out_df[c].dtype == object:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

    # 1) Standard Parquet (fast, compact)
    out_df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    # 2) Ultra-compatible Parquet for DuckDB-based viewers (sanitized names, uncompressed, no dictionary)
    safe_cols, mapping = _sanitize_column_names(list(out_df.columns))
    out_safe = out_df.copy()
    out_safe.columns = safe_cols

    safe_path = out_path.with_name(out_path.stem + "__duckdb" + out_path.suffix)
    _write_duckdb_compatible_parquet(out_safe, safe_path)

    # mapping csv
    map_path = out_path.with_name(out_path.stem + "__duckdb_colmap.csv")
    pd.DataFrame({"original": list(mapping.keys()), "duckdb_safe": list(mapping.values())}).to_csv(map_path, index=False)

    print(
        f"[LSTMInput] Wrote {spec.exposure_id} pack: {out_path} (snappy) | {safe_path} (duckdb) "
        f"rows={len(out_df):,}, cols={len(out_df.columns)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, required=True, help="MasterData parquet/csv")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for LSTM packs")

    ap.add_argument("--selected_wti", type=str, default="", help="selected_features CSV for WTI")
    ap.add_argument("--selected_brent", type=str, default="", help="selected_features CSV for Brent")
    ap.add_argument("--selected_opec", type=str, default="", help="selected_features CSV for OPEC")

    ap.add_argument("--no_spreads", action="store_true", help="Disable adding spreads/basis engineered features")
    ap.add_argument("--no_missing_masks", action="store_true", help="Do not add isnan_* indicator columns")

    args = ap.parse_args()

    master = _read_df(args.master)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_wti = _read_selected_features(args.selected_wti) if args.selected_wti else None
    sel_brent = _read_selected_features(args.selected_brent) if args.selected_brent else None
    sel_opec = _read_selected_features(args.selected_opec) if args.selected_opec else None

    build_exposure_pack(
        master,
        EXPOSURES["WTI_SPOT"],
        out_dir / "WTI_SPOT_lstm_features.parquet",
        selected_features=sel_wti,
        make_spreads=(not args.no_spreads),
        keep_sparse=(not args.no_missing_masks),
    )
    build_exposure_pack(
        master,
        EXPOSURES["BRENT_SPOT"],
        out_dir / "BRENT_SPOT_lstm_features.parquet",
        selected_features=sel_brent,
        make_spreads=(not args.no_spreads),
        keep_sparse=(not args.no_missing_masks),
    )
    build_exposure_pack(
        master,
        EXPOSURES["OPEC_BASKET"],
        out_dir / "OPEC_BASKET_lstm_features.parquet",
        selected_features=sel_opec,
        make_spreads=(not args.no_spreads),
        keep_sparse=(not args.no_missing_masks),
    )


if __name__ == "__main__":
    main()
