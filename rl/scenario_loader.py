"""Scenario loader for RL.

This module loads scenario datasets (baseline/oracle_universe/oracle_all) stored as parquet
and converts start/end dates to indices in the precomputed universe cache (dates_int).

Why we need this
----------------
- Scenarios store dates as calendar dates (start_date/end_date).
- RL env uses fast NumPy arrays indexed by an integer time index.
- Precompute caches store `dates_int` as int64 = days since Unix epoch (np.datetime64[D]).

Outputs
-------
A list of Python dicts (one per scenario) ready to pass into `OilHedgingDailyEnv`.

Design goals
------------
- Reproducible sampling (seed)
- No look-ahead (we only map dates)
- Fast (vectorized mapping, optional dict mapping)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------

def _to_day_int(x: Any) -> int:
    """Convert a date-like object to int days since epoch (np.datetime64[D] -> int64)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("date is NaN/None")
    if isinstance(x, (np.datetime64,)):
        return int(x.astype("datetime64[D]").astype("int64"))
    # pandas Timestamp
    if isinstance(x, pd.Timestamp):
        return int(x.to_datetime64().astype("datetime64[D]").astype("int64"))
    # string
    return int(np.datetime64(str(x), "D").astype("int64"))


def _build_date_index(dates_int: np.ndarray, use_dict: bool = False) -> Any:
    """Return a mapping object to convert day_int -> position index."""
    d = np.asarray(dates_int, dtype=np.int64)
    if d.ndim != 1:
        raise ValueError("dates_int must be 1d")
    if use_dict:
        return {int(v): i for i, v in enumerate(d)}
    # otherwise we will use searchsorted
    return d


def _map_day_ints_to_pos(
    day_ints: np.ndarray,
    dates_index: Any,
) -> np.ndarray:
    """Map day_ints to indices in dates_int; return -1 when not found."""
    x = np.asarray(day_ints, dtype=np.int64)
    if isinstance(dates_index, dict):
        out = np.empty(len(x), dtype=np.int64)
        for i, v in enumerate(x):
            out[i] = int(dates_index.get(int(v), -1))
        return out

    d = np.asarray(dates_index, dtype=np.int64)
    pos = np.searchsorted(d, x)
    # mark out of bounds
    bad = (pos < 0) | (pos >= len(d))
    pos[bad] = -1
    ok = pos != -1
    # verify exact matches
    ok_idx = np.where(ok)[0]
    if ok_idx.size:
        mismatch = d[pos[ok_idx]] != x[ok_idx]
        if np.any(mismatch):
            pos[ok_idx[mismatch]] = -1
    return pos.astype(np.int64)


# -------------------------
# Public API
# -------------------------

@dataclass(frozen=True)
class ScenarioLoaderConfig:
    require_ok_coverage: bool = True
    allow_shortened: bool = False
    max_scenarios: Optional[int] = None
    seed: int = 2026
    use_dict_index: bool = False


def load_scenarios_from_parquet(
    parquet_path: str,
    *,
    dates_int: np.ndarray,
    exposure_id: Optional[str] = None,
    window_start_day_int: Optional[int] = None,
    window_end_day_int: Optional[int] = None,
    require_full_containment: bool = True,
    cfg: Optional[ScenarioLoaderConfig] = None,
) -> List[Dict[str, Any]]:
    """Load scenarios and map them to precompute indices.

    Parameters
    ----------
    parquet_path:
        Path to scenario parquet.
    dates_int:
        1d int64 array from precompute (days since epoch).
    exposure_id:
        If provided, keep only scenarios with matching exposure_id column.
    window_start_day_int/window_end_day_int:
        Keep scenarios within the provided day-int range.
    require_full_containment:
        If True, require start>=window_start and end<=window_end; if False, allow partial overlap.
    cfg:
        Loader config (coverage filtering, sampling, seed).

    Returns
    -------
    List[dict]
        Each dict contains at least:
        - scenario_id, exposure_id
        - start_idx, end_idx (inclusive indices)
        - start_day_int, end_day_int
        - volume_bbl
        - horizon_days_target, horizon_days_realized
        Plus a small subset of metadata helpful for analysis.
    """

    if cfg is None:
        cfg = ScenarioLoaderConfig()

    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_parquet(p)

    # Required identifier field
    if "scenario_id" not in df.columns:
        raise ValueError("Scenario parquet must contain scenario_id")

    # Optional exposure filter
    if exposure_id is not None and "exposure_id" in df.columns:
        df = df.loc[df["exposure_id"].astype(str) == str(exposure_id)]

    # Coverage policy
    if cfg.require_ok_coverage and "data_coverage_flag" in df.columns:
        if cfg.allow_shortened:
            df = df.loc[df["data_coverage_flag"].isin(["OK", "SHORTENED"])]
        else:
            df = df.loc[df["data_coverage_flag"] == "OK"]

    if "start_date" not in df.columns or "end_date" not in df.columns:
        raise ValueError("Scenario parquet must contain start_date and end_date")

    # Convert dates to day-int
    # Use vectorized pandas -> numpy conversion for speed
    sd = pd.to_datetime(df["start_date"], errors="coerce").dt.floor("D")
    ed = pd.to_datetime(df["end_date"], errors="coerce").dt.floor("D")
    if sd.isna().any() or ed.isna().any():
        # fall back row-wise to keep precise error locations
        bad_rows = df.loc[sd.isna() | ed.isna(), ["scenario_id", "start_date", "end_date"]].head(5)
        raise ValueError(f"Found unparsable dates in scenarios (showing up to 5):\n{bad_rows}")

    start_day_int = sd.to_numpy(dtype="datetime64[D]").astype("int64")
    end_day_int = ed.to_numpy(dtype="datetime64[D]").astype("int64")

    # Strict walk-forward safety: filter by full containment in the requested window
    if window_start_day_int is not None or window_end_day_int is not None:
        ws = window_start_day_int
        we = window_end_day_int
        if ws is None:
            ws = int(np.min(dates_int))
        if we is None:
            we = int(np.max(dates_int))
        if require_full_containment:
            keep = (start_day_int >= int(ws)) & (end_day_int <= int(we))
        else:
            # allow any overlap
            keep = (end_day_int >= int(ws)) & (start_day_int <= int(we))
        keep_idx = np.where(keep)[0]
        df = df.iloc[keep_idx].copy()
        start_day_int = start_day_int[keep_idx]
        end_day_int = end_day_int[keep_idx]

    if df.empty:
        return []

    # Map to universe indices
    dates_index = _build_date_index(dates_int, use_dict=cfg.use_dict_index)
    start_idx = _map_day_ints_to_pos(start_day_int, dates_index)
    end_idx = _map_day_ints_to_pos(end_day_int, dates_index)

    # Keep only in-range and valid order
    ok = (start_idx >= 0) & (end_idx >= 0) & (end_idx >= start_idx)
    ok_idx = np.where(ok)[0]
    df = df.iloc[ok_idx].copy()
    start_idx = start_idx[ok_idx]
    end_idx = end_idx[ok_idx]
    start_day_int = start_day_int[ok_idx]
    end_day_int = end_day_int[ok_idx]

    if df.empty:
        return []

    # Optional sampling for speed
    if cfg.max_scenarios is not None and len(df) > int(cfg.max_scenarios):
        df = df.sample(n=int(cfg.max_scenarios), random_state=int(cfg.seed)).copy()

        # Recompute mapping for the sampled df (still fast at this size)
        sd2 = pd.to_datetime(df["start_date"], errors="coerce").dt.floor("D")
        ed2 = pd.to_datetime(df["end_date"], errors="coerce").dt.floor("D")
        start_day_int = sd2.to_numpy(dtype="datetime64[D]").astype("int64")
        end_day_int = ed2.to_numpy(dtype="datetime64[D]").astype("int64")
        start_idx = _map_day_ints_to_pos(start_day_int, dates_index)
        end_idx = _map_day_ints_to_pos(end_day_int, dates_index)

        # Keep only in-range and valid order (sampling can reintroduce -1 via remap)
        ok2 = (start_idx >= 0) & (end_idx >= 0) & (end_idx >= start_idx)
        ok2_idx = np.where(ok2)[0]
        df = df.iloc[ok2_idx].copy()
        start_idx = start_idx[ok2_idx]
        end_idx = end_idx[ok2_idx]
        start_day_int = start_day_int[ok2_idx]
        end_day_int = end_day_int[ok2_idx]

    # Required numeric fields
    if "volume_bbl" not in df.columns:
        raise ValueError("Scenario parquet must contain volume_bbl")

    # Build output dicts
    out: List[Dict[str, Any]] = []
    cols_optional = [
        "scenario_kind",
        "tag",
        "company_id",
        "company_size",
        "oracle_pool",
        "oracle_candidate",
        "oracle_series",
        "scenario_record_id",
        "scenario_id_with_label",
        "horizon_days_target",
        "horizon_days_realized",
    ]

    records = df.to_dict("records")
    for i, rowd in enumerate(records):
        sc: Dict[str, Any] = {
            "scenario_id": str(rowd.get("scenario_id")),
            "exposure_id": str(rowd.get("exposure_id")) if "exposure_id" in rowd else (str(exposure_id) if exposure_id else ""),
            "scenario_kind": str(rowd.get("scenario_kind", "")),
            "start_idx": int(start_idx[i]),
            "end_idx": int(end_idx[i]),
            "start_day_int": int(start_day_int[i]),
            "end_day_int": int(end_day_int[i]),
            "volume_bbl": float(rowd.get("volume_bbl")),
        }
        for c in cols_optional:
            if c in rowd:
                v = rowd.get(c)
                if c in ("horizon_days_target", "horizon_days_realized") and v is not None:
                    try:
                        sc[c] = int(v)
                    except Exception:
                        sc[c] = v
                else:
                    sc[c] = v
        out.append(sc)

    return out


# -------------------------
# CLI
# -------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load parquet scenarios and map to precompute indices")
    p.add_argument("--cache", required=True, help="Path to precompute_*.npz")
    p.add_argument("--parquet", required=True, help="Path to scenarios parquet")
    p.add_argument("--exposure_id", default=None, help="Filter exposure_id")
    p.add_argument("--max_scenarios", type=int, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--allow_shortened", action="store_true")
    p.add_argument("--no_require_ok", action="store_true")
    p.add_argument("--use_dict_index", action="store_true", help="Use dict mapping instead of searchsorted")
    p.add_argument("--window_start", type=str, default=None, help="Window start date (YYYY-MM-DD) or day-int")
    p.add_argument("--window_end", type=str, default=None, help="Window end date (YYYY-MM-DD) or day-int")
    p.add_argument("--allow_partial_overlap", action="store_true", help="Allow scenarios that overlap the window (not fully contained)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    z = np.load(args.cache, allow_pickle=True)
    dates_int = z["dates_int"].astype(np.int64)

    def _parse_window_arg(x: Optional[str]) -> Optional[int]:
        if x is None:
            return None
        xs = str(x).strip()
        if xs == "":
            return None
        # if it looks like an int, treat as day-int
        if xs.lstrip("-").isdigit():
            return int(xs)
        # otherwise parse as date
        return _to_day_int(xs)

    window_start_day_int = _parse_window_arg(args.window_start)
    window_end_day_int = _parse_window_arg(args.window_end)

    cfg = ScenarioLoaderConfig(
        require_ok_coverage=not bool(args.no_require_ok),
        allow_shortened=bool(args.allow_shortened),
        max_scenarios=(int(args.max_scenarios) if args.max_scenarios is not None else None),
        seed=int(args.seed),
        use_dict_index=bool(args.use_dict_index),
    )

    sc = load_scenarios_from_parquet(
        args.parquet,
        dates_int=dates_int,
        exposure_id=args.exposure_id,
        window_start_day_int=window_start_day_int,
        window_end_day_int=window_end_day_int,
        require_full_containment=not bool(args.allow_partial_overlap),
        cfg=cfg,
    )

    print(f"loaded_scenarios={len(sc)}")
    if sc:
        # print a tiny sample
        for x in sc[:3]:
            print(x)


if __name__ == "__main__":
    main()