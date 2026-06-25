

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline validation gate (thesis-grade).

This script is intentionally strict. If a check fails, it exits with non-zero
status to prevent running training/backtests on inconsistent universes.

Core responsibilities:
1) Validate canonical scenario universe (schema, uniqueness, date logic).
2) Validate that generated scenario parquet files are internally consistent.
3) (Optional) Validate that a results table aligns to the scenario universe
   and that intersection logic is safe for reporting.

Usage examples:
  python validate_pipeline.py --scenarios_root scenarios --exposure OPEC_BASKET
  python validate_pipeline.py --scenarios_root scenarios --exposure OPEC_BASKET \
      --results results/hedge_results_master.parquet

Notes:
- This validator assumes Scenario_Generator writes:
  baseline.parquet, companies.parquet, oracle_universe.parquet, oracle_all.parquet
  under scenarios/<EXPOSURE>/
- If you later add scenario_universe.parquet, this script will use it if present.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ALLOWED_SCENARIO_KINDS = {
    "baseline",
    "company",
    "oracle_universe",
    "oracle_all",
}

# Minimal columns required to uniquely define a *trade scenario* row.
TRADE_KEY_COLS = [
    "scenario_id",
    "exposure_id",
    "scenario_kind",
    "start_date",
    "end_date",
    "horizon_days_target",
    "horizon_days_realized",
    "volume_bbl",
]

# Results schema (minimum to do alignment checks)
RESULTS_KEY_COLS = [
    "scenario_id",
    "exposure_id",
    "scenario_kind",
    "strategy",
    "split",
]


@dataclass
class CheckResult:
    ok: bool
    name: str
    detail: str


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _coerce_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.normalize()
    return out


def _fail(msg: str) -> None:
    print(f"\n❌ VALIDATION FAILED: {msg}", file=sys.stderr)
    sys.exit(2)


def _warn(msg: str) -> None:
    print(f"⚠️  {msg}")



def _ok(msg: str) -> None:
    print(f"✅ {msg}")


# --- Exposure folder helpers ---
from typing import List

def _list_exposures(scenarios_root: Path) -> List[str]:
    if not scenarios_root.exists():
        return []
    out = []
    for p in scenarios_root.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            out.append(p.name)
    return sorted(out)


def _normalize_exposure_name(exposure: str) -> List[str]:
    """Return candidate folder names for an exposure string."""
    e = str(exposure)
    cands = [e]
    if "-" in e:
        cands.append(e.replace("-", "_"))
    if "_" in e:
        cands.append(e.replace("_", "-"))
    # also try upper/lower variants (folder names are typically uppercase)
    cands.append(e.upper())
    cands.append(e.lower())
    # de-duplicate preserving order
    seen = set()
    uniq = []
    for x in cands:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _check_required_cols(df: pd.DataFrame, required: List[str], name: str) -> CheckResult:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return CheckResult(False, f"{name}:required_cols", f"missing={missing}")
    return CheckResult(True, f"{name}:required_cols", "ok")


def _check_nonnull(df: pd.DataFrame, cols: List[str], name: str) -> CheckResult:
    bad = []
    for c in cols:
        if c in df.columns:
            n = int(df[c].isna().sum())
            if n > 0:
                bad.append((c, n))
    if bad:
        return CheckResult(False, f"{name}:nonnull", f"null_counts={bad}")
    return CheckResult(True, f"{name}:nonnull", "ok")


def _check_unique(df: pd.DataFrame, cols: List[str], name: str) -> CheckResult:
    if any(c not in df.columns for c in cols):
        return CheckResult(False, f"{name}:unique", f"missing_key_cols={cols}")
    dup_mask = df.duplicated(subset=cols, keep=False)
    if dup_mask.any():
        sample = df.loc[dup_mask, cols].head(10).to_dict("records")
        return CheckResult(False, f"{name}:unique", f"duplicates_on={cols}; sample={sample}")
    return CheckResult(True, f"{name}:unique", "ok")


def _check_allowed_kinds(df: pd.DataFrame, col: str, name: str) -> CheckResult:
    if col not in df.columns:
        return CheckResult(False, f"{name}:kinds", f"missing={col}")
    kinds = set(map(str, df[col].dropna().unique().tolist()))
    bad = sorted(list(kinds - ALLOWED_SCENARIO_KINDS))
    if bad:
        return CheckResult(False, f"{name}:kinds", f"unexpected={bad}; allowed={sorted(ALLOWED_SCENARIO_KINDS)}")
    return CheckResult(True, f"{name}:kinds", "ok")


def _check_dates_logic(df: pd.DataFrame, name: str) -> CheckResult:
    for c in ["start_date", "end_date"]:
        if c not in df.columns:
            return CheckResult(False, f"{name}:dates", f"missing={c}")
    bad_parse = int(df["start_date"].isna().sum() + df["end_date"].isna().sum())
    if bad_parse > 0:
        return CheckResult(False, f"{name}:dates", f"unparseable_dates={bad_parse}")

    # end >= start
    bad_order = df["end_date"] < df["start_date"]
    if bad_order.any():
        sample = df.loc[bad_order, ["scenario_id", "start_date", "end_date"]].head(10).to_dict("records")
        return CheckResult(False, f"{name}:dates", f"end_before_start sample={sample}")

    # horizon consistency (realized must be <= target, and roughly matches date span)
    if "horizon_days_target" in df.columns and "horizon_days_realized" in df.columns:
        tgt = pd.to_numeric(df["horizon_days_target"], errors="coerce")
        real = pd.to_numeric(df["horizon_days_realized"], errors="coerce")
        bad_h = (real > tgt)
        if bad_h.any():
            sample = df.loc[bad_h, ["scenario_id", "horizon_days_target", "horizon_days_realized"]].head(10).to_dict("records")
            return CheckResult(False, f"{name}:horizon", f"realized_gt_target sample={sample}")

        # approximate span in business days: allow mismatch because of tradable-day shortening
        span_days = (df["end_date"] - df["start_date"]).dt.days
        # span_days should be >=0 always; already checked.
        # sanity: realized horizon should not exceed calendar span + 5 (buffer)
        too_big = real > (span_days + 5)
        if too_big.any():
            sample = df.loc[too_big, ["scenario_id", "horizon_days_realized", "start_date", "end_date"]].head(10).to_dict("records")
            return CheckResult(False, f"{name}:horizon", f"realized_gt_calendar_span sample={sample}")

    return CheckResult(True, f"{name}:dates", "ok")


def _check_oracle_all_labels(df: pd.DataFrame, name: str) -> List[CheckResult]:
    results: List[CheckResult] = []
    # oracle_all must have oracle_series and should have scenario_record_id.
    results.append(_check_required_cols(df, ["oracle_series"], name))
    if "oracle_series" in df.columns:
        # no empty oracle_series
        empt = df["oracle_series"].astype(str).fillna("").str.strip() == ""
        if empt.any():
            sample = df.loc[empt, ["scenario_id", "oracle_series"]].head(10).to_dict("records")
            results.append(CheckResult(False, f"{name}:oracle_series_nonempty", f"empty_labels sample={sample}"))
        else:
            results.append(CheckResult(True, f"{name}:oracle_series_nonempty", "ok"))

    if "scenario_record_id" in df.columns:
        results.append(_check_unique(df, ["scenario_record_id"], name))
        # also enforce uniqueness of (scenario_id, oracle_series)
        results.append(_check_unique(df, ["scenario_id", "oracle_series"], name + ":pair"))
    else:
        results.append(CheckResult(False, f"{name}:scenario_record_id", "missing scenario_record_id"))

    return results


def load_scenarios(scenarios_root: Path, exposure: str) -> Dict[str, pd.DataFrame]:
    # Resolve exposure folder name (tolerate '-' vs '_' etc.)
    exp_dir = None
    for cand in _normalize_exposure_name(exposure):
        p = scenarios_root / cand
        if p.exists() and p.is_dir():
            exp_dir = p
            break
    if exp_dir is None:
        avail = _list_exposures(scenarios_root)
        _fail(f"Exposure folder not found for '{exposure}'. Available exposures: {avail}")

    # Preferred canonical universe file (if you add it later)
    uni_path = exp_dir / "scenario_universe.parquet"
    if uni_path.exists():
        uni = _read_any(uni_path)
        return {"scenario_universe": uni}

    # Otherwise use generator outputs
    paths = {
        "baseline": exp_dir / "baseline.parquet",
        "company": exp_dir / "companies.parquet",
        "oracle_universe": exp_dir / "oracle_universe.parquet",
        "oracle_all": exp_dir / "oracle_all.parquet",
    }

    out: Dict[str, pd.DataFrame] = {}
    for k, p in paths.items():
        if not p.exists():
            _fail(f"Missing required scenario file: {p}")
        out[k] = _read_any(p)

    return out


def validate_scenarios(frames: Dict[str, pd.DataFrame], exposure: str) -> None:
    checks: List[CheckResult] = []

    for name, df0 in frames.items():
        df = df0.copy()
        # normalize date columns
        df = _coerce_dates(df, ["start_date", "end_date"])

        checks.append(_check_required_cols(df, TRADE_KEY_COLS, name))
        checks.append(_check_nonnull(df, ["scenario_id", "exposure_id", "scenario_kind"], name))
        checks.append(_check_allowed_kinds(df, "scenario_kind", name))
        checks.append(_check_dates_logic(df, name))

        # exposure must match
        if "exposure_id" in df.columns:
            bad = df["exposure_id"].astype(str) != str(exposure)
            if bad.any():
                sample = df.loc[bad, ["scenario_id", "exposure_id"]].head(10).to_dict("records")
                checks.append(CheckResult(False, f"{name}:exposure_match", f"expected={exposure}; sample={sample}"))
            else:
                checks.append(CheckResult(True, f"{name}:exposure_match", "ok"))

        # scenario_id must be unique within each file (trade universe)
        # Exception: oracle_all can have repeated scenario_id due to labels.
        if name == "oracle_all":
            checks.extend(_check_oracle_all_labels(df, name))
        else:
            checks.append(_check_unique(df, ["scenario_id"], name))

    # Print check summary
    failed = [c for c in checks if not c.ok]
    for c in checks:
        (print if c.ok else print)(f"{'OK ' if c.ok else 'FAIL'} | {c.name} | {c.detail}")

    if failed:
        # show a compact failure list at end
        msg = "\n".join([f"- {c.name}: {c.detail}" for c in failed[:20]])
        _fail(f"Scenario validation failed with {len(failed)} failing checks. First failures:\n{msg}")

    _ok("Scenario files passed strict validation.")

    # Additional cross-file checks (if we have generator outputs)
    if "scenario_universe" not in frames:
        # Joinability: baseline/company/oracle_universe scenario_id must not overlap (they represent different universes)
        # This is conservative. If you intentionally allow overlap, remove/relax this check.
        keys = [k for k in ["baseline", "company", "oracle_universe"] if k in frames]
        id_sets = {k: set(frames[k]["scenario_id"].astype(str).tolist()) for k in keys}
        overlaps: List[Tuple[str, str, int]] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                inter = id_sets[a].intersection(id_sets[b])
                if len(inter) > 0:
                    overlaps.append((a, b, len(inter)))
        if overlaps:
            # Not failing by default, but warn loudly
            _warn(f"Scenario_id overlaps detected across universes: {overlaps}. If this is unintended, investigate Scenario_Generator.")
        else:
            _ok("No scenario_id overlaps across baseline/company/oracle_universe (expected).")

        # oracle_all label coverage: each oracle_all row should match a scenario_id from oracle_universe or company/baseline depending on your design.
        # Current generator builds oracle_all from oracle_universe selection, so check subset.
        if "oracle_all" in frames and "oracle_universe" in frames:
            all_ids = set(frames["oracle_all"]["scenario_id"].astype(str).tolist())
            uni_ids = set(frames["oracle_universe"]["scenario_id"].astype(str).tolist())
            missing = list(all_ids - uni_ids)
            if missing:
                _warn(f"oracle_all contains scenario_id not present in oracle_universe (count={len(missing)}). First: {missing[:5]}")
            else:
                _ok("oracle_all scenario_id are subset of oracle_universe (expected).")


def validate_results(results_path: Path, frames: Dict[str, pd.DataFrame]) -> None:
    if not results_path.exists():
        _warn(f"Results file not found, skipping results validation: {results_path}")
        return

    res = _read_any(results_path)
    if len(res) == 0:
        _warn("Results table is empty; skipping alignment checks.")
        return

    res = _coerce_dates(res, ["start_date", "end_date"])  # if present

    # Minimum schema
    cr = _check_required_cols(res, RESULTS_KEY_COLS, "results")
    if not cr.ok:
        _fail(cr.detail)

    # scenario_kind allowed
    ck = _check_allowed_kinds(res, "scenario_kind", "results")
    if not ck.ok:
        _fail(ck.detail)

    # Build canonical scenario set
    if "scenario_universe" in frames:
        uni = frames["scenario_universe"].copy()
        uni = _coerce_dates(uni, ["start_date", "end_date"])
        scenario_set = set(uni["scenario_id"].astype(str).tolist())
    else:
        # Use baseline+company+oracle_universe as the "universe" of tradable scenarios
        scenario_set = set()
        for k in ["baseline", "company", "oracle_universe"]:
            if k in frames:
                scenario_set |= set(frames[k]["scenario_id"].astype(str).tolist())

    res_ids = set(res["scenario_id"].astype(str).tolist())
    missing_from_universe = list(res_ids - scenario_set)
    if missing_from_universe:
        _fail(f"Results contain scenario_id not in scenario universe. Count={len(missing_from_universe)} First={missing_from_universe[:10]}")
    _ok("All results scenario_id exist in scenario universe.")

    # Intersection safety for reporting
    by_strategy = res.groupby("strategy")["scenario_id"].nunique().sort_values(ascending=False)
    _ok(f"Results unique scenarios by strategy: {by_strategy.to_dict()}")

    # NoHedge variance sanity (if present)
    # We accept either net_pnl_total or equity_return; choose what's available.
    if "strategy" in res.columns:
        nohedge = res[res["strategy"].astype(str).str.upper().isin(["NOHEDGE", "NO_HEDGE", "NOHEDGE_BASELINE"])]
        if len(nohedge) > 0:
            if "net_pnl_total" in nohedge.columns:
                v = float(np.nanvar(pd.to_numeric(nohedge["net_pnl_total"], errors="coerce").to_numpy(dtype=float)))
                if not np.isfinite(v) or v <= 0:
                    _fail(f"Var(NoHedge) is not positive/finite: {v}")
                _ok(f"Var(NoHedge) OK: {v:.6g}")
            elif "equity_return" in nohedge.columns:
                v = float(np.nanvar(pd.to_numeric(nohedge["equity_return"], errors="coerce").to_numpy(dtype=float)))
                if not np.isfinite(v) or v <= 0:
                    _fail(f"Var(NoHedge equity_return) is not positive/finite: {v}")
                _ok(f"Var(NoHedge equity_return) OK: {v:.6g}")
            else:
                _warn("NoHedge found but neither net_pnl_total nor equity_return exists; skipping variance sanity.")
        else:
            _warn("NoHedge rows not found in results; variance sanity skipped.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict pipeline validation gate")
    ap.add_argument("--scenarios_root", type=str, default="scenarios", help="Root folder containing exposure subfolders")
    ap.add_argument("--exposure", type=str, required=True, help="Exposure folder name, e.g., OPEC_BASKET")
    ap.add_argument("--results", type=str, default="", help="Optional results parquet/csv to validate alignment")
    args = ap.parse_args()

    scenarios_root = Path(args.scenarios_root)
    exposure = args.exposure

    if str(exposure).upper() == "ALL":
        exposures = _list_exposures(scenarios_root)
        if not exposures:
            _fail(f"No exposure folders found under: {scenarios_root}")
        _ok(f"Validating ALL exposures: {exposures}")
        for exp in exposures:
            print(f"\n=== EXPOSURE: {exp} ===")
            frames = load_scenarios(scenarios_root, exp)
            validate_scenarios(frames, exp)
            if args.results:
                validate_results(Path(args.results), frames)
        _ok("All validations passed for ALL exposures.")
        return

    frames = load_scenarios(scenarios_root, exposure)
    validate_scenarios(frames, exposure)

    if args.results:
        validate_results(Path(args.results), frames)

    _ok("All validations passed.")


if __name__ == "__main__":
    main()