

# -*- coding: utf-8 -*-
"""Scenario_Generator.py

گام 4) Physical Trade Scenario Generator

طبق تصمیم‌های قفل‌شده در گفتگو:
1) سناریوهای بدون برچسب شرکت (unlabeled) فقط یک نوع baseline هستند:
   - BASIC_BASELINE_20D_1MMB : افق دقیقاً 20 روز تقویمی، حجم دقیقاً 1,000,000 بشکه
   - از همه تاریخ‌های شروع معتبر تولید می‌شود (به ازای هر exposure universe جدا)

2) سناریوهای شرکتی (company scenarios):
   - هر ترید منحصر به‌فرد است (scenario_id یکتا)
   - 3 دسته شرکت (Small/Medium/Large) و هر کدام 300 شرکت
   - شدت فعالیت ماهانه (تعداد ترید در ماه) قفل‌شده:
       Small  : 1..3
       Medium : 4..8
       Large  : 9..15
   - برای هر ترید: volume ~ U(1e6,2e6)، horizon ~ DiscreteUniform(20..40) (روزهای تقویمی)
   - در هر شرکت، در هر ماه، start_date ها یکتا هستند (حداکثر 1 ترید در هر روز شروع برای همان شرکت)

3) سناریوهای oracle:
   - در این گام oracle universe ساخته می‌شود و با استفاده از متریک‌های قطعی فیزیکی (pnl_physical و ...) 4 سری oracle استخراج می‌شود.
   - محاسبه بهترین/بدترین ex-post و حل OR ماهانه (3..10 trade و max 1 per start day)
     در گام‌های بعدی (پس از ساخت PnL کامل) انجام می‌شود.

قانون کمبود دیتا در افق (LOCKED):
- اگر در افق هدف داده کافی نباشد، افق کوتاه می‌شود تا آخرین روز معتبر
- اگر افق تحقق‌یافته < min_realized_horizon (=15) → سناریو REJECTED

نکته:
- این ماژول hedge ratio یا PnL آتی/فیزیکی محاسبه نمی‌کند.
- ورودی داده باید از DataAdapter بیاید تا قوانین missing/zero/availability enforce شود.

خروجی‌ها (قطعی):
- scenarios/<EXPOSURE>/baseline.parquet (+ baseline.npz)
- scenarios/<EXPOSURE>/companies.parquet (+ companies.npz)
- scenarios/<EXPOSURE>/oracle_universe.parquet (+ oracle_universe.npz)
- scenarios/<EXPOSURE>/oracle_all.parquet (+ oracle_all.npz)
- scenarios/<EXPOSURE>/report.xlsx

نیازمندی‌ها:
  pip install pandas numpy pyyaml
  (اختیاری برای parquet) pip install pyarrow

"""

from __future__ import annotations

import os
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from data_adapter import DataAdapter


# -----------------------------
# Config (locked defaults for step4)
# -----------------------------

BASELINE_NAME = "BASIC_BASELINE_20D_1MMB"
BASELINE_H_DAYS = 20
BASELINE_Q_BBL = 1_000_000

MIN_REALIZED_HORIZON = 15  # LOCKED by user

N_COMPANIES_PER_SIZE = 300
COMPANY_SIZES = ["SMALL", "MEDIUM", "LARGE"]

TRADES_PER_MONTH_RANGES = {
    "SMALL": (1, 3),
    "MEDIUM": (4, 8),
    "LARGE": (9, 15),
}

HORIZON_MIN = 20
HORIZON_MAX = 40
VOLUME_MIN = 500_000
VOLUME_MAX = 1_500_000

# -----------------------------
# Oracle design (LOCKED for this iteration)
# Step 4 will build an "oracle universe" of candidate trades, then SELECT 4 oracle series from it:
#   (1) EXTREME_DAILY_BEST   : best trade per day (max score)
#   (2) EXTREME_DAILY_WORST  : worst trade per day (min score)
#   (3) FEASIBLE_MONTHLY_BEST  : 3..4 trades per month, max 1 per start-day (OR), maximize score
#   (4) FEASIBLE_MONTHLY_WORST : same constraints, minimize score
#
# IMPORTANT: In this step we compute a *provisional* score for oracle selection:
#   score_physical = (Spot_end - Spot_start) * volume_bbl
# This is ONLY for building oracle labels. In later steps we can replace score with full hedged/cost-adjusted PnL.
ORACLE_UNIVERSE_K_PER_DAY = 10
ORACLE_MONTHLY_TRADES_BOUNDS = (3, 10)
ORACLE_POOL_TAG = "ORACLE_UNIVERSE"

ORACLE_SCORE_COL = "score_physical"

# Physical metric columns that must be present in all scenario outputs
PHYS_COLS = [
    "spot_start",
    "spot_end",
    "spot_high",
    "spot_low",
    "pnl_physical",
    "return_simple",
    "mdd_abs_usd",
    "mdd_pct",
    "mae_abs_usd",
    "missed_upside_usd",
    "score_physical",
]


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass
class ScenarioSpec:
    scenario_id: str
    exposure_id: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    horizon_days_target: int
    horizon_days_realized: int
    volume_bbl: int
    scenario_kind: str  # baseline / company / oracle_universe / oracle_all
    tag: str
    oracle_series: Optional[str] = None  # EXTREME_DAILY_BEST/WORST, FEASIBLE_MONTHLY_BEST/WORST (only in oracle_all)
    scenario_record_id: Optional[str] = None  # unique row id when labels create multiple rows per scenario_id
    company_id: Optional[str] = None
    company_size: Optional[str] = None
    oracle_pool: Optional[str] = None  # "EXTREME_DAILY" or "FEASIBLE_MONTHLY"
    oracle_candidate: int = 0
    spot_start: Optional[float] = None
    spot_end: Optional[float] = None
    score_physical: Optional[float] = None
    spot_high: Optional[float] = None
    spot_low: Optional[float] = None
    pnl_physical: Optional[float] = None
    return_simple: Optional[float] = None
    mdd_abs_usd: Optional[float] = None
    mdd_pct: Optional[float] = None
    mae_abs_usd: Optional[float] = None
    missed_upside_usd: Optional[float] = None
    data_coverage_flag: str = "OK"  # OK / SHORTENED / REJECTED
# -----------------------------
# Core utilities
# -----------------------------


def _stable_hash_id(key: str, n: int = 16) -> str:
    """Deterministic, reproducible id from a string key."""
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return h[: int(n)]


def _make_scenario_id(
    exposure_id: str,
    scenario_kind: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    volume_bbl: int,
    company_id: Optional[str] = None,
    oracle_pool: Optional[str] = None,
    oracle_candidate: int = 0,
) -> str:
    """Stable id for a *trade scenario* (does NOT include oracle_series label).

    Rationale: scenario_id must be reproducible across runs and usable for joins
    between baselines/RL/oracle selection. Oracle labels are stored separately.
    """
    s = pd.Timestamp(start_date).normalize().date().isoformat()
    e = pd.Timestamp(end_date).normalize().date().isoformat()
    cid = "" if company_id is None else str(company_id)
    op = "" if oracle_pool is None else str(oracle_pool)
    key = f"{exposure_id}|{scenario_kind}|{s}|{e}|{int(volume_bbl)}|{cid}|{op}|{int(oracle_candidate)}"
    return _stable_hash_id(key, n=20)


def _make_scenario_record_id(scenario_id: str, oracle_series: Optional[str]) -> str:
    """Stable id for a *scenario record*.

    For oracle_all, the same scenario_id (trade) can appear with different oracle_series
    labels. This record id disambiguates rows while keeping scenario_id joinable.
    """
    lab = "" if oracle_series is None else str(oracle_series)
    return _stable_hash_id(f"{scenario_id}|{lab}", n=24)


# -----------------------------
# Core utilities
# -----------------------------


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _month_key(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.year:04d}-{ts.month:02d}"


def _safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return int(np.round(float(x)))


def _build_date_index(df: pd.DataFrame, date_col: str) -> Tuple[pd.DatetimeIndex, set]:
    dates = pd.DatetimeIndex(pd.to_datetime(df[date_col]).dt.normalize())
    return dates, set(dates)


def _shorten_end_date(
    start_date: pd.Timestamp,
    target_h: int,
    dates_set: set,
    max_backtrack: int = 60,
) -> Tuple[pd.Timestamp, int, str]:
    """Apply shortening rule.

    We interpret horizon as calendar days. Target end = start + target_h days.
    If end date not present in available dates, we backtrack day-by-day until found.

    Returns: (end_date_realized, realized_h, flag)
      flag: OK if realized_h==target_h and end exists
            SHORTENED if shortened but >= MIN_REALIZED_HORIZON
            REJECTED if shortened below MIN_REALIZED_HORIZON
    """
    s = pd.Timestamp(start_date).normalize()
    end_target = (s + pd.Timedelta(days=int(target_h))).normalize()

    if end_target in dates_set:
        return end_target, int(target_h), "OK"

    # backtrack
    end_real = end_target
    for _ in range(max_backtrack):
        end_real = (end_real - pd.Timedelta(days=1)).normalize()
        if end_real in dates_set:
            realized_h = int((end_real - s).days)
            if realized_h < MIN_REALIZED_HORIZON:
                return end_real, realized_h, "REJECTED"
            return end_real, realized_h, "SHORTENED"

    # could not find
    realized_h = int((end_real - s).days)
    return end_real, realized_h, "REJECTED"


def _validate_start(df: pd.DataFrame, date_col: str, spot_col: str, cl1_col: str, start_date: pd.Timestamp) -> bool:
    """Start date must exist and have spot+CL1 present."""
    s = pd.Timestamp(start_date).normalize()
    mask = pd.to_datetime(df[date_col]).dt.normalize() == s
    if not mask.any():
        return False
    row = df.loc[mask].iloc[0]
    return pd.notna(row.get(spot_col)) and pd.notna(row.get(cl1_col))



def _get_spot_on_date(df: pd.DataFrame, date_col: str, spot_col: str, d: pd.Timestamp) -> Optional[float]:
    dd = pd.Timestamp(d).normalize()
    m = pd.to_datetime(df[date_col]).dt.normalize() == dd
    if not m.any():
        return None
    v = df.loc[m, spot_col].iloc[0]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except Exception:
        return None


# -----------------------------
# Physical metrics helper
# -----------------------------

def _compute_physical_metrics_from_window(spot_win: np.ndarray, entry: float, exit_: float, Q: float) -> Dict[str, float]:
    """Compute deterministic physical-trade metrics over the holding window.

    Notes on edge cases:
    - Returns use simple return (exit-entry)/entry. If entry == 0, return_simple is NaN.
    - Drawdown is computed on spot level relative to running peak (absolute and percent).
      If running peak <= 0 at any point, percent drawdown is set to NaN for those points.
    """
    spot_high = float(np.nanmax(spot_win))
    spot_low = float(np.nanmin(spot_win))

    pnl_physical = float((exit_ - entry) * Q)

    # simple return (robust with negative prices; log-return not used here)
    if entry == 0 or np.isnan(entry):
        return_simple = float("nan")
    else:
        return_simple = float((exit_ - entry) / entry)

    # Drawdown on spot level relative to running peak
    run_peak = np.maximum.accumulate(spot_win)
    dd_abs = spot_win - run_peak  # <= 0
    mdd_abs_usd = float(np.nanmin(dd_abs) * Q)

    # percent drawdown only meaningful when run_peak > 0
    dd_pct = np.full_like(spot_win, np.nan, dtype=float)
    mask = run_peak > 0
    dd_pct[mask] = (spot_win[mask] - run_peak[mask]) / run_peak[mask]
    mdd_pct = float(np.nanmin(dd_pct)) if np.isfinite(dd_pct).any() else float("nan")

    # Max adverse excursion (absolute) from entry
    mae_abs_usd = float((spot_low - entry) * Q)

    # Missed upside relative to best in-window exit
    best_possible = float((spot_high - entry) * Q)
    missed_upside_usd = float(best_possible - pnl_physical)

    return {
        "spot_high": spot_high,
        "spot_low": spot_low,
        "pnl_physical": pnl_physical,
        "return_simple": return_simple,
        "mdd_abs_usd": mdd_abs_usd,
        "mdd_pct": mdd_pct,
        "mae_abs_usd": mae_abs_usd,
        "missed_upside_usd": missed_upside_usd,
    }



def _ensure_solver():
    """Return (solver_name, module) or (None, None) if no MILP solver available."""
    try:
        import pulp  # type: ignore
        return "pulp", pulp
    except Exception:
        return None, None


# -----------------------------
# File existence/read helpers
# -----------------------------

def _path_exists_any(path_parquet: str) -> Optional[str]:
    """Return the existing path among parquet/csv variants, else None."""
    if os.path.exists(path_parquet):
        return path_parquet
    if path_parquet.endswith(".parquet"):
        path_csv = path_parquet.replace(".parquet", ".csv")
        if os.path.exists(path_csv):
            return path_csv
    return None



def _read_df_any(path_parquet: str) -> pd.DataFrame:
    """Read parquet if exists else csv fallback; raises if neither exists."""
    p = _path_exists_any(path_parquet)
    if p is None:
        raise FileNotFoundError(path_parquet)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


# Helper to enrich oracle dfs with required physical columns from universe
def _enrich_from_universe(sel: pd.DataFrame, uni: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Ensure selected oracle df has all desired cols by left-joining from universe on scenario_id."""
    if sel is None or len(sel) == 0:
        return sel
    if uni is None or len(uni) == 0:
        return sel
    if "scenario_id" not in sel.columns or "scenario_id" not in uni.columns:
        return sel

    missing = [c for c in cols if c not in sel.columns]
    # Also treat as missing if column exists but all NA (happens with cached older outputs)
    allna = [c for c in cols if c in sel.columns and sel[c].isna().all()]
    need = sorted(set(missing + allna))
    if not need:
        return sel

    base_cols = ["scenario_id"] + [c for c in need if c in uni.columns]
    if len(base_cols) <= 1:
        return sel

    add = uni[base_cols].drop_duplicates(subset=["scenario_id"]).copy()
    out = sel.merge(add, on="scenario_id", how="left", suffixes=("", "_uni"))

    # If some columns already existed but were NA, prefer universe values
    for c in need:
        if c in out.columns and f"{c}_uni" in out.columns:
            out[c] = out[c].where(out[c].notna(), out[f"{c}_uni"])
            out.drop(columns=[f"{c}_uni"], inplace=True)

    return out


# -----------------------------
# Progress/logging helpers
# -----------------------------

def _now() -> float:
    return time.time()


def _fmt_s(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _pbar(it, desc: str, total: Optional[int] = None):
    """Progress bar wrapper (tqdm if available, else plain iterator)."""
    if tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total, dynamic_ncols=True)


def _log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------------
# Generators
# -----------------------------


class PhysicalScenarioGenerator:
    def __init__(
        self,
        adapter: DataAdapter,
        exposure_id: str,
        out_dir: str = "scenarios",
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.adapter = adapter
        self.exposure_id = exposure_id
        self.verbose = verbose
        self.out_dir = out_dir

        # seed priority: provided > config.project.seed_global > 2026
        cfg_seed = adapter.cfg.get("project", {}).get("seed_global", 2026)
        self.seed = int(seed if seed is not None else cfg_seed)
        self.rng = _make_rng(self.seed)
        _log(f"[ScenarioGen:{self.exposure_id}] Init | seed={self.seed} | out_dir='{self.out_dir}'")

        # universe dataframe (spot + CL1/CL2 + features if exist)
        self.df_uni = self.adapter.get_universe(exposure_id, include_features=True, feature_role="both")
        _log(f"[ScenarioGen:{self.exposure_id}] Universe rows: {len(self.df_uni)} | columns: {len(self.df_uni.columns)}")
        self.date_col = self.adapter.date_col

        exp = self.adapter._get_exposure(exposure_id)
        self.spot_col = exp["price_column"]
        self.cl1_col = self.adapter.cfg["assets"]["hedge_instrument"]["price_column_front"]

        # Precompute fast access structures for physical metrics
        tmp_dates = pd.to_datetime(self.df_uni[self.date_col]).dt.normalize().to_numpy()
        self._dates_arr = tmp_dates
        self._spot_arr = self.df_uni[self.spot_col].to_numpy(dtype=float)
        # map date -> first position (df_uni dates are expected unique after adapter filtering)
        self._date_to_pos: Dict[pd.Timestamp, int] = {}
        for i, d in enumerate(tmp_dates):
            ts = pd.Timestamp(d)
            if ts not in self._date_to_pos:
                self._date_to_pos[ts] = int(i)

        # date set for shortening rule
        self.dates, self.dates_set = _build_date_index(self.df_uni, self.date_col)
        _log(f"[ScenarioGen:{self.exposure_id}] Available dates: {len(self.dates)} | date range: {self.dates.min().date()} .. {self.dates.max().date()}")

        # Precompute normalized dates and valid start masks for speedups
        self._date_norm = pd.to_datetime(self.df_uni[self.date_col]).dt.normalize()
        self._valid_start_mask = self.df_uni[self.spot_col].notna() & self.df_uni[self.cl1_col].notna()
        self._valid_start_dates = pd.DatetimeIndex(self._date_norm[self._valid_start_mask].unique()).sort_values()
        # Precompute calendar-to-available-trading-date forward-fill map
        avail = pd.DatetimeIndex(self._dates_arr)
        cal = pd.date_range(avail.min(), avail.max(), freq="D")
        m = pd.Series(pd.NaT, index=cal, dtype="datetime64[ns]")
        m.loc[avail] = avail
        self._cal_to_trade = m.ffill()

        os.makedirs(self.out_dir, exist_ok=True)

    def _physical_metrics(self, start_date: pd.Timestamp, end_date: pd.Timestamp, volume_bbl: int) -> Dict[str, float]:
        s = pd.Timestamp(start_date).normalize()
        e = pd.Timestamp(end_date).normalize()
        if s not in self._date_to_pos or e not in self._date_to_pos:
            return {}
        i0 = self._date_to_pos[s]
        i1 = self._date_to_pos[e]
        if i1 < i0:
            return {}
        spot_win = self._spot_arr[i0 : i1 + 1]
        if spot_win.size == 0:
            return {}
        entry = float(spot_win[0])
        exit_ = float(spot_win[-1])
        Q = float(volume_bbl)
        out = {
            "spot_start": entry,
            "spot_end": exit_,
            "score_physical": float((exit_ - entry) * Q),
        }
        out.update(_compute_physical_metrics_from_window(spot_win, entry, exit_, Q))
        return out

    def _physical_metrics_core(self, start_date: pd.Timestamp, end_date: pd.Timestamp, volume_bbl: int) -> Dict[str, float]:
        s = pd.Timestamp(start_date).normalize()
        e = pd.Timestamp(end_date).normalize()
        if s not in self._date_to_pos or e not in self._date_to_pos:
            return {}
        i0 = self._date_to_pos[s]
        i1 = self._date_to_pos[e]
        if i1 < i0:
            return {}
        entry = float(self._spot_arr[i0])
        exit_ = float(self._spot_arr[i1])
        Q = float(volume_bbl)
        pnl_physical = float((exit_ - entry) * Q)
        if entry == 0 or np.isnan(entry):
            return_simple = float("nan")
        else:
            return_simple = float((exit_ - entry) / entry)
        return {
            "spot_start": entry,
            "spot_end": exit_,
            "pnl_physical": pnl_physical,
            "return_simple": return_simple,
            "score_physical": float((exit_ - entry) * Q),
        }

    def _shorten_end_date_fast(self, start_date: pd.Timestamp, target_h: int) -> Tuple[pd.Timestamp, int, str]:
        """Fast shortening rule using precomputed calendar->trading-date map."""
        s = pd.Timestamp(start_date).normalize()
        end_target = (s + pd.Timedelta(days=int(target_h))).normalize()
        if end_target not in self._cal_to_trade.index:
            # beyond available calendar span
            return end_target, int((end_target - s).days), "REJECTED"
        end_real = self._cal_to_trade.loc[end_target]
        if pd.isna(end_real):
            return end_target, int((end_target - s).days), "REJECTED"
        end_real = pd.Timestamp(end_real).normalize()
        realized_h = int((end_real - s).days)
        if realized_h == int(target_h):
            flag = "OK"
        else:
            flag = "SHORTENED"
        if realized_h < MIN_REALIZED_HORIZON:
            return end_real, realized_h, "REJECTED"
        return end_real, realized_h, flag

    def generate_baseline(self) -> pd.DataFrame:
        """Generate BASIC_BASELINE_20D_1MMB for all valid start dates."""
        t0 = _now()
        _log(f"[ScenarioGen:{self.exposure_id}] Stage 1/3: Generating BASELINE '{BASELINE_NAME}' | H={BASELINE_H_DAYS} | Q={BASELINE_Q_BBL:,}")
        rows: List[Dict] = []
        rejected = 0
        shortened = 0
        valid_starts = len(self._valid_start_dates)
        for s in _pbar(self._valid_start_dates, desc=f"Baseline {self.exposure_id}", total=valid_starts):
            end_date, h_real, flag = self._shorten_end_date_fast(s, BASELINE_H_DAYS)
            if flag == "SHORTENED":
                shortened += 1
            if flag == "REJECTED":
                rejected += 1
                continue

            pm = self._physical_metrics(s, end_date, BASELINE_Q_BBL)

            sc = ScenarioSpec(
                scenario_id=_make_scenario_id(
                    exposure_id=self.exposure_id,
                    scenario_kind="baseline",
                    start_date=s,
                    end_date=end_date,
                    volume_bbl=BASELINE_Q_BBL,
                ),
                exposure_id=self.exposure_id,
                start_date=s,
                end_date=end_date,
                horizon_days_target=BASELINE_H_DAYS,
                horizon_days_realized=h_real,
                volume_bbl=BASELINE_Q_BBL,
                scenario_kind="baseline",
                tag=BASELINE_NAME,
                company_id=None,
                company_size=None,
                spot_start=pm.get("spot_start"),
                spot_end=pm.get("spot_end"),
                score_physical=pm.get("score_physical"),
                spot_high=pm.get("spot_high"),
                spot_low=pm.get("spot_low"),
                pnl_physical=pm.get("pnl_physical"),
                return_simple=pm.get("return_simple"),
                mdd_abs_usd=pm.get("mdd_abs_usd"),
                mdd_pct=pm.get("mdd_pct"),
                mae_abs_usd=pm.get("mae_abs_usd"),
                missed_upside_usd=pm.get("missed_upside_usd"),
                data_coverage_flag=flag,
            )
            rows.append(sc.__dict__)

        df = pd.DataFrame(rows)
        if self.verbose:
            _log(
                f"[ScenarioGen:{self.exposure_id}] Baseline done | generated={len(df)} | valid_starts={valid_starts} | shortened={shortened} | rejected={rejected} | time={_fmt_s(_now()-t0)}"
            )
        return df

    def generate_company_scenarios(self) -> pd.DataFrame:
        """Generate company-tagged scenarios (large unique set)."""
        t0 = _now()
        _log(
            f"[ScenarioGen:{self.exposure_id}] Stage 2/3: Generating COMPANY scenarios | companies_per_size={N_COMPANIES_PER_SIZE} | monthly_ranges={TRADES_PER_MONTH_RANGES} | H~U[{HORIZON_MIN},{HORIZON_MAX}] | Q~U[{VOLUME_MIN:,},{VOLUME_MAX:,}]"
        )
        # Build companies table
        companies = []
        for size in COMPANY_SIZES:
            for i in range(N_COMPANIES_PER_SIZE):
                companies.append({
                    "company_id": f"{size}_{i:03d}",
                    "company_size": size,
                })
        companies_df = pd.DataFrame(companies)

        # Use precomputed _date_norm and _valid_start_mask and group valid dates by month
        date_norm = self._date_norm
        month_str = date_norm.dt.to_period("M").astype(str)
        df_valid = pd.DataFrame({"_date": date_norm[self._valid_start_mask].to_numpy(), "_month": month_str[self._valid_start_mask].to_numpy()})
        month_to_dates: Dict[str, np.ndarray] = {m: g["_date"].to_numpy() for m, g in df_valid.groupby("_month")}

        months = sorted(month_to_dates.keys())
        if not months:
            return pd.DataFrame()
        _log(f"[ScenarioGen:{self.exposure_id}] Company months available: {len(months)} | first={months[0]} | last={months[-1]}")

        rows: List[Dict] = []
        rejected = 0
        valid_starts = 0
        shortened = 0
        generated = 0
        total_companies = len(companies_df)
        total_months = len(months)
        total_iterations = total_companies * total_months
        iter_counter = 0

        for _, comp in _pbar(companies_df.iterrows(), desc=f"Companies {self.exposure_id}", total=total_companies):
            cid = comp["company_id"]
            csize = comp["company_size"]
            lo, hi = TRADES_PER_MONTH_RANGES[csize]

            for m in months:
                iter_counter += 1
                dates_m = month_to_dates[m]
                if len(dates_m) == 0:
                    continue

                n_trades = int(self.rng.integers(lo, hi + 1))
                # If not enough unique dates, cap
                n_trades = min(n_trades, len(dates_m))

                # choose unique start dates for this company-month
                chosen = self.rng.choice(dates_m, size=n_trades, replace=False)

                for s in chosen:
                    s = pd.Timestamp(s).normalize()
                    # sample H and Q
                    H = int(self.rng.integers(HORIZON_MIN, HORIZON_MAX + 1))
                    Q = int(self.rng.integers(VOLUME_MIN, VOLUME_MAX + 1))
                    valid_starts += 1

                    end_date, h_real, flag = self._shorten_end_date_fast(s, H)
                    if flag == "SHORTENED":
                        shortened += 1
                    if flag == "REJECTED":
                        rejected += 1
                        continue

                    pm_core = self._physical_metrics_core(s, end_date, Q)
                    # keep schema: fill other physical columns with np.nan
                    sc = ScenarioSpec(
                        scenario_id=_make_scenario_id(
                            exposure_id=self.exposure_id,
                            scenario_kind="company",
                            start_date=s,
                            end_date=end_date,
                            volume_bbl=Q,
                            company_id=cid,
                        ),
                        exposure_id=self.exposure_id,
                        start_date=s,
                        end_date=end_date,
                        horizon_days_target=H,
                        horizon_days_realized=h_real,
                        volume_bbl=Q,
                        scenario_kind="company",
                        tag="COMPANY_RANDOM",
                        company_id=cid,
                        company_size=csize,
                        spot_start=pm_core.get("spot_start"),
                        spot_end=pm_core.get("spot_end"),
                        score_physical=pm_core.get("score_physical"),
                        spot_high=np.nan,
                        spot_low=np.nan,
                        pnl_physical=pm_core.get("pnl_physical"),
                        return_simple=pm_core.get("return_simple"),
                        mdd_abs_usd=np.nan,
                        mdd_pct=np.nan,
                        mae_abs_usd=np.nan,
                        missed_upside_usd=np.nan,
                        data_coverage_flag=flag,
                    )
                    rows.append(sc.__dict__)
                    generated += 1

        out = pd.DataFrame(rows)
        if self.verbose:
            _log(
                f"[ScenarioGen:{self.exposure_id}] Company done | generated={len(out)} | attempts={valid_starts} | shortened={shortened} | rejected={rejected} | time={_fmt_s(_now()-t0)}"
            )
        return out


    def generate_oracle_universe(self) -> pd.DataFrame:
        """Build oracle candidate universe: for each valid start day generate K trades.

        Each candidate gets a provisional physical score:
          score_physical = (spot_end - spot_start) * volume_bbl

        This makes it possible to select 4 oracle series *now*.
        """
        t0 = _now()
        _log(
            f"[ScenarioGen:{self.exposure_id}] Stage 3a/3: Generating ORACLE UNIVERSE | K_per_day={ORACLE_UNIVERSE_K_PER_DAY} | H~U[{HORIZON_MIN},{HORIZON_MAX}] | Q~U[{VOLUME_MIN:,},{VOLUME_MAX:,}]"
        )

        valid_dates = self._valid_start_dates
        if len(valid_dates) == 0:
            _log(f"[ScenarioGen:{self.exposure_id}] Oracle universe: no valid start dates")
            return pd.DataFrame()

        rows: List[Dict] = []
        rejected = 0
        shortened = 0

        for s in _pbar(valid_dates, desc=f"OracleUniverse {self.exposure_id}", total=len(valid_dates)):
            for _ in range(int(ORACLE_UNIVERSE_K_PER_DAY)):
                H = int(self.rng.integers(HORIZON_MIN, HORIZON_MAX + 1))
                Q = int(self.rng.integers(VOLUME_MIN, VOLUME_MAX + 1))

                end_date, h_real, flag = self._shorten_end_date_fast(pd.Timestamp(s), H)
                if flag == "SHORTENED":
                    shortened += 1
                if flag == "REJECTED":
                    rejected += 1
                    continue

                pm = self._physical_metrics(pd.Timestamp(s).normalize(), end_date, Q)
                if not pm:
                    rejected += 1
                    continue

                sc = ScenarioSpec(
                    scenario_id=_make_scenario_id(
                        exposure_id=self.exposure_id,
                        scenario_kind="oracle_universe",
                        start_date=pd.Timestamp(s).normalize(),
                        end_date=end_date,
                        volume_bbl=Q,
                        oracle_pool="UNIVERSE",
                        oracle_candidate=1,
                    ),
                    exposure_id=self.exposure_id,
                    start_date=pd.Timestamp(s).normalize(),
                    end_date=end_date,
                    horizon_days_target=H,
                    horizon_days_realized=h_real,
                    volume_bbl=Q,
                    scenario_kind="oracle_universe",
                    tag=ORACLE_POOL_TAG,
                    company_id=None,
                    company_size=None,
                    oracle_pool="UNIVERSE",
                    oracle_candidate=1,
                    spot_start=pm.get("spot_start"),
                    spot_end=pm.get("spot_end"),
                    score_physical=pm.get("score_physical"),
                    spot_high=pm.get("spot_high"),
                    spot_low=pm.get("spot_low"),
                    pnl_physical=pm.get("pnl_physical"),
                    return_simple=pm.get("return_simple"),
                    mdd_abs_usd=pm.get("mdd_abs_usd"),
                    mdd_pct=pm.get("mdd_pct"),
                    mae_abs_usd=pm.get("mae_abs_usd"),
                    missed_upside_usd=pm.get("missed_upside_usd"),
                    data_coverage_flag=flag,
                )
                rows.append(sc.__dict__)

        out = pd.DataFrame(rows)
        if self.verbose:
            _log(
                f"[ScenarioGen:{self.exposure_id}] Oracle universe done | total={len(out)} | shortened={shortened} | rejected={rejected} | time={_fmt_s(_now()-t0)}"
            )
        return out

    def select_oracle_extremes_daily(self, oracle_universe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select best/worst per day from oracle universe (EXTREME_DAILY_BEST/WORST)."""
        t0 = _now()
        if oracle_universe is None or len(oracle_universe) == 0:
            return pd.DataFrame(), pd.DataFrame()

        df = oracle_universe.copy()
        df["start_day"] = pd.to_datetime(df["start_date"]).dt.normalize()

        idx_best = df.groupby("start_day")[ORACLE_SCORE_COL].idxmax()
        best = df.loc[idx_best].copy().reset_index(drop=True)
        best["oracle_series"] = "EXTREME_DAILY_BEST"
        best = _enrich_from_universe(best, oracle_universe, PHYS_COLS)

        idx_worst = df.groupby("start_day")[ORACLE_SCORE_COL].idxmin()
        worst = df.loc[idx_worst].copy().reset_index(drop=True)
        worst["oracle_series"] = "EXTREME_DAILY_WORST"
        worst = _enrich_from_universe(worst, oracle_universe, PHYS_COLS)

        _log(
            f"[ScenarioGen:{self.exposure_id}] Oracle extremes selected | best={len(best)} | worst={len(worst)} | time={_fmt_s(_now()-t0)}"
        )
        return best, worst

    def select_oracle_feasible_monthly_or(self, oracle_universe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select feasible monthly best/worst using OR (MILP) from oracle universe.

        Constraints per month:
          - number of trades in [L,U] where (L,U)=ORACLE_MONTHLY_TRADES_BOUNDS
          - max 1 trade per start_day

        If a MILP solver (pulp) is not available, raise a clear error.
        """
        t0 = _now()
        if oracle_universe is None or len(oracle_universe) == 0:
            return pd.DataFrame(), pd.DataFrame()

        L, U = ORACLE_MONTHLY_TRADES_BOUNDS
        df = oracle_universe.copy()
        df["start_day"] = pd.to_datetime(df["start_date"]).dt.normalize()
        df["month"] = df["start_day"].dt.to_period("M").astype(str)

        solver_name, pulp = _ensure_solver()
        if solver_name is None:
            raise RuntimeError(
                f"pulp not installed. Install with: pip install pulp (needed for feasible monthly oracle OR selection)"
            )

        best_rows = []
        worst_rows = []

        months = sorted(df["month"].unique())
        for m in _pbar(months, desc=f"OracleOR {self.exposure_id}", total=len(months)):
            g = df[df["month"] == m].copy()
            if len(g) == 0:
                continue

            # Cap candidate set for solver speed (reproducible + includes tails)
            cap = 2000
            if len(g) > cap:
                g_best = g.nlargest(cap // 2, ORACLE_SCORE_COL)
                g_worst = g.nsmallest(cap // 2, ORACLE_SCORE_COL)
                g = pd.concat([g_best, g_worst], axis=0).drop_duplicates(subset=["scenario_id"]).reset_index(drop=True)
            # Ensure indices are 0..len(g)-1 after cap or if not capped
            g = g.reset_index(drop=True)

            # MILP maximize
            prob = pulp.LpProblem(f"oracle_best_{m}", pulp.LpMaximize)
            x = pulp.LpVariable.dicts("x", list(range(len(g))), lowBound=0, upBound=1, cat=pulp.LpBinary)
            prob += pulp.lpSum([x[i] * float(g.iloc[i][ORACLE_SCORE_COL]) for i in range(len(g))])
            prob += pulp.lpSum([x[i] for i in range(len(g))]) >= L
            prob += pulp.lpSum([x[i] for i in range(len(g))]) <= U
            for d, idxs in g.groupby("start_day").groups.items():
                idxs2 = [int(i) for i in idxs if int(i) in x]
                if idxs2:
                    prob += pulp.lpSum([x[i] for i in idxs2]) <= 1
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            chosen_best = [i for i in range(len(g)) if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5]

            # MILP minimize
            prob2 = pulp.LpProblem(f"oracle_worst_{m}", pulp.LpMinimize)
            y = pulp.LpVariable.dicts("y", list(range(len(g))), lowBound=0, upBound=1, cat=pulp.LpBinary)
            prob2 += pulp.lpSum([y[i] * float(g.iloc[i][ORACLE_SCORE_COL]) for i in range(len(g))])
            prob2 += pulp.lpSum([y[i] for i in range(len(g))]) >= L
            prob2 += pulp.lpSum([y[i] for i in range(len(g))]) <= U
            for d, idxs in g.groupby("start_day").groups.items():
                idxs2 = [int(i) for i in idxs if int(i) in y]
                if idxs2:
                    prob2 += pulp.lpSum([y[i] for i in idxs2]) <= 1
            prob2.solve(pulp.PULP_CBC_CMD(msg=False))
            chosen_worst = [i for i in range(len(g)) if pulp.value(y[i]) is not None and pulp.value(y[i]) > 0.5]

            if chosen_best:
                sel = g.iloc[chosen_best].copy()
                sel["oracle_series"] = "FEASIBLE_MONTHLY_BEST"
                best_rows.append(sel)
            if chosen_worst:
                sel = g.iloc[chosen_worst].copy()
                sel["oracle_series"] = "FEASIBLE_MONTHLY_WORST"
                worst_rows.append(sel)

        best = pd.concat(best_rows, axis=0).reset_index(drop=True) if best_rows else pd.DataFrame()
        worst = pd.concat(worst_rows, axis=0).reset_index(drop=True) if worst_rows else pd.DataFrame()
        best = _enrich_from_universe(best, oracle_universe, PHYS_COLS)
        worst = _enrich_from_universe(worst, oracle_universe, PHYS_COLS)

        _log(
            f"[ScenarioGen:{self.exposure_id}] Oracle feasible monthly selected | best_rows={len(best)} | worst_rows={len(worst)} | time={_fmt_s(_now()-t0)}"
        )
        return best, worst

    def write_outputs(
        self,
        baseline_df: pd.DataFrame,
        company_df: pd.DataFrame,
        oracle_universe_df: pd.DataFrame,
        oracle_extreme_best_df: pd.DataFrame,
        oracle_extreme_worst_df: pd.DataFrame,
        oracle_feasible_best_df: pd.DataFrame,
        oracle_feasible_worst_df: pd.DataFrame,
    ) -> None:
        # Place outputs under per-exposure folder
        exp_dir = os.path.join(self.out_dir, self.exposure_id)
        os.makedirs(exp_dir, exist_ok=True)

        base_path = os.path.join(exp_dir, "baseline.parquet")
        comp_path = os.path.join(exp_dir, "companies.parquet")
        oraU_path = os.path.join(exp_dir, "oracle_universe.parquet")
        oraALL_path = os.path.join(exp_dir, "oracle_all.parquet")
        report_path = os.path.join(exp_dir, "report.xlsx")

        def _ensure_column(df: pd.DataFrame, col: str, default):
            if df is not None and len(df) > 0 and col not in df.columns:
                df[col] = default
            return df

        # Add missing columns for schema consistency
        for _df in [baseline_df, company_df, oracle_universe_df]:
            if _df is not None and len(_df) > 0:
                if "oracle_series" not in _df.columns:
                    _df["oracle_series"] = ""
                # Migrate scenario_type to scenario_kind if needed
                if "scenario_kind" not in _df.columns:
                    if "scenario_type" in _df.columns:
                        # Map known types
                        typemap = {"BASELINE": "baseline", "COMPANY": "company", "ORACLE": "oracle_universe"}
                        _df["scenario_kind"] = _df["scenario_type"].map(typemap).fillna(_df["scenario_type"])
                    else:
                        _df["scenario_kind"] = ""
                if "scenario_id" not in _df.columns:
                    _df["scenario_id"] = ""
        def _write(df: pd.DataFrame, path_parquet: str, overwrite: bool = False):
            if df is None or len(df) == 0:
                return

            # parquet if available, else csv fallback
            existing = _path_exists_any(path_parquet)
            if existing is not None and (not overwrite):
                _log(f"[ScenarioGen:{self.exposure_id}] Skip write (exists): {existing}")
                return

            # Write dataframe
            try:
                df.to_parquet(path_parquet, index=False)
            except Exception:
                df.to_csv(path_parquet.replace(".parquet", ".csv"), index=False)

            # Also write a fast-load NPZ file with essential arrays (same row count)
            npz_path = path_parquet.replace(".parquet", ".npz")
            n = len(df)

            def _col_datetime(col: str) -> np.ndarray:
                if col in df.columns:
                    return pd.to_datetime(df[col], errors="coerce").dt.normalize().to_numpy(dtype="datetime64[ns]")
                return np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

            def _col_float(col: str) -> np.ndarray:
                if col in df.columns:
                    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                return np.full(n, np.nan, dtype=float)

            def _col_int(col: str) -> np.ndarray:
                if col in df.columns:
                    return pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=int)
                return np.zeros(n, dtype=int)

            def _col_obj(col: str) -> np.ndarray:
                if col in df.columns:
                    return df[col].astype(str).fillna("").to_numpy(dtype=object)
                return np.full(n, "", dtype=object)

            arrs = {
                "scenario_id": _col_obj("scenario_id"),
                "start_date": _col_datetime("start_date"),
                "end_date": _col_datetime("end_date"),
                "volume_bbl": _col_int("volume_bbl"),
                "horizon_days_target": _col_int("horizon_days_target"),
                "horizon_days_realized": _col_int("horizon_days_realized"),
                "company_id": _col_obj("company_id"),
                "company_size": _col_obj("company_size"),
                "oracle_series": _col_obj("oracle_series"),
                "scenario_kind": _col_obj("scenario_kind"),
                "scenario_record_id": _col_obj("scenario_record_id"),
                "scenario_id_with_label": _col_obj("scenario_id_with_label"),
            }

            try:
                np.savez_compressed(npz_path, **arrs)
            except Exception as e:
                _log(f"[ScenarioGen:{self.exposure_id}] WARNING: failed to write NPZ ({npz_path}): {e}")

        _write(baseline_df, base_path)
        _write(company_df, comp_path)
        _write(oracle_universe_df, oraU_path)

        # Consolidated ORACLE output (single file; EXCLUDES universe)
        # Contains ONLY the 4 oracle series:
        #   EXTREME_DAILY_BEST / EXTREME_DAILY_WORST / FEASIBLE_MONTHLY_BEST / FEASIBLE_MONTHLY_WORST
        oracle_parts = []

        if oracle_extreme_best_df is not None and len(oracle_extreme_best_df) > 0:
            b = oracle_extreme_best_df.copy()
            if "oracle_series" not in b.columns:
                b["oracle_series"] = "EXTREME_DAILY_BEST"
            oracle_parts.append(b)

        if oracle_extreme_worst_df is not None and len(oracle_extreme_worst_df) > 0:
            w = oracle_extreme_worst_df.copy()
            if "oracle_series" not in w.columns:
                w["oracle_series"] = "EXTREME_DAILY_WORST"
            oracle_parts.append(w)

        if oracle_feasible_best_df is not None and len(oracle_feasible_best_df) > 0:
            fb = oracle_feasible_best_df.copy()
            if "oracle_series" not in fb.columns:
                fb["oracle_series"] = "FEASIBLE_MONTHLY_BEST"
            oracle_parts.append(fb)

        if oracle_feasible_worst_df is not None and len(oracle_feasible_worst_df) > 0:
            fw = oracle_feasible_worst_df.copy()
            if "oracle_series" not in fw.columns:
                fw["oracle_series"] = "FEASIBLE_MONTHLY_WORST"
            oracle_parts.append(fw)

        oracle_all = pd.concat(oracle_parts, axis=0, ignore_index=True) if oracle_parts else pd.DataFrame()

        if len(oracle_all) > 0:
            oracle_all["scenario_kind"] = "oracle_all"
            # Disambiguate labeled rows while keeping scenario_id joinable
            if "oracle_series" not in oracle_all.columns:
                oracle_all["oracle_series"] = ""
            oracle_all["scenario_record_id"] = oracle_all.apply(
                lambda r: _make_scenario_record_id(str(r.get("scenario_id")), str(r.get("oracle_series"))),
                axis=1,
            )
            oracle_all["scenario_id_with_label"] = oracle_all.apply(
                lambda r: f"{r.get('scenario_id')}__{r.get('oracle_series')}",
                axis=1,
            )
        else:
            # Ensure columns exist for empty df
            for col in ["scenario_kind", "scenario_record_id", "scenario_id_with_label", "oracle_series"]:
                if col not in oracle_all.columns:
                    oracle_all[col] = ""

        # Always overwrite oracle_all to guarantee consistency
        _write(oracle_all, oraALL_path, overwrite=True)

        def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
            if df is None or len(df) == 0 or col not in df.columns:
                return pd.Series(dtype=float)
            return pd.to_numeric(df[col], errors="coerce")

        def _var_es(x: pd.Series, alpha: float) -> Tuple[float, float]:
            """Historical VaR/ES on a PnL series (left tail)."""
            x = pd.to_numeric(x, errors="coerce").dropna()
            if len(x) == 0:
                return float("nan"), float("nan")
            q = float(x.quantile(alpha))
            es = float(x[x <= q].mean()) if (x <= q).any() else q
            return q, es

        def _kpi_row(df: pd.DataFrame, name: str) -> Dict:
            if df is None or len(df) == 0:
                return {"dataset": name, "n": 0}

            h_t = _safe_series(df, "horizon_days_target")
            h_r = _safe_series(df, "horizon_days_realized")
            vol = _safe_series(df, "volume_bbl")
            pnl = _safe_series(df, "pnl_physical")
            ret = _safe_series(df, "return_simple")
            mdd = _safe_series(df, "mdd_abs_usd")
            mae = _safe_series(df, "mae_abs_usd")
            mu = _safe_series(df, "missed_upside_usd")
            rng = None
            if (df is not None) and ("spot_high" in df.columns) and ("spot_low" in df.columns):
                rng = pd.to_numeric(df["spot_high"], errors="coerce") - pd.to_numeric(df["spot_low"], errors="coerce")

            var01, es01 = _var_es(pnl, 0.01)
            var05, es05 = _var_es(pnl, 0.05)

            out = {
                "dataset": name,
                "n": int(len(df)),
                "shortened_pct": float((df["data_coverage_flag"] == "SHORTENED").mean()) if "data_coverage_flag" in df.columns else float("nan"),
                "h_target_mean": float(h_t.mean()) if len(h_t) else float("nan"),
                "h_real_mean": float(h_r.mean()) if len(h_r) else float("nan"),
                "h_real_min": float(h_r.min()) if len(h_r) else float("nan"),
                "h_real_max": float(h_r.max()) if len(h_r) else float("nan"),
                "vol_mean": float(vol.mean()) if len(vol) else float("nan"),
                "vol_min": float(vol.min()) if len(vol) else float("nan"),
                "vol_max": float(vol.max()) if len(vol) else float("nan"),
            }

            if len(pnl):
                out.update({
                    "pnl_mean": float(pnl.mean()),
                    "pnl_std": float(pnl.std(ddof=1)) if len(pnl) > 1 else float("nan"),
                    "pnl_min": float(pnl.min()),
                    "pnl_p01": float(pnl.quantile(0.01)),
                    "pnl_p05": float(pnl.quantile(0.05)),
                    "pnl_p50": float(pnl.quantile(0.50)),
                    "pnl_p95": float(pnl.quantile(0.95)),
                    "pnl_p99": float(pnl.quantile(0.99)),
                    "pnl_max": float(pnl.max()),
                    "hit_ratio": float((pnl > 0).mean()),
                    "VaR_1%": var01,
                    "ES_1%": es01,
                    "VaR_5%": var05,
                    "ES_5%": es05,
                })

            if len(ret):
                out.update({
                    "ret_mean": float(ret.mean()),
                    "ret_std": float(ret.std(ddof=1)) if len(ret) > 1 else float("nan"),
                    "ret_p05": float(ret.quantile(0.05)),
                    "ret_p50": float(ret.quantile(0.50)),
                    "ret_p95": float(ret.quantile(0.95)),
                })

            if len(mdd):
                out.update({
                    "mdd_abs_mean": float(mdd.mean()),
                    "mdd_abs_p95": float(mdd.quantile(0.95)),
                    "mdd_abs_min": float(mdd.min()),
                })

            if "mdd_pct" in df.columns:
                mddp = _safe_series(df, "mdd_pct")
                if len(mddp.dropna()):
                    out.update({
                        "mdd_pct_mean": float(mddp.mean()),
                        "mdd_pct_p95": float(mddp.quantile(0.95)),
                        "mdd_pct_min": float(mddp.min()),
                    })

            if len(mae):
                out.update({
                    "mae_abs_mean": float(mae.mean()),
                    "mae_abs_min": float(mae.min()),
                })

            if len(mu):
                out.update({
                    "missed_upside_mean": float(mu.mean()),
                    "missed_upside_p95": float(mu.quantile(0.95)),
                })

            if rng is not None:
                rng = pd.to_numeric(rng, errors="coerce")
                out.update({
                    "spot_range_mean": float(rng.mean()),
                    "spot_range_p95": float(rng.quantile(0.95)),
                })

            return out

        def _describe_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
            if df is None or len(df) == 0:
                return pd.DataFrame()
            cols = [c for c in [
                "horizon_days_target",
                "horizon_days_realized",
                "volume_bbl",
                "spot_start",
                "spot_end",
                "spot_high",
                "spot_low",
                "pnl_physical",
                "return_simple",
                "mdd_abs_usd",
                "mdd_pct",
                "mae_abs_usd",
                "missed_upside_usd",
            ] if c in df.columns]
            if not cols:
                return pd.DataFrame()
            X = df[cols].apply(pd.to_numeric, errors="coerce")
            desc = X.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
            # skew/kurtosis are thesis-friendly
            try:
                desc["skew"] = X.skew(numeric_only=True)
                desc["kurtosis"] = X.kurtosis(numeric_only=True)
            except Exception:
                pass
            desc.insert(0, "dataset", name)
            desc.insert(1, "metric", desc.index)
            desc.reset_index(drop=True, inplace=True)
            return desc

        def _monthly_agg(df: pd.DataFrame, name: str) -> pd.DataFrame:
            if df is None or len(df) == 0:
                return pd.DataFrame()
            if "start_date" not in df.columns:
                return pd.DataFrame()
            tmp = df.copy()
            tmp["start_month"] = pd.to_datetime(tmp["start_date"]).dt.to_period("M").astype(str)
            pnl = _safe_series(tmp, "pnl_physical")
            tmp["_pnl"] = pnl
            tmp["_hit"] = (pnl > 0).astype(int)
            out = tmp.groupby("start_month", as_index=False).agg(
                n=("scenario_id", "count"),
                pnl_mean=("_pnl", "mean"),
                pnl_std=("_pnl", "std"),
                hit_ratio=("_hit", "mean"),
            )
            # add tail risk per month
            var_es = []
            for m in out["start_month"].tolist():
                x = tmp.loc[tmp["start_month"] == m, "_pnl"]
                v1, e1 = _var_es(x, 0.01)
                v5, e5 = _var_es(x, 0.05)
                var_es.append((v1, e1, v5, e5))
            out[["VaR_1%", "ES_1%", "VaR_5%", "ES_5%"]] = pd.DataFrame(var_es)
            out.insert(0, "dataset", name)
            return out

        def _company_profile(company_df: pd.DataFrame) -> pd.DataFrame:
            """Company-level table: 900 rows (manageable) with PnL and risk summaries."""
            if company_df is None or len(company_df) == 0:
                return pd.DataFrame()
            if "company_id" not in company_df.columns:
                return pd.DataFrame()
            tmp = company_df.copy()
            tmp["_pnl"] = _safe_series(tmp, "pnl_physical")
            tmp["_mdd"] = _safe_series(tmp, "mdd_abs_usd")
            tmp["_mu"] = _safe_series(tmp, "missed_upside_usd")
            tmp["_hit"] = (tmp["_pnl"] > 0).astype(int)
            grp_cols = ["company_id"]
            if "company_size" in tmp.columns:
                grp_cols.append("company_size")
            prof = tmp.groupby(grp_cols, as_index=False).agg(
                n_trades=("scenario_id", "count"),
                pnl_mean=("_pnl", "mean"),
                pnl_std=("_pnl", "std"),
                pnl_min=("_pnl", "min"),
                pnl_p05=("_pnl", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.05))),
                pnl_p50=("_pnl", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.50))),
                pnl_p95=("_pnl", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.95))),
                hit_ratio=("_hit", "mean"),
                mdd_mean=("_mdd", "mean"),
                mdd_min=("_mdd", "min"),
                missed_upside_mean=("_mu", "mean"),
            )
            # optional: worst 1% tail per company (VaR/ES)
            v_list = []
            for _, r in prof.iterrows():
                cid = r["company_id"]
                x = tmp.loc[tmp["company_id"] == cid, "_pnl"]
                v1, e1 = _var_es(x, 0.01)
                v5, e5 = _var_es(x, 0.05)
                v_list.append((v1, e1, v5, e5))
            prof[["VaR_1%", "ES_1%", "VaR_5%", "ES_5%"]] = pd.DataFrame(v_list)
            return prof

        # Build dataset map for reporting
        datasets: Dict[str, pd.DataFrame] = {
            "baseline": baseline_df,
            "company": company_df,
            "oracle_universe": oracle_universe_df,
            "oracle_extreme_daily_best": oracle_extreme_best_df,
            "oracle_extreme_daily_worst": oracle_extreme_worst_df,
            "oracle_feasible_monthly_best": oracle_feasible_best_df,
            "oracle_feasible_monthly_worst": oracle_feasible_worst_df,
        }

        # KPIs (overall)
        kpi = pd.DataFrame([_kpi_row(df, name) for name, df in datasets.items() if df is not None])

        # Descriptive stats (stacked long table)
        desc_tables = []
        for name, df in datasets.items():
            dtab = _describe_table(df, name)
            if len(dtab):
                desc_tables.append(dtab)
        desc_all = pd.concat(desc_tables, axis=0, ignore_index=True) if desc_tables else pd.DataFrame()

        # Monthly aggregates (for narrative + crisis windows)
        month_tables = []
        for name, df in datasets.items():
            mtab = _monthly_agg(df, name)
            if len(mtab):
                month_tables.append(mtab)
        monthly_all = pd.concat(month_tables, axis=0, ignore_index=True) if month_tables else pd.DataFrame()

        # Company profile (900-ish rows; thesis-friendly)
        company_prof = _company_profile(company_df)

        # Write ONE Excel report (overwrite each run; avoids stale partial reports)
        try:
            with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
                kpi.to_excel(writer, sheet_name="KPI_overall", index=False)
                if len(desc_all):
                    desc_all.to_excel(writer, sheet_name="DescriptiveStats", index=False)
                if len(monthly_all):
                    monthly_all.to_excel(writer, sheet_name="MonthlyAgg", index=False)
                if len(company_prof):
                    company_prof.to_excel(writer, sheet_name="CompanyProfile", index=False)

                # small sheet: config hints
                meta = pd.DataFrame([
                    {"key": "exposure_id", "value": self.exposure_id},
                    {"key": "seed", "value": self.seed},
                    {"key": "min_realized_horizon", "value": MIN_REALIZED_HORIZON},
                    {"key": "baseline_name", "value": BASELINE_NAME},
                    {"key": "oracle_k_per_day", "value": ORACLE_UNIVERSE_K_PER_DAY},
                    {"key": "oracle_monthly_bounds", "value": str(ORACLE_MONTHLY_TRADES_BOUNDS)},
                    {"key": "horizon_range", "value": f"{HORIZON_MIN}..{HORIZON_MAX} calendar days"},
                    {"key": "volume_range", "value": f"{VOLUME_MIN}..{VOLUME_MAX} bbl"},
                ])
                meta.to_excel(writer, sheet_name="Meta", index=False)
        except Exception as e:
            _log(f"[ScenarioGen:{self.exposure_id}] WARNING: failed to write Excel report ({report_path}): {e}")

        _log(f"[ScenarioGen:{self.exposure_id}] Report: {report_path}")

        # Print output file hints (parquet or csv fallback)
        for p in [base_path, comp_path, oraU_path, oraALL_path, report_path]:
            p2 = p
            if (not os.path.exists(p2)) and p2.endswith('.parquet'):
                p2 = p2.replace('.parquet', '.csv')
            if os.path.exists(p2):
                try:
                    sz = os.path.getsize(p2)
                    _log(f"[ScenarioGen:{self.exposure_id}] Output: {p2} | {sz/1024/1024:.2f} MB")
                except Exception:
                    _log(f"[ScenarioGen:{self.exposure_id}] Output: {p2}")

        if self.verbose:
            _log(f"[ScenarioGen:{self.exposure_id}] Wrote outputs to: {exp_dir}")


# -----------------------------
# CLI
# -----------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default="MasterData.parquet")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--out_dir", type=str, default="scenarios")
    ap.add_argument(
        "--exposure",
        type=str,
        default="WTI_SPOT",
        help="Exposure id: WTI_SPOT | BRENT_SPOT | OPEC_BASKET | ALL",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no_company", action="store_true")
    ap.add_argument("--no_oracle", action="store_true")
    args = ap.parse_args()

    allowed = {"WTI_SPOT", "BRENT_SPOT", "OPEC_BASKET", "ALL"}
    if args.exposure not in allowed:
        raise ValueError(f"Invalid exposure: {args.exposure}. Allowed: {sorted(list(allowed))}")

    exposures = ["WTI_SPOT", "BRENT_SPOT", "OPEC_BASKET"] if args.exposure == "ALL" else [args.exposure]

    _log(
        f"[ScenarioGen] Start | exposures={exposures} | master='{args.master}' | config='{args.config}' | out_dir='{args.out_dir}'"
    )

    ad = DataAdapter(master_csv=args.master, config_path=args.config, verbose=True)

    for exposure_id in exposures:
        _log(f"\n[ScenarioGen] ===== RUN exposure={exposure_id} =====")

        gen = PhysicalScenarioGenerator(
            adapter=ad,
            exposure_id=exposure_id,
            out_dir=args.out_dir,
            seed=args.seed,
            verbose=True,
        )

        # Output paths (for cache/reuse) - per exposure bundle
        exp_dir = os.path.join(args.out_dir, exposure_id)
        base_path = os.path.join(exp_dir, "baseline.parquet")
        comp_path = os.path.join(exp_dir, "companies.parquet")
        oraU_path = os.path.join(exp_dir, "oracle_universe.parquet")
        oraALL_path = os.path.join(exp_dir, "oracle_all.parquet")
        report_path = os.path.join(exp_dir, "report.xlsx")

        # Note: individual oracle series files are not persisted anymore; they are consolidated into oracle_all

        def _maybe_load(path_parquet: str, name: str) -> Optional[pd.DataFrame]:
            existing = _path_exists_any(path_parquet)
            if existing is None:
                return None
            try:
                _log(f"[ScenarioGen] Reuse existing {name}: {existing}")
                df = _read_df_any(path_parquet)
                # If dataframe is empty, allow as-is (for oracle_all or any)
                if df is not None and len(df) == 0:
                    return df
                # If cached outputs are from an older schema (missing physical cols), force regeneration
                if name in {"oracle_universe", "baseline", "company"}:
                    missing = [c for c in PHYS_COLS if c not in df.columns] + [c for c in ["scenario_id", "scenario_kind"] if c not in df.columns]
                    if missing:
                        _log(f"[ScenarioGen] Cache schema mismatch for {name}: missing {missing} | will regenerate")
                        return None
                if name == "oracle_all":
                    required_cols = ["scenario_id", "scenario_kind", "oracle_series"]
                    missing = [c for c in required_cols if c not in df.columns]
                    if missing:
                        _log(f"[ScenarioGen] Cache schema mismatch for oracle_all: missing {missing} | will regenerate")
                        return None
                return df
            except Exception as e:
                _log(f"[ScenarioGen] WARNING: failed to read existing {name} ({existing}): {e} | will regenerate")
                return None

        baseline = _maybe_load(base_path, "baseline")
        if baseline is None:
            baseline = gen.generate_baseline()

        if args.no_company:
            company = pd.DataFrame()
        else:
            company = _maybe_load(comp_path, "company")
            if company is None:
                company = gen.generate_company_scenarios()

        if args.no_oracle:
            oracle_universe = pd.DataFrame()
            ob = ow = ofb = ofw = pd.DataFrame()
        else:
            oracle_universe = _maybe_load(oraU_path, "oracle_universe")
            oracle_all_cached = _maybe_load(oraALL_path, "oracle_all")

            if oracle_all_cached is not None:
                _log(f"[ScenarioGen] Reuse existing oracle_all: {oraALL_path}")
                # Do not recompute selections; keep placeholders empty
                ob = ow = ofb = ofw = pd.DataFrame()
                # Ensure universe is at least available for reporting; if missing, generate it quickly
                if oracle_universe is None:
                    oracle_universe = gen.generate_oracle_universe()
            else:
                if oracle_universe is None:
                    oracle_universe = gen.generate_oracle_universe()
                ob, ow = gen.select_oracle_extremes_daily(oracle_universe)
                ofb, ofw = gen.select_oracle_feasible_monthly_or(oracle_universe)

        gen.write_outputs(baseline, company, oracle_universe, ob, ow, ofb, ofw)


if __name__ == "__main__":
    main()