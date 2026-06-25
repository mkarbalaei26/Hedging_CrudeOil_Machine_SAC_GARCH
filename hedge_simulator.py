"""
hedge_simulator.py
---------------------
Core hedging simulator (locked mechanics).

This module simulates PnL of a physical crude oil trade hedged with CL futures.
All models/strategies are plugins that only output hedge ratio h_t.

Locked mechanics:
- Physical PnL: dS_t * Q
- Futures PnL:  N_{t-1} * dF_t * 1000
- N_t integer contracts: N_t = round(h_t * Q / 1000)
- Costs:
    trade: |N_t - N_{t-1}| * 10
    roll:  2*|N_t|*10 if roll_flag and N_t!=0
- roll_flag comes from price_engine output

Leakage rule:
- Strategy only sees history up to t (inclusive).

Outputs:
- summary parquet
- optional daily path parquet (for first N scenarios)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

from window_engine import WindowEngine
from cost_model import ExecutionCostModel, ExecutionCostConfig

# Import strategy plugin system
from strategy import make_strategy, available_strategies




# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class SimulatorConfig:
    exposure_id: str
    mode_roll: bool = True
    physical_side: int = 1  # +1: long physical (default), -1: short physical
    dynamic: bool = True  # True: daily updates; False: static (first-day only)
    cost_per_contract_trade_usd: float = 10.0

    # Optional position constraints
    h_clip: Optional[Tuple[float, float]] = (-3.0, 3.0)  # None => no clip
    max_contracts_abs: Optional[int] = None  # None => no cap

    # Output options
    save_daily_paths: bool = False
    daily_paths_max_scenarios: int = 50


# ------------------------------------------------------------
# Core simulator
# ------------------------------------------------------------

class HedgingSimulator:
    def __init__(self, window_engine: WindowEngine, cost_model: ExecutionCostModel, cfg: SimulatorConfig):
        self.we = window_engine
        self.cm = cost_model
        self.cfg = cfg

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if equity.size == 0:
            return float("nan")
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        return float(np.min(dd))

    def _round_contracts(self, h: float, Q_bbl: float) -> int:
        if self.cfg.h_clip is not None:
            lo, hi = self.cfg.h_clip
            h = float(np.clip(h, lo, hi))

        # Hedge direction convention (locked):
        # For a LONG physical exposure (physical_side=+1), the hedge should be SHORT futures.
        # For a SHORT physical exposure (physical_side=-1), the hedge should be LONG futures.
        hedge_dir = -1 * int(self.cfg.physical_side)

        n_float = float(h) * float(Q_bbl) / 1000.0
        n_abs = self.cm.to_int_contracts(n_float)
        n = int(hedge_dir * int(n_abs))

        if self.cfg.max_contracts_abs is not None:
            cap = int(self.cfg.max_contracts_abs)
            n = int(np.clip(n, -cap, cap))

        return int(n)

    def simulate_one(
        self,
        scenario_row: Dict[str, Any],
        strategy,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        sid = scenario_row.get("scenario_id")
        s = pd.Timestamp(scenario_row["start_date"]).normalize()
        e = pd.Timestamp(scenario_row["end_date"]).normalize()
        Q = float(scenario_row["volume_bbl"])

        winA = self.we.build_window_arrays(s, e, feature_cols=feature_cols, drop_nan=True)

        # Hot path arrays
        dS_arr = np.asarray(winA["dS"], dtype=float)
        dF_arr = np.asarray(winA["dF"], dtype=float)
        dates_arr = np.asarray(winA["dates"], dtype="datetime64[ns]")
        spot_arr = np.asarray(winA["spot"], dtype=float)
        fut_arr = np.asarray(winA["fut"], dtype=float)
        roll_arr = winA.get("roll_flag", None)

        # ----------------------------------------------------
        # Pre-trade history (no leakage) - used for rolling strategies
        # ----------------------------------------------------
        try:
            pre_hist = self.we.history_before(s, max_rows=2520)
        except Exception:
            pre_hist = pd.DataFrame(columns=["date", "spot", "fut", "dS", "dF"])

        # Reset strategy state per scenario
        if hasattr(strategy, "reset"):
            strategy.reset()

        win_len = int(len(dS_arr))

        # Build full (pre + window) price levels for strategy estimation
        if pre_hist is not None and len(pre_hist) > 0:
            spot_pre = pd.to_numeric(pre_hist.get("spot", np.nan), errors="coerce").to_numpy(dtype=float)
            fut_pre = pd.to_numeric(pre_hist.get("fut", np.nan), errors="coerce").to_numpy(dtype=float)
            pre_len = int(len(spot_pre))
        else:
            spot_pre = np.empty(0, dtype=float)
            fut_pre = np.empty(0, dtype=float)
            pre_len = 0

        spot_win = np.asarray(spot_arr, dtype=float)
        fut_win = np.asarray(fut_arr, dtype=float)

        spot_full = np.concatenate([spot_pre, spot_win])
        fut_full = np.concatenate([fut_pre, fut_win])

        # Build diffs for the full series (first diff = 0)
        dS_full = np.empty_like(spot_full, dtype=float)
        dF_full = np.empty_like(fut_full, dtype=float)
        if len(spot_full) > 0:
            dS_full[0] = 0.0
            dF_full[0] = 0.0
        if len(spot_full) > 1:
            dS_full[1:] = spot_full[1:] - spot_full[:-1]
            dF_full[1:] = fut_full[1:] - fut_full[:-1]

        # ----------------------------------------------------
        # Strategy fast path: build h_t once (preferred)
        # ----------------------------------------------------
        scenario_meta = dict(scenario_row)

        oracle_series_val = scenario_row.get("oracle_series")
        parsed_pool, parsed_freq, parsed_label = _parse_oracle_series(oracle_series_val)

        # Prefer explicit columns if provided by generator
        oracle_pool_val = scenario_row.get("oracle_pool", parsed_pool)
        oracle_freq_val = scenario_row.get("oracle_freq", parsed_freq)
        label_val = scenario_row.get("label", parsed_label)

        # Provide required feature arrays (aligned to full length). Pre part = NaN.
        if feature_cols:
            for col in feature_cols:
                if col in winA:
                    full_feat = np.full(pre_len + win_len, np.nan, dtype=float)
                    full_feat[pre_len:] = np.asarray(winA[col], dtype=float)
                    scenario_meta[col] = full_feat

        # Dates for full history are optional
        if dates_arr is not None:
            # dates_arr corresponds to window only; pad pre_len with NaT
            dates_win = pd.to_datetime(dates_arr).astype("datetime64[ns]")
            if pre_len > 0:
                dates_full = np.concatenate([np.full(pre_len, np.datetime64("NaT"), dtype="datetime64[ns]"), dates_win])
            else:
                dates_full = dates_win
        else:
            dates_full = None

        if hasattr(strategy, "build_h_path"):
            h_full = strategy.build_h_path(
                dS=dS_full,
                dF=dF_full,
                dates=dates_full,
                scenario_meta=scenario_meta,
                spot=spot_full,
                fut=fut_full,
            )
            h_full = np.asarray(h_full, dtype=float)
            if len(h_full) != len(dS_full):
                raise ValueError("Strategy build_h_path returned wrong length")
            h_path = h_full[-win_len:]
        else:
            # Fallback: legacy per-step API (slower)
            h_path = np.zeros(win_len, dtype=float)
            # Build a minimal history DataFrame once and slice by iloc
            hist_full = pd.DataFrame({
                "spot": spot_full,
                "fut": fut_full,
                "dS": dS_full,
                "dF": dF_full,
            })
            for t in range(win_len):
                hist = hist_full.iloc[: pre_len + t + 1]
                h_path[t] = float(strategy.get_h(t, hist, scenario_row))

        # Static mode: freeze at first-day hedge ratio
        if not self.cfg.dynamic and win_len > 0:
            h_path[:] = float(h_path[0])

        # ----------------------------------------------------
        # Vectorized contracts, PnL, turnover, and costs
        # ----------------------------------------------------
        # Clip h if configured
        if self.cfg.h_clip is not None:
            lo, hi = self.cfg.h_clip
            h_path = np.clip(h_path, float(lo), float(hi))

        hedge_dir = -1 * int(self.cfg.physical_side)
        n_float = (h_path.astype(float) * float(Q) / 1000.0)
        n_abs = self.cm.to_int_contracts_vec(n_float)
        n_new = (hedge_dir * n_abs).astype(int)

        if self.cfg.max_contracts_abs is not None:
            cap = int(self.cfg.max_contracts_abs)
            n_new = np.clip(n_new, -cap, cap).astype(int)

        # n_prev is yesterday's position (0 at t=0)
        n_prev = np.empty_like(n_new)
        if win_len > 0:
            n_prev[0] = 0
        if win_len > 1:
            n_prev[1:] = n_new[:-1]

        # Physical and futures PnL
        spot_pnl = dS_arr.astype(float) * float(Q)
        fut_pnl = n_prev.astype(float) * dF_arr.astype(float) * 1000.0

        # Costs (vectorized)
        if roll_arr is None or (not self.cfg.mode_roll):
            rf = np.zeros(win_len, dtype=int)
        else:
            # roll_arr is typically already a NumPy array; coerce safely without pandas fillna
            rf = np.asarray(roll_arr)
            if rf.dtype.kind in ("U", "S", "O"):
                # object/string -> numeric via pandas (small, per-window)
                rf = pd.to_numeric(pd.Series(rf), errors="coerce").to_numpy(dtype=float)
            else:
                rf = rf.astype(float, copy=False)
            rf = np.nan_to_num(rf, nan=0.0, posinf=0.0, neginf=0.0).astype(int)

        c = self.cm.total_cost_vec(n_prev=n_prev, n_new=n_new, roll_flag=rf)
        cost_trade = c["cost_trade"]
        cost_roll = c["cost_roll"]

        # Turnover
        turnover_n = np.abs(n_new - n_prev).astype(float)
        h_prev_arr = np.empty_like(h_path, dtype=float)
        if win_len > 0:
            h_prev_arr[0] = 0.0
        if win_len > 1:
            h_prev_arr[1:] = h_path[:-1]
        turnover_h = np.abs(h_path - h_prev_arr)

        trade_contracts = float(np.sum(turnover_n))
        roll_contracts = float(np.sum((rf.astype(bool) & (np.abs(n_new) > 0)).astype(float) * (2.0 * np.abs(n_new))))

        n_pos = n_new
        h_series = h_path

        net_pnl = spot_pnl + fut_pnl - cost_trade - cost_roll
        equity = np.cumsum(net_pnl)

        daily_df = pd.DataFrame()
        if self.cfg.save_daily_paths:
            daily_df = pd.DataFrame({
                "scenario_id": sid,
                "date": dates_arr,
                "spot": spot_arr,
                "fut": fut_arr,
                "dS": dS_arr,
                "dF": dF_arr,
                "spot_pnl": spot_pnl,
                "fut_pnl": fut_pnl,
                "cost_trade": cost_trade,
                "cost_roll": cost_roll,
                "net_pnl": net_pnl,
                "equity": equity,
                "h_t": h_series,
                "N_t": n_pos,
                "turnover_contracts": turnover_n,
                "turnover_h": turnover_h,
                "scenario_kind": scenario_row.get("scenario_kind"),
                "scenario_record_id": scenario_row.get("scenario_record_id"),
                "oracle_series": oracle_series_val,
                "oracle_pool": oracle_pool_val,
                "oracle_freq": oracle_freq_val,
                "label": label_val,
                "tag": scenario_row.get("tag"),
            })
            if rf is not None:
                daily_df["roll_flag"] = rf.astype(int)

        summary = {
            "scenario_id": sid,
            "exposure_id": self.cfg.exposure_id,
            # Scenario window (reporting-grade)
            "start_date": s,
            "end_date": e,
            "horizon_days_target": int(scenario_row.get("horizon_days_target", scenario_row.get("horizon_days", win_len - 1))),
            "horizon_days_realized": int(scenario_row.get("horizon_days_realized", win_len - 1)),

            # Oracle / scenario metadata
            "scenario_kind": scenario_row.get("scenario_kind"),
            "scenario_record_id": scenario_row.get("scenario_record_id"),
            "tag": scenario_row.get("tag"),

            "oracle_series": oracle_series_val,
            "oracle_pool": oracle_pool_val,
            "oracle_freq": oracle_freq_val,
            "label": label_val,
            "oracle_candidate": scenario_row.get("oracle_candidate"),

            "company_id": scenario_row.get("company_id"),
            "company_size": scenario_row.get("company_size"),
            "volume_bbl": Q,
            "strategy": getattr(strategy, "name", strategy.__class__.__name__),
            "dynamic": int(self.cfg.dynamic),
            "mode_roll": int(self.cfg.mode_roll),
            "spot_pnl_total": float(np.sum(spot_pnl)),
            "fut_pnl_total": float(np.sum(fut_pnl)),
            "cost_trade_total": float(np.sum(cost_trade)),
            "cost_roll_total": float(np.sum(cost_roll)),
            "net_pnl_total": float(np.sum(net_pnl)),
            "turnover_contracts": float(np.sum(turnover_n)),
            "turnover_h": float(np.sum(turnover_h)),
            "trade_contracts": float(trade_contracts),
            "roll_contracts": float(roll_contracts),
            "max_abs_contracts": int(np.max(np.abs(n_pos))) if len(n_pos) else 0,
            "mdd_equity": self._max_drawdown(equity),
        }

        return daily_df, summary

    def simulate_many(
        self,
        scenarios_df: pd.DataFrame,
        strategy,
        feature_cols: Optional[List[str]] = None,
        progress_desc: str = "Simulate",
        jobs: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        summaries: List[Dict[str, Any]] = []
        daily_paths: List[pd.DataFrame] = []

        max_daily = int(self.cfg.daily_paths_max_scenarios) if self.cfg.save_daily_paths else 0
        it = scenarios_df.to_dict("records")

        def _run_one(i_row: Tuple[int, Dict[str, Any]]):
            i, row = i_row
            try:
                strat_local = copy.deepcopy(strategy)
            except Exception:
                raise RuntimeError(
                    "Strategy is not deepcopy-able; run with --jobs 1 or implement clone()/factory for this strategy."
                )
            daily_df, summ = self.simulate_one(row, strategy=strat_local, feature_cols=feature_cols)
            return i, daily_df, summ

        if int(jobs) <= 1:
            for i, row in enumerate(tqdm(it, desc=progress_desc, total=len(it))):
                daily_df, summ = self.simulate_one(row, strategy=strategy, feature_cols=feature_cols)
                summaries.append(summ)
                if self.cfg.save_daily_paths and i < max_daily and len(daily_df):
                    daily_paths.append(daily_df)
        else:
            jobs = int(jobs)
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futures = [ex.submit(_run_one, (i, row)) for i, row in enumerate(it)]
                out_summ: List[Optional[Dict[str, Any]]] = [None] * len(futures)
                out_daily: List[Optional[pd.DataFrame]] = [None] * len(futures)

                for fut in tqdm(as_completed(futures), total=len(futures), desc=progress_desc):
                    i, daily_df, summ = fut.result()
                    out_summ[i] = summ
                    if self.cfg.save_daily_paths and i < max_daily and len(daily_df):
                        out_daily[i] = daily_df

                summaries = [s for s in out_summ if s is not None]
                daily_paths = [d for d in out_daily if d is not None]

        summary_df = pd.DataFrame(summaries)
        daily_df_all = pd.concat(daily_paths, axis=0, ignore_index=True) if daily_paths else pd.DataFrame()
        return daily_df_all, summary_df


# ------------------------------------------------------------
# CLI helpers
# --------
def _ensure_parquet_cache(path: str, cache_dir: str = "._cache") -> str:
    """If `path` is a CSV, create/use a cached parquet copy for faster reloads.

    Returns a path that can be loaded quickly (parquet if possible).
    """
    path = str(path)
    if path.endswith(".parquet"):
        return path

    if not path.endswith(".csv"):
        # Unknown type; just return as-is
        return path

    os.makedirs(cache_dir, exist_ok=True)

    # Deterministic cache name (avoid collisions across different folders)
    h = hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:10]
    stem = os.path.splitext(os.path.basename(path))[0]
    pq = os.path.join(cache_dir, f"{stem}_{h}.parquet")

    try:
        csv_mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return path

    if os.path.exists(pq):
        try:
            pq_mtime = os.path.getmtime(pq)
            if pq_mtime >= csv_mtime:
                return pq
        except Exception:
            pass

    print(f"[Cache] Building parquet cache for: {path} -> {pq}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    df.to_parquet(pq, index=False)
    return pq


def _load_price_engine(path: str) -> pd.DataFrame:
    path = str(path)

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        if "Date" not in z.files or "F_mark" not in z.files:
            raise ValueError("price_engine npz must contain arrays: Date, F_mark")
        df = pd.DataFrame({
            "Date": pd.to_datetime(z["Date"]).astype("datetime64[ns]"),
            "F_mark": z["F_mark"].astype(float),
        })
    else:
        fast_path = _ensure_parquet_cache(path)
        if fast_path.endswith(".parquet"):
            df = pd.read_parquet(fast_path)
        else:
            df = pd.read_csv(fast_path)

    if "Date" not in df.columns or "F_mark" not in df.columns:
        raise ValueError("price_engine must have columns: Date, F_mark")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df


def _load_scenarios(path: str) -> pd.DataFrame:
    path = str(path)

    if path.endswith(".parquet"):
        return pd.read_parquet(path)

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        # Required columns
        out = {
            "scenario_id": z["scenario_id"],
            "start_date": pd.to_datetime(z["start_date"], errors="coerce").dt.normalize().astype("datetime64[ns]"),
            "end_date": pd.to_datetime(z["end_date"], errors="coerce").dt.normalize().astype("datetime64[ns]"),
            "volume_bbl": z["volume_bbl"].astype(int),
            "horizon_days_target": z["horizon_days_target"].astype(int),
            "horizon_days_realized": z["horizon_days_realized"].astype(int),
        }
        # Optional columns (present for companies/oracle)
        for k in ["company_id", "company_size", "oracle_series"]:
            if k in z.files:
                out[k] = z[k]
        return pd.DataFrame(out)

    if path.endswith(".csv"):
        fast_path = _ensure_parquet_cache(path)
        if fast_path.endswith(".parquet"):
            df = pd.read_parquet(fast_path)
        else:
            df = pd.read_csv(path)
        if "start_date" in df.columns:
            df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.normalize()
        if "end_date" in df.columns:
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce").dt.normalize()
        return df

    # fallback
    return pd.read_parquet(path)

def _infer_spot_col(universe_df: pd.DataFrame, exposure_id: str) -> str:
    if "spot" in universe_df.columns:
        return "spot"
    if exposure_id in universe_df.columns:
        return exposure_id

    mapping = {
        "WTI_SPOT": "WTI",
        "BRENT_SPOT": "Brent",
        "OPEC_BASKET": "OPEC",
        "WTI": "WTI",
        "Brent": "Brent",
        "OPEC": "OPEC",
    }
    if exposure_id in mapping and mapping[exposure_id] in universe_df.columns:
        return mapping[exposure_id]

    candidates: List[str] = []
    if isinstance(exposure_id, str) and exposure_id.endswith("_SPOT"):
        base = exposure_id.replace("_SPOT", "")
        candidates += [base, base.upper(), base.title()]
    if isinstance(exposure_id, str) and "OPEC" in exposure_id.upper():
        candidates += ["OPEC", "Opec"]

    for c in candidates:
        if c in universe_df.columns:
            return c

    cols_preview = ",".join(list(universe_df.columns)[:30])
    raise ValueError(
        "Cannot infer spot column name from universe_df for exposure_id="
        f"{exposure_id}. Available columns preview: {cols_preview}"
    )


def _parse_oracle_series(val):
    """
    Parse strings like 'EXTREME_DAILY_BEST' -> (pool='EXTREME', freq='DAILY', label='BEST').
    Returns (oracle_pool, oracle_freq, label). If not parsable, returns (None, None, None).
    """
    if val is None:
        return None, None, None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None, None, None
    s = s.upper()
    parts = [p for p in s.split("_") if p]
    if not parts:
        return None, None, None
    oracle_pool = parts[0] if len(parts) >= 1 else None
    oracle_freq = parts[1] if len(parts) >= 2 else None
    label = parts[-1] if parts[-1] in ("BEST", "WORST") else None
    return oracle_pool, oracle_freq, label

def main():
    import argparse
    from data_adapter import DataAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default="MasterData.parquet")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--price_engine", type=str, default="MasterData_price_engine.parquet")

    ap.add_argument("--exposure", type=str, default="WTI_SPOT")
    ap.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        required=True,
        help="One or more scenario files: parquet/csv/npz",
    )

    ap.add_argument(
        "--scenario_names",
        type=str,
        nargs="+",
        default=None,
        help="Optional names (same count as --scenarios) used for output subfolders; default is file stem.",
    )

    ap.add_argument(
        "--scenario_stem",
        type=str,
        default=None,
        help=(
            "Override output subfolder stem for scenarios (used by orchestrator for chunked runs). "
            "Only valid when a single --scenarios file is provided."
        ),
    )

    ap.add_argument("--strategy", type=str, default="nohedge")
    ap.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of strategies to run in one invocation. If omitted, uses --strategy.",
    )
    ap.add_argument("--h", type=float, default=1.0, help="For naive/constant strategies: hedge ratio h")
    ap.add_argument("--window", type=int, default=120, help="For rolling strategies (e.g., ols_roll): lookback window")
    ap.add_argument("--intercept", action="store_true", help="Force OLS with intercept (if supported by strategy)")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--static", action="store_true")
    ap.add_argument("--no_roll", action="store_true")

    ap.add_argument("--save_daily", action="store_true")
    ap.add_argument("--daily_max", type=int, default=25)
    ap.add_argument("--cap", type=int, default=None, help="Limit number of scenarios loaded from each scenario file (smoke test)")

    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--jobs", type=int, default=1, help="Parallelism (threads) across scenarios. 1 = off")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.scenario_names is not None and len(args.scenario_names) != len(args.scenarios):
        raise ValueError("--scenario_names must have the same count as --scenarios")

    if args.scenario_stem is not None:
        if len(args.scenarios) != 1:
            raise ValueError("--scenario_stem can only be used when exactly one --scenarios file is provided")

    dyn = True
    if args.static:
        dyn = False
    if args.dynamic:
        dyn = True

    master_fast = _ensure_parquet_cache(args.master)
    ad = DataAdapter(master_fast, args.config)
    
    # Strategy via plugin factory (single or multiple)
    keys: List[str]
    if args.strategies is None:
        keys = [args.strategy]
    else:
        keys = list(args.strategies)

    strategies = []
    feature_cols_union: List[str] = []

    for key in keys:
        try:
            s_obj = make_strategy(
                key,
                h=args.h,
                window=args.window,
                intercept=bool(args.intercept),
                exposure_id=args.exposure,
            )
        except ValueError:
            print("[HedgingSimulator] Unknown strategy. Available:")
            for k, v in available_strategies().items():
                print(f"  - {k}: {v}")
            raise

        strategies.append(s_obj)

        if hasattr(s_obj, "required_feature_cols"):
            req = s_obj.required_feature_cols()
            for c in req:
                if c not in feature_cols_union:
                    feature_cols_union.append(c)

    uni = ad.get_universe(args.exposure, include_features=True, feature_role="both")
    # Ensure strategy-required feature columns exist even if config feature list is outdated.
    if feature_cols_union:
        missing_feats = [c for c in feature_cols_union if c not in uni.columns]
        if missing_feats:
            print(f"[HedgingSimulator] Injecting missing feature cols into universe: {missing_feats}")
            if str(master_fast).endswith(".parquet"):
                m = pd.read_parquet(master_fast)
            else:
                m = pd.read_csv(master_fast, parse_dates=["Date"])
            m["Date"] = pd.to_datetime(m["Date"]).dt.normalize()
            use_cols = ["Date"] + [c for c in missing_feats if c in m.columns]
            if len(use_cols) == 1:
                raise ValueError(
                    f"Required feature columns not found in master file: {missing_feats}. "
                    f"Make sure BaseGARCH output replaced MasterData.csv or update DataAdapter/config."
                )
            add = m[use_cols].copy()
            # Normalize universe date column name
            if "Date" not in uni.columns and "date" in uni.columns:
                uni = uni.rename(columns={"date": "Date"})
            uni["Date"] = pd.to_datetime(uni["Date"]).dt.normalize()
            uni = uni.merge(add, on="Date", how="left")

    pe = _load_price_engine(args.price_engine)
    spot_col = _infer_spot_col(uni, args.exposure)

    we = WindowEngine(
        universe_df=uni.rename(columns={spot_col: "spot"}),
        price_engine_df=pe,
        date_col="Date",
        spot_col="spot",
        futures_col="F_mark",
    )

    cm = ExecutionCostModel(ExecutionCostConfig(cost_per_contract_trade_usd=10.0))

    cfg = SimulatorConfig(
        exposure_id=args.exposure,
        mode_roll=not args.no_roll,
        dynamic=dyn,
        cost_per_contract_trade_usd=10.0,
        save_daily_paths=args.save_daily,
        daily_paths_max_scenarios=args.daily_max,
    )

    sim = HedgingSimulator(window_engine=we, cost_model=cm, cfg=cfg)

    tag = args.tag.strip()
    tag2 = f"_{tag}" if tag else ""

    for idx, scen_path in enumerate(args.scenarios):
        scen_name = None
        if args.scenario_stem is not None:
            scen_name = str(args.scenario_stem)
        elif args.scenario_names is not None:
            scen_name = args.scenario_names[idx]
        else:
            base = os.path.basename(scen_path)
            scen_name = os.path.splitext(base)[0]

        out_subdir = os.path.join(args.out_dir, scen_name)
        os.makedirs(out_subdir, exist_ok=True)

        sc = _load_scenarios(scen_path)
        if args.cap is not None:
            sc = sc.head(int(args.cap)).copy()

        for strat in strategies:
            # Determine required feature columns for this specific strategy
            feature_cols = strat.required_feature_cols() if hasattr(strat, "required_feature_cols") else None

            daily_df, summary_df = sim.simulate_many(
                sc,
                strategy=strat,
                feature_cols=feature_cols,
                jobs=int(args.jobs),
                progress_desc=(
                    f"Sim[{args.exposure}|{getattr(strat,'name',strat.__class__.__name__)}|"
                    f"{'dyn' if dyn else 'static'}|{'roll' if cfg.mode_roll else 'no-roll'}|{scen_name}]"
                ),
            )

            strat_name = getattr(strat, "name", strat.__class__.__name__)

            out_sum = os.path.join(
                out_subdir,
                f"hedge_summary_{args.exposure}_{strat_name}_{'dyn' if dyn else 'static'}_{'roll' if cfg.mode_roll else 'noroll'}{tag2}.parquet",
            )
            summary_df.to_parquet(out_sum, index=False)
            print(f"[HedgingSimulator] wrote: {out_sum} | rows={len(summary_df)}")

            if cfg.save_daily_paths and len(daily_df):
                out_day = os.path.join(
                    out_subdir,
                    f"hedge_daily_{args.exposure}_{strat_name}_{'dyn' if dyn else 'static'}_{'roll' if cfg.mode_roll else 'noroll'}{tag2}.parquet",
                )
                daily_df.to_parquet(out_day, index=False)
                print(f"[HedgingSimulator] wrote: {out_day} | rows={len(daily_df)}")


if __name__ == "__main__":
    main()
