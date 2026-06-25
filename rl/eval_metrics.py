

"""rl.eval_metrics

ارزیابی خروجی‌های RL (پوشش ریسک) و مقایسه با baselineها (Naive و DCC-GARCH).

این اسکریپت دو حالت دارد:
1) Aggregate from saved RL outputs:
   - خروجی‌های هر پنجره از مسیر `--rl_dir` خوانده می‌شوند:
     - جدید: `test_episodes_all.parquet` و `window_summary.json` (در هر پنجره)
     - قدیمی: `test_episodes.csv` (برای سازگاری)
   - اگر خروجی ادغام‌شده‌ی کل اجرا موجود باشد: `results_all_windows.parquet` (در ریشه‌ی run)
   - سپس خلاصه‌ی کلی و خلاصه‌ی پنجره‌ای تولید می‌شود.

2) Re-simulate baselines (و در صورت نیاز RL policy در آینده):
   - با داشتن cache `precompute_*.npz` و فایل سناریو (parquet/npz/csv) می‌توان PnL را برای سیاست‌های baseline محاسبه کرد.
   - سیاست‌های پشتیبانی‌شده:
     - unhedged (h=0)
     - naive (h ثابت)
     - dcc (h_t از ستون hedge ratio در MasterData)

نکته مهم:
- در پروژه‌ی فعلی، سری h_t مدل DCC باید از MasterData (ستون آماده) به اسکریپت داده شود.
  اگر ستون موجود نباشد، ارزیابی DCC انجام نمی‌شود و اسکریپت هشدار می‌دهد.

خروجی‌ها:
- out_dir/summary_windows.csv
- out_dir/summary_overall.csv
- out_dir/episodes_<policy>.csv  (در صورت استفاده از --scenario_file)
- out_dir/summary_grouped.csv (جدول مقایسه گروه‌بندی‌شده RL و baselineها)
- out_dir/compare_strategies.csv (pivot مقایسه استراتژی‌ها اگر داده کافی باشد)

Usage examples:

# فقط جمع‌بندی خروجی‌های RL
python -m rl.eval_metrics --rl_dir rl_runs_wti_lpm_parallel --out_dir eval_out

# مقایسه RL aggregation + شبیه‌سازی baselineها روی oracle_universe
python -m rl.eval_metrics \
  --rl_dir rl_runs_wti_lpm_parallel \
  --cache rl_cache/precompute_WTI.npz \
  --scenario_file scenarios/WTI_SPOT/oracle_universe.parquet \
  --policy naive --naive_h 1.0 \
  --policy unhedged \
  --policy dcc --master MasterData.csv --dcc_col WTI_h_dcc \
  --out_dir eval_out

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# IO helpers
# -------------------------

def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Precomputed:
    dates_int: np.ndarray
    spot: np.ndarray
    dS: np.ndarray
    pnl_1c: np.ndarray
    roll_flag: np.ndarray
    tradable: np.ndarray
    feature_matrix: np.ndarray


def load_precompute_npz(path: str) -> Precomputed:
    z = np.load(path, allow_pickle=False)
    required = ["dates_int", "spot", "dS", "pnl_1c", "roll_flag", "tradable", "feature_matrix"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise ValueError(f"precompute npz missing keys: {missing}")
    return Precomputed(
        dates_int=z["dates_int"],
        spot=z["spot"].astype(np.float64),
        dS=z["dS"].astype(np.float64),
        pnl_1c=z["pnl_1c"].astype(np.float64),
        roll_flag=z["roll_flag"].astype(np.int8),
        tradable=z["tradable"].astype(np.int8),
        feature_matrix=z["feature_matrix"].astype(np.float32),
    )


def load_scenarios(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    elif p.suffix.lower() == ".npz":
        z = np.load(p, allow_pickle=True)
        # heuristics: accept arrays for start/end/Q
        cols = {}
        for key in z.files:
            cols[key] = z[key]
        df = pd.DataFrame(cols)
    else:
        raise ValueError(f"Unsupported scenario file: {p}")

    # Normalize common column names
    rename_map = {
        "start": "start_date_int",
        "start_int": "start_date_int",
        "start_date": "start_date_int",
        "end": "end_date_int",
        "end_int": "end_date_int",
        "end_date": "end_date_int",
        "Q": "volume_bbl",
        "q": "volume_bbl",
        "volume": "volume_bbl",
        "volume_bbl": "volume_bbl",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for c in ["start_date_int", "end_date_int", "volume_bbl"]:
        if c not in df.columns:
            raise ValueError(
                f"Scenario file must contain '{c}'. Available columns: {list(df.columns)[:50]}"
            )

    df = df.copy()
    df["start_date_int"] = df["start_date_int"].astype(int)
    df["end_date_int"] = df["end_date_int"].astype(int)
    df["volume_bbl"] = df["volume_bbl"].astype(float)

    # Ensure end >= start
    df = df[df["end_date_int"] >= df["start_date_int"]]
    df = df.reset_index(drop=True)

    return df


# -------------------------
# Core metrics
# -------------------------

def _var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.var(x, ddof=1))


def _downside_deviation(r: np.ndarray, target: float = 0.0) -> float:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    d = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.mean(d * d)))


def _var_es(r: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    """Returns (VaR, ES) for losses, i.e., on -r."""
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return (float("nan"), float("nan"))
    losses = -r
    q = float(np.quantile(losses, alpha))
    tail = losses[losses >= q]
    es = float(np.mean(tail)) if tail.size else float("nan")
    return q, es


def _max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(e)
    dd = (peak - e)
    return float(np.max(dd))


def summarize_episodes(ep: pd.DataFrame, *, alpha: float = 0.95) -> Dict[str, float]:
    """Summaries over episodes. Expects columns in test_episodes.csv-like format."""
    out: Dict[str, float] = {}

    # Episode totals
    for col in ["pnl_net_sum", "cost_sum", "turnover_contract", "mdd", "reward_sum"]:
        if col in ep.columns:
            out[f"mean_{col}"] = float(np.nanmean(ep[col].to_numpy(dtype=float)))
            out[f"median_{col}"] = float(np.nanmedian(ep[col].to_numpy(dtype=float)))

    # Optional daily series if present
    if "daily_return" in ep.columns:
        r = ep["daily_return"].to_numpy(dtype=float)
        out["var_daily"] = _var(r)
        out["dd_daily"] = _downside_deviation(r, target=0.0)
        var95, es95 = _var_es(r, alpha=alpha)
        out["VaR95_loss"] = var95
        out["ES95_loss"] = es95

    return out


# -------------------------
# Baseline simulation on scenarios
# -------------------------

@dataclass
class Scenario:
    start_idx: int
    end_idx: int
    Q: float


def _build_index_map(dates_int: np.ndarray) -> Dict[int, int]:
    return {int(d): int(i) for i, d in enumerate(dates_int)}


def scenarios_to_indexed(pre: Precomputed, df_sc: pd.DataFrame) -> List[Scenario]:
    m = _build_index_map(pre.dates_int)
    out: List[Scenario] = []
    for row in df_sc.itertuples(index=False):
        s_int = int(getattr(row, "start_date_int"))
        e_int = int(getattr(row, "end_date_int"))
        Q = float(getattr(row, "volume_bbl"))
        if s_int not in m or e_int not in m:
            continue
        s_i = m[s_int]
        e_i = m[e_int]
        if e_i <= s_i:
            continue
        out.append(Scenario(start_idx=s_i, end_idx=e_i, Q=Q))
    return out


def _contracts_from_h(Q: float, h: float) -> int:
    # 1 CL contract = 1000 bbl
    return int(np.round((float(Q) / 1000.0) * float(h)))


def simulate_policy(
    pre: Precomputed,
    scenarios: List[Scenario],
    *,
    policy: str,
    naive_h: float = 1.0,
    dcc_h_series: Optional[pd.Series] = None,
    cost_per_contract_trade_usd: float = 10.0,
    lpm_target: float = 0.0,
    exposure_id: str = "",
    dataset: str = "scenario_file",
) -> pd.DataFrame:
    """Simulate a baseline policy over scenarios using precompute series.

    Returns episode-level dataframe with pnl/cost/turnover/mdd and simple daily return stats.

    policy:
      - unhedged: h=0
      - naive: h=naive_h
      - dcc: h_t from dcc_h_series indexed by dates_int

    Notes:
    - Costs here use the same simplified tick proxy: $10 per trade per contract.
      Roll costs are approximated by applying trade costs on roll days to the carried position.
      (If you want exact roll breakdown, use env_daily + cost_model; this is a fast evaluator.)
    - New: output includes strategy, dataset, exposure_id for comparison with RL.
    """

    if policy not in {"unhedged", "naive", "dcc"}:
        raise ValueError(f"Unknown policy: {policy}")

    # map date_int -> h_t for dcc
    if policy == "dcc":
        if dcc_h_series is None or dcc_h_series.empty:
            raise ValueError("dcc policy requested but dcc_h_series is missing/empty")
        dcc_map = {int(k): float(v) for k, v in dcc_h_series.items() if np.isfinite(v)}
    else:
        dcc_map = {}

    rows: List[Dict[str, Any]] = []

    for si, sc in enumerate(scenarios):
        s = sc.start_idx
        e = sc.end_idx
        Q = float(sc.Q)

        N_prev = 0
        equity = 0.0
        peak = 0.0
        mdd = 0.0
        turnover = 0.0
        cost_sum = 0.0
        pnl_net_sum = 0.0

        # daily returns (return-like) for risk metrics
        daily_r: List[float] = []

        # step from s+1..e inclusive (needs t-1)
        for t in range(s + 1, e + 1):
            date_int = int(pre.dates_int[t])

            # choose h
            if policy == "unhedged":
                h = 0.0
            elif policy == "naive":
                h = float(naive_h)
            else:
                h = float(dcc_map.get(date_int, float(naive_h)))

            N_new = _contracts_from_h(Q, h)

            # PnL components
            pnl_phys = Q * float(pre.dS[t])
            pnl_fut = int(N_prev) * float(pre.pnl_1c[t])

            # trading cost
            dN = abs(int(N_new) - int(N_prev))
            trade_cost = float(dN) * float(cost_per_contract_trade_usd)

            # roll cost proxy: if roll day, closing + opening of carried position
            roll_cost = 0.0
            if int(pre.roll_flag[t]) != 0:
                roll_cost = 2.0 * abs(int(N_prev)) * float(cost_per_contract_trade_usd)

            cost_t = trade_cost + roll_cost

            pnl_net = pnl_phys + pnl_fut - cost_t

            equity += pnl_net
            peak = max(peak, equity)
            mdd = max(mdd, peak - equity)

            turnover += abs(float(N_new) - float(N_prev))
            cost_sum += cost_t
            pnl_net_sum += pnl_net

            # return-like (normalize by notional)
            notional = max(Q * float(pre.spot[t - 1]), 1e-9)
            r_t = pnl_net / notional
            daily_r.append(r_t)

            N_prev = int(N_new)

        daily_r_arr = np.asarray(daily_r, dtype=np.float64)
        var95, es95 = _var_es(daily_r_arr, alpha=0.95)

        rows.append(
            {
                "scenario_id": si,
                "start_date_int": int(pre.dates_int[s]),
                "end_date_int": int(pre.dates_int[e]),
                "Q": Q,
                "policy": policy,
                "strategy": policy,
                "dataset": dataset,
                "exposure_id": exposure_id,
                "pnl_net_sum": float(pnl_net_sum),
                "cost_sum": float(cost_sum),
                "turnover_contract": float(turnover),
                "mdd": float(mdd),
                "mean_daily_return": float(np.mean(daily_r_arr)) if daily_r_arr.size else float("nan"),
                "var_daily": _var(daily_r_arr),
                "dd_daily": _downside_deviation(daily_r_arr, target=float(lpm_target)),
                "VaR95_loss": float(var95),
                "ES95_loss": float(es95),
            }
        )

    return pd.DataFrame(rows)


# -------------------------
# RL aggregation from run folders
# -------------------------


def collect_rl_windows(rl_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (windows_summary_df, episodes_df_concat).

    - windows_summary_df has one row per window (from window_summary.json if exists else val_summary.json)
    - episodes_df_concat concatenates test_episodes_all.parquet or test_episodes.csv across windows,
      or loads merged results_all_windows.parquet if present.
    - Ensures columns: dataset, strategy, exposure_id, window_name.
    """
    import warnings
    base = Path(rl_dir).expanduser()
    # allow relative paths from current working directory
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()

    if not base.exists():
        # provide helpful suggestions
        cwd = Path.cwd()
        candidates = sorted([p.name for p in cwd.iterdir() if p.is_dir() and (p.name.startswith("rl_runs") or p.name.startswith("runs") or p.name.startswith("WF_"))])
        hint = ""
        if candidates:
            hint = "\nAvailable run directories in current folder:\n  - " + "\n  - ".join(candidates[:30])
            if len(candidates) > 30:
                hint += f"\n  ... (+{len(candidates)-30} more)"
        raise FileNotFoundError(f"rl_dir not found: {str(base)}. You passed: '{rl_dir}'.{hint}")

    # First, check for merged results parquet
    merged_parquet = base / "results_all_windows.parquet"
    windows_summary_json = base / "window_summary.json"
    if merged_parquet.exists():
        # Load merged episodes
        ep_df = pd.read_parquet(merged_parquet)
        # Try to load window_summary.json for summary
        rows = []
        if windows_summary_json.exists():
            ws = _read_json(windows_summary_json)
            if isinstance(ws, list):
                rows.extend(ws)
            elif isinstance(ws, dict):
                rows.append(ws)
        else:
            # Try to build from unique window_names in ep_df
            for wname in sorted(set(ep_df["window_name"])) if "window_name" in ep_df.columns else []:
                rows.append({"window": wname, "path": ""})
        win_df = pd.DataFrame(rows)
        if win_df.empty:
            win_df = pd.DataFrame(columns=["window", "path"])
        # Ensure required columns
        for c, default in [
            ("dataset", "oracle_universe"),
            ("strategy", "RL_PPO"),
            ("exposure_id", ""),
            ("window_name", ""),
        ]:
            if c not in ep_df.columns:
                ep_df[c] = default
        return win_df, ep_df

    # Otherwise, iterate window folders
    rows = []
    eps_all = []
    for wdir in sorted([p for p in base.iterdir() if p.is_dir() and (p.name.startswith("WF_") or ("train" in p.name and "val" in p.name))]):
        wname = wdir.name
        summ_path = wdir / "window_summary.json"
        if summ_path.exists():
            summ = _read_json(summ_path)
        else:
            # fallback
            vs = wdir / "val_summary.json"
            summ = _read_json(vs) if vs.exists() else {}
        summ["window"] = wname
        summ["path"] = str(wdir)
        rows.append(summ)

        # Prefer test_episodes_all.parquet, fallback to test_episodes.csv
        ep_path_parquet = wdir / "test_episodes_all.parquet"
        ep_path_csv = wdir / "test_episodes.csv"
        ep = None
        if ep_path_parquet.exists():
            ep = pd.read_parquet(ep_path_parquet)
        elif ep_path_csv.exists():
            ep = pd.read_csv(ep_path_csv)
            # For legacy CSV, add required columns if missing
            if "dataset" not in ep.columns:
                ep["dataset"] = "oracle_universe"
            if "strategy" not in ep.columns:
                ep["strategy"] = "RL_PPO"
        if ep is not None:
            # Try to get exposure_id from window_summary.json if present
            exposure_id = ""
            if isinstance(summ, dict) and "exposure_id" in summ:
                exposure_id = summ["exposure_id"]
            if "exposure_id" not in ep.columns:
                ep["exposure_id"] = exposure_id
            else:
                ep["exposure_id"] = ep["exposure_id"].fillna(exposure_id)
            ep["window_name"] = wname
            eps_all.append(ep)
    win_df = pd.DataFrame(rows)
    ep_df = pd.concat(eps_all, ignore_index=True) if eps_all else pd.DataFrame()
    # Ensure required columns in episodes df
    for c, default in [
        ("dataset", "oracle_universe"),
        ("strategy", "RL_PPO"),
        ("exposure_id", ""),
        ("window_name", ""),
    ]:
        if c not in ep_df.columns:
            ep_df[c] = default
    return win_df, ep_df


# -------------------------
# Grouped episode summarization
# -------------------------

def summarize_grouped_episodes(ep_df: pd.DataFrame, *, alpha: float = 0.95) -> pd.DataFrame:
    """Summarize episode-level outputs grouped by exposure_id/strategy/dataset (and window_name if present)."""
    if ep_df is None or ep_df.empty:
        return pd.DataFrame()

    # Ensure numeric columns exist
    for c in ["pnl_net_sum", "cost_sum", "turnover_contract", "mdd", "reward_sum"]:
        if c in ep_df.columns:
            ep_df[c] = pd.to_numeric(ep_df[c], errors="coerce")

    group_cols = [c for c in ["exposure_id", "strategy", "dataset", "window_name"] if c in ep_df.columns]
    if not group_cols:
        group_cols = ["strategy"]

    rows = []
    for keys, g in ep_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        d = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        d["n_episodes"] = int(len(g))
        for col in ["pnl_net_sum", "cost_sum", "turnover_contract", "mdd", "reward_sum"]:
            if col in g.columns:
                arr = g[col].to_numpy(dtype=float)
                d[f"mean_{col}"] = float(np.nanmean(arr))
                d[f"median_{col}"] = float(np.nanmedian(arr))
        rows.append(d)

    return pd.DataFrame(rows)


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RL hedging runs vs baselines")

    p.add_argument("--rl_dir", type=str, default="", help="Folder containing RL outputs: per-window test_episodes_all.parquet or test_episodes.csv, or run-level results_all_windows.parquet")

    p.add_argument("--cache", type=str, default="", help="precompute_*.npz for baseline simulation")
    p.add_argument("--scenario_file", type=str, default="", help="Scenario file (parquet/npz/csv) for baseline simulation")

    p.add_argument(
        "--policy",
        action="append",
        default=[],
        help="Policy to simulate: unhedged | naive | dcc. Can be repeated.",
    )
    p.add_argument("--naive_h", type=float, default=1.0, help="Fixed hedge ratio for naive")

    p.add_argument("--master", type=str, default="", help="MasterData.csv (needed for dcc h_t series)")
    p.add_argument("--dcc_col", type=str, default="", help="Column name in master containing DCC hedge ratio series")

    p.add_argument("--cost_per_contract_trade_usd", type=float, default=10.0)
    p.add_argument("--lpm_target", type=float, default=0.0)

    p.add_argument("--alpha", type=float, default=0.95, help="VaR/ES confidence level")

    p.add_argument("--out_dir", type=str, required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_win_df = None
    rl_ep_df = None
    baseline_episodes_list = []
    baseline_policy_names = []

    # 1) RL aggregation
    if args.rl_dir:
        rl_win_df, rl_ep_df = collect_rl_windows(args.rl_dir)
        rl_win_df.to_csv(out_dir / "summary_windows.csv", index=False)

        overall = {}
        if rl_ep_df is not None and not rl_ep_df.empty:
            overall = summarize_episodes(rl_ep_df, alpha=float(args.alpha))
        else:
            overall["note"] = "No RL episode data found (episodes dataframe is empty)."
        # also include mean window metrics if present
        if "val_mean_reward" in rl_win_df.columns:
            overall["mean_val_mean_reward_over_windows"] = float(
                np.nanmean(pd.to_numeric(rl_win_df["val_mean_reward"], errors="coerce").to_numpy())
            )
        if "test_pnl_mean" in rl_win_df.columns:
            overall["mean_test_pnl_mean_over_windows"] = float(
                np.nanmean(pd.to_numeric(rl_win_df["test_pnl_mean"], errors="coerce").to_numpy())
            )
        pd.DataFrame([overall]).to_csv(out_dir / "summary_overall.csv", index=False)

        if rl_ep_df is not None and not rl_ep_df.empty:
            rl_ep_df.to_csv(out_dir / "episodes_RL_concat.csv", index=False)

        print(f"[eval] RL windows: {len(rl_win_df)} | episodes rows: {len(rl_ep_df) if rl_ep_df is not None else 0}")
        print(f"[eval] wrote: {out_dir / 'summary_windows.csv'}")

    # 2) Baseline simulation on a scenario file
    if args.cache and args.scenario_file and args.policy:
        pre = load_precompute_npz(args.cache)
        sc_df = load_scenarios(args.scenario_file)
        sc = scenarios_to_indexed(pre, sc_df)
        if not sc:
            raise RuntimeError("No scenarios matched the cache dates. Check date_int formats.")

        # dcc series
        dcc_series = None
        if "dcc" in set(args.policy):
            if not args.master or not args.dcc_col:
                print("[eval] WARNING: dcc policy requested but --master/--dcc_col not provided. Skipping dcc.")
            else:
                mdf = pd.read_csv(args.master)
                if "Date" in mdf.columns:
                    # assume same date_int format as cache (e.g., YYYYMMDD int or similar)
                    # If Date is like '1986/1/2', try to parse.
                    if np.issubdtype(mdf["Date"].dtype, np.number):
                        mdf["date_int"] = mdf["Date"].astype(int)
                    else:
                        dt = pd.to_datetime(mdf["Date"], errors="coerce")
                        # int format yyyyMMdd
                        mdf["date_int"] = dt.dt.strftime("%Y%m%d").astype(float).astype("Int64")
                else:
                    raise ValueError("MasterData must have a 'Date' column")

                if args.dcc_col not in mdf.columns:
                    print(
                        f"[eval] WARNING: dcc_col '{args.dcc_col}' not found in master. Available: {list(mdf.columns)[:30]}..."
                    )
                else:
                    s = pd.Series(mdf[args.dcc_col].astype(float).to_numpy(), index=mdf["date_int"].astype(int).to_numpy())
                    dcc_series = s

        # Run policies
        summaries = []
        for pol in args.policy:
            if pol == "dcc" and dcc_series is None:
                continue
            ep = simulate_policy(
                pre,
                sc,
                policy=pol,
                naive_h=float(args.naive_h),
                dcc_h_series=dcc_series,
                cost_per_contract_trade_usd=float(args.cost_per_contract_trade_usd),
                lpm_target=float(args.lpm_target),
                exposure_id="",
                dataset="scenario_file",
            )
            ep["window_name"] = "GLOBAL"
            ep_path = out_dir / f"episodes_{pol}.csv"
            ep.to_csv(ep_path, index=False)
            baseline_episodes_list.append(ep)
            baseline_policy_names.append(pol)
            summ = {
                "policy": pol,
                "n_scenarios": int(len(ep)),
                "mean_pnl_net_sum": float(np.nanmean(ep["pnl_net_sum"])),
                "mean_cost_sum": float(np.nanmean(ep["cost_sum"])),
                "mean_turnover_contract": float(np.nanmean(ep["turnover_contract"])),
                "mean_mdd": float(np.nanmean(ep["mdd"])),
                "mean_var_daily": float(np.nanmean(ep["var_daily"])),
                "mean_dd_daily": float(np.nanmean(ep["dd_daily"])),
                "mean_VaR95_loss": float(np.nanmean(ep["VaR95_loss"])),
                "mean_ES95_loss": float(np.nanmean(ep["ES95_loss"])),
            }
            summaries.append(summ)
            print(f"[eval] simulated policy={pol} scenarios={len(ep)} -> {ep_path}")

        if summaries:
            pd.DataFrame(summaries).to_csv(out_dir / "baseline_policy_summary.csv", index=False)
            print(f"[eval] wrote: {out_dir / 'baseline_policy_summary.csv'}")

    # 3) Comparison report
    # Combine RL and baseline episodes, summarize, and pivot
    compare_cols = [
        "exposure_id", "strategy", "dataset", "window_name",
        "pnl_net_sum", "cost_sum", "turnover_contract", "mdd", "reward_sum"
    ]
    combined_ep_df = pd.DataFrame()
    rl_ep_df_selected = None
    if rl_ep_df is not None and not rl_ep_df.empty:
        rl_ep_df_selected = rl_ep_df[[c for c in compare_cols if c in rl_ep_df.columns]].copy()
        combined_ep_df = rl_ep_df_selected.copy()
    if baseline_episodes_list:
        for bepdf in baseline_episodes_list:
            bepdf_selected = bepdf[[c for c in compare_cols if c in bepdf.columns]].copy()
            bepdf_selected["window_name"] = "GLOBAL"
            combined_ep_df = pd.concat([combined_ep_df, bepdf_selected], ignore_index=True)
    if not combined_ep_df.empty:
        grouped_summary = summarize_grouped_episodes(combined_ep_df, alpha=float(args.alpha))
        grouped_summary.to_csv(out_dir / "summary_grouped.csv", index=False)
        print(f"[eval] wrote: {out_dir / 'summary_grouped.csv'}")
        # Pivoted strategy comparison if possible
        strategies = set(combined_ep_df["strategy"].unique())
        if (
            "RL_PPO" in strategies
            and any(s in strategies for s in {"naive", "dcc", "unhedged"})
            and "exposure_id" in combined_ep_df.columns
            and "dataset" in combined_ep_df.columns
        ):
            pivot_cols = ["exposure_id", "dataset"]
            value_cols = ["mean_pnl_net_sum", "mean_mdd", "mean_cost_sum"]
            # Compute mean per (exposure_id, strategy, dataset)
            gdf = summarize_grouped_episodes(combined_ep_df, alpha=float(args.alpha))
            # Only keep strategies of interest
            gdf = gdf[gdf["strategy"].isin(["RL_PPO", "naive", "dcc", "unhedged"])]
            if gdf.empty:
                print("[eval] Not enough data to build compare_strategies.csv (grouped summary empty).")
                return
            # Pivot
            pivot = pd.pivot_table(
                gdf,
                index=pivot_cols,
                columns="strategy",
                values=[c for c in gdf.columns if c.startswith("mean_")],
                aggfunc="first",
            )
            # Flatten columns
            pivot.columns = ["%s_%s" % (col[1], col[0]) for col in pivot.columns]
            pivot = pivot.reset_index()
            pivot.to_csv(out_dir / "compare_strategies.csv", index=False)
            print(f"[eval] wrote: {out_dir / 'compare_strategies.csv'}")

    if not (args.rl_dir or (args.cache and args.scenario_file and args.policy)):
        print("[eval] Nothing to do. Provide --rl_dir and/or --cache + --scenario_file + --policy.")


if __name__ == "__main__":
    main()