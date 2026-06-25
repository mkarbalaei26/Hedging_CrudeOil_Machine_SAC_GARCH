

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel walk-forward runner for SACPortfolioLPMEnv.

This script keeps the current portfolio-LPM reward unchanged. Its job is to run the
same SAC-LPM model across multiple expanding walk-forward windows, optionally in
parallel, and export files that can be consumed by ``finalreport.py``.

Main outputs per run:
    <out_dir>/results_all_windows.parquet      # finalreport-compatible RL rows
    <out_dir>/results_all_windows.csv
    <out_dir>/daily_logs_all_windows.parquet   # daily decision logs
    <out_dir>/window_XX/...                    # model, logs, summaries per window
    <out_dir>/dashboard_summary.csv
    <out_dir>/run_config.json

Example, generated rolling scenarios over all OPEC data:
    python -m rl.SAC_Walkforard_LPM \
      --asset OPEC \
      --timesteps 100000 \
      --n-windows 10 \
      --max-parallel 10 \
      --episode-len 30 \
      --stride 5

Example, scenario-root mode for comparison with finalreport scenario metadata:
    python -m rl.SAC_Walkforard_LPM \
      --asset OPEC \
      --scenario-root scenarios \
      --scenario-kinds oracle_universe,oracle_all,baseline \
      --timesteps 100000 \
      --n-windows 10 \
      --max-parallel 10

Notes:
- ``dates_int`` in precompute files are Python ordinal integers.
- The report script recognizes RL files through columns such as net_pnl_total,
  turnover_h, turnover_contracts, mdd_equity, scenario_id, exposure_id, window,
  mode, and roll.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as exc:  # pragma: no cover
    raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3") from exc

from rl.SACPortfolioLPMEnv import SACPortfolioLPMConfig, SACPortfolioLPMEnv
from rl.train_sac_portfolio_lpm import (
    ASSET_TO_FILE,
    CACHE_DIR,
    compute_episode_summary,
    filter_feature_matrix,
    load_precompute,
    make_rolling_scenarios,
    plot_sample_episode,
    run_policy_on_scenarios,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "rl_runs" / "SAC_LPM_WALKFORWARD"

ASSET_TO_EXPOSURE_ID = {
    "WTI": "WTI_SPOT",
    "BRENT": "BRENT_SPOT",
    "OPEC": "OPEC_BASKET",
}

SCENARIO_FILES_BY_KIND = {
    "baseline": ["baseline.parquet", "scenarios_baseline.parquet"],
    "company": ["companies.parquet", "company.parquet", "scenarios_company.parquet"],
    "oracle_universe": ["oracle_universe.parquet"],
    "oracle_all": ["oracle_all.parquet"],
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[WARN] parquet write failed for {path}: {exc}. Wrote CSV: {csv_path}")


def ordinal_to_timestamp(x: Any) -> pd.Timestamp:
    try:
        return pd.Timestamp.fromordinal(int(x))
    except Exception:
        return pd.NaT


def date_to_ordinal(x: Any) -> Optional[int]:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return None
    return int(ts.to_pydatetime().date().toordinal())


def window_name(w: int) -> str:
    return f"w{int(w):02d}"


def set_process_threads(n: int) -> None:
    n = max(1, int(n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    if torch is not None:
        try:
            torch.set_num_threads(n)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Scenario loading
# -----------------------------------------------------------------------------


def read_scenario_file(path: Path, exposure_id: str, kind: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "scenario_id" not in df.columns:
        return pd.DataFrame()
    if "start_date" not in df.columns or "end_date" not in df.columns:
        return pd.DataFrame()
    if "volume_bbl" not in df.columns:
        df["volume_bbl"] = 1_000_000.0
    if "exposure_id" not in df.columns:
        df["exposure_id"] = exposure_id
    if "scenario_kind" not in df.columns:
        df["scenario_kind"] = kind
    df["scenario_kind"] = df["scenario_kind"].fillna(kind).astype(str)
    df["exposure_id"] = df["exposure_id"].fillna(exposure_id).astype(str)
    df["scenario_id"] = df["scenario_id"].astype(str)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["volume_bbl"] = pd.to_numeric(df["volume_bbl"], errors="coerce").fillna(1_000_000.0)
    df["scenario_source_file"] = str(path)
    return df


def load_scenarios_from_root(
    scenario_root: Path,
    *,
    exposure_id: str,
    kinds: Sequence[str],
    dates_int: np.ndarray,
) -> pd.DataFrame:
    asset_dir = scenario_root / exposure_id
    if not asset_dir.exists():
        raise FileNotFoundError(f"Scenario asset directory not found: {asset_dir}")

    frames: List[pd.DataFrame] = []
    for kind in kinds:
        kind = str(kind).strip()
        if not kind:
            continue
        if kind not in SCENARIO_FILES_BY_KIND:
            raise ValueError(f"Unknown scenario kind {kind!r}. Valid: {sorted(SCENARIO_FILES_BY_KIND)}")
        for filename in SCENARIO_FILES_BY_KIND[kind]:
            p = asset_dir / filename
            if p.exists():
                d = read_scenario_file(p, exposure_id=exposure_id, kind=kind)
                if not d.empty:
                    frames.append(d)
                break

    if not frames:
        raise ValueError(f"No scenario files found under {asset_dir} for kinds={kinds}")

    meta = pd.concat(frames, ignore_index=True)
    meta["start_ord"] = meta["start_date"].map(date_to_ordinal)
    meta["end_ord"] = meta["end_date"].map(date_to_ordinal)
    meta = meta.dropna(subset=["start_ord", "end_ord"]).copy()
    meta["start_ord"] = meta["start_ord"].astype(int)
    meta["end_ord"] = meta["end_ord"].astype(int)

    # Convert dates to precompute index windows.
    meta["start_idx"] = np.searchsorted(dates_int, meta["start_ord"].to_numpy(), side="left")
    meta["end_idx"] = np.searchsorted(dates_int, meta["end_ord"].to_numpy(), side="left")
    meta["start_idx"] = meta["start_idx"].clip(0, len(dates_int) - 1).astype(int)
    meta["end_idx"] = meta["end_idx"].clip(1, len(dates_int)).astype(int)
    meta = meta[meta["end_idx"] > meta["start_idx"] + 1].copy()

    # Keep only scenarios that intersect the available precompute date range.
    meta = meta.drop_duplicates(subset=["scenario_id", "exposure_id", "scenario_kind"], keep="first")
    meta = meta.sort_values(["start_idx", "end_idx", "scenario_id"]).reset_index(drop=True)
    if meta.empty:
        raise ValueError("Scenario metadata became empty after date/index filtering")
    return meta


def generated_rolling_scenario_meta(
    dates_int: np.ndarray,
    tradable: np.ndarray,
    *,
    exposure_id: str,
    episode_len: int,
    stride: int,
    volume_bbl: float,
) -> pd.DataFrame:
    scenarios = make_rolling_scenarios(
        dates_int,
        tradable,
        start_idx=0,
        end_idx=len(dates_int),
        episode_len=episode_len,
        stride=stride,
        volume_bbl=volume_bbl,
    )
    rows = []
    for i, s in enumerate(scenarios):
        start_idx = int(s["start_idx"])
        end_idx = int(s["end_idx"])
        rows.append({
            "scenario_id": f"RLGEN_{exposure_id}_{start_idx}_{end_idx}",
            "exposure_id": exposure_id,
            "scenario_kind": "oracle_universe",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_date": ordinal_to_timestamp(dates_int[start_idx]),
            "end_date": ordinal_to_timestamp(dates_int[min(end_idx - 1, len(dates_int) - 1)]),
            "volume_bbl": float(s["volume_bbl"]),
            "scenario_record_id": i,
        })
    return pd.DataFrame(rows)


def scenario_rows_to_env_scenarios(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert scenario metadata rows to env scenarios while preserving IDs.

    The environment itself only consumes start_idx/end_idx/volume_bbl, but the
    evaluator reads scenario_id/scenario_kind/exposure_id/scenario_record_id and
    any extra metadata from this dict. Therefore we preserve all scalar columns
    instead of rebuilding synthetic IDs.
    """
    out: List[Dict[str, Any]] = []
    for pos, (_, r) in enumerate(df.iterrows()):
        item: Dict[str, Any] = {}
        for col, val in r.items():
            if isinstance(val, pd.Timestamp):
                item[col] = val.isoformat()
            elif pd.isna(val) if not isinstance(val, (list, tuple, dict, np.ndarray)) else False:
                item[col] = None
            else:
                item[col] = val.item() if hasattr(val, "item") else val

        item["start_idx"] = int(r["start_idx"])
        item["end_idx"] = int(r["end_idx"])
        item["volume_bbl"] = float(r["volume_bbl"])
        item["scenario_id"] = str(r.get("scenario_id", f"GEN_{item['start_idx']}_{item['end_idx']}"))
        item["scenario_kind"] = str(r.get("scenario_kind", "oracle_universe"))
        item["exposure_id"] = str(r.get("exposure_id", "unknown"))
        item["scenario_record_id"] = r.get("scenario_record_id", pos)
        out.append(item)
    return out


# -----------------------------------------------------------------------------
# Walk-forward split construction
# -----------------------------------------------------------------------------


def build_walkforward_windows(
    scenario_meta: pd.DataFrame,
    *,
    n_total_dates: int,
    n_windows: int,
    min_train_frac: float,
    val_frac_of_train: float,
    min_train_scenarios: int,
) -> List[Dict[str, Any]]:
    """Create expanding train/validation/test windows.

    Test periods are contiguous chunks from min_train_frac*T to T. Training always
    uses scenarios before the validation block, and validation uses the block right
    before the test block. Splitting is done by scenario start_idx to avoid mixing
    future scenarios into earlier train windows.
    """
    n_windows = max(1, int(n_windows))
    min_start = int(np.clip(round(float(min_train_frac) * n_total_dates), 1, n_total_dates - 2))
    cuts = np.linspace(min_start, n_total_dates, n_windows + 1).round().astype(int)
    cuts = np.unique(cuts)
    if len(cuts) < 2:
        raise ValueError("Not enough date range to construct walk-forward windows")

    windows: List[Dict[str, Any]] = []
    for j in range(len(cuts) - 1):
        test_start, test_end = int(cuts[j]), int(cuts[j + 1])
        if test_end <= test_start + 1:
            continue
        val_len = max(30, int(round(float(val_frac_of_train) * max(test_start, 1))))
        val_start = max(0, test_start - val_len)
        train_end = val_start

        train_df = scenario_meta[scenario_meta["start_idx"] < train_end].copy()
        val_df = scenario_meta[(scenario_meta["start_idx"] >= val_start) & (scenario_meta["start_idx"] < test_start)].copy()
        test_df = scenario_meta[(scenario_meta["start_idx"] >= test_start) & (scenario_meta["start_idx"] < test_end)].copy()

        if len(train_df) < int(min_train_scenarios) or val_df.empty or test_df.empty:
            continue

        windows.append({
            "window": int(j),
            "train_start_idx": 0,
            "train_end_idx": int(train_end),
            "val_start_idx": int(val_start),
            "val_end_idx": int(test_start),
            "test_start_idx": int(test_start),
            "test_end_idx": int(test_end),
            "train_meta": train_df.reset_index(drop=True),
            "val_meta": val_df.reset_index(drop=True),
            "test_meta": test_df.reset_index(drop=True),
        })
    if not windows:
        raise ValueError("No valid walk-forward windows were built. Reduce n_windows/min_train_scenarios or check scenarios.")
    return windows


# -----------------------------------------------------------------------------
# Training/evaluation worker
# -----------------------------------------------------------------------------


def make_env(pre_dict: Dict[str, np.ndarray], scenarios: Sequence[Dict[str, Any]], cfg: SACPortfolioLPMConfig, seed: int):
    def _make():
        env = SACPortfolioLPMEnv(pre_dict, scenarios, cfg=cfg, seed=seed)
        return Monitor(env)
    return DummyVecEnv([_make])


def add_metadata_to_eval_outputs(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    test_meta: pd.DataFrame,
    *,
    exposure_id: str,
    window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach scenario IDs and build finalreport-compatible episode rows."""
    meta = test_meta.reset_index(drop=True).copy()
    meta["eval_scenario_order"] = np.arange(len(meta), dtype=int)
    # Preserve all scenario metadata columns, not only a small whitelist. This is
    # important because finalreport and later diagnostics may rely on columns such
    # as scenario_id, company_id, oracle settings, hold_days, source tags, etc.
    keep_meta = list(meta.columns)
    m = meta[keep_meta].copy()

    if not daily.empty:
        daily = daily.merge(m, on="eval_scenario_order", how="left", suffixes=("", "_meta"))
        daily["window"] = int(window)
        daily["exposure_id"] = daily.get("exposure_id", exposure_id).fillna(exposure_id)
        daily["scenario_kind"] = daily.get("scenario_kind", "oracle_universe").fillna("oracle_universe")
        for base_col in ["scenario_id", "scenario_kind", "exposure_id", "scenario_record_id"]:
            meta_col = f"{base_col}_meta"
            if meta_col in daily.columns:
                if base_col not in daily.columns:
                    daily[base_col] = daily[meta_col]
                else:
                    daily[base_col] = daily[base_col].where(daily[base_col].notna(), daily[meta_col])

    if not summary.empty:
        summary = summary.merge(m, on="eval_scenario_order", how="left", suffixes=("", "_meta"))
        summary["window"] = int(window)
        summary["exposure_id"] = summary.get("exposure_id", exposure_id).fillna(exposure_id)
        summary["scenario_kind"] = summary.get("scenario_kind", "oracle_universe").fillna("oracle_universe")
        for base_col in ["scenario_id", "scenario_kind", "exposure_id", "scenario_record_id"]:
            meta_col = f"{base_col}_meta"
            if meta_col in summary.columns:
                if base_col not in summary.columns:
                    summary[base_col] = summary[meta_col]
                else:
                    summary[base_col] = summary[base_col].where(summary[base_col].notna(), summary[meta_col])

    # Finalreport-compatible rows.
    if summary.empty:
        results = pd.DataFrame()
    else:
        results = pd.DataFrame({
            "scenario_id": summary["scenario_id"].astype(str),
            "exposure_id": summary["exposure_id"].astype(str),
            "scenario_kind": summary["scenario_kind"].astype(str),
            "strategy": "RL_SAC_LPM",
            "window": int(window),
            "mode": "dynamic",
            "roll": "roll",
            "dynamic": 1,
            "start_date": pd.to_datetime(summary.get("start_date", pd.NaT), errors="coerce"),
            "end_date": pd.to_datetime(summary.get("end_date", pd.NaT), errors="coerce"),
            "volume_bbl": pd.to_numeric(summary.get("volume_bbl", np.nan), errors="coerce"),
            "spot_pnl_total": pd.to_numeric(summary["physical_pnl"], errors="coerce"),
            "fut_pnl_total": pd.to_numeric(summary["futures_pnl"], errors="coerce"),
            "cost_trade_total": pd.to_numeric(summary["decision_cost"], errors="coerce"),
            "cost_roll_total": pd.to_numeric(summary["roll_accounting_cost"], errors="coerce"),
            "net_pnl_total": pd.to_numeric(summary["total_pnl"], errors="coerce"),
            "turnover_contracts": pd.to_numeric(summary["turnover_contracts"], errors="coerce"),
            "turnover_h": pd.to_numeric(summary["turnover_h"], errors="coerce"),
            "trade_contracts": pd.to_numeric(summary["turnover_contracts"], errors="coerce"),
            "roll_contracts": np.nan,
            "max_abs_contracts": np.nan,
            "mdd_equity": pd.to_numeric(summary["mdd"], errors="coerce"),
            "mean_h": pd.to_numeric(summary["mean_h"], errors="coerce"),
            "min_h": pd.to_numeric(summary["min_h"], errors="coerce"),
            "max_h": pd.to_numeric(summary["max_h"], errors="coerce"),
            "no_hedge_pnl": pd.to_numeric(summary.get("no_hedge_pnl", np.nan), errors="coerce"),
            "naive_pnl": pd.to_numeric(summary.get("naive_pnl", np.nan), errors="coerce"),
            "steps": pd.to_numeric(summary.get("steps", np.nan), errors="coerce"),
        })
        # Preserve optional scenario metadata for finalreport joins/plots.
        for col in ["oracle_series", "oracle_pool", "oracle_freq", "label", "tag", "scenario_record_id"]:
            if col in summary.columns:
                results[col] = summary[col]

    return daily, summary, results


def train_one_window(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function. Loads data inside the process to avoid pickling large arrays."""
    start_time = time.time()
    window = int(task["window"])
    seed = int(task["seed"]) + window * 1009
    set_process_threads(int(task["torch_threads"]))

    out_dir = Path(task["out_dir"])
    wdir = ensure_dir(out_dir / window_name(window))

    try:
        raw_pre, pre_path = load_precompute(task["asset"])
        pre = filter_feature_matrix(raw_pre, task["feature_mode"])

        train_meta = pd.DataFrame(task["train_meta"])
        val_meta = pd.DataFrame(task["val_meta"])
        test_meta = pd.DataFrame(task["test_meta"])
        train_scenarios = scenario_rows_to_env_scenarios(train_meta)
        val_scenarios = scenario_rows_to_env_scenarios(val_meta)
        test_scenarios = scenario_rows_to_env_scenarios(test_meta)

        cfg = SACPortfolioLPMConfig(
            initial_h=1.0,
            h_min=float(task["h_min"]),
            h_max=float(task["h_max"]),
            delta_h_bounds=(-float(task["delta_h"]), float(task["delta_h"])),
            lambda_lpm=float(task["lambda_lpm"]),
            eta_decision_cost=float(task["eta_cost"]),
            lambda_smooth=float(task["lambda_smooth"]),
            reward_scale=float(task["reward_scale"]),
            reward_clip=float(task["reward_clip"]) if task["reward_clip"] is not None else None,
            info_mode="train",
        )

        env = make_env(pre, train_scenarios, cfg, seed=seed)
        net_arch = [int(x) for x in str(task["net_arch"]).split(",") if str(x).strip()]
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=float(task["learning_rate"]),
            buffer_size=int(task["buffer_size"]),
            batch_size=int(task["batch_size"]),
            gamma=float(task["gamma"]),
            tau=float(task["tau"]),
            learning_starts=int(task["learning_starts"]),
            policy_kwargs={"net_arch": net_arch},
            verbose=0,
            seed=seed,
            tensorboard_log=str(wdir / "tb"),
        )
        model.learn(total_timesteps=int(task["timesteps"]), progress_bar=False)
        model.save(wdir / "model.zip")

        # Evaluate all test scenarios by default. Validation is exported for diagnostics.
        val_log = run_policy_on_scenarios(
            model, pre, val_scenarios, cfg, deterministic=True, max_episodes=None
        )
        test_log = run_policy_on_scenarios(
            model, pre, test_scenarios, cfg, deterministic=True, max_episodes=None
        )
        val_summary = compute_episode_summary(val_log)
        test_summary = compute_episode_summary(test_log)

        val_log, val_summary, _ = add_metadata_to_eval_outputs(
            val_log, val_summary, val_meta, exposure_id=task["exposure_id"], window=window
        )
        test_log, test_summary, results = add_metadata_to_eval_outputs(
            test_log, test_summary, test_meta, exposure_id=task["exposure_id"], window=window
        )

        safe_to_parquet(val_log, wdir / "val_daily_log.parquet")
        safe_to_parquet(test_log, wdir / "test_daily_log.parquet")
        safe_to_parquet(val_summary, wdir / "val_episode_summary.parquet")
        safe_to_parquet(test_summary, wdir / "test_episode_summary.parquet")
        safe_to_parquet(results, wdir / "results.parquet")
        val_log.to_csv(wdir / "val_daily_log.csv", index=False)
        test_log.to_csv(wdir / "test_daily_log.csv", index=False)
        test_summary.to_csv(wdir / "test_episode_summary.csv", index=False)
        results.to_csv(wdir / "results.csv", index=False)

        if not test_log.empty:
            plot_sample_episode(test_log, wdir / "sample_test_episode_0.png", scenario_order=0)

        metric = {
            "window": window,
            "status": "ok",
            "seed": seed,
            "train_n": len(train_scenarios),
            "val_n": len(val_scenarios),
            "test_n": len(test_scenarios),
            "mean_net_pnl": float(results["net_pnl_total"].mean()) if not results.empty else np.nan,
            "median_net_pnl": float(results["net_pnl_total"].median()) if not results.empty else np.nan,
            "mean_no_hedge_pnl": float(results["no_hedge_pnl"].mean()) if not results.empty and "no_hedge_pnl" in results else np.nan,
            "mean_naive_pnl": float(results["naive_pnl"].mean()) if not results.empty and "naive_pnl" in results else np.nan,
            "mean_turnover_h": float(results["turnover_h"].mean()) if not results.empty else np.nan,
            "mean_mdd": float(results["mdd_equity"].mean()) if not results.empty else np.nan,
            "duration_sec": float(time.time() - start_time),
            "window_dir": str(wdir),
        }
        with open(wdir / "window_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metric, f, ensure_ascii=False, indent=2)
        return metric

    except Exception as exc:
        err = {
            "window": window,
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "duration_sec": float(time.time() - start_time),
            "window_dir": str(wdir),
        }
        with open(wdir / "ERROR.json", "w", encoding="utf-8") as f:
            json.dump(err, f, ensure_ascii=False, indent=2)
        return err


# -----------------------------------------------------------------------------
# Dashboard / orchestration
# -----------------------------------------------------------------------------


def run_parallel(tasks: List[Dict[str, Any]], max_parallel: int) -> List[Dict[str, Any]]:
    max_parallel = max(1, min(int(max_parallel), len(tasks)))
    results: List[Dict[str, Any]] = []

    try:
        from rich.console import Console
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
        from rich.table import Table
        from rich.live import Live

        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        task_id = progress.add_task("SAC-LPM walk-forward", total=len(tasks))

        def make_table() -> Table:
            t = Table(title="SAC-LPM Walk-Forward Dashboard")
            for col in ["window", "status", "test_n", "mean_net_pnl", "mean_naive_pnl", "mean_turnover_h", "duration_sec"]:
                t.add_column(col)
            for r in sorted(results, key=lambda x: int(x.get("window", -1))):
                t.add_row(
                    str(r.get("window", "")),
                    str(r.get("status", "")),
                    str(r.get("test_n", "")),
                    f"{float(r.get('mean_net_pnl', np.nan)):,.0f}" if r.get("status") == "ok" else str(r.get("error", ""))[:50],
                    f"{float(r.get('mean_naive_pnl', np.nan)):,.0f}" if r.get("status") == "ok" else "",
                    f"{float(r.get('mean_turnover_h', np.nan)):,.3f}" if r.get("status") == "ok" else "",
                    f"{float(r.get('duration_sec', np.nan)):.1f}",
                )
            return t

        with ProcessPoolExecutor(max_workers=max_parallel) as ex:
            futures = [ex.submit(train_one_window, t) for t in tasks]
            with Live(make_table(), console=console, refresh_per_second=1) as live:
                progress.start()
                for fut in as_completed(futures):
                    r = fut.result()
                    results.append(r)
                    progress.advance(task_id)
                    live.update(make_table())
                progress.stop()
        return results

    except Exception:
        print(f"[{now_str()}] rich dashboard unavailable; using plain progress.")
        with ProcessPoolExecutor(max_workers=max_parallel) as ex:
            futures = [ex.submit(train_one_window, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                print(f"[{i}/{len(tasks)}] window={r.get('window')} status={r.get('status')} mean_pnl={r.get('mean_net_pnl')}")
        return results


def collect_outputs(out_dir: Path, metrics: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    result_frames: List[pd.DataFrame] = []
    daily_frames: List[pd.DataFrame] = []
    for r in metrics:
        if r.get("status") != "ok":
            continue
        wdir = Path(str(r["window_dir"]))
        rp = wdir / "results.parquet"
        dp = wdir / "test_daily_log.parquet"
        try:
            if rp.exists():
                result_frames.append(pd.read_parquet(rp))
            elif (wdir / "results.csv").exists():
                result_frames.append(pd.read_csv(wdir / "results.csv"))
        except Exception as exc:
            print(f"[WARN] cannot collect results from {wdir}: {exc}")
        try:
            if dp.exists():
                daily_frames.append(pd.read_parquet(dp))
            elif (wdir / "test_daily_log.csv").exists():
                daily_frames.append(pd.read_csv(wdir / "test_daily_log.csv"))
        except Exception as exc:
            print(f"[WARN] cannot collect daily log from {wdir}: {exc}")

    all_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    all_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()

    if not all_results.empty:
        safe_to_parquet(all_results, out_dir / "results_all_windows.parquet")
        all_results.to_csv(out_dir / "results_all_windows.csv", index=False)
    if not all_daily.empty:
        safe_to_parquet(all_daily, out_dir / "daily_logs_all_windows.parquet")
        all_daily.to_csv(out_dir / "daily_logs_all_windows.csv", index=False)
    return all_results, all_daily


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel walk-forward SAC-LPM runner.")
    p.add_argument("--asset", choices=sorted(ASSET_TO_FILE), default="OPEC")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--scenario-root",
        type=Path,
        default=PROJECT_ROOT / "scenarios",
        help="Scenario root folder. Default uses project scenarios/. Pass --generated-scenarios to ignore it.",
    )
    p.add_argument(
        "--generated-scenarios",
        action="store_true",
        help="Ignore --scenario-root and generate rolling scenarios from the precompute date range.",
    )
    p.add_argument("--scenario-kinds", type=str, default="oracle_universe,oracle_all,baseline")
    p.add_argument("--feature-mode", choices=["all", "core_no_nan"], default="core_no_nan")

    # Walk-forward setup
    p.add_argument("--n-windows", type=int, default=10)
    p.add_argument("--max-parallel", type=int, default=10)
    p.add_argument("--min-train-frac", type=float, default=0.40)
    p.add_argument("--val-frac-of-train", type=float, default=0.15)
    p.add_argument("--min-train-scenarios", type=int, default=50)

    # Generated rolling scenarios, used only if --scenario-root is omitted
    p.add_argument("--episode-len", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--volume-bbl", type=float, default=1_000_000.0)

    # Environment reward/action parameters; reward formulation is unchanged
    p.add_argument("--h-min", type=float, default=0.0)
    p.add_argument("--h-max", type=float, default=1.5)
    p.add_argument("--delta-h", type=float, default=0.20)
    p.add_argument("--lambda-lpm", type=float, default=1.0)
    p.add_argument("--eta-cost", type=float, default=1.0)
    p.add_argument("--lambda-smooth", type=float, default=0.01)
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--reward-clip", type=float, default=50.0)

    # SAC hyperparameters
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--net-arch", type=str, default="128,128")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-threads", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asset = args.asset.upper()
    exposure_id = ASSET_TO_EXPOSURE_ID[asset]
    out_dir = ensure_dir(Path(args.out_dir))

    raw_pre, pre_path = load_precompute(asset)
    # Only load arrays here for planning; each worker loads data again.
    dates_int = np.asarray(raw_pre["dates_int"], dtype=np.int64)
    tradable = np.asarray(raw_pre["tradable"], dtype=np.int8)

    if (not args.generated_scenarios) and args.scenario_root is not None and Path(args.scenario_root).exists():
        kinds = [x.strip() for x in str(args.scenario_kinds).split(",") if x.strip()]
        scenario_meta = load_scenarios_from_root(
            Path(args.scenario_root),
            exposure_id=exposure_id,
            kinds=kinds,
            dates_int=dates_int,
        )
        scenario_mode = "scenario_root"
        print(f"Loaded scenario kinds: {kinds}")
        print(f"Loaded scenario_id examples: {scenario_meta['scenario_id'].head(5).tolist()}")
    else:
        scenario_meta = generated_rolling_scenario_meta(
            dates_int,
            tradable,
            exposure_id=exposure_id,
            episode_len=args.episode_len,
            stride=args.stride,
            volume_bbl=args.volume_bbl,
        )
        scenario_mode = "generated_rolling"

    windows = build_walkforward_windows(
        scenario_meta,
        n_total_dates=len(dates_int),
        n_windows=args.n_windows,
        min_train_frac=args.min_train_frac,
        val_frac_of_train=args.val_frac_of_train,
        min_train_scenarios=args.min_train_scenarios,
    )

    print("=" * 90)
    print("SAC-LPM Parallel Walk-Forward")
    print(f"asset/exposure:     {asset} / {exposure_id}")
    print(f"precompute:         {pre_path}")
    print(f"scenario mode:      {scenario_mode}")
    print(f"scenario rows:      {len(scenario_meta)}")
    print(f"windows built:      {len(windows)}")
    print(f"max parallel:       {args.max_parallel}")
    print(f"timesteps/window:   {args.timesteps}")
    print(f"output dir:         {out_dir}")
    print("=" * 90)

    # Store planned windows.
    plan_rows = []
    tasks: List[Dict[str, Any]] = []
    for w in windows:
        plan_rows.append({
            "window": int(w["window"]),
            "train_n": len(w["train_meta"]),
            "val_n": len(w["val_meta"]),
            "test_n": len(w["test_meta"]),
            "train_end_idx": int(w["train_end_idx"]),
            "val_start_idx": int(w["val_start_idx"]),
            "val_end_idx": int(w["val_end_idx"]),
            "test_start_idx": int(w["test_start_idx"]),
            "test_end_idx": int(w["test_end_idx"]),
        })
        t = {
            "window": int(w["window"]),
            "asset": asset,
            "exposure_id": exposure_id,
            "out_dir": str(out_dir),
            "feature_mode": args.feature_mode,
            "seed": int(args.seed),
            "torch_threads": int(args.torch_threads),
            "train_meta": w["train_meta"].to_dict(orient="records"),
            "val_meta": w["val_meta"].to_dict(orient="records"),
            "test_meta": w["test_meta"].to_dict(orient="records"),
            "h_min": args.h_min,
            "h_max": args.h_max,
            "delta_h": args.delta_h,
            "lambda_lpm": args.lambda_lpm,
            "eta_cost": args.eta_cost,
            "lambda_smooth": args.lambda_smooth,
            "reward_scale": args.reward_scale,
            "reward_clip": args.reward_clip,
            "timesteps": args.timesteps,
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "learning_starts": args.learning_starts,
            "net_arch": args.net_arch,
        }
        tasks.append(t)

    plan = pd.DataFrame(plan_rows)
    plan.to_csv(out_dir / "walkforward_plan.csv", index=False)
    safe_to_parquet(plan, out_dir / "walkforward_plan.parquet")

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "asset": asset,
                "exposure_id": exposure_id,
                "precompute_path": str(pre_path),
                "scenario_mode": scenario_mode,
                "n_scenarios": int(len(scenario_meta)),
                "n_windows": int(len(windows)),
                "created_at": now_str(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metrics = run_parallel(tasks, max_parallel=args.max_parallel)
    metrics_df = pd.DataFrame(metrics).sort_values("window")
    metrics_df.to_csv(out_dir / "dashboard_summary.csv", index=False)
    safe_to_parquet(metrics_df, out_dir / "dashboard_summary.parquet")

    all_results, all_daily = collect_outputs(out_dir, metrics)
    print("\n" + "=" * 90)
    print("DONE")
    print(f"successful windows: {sum(1 for m in metrics if m.get('status') == 'ok')} / {len(metrics)}")
    print(f"results rows:       {len(all_results)}")
    print(f"daily log rows:     {len(all_daily)}")
    print(f"report input:       {out_dir / 'results_all_windows.parquet'}")
    print("=" * 90)


if __name__ == "__main__":
    main()