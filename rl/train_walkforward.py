

"""Walk-forward training runner for the daily hedging RL environment.

This module is intentionally pragmatic:
- Uses the project's precomputed arrays (rl_cache/precompute_<EXPOSURE>.npz)
- Trains with rolling walk-forward windows: 2y train, 1y validation, 1y test
- Advances the window by 1 year each iteration
- Saves one "best" checkpoint per window based on validation mean episode reward

Notes
-----
1) Scenario generation: to keep this runner self-contained and reproducible, we
   generate simple physical-trade scenarios (volume + holding period) directly
   from the date index. You can later swap `build_scenarios_*` to use the richer
   Scenario_Generator outputs.

2) Speed: environment is NumPy-based and can be parallelized via SubprocVecEnv.
   The policy network runs on MPS when available.

Outputs
-------
For each window under `--out_dir`:
- model_best.zip (SB3)
- val_summary.json
- test_episodes.csv (episode-level PnL/cost/turnover/MDD and reward totals)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Timing/progress
import time
from datetime import timedelta

from rl.precompute import load_npz
from rl.env_daily import EnvConfig, OilHedgingDailyEnv, make_env

import pandas as pd

# Add imports for CLI/log redirection
import sys
import contextlib

# Parallel/process helpers
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
# -------------------------
# Scenario loading (Parquet; reproducible)
# -------------------------


from rl.scenario_loader import load_scenarios_from_parquet, ScenarioLoaderConfig

# Optional progress bars (CLI + per-window training)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _load_pre_from_cache(cache_path: str) -> Any:
    """Load precompute bundle. Returns PrecomputeResult (preferred) for speed and clarity."""
    return load_npz(cache_path)


def _load_window_scenarios(
    *,
    scenario_path: str,
    dates_int: np.ndarray,
    exposure_id: str,
    window_start_int: int,
    window_end_int: int,
    max_scenarios: int,
    seed: int,
    require_ok_coverage: bool = True,
    allow_shortened: bool = False,
) -> List[Dict[str, Any]]:
    cfg = ScenarioLoaderConfig(
        require_ok_coverage=bool(require_ok_coverage),
        allow_shortened=bool(allow_shortened),
        max_scenarios=int(max_scenarios) if int(max_scenarios) > 0 else None,
        seed=int(seed),
    )
    return load_scenarios_from_parquet(
        scenario_path,
        dates_int=dates_int,
        exposure_id=str(exposure_id),
        window_start_day_int=int(window_start_int),
        window_end_day_int=int(window_end_int),
        require_full_containment=True,
        cfg=cfg,
    )
# -------------------------
# Walk-forward loop
# -------------------------


class _TeeIO:
    """Write to two streams (used to keep train.log while still printing to terminal)."""
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def write(self, s):
        try:
            self.a.write(s)
        except Exception:
            pass
        try:
            self.b.write(s)
        except Exception:
            pass
        return len(s)
    def flush(self):
        try:
            self.a.flush()
        except Exception:
            pass
        try:
            self.b.flush()
        except Exception:
            pass

def _run_window_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entrypoint for one walk-forward window.

    Notes:
    - Loads cache inside the worker to avoid pickling large NumPy arrays.
    - When running many workers, forces DummyVecEnv to avoid nested multiprocessing overload.
    """
    try:
        cache_path = str(job["cache"])
        out_dir = Path(job["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        log_path = out_dir / "train.log"

        # Optional: reduce oversubscription inside each process
        # (outer parallelism is across windows)
        try:
            import torch
            tt = int(job.get("torch_threads", 0))
            ti = int(job.get("torch_interop_threads", 0))
            if tt > 0:
                torch.set_num_threads(tt)
            if ti > 0:
                torch.set_num_interop_threads(ti)
        except Exception:
            pass

        cli_live = bool(job.get("cli_live", False))

        with open(log_path, "w", encoding="utf-8") as _log_f:
            # If sequential (cli_live=True): tee stdout to both terminal and log,
            # and DO NOT redirect stderr so tqdm bars (stderr by default) show in terminal.
            if cli_live:
                tee = _TeeIO(_log_f, sys.__stdout__)
                _ctx_out = contextlib.redirect_stdout(tee)
                _ctx_err = contextlib.nullcontext()  # keep stderr live
            else:
                # Parallel windows: keep logs quiet in terminal
                _ctx_out = contextlib.redirect_stdout(_log_f)
                _ctx_err = contextlib.redirect_stderr(_log_f)

            with _ctx_out, _ctx_err:

                pre = _load_pre_from_cache(cache_path)

                print("\n" + "═" * 100)
                print(f"🚀 WINDOW: {job.get('window_name')}")
                print(f"   Exposure: {job.get('exposure_id')} | Mode: {job.get('train_mode')} | Device: {job.get('device')}")
                print("═" * 100)

                # Build env config
                env_cfg = EnvConfig(
                    h_max=float(job["h_max"]),
                    delta_h_max=float(job.get("delta_h_max", 0.10)),
                    delta_h_step=float(job.get("delta_h_step", 0.05)),
                    action_mode=str(job["action_mode"]),
                    risk_mode=str(job["risk_mode"]),
                    mu_pnl=float(job.get("mu_pnl", 0.10)),
                    lambda_rollvar=float(job.get("lambda_rollvar", 5.0)),
                    roll_var_L=int(job.get("roll_var_L", 20)),
                    lambda_var=float(job["lambda_var"]),
                    lambda_lpm=float(job["lambda_lpm"]),
                    lpm_order=int(job["lpm_order"]),
                    lpm_target=float(job["lpm_target"]),
                    eta_cost=float(job["eta_cost"]),
                    info_mode=str(job.get("info_mode", "train")),
                )

                # -------------------------
                # Load scenarios from Parquet (no synthetic scenarios)
                # -------------------------
                exposure_id = str(job["exposure_id"])
                scenario_dir = str(job["scenario_dir"])

                # Paths
                p_train = os.path.join(scenario_dir, exposure_id, "oracle_universe.parquet")
                p_oracle_all = os.path.join(scenario_dir, exposure_id, "oracle_all.parquet")
                p_baseline = os.path.join(scenario_dir, exposure_id, "baseline.parquet")

                # Window years
                train_y0 = int(job["train_y0"])
                train_y1 = int(job["train_y1"])
                val_y = int(job["val_y"])
                test_y = int(job["test_y"])

                # Load train/val/test scenarios using window ints
                dates_int = pre.dates_int
                seed = int(job["seed"])
                max_train = int(job.get("max_train_scenarios", 0))
                max_eval = int(job.get("max_eval_scenarios", 0))
                # Evaluation scope. Default is strict out-of-sample test year.
                # In eval_only mode we may want broader coverage without retraining:
                #   test_year     : only scenarios fully contained in the test year (fair OOS default)
                #   all_until_test: scenarios from the first available date through the test-year end
                #   all_available : all scenarios available in the cache date range
                eval_scope = str(job.get("eval_scope", "test_year")).lower().strip()
                eval_start_int = int(job["test_start_int"])
                eval_end_int = int(job["test_end_int"])
                if bool(job.get("eval_only", False)):
                    if eval_scope == "all_until_test":
                        eval_start_int = int(np.asarray(dates_int, dtype=np.int64).min())
                        eval_end_int = int(job["test_end_int"])
                    elif eval_scope == "all_available":
                        eval_start_int = int(np.asarray(dates_int, dtype=np.int64).min())
                        eval_end_int = int(np.asarray(dates_int, dtype=np.int64).max())
                    elif eval_scope == "test_year":
                        pass
                    else:
                        raise ValueError(f"Unknown eval_scope={eval_scope}; expected test_year/all_until_test/all_available")
                train_sc = _load_window_scenarios(
                    scenario_path=p_train,
                    dates_int=dates_int,
                    exposure_id=exposure_id,
                    window_start_int=int(job["train_start_int"]),
                    window_end_int=int(job["train_end_int"]),
                    max_scenarios=max_train,
                    seed=seed + 11,
                    require_ok_coverage=True,
                    allow_shortened=False,
                )
                val_sc = _load_window_scenarios(
                    scenario_path=p_train,
                    dates_int=dates_int,
                    exposure_id=exposure_id,
                    window_start_int=int(job["val_start_int"]),
                    window_end_int=int(job["val_end_int"]),
                    max_scenarios=max_eval,
                    seed=seed + 22,
                    require_ok_coverage=True,
                    allow_shortened=False,
                )
                test_sc = _load_window_scenarios(
                    scenario_path=p_train,
                    dates_int=dates_int,
                    exposure_id=exposure_id,
                    window_start_int=int(eval_start_int),
                    window_end_int=int(eval_end_int),
                    max_scenarios=max_eval,
                    seed=seed + 33,
                    require_ok_coverage=True,
                    allow_shortened=False,
                )

                # Extra evaluation sets on the SAME test year
                test_sc_oracle_all = _load_window_scenarios(
                    scenario_path=p_oracle_all,
                    dates_int=dates_int,
                    exposure_id=exposure_id,
                    window_start_int=int(eval_start_int),
                    window_end_int=int(eval_end_int),
                    max_scenarios=max_eval,
                    seed=seed + 44,
                    require_ok_coverage=True,
                    allow_shortened=True,
                )
                test_sc_baseline = _load_window_scenarios(
                    scenario_path=p_baseline,
                    dates_int=dates_int,
                    exposure_id=exposure_id,
                    window_start_int=int(eval_start_int),
                    window_end_int=int(eval_end_int),
                    max_scenarios=max_eval,
                    seed=seed + 55,
                    require_ok_coverage=True,
                    allow_shortened=True,
                )

                print("\n📊 SCENARIO SUMMARY")
                print(f"   Train scenarios:        {len(train_sc):,}")
                print(f"   Validation scenarios:   {len(val_sc):,}")
                print(f"   Test (oracle_universe): {len(test_sc):,}")
                print(f"   Test (oracle_all):      {len(test_sc_oracle_all):,}")
                print(f"   Test (baseline):        {len(test_sc_baseline):,}")
                print(f"   Train window ints: {int(job['train_start_int'])}..{int(job['train_end_int'])}")

                # Guardrails
                if len(train_sc) < 50 or len(val_sc) < 20 or len(test_sc) < 20:
                    return {
                        "status": "skipped",
                        "window": str(job["window_name"]),
                        "train": len(train_sc),
                        "val": len(val_sc),
                        "test": len(test_sc),
                        "seconds": float(time.time() - t0),
                        "log": str(log_path),
                    }
                if len(test_sc_oracle_all) < 10 or len(test_sc_baseline) < 10:
                    return {
                        "status": "skipped",
                        "window": str(job["window_name"]),
                        "train": len(train_sc),
                        "val": len(val_sc),
                        "test": len(test_sc),
                        "test_oracle_all": len(test_sc_oracle_all),
                        "test_baseline": len(test_sc_baseline),
                        "seconds": float(time.time() - t0),
                        "log": str(log_path),
                    }

                # When running many workers, avoid nested multiprocessing: force dummy vec.
                vec_backend = str(job.get("vec", "auto"))
                if int(job.get("parallel_windows", 1)) > 1:
                    vec_backend = "dummy"

                # -------------------------
                # Eval-only mode: reuse existing model_best.zip and evaluate all requested scenarios
                # without retraining. This is useful when the trained checkpoints are already valid
                # but the previous run saved too few evaluation episodes.
                # -------------------------
                if bool(job.get("eval_only", False)):
                    from stable_baselines3 import PPO

                    model_path = out_dir / "model_best.zip"
                    if not model_path.exists():
                        raise FileNotFoundError(f"eval_only requested but model checkpoint not found: {model_path}")

                    print("\n🔁 EVAL-ONLY MODE — loading saved checkpoint and skipping training")
                    print(f"   model: {model_path}")
                    model = PPO.load(str(model_path), device=str(job["device"]))

                    def _cap_eval(n_req: int, n_avail: int) -> int:
                        if int(n_req) <= 0:
                            return int(n_avail)
                        return int(min(int(n_req), int(n_avail)))

                    n_test_eval = _cap_eval(int(job["eval_episodes"]), len(test_sc))
                    n_oa_eval = _cap_eval(int(job["eval_episodes"]), len(test_sc_oracle_all))
                    n_bl_eval = _cap_eval(int(job["eval_episodes"]), len(test_sc_baseline))

                    def _eps_df(
                        eps: List[Dict[str, Any]],
                        dataset: str,
                        scenarios: List[Dict[str, Any]],
                        split: str,
                    ) -> pd.DataFrame:
                        df = pd.DataFrame(eps)
                        if df.empty:
                            df = pd.DataFrame(columns=[
                                "scenario_idx", "Q", "steps", "reward_sum", "pnl_net_sum",
                                "cost_sum", "turnover_contract", "mdd",
                            ])

                        ds = str(dataset)
                        if ds not in ("oracle_universe", "oracle_all", "baseline"):
                            ds = "unknown"

                        n_sc = len(scenarios)

                        def _get(i: int, key: str, default=None):
                            if 0 <= i < n_sc:
                                return scenarios[i].get(key, default)
                            return default

                        if "scenario_idx" not in df.columns:
                            df["scenario_idx"] = -1
                        df["scenario_idx"] = pd.to_numeric(df["scenario_idx"], errors="coerce").fillna(-1).astype(int)

                        idxs = df["scenario_idx"].to_numpy(dtype=int, copy=False)
                        df["scenario_id"] = [_get(int(i), "scenario_id", None) for i in idxs]
                        df["tag"] = [_get(int(i), "tag", None) for i in idxs]
                        df["oracle_series"] = [_get(int(i), "oracle_series", None) for i in idxs]
                        df["oracle_pool"] = [_get(int(i), "oracle_pool", None) for i in idxs]
                        df["oracle_candidate"] = [_get(int(i), "oracle_candidate", None) for i in idxs]
                        df["company_id"] = [_get(int(i), "company_id", None) for i in idxs]
                        df["company_size"] = [_get(int(i), "company_size", None) for i in idxs]
                        df["start_idx"] = [_get(int(i), "start_idx", None) for i in idxs]
                        df["end_idx"] = [_get(int(i), "end_idx", None) for i in idxs]
                        df["start_date_int"] = [_get(int(i), "start_date_int", None) for i in idxs]
                        df["end_date_int"] = [_get(int(i), "end_date_int", None) for i in idxs]
                        df["horizon_days_target"] = [_get(int(i), "horizon_days_target", None) for i in idxs]
                        df["horizon_days_realized"] = [_get(int(i), "horizon_days_realized", None) for i in idxs]

                        df.insert(0, "split", str(split))
                        df.insert(0, "dataset", ds)
                        df.insert(0, "strategy", "RL_PPO")
                        df.insert(0, "exposure_id", str(exposure_id))

                        for k, v in {
                            "train_y0": train_y0,
                            "train_y1": train_y1,
                            "val_y": val_y,
                            "test_y": test_y,
                            "window_name": str(job["window_name"]),
                        }.items():
                            df[k] = v

                        return df

                    print(f"\n🔎 EVAL-ONLY: test (oracle_universe) | episodes={n_test_eval:,} / available={len(test_sc):,}")
                    test_env = OilHedgingDailyEnv(pre, test_sc, cfg=env_cfg, seed=seed + 2020)
                    test_eps = run_episodes_with_progress(
                        test_env,
                        model,
                        n_episodes=n_test_eval,
                        deterministic=True,
                        seed=seed + 2021,
                        desc=f"{out_dir.name} | eval-only:test:univ",
                        enable_pbar=bool(cli_live),
                    )

                    print(f"\n🔎 EVAL-ONLY: test (oracle_all) | episodes={n_oa_eval:,} / available={len(test_sc_oracle_all):,}")
                    test_env_oa = OilHedgingDailyEnv(pre, test_sc_oracle_all, cfg=env_cfg, seed=seed + 3030)
                    test_eps_oa = run_episodes_with_progress(
                        test_env_oa,
                        model,
                        n_episodes=n_oa_eval,
                        deterministic=True,
                        seed=seed + 3031,
                        desc=f"{out_dir.name} | eval-only:test:all",
                        enable_pbar=bool(cli_live),
                    )

                    print(f"\n🔎 EVAL-ONLY: test (baseline) | episodes={n_bl_eval:,} / available={len(test_sc_baseline):,}")
                    test_env_bl = OilHedgingDailyEnv(pre, test_sc_baseline, cfg=env_cfg, seed=seed + 4040)
                    test_eps_bl = run_episodes_with_progress(
                        test_env_bl,
                        model,
                        n_episodes=n_bl_eval,
                        deterministic=True,
                        seed=seed + 4041,
                        desc=f"{out_dir.name} | eval-only:test:base",
                        enable_pbar=bool(cli_live),
                    )

                    df_test = _eps_df(test_eps, "oracle_universe", test_sc, "test")
                    df_oa = _eps_df(test_eps_oa, "oracle_all", test_sc_oracle_all, "test")
                    df_bl = _eps_df(test_eps_bl, "baseline", test_sc_baseline, "test")
                    df_all = pd.concat([df_test, df_oa, df_bl], ignore_index=True)
                    df_all.to_parquet(out_dir / "test_episodes_all.parquet", index=False)

                    def _mean(eps: List[Dict[str, Any]], key: str) -> float:
                        if not eps:
                            return float("nan")
                        vals = [float(x.get(key, float("nan"))) for x in eps]
                        vals = [v for v in vals if np.isfinite(v)]
                        return float(np.mean(vals)) if vals else float("nan")

                    eval_summary = {
                        "eval_only": True,
                        "test_pnl_mean": _mean(test_eps, "pnl_net_sum"),
                        "test_pnl_mean_oracle_all": _mean(test_eps_oa, "pnl_net_sum"),
                        "test_pnl_mean_baseline": _mean(test_eps_bl, "pnl_net_sum"),
                        "test_mdd_mean": _mean(test_eps, "mdd"),
                        "test_mdd_mean_oracle_all": _mean(test_eps_oa, "mdd"),
                        "test_mdd_mean_baseline": _mean(test_eps_bl, "mdd"),
                        "n_test_oracle_universe": int(len(test_eps)),
                        "n_test_oracle_all": int(len(test_eps_oa)),
                        "n_test_baseline": int(len(test_eps_bl)),
                        "available_oracle_universe": int(len(test_sc)),
                        "available_oracle_all": int(len(test_sc_oracle_all)),
                        "available_baseline": int(len(test_sc_baseline)),
                    }
                    with open(out_dir / "eval_only_summary.json", "w", encoding="utf-8") as f:
                        json.dump(eval_summary, f, ensure_ascii=False, indent=2)

                    return eval_summary
                # Train + eval
                metrics = train_one_window(
                    pre=pre,
                    env_cfg=env_cfg,
                    train_scenarios=train_sc,
                    val_scenarios=val_sc,
                    test_scenarios=test_sc,
                    test_scenarios_oracle_all=test_sc_oracle_all,
                    test_scenarios_baseline=test_sc_baseline,
                    out_dir=out_dir,
                    seed=int(job["seed"]),
                    n_envs=int(job["n_envs"]),
                    total_timesteps=int(job["timesteps"]),
                    eval_episodes=int(job["eval_episodes"]),
                    dummy_policy=str(job.get("dummy_policy", "none")),
                    eta_print_freq=int(job["eta_print_freq"]),
                    device=str(job["device"]),
                    cli_live=bool(job.get("cli_live", False)),
                    vec_backend=vec_backend,
                    n_steps=int(job["n_steps"]),
                    batch_size=int(job["batch_size"]),
                    net_arch=tuple(int(x) for x in job["net_arch"]),
                    ppo_gamma=float(job.get("ppo_gamma", 0.99)),
                    ppo_gae_lambda=float(job.get("ppo_gae_lambda", 0.95)),
                    ppo_ent_coef=float(job.get("ppo_ent_coef", 0.0)),
                    ppo_vf_coef=float(job.get("ppo_vf_coef", 0.5)),
                    ppo_max_grad_norm=float(job.get("ppo_max_grad_norm", 0.5)),
                    ppo_n_epochs=int(job.get("ppo_n_epochs", 10)),
                    ppo_learning_rate=float(job.get("ppo_learning_rate", 3e-4)),
                    ppo_clip_range=float(job.get("ppo_clip_range", 0.2)),
                    ppo_target_kl=float(job.get("ppo_target_kl", 0.0)),
                    exposure_id=exposure_id,
                    window_meta={"train_y0": train_y0, "train_y1": train_y1, "val_y": val_y, "test_y": test_y, "window_name": str(job["window_name"])},
                    
                )

        # End redirect context; return compact summary to parent.
        out = {"status": "ok", "window": str(job["window_name"]), "seconds": float(time.time() - t0), "log": str(log_path)}
        if isinstance(metrics, dict):
            out.update(metrics)
        return out

    except Exception as e:
        return {
            "status": "error",
            "window": str(job.get("window_name", "?")),
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


# -------------------------
# Date helpers
# -------------------------


def intdays_to_datetime64(d: np.ndarray) -> np.ndarray:
    """Convert int64 days since epoch to datetime64[D]."""
    d = np.asarray(d, dtype=np.int64)
    return d.astype("datetime64[D]")


def year_of_intdays(d: np.ndarray) -> np.ndarray:
    dt = intdays_to_datetime64(d)
    # convert to year integer
    return (dt.astype("datetime64[Y]").astype(int) + 1970).astype(int)


def first_date_of_year(y: int) -> np.datetime64:
    return np.datetime64(f"{y}-01-01", "D")


def last_date_of_year(y: int) -> np.datetime64:
    return np.datetime64(f"{y}-12-31", "D")


# -------------------------
# Scenario generation (baseline, reproducible)
# -------------------------


def build_scenarios_baseline(
    *,
    dates_int: np.ndarray,
    start_int: int,
    end_int: int,
    hold_days: int = 20,
    volume_bbl: float = 1_000_000.0,
    stride_days: int = 1,
) -> List[Dict[str, Any]]:
    """Build simple fixed-horizon scenarios starting inside [start,end].

    A scenario is included only if both start and end dates exist in dates_int
    and the end is <= end_int.
    """
    dates_int = np.asarray(dates_int, dtype=np.int64)
    # Map int date -> position
    idx = {int(d): i for i, d in enumerate(dates_int)}

    dt_start = np.datetime64(start_int, "D")
    dt_end = np.datetime64(end_int, "D")

    # iterate by calendar days; keep only those present in the trading calendar
    scenarios: List[Dict[str, Any]] = []
    d = dt_start
    stride = max(int(stride_days), 1)
    while d <= dt_end:
        s_int = int(d.astype("int64"))
        if s_int in idx:
            e_dt = d + np.timedelta64(int(hold_days), "D")
            e_int = int(e_dt.astype("int64"))
            if e_int <= end_int and e_int in idx:
                scenarios.append({
                    "start_date_int": int(s_int),
                    "end_date_int": int(e_int),
                    "volume_bbl": float(volume_bbl),
                })
        d = d + np.timedelta64(stride, "D")

    return scenarios


def build_scenarios_randomized(
    *,
    dates_int: np.ndarray,
    start_int: int,
    end_int: int,
    hold_min: int = 20,
    hold_max: int = 40,
    vol_min: float = 1_000_000.0,
    vol_max: float = 2_000_000.0,
    stride_days: int = 1,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    """Build randomized-but-reproducible scenarios (duration + volume)."""
    rng = np.random.default_rng(int(seed))
    dates_int = np.asarray(dates_int, dtype=np.int64)
    idx = {int(d): i for i, d in enumerate(dates_int)}

    dt_start = np.datetime64(start_int, "D")
    dt_end = np.datetime64(end_int, "D")

    scenarios: List[Dict[str, Any]] = []
    d = dt_start
    stride = max(int(stride_days), 1)
    while d <= dt_end:
        s_int = int(d.astype("int64"))
        if s_int in idx:
            hold = int(rng.integers(hold_min, hold_max + 1))
            vol = float(rng.uniform(vol_min, vol_max))
            e_dt = d + np.timedelta64(hold, "D")
            e_int = int(e_dt.astype("int64"))
            if e_int <= end_int and e_int in idx:
                scenarios.append({
                    "start_date_int": int(s_int),
                    "end_date_int": int(e_int),
                    "volume_bbl": float(vol),
                })
        d = d + np.timedelta64(stride, "D")
    return scenarios


# -------------------------
# Evaluation
# -------------------------


# -------------------------
# CLI utilities
# -------------------------


def _fmt_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    return str(timedelta(seconds=int(seconds)))


class ETACallback:
    """Minimal ETA/progress callback for SB3.

    - If tqdm is available, drives a per-window progress bar for timesteps.
    - For PPO, we also show an approximate "updates" counter:
        updates ≈ timesteps / (n_steps * n_envs)
    """

    def __init__(
        self,
        total_timesteps: int,
        *,
        print_freq: int = 10_000,
        prefix: str = "train",
        use_tqdm: bool = False,
    ):
        self.total_timesteps = int(total_timesteps)
        self.print_freq = int(max(print_freq, 1))
        self.prefix = str(prefix)
        self.use_tqdm = bool(use_tqdm)
        self._t0 = None
        self._last_print = 0
        self._pbar = None
        self._last_pbar_n = 0
        self._steps_per_update = 0

    def init_callback(self, model) -> None:
        self.model = model
        self._t0 = time.time()
        self._last_print = 0
        self._last_pbar_n = 0

        # Approx PPO "updates" counter
        try:
            n_steps = int(getattr(model, "n_steps", 0))
            n_envs = int(getattr(getattr(model, "env", None), "num_envs", 0))
            if n_steps > 0 and n_envs > 0:
                self._steps_per_update = int(n_steps * n_envs)
        except Exception:
            self._steps_per_update = 0

        # SB3 (especially PPO) will stop at a multiple of (n_steps * n_envs),
        # so num_timesteps may slightly exceed the requested total_timesteps.
        eff_total = int(self.total_timesteps)
        if self._steps_per_update > 0:
            eff_total = int(((eff_total + self._steps_per_update - 1) // self._steps_per_update) * self._steps_per_update)
        self.total_timesteps = eff_total

        if self.use_tqdm and tqdm is not None:
            try:
                self._pbar = tqdm(
                    total=int(self.total_timesteps),
                    desc=str(self.prefix),
                    unit="ts",
                    leave=True,
                    dynamic_ncols=True,
                    file=sys.__stdout__,
                )
            except Exception:
                self._pbar = None

    def close(self) -> None:
        try:
            if self._pbar is not None:
                self._pbar.close()
        except Exception:
            pass
        self._pbar = None

    def on_step(self) -> bool:
        n = int(getattr(self.model, "num_timesteps", 0))

        # tqdm update (every callback step)
        if self._pbar is not None:
            dn = max(n - self._last_pbar_n, 0)
            if dn:
                self._pbar.update(dn)
                self._last_pbar_n = n

            dt = max(time.time() - float(self._t0 or time.time()), 1e-6)
            fps = n / dt
            remaining = max(self.total_timesteps - n, 0)
            eta_s = remaining / max(fps, 1e-6)
            pct = 100.0 * (n / max(self.total_timesteps, 1))
            updates = (n // self._steps_per_update) if self._steps_per_update > 0 else int(n > 0)
            try:
                self._pbar.set_postfix({"fps": f"{fps:,.0f}", "eta": _fmt_eta(eta_s), "%": f"{pct:5.1f}", "upd": int(updates)})
            except Exception:
                pass

        # If tqdm bar is active, do not also print fallback lines (prevents messy output).
        if self._pbar is not None:
            return True

        # Fallback periodic print (when tqdm is not available)
        if n - self._last_print < self.print_freq and n < self.total_timesteps:
            return True
        self._last_print = n

        dt = max(time.time() - float(self._t0 or time.time()), 1e-6)
        fps = n / dt
        remaining = max(self.total_timesteps - n, 0)
        eta_s = remaining / max(fps, 1e-6)
        pct = 100.0 * (n / max(self.total_timesteps, 1))
        updates = (n // self._steps_per_update) if self._steps_per_update > 0 else 0

        print(
            f"[{self.prefix}] {n:,}/{self.total_timesteps:,} ({pct:5.1f}%)  fps={fps:,.0f}  "
            f"ETA={_fmt_eta(eta_s)}  updates={int(updates)}"
        )
        return True


# -------------------------
# Evaluation
# -------------------------

def run_episodes_with_progress(
    env: OilHedgingDailyEnv,
    model,
    n_episodes: int,
    *,
    deterministic: bool = True,
    seed: Optional[int] = None,
    desc: str = "eval",
    enable_pbar: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if seed is not None:
        env.reset(seed=int(seed))

    pbar = None
    if enable_pbar and tqdm is not None:
        try:
            pbar = tqdm(
                total=int(n_episodes),
                desc=str(desc),
                unit="ep",
                leave=True,
                dynamic_ncols=True,
                file=sys.__stdout__,
            )
        except Exception:
            pbar = None

    try:
        for _ in range(int(n_episodes)):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_pnl = 0.0
            ep_cost = 0.0
            ep_turnover = 0.0
            last_n = None
            mdd = 0.0
            steps = 0
            action_counts: Dict[int, int] = {}
            h_values: List[float] = []
            n_values: List[int] = []
            h_turnover = 0.0
            last_h = None

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                try:
                    action_int = int(np.asarray(action).reshape(-1)[0])
                except Exception:
                    action_int = -999
                action_counts[action_int] = action_counts.get(action_int, 0) + 1
                obs, r, terminated, truncated, inf = env.step(action)
                done = bool(terminated or truncated)
                ep_reward += float(r)
                ep_pnl += float(inf.get("pnl_net", 0.0))
                ep_cost += float(inf.get("cost", 0.0))
                mdd = float(inf.get("mdd", mdd))

                n_now = int(inf.get("n_new", inf.get("n_prev", 0)))
                h_now = float(inf.get("h", inf.get("h_new", inf.get("h_prev", 0.0))))
                h_values.append(h_now)
                n_values.append(n_now)
                if last_n is None:
                    ep_turnover += abs(n_now)
                else:
                    ep_turnover += abs(n_now - last_n)
                if last_h is None:
                    h_turnover += abs(h_now)
                else:
                    h_turnover += abs(h_now - last_h)
                last_n = n_now
                last_h = h_now
                steps += 1

            out.append({
                "scenario_idx": int(info.get("scenario_idx", -1)),
                "Q": float(info.get("Q", np.nan)),
                "steps": int(steps),
                "reward_sum": float(ep_reward),
                "pnl_net_sum": float(ep_pnl),
                "cost_sum": float(ep_cost),
                "turnover_contract": float(ep_turnover),
                "turnover_h": float(h_turnover),
                "h_abs_mean": float(np.mean(np.abs(h_values))) if h_values else 0.0,
                "h_mean": float(np.mean(h_values)) if h_values else 0.0,
                "h_std": float(np.std(h_values)) if h_values else 0.0,
                "h_nonzero_share": float(np.mean(np.abs(h_values) > 1e-6)) if h_values else 0.0,
                "n_abs_mean": float(np.mean(np.abs(n_values))) if n_values else 0.0,
                "action_counts": json.dumps(action_counts, sort_keys=True),
                "mdd": float(mdd),
            })

            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    return out

def run_episodes(
    env: OilHedgingDailyEnv,
    model,
    n_episodes: int,
    *,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run episodes on a single (non-vector) env and return episode summaries."""
    out: List[Dict[str, Any]] = []
    if seed is not None:
        env.reset(seed=int(seed))

    for _ in range(int(n_episodes)):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_pnl = 0.0
        ep_cost = 0.0
        ep_turnover = 0.0
        last_n = None
        last_h = None
        mdd = 0.0
        steps = 0
        action_counts: Dict[int, int] = {}
        h_values: List[float] = []
        n_values: List[int] = []
        h_turnover = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            try:
                action_int = int(np.asarray(action).reshape(-1)[0])
            except Exception:
                action_int = -999
            action_counts[action_int] = action_counts.get(action_int, 0) + 1
            obs, r, terminated, truncated, inf = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(r)
            ep_pnl += float(inf.get("pnl_net", 0.0))
            ep_cost += float(inf.get("cost", 0.0))
            mdd = float(inf.get("mdd", mdd))

            n_now = int(inf.get("n_new", inf.get("n_prev", 0)))
            h_now = float(inf.get("h", inf.get("h_new", inf.get("h_prev", 0.0))))
            h_values.append(h_now)
            n_values.append(n_now)
            if last_n is None:
                ep_turnover += abs(n_now)
            else:
                ep_turnover += abs(n_now - last_n)
            if last_h is None:
                h_turnover += abs(h_now)
            else:
                h_turnover += abs(h_now - last_h)
            last_n = n_now
            last_h = h_now
            steps += 1

        out.append({
            "scenario_idx": int(info.get("scenario_idx", -1)),
            "Q": float(info.get("Q", np.nan)),
            "steps": int(steps),
            "reward_sum": float(ep_reward),
            "pnl_net_sum": float(ep_pnl),
            "cost_sum": float(ep_cost),
            "turnover_contract": float(ep_turnover),
            "turnover_h": float(h_turnover),
            "h_abs_mean": float(np.mean(np.abs(h_values))) if h_values else 0.0,
            "h_mean": float(np.mean(h_values)) if h_values else 0.0,
            "h_std": float(np.std(h_values)) if h_values else 0.0,
            "h_nonzero_share": float(np.mean(np.abs(h_values) > 1e-6)) if h_values else 0.0,
            "n_abs_mean": float(np.mean(np.abs(n_values))) if n_values else 0.0,
            "action_counts": json.dumps(action_counts, sort_keys=True),
            "mdd": float(mdd),
        })

    return out


# -------------------------
# Walk-forward loop
# -------------------------


def walkforward_windows(year_start: int, year_end: int) -> List[Tuple[int, int, int, int]]:
    """Return list of (train_y0, train_y1, val_y, test_y) windows.

    Base structure (step=1y):
      train: years [y, y+1]  (2 years, inclusive)
      val:   year  y+2
      test:  year  y+3

    The caller may convert this to expanding/hybrid by overriding train_y0.
    """
    out: List[Tuple[int, int, int, int]] = []
    y = int(year_start)
    while True:
        train_y0 = y
        train_y1 = y + 1
        val_y = y + 2
        test_y = y + 3
        if test_y > int(year_end):
            break
        out.append((train_y0, train_y1, val_y, test_y))
        y += 1
    return out


def year_range_to_int_bounds(y0: int, y1: int) -> Tuple[int, int]:
    """Inclusive bounds for [Jan1 y0, Dec31 y1]."""
    s = int(first_date_of_year(y0).astype("int64"))
    e = int(last_date_of_year(y1).astype("int64"))
    return s, e



def _select_device(device_arg: str) -> str:
    device_arg = str(device_arg)
    if device_arg in ("cpu", "mps"):
        return device_arg
    # auto
    try:
        import torch
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def train_one_window(
    *,
    pre: Any,
    env_cfg: EnvConfig,
    train_scenarios: List[Dict[str, Any]],
    val_scenarios: List[Dict[str, Any]],
    test_scenarios: List[Dict[str, Any]],
    test_scenarios_oracle_all: List[Dict[str, Any]],
    test_scenarios_baseline: List[Dict[str, Any]],
    out_dir: Path,
    seed: int,
    n_envs: int,
    total_timesteps: int,
    eval_episodes: int,
    dummy_policy: str = "none",
    policy_kwargs: Optional[Dict[str, Any]] = None,
    eta_print_freq: int = 50_000,
    device: str = "cpu",
    cli_live: bool = False,
    vec_backend: str = "auto",
    n_steps: int = 8192,
    batch_size: int = 8192,
    net_arch: Tuple[int, ...] = (256, 256),
    exposure_id: str = "",
    window_meta: Optional[Dict[str, Any]] = None,
    # PPO hyperparameters (argparse-exposed)
    ppo_gamma: float = 0.99,
    ppo_gae_lambda: float = 0.95,
    ppo_ent_coef: float = 0.0,
    ppo_vf_coef: float = 0.5,
    ppo_max_grad_norm: float = 0.5,
    ppo_n_epochs: int = 10,
    ppo_learning_rate: float = 3e-4,
    ppo_clip_range: float = 0.2,
    ppo_target_kl: float = 0.0,
) -> Dict[str, Any]:
    """Train PPO on train scenarios, select best on val, evaluate on test."""

    def _cap(n_req: int, n_avail: int) -> int:
        if int(n_req) <= 0:
            return int(n_avail)
        return int(min(int(n_req), int(n_avail)))

    dp = str(dummy_policy or "none").lower().strip()

    # Cap evaluation counts to available scenarios to avoid hangs when eval_episodes is huge
    n_val_eval = _cap(eval_episodes, len(val_scenarios))
    n_test_eval = _cap(eval_episodes, len(test_scenarios))
    n_oa_eval = _cap(eval_episodes, len(test_scenarios_oracle_all))
    n_bl_eval = _cap(eval_episodes, len(test_scenarios_baseline))


    class _DummyPolicy:
        def __init__(self, env_cfg: EnvConfig, target_h: float, feature_dim: int):
            self.env_cfg = env_cfg
            self.target_h = float(target_h)
            self.feature_dim = int(feature_dim)
            explicit_grid = getattr(env_cfg, "delta_h_grid", None)
            if explicit_grid is not None:
                self._grid = np.asarray(explicit_grid, dtype=np.float32)
            else:
                dh_max = float(getattr(env_cfg, "delta_h_max", 0.10))
                dh_step = float(getattr(env_cfg, "delta_h_step", 0.05))
                if dh_step <= 0:
                    dh_step = 0.05
                n_grid_steps = int(np.floor(dh_max / dh_step + 1e-9))
                self._grid = (
                    np.arange(-n_grid_steps, n_grid_steps + 1, dtype=np.float32)
                    * np.float32(dh_step)
                ).astype(np.float32)
                if self._grid.size == 0 or not np.any(np.isclose(self._grid, 0.0)):
                    self._grid = np.sort(
                        np.unique(np.append(self._grid, np.float32(0.0)))
                    ).astype(np.float32)

        def predict(self, obs, deterministic: bool = True):
            obs = np.asarray(obs, dtype=np.float32)
            h_idx = self.feature_dim  # h_prev right after feature block
            h_prev = float(obs[h_idx]) if obs.size > h_idx else 0.0
            dh = float(self.target_h - h_prev)
            j = int(np.argmin(np.abs(self._grid - dh)))
            return np.asarray([j], dtype=np.int64), None

    if dp != "none":
        if dp == "nohedge":
            target_h = 0.0
            strategy_name = "DUMMY_NOHEDGE"
        else:
            target_h = 1.0
            strategy_name = "DUMMY_NAIVE"

        print("\n🧪 DUMMY POLICY MODE — skipping training")
        print(f"   policy={strategy_name} target_h={target_h}")

        model = _DummyPolicy(env_cfg, target_h=target_h, feature_dim=int(pre.feature_matrix.shape[1]))

        val_env = OilHedgingDailyEnv(pre, val_scenarios, cfg=env_cfg, seed=seed + 999)
        print(f"\n🔎 EVAL: validation (oracle_universe) | episodes={n_val_eval:,}")
        val_eps = run_episodes_with_progress(
            val_env,
            model,
            n_episodes=n_val_eval,
            deterministic=True,
            seed=seed + 777,
            desc=f"{out_dir.name} | val",
            enable_pbar=bool(cli_live),
        )
        val_mean = float(np.mean([x["reward_sum"] for x in val_eps])) if val_eps else float("nan")

        test_env = OilHedgingDailyEnv(pre, test_scenarios, cfg=env_cfg, seed=seed + 2020)
        print(f"\n🔎 EVAL: test (oracle_universe) | episodes={n_test_eval:,}")
        test_eps = run_episodes_with_progress(
            test_env,
            model,
            n_episodes=n_test_eval,
            deterministic=True,
            seed=seed + 2021,
            desc=f"{out_dir.name} | test:univ",
            enable_pbar=bool(cli_live),
        )

        test_env_oa = OilHedgingDailyEnv(pre, test_scenarios_oracle_all, cfg=env_cfg, seed=seed + 3030)
        print(f"\n🔎 EVAL: test (oracle_all) | episodes={n_oa_eval:,}")
        test_eps_oa = run_episodes_with_progress(
            test_env_oa,
            model,
            n_episodes=n_oa_eval,
            deterministic=True,
            seed=seed + 3031,
            desc=f"{out_dir.name} | test:all",
            enable_pbar=bool(cli_live),
        )

        test_env_bl = OilHedgingDailyEnv(pre, test_scenarios_baseline, cfg=env_cfg, seed=seed + 4040)
        print(f"\n🔎 EVAL: test (baseline) | episodes={n_bl_eval:,}")
        test_eps_bl = run_episodes_with_progress(
            test_env_bl,
            model,
            n_episodes=n_bl_eval,
            deterministic=True,
            seed=seed + 4041,
            desc=f"{out_dir.name} | test:base",
            enable_pbar=bool(cli_live),
        )

        def _eps_df(
            eps: List[Dict[str, Any]],
            dataset: str,
            scenarios: List[Dict[str, Any]],
            split: str,
        ) -> pd.DataFrame:
            df = pd.DataFrame(eps)
            if df.empty:
                df = pd.DataFrame(columns=[
                    "scenario_idx",
                    "Q",
                    "steps",
                    "reward_sum",
                    "pnl_net_sum",
                    "cost_sum",
                    "turnover_contract",
                    "mdd",
                ])

            ds = str(dataset)
            if ds not in ("oracle_universe", "oracle_all", "baseline"):
                ds = "unknown"

            n_sc = len(scenarios)
            def _get(i: int, key: str, default=None):
                if 0 <= i < n_sc:
                    return scenarios[i].get(key, default)
                return default

            if "scenario_idx" not in df.columns:
                df["scenario_idx"] = -1
            df["scenario_idx"] = pd.to_numeric(df["scenario_idx"], errors="coerce").fillna(-1).astype(int)

            idxs = df["scenario_idx"].to_numpy(dtype=int, copy=False)
            df["scenario_id"] = [_get(int(i), "scenario_id", None) for i in idxs]
            df["tag"] = [_get(int(i), "tag", None) for i in idxs]
            df["oracle_series"] = [_get(int(i), "oracle_series", None) for i in idxs]
            df["oracle_pool"] = [_get(int(i), "oracle_pool", None) for i in idxs]
            df["oracle_candidate"] = [_get(int(i), "oracle_candidate", None) for i in idxs]
            df["company_id"] = [_get(int(i), "company_id", None) for i in idxs]
            df["company_size"] = [_get(int(i), "company_size", None) for i in idxs]
            df["start_idx"] = [_get(int(i), "start_idx", None) for i in idxs]
            df["end_idx"] = [_get(int(i), "end_idx", None) for i in idxs]
            df["start_date_int"] = [_get(int(i), "start_date_int", None) for i in idxs]
            df["end_date_int"] = [_get(int(i), "end_date_int", None) for i in idxs]
            df["horizon_days_target"] = [_get(int(i), "horizon_days_target", None) for i in idxs]
            df["horizon_days_realized"] = [_get(int(i), "horizon_days_realized", None) for i in idxs]

            df.insert(0, "split", str(split))
            df.insert(0, "dataset", ds)
            df.insert(0, "strategy", strategy_name)
            df.insert(0, "exposure_id", str(exposure_id))

            for k, v in (window_meta or {}).items():
                df[k] = v

            return df

        df_test = _eps_df(test_eps, "oracle_universe", test_scenarios, "test")
        df_oa = _eps_df(test_eps_oa, "oracle_all", test_scenarios_oracle_all, "test")
        df_bl = _eps_df(test_eps_bl, "baseline", test_scenarios_baseline, "test")

        df_all = pd.concat([df_test, df_oa, df_bl], ignore_index=True)
        df_all.to_parquet(out_dir / "test_episodes_all.parquet", index=False)

        def _mean(eps: List[Dict[str, Any]], key: str) -> float:
            if not eps:
                return float("nan")
            vals = [float(x.get(key, float("nan"))) for x in eps]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        window_summary = {
            "val_mean_reward": float(val_mean),
            "test_reward_mean": float(_mean(test_eps, "reward_sum")),
            "test_pnl_mean": float(_mean(test_eps, "pnl_net_sum")),
            "test_cost_mean": float(_mean(test_eps, "cost_sum")),
            "test_mdd_mean": float(_mean(test_eps, "mdd")),
            "n_eval_episodes": int(eval_episodes),
            "device": str(device),
        "risk_mode": str(env_cfg.risk_mode),
        "eta_cost": float(env_cfg.eta_cost),
            "n_envs": int(n_envs),
            "timesteps": int(total_timesteps),
            "n_steps": int(n_steps),
            "batch_size": int(batch_size),
            "net_arch": list(net_arch),
            "dummy_policy": strategy_name,
            "dummy_mode": str(dp),
        }

        with open(out_dir / "window_summary.json", "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(window_summary, f, ensure_ascii=False, indent=2)

        return window_summary

    # Lazy import SB3
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    out_dir.mkdir(parents=True, exist_ok=True)

    # Vec env for training
    # Subproc is faster when env step is heavy; on mac spawn is default.
    env_fns = [make_env(pre, train_scenarios, cfg=env_cfg, seed=seed + i) for i in range(int(n_envs))]
    # Vec env selection: on mac, SubprocVecEnv can be slower due to spawn overhead.
    if vec_backend == "dummy":
        venv = DummyVecEnv(env_fns)
    elif vec_backend == "subproc":
        venv = SubprocVecEnv(env_fns)
    else:
        # auto
        try:
            venv = SubprocVecEnv(env_fns)
        except Exception:
            venv = DummyVecEnv(env_fns)

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        verbose=0,
        seed=int(seed),
        device=device,
        policy_kwargs=policy_kwargs or dict(net_arch=list(net_arch)),
        n_steps=int(n_steps),
        batch_size=int(batch_size),

        n_epochs=int(ppo_n_epochs),
        learning_rate=float(ppo_learning_rate),
        clip_range=float(ppo_clip_range),
        target_kl=None if float(ppo_target_kl) <= 0 else float(ppo_target_kl),

        gae_lambda=float(ppo_gae_lambda),
        gamma=float(ppo_gamma),
        ent_coef=float(ppo_ent_coef),
        vf_coef=float(ppo_vf_coef),
        max_grad_norm=float(ppo_max_grad_norm),
    )

    # Progress / ETA callback
    eta_cb = ETACallback(
        total_timesteps=int(total_timesteps),
        print_freq=int(eta_print_freq),
        prefix=f"🧠 TRAIN {out_dir.name}",
        use_tqdm=True,
    )
    eta_cb.init_callback(model)

    # SB3 expects a BaseCallback; we provide a tiny adapter to keep dependencies minimal.
    from stable_baselines3.common.callbacks import BaseCallback

    class _Adapter(BaseCallback):
        def __init__(self, inner: ETACallback):
            super().__init__()
            self.inner = inner

        def _on_step(self) -> bool:
            return self.inner.on_step()

    cb = _Adapter(eta_cb)

    # Train
    model.learn(total_timesteps=int(total_timesteps), callback=cb)

    # Close tqdm bar if used
    try:
        eta_cb.close()
    except Exception:
        pass

    # Validation evaluation (single env)
    val_env = OilHedgingDailyEnv(pre, val_scenarios, cfg=env_cfg, seed=seed + 999)
    print("\n✅ TRAINING DONE — starting evaluation...")
    print(f"\n🔎 EVAL: validation (oracle_universe) | episodes={n_val_eval:,}")
    val_eps = run_episodes_with_progress(
        val_env,
        model,
        n_episodes=n_val_eval,
        deterministic=True,
        seed=seed + 777,
        desc=f"{out_dir.name} | val",
        enable_pbar=bool(cli_live),
    )
    val_mean = float(np.mean([x["reward_sum"] for x in val_eps])) if val_eps else float("nan")

    # Save best (for now: single checkpoint)
    model_path = out_dir / "model_best.zip"
    model.save(str(model_path))

    # Test evaluation
    test_env = OilHedgingDailyEnv(pre, test_scenarios, cfg=env_cfg, seed=seed + 2020)
    print(f"\n🔎 EVAL: test (oracle_universe) | episodes={n_test_eval:,}")
    test_eps = run_episodes_with_progress(
        test_env,
        model,
        n_episodes=n_test_eval,
        deterministic=True,
        seed=seed + 2021,
        desc=f"{out_dir.name} | test:univ",
        enable_pbar=bool(cli_live),
    )

    # Evaluate on oracle_all and baseline test scenario sets
    test_env_oa = OilHedgingDailyEnv(pre, test_scenarios_oracle_all, cfg=env_cfg, seed=seed + 3030)
    print(f"\n🔎 EVAL: test (oracle_all) | episodes={n_oa_eval:,}")
    test_eps_oa = run_episodes_with_progress(
        test_env_oa,
        model,
        n_episodes=n_oa_eval,
        deterministic=True,
        seed=seed + 3031,
        desc=f"{out_dir.name} | test:all",
        enable_pbar=bool(cli_live),
    )

    test_env_bl = OilHedgingDailyEnv(pre, test_scenarios_baseline, cfg=env_cfg, seed=seed + 4040)
    print(f"\n🔎 EVAL: test (baseline) | episodes={n_bl_eval:,}")
    test_eps_bl = run_episodes_with_progress(
        test_env_bl,
        model,
        n_episodes=n_bl_eval,
        deterministic=True,
        seed=seed + 4041,
        desc=f"{out_dir.name} | test:base",
        enable_pbar=bool(cli_live),
    )

    # Save unified test episodes Parquet
    def _eps_df(
        eps: List[Dict[str, Any]],
        dataset: str,
        scenarios: List[Dict[str, Any]],
        split: str,
    ) -> pd.DataFrame:
        """Build a unified episode-level dataframe and attach scenario metadata.

        Important: `scenario_idx` emitted by the env is the index into the `scenarios` list
        passed to the env constructor. We exploit that to attach `scenario_id` and other
        scenario columns so downstream evaluation can do exact scenario-level joins.
        """
        df = pd.DataFrame(eps)
        if df.empty:
            df = pd.DataFrame(columns=[
                "scenario_idx",
                "Q",
                "steps",
                "reward_sum",
                "pnl_net_sum",
                "cost_sum",
                "turnover_contract",
                "mdd",
            ])

        # Force a clean dataset label
        ds = str(dataset)
        if ds not in ("oracle_universe", "oracle_all", "baseline"):
            ds = "unknown"

        # Attach scenario metadata (scenario_id, tag, oracle_series, dates, etc.)
        # Build lookup arrays to keep it fast (no pandas apply loops).
        n_sc = len(scenarios)
        def _get(i: int, key: str, default=None):
            if 0 <= i < n_sc:
                return scenarios[i].get(key, default)
            return default

        # Ensure scenario_idx exists and is int
        if "scenario_idx" not in df.columns:
            df["scenario_idx"] = -1
        df["scenario_idx"] = pd.to_numeric(df["scenario_idx"], errors="coerce").fillna(-1).astype(int)

        # Vectorized-ish mapping via list comprehensions (still fast for ~thousands rows)
        idxs = df["scenario_idx"].to_numpy(dtype=int, copy=False)
        df["scenario_id"] = [_get(int(i), "scenario_id", None) for i in idxs]
        df["tag"] = [_get(int(i), "tag", None) for i in idxs]
        # oracle_series exists only in oracle_all; keep None otherwise
        df["oracle_series"] = [_get(int(i), "oracle_series", None) for i in idxs]
        df["oracle_pool"] = [_get(int(i), "oracle_pool", None) for i in idxs]
        df["oracle_candidate"] = [_get(int(i), "oracle_candidate", None) for i in idxs]
        df["company_id"] = [_get(int(i), "company_id", None) for i in idxs]
        df["company_size"] = [_get(int(i), "company_size", None) for i in idxs]
        df["start_idx"] = [_get(int(i), "start_idx", None) for i in idxs]
        df["end_idx"] = [_get(int(i), "end_idx", None) for i in idxs]
        df["start_date_int"] = [_get(int(i), "start_date_int", None) for i in idxs]
        df["end_date_int"] = [_get(int(i), "end_date_int", None) for i in idxs]
        df["horizon_days_target"] = [_get(int(i), "horizon_days_target", None) for i in idxs]
        df["horizon_days_realized"] = [_get(int(i), "horizon_days_realized", None) for i in idxs]

        # Core identifiers
        df.insert(0, "split", str(split))
        df.insert(0, "dataset", ds)
        df.insert(0, "strategy", "RL_PPO")
        df.insert(0, "exposure_id", str(exposure_id))

        # Window metadata
        for k, v in (window_meta or {}).items():
            df[k] = v

        return df

    df_test = _eps_df(test_eps, "oracle_universe", test_scenarios, "test")
    df_oa = _eps_df(test_eps_oa, "oracle_all", test_scenarios_oracle_all, "test")
    df_bl = _eps_df(test_eps_bl, "baseline", test_scenarios_baseline, "test")

    df_all = pd.concat([df_test, df_oa, df_bl], ignore_index=True)
    # Sanity: in evaluation, we rely on scenario_id to do exact joins.
    n_missing_sid = int(df_all["scenario_id"].isna().sum()) if "scenario_id" in df_all.columns else len(df_all)
    if n_missing_sid > 0:
        print(f"⚠️  WARNING: missing scenario_id for {n_missing_sid}/{len(df_all)} RL episode rows (check scenario_idx mapping)")
    df_all.to_parquet(out_dir / "test_episodes_all.parquet", index=False)

    # Save summaries
    # Aggregate episode metrics for quick CLI summaries
    def _mean(eps: List[Dict[str, Any]], key: str) -> float:
        if not eps:
            return float("nan")
        vals = [float(x.get(key, float("nan"))) for x in eps]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    test_reward_mean = _mean(test_eps, "reward_sum")
    test_pnl_mean = _mean(test_eps, "pnl_net_sum")
    test_cost_mean = _mean(test_eps, "cost_sum")
    test_mdd_mean = _mean(test_eps, "mdd")

    test_reward_mean_oa = _mean(test_eps_oa, "reward_sum")
    test_pnl_mean_oa = _mean(test_eps_oa, "pnl_net_sum")
    test_cost_mean_oa = _mean(test_eps_oa, "cost_sum")
    test_mdd_mean_oa = _mean(test_eps_oa, "mdd")

    test_reward_mean_bl = _mean(test_eps_bl, "reward_sum")
    test_pnl_mean_bl = _mean(test_eps_bl, "pnl_net_sum")
    test_cost_mean_bl = _mean(test_eps_bl, "cost_sum")
    test_mdd_mean_bl = _mean(test_eps_bl, "mdd")

    window_summary = {
        "val_mean_reward": float(val_mean),
        "test_reward_mean": float(test_reward_mean),
        "test_pnl_mean": float(test_pnl_mean),
        "test_cost_mean": float(test_cost_mean),
        "test_mdd_mean": float(test_mdd_mean),
        "test_reward_mean_oracle_all": float(test_reward_mean_oa),
        "test_pnl_mean_oracle_all": float(test_pnl_mean_oa),
        "test_cost_mean_oracle_all": float(test_cost_mean_oa),
        "test_mdd_mean_oracle_all": float(test_mdd_mean_oa),
        "test_reward_mean_baseline": float(test_reward_mean_bl),
        "test_pnl_mean_baseline": float(test_pnl_mean_bl),
        "test_cost_mean_baseline": float(test_cost_mean_bl),
        "test_mdd_mean_baseline": float(test_mdd_mean_bl),
        "n_eval_episodes": int(eval_episodes),
        "device": str(device),
        "risk_mode": str(env_cfg.risk_mode),
        "eta_cost": float(env_cfg.eta_cost),
        "n_envs": int(n_envs),
        "timesteps": int(total_timesteps),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "net_arch": list(net_arch),
    }

    # Backward-compatible file name
    with open(out_dir / "val_summary.json", "w", encoding="utf-8") as f:
        json.dump({"val_mean_reward": float(val_mean), "n_eval_episodes": int(eval_episodes), "device": str(device)}, f, ensure_ascii=False, indent=2)

    with open(out_dir / "window_summary.json", "w", encoding="utf-8") as f:
        json.dump(window_summary, f, ensure_ascii=False, indent=2)

    # Save test episodes CSV for backward-compatibility
    import csv

    csv_path = out_dir / "test_episodes.csv"
    if test_eps:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(test_eps[0].keys()))
            w.writeheader()
            for row in test_eps:
                w.writerow(row)

    # Clean up vec env
    try:
        venv.close()
    except Exception:
        pass

    return window_summary


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward PPO training for oil hedging")
    p.add_argument("--cache", required=True, help="Path to precompute_<EXPOSURE>.npz")
    p.add_argument("--out_dir", default="rl_runs", help="Output directory")
    p.add_argument("--year_start", type=int, default=None, help="First year for windows (optional)")
    p.add_argument("--year_end", type=int, default=None, help="Last year for windows (optional)")

    # Scenario settings
    p.add_argument("--scenario_mode", choices=["baseline", "random"], default="random")  # (deprecated)
    p.add_argument("--stride_days", type=int, default=5, help="Start-date stride in calendar days")  # (deprecated)
    p.add_argument("--hold_days", type=int, default=20, help="Fixed holding period for baseline")  # (deprecated)
    p.add_argument("--hold_min", type=int, default=20)  # (deprecated)
    p.add_argument("--hold_max", type=int, default=40)  # (deprecated)
    p.add_argument("--vol_min", type=float, default=1_000_000.0)  # (deprecated)
    p.add_argument("--vol_max", type=float, default=2_000_000.0)  # (deprecated)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--exposure_id", required=True, choices=["WTI_SPOT","BRENT_SPOT","OPEC_BASKET"], help="Exposure id")
    p.add_argument("--scenario_dir", default="scenarios", help="Base scenarios directory (contains <EXPOSURE>/oracle_universe.parquet etc.)")
    p.add_argument("--train_mode", choices=["rolling","expanding","hybrid_expanding"], default="hybrid_expanding")
    p.add_argument("--max_train_scenarios", type=int, default=0, help="Cap train scenarios per window (0=no cap)")
    p.add_argument("--max_eval_scenarios", type=int, default=0, help="Cap eval scenarios per window (0=no cap)")

    # RL settings
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu", help="Training device")
    p.add_argument("--vec", choices=["auto", "dummy", "subproc"], default="auto", help="VecEnv backend")
    p.add_argument("--torch_threads", type=int, default=0, help="Set torch num threads (0=leave default)")
    p.add_argument("--torch_interop_threads", type=int, default=0, help="Set torch interop threads (0=leave default)")
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--eval_episodes", type=int, default=200)
    p.add_argument(
        "--dummy_policy",
        choices=["none", "hold", "naive", "nohedge"],
        default="none",
        help="Skip training and run evaluation with a fixed dummy policy (smoke-test) (hold=do-nothing delta=0).",
    )
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, load each WF_*/model_best.zip checkpoint, and re-evaluate the requested scenarios.",
    )
    p.add_argument(
        "--eval_scope",
        choices=["test_year", "all_until_test", "all_available"],
        default="test_year",
        help=(
            "Scenario coverage used only in --eval_only mode. "
            "test_year=fair OOS test year; all_until_test=all scenarios up to each test-year end; "
            "all_available=all scenarios in the cache date range."
        ),
    )

    p.add_argument("--eta_print_freq", type=int, default=50_000, help="Print training ETA every N timesteps")
    p.add_argument("--n_steps", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--net_arch", type=str, default="256,256", help="MLP hidden sizes, e.g. '128,128'")

    # PPO hyperparameters (keep defaults aligned with SB3/common defaults)
    p.add_argument("--ppo_gamma", type=float, default=0.99)
    p.add_argument("--ppo_gae_lambda", type=float, default=0.95)
    p.add_argument("--ppo_ent_coef", type=float, default=0.0)
    p.add_argument("--ppo_vf_coef", type=float, default=0.5)
    p.add_argument("--ppo_max_grad_norm", type=float, default=0.5)
    p.add_argument("--ppo_n_epochs", type=int, default=10)
    p.add_argument("--ppo_learning_rate", type=float, default=3e-4)
    p.add_argument("--ppo_clip_range", type=float, default=0.2)
    p.add_argument("--ppo_target_kl", type=float, default=0.0, help="0 disables target_kl")

    # Env config
    p.add_argument("--h_max", type=float, default=2.0)
    p.add_argument(
    "--delta_h_max",
    type=float,
    default=0.10,
    help="Maximum absolute one-step hedge-ratio change for delta_h_discrete actions",
    )
    p.add_argument(
        "--delta_h_step",
        type=float,
        default=0.05,
        help="Step size for delta_h_discrete action grid",
    )
    p.add_argument("--action_mode", choices=["delta_h_discrete", "h_levels"], default="delta_h_discrete")
    p.add_argument("--risk_mode", choices=["none", "quad", "lpm", "rollvar_lpm"], default="rollvar_lpm")
    p.add_argument("--lambda_var", type=float, default=1.0, help="Used when risk_mode=quad")
    p.add_argument("--lambda_lpm", type=float, default=1.0, help="Used when risk_mode=lpm")
    p.add_argument("--lpm_order", type=int, default=2)
    p.add_argument("--lpm_target", type=float, default=0.0)
    p.add_argument("--mu_pnl", type=float, default=0.10, help="Small profit incentive in reward")
    p.add_argument("--lambda_rollvar", type=float, default=5.0, help="Rolling variance penalty weight (risk_mode=rollvar_lpm)")
    p.add_argument("--roll_var_L", type=int, default=20, help="Rolling variance window length L (risk_mode=rollvar_lpm)")
    p.add_argument("--eta_cost", type=float, default=1.0)
    p.add_argument("--info_mode", choices=["train", "eval"], default="train")

    # Parallelization
    p.add_argument("--parallel_windows", type=int, default=1, help="Run N walk-forward windows in parallel (processes)")
    p.add_argument("--max_windows", type=int, default=0, help="If >0, only run the first N windows (for quick tests)")

    # Expanding/hybrid-expanding cap
    p.add_argument("--expanding_max_years", type=int, default=8, help="Max train length in years for hybrid_expanding (cap) and expanding (optional cap)")

    return p.parse_args()



def merge_all_window_parquets(base_out: Path) -> None:
    paths = sorted(base_out.glob("WF_*/test_episodes_all.parquet"))
    if not paths:
        return
    dfs = []
    for pth in paths:
        try:
            dfs.append(pd.read_parquet(pth))
        except Exception:
            continue
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(base_out / "results_all_windows.parquet", index=False)


def main() -> None:
    args = parse_args()

    # Torch threading
    try:
        import torch
        if int(args.torch_threads) > 0:
            torch.set_num_threads(int(args.torch_threads))
        if int(args.torch_interop_threads) > 0:
            torch.set_num_interop_threads(int(args.torch_interop_threads))
    except Exception:
        pass

    # Parse net_arch
    net_arch = tuple(int(x.strip()) for x in str(args.net_arch).split(",") if x.strip())
    if not net_arch:
        net_arch = (256, 256)

    device = _select_device(args.device)
    if device == "mps":
        print("⚠️  NOTE: PPO MLP on MPS is not GPU-efficient. CPU may be equally fast.")

    # Load once here for year range discovery; workers will reload cache.
    pre_res = load_npz(args.cache)

    years = year_of_intdays(pre_res.dates_int)
    y_min = int(years.min())
    y_max = int(years.max())

    year_start = int(args.year_start) if args.year_start is not None else max(y_min, y_min + 5)
    year_end = int(args.year_end) if args.year_end is not None else y_max

    windows = walkforward_windows(year_start, year_end)
    if int(args.max_windows) > 0:
        windows = windows[: int(args.max_windows)]
    if not windows:
        raise SystemExit(f"No walk-forward windows possible in [{year_start}, {year_end}]. Available years: {y_min}..{y_max}")

    print(f"[walkforward] exposure cache: {args.cache}")
    print(f"[walkforward] years available: {y_min}..{y_max} | windows: {len(windows)} | step=1y | train=2y val=1y test=1y")
    print(f"[walkforward] device={device} vec={args.vec} n_envs={args.n_envs} n_steps={args.n_steps} batch={args.batch_size} net_arch={net_arch}")
    print(
        f"[walkforward] risk_mode={args.risk_mode} mu_pnl={args.mu_pnl} "
        f"lambda_rollvar={args.lambda_rollvar} L={args.roll_var_L} "
        f"lambda_lpm={args.lambda_lpm} order={args.lpm_order} target={args.lpm_target} eta_cost={args.eta_cost}"
    )
    if int(args.parallel_windows) > 1:
        print("[walkforward] NOTE: parallel window training forces vec=dummy inside workers to avoid nested multiprocessing")

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Save run config
    with open(base_out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    jobs: List[Dict[str, Any]] = []

    anchor_train_y0 = int(year_start)
    max_train_years = int(args.expanding_max_years)
    if max_train_years <= 0:
        max_train_years = 8

    for wi, (train_y0, train_y1, val_y, test_y) in enumerate(windows, start=1):
        train_mode = str(args.train_mode)
        train_y0_eff = int(train_y0)
        train_y1_eff = int(train_y1)
        if train_mode == "expanding":
            train_y0_eff = int(anchor_train_y0)
        elif train_mode == "hybrid_expanding":
            # expanding until max_train_years, then rolling with fixed length
            train_y0_eff = int(max(anchor_train_y0, train_y1_eff - (max_train_years - 1)))
        # rolling keeps original train_y0/train_y1
        train_y0 = train_y0_eff
        train_y1 = train_y1_eff

        train_start_int, train_end_int = year_range_to_int_bounds(train_y0, train_y1)
        val_start_int, val_end_int = year_range_to_int_bounds(val_y, val_y)
        test_start_int, test_end_int = year_range_to_int_bounds(test_y, test_y)

        win_dir = base_out / f"WF_train{train_y0}-{train_y1}_val{val_y}_test{test_y}"
        window_name = win_dir.name

        jobs.append({
            "cache": args.cache,
            "out_dir": str(win_dir),
            "window_name": window_name,
            "parallel_windows": int(args.parallel_windows),
            "cli_live": bool(int(args.parallel_windows) <= 1),

            # window ids (used for seeding)
            "train_y0": int(train_y0),
            "train_y1": int(train_y1),
            "val_y": int(val_y),
            "test_y": int(test_y),

            # date bounds (kept for logging/debug)
            "train_start_int": int(train_start_int),
            "train_end_int": int(train_end_int),
            "val_start_int": int(val_start_int),
            "val_end_int": int(val_end_int),
            "test_start_int": int(test_start_int),
            "test_end_int": int(test_end_int),

            # scenario settings
            "scenario_mode": str(args.scenario_mode),
            "stride_days": int(args.stride_days),
            "hold_days": int(args.hold_days),
            "hold_min": int(args.hold_min),
            "hold_max": int(args.hold_max),
            "vol_min": float(args.vol_min),
            "vol_max": float(args.vol_max),
            "exposure_id": str(args.exposure_id),
            "scenario_dir": str(args.scenario_dir),
            "train_mode": str(args.train_mode),
            "expanding_max_years": int(max_train_years),
            "max_train_scenarios": int(args.max_train_scenarios),
            "max_eval_scenarios": int(args.max_eval_scenarios),

            # rl settings
            "seed": int(args.seed),
            "n_envs": int(args.n_envs),
            "timesteps": int(args.timesteps),
            "eval_episodes": int(args.eval_episodes),
            "dummy_policy": str(args.dummy_policy),
            "eval_only": bool(args.eval_only),
            "eval_scope": str(args.eval_scope),
            "eta_print_freq": int(args.eta_print_freq),
            "device": device,
            "vec": str(args.vec),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "net_arch": list(net_arch),
            "torch_threads": int(args.torch_threads),
            "torch_interop_threads": int(args.torch_interop_threads),

            "ppo_gamma": float(args.ppo_gamma),
            "ppo_gae_lambda": float(args.ppo_gae_lambda),
            "ppo_ent_coef": float(args.ppo_ent_coef),
            "ppo_vf_coef": float(args.ppo_vf_coef),
            "ppo_max_grad_norm": float(args.ppo_max_grad_norm),
            "ppo_n_epochs": int(args.ppo_n_epochs),
            "ppo_learning_rate": float(args.ppo_learning_rate),
            "ppo_clip_range": float(args.ppo_clip_range),
            "ppo_target_kl": float(args.ppo_target_kl),

            # env cfg
            "h_max": float(args.h_max),
            "delta_h_max": float(args.delta_h_max),
            "delta_h_step": float(args.delta_h_step),
            "action_mode": str(args.action_mode),
            "risk_mode": str(args.risk_mode),
            "mu_pnl": float(args.mu_pnl),
            "lambda_rollvar": float(args.lambda_rollvar),
            "roll_var_L": int(args.roll_var_L),
            "lambda_var": float(args.lambda_var),
            "lambda_lpm": float(args.lambda_lpm),
            "lpm_order": int(args.lpm_order),
            "lpm_target": float(args.lpm_target),
            "eta_cost": float(args.eta_cost),
            "info_mode": str(args.info_mode),
        })

    # Run jobs
    pw = int(args.parallel_windows)
    if pw <= 1:
        print(f"[walkforward] running windows sequentially: {len(jobs)}")
        ok = 0
        skipped = 0
        failed = 0

        it = jobs
        if tqdm is not None:
            it = tqdm(jobs, total=len(jobs), desc="walkforward windows", unit="win", dynamic_ncols=True)

        for i, job in enumerate(it, start=1):
            if tqdm is None:
                print(f"[walkforward] ({i}/{len(jobs)}) start {job['window_name']}")

            res = _run_window_job(job)
            st = res.get("status")
            if st == "ok":
                ok += 1
                if tqdm is None:
                    print(f"[walkforward] ({i}/{len(jobs)}) done  {job['window_name']}")
            elif st == "skipped":
                skipped += 1
                if tqdm is None:
                    print(
                        f"[walkforward] ({i}/{len(jobs)}) skip  {job['window_name']} | "
                        f"scenarios train/val/test={res.get('train')}/{res.get('val')}/{res.get('test')}"
                    )
            else:
                failed += 1
                if tqdm is None:
                    print(f"[walkforward] ({i}/{len(jobs)}) ERROR {job['window_name']} | {res.get('error')}")
                    print(res.get("traceback", ""))

            if tqdm is not None:
                try:
                    it.set_postfix({"ok": ok, "skip": skipped, "fail": failed})
                except Exception:
                    pass
    else:
        print(f"[walkforward] running windows in parallel: {len(jobs)} windows | workers={pw}")
        pbar = None
        if tqdm is not None:
            try:
                pbar = tqdm(total=len(jobs), desc="walkforward windows", unit="win", dynamic_ncols=True)
            except Exception:
                pbar = None
        t_global0 = time.time()
        done = 0
        ok = 0
        skipped = 0
        failed = 0
        durations: List[float] = []

        def _eta_overall() -> str:
            remaining = max(len(jobs) - done, 0)
            if not durations:
                return "--:--:--"
            avg = float(np.mean(durations))
            eta_s = (avg * remaining) / max(pw, 1)
            return _fmt_eta(eta_s)

        with ProcessPoolExecutor(max_workers=pw) as ex:
            futs = {ex.submit(_run_window_job, job): job for job in jobs}
            for fut in as_completed(futs):
                job = futs[fut]
                done += 1
                if pbar is not None:
                    try:
                        pbar.update(1)
                    except Exception:
                        pass
                try:
                    res = fut.result()
                except Exception as e:
                    failed += 1
                    print(f"[walkforward] ({done}/{len(jobs)}) ERROR {job['window_name']} | {repr(e)}")
                    continue

                st = res.get("status")
                sec = float(res.get("seconds", float("nan")))
                if np.isfinite(sec):
                    durations.append(sec)

                if st == "ok":
                    ok += 1
                    val_mean = res.get("val_mean_reward", float("nan"))
                    test_pnl = res.get("test_pnl_mean", float("nan"))
                    test_mdd = res.get("test_mdd_mean", float("nan"))
                    logp = res.get("log", "")
                    print(
                        f"[walkforward] ({done}/{len(jobs)}) DONE {job['window_name']} | "
                        f"sec={sec:.1f} | val_mean={val_mean:.4g} | test_pnl_mean={test_pnl:.4g} | "
                        f"test_mdd_mean={test_mdd:.4g} | ETA_all={_eta_overall()}"
                    )
                    if logp:
                        print(f"           log: {logp}")
                elif st == "skipped":
                    skipped += 1
                    print(
                        f"[walkforward] ({done}/{len(jobs)}) SKIP {job['window_name']} | "
                        f"train/val/test={res.get('train')}/{res.get('val')}/{res.get('test')} | "
                        f"sec={sec:.1f} | ETA_all={_eta_overall()}"
                    )
                else:
                    failed += 1
                    print(
                        f"[walkforward] ({done}/{len(jobs)}) FAIL {job['window_name']} | {res.get('error')} | "
                        f"sec={sec:.1f} | ETA_all={_eta_overall()}"
                    )
                    tb = res.get("traceback", "")
                    if tb:
                        print(tb)

                elapsed = time.time() - t_global0
                print(
                    f"[walkforward] progress: ok={ok} skip={skipped} fail={failed} "
                    f"done={done}/{len(jobs)} elapsed={_fmt_eta(elapsed)} ETA_all={_eta_overall()}"
                )
                if pbar is not None:
                    try:
                        pbar.set_postfix({"ok": ok, "skip": skipped, "fail": failed, "ETA": _eta_overall()})
                    except Exception:
                        pass
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    # Merge all per-window parquets into one (for downstream baselines comparison)
    merge_all_window_parquets(base_out)


if __name__ == "__main__":
    main()