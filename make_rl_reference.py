#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_rl_reference.py

Re-evaluate a trained SB3 model on scenario parquets and export RL results WITH scenario_id
so Baseline_report can align and compare apples-to-apples.

Outputs:
  <out_dir>/rl_reference_<EXPOSURE>.parquet
  <out_dir>/rl_reference_<EXPOSURE>.csv

Example:
  python make_rl_reference.py \
    --run_dir rl_runs/OPEC_QUICK \
    --cache rl_cache/precompute_OPEC_BASKET.npz \
    --scenario_dir scenarios \
    --exposure_id OPEC_BASKET \
    --out_dir rl_runs/OPEC_QUICK
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from rl.precompute import load_npz
from rl.env_daily import OilHedgingDailyEnv, EnvConfig
from rl.scenario_loader import load_scenarios_from_parquet, ScenarioLoaderConfig


# ----------------------------
# Helpers
# ----------------------------

WIN_RE = re.compile(r"WF_train(?P<tr0>\d{4})-(?P<tr1>\d{4})_val(?P<va>\d{4})_test(?P<te>\d{4})", re.IGNORECASE)

def find_model_zip(run_dir: Path) -> Optional[Path]:
    # try common names first
    candidates = []
    for pat in ["**/*.zip", "**/best_model.zip", "**/final_model.zip", "**/model.zip"]:
        candidates += list(run_dir.glob(pat))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    # choose newest
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def extract_test_years_from_run_dir(run_dir: Path) -> List[int]:
    # if folders like WF_train.... exist, parse test year
    years = []
    for p in run_dir.glob("**/*"):
        if not p.is_dir():
            continue
        m = WIN_RE.search(p.name)
        if m:
            years.append(int(m.group("te")))
    years = sorted(set(years))
    return years

def year_from_dayint(day_int: int) -> int:
    # day_int is days since 1970-01-01 in your pipeline (based on your precompute prints).
    # We don't rely on exact conversion here; scenario_loader already filters by year_start/year_end using dates_int mapping.
    raise RuntimeError("Not used")

def episode_rollup(env: OilHedgingDailyEnv, model: PPO, deterministic: bool = True) -> Dict[str, float]:
    """
    Runs ONE episode (one scenario), returns totals.
    Assumes env.reset() already called and points to desired scenario.
    """
    # Collect opening costs from reset info if present
    obs, info = env.reset()
    opening_cost = float(info.get("opening_cost", 0.0))
    opening_breakdown = info.get("opening_cost_breakdown", {}) or {}
    opening_trade = float(opening_breakdown.get("cost_trade", opening_cost))
    opening_roll = float(opening_breakdown.get("cost_roll", 0.0))

    pnl_net_total = 0.0
    cost_trade_total = opening_trade
    cost_roll_total = opening_roll

    turnover_contracts = 0.0
    max_abs_contracts = 0.0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, terminated, truncated, inf = env.step(action)

        # per-step pnl_net (your env prints show pnl_net exists)
        pnl_net_total += float(inf.get("pnl_net", 0.0))

        # per-step cost breakdown
        cb = inf.get("cost_breakdown", {}) or {}
        # IMPORTANT: some versions store totals inconsistently; we sum trade+roll separately
        cost_trade_total += float(cb.get("cost_trade", 0.0))
        cost_roll_total += float(cb.get("cost_roll", 0.0))

        # turnover proxy from contract changes if present
        if "n_used" in inf and "n_new" in inf:
            n_used = float(inf.get("n_used", 0.0))
            n_new = float(inf.get("n_new", 0.0))
            turnover_contracts += abs(n_new - n_used)
            max_abs_contracts = max(max_abs_contracts, abs(n_new))

    return {
        "net_pnl_total": pnl_net_total,
        "cost_trade_total": cost_trade_total,
        "cost_roll_total": cost_roll_total,
        "turnover_contracts": turnover_contracts,
        "max_abs_contracts": max_abs_contracts,
    }


def load_dataset_scenarios(scenario_dir: Path, exposure_id: str, dataset: str,
                          pre_dates_int: np.ndarray, year: int,
                          max_scenarios: Optional[int] = None, seed: int = 2026) -> List[Dict]:
    """
    dataset in: oracle_universe / oracle_all / baseline
    Returns list of scenario dicts as expected by env.
    """
    # file mapping
    if dataset == "oracle_universe":
        rel = f"{exposure_id}/oracle_universe.parquet"
    elif dataset == "oracle_all":
        rel = f"{exposure_id}/oracle_all.parquet"
    elif dataset == "baseline":
        rel = f"{exposure_id}/baseline.parquet"
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    path = scenario_dir / rel
    if not path.exists():
        raise FileNotFoundError(f"scenario file not found: {path}")

    cfg = ScenarioLoaderConfig(
        require_ok_coverage=True,
        allow_shortened=False,
        max_scenarios=max_scenarios,
        seed=seed,
    )

    sc = load_scenarios_from_parquet(
        str(path),
        dates_int=pre_dates_int,
        exposure_id=exposure_id,
        year_start=year,
        year_end=year,
        cfg=cfg,
    )
    return sc


# ----------------------------
# Main
# ----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="RL run folder (where model zip is located)")
    ap.add_argument("--cache", required=True, help="precompute npz path, e.g. rl_cache/precompute_OPEC_BASKET.npz")
    ap.add_argument("--scenario_dir", default="scenarios", help="scenarios root dir")
    ap.add_argument("--exposure_id", required=True, choices=["WTI_SPOT", "BRENT_SPOT", "OPEC_BASKET"])
    ap.add_argument("--out_dir", required=True, help="output dir for rl_reference parquet/csv")
    ap.add_argument("--device", default="cpu", choices=["cpu","mps"], help="SB3 device")
    ap.add_argument("--years", default=None, help="Comma-separated test years (optional). If not provided, infer from run_dir.")
    ap.add_argument("--max_scenarios", type=int, default=None, help="Optional cap for faster runs")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic policy for eval")
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    scenario_dir = Path(args.scenario_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = find_model_zip(run_dir)
    if model_path is None:
        raise SystemExit(f"[ERR] No SB3 model .zip found under: {run_dir}. Make sure train saves a model.")

    print(f"[OK] model: {model_path}")

    pre = load_npz(args.cache)
    dates_int = pre.dates_int  # numpy int array

    # decide years
    if args.years:
        years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    else:
        years = extract_test_years_from_run_dir(run_dir)
        if not years:
            # fallback: just do a sane default single year if we can't infer
            years = [2011]
            print("[WARN] could not infer test years from run_dir. Falling back to 2011. Use --years to set explicitly.")
    print(f"[OK] test years: {years}")

    # load model
    model = PPO.load(str(model_path), device=args.device)
    print(f"[OK] loaded model on device={args.device}")

    rows = []
    env_cfg = EnvConfig(info_mode="eval")

    for year in years:
        for dataset in ["oracle_universe", "oracle_all", "baseline"]:
            sc = load_dataset_scenarios(
                scenario_dir=scenario_dir,
                exposure_id=args.exposure_id,
                dataset=dataset,
                pre_dates_int=dates_int,
                year=year,
                max_scenarios=args.max_scenarios,
                seed=args.seed,
            )
            print(f"[YEAR {year}] dataset={dataset} scenarios={len(sc)}")

            if not sc:
                continue

            env = OilHedgingDailyEnv(pre, sc, cfg=env_cfg, seed=args.seed)

            # evaluate each scenario
            for j in range(len(sc)):
                # force env to scenario j (your env increments scenario_idx internally on reset; simplest: set then reset)
                env._scenario_idx = j  # noqa: relies on your env internal; if name differs, set via a method.
                obs, info = env.reset()

                # scenario identifiers from loaded scenario dict
                sid = sc[j].get("scenario_id")
                if sid is None:
                    # if scenario_loader didn't keep it, we cannot compare properly
                    raise SystemExit("[ERR] scenario_id missing in scenarios. Fix scenario_loader to keep scenario_id.")
                oracle_series = sc[j].get("oracle_series", None)

                totals = episode_rollup(env, model, deterministic=bool(args.deterministic))

                rows.append({
                    "exposure_id": args.exposure_id,
                    "dataset": dataset,
                    "scenario_id": str(sid),
                    "oracle_series": oracle_series,
                    "test_year": int(year),
                    "net_pnl_total": float(totals["net_pnl_total"]),
                    "cost_trade_total": float(totals["cost_trade_total"]),
                    "cost_roll_total": float(totals["cost_roll_total"]),
                    "turnover_contracts": float(totals["turnover_contracts"]),
                    "max_abs_contracts": float(totals["max_abs_contracts"]),
                    # locked tags for report schema
                    "strategy": "RL_PPO",
                    "mode": "dynamic",
                    "roll": "roll",
                })

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("[ERR] No rows produced. Check years/scenario files/model path.")

    out_path_pq = out_dir / f"rl_reference_{args.exposure_id}.parquet"
    out_path_csv = out_dir / f"rl_reference_{args.exposure_id}.csv"
    out.to_parquet(out_path_pq, index=False)
    out.to_csv(out_path_csv, index=False)

    print(f"[DONE] wrote: {out_path_pq}")
    print(f"[DONE] wrote: {out_path_csv}")
    print("[INFO] Now run Baseline_report with --rl_reference and --include_rl for apples-to-apples comparison.")


if __name__ == "__main__":
    main()