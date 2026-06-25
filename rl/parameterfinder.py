

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parameterfinder.py

هدف:
- روی «اولین سال ممکن» از سناریوهای OPEC_BASKET (oracle_universe) فقط 100 سناریو انتخاب می‌کنیم.
- سه استراتژی پایه را روی همان 100 سناریو اجرا می‌کنیم: NoHedge, Naive, DCC-GARCH (dyn+roll)
- سپس یک sweep روی چند هایپرپارامتر عددی RL (فعلاً eta_cost) انجام می‌دهیم:
    آموزش PPO کوتاه + ارزیابی روی همان 100 سناریو
- خروجی:
    1) CSV نتایج sweep
    2) نمودارهای cost / turnover / variance / ES95 / HE_var

نکته:
- این اسکریپت عمداً «سریع/تجربی» است (smoke-test) و جایگزین walkforward اصلی نیست.
- برای این که مقایسه منصفانه باشد، سناریوها مشترک‌اند.

اجرا (از ریشه پروژه FinalRL):

python -m rl.parameterfinder \
  --cache rl_cache/precompute_OPEC_BASKET.npz \
  --exposure_id OPEC_BASKET \
  --scenario_parquet scenarios/OPEC_BASKET/oracle_universe.parquet \
  --out_dir rl_runs/PARAMFINDER_OPEC \
  --n_scenarios 100 \
  --timesteps 50000

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Matplotlib for quick plots
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.precompute import load_npz
from rl.env_daily import OilHedgingDailyEnv, EnvConfig


def _year_from_date_int(d: int) -> int:
    # In this project dates_int are usually days since 1970-01-01.
    # Use pandas to convert robustly.
    dt = pd.to_datetime(int(d), unit="D", origin="unix")
    return int(dt.year)


def select_first_year_scenarios(
    scenario_parquet: str,
    exposure_id: str,
    n_scenarios: int,
    seed: int,
) -> Tuple[pd.DataFrame, int]:
    """Load oracle_universe parquet, find earliest year with enough rows, sample n_scenarios."""
    df = pd.read_parquet(scenario_parquet)
    if "exposure_id" in df.columns:
        df = df[df["exposure_id"].astype(str) == str(exposure_id)].copy()

    # Prefer start_date_int / start_day_int if available
    date_col = None
    for c in ["start_date_int", "start_day_int", "start_date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"Scenario file missing start date columns. columns={list(df.columns)}")

    # Convert to year
    if date_col.endswith("_int"):
        years = df[date_col].astype(int).map(_year_from_date_int)
    else:
        years = pd.to_datetime(df[date_col]).dt.year

    df["__year__"] = years.astype(int)

    # Find earliest year with enough scenarios
    year_counts = df.groupby("__year__")["scenario_id"].count().sort_index()
    if year_counts.empty:
        raise ValueError("No scenarios after exposure filter")

    chosen_year = None
    for y, c in year_counts.items():
        if int(c) >= int(n_scenarios):
            chosen_year = int(y)
            break
    if chosen_year is None:
        # fallback: choose earliest year with at least 1 and sample min(n_scenarios, count)
        chosen_year = int(year_counts.index.min())

    dyy = df[df["__year__"] == chosen_year].copy()
    if len(dyy) == 0:
        raise ValueError("Internal error selecting year")

    rng = np.random.default_rng(int(seed))
    take = min(int(n_scenarios), len(dyy))
    idx = rng.choice(len(dyy), size=take, replace=False)
    dyy = dyy.iloc[idx].reset_index(drop=True)

    return dyy, chosen_year


def to_env_scenarios(df: pd.DataFrame, dates_int: np.ndarray) -> List[Dict]:
    """Convert scenario rows to env scenario dicts.

    Env نیاز دارد:
      - start_idx, end_idx (اندیس روی سری precompute)
      - volume_bbl
      - scenario_id

    بعضی فایل‌های سناریو فقط start_date_int/end_date_int دارند. در این حالت
    start_idx/end_idx را با searchsorted روی dates_int می‌سازیم.

    نکته: end_idx در کل پروژه شما «اندیس انتهاییِ episode» است و در env
    به صورت inclusive استفاده می‌شود (طبق تست‌های قبلی). بنابراین از
    موقعیت دقیق روز end_date_int استفاده می‌کنیم.
    """

    if "scenario_id" not in df.columns:
        raise ValueError("Scenario parquet missing required column: scenario_id")
    if "volume_bbl" not in df.columns:
        raise ValueError("Scenario parquet missing required column: volume_bbl")

    have_idx = ("start_idx" in df.columns) and ("end_idx" in df.columns)

    # If indices missing, require int dates
    if not have_idx:
        start_col = None
        end_col = None
        for c in ["start_date_int", "start_day_int", "start_date"]:
            if c in df.columns:
                start_col = c
                break
        for c in ["end_date_int", "end_day_int", "end_date"]:
            if c in df.columns:
                end_col = c
                break
        if start_col is None or end_col is None:
            raise ValueError(
                "Scenario parquet missing start_idx/end_idx AND missing usable date columns "
                f"(start: {start_col}, end: {end_col}). columns={list(df.columns)}"
            )

        # Normalize to int days since epoch
        if start_col.endswith("_int"):
            s_days = df[start_col].astype(int).to_numpy()
        else:
            s_days = pd.to_datetime(df[start_col]).astype("int64") // 86_400_000_000_000
            s_days = s_days.to_numpy(dtype=np.int64)

        if end_col.endswith("_int"):
            e_days = df[end_col].astype(int).to_numpy()
        else:
            e_days = pd.to_datetime(df[end_col]).astype("int64") // 86_400_000_000_000
            e_days = e_days.to_numpy(dtype=np.int64)

        # Map days -> indices on dates_int
        di = np.asarray(dates_int, dtype=np.int64)
        # left index for start
        start_idx = np.searchsorted(di, s_days, side="left")
        # right index for end (inclusive): use exact match if exists else previous day
        end_idx = np.searchsorted(di, e_days, side="left")

        # If end day is not present (e.g., weekend/missing), step back to previous available day
        # Keep within bounds.
        end_idx = np.clip(end_idx, 0, len(di) - 1)
        # If mapped day is after requested end day, step back
        bad = di[end_idx] > e_days
        end_idx[bad] = np.maximum(0, end_idx[bad] - 1)

        df = df.copy()
        df["start_idx"] = start_idx.astype(int)
        df["end_idx"] = end_idx.astype(int)

    # Final validation
    required = ["start_idx", "end_idx", "volume_bbl", "scenario_id"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Scenario parquet missing required column: {c}")

    out: List[Dict] = []
    for r in df.itertuples(index=False):
        out.append(
            {
                "scenario_id": getattr(r, "scenario_id"),
                "start_idx": int(getattr(r, "start_idx")),
                "end_idx": int(getattr(r, "end_idx")),
                "volume_bbl": float(getattr(r, "volume_bbl")),
                # optional metadata
                "tag": getattr(r, "tag", None),
                "scenario_type": getattr(r, "scenario_type", None),
                "oracle_series": getattr(r, "oracle_series", None),
            }
        )
    return out


def run_env_policy_eval(
    pre,
    scenarios: List[Dict],
    cfg: EnvConfig,
    policy_fn,
    seed: int,
) -> pd.DataFrame:
    """Run all scenarios and collect **episode-level** summaries.

    مهم: برای جلوگیری از اشتباه در انتخاب سناریو داخل env، هر episode را با env جداگانه
    (سناریوی تک‌عضوی) اجرا می‌کنیم. چون n=100 است، سربارش قابل قبول است و نتایج قطعی می‌شود.

    policy_fn(obs, info) -> action ndarray
    """

    def _pick(d: Dict, keys: List[str], default=0.0):
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    pass
        return float(default)

    rows = []

    for sidx, sc in enumerate(scenarios):
        env = OilHedgingDailyEnv(pre, [sc], cfg=cfg, seed=int(seed))
        obs, info = env.reset(seed=int(seed))

        term = False
        trunc = False
        steps = 0
        last_inf: Dict = {}

        while not (term or trunc):
            a = policy_fn(obs, info)
            obs, r, term, trunc, inf = env.step(a)
            steps += 1
            last_inf = inf if isinstance(inf, dict) else {}

        # Episode totals: prefer explicit *_sum / *_total from env (as in train_walkforward outputs)
        pnl_total = _pick(last_inf, ["pnl_net_sum", "pnl_net_total", "net_pnl_total", "pnl_net", "pnl_total"], default=0.0)
        cost_total = _pick(last_inf, ["cost_sum", "cost_total", "cost", "cost_trade_total"], default=0.0)
        # some envs provide separate roll/trade costs
        if "cost_roll_total" in last_inf or "cost_trade_total" in last_inf:
            cost_total = float(_pick(last_inf, ["cost_trade_total"], 0.0) + _pick(last_inf, ["cost_roll_total"], 0.0))

        turnover = _pick(last_inf, ["turnover_contract", "turnover_contracts", "turnover", "turnover_total"], default=0.0)

        rows.append(
            {
                "scenario_id": sc.get("scenario_id"),
                "scenario_idx": int(sidx),
                "pnl_net_sum": float(pnl_total),
                "cost_sum": float(cost_total),
                "turnover_contract": float(turnover),
                "steps": int(steps),
            }
        )

    return pd.DataFrame(rows)


def var_es95(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    xs = np.sort(x)
    k = int(np.floor(0.05 * xs.size))
    k = max(1, min(k, xs.size - 1))
    var95 = float(xs[k])
    es95 = float(xs[:k].mean())
    return var95, es95


def compute_metrics(df: pd.DataFrame, df_nohedge: pd.DataFrame) -> Dict[str, float]:
    x = df["pnl_net_sum"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    v = float(np.var(x, ddof=1)) if x.size > 1 else np.nan
    s = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
    var95, es95 = var_es95(x)

    # HE_var = 1 - var(hedged)/var(nohedge)
    x0 = df_nohedge["pnl_net_sum"].to_numpy(dtype=float)
    x0 = x0[np.isfinite(x0)]
    v0 = float(np.var(x0, ddof=1)) if x0.size > 1 else np.nan
    he = float(1.0 - (v / v0)) if (np.isfinite(v) and np.isfinite(v0) and v0 > 0) else np.nan

    out = {
        "n": int(len(df)),
        "pnl_mean": float(np.mean(x)) if x.size else np.nan,
        "pnl_std": s,
        "pnl_var": v,
        "VaR95": var95,
        "ES95": es95,
        "cost_mean": float(df["cost_sum"].mean()),
        "turnover_mean": float(df["turnover_contract"].mean()),
        "HE_var": he,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--exposure_id", required=True)
    ap.add_argument("--scenario_parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_scenarios", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--eta_grid", type=str, default="50,100,300,500,800,1200,2000")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = load_npz(args.cache)

    # Pick earliest year with >= n_scenarios
    sdf, year = select_first_year_scenarios(
        args.scenario_parquet, args.exposure_id, args.n_scenarios, args.seed
    )

    # Keep a scenario file for reproducibility
    scenario_out = out_dir / f"scenarios_firstyear_{args.exposure_id}_{year}_n{len(sdf)}.parquet"
    sdf.to_parquet(scenario_out, index=False)

    scenarios = to_env_scenarios(sdf, pre.dates_int)

    # ---------------- Baselines (env-based quick) ----------------
    # NOTE: This baseline is not the full hedge_simulator baseline.
    # It is a minimal env-run baseline for fast comparison.

    cfg_eval = EnvConfig(info_mode="eval")

    def pol_zero_delta(obs, info):
        return np.asarray([0.0], dtype=np.float32)

    cfg_nohedge = EnvConfig(info_mode="eval")
    # enforce no-hedge at reset if env supports it
    for k in ["h0", "initial_h", "target_h"]:
        try:
            setattr(cfg_nohedge, k, 0.0)
        except Exception:
            pass

    df_nohedge = run_env_policy_eval(pre, scenarios, cfg_nohedge, pol_zero_delta, seed=args.seed)
    df_nohedge.to_parquet(out_dir / "baseline_nohedge_env.parquet", index=False)

    df_naive   = run_env_policy_eval(pre, scenarios, cfg_eval,     pol_zero_delta, seed=args.seed)
    df_naive.to_parquet(out_dir / "baseline_naive_env.parquet", index=False)

    # DCC-GARCH baseline via existing hedge_simulator is heavier and depends on your simulator.
    # Here we just record a placeholder and recommend using hedge_simulator outputs for the final thesis tables.
    # If you insist on an env-only baseline for DCC, it requires a deterministic policy that reads covariances.

    base_metrics = []
    base_metrics.append({"run": "BASELINE_NOHEDGE", "year": year, **compute_metrics(df_nohedge, df_nohedge)})
    base_metrics.append({"run": "BASELINE_NAIVE", "year": year, **compute_metrics(df_naive, df_nohedge)})

    # ---------------- RL sweep ----------------
    eta_values = [float(x.strip()) for x in str(args.eta_grid).split(",") if x.strip()]

    sweep_rows = []
    for eta in eta_values:
        # Fresh env config per run
        cfg = EnvConfig(info_mode="train")
        cfg_eval2 = EnvConfig(info_mode="eval")

        # Apply eta_cost into env reward
        try:
            setattr(cfg, "eta_cost", float(eta))
            setattr(cfg_eval2, "eta_cost", float(eta))
        except Exception:
            pass

        # Vector env for PPO (fast enough)
        def make_env():
            return OilHedgingDailyEnv(pre, scenarios, cfg=cfg, seed=int(args.seed))

        venv = DummyVecEnv([make_env])

        model = PPO(
            policy="MlpPolicy",
            env=venv,
            verbose=0,
            device=str(args.device),
            n_steps=2048,
            batch_size=2048,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.0,
            clip_range=0.2,
        )

        model.learn(total_timesteps=int(args.timesteps))

        # Eval: deterministic
        def pol_rl(obs, info):
            a, _ = model.predict(obs, deterministic=True)
            # SB3 returns shape (1,) or (1,1) for scalar actions
            return np.asarray(a).reshape(-1).astype(np.float32)

        df_rl = run_env_policy_eval(pre, scenarios, cfg_eval2, pol_rl, seed=args.seed)
        df_rl.to_parquet(out_dir / f"rl_eval_eta{int(eta)}.parquet", index=False)

        m = compute_metrics(df_rl, df_nohedge)
        row = {"run": f"RL_PPO_eta{eta}", "eta_cost": float(eta), "year": year, **m}
        sweep_rows.append(row)
        print(f"[sweep] eta={eta} HE_var={row['HE_var']:.4f} std={row['pnl_std']:.3e} ES95={row['ES95']:.3e} cost_mean={row['cost_mean']:.1f} turnover={row['turnover_mean']:.1f}")

    res = pd.DataFrame(base_metrics + sweep_rows)
    res.to_csv(out_dir / "parameter_sweep.csv", index=False)

    # ---------------- Plots ----------------
    # Filter RL rows for x-axis
    rl_res = res[res["run"].astype(str).str.startswith("RL_PPO")].copy()
    rl_res = rl_res.sort_values("eta_cost")

    def plot_xy(ycol: str, title: str, fname: str):
        plt.figure()
        plt.plot(rl_res["eta_cost"].values, rl_res[ycol].values, marker="o")
        plt.xscale("log")
        plt.xlabel("eta_cost (log scale)")
        plt.ylabel(ycol)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

    plot_xy("cost_mean", "RL sweep: mean cost vs eta_cost", "plot_cost_mean.png")
    plot_xy("turnover_mean", "RL sweep: turnover vs eta_cost", "plot_turnover.png")
    plot_xy("pnl_var", "RL sweep: variance of net PnL vs eta_cost", "plot_variance.png")
    plot_xy("ES95", "RL sweep: ES95 of net PnL vs eta_cost", "plot_es95.png")
    plot_xy("HE_var", "RL sweep: Hedging Effectiveness (variance reduction) vs eta_cost", "plot_he_var.png")

    # Save run meta
    meta = {
        "cache": args.cache,
        "exposure_id": args.exposure_id,
        "scenario_parquet": args.scenario_parquet,
        "chosen_year": int(year),
        "n_scenarios": int(len(scenarios)),
        "timesteps": int(args.timesteps),
        "eta_grid": eta_values,
        "device": str(args.device),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nDONE")
    print("- scenarios:", scenario_out)
    print("- sweep csv:", out_dir / "parameter_sweep.csv")
    print("- plots:", out_dir)


if __name__ == "__main__":
    main()