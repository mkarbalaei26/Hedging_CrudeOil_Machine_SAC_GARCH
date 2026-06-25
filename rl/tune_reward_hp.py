# rl/tune_reward_hp.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import time
import random
from datetime import timedelta
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# --- import from your project ---
# Support running as a module (`python -m rl.tune_reward_hp`) OR as a script (`PYTHONPATH=. python rl/tune_reward_hp.py`).
# Note: project modules live under the `rl` package.
try:
    # When running as a module: `python -m rl.tune_reward_hp`
    from .env_daily import OilHedgingDailyEnv as DailyHedgeEnv, EnvConfig
    from .train_walkforward import (
        walkforward_windows,
        _load_pre_from_cache,
        _load_window_scenarios,
    )
except ImportError:  # pragma: no cover
    # When running as a script from project root with PYTHONPATH=.
    from rl.env_daily import OilHedgingDailyEnv as DailyHedgeEnv, EnvConfig
    from rl.train_walkforward import (
        walkforward_windows,
        _load_pre_from_cache,
        _load_window_scenarios,
    )


def ensure_seed(seed: int) -> None:
    """Set seeds for Python/NumPy/Torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # make CuDNN deterministic if it exists
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        # torch not installed or not needed
        pass


# =========================
# Progress reporting helpers
# =========================

def _fmt_td(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except Exception:
        return "?"
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    return str(timedelta(seconds=int(seconds)))


def _print_progress(
    *,
    done: int,
    total: int,
    start_ts: float,
    ema_sec_per_trial: Optional[float],
    best_score: Optional[float],
    last_score: Optional[float],
    n_workers: int,
) -> float:
    """Print a single-line progress update and return updated EMA seconds/trial."""
    now = time.time()
    elapsed = now - start_ts

    # instantaneous avg seconds per completed trial (wall time)
    inst = (elapsed / done) if done > 0 else None

    # EMA over inst rate to stabilize ETA
    if inst is None:
        ema = ema_sec_per_trial
    elif ema_sec_per_trial is None:
        ema = inst
    else:
        alpha = 0.15
        ema = (1 - alpha) * ema_sec_per_trial + alpha * inst

    remaining = total - done
    eta = (ema * remaining) if (ema is not None) else float("nan")

    pct = (100.0 * done / total) if total > 0 else 0.0
    best_s = f"{best_score:.6f}" if best_score is not None and np.isfinite(best_score) else "?"
    last_s = f"{last_score:.6f}" if last_score is not None and np.isfinite(last_score) else "?"

    msg = (
        f"[tune] {done}/{total} ({pct:6.2f}%) | "
        f"workers={n_workers} | elapsed={_fmt_td(elapsed)} | "
        f"ETA={_fmt_td(eta)} | best_val={best_s} | last_val={last_s}"
    )

    # carriage-return overwrite; print newline when finished
    end = "\n" if done >= total else "\r"
    print(msg, end=end, file=sys.stdout, flush=True)
    return ema

# =========================
# Metrics helpers (no leakage)
# =========================

def var_es(losses: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    """
    losses: array of PnL (can be negative). We compute VaR/ES on the left tail.
    VaR95 = 5th percentile (since alpha=0.95).
    ES95  = mean of values <= VaR95.
    """
    if losses.size == 0:
        return float("nan"), float("nan")
    q = np.quantile(losses, 1 - alpha)
    tail = losses[losses <= q]
    es = float(np.mean(tail)) if tail.size else float(q)
    return float(q), float(es)

def max_drawdown(equity_curve: np.ndarray) -> float:
    if equity_curve.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak)
    return float(np.min(dd))  # negative number (drawdown in $)

def evaluate_policy_on_scenarios(
    model: PPO,
    scenarios: List[Dict[str, Any]],
    env_cfg: EnvConfig,
    seed: int,
    pre: Any,
    max_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs deterministic evaluation on a list of scenarios.
    Collects:
      - mean episode reward
      - mean episode PnL
      - MDD over equity (per episode, averaged)
      - VaR95/ES95 over all step PnL pooled (more informative than episode-only)
    """
    if max_episodes is not None:
        scenarios = scenarios[:max_episodes]

    env = DailyHedgeEnv(pre, scenarios=scenarios, cfg=env_cfg, seed=seed)
    vec = DummyVecEnv([lambda: env])

    ep_rewards: List[float] = []
    ep_pnls: List[float] = []
    ep_mdds: List[float] = []

    # pooled step PnL across all episodes (used for VaR/ES)
    step_pnls_all: List[float] = []

    for _ in range(len(scenarios)):
        obs = vec.reset()
        done = False
        total_r = 0.0

        # Build equity curve ourselves from step PnL (no dependency on env.equity)
        eq0 = float(getattr(env, "equity", 0.0))
        equity_curve: List[float] = [eq0]
        eq = eq0

        # Track step pnl for THIS episode (fixes previous cross-episode leakage)
        step_pnls_ep: List[float] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec.step(action)
            r = float(reward[0])
            total_r += r
            done = bool(dones[0])

            info0 = infos[0] if infos and isinstance(infos, (list, tuple)) else {}
            pnl_step = float(info0.get("pnl_net", info0.get("pnl", 0.0)))

            step_pnls_all.append(pnl_step)
            step_pnls_ep.append(pnl_step)

            # update synthetic equity
            eq += pnl_step
            equity_curve.append(eq)

        ep_rewards.append(total_r)

        # Prefer env totals if available, otherwise sum per-episode steps
        if hasattr(env, "total_pnl") and env.total_pnl is not None:
            ep_pnls.append(float(env.total_pnl))
        else:
            ep_pnls.append(float(np.sum(step_pnls_ep)))

        # MDD in dollars from synthetic equity curve
        ep_mdds.append(max_drawdown(np.asarray(equity_curve, dtype=float)))

    step_pnls_arr = np.asarray(step_pnls_all, dtype=float)
    VaR95, ES95 = var_es(step_pnls_arr, alpha=0.95)

    # robust aggregates (avoid numpy empty-slice warnings)
    mdd_mean = float(np.mean(ep_mdds)) if len(ep_mdds) > 0 else float("nan")
    mdd_std = float(np.std(ep_mdds)) if len(ep_mdds) > 0 else float("nan")

    return {
        "episodes": len(scenarios),
        "reward_mean": float(np.mean(ep_rewards)) if len(ep_rewards) > 0 else float("nan"),
        "reward_std": float(np.std(ep_rewards)) if len(ep_rewards) > 0 else float("nan"),
        "pnl_mean": float(np.mean(ep_pnls)) if len(ep_pnls) > 0 else float("nan"),
        "pnl_std": float(np.std(ep_pnls)) if len(ep_pnls) > 0 else float("nan"),
        "mdd_mean": mdd_mean,
        "mdd_std": mdd_std,
        "VaR95_step": VaR95,
        "ES95_step": ES95,
    }

def composite_score(metrics: Dict[str, Any]) -> float:
    """
    One scalar to rank configs on VAL only.
    You can tweak weights later.
    Convention:
      - higher pnl is better
      - less negative mdd is better (closer to 0)
      - VaR/ES: higher is better if they are less negative (loss tail smaller)
    """
    def _finite_or(x: Any, fallback: float) -> float:
        try:
            v = float(x)
        except Exception:
            return float(fallback)
        return v if np.isfinite(v) else float(fallback)

    pnl = _finite_or(metrics.get("pnl_mean"), -1e9)
    mdd = _finite_or(metrics.get("mdd_mean"), -1e9)      # negative; closer to 0 is better
    var95 = _finite_or(metrics.get("VaR95_step"), -1e9)  # typically negative; closer to 0 is better
    es95 = _finite_or(metrics.get("ES95_step"), -1e9)    # typically negative; closer to 0 is better

    # Weights: adjust later if you want risk-first vs pnl-first
    w_pnl = 1.0
    w_mdd = 0.5
    w_var = 0.3
    w_es = 0.3

    return float(w_pnl * pnl + w_mdd * mdd + w_var * var95 + w_es * es95)

# =========================
# Sampling search space
# =========================

def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))

def sample_hparams(trial_id: int, base_seed: int) -> Dict[str, Any]:
    """
    Fixed in env (per your decision):
      - risk_mode = lpm
      - roll_var_L = 10
      - include_elapsed = True
      - include_equity  = True

    Fixed here in tuning (your request):
      - PPO: gamma=0.97, gae_lambda=0.97, ent_coef=0.01, clip_range=0.16,
             n_steps=256, batch_size=128, n_epochs=5
      - Env: h_max=0.78, lpm_target=0.002

    Tuned:
      - mu_pnl in [0.08, 0.25]
      - lambda_lpm in [3, 15]
      - lpm_order in {1,2}
      - eta_cost in [30, 100] (log-uniform)
      - cost_penalty_mult in [10, 30]
      - min_action_change_threshold in [0.05, 0.18]
      - delta_h_bounds magnitude b in [0.6, 1.4]
      - learning_rate in [8e-6, 1.2e-4] (log-uniform)
    """
    rng = np.random.default_rng(base_seed + trial_id * 10007)

    hp = {}

    # =============================
    # Reward / risk hyperparameters
    # =============================
    # You asked to FIX the previously "common" PPO/env values (category A)
    # and only tune the sensitive ones (category B).

    # --- Env reward/risk weights ---
    hp["mu_pnl"] = float(rng.uniform(0.08, 0.25))          # unitless reward weight on PnL
    hp["lambda_lpm"] = float(rng.uniform(3.0, 15.0))      # downside penalty strength (unitless)

    # Try both orders again (explicitly requested)
    hp["lpm_order"] = int(rng.choice([1, 2]))

    # Fixed positive target (requested)
    hp["lpm_target"] = 0.002

    # Costs / churn
    # Search a reasonably high range (requested). Log-uniform covers 30..100 without biasing too hard.
    hp["eta_cost"] = log_uniform(rng, 30.0, 100.0)
    hp["cost_penalty_mult"] = float(rng.uniform(10.0, 30.0))

    # Deadband for tiny action changes: tuned, but keep it away from ~0 to avoid churny policies
    # (still bounded by action range; unitless)
    hp["min_action_change_threshold"] = float(rng.uniform(0.05, 0.18))

    # Hedge/action constraints
    # Keep h_max near the best cluster (fixed)
    hp["h_max"] = 0.78

    # Tune delta_h bounds but bias toward smaller jumps (risk/churn control)
    b = float(rng.uniform(0.6, 1.4))
    hp["delta_h_bounds"] = (-b, b)

    # =============================
    # PPO hyperparameters
    # =============================
    # Keep these FIXED as "common" settings (requested: no ranges here)
    hp["gamma"] = 0.97
    hp["gae_lambda"] = 0.97
    hp["ent_coef"] = 0.01
    hp["clip_range"] = 0.16
    hp["n_steps"] = 256
    hp["batch_size"] = 128
    hp["n_epochs"] = 5

    # Tune only learning rate in a tighter, safer band (log scale)
    hp["learning_rate"] = log_uniform(rng, 8e-6, 1.2e-4)

    return hp

# =========================
# One trial worker (parallel)
# =========================

def run_one_trial(payload: Dict[str, Any]) -> Dict[str, Any]:
    trial_id = payload["trial_id"]
    base_seed = payload["base_seed"]
    timesteps = payload["timesteps"]
    max_val_episodes = payload["max_val_episodes"]
    cache_path = payload["cache_path"]
    pre = _load_pre_from_cache(cache_path)

    # data/scenarios
    train_scenarios = payload["train_scenarios"]
    val_scenarios = payload["val_scenarios"]

    hp = sample_hparams(trial_id, base_seed)
    seed = base_seed + trial_id

    ensure_seed(seed)
    set_random_seed(seed)

    # build env config with your FIXED choices
    env_cfg = EnvConfig(
        # fixed per your decision
        risk_mode="lpm",
        roll_var_L=10,
        include_elapsed=True,
        include_equity=True,

        # keep other defaults unless tuned
        mu_pnl=hp["mu_pnl"],
        lambda_lpm=hp["lambda_lpm"],
        lpm_order=hp["lpm_order"],
        lpm_target=hp["lpm_target"],
        eta_cost=hp["eta_cost"],
        cost_penalty_mult=hp["cost_penalty_mult"],
        min_action_change_threshold=hp["min_action_change_threshold"],
        h_max=hp["h_max"],
        delta_h_bounds=hp["delta_h_bounds"],

        # info_mode train for training env
        info_mode="train",
    )

    # --- training env ---
    train_env = DailyHedgeEnv(pre, scenarios=train_scenarios, cfg=env_cfg, seed=seed)
    vec_train = DummyVecEnv([lambda: train_env])

    model = PPO(
        "MlpPolicy",
        vec_train,
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        ent_coef=hp["ent_coef"],
        clip_range=hp["clip_range"],
        verbose=0,
        seed=seed,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - t0

    # --- validation env config (eval mode) ---
    env_cfg_eval = EnvConfig(**{**asdict(env_cfg), "info_mode": "eval"})

    val_metrics = evaluate_policy_on_scenarios(
        model=model,
        scenarios=val_scenarios,
        env_cfg=env_cfg_eval,
        seed=seed,
        pre=pre,
        max_episodes=max_val_episodes,
    )
    score = composite_score(val_metrics)

    out = {
        "trial_id": trial_id,
        "seed": seed,
        "train_time_sec": train_time,
        "timesteps": timesteps,
        "score_val": score,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"hp_{k}": v for k, v in hp.items()},
    }
    return out

# =========================
# Plotting & report
# =========================

def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Create thesis-ready figures.

    Notes on units:
      - PnL/MDD/VaR/ES are in dollars ($) because the environment reports step/episode PnL in $.
        We format axes in $ millions to keep figures readable.
      - Composite score is unitless (a linear combination of $-valued metrics).
      - Hyperparameters such as mu_pnl / lambda_lpm / eta_cost are *unitless weights* applied inside the reward.
      - Learning rate is unitless.

    The function is defensive against NaNs and small sample sizes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: nicer default style via seaborn if installed
    try:
        import seaborn as sns  # type: ignore
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
        _use_sns = True
    except Exception:
        _use_sns = False
        # Matplotlib defaults for paper-like plots
        plt.rcParams.update(
            {
                "figure.dpi": 160,
                "savefig.dpi": 220,
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

    import matplotlib.ticker as mticker
    from matplotlib.ticker import FuncFormatter

    def _millions_formatter(prefix: str = "$", suffix: str = "M"):
        def _f(x, _pos):
            if x is None or not np.isfinite(x):
                return ""
            return f"{prefix}{x/1e6:.2f}{suffix}"
        return FuncFormatter(_f)

    def _apply_grid(ax):
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.15)

    def _clean(sub: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        # Keep only rows where all required cols are finite
        out = sub.copy()
        for c in cols:
            out = out[np.isfinite(out[c].astype(float))]
        return out

    # -----------------------------
    # 1) Score distribution
    # -----------------------------
    d1 = _clean(df, ["score_val"]) if "score_val" in df.columns else pd.DataFrame()
    if len(d1) > 0:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        if _use_sns:
            import seaborn as sns  # type: ignore
            sns.histplot(d1["score_val"], bins=30, kde=True, ax=ax)
        else:
            ax.hist(d1["score_val"].to_numpy(), bins=30)
        ax.set_title("Validation window: composite score distribution")
        ax.set_xlabel("Composite score (unitless)")
        ax.set_ylabel("Count")
        _apply_grid(ax)
        fig.savefig(out_dir / "score_distribution.png")
        plt.close(fig)

    # -----------------------------
    # 2) PnL vs MDD (VAL)
    # -----------------------------
    cols2 = ["val_pnl_mean", "val_mdd_mean"]
    d2 = _clean(df, cols2) if all(c in df.columns for c in cols2) else pd.DataFrame()
    if len(d2) > 0:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        if _use_sns:
            import seaborn as sns  # type: ignore
            sns.scatterplot(
                data=d2,
                x="val_pnl_mean",
                y="val_mdd_mean",
                s=55,
                edgecolor="none",
                ax=ax,
            )
        else:
            ax.scatter(d2["val_pnl_mean"], d2["val_mdd_mean"], s=35)

        ax.set_title("Validation window: mean episode PnL vs mean episode maximum drawdown")
        ax.set_xlabel("Mean episode PnL ($)")
        ax.set_ylabel("Mean episode max drawdown ($; more negative = worse)")

        # Format in $ millions to avoid 1e6 scientific offset clutter
        ax.xaxis.set_major_formatter(_millions_formatter("$", "M"))
        ax.yaxis.set_major_formatter(_millions_formatter("$", "M"))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        _apply_grid(ax)

        # Reference lines
        ax.axhline(0.0, linewidth=1.0, alpha=0.4)
        ax.axvline(0.0, linewidth=1.0, alpha=0.4)

        fig.savefig(out_dir / "pnl_vs_mdd.png")
        plt.close(fig)

    # -----------------------------
    # 3) Learning rate vs score
    # -----------------------------
    cols3 = ["hp_learning_rate", "score_val"]
    d3 = _clean(df, cols3) if all(c in df.columns for c in cols3) else pd.DataFrame()
    if len(d3) > 0:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        if _use_sns:
            import seaborn as sns  # type: ignore
            sns.scatterplot(
                data=d3,
                x="hp_learning_rate",
                y="score_val",
                s=55,
                edgecolor="none",
                ax=ax,
            )
        else:
            ax.scatter(d3["hp_learning_rate"], d3["score_val"], s=35)

        ax.set_xscale("log")
        ax.set_title("Validation window: PPO learning rate vs composite score")
        ax.set_xlabel("Learning rate (log scale)")
        ax.set_ylabel("Composite score (unitless)")
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        _apply_grid(ax)

        fig.savefig(out_dir / "lr_vs_score.png")
        plt.close(fig)

    # -----------------------------
    # 4) 3D plots (mu_pnl, lambda_lpm, log10(lr))
    # -----------------------------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Pretty labels for colorbars
    cbar_label = {
        "val_mdd_mean": "Mean episode max drawdown ($)",
        "val_VaR95_step": "Step VaR (95%) ($)",
        "val_ES95_step": "Step ES (95%) ($)",
        "score_val": "Composite score (unitless)",
    }

    def _colorbar_formatter_for(col: str):
        # Use $ millions for $-valued metrics
        if col in {"val_mdd_mean", "val_VaR95_step", "val_ES95_step"}:
            return _millions_formatter("$", "M")
        return None

    def plot3d(color_col: str, fname: str, title: str):
        cols = ["hp_mu_pnl", "hp_lambda_lpm", "hp_learning_rate", color_col]
        d = _clean(df, cols) if all(c in df.columns for c in cols) else pd.DataFrame()
        if len(d) == 0:
            return

        fig = plt.figure(figsize=(8.2, 6.2), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        x = d["hp_mu_pnl"].to_numpy(dtype=float)
        y = d["hp_lambda_lpm"].to_numpy(dtype=float)
        lr = d["hp_learning_rate"].to_numpy(dtype=float)
        # Show lr on a log scale without turning the 3D axis into unreadable tiny numbers
        z = np.log10(np.clip(lr, 1e-12, None))

        c = d[color_col].to_numpy(dtype=float)

        sc = ax.scatter(x, y, z, c=c, s=45, depthshade=True)

        ax.set_title(title, pad=18)
        ax.set_xlabel(r"$\mu_{\mathrm{PnL}}$ (unitless reward weight)", labelpad=12)
        ax.set_ylabel(r"$\lambda_{\mathrm{LPM}}$ (risk penalty weight; unitless)", labelpad=12)
        ax.set_zlabel(r"$\log_{10}$(learning rate)", labelpad=12)

        # Better viewing angle for papers
        ax.view_init(elev=22, azim=-58)

        # Grid
        ax.grid(True)

        # Colorbar with sensible units
        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08)
        cbar.set_label(cbar_label.get(color_col, color_col), rotation=90, labelpad=14)
        fmt = _colorbar_formatter_for(color_col)
        if fmt is not None:
            cbar.ax.yaxis.set_major_formatter(fmt)

        fig.savefig(out_dir / fname)
        plt.close(fig)

    plot3d("val_mdd_mean", "3d_mu_lpm_lr_mdd.png", "3D: Reward/risk hyperparameters colored by validation MDD")
    plot3d("val_VaR95_step", "3d_mu_lpm_lr_var95.png", "3D: Reward/risk hyperparameters colored by validation VaR (95%)")
    plot3d("val_ES95_step", "3d_mu_lpm_lr_es95.png", "3D: Reward/risk hyperparameters colored by validation ES (95%)")
    plot3d("score_val", "3d_mu_lpm_lr_score.png", "3D: Reward/risk hyperparameters colored by validation composite score")

def write_report(
    df_all: pd.DataFrame,
    df_top_stable: pd.DataFrame,
    final_test_rows: Optional[pd.DataFrame],
    window_info: Dict[str, Any],
    out_dir: Path
) -> None:
    lines = []
    lines.append("# Hyperparameter tuning report (first walk-forward window)\n")
    lines.append("## Window definition\n")
    lines.append(f"- Train years: {window_info['train_years'][0]}-{window_info['train_years'][-1]}\n")
    lines.append(f"- Val year: {window_info['val_years'][0]}\n")
    lines.append(f"- Test year: {window_info['test_years'][0]}\n")
    lines.append("\n## Fixed choices (per your decision)\n")
    lines.append("- risk_mode = lpm\n")
    lines.append("- roll_var_L = 10\n")
    lines.append("- include_elapsed = True\n")
    lines.append("- include_equity  = True\n")

    lines.append("\n## Search summary\n")
    lines.append(f"- Trials: {len(df_all)}\n")
    lines.append(f"- Best VAL score: {df_all['score_val'].max():.6f}\n")
    best = df_all.sort_values("score_val", ascending=False).iloc[0].to_dict()
    lines.append("\n## Best single trial on VAL (not yet stability-checked)\n")
    lines.append(f"- trial_id: {int(best['trial_id'])}\n")
    lines.append(f"- score_val: {best['score_val']:.6f}\n")
    lines.append(f"- val_pnl_mean: {best['val_pnl_mean']:.6f}\n")
    lines.append(f"- val_mdd_mean: {best['val_mdd_mean']:.6f}\n")
    lines.append(f"- val_VaR95_step: {best['val_VaR95_step']:.6f}\n")
    lines.append(f"- val_ES95_step: {best['val_ES95_step']:.6f}\n")

    lines.append("\n## Stable top configs (multi-seed re-check)\n")
    if len(df_top_stable) == 0:
        lines.append("- (none)\n")
    else:
        # show top 10 rows with key stats
        cols = [
            "rank", "score_val_mean", "score_val_std",
            "val_pnl_mean_mean", "val_pnl_mean_std",
            "val_mdd_mean_mean", "val_mdd_mean_std",
            "val_VaR95_step_mean", "val_ES95_step_mean",
        ]
        lines.append(df_top_stable[cols].head(10).to_markdown(index=False))
        lines.append("\n")

    if final_test_rows is not None and len(final_test_rows) > 0:
        lines.append("\n## Held-out TEST (2011) evaluation for final selected configs\n")
        lines.append(final_test_rows.to_markdown(index=False))
        lines.append("\n")

    lines.append("\n## Generated plots\n")
    lines.append("- score_distribution.png\n")
    lines.append("- pnl_vs_mdd.png\n")
    lines.append("- lr_vs_score.png\n")
    lines.append("- 3d_mu_lpm_lr_score.png\n")
    lines.append("- 3d_mu_lpm_lr_mdd.png\n")
    lines.append("- 3d_mu_lpm_lr_var95.png\n")
    lines.append("- 3d_mu_lpm_lr_es95.png\n")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

# =========================
# Multiprocessing-safe workers (must be top-level for pickling on macOS spawn)
# =========================

def _stability_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one stability re-check job (top-level so it can be pickled)."""
    config_id = int(payload["config_id"])
    seed = int(payload["seed"])
    hp = payload["hp"]
    cache_path = payload["cache_path"]
    timesteps = int(payload["timesteps"])

    ensure_seed(seed)
    set_random_seed(seed)
    pre = _load_pre_from_cache(cache_path)

    env_cfg = EnvConfig(
        risk_mode="lpm",
        roll_var_L=10,
        include_elapsed=True,
        include_equity=True,
        info_mode="train",

        mu_pnl=float(hp["mu_pnl"]),
        lambda_lpm=float(hp["lambda_lpm"]),
        lpm_order=int(hp["lpm_order"]),
        lpm_target=float(hp["lpm_target"]),
        eta_cost=float(hp["eta_cost"]),
        cost_penalty_mult=float(hp["cost_penalty_mult"]),
        min_action_change_threshold=float(hp["min_action_change_threshold"]),
        h_max=float(hp["h_max"]),
        delta_h_bounds=tuple(hp["delta_h_bounds"]),
    )

    train_env = DailyHedgeEnv(pre, scenarios=payload["train_scenarios"], cfg=env_cfg, seed=seed)
    vec_train = DummyVecEnv([lambda: train_env])

    model = PPO(
        "MlpPolicy",
        vec_train,
        learning_rate=float(hp["learning_rate"]),
        n_steps=int(hp["n_steps"]),
        batch_size=int(hp["batch_size"]),
        n_epochs=int(hp["n_epochs"]),
        gamma=float(hp["gamma"]),
        gae_lambda=float(hp["gae_lambda"]),
        ent_coef=float(hp["ent_coef"]),
        clip_range=float(hp["clip_range"]),
        verbose=0,
        seed=seed,
    )

    model.learn(total_timesteps=timesteps)

    env_cfg_eval = EnvConfig(**{**asdict(env_cfg), "info_mode": "eval"})
    val_metrics = evaluate_policy_on_scenarios(
        model=model,
        scenarios=payload["val_scenarios"],
        env_cfg=env_cfg_eval,
        seed=seed,
        pre=pre,
        max_episodes=payload.get("max_val_episodes", None),
    )
    score = composite_score(val_metrics)

    return {
        "config_id": config_id,
        "seed": seed,
        "score_val": float(score),
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }


def _final_test_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Train one chosen config and evaluate on TEST (top-level so it can be pickled)."""
    config_id = int(payload["config_id"])
    seed = int(payload["seed"])
    hp = payload["hp"]
    cache_path = payload["cache_path"]
    timesteps = int(payload["timesteps"])

    ensure_seed(seed)
    set_random_seed(seed)
    pre = _load_pre_from_cache(cache_path)

    env_cfg = EnvConfig(
        risk_mode="lpm",
        roll_var_L=10,
        include_elapsed=True,
        include_equity=True,
        info_mode="train",

        mu_pnl=float(hp["mu_pnl"]),
        lambda_lpm=float(hp["lambda_lpm"]),
        lpm_order=int(hp["lpm_order"]),
        lpm_target=float(hp["lpm_target"]),
        eta_cost=float(hp["eta_cost"]),
        cost_penalty_mult=float(hp["cost_penalty_mult"]),
        min_action_change_threshold=float(hp["min_action_change_threshold"]),
        h_max=float(hp["h_max"]),
        delta_h_bounds=tuple(hp["delta_h_bounds"]),
    )

    train_env = DailyHedgeEnv(pre, scenarios=payload["train_scenarios"], cfg=env_cfg, seed=seed)
    vec_train = DummyVecEnv([lambda: train_env])

    model = PPO(
        "MlpPolicy",
        vec_train,
        learning_rate=float(hp["learning_rate"]),
        n_steps=int(hp["n_steps"]),
        batch_size=int(hp["batch_size"]),
        n_epochs=int(hp["n_epochs"]),
        gamma=float(hp["gamma"]),
        gae_lambda=float(hp["gae_lambda"]),
        ent_coef=float(hp["ent_coef"]),
        clip_range=float(hp["clip_range"]),
        verbose=0,
        seed=seed,
    )

    model.learn(total_timesteps=timesteps)

    env_cfg_eval = EnvConfig(**{**asdict(env_cfg), "info_mode": "eval"})
    test_metrics = evaluate_policy_on_scenarios(
        model=model,
        scenarios=payload["test_scenarios"],
        env_cfg=env_cfg_eval,
        seed=seed,
        pre=pre,
        max_episodes=payload.get("max_test_episodes", None),
    )

    return {
        "config_id": config_id,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

# =========================
# Stability re-check
# =========================

def stability_check(
    best_rows: pd.DataFrame,
    payload_base: Dict[str, Any],
    out_dir: Path,
    n_seeds: int = 5,
    max_workers: int = 10
) -> pd.DataFrame:
    """
    Re-run top configs with multiple seeds to measure stability.
    We re-use their hp_* columns (no resampling).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    base_seed = payload_base["base_seed"]

    # capture hp keys
    hp_cols = [c for c in best_rows.columns if c.startswith("hp_")]

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = []
        for _, row in best_rows.iterrows():
            trial_id = int(row["trial_id"])

            # Build a clean hp dict (strip hp_ prefix)
            hp_dict = {k.replace("hp_", ""): row[k] for k in hp_cols}

            # delta_h_bounds may be stored as JSON string
            dhb = hp_dict.get("delta_h_bounds")
            if isinstance(dhb, str):
                try:
                    hp_dict["delta_h_bounds"] = tuple(json.loads(dhb))
                except Exception:
                    hp_dict["delta_h_bounds"] = tuple(json.loads(dhb.replace("(", "[").replace(")", "]")))

            for s_i in range(int(n_seeds)):
                seed = base_seed + 100000 + trial_id * 101 + s_i
                payload = {
                    "config_id": trial_id,
                    "seed": seed,
                    "hp": hp_dict,
                    "cache_path": payload_base["cache_path"],
                    "timesteps": payload_base["timesteps"],
                    "train_scenarios": payload_base["train_scenarios"],
                    "val_scenarios": payload_base["val_scenarios"],
                    "max_val_episodes": payload_base.get("max_val_episodes", None),
                }
                futs.append(ex.submit(_stability_worker, payload))

        rows = []
        for f in as_completed(futs):
            rows.append(f.result())

    df = pd.DataFrame(rows)

    # aggregate stability
    agg = df.groupby("config_id").agg(
        score_val_mean=("score_val", "mean"),
        score_val_std=("score_val", "std"),
        val_pnl_mean_mean=("val_pnl_mean", "mean"),
        val_pnl_mean_std=("val_pnl_mean", "std"),
        val_mdd_mean_mean=("val_mdd_mean", "mean"),
        val_mdd_mean_std=("val_mdd_mean", "std"),
        val_VaR95_step_mean=("val_VaR95_step", "mean"),
        val_ES95_step_mean=("val_ES95_step", "mean"),
    ).reset_index()

    # rank by mean - penalty*std (stability-aware)
    agg["stable_score"] = agg["score_val_mean"] - 0.5 * agg["score_val_std"].fillna(0.0)
    agg = agg.sort_values("stable_score", ascending=False).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    agg.to_csv(out_dir / "stability_summary.csv", index=False)
    df.to_csv(out_dir / "stability_runs.csv", index=False)
    return agg

# =========================
# Final test evaluation
# =========================

def final_test_eval(
    chosen_configs: pd.DataFrame,
    df_all: pd.DataFrame,
    payload_base: Dict[str, Any],
    out_dir: Path,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Train each chosen config on TRAIN, pick best checkpoint implicitly (single run),
    evaluate on TEST scenarios. (We DO NOT use test for selection.)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_seed = payload_base["base_seed"]
    hp_cols = [c for c in df_all.columns if c.startswith("hp_")]

    ctx = mp.get_context("spawn")
    rows = []

    # Build a lookup dict from df_all for hp columns
    hp_cols = [c for c in df_all.columns if c.startswith("hp_")]
    df_idx = df_all.set_index("trial_id")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = []
        for cid in chosen_configs["config_id"].tolist():
            cid = int(cid)
            row = df_idx.loc[cid]
            hp_dict = {k.replace("hp_", ""): row[k] for k in hp_cols}

            dhb = hp_dict.get("delta_h_bounds")
            if isinstance(dhb, str):
                try:
                    hp_dict["delta_h_bounds"] = tuple(json.loads(dhb))
                except Exception:
                    hp_dict["delta_h_bounds"] = tuple(json.loads(dhb.replace("(", "[").replace(")", "]")))

            seed = int(payload_base["base_seed"]) + 200000 + cid
            payload = {
                "config_id": cid,
                "seed": seed,
                "hp": hp_dict,
                "cache_path": payload_base["cache_path"],
                "timesteps": payload_base["timesteps"],
                "train_scenarios": payload_base["train_scenarios"],
                "test_scenarios": payload_base["test_scenarios"],
                "max_test_episodes": payload_base.get("max_test_episodes", None),
            }
            futs.append(ex.submit(_final_test_worker, payload))

        for f in as_completed(futs):
            rows.append(f.result())

    df = pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)
    df.to_csv(out_dir / "final_test_eval.csv", index=False)
    return df

# =========================
# Main
# =========================

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True, help="precomputed NPZ cache (same as train_walkforward --cache)")
    ap.add_argument("--scenario_dir", type=str, required=True, help="scenario directory (same as train_walkforward --scenario_dir)")
    ap.add_argument("--exposure_id", type=str, required=True, help="exposure id (same as train_walkforward --exposure_id)")
    ap.add_argument("--out_dir", type=str, default="tune_out", help="output directory")
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--n_workers", type=int, default=10)
    ap.add_argument("--timesteps", type=int, default=50000, help="training timesteps per trial (keep light for big n_trials)")
    ap.add_argument("--base_seed", type=int, default=2026)

    # first window (expected 2008-2009 train, 2010 val, 2011 test)
    ap.add_argument("--year_start", type=int, default=2008)
    ap.add_argument("--year_end", type=int, default=2011)

    # evaluation caps to keep runtime bounded
    ap.add_argument("--max_train_scenarios", type=int, default=300)
    ap.add_argument("--max_val_episodes", type=int, default=120)
    ap.add_argument("--max_test_episodes", type=int, default=120)

    # stability
    ap.add_argument("--top_k", type=int, default=20, help="top K configs to re-check for stability")
    ap.add_argument("--stable_seeds", type=int, default=5, help="seeds per config for stability check")
    ap.add_argument("--final_k", type=int, default=5, help="how many stable configs to evaluate on TEST")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_seed(args.base_seed)
    set_random_seed(args.base_seed)

    # --- load precomputed cache once ---
    pre = _load_pre_from_cache(args.cache)

    # dates_int is needed by the scenario loader helpers.
    # train_walkforward may return either a dict-like cache OR a dataclass (e.g. PrecomputeResult).
    if isinstance(pre, dict):
        dates_int = pre.get("dates_int", None)
    else:
        dates_int = getattr(pre, "dates_int", None)

    if dates_int is None:
        raise RuntimeError(
            "Cache missing 'dates_int'. Rebuild cache using train_walkforward precompute."
        )

    # --- helper: convert (year range) -> (date_int range) when scenario loader expects date_ints ---
    # In this project, `dates_int` are days since 1970-01-01 (confirmed by your printed range).
    import datetime as _dt

    _ORIGIN = _dt.date(1970, 1, 1)

    def _year_range_to_window_ints(y0: int, y1: int) -> Tuple[int, int]:
        """Return (start_int, end_int) aligned to available trading dates.

        We target [Jan 1 of y0, Dec 31 of y1] and then snap to the nearest
        available entries in `dates_int`:
          - start_int = first dates_int >= target_start
          - end_int   = last  dates_int <= target_end

        Raises if the requested years have no overlap with available dates.
        """
        if dates_int is None or len(dates_int) == 0:
            raise RuntimeError("dates_int is empty; cannot build scenario windows")

        # target bounds (calendar)
        target_start = ( _dt.date(int(y0), 1, 1) - _ORIGIN ).days
        target_end   = ( _dt.date(int(y1), 12, 31) - _ORIGIN ).days

        di = np.asarray(dates_int, dtype=int)
        di_sorted = di  # precompute cache should already be sorted

        # snap start: first >= target_start
        i0 = int(np.searchsorted(di_sorted, target_start, side="left"))
        if i0 >= len(di_sorted):
            raise RuntimeError(f"Requested start year {y0} is after available data")
        start_int = int(di_sorted[i0])

        # snap end: last <= target_end
        i1 = int(np.searchsorted(di_sorted, target_end, side="right")) - 1
        if i1 < 0:
            raise RuntimeError(f"Requested end year {y1} is before available data")
        end_int = int(di_sorted[i1])

        if start_int > end_int:
            raise RuntimeError(
                f"No overlap between requested years {y0}-{y1} and available dates_int range {int(di_sorted[0])}-{int(di_sorted[-1])}"
            )

        return start_int, end_int

    # --- build first walk-forward window ---
    # --- build first walk-forward window ---
# train_walkforward.walkforward_windows signature is: (year_start, year_end) -> List[(train_y0, train_y1, val_y, test_y)]
    windows = walkforward_windows(args.year_start, args.year_end)
    if not windows:
        raise RuntimeError("No windows generated. Check year_start/year_end vs your data coverage.")

    train_y0, train_y1, val_y, test_y = windows[0]
    train_years = [int(train_y0), int(train_y1)]
    val_years = [int(val_y)]
    test_years = [int(test_y)]

    # --- load scenarios for that window (match train_walkforward logic) ---
    # train_walkforward expects `scenario_path` to point to the *exposure folder*.
    # Otherwise loaders may scan other exposures and try to read non-parquet files (e.g. baseline.npz).
    scenario_root = Path(args.scenario_dir)
    exposure_dir = scenario_root / args.exposure_id

    # IMPORTANT: pass a SINGLE parquet file, not a directory
    preferred_file = exposure_dir / "oracle_universe.parquet"
    if not preferred_file.exists():
        raise FileNotFoundError(f"Missing scenario file: {preferred_file}")

    scenario_path = str(preferred_file)

    # ---- compatibility wrapper: train_walkforward versions differ in _load_window_scenarios signature ----
    import inspect

    def load_window_scenarios_compat(*, y0: int, y1: int, max_scenarios: int, seed: int):
        """Call _load_window_scenarios with the right argument names across repo versions.

        IMPORTANT: Some versions define `_load_window_scenarios` as keyword-only (leading `*` in signature).
        So we should prefer **kwargs** calls and avoid positional fallbacks.
        """
        sig = inspect.signature(_load_window_scenarios)
        params = set(sig.parameters.keys())

        # Common required args
        base = {
            "scenario_path": scenario_path,
            "dates_int": dates_int,
            "exposure_id": args.exposure_id,
            "max_scenarios": max_scenarios,
            "seed": seed,
            "require_ok_coverage": True,
            "allow_shortened": False,
        }

        # Newer/alternate naming: window_start_int/window_end_int
        # In some repo versions these are *date_int* bounds (days since 1970), not years.
        if "window_start_int" in params and "window_end_int" in params:
            ws, we = _year_range_to_window_ints(int(y0), int(y1))
            return _load_window_scenarios(
                **base,
                window_start_int=int(ws),
                window_end_int=int(we),
            )

        # Older naming: y0/y1
        if "y0" in params and "y1" in params:
            return _load_window_scenarios(**base, y0=int(y0), y1=int(y1))

        # Older naming: year_start/year_end
        if "year_start" in params and "year_end" in params:
            return _load_window_scenarios(**base, year_start=int(y0), year_end=int(y1))

        # Older naming: window_start/window_end
        # Some versions also expect these as date_int bounds.
        if "window_start" in params and "window_end" in params:
            ws, we = _year_range_to_window_ints(int(y0), int(y1))
            return _load_window_scenarios(**base, window_start=int(ws), window_end=int(we))

        raise TypeError(
            f"Unsupported _load_window_scenarios signature: {sig}. "
            "Expected one of (y0/y1), (year_start/year_end), (window_start/window_end), or (window_start_int/window_end_int)."
        )

    train_scenarios = load_window_scenarios_compat(
        y0=int(train_y0),
        y1=int(train_y1),
        max_scenarios=args.max_train_scenarios,
        seed=args.base_seed + 11,
    )
    val_scenarios = load_window_scenarios_compat(
        y0=int(val_y),
        y1=int(val_y),
        max_scenarios=args.max_val_episodes,
        seed=args.base_seed + 22,
    )
    test_scenarios = load_window_scenarios_compat(
        y0=int(test_y),
        y1=int(test_y),
        max_scenarios=args.max_test_episodes,
        seed=args.base_seed + 33,
    )

    # Defensive checks: if any list is empty, fail fast with a clear message.
    if not train_scenarios:
        raise RuntimeError(
            f"No TRAIN scenarios loaded for years {train_y0}-{train_y1}. "
            f"Check scenario_path={scenario_path} and loader window conversion."
        )
    if not val_scenarios:
        raise RuntimeError(
            f"No VAL scenarios loaded for year {val_y}. "
            f"Check scenario_path={scenario_path} and loader window conversion."
        )
    if not test_scenarios:
        raise RuntimeError(
            f"No TEST scenarios loaded for year {test_y}. "
            f"Check scenario_path={scenario_path} and loader window conversion."
        )

    payload_base = {
        "train_scenarios": train_scenarios,
        "val_scenarios": val_scenarios,
        "test_scenarios": test_scenarios,
        "base_seed": args.base_seed,
        "timesteps": args.timesteps,
        "max_val_episodes": args.max_val_episodes,
        "max_test_episodes": args.max_test_episodes,
        "cache_path": args.cache,
    }

    # --- parallel random search ---
    ctx = mp.get_context("spawn")
    results: List[Dict[str, Any]] = []

    total_trials = int(args.n_trials)
    done_trials = 0
    best_score: Optional[float] = None
    last_score: Optional[float] = None
    start_ts = time.time()
    ema_sec_per_trial: Optional[float] = None

    # initial banner
    print(
        f"[tune] starting: trials={total_trials}, workers={args.n_workers}, timesteps={args.timesteps} (per trial)",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        futs = []
        for trial_id in range(total_trials):
            payload = dict(payload_base)
            payload.update({"trial_id": trial_id})
            futs.append(ex.submit(run_one_trial, payload))

        # consume as they complete and print progress
        for f in as_completed(futs):
            res = f.result()
            results.append(res)

            done_trials += 1
            last_score = float(res.get("score_val", float("nan")))
            if best_score is None or (np.isfinite(last_score) and last_score > best_score):
                best_score = last_score

            ema_sec_per_trial = _print_progress(
                done=done_trials,
                total=total_trials,
                start_ts=start_ts,
                ema_sec_per_trial=ema_sec_per_trial,
                best_score=best_score,
                last_score=last_score,
                n_workers=int(args.n_workers),
            )

    df = pd.DataFrame(results)

    # make delta_h_bounds JSON-safe for csv
    if "hp_delta_h_bounds" in df.columns:
        df["hp_delta_h_bounds"] = df["hp_delta_h_bounds"].apply(lambda x: json.dumps(list(x)) if isinstance(x, (tuple, list)) else x)

    df = df.sort_values("score_val", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "results.csv", index=False)

    # plots
    save_plots(df, out_dir / "plots")

    # stability check on top-k
    top_k = df.head(args.top_k).copy()
    print(f"\n[tune] stability check: top_k={args.top_k}, seeds_per_config={args.stable_seeds}", flush=True)
    stable_summary = stability_check(
        best_rows=top_k,
        payload_base=payload_base,
        out_dir=out_dir / "stability",
        n_seeds=args.stable_seeds,
        max_workers=args.n_workers,
    )

    # final test eval for best stable configs
    final_test_rows = None
    if len(stable_summary) > 0:
        chosen = stable_summary.head(args.final_k)[["config_id"]].copy()
        print(f"[tune] final test eval: final_k={args.final_k}", flush=True)
        final_test_rows = final_test_eval(
            chosen_configs=chosen,
            df_all=df,
            payload_base=payload_base,
            out_dir=out_dir / "final_test",
            max_workers=args.n_workers,
        )

    # report
    write_report(
        df_all=df,
        df_top_stable=stable_summary,
        final_test_rows=final_test_rows,
        window_info={"train_years": train_years, "val_years": val_years, "test_years": test_years},
        out_dir=out_dir,
    )

    print(f"[DONE] outputs saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()