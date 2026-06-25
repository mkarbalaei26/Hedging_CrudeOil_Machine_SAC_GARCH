#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune RL dynamic-policy behavior for crude-oil hedging.

This tuner is intentionally narrower than the earlier reward tuner. It keeps the
major PPO/backtest structure fixed and searches only the parameters that are most
likely to determine whether the learned policy becomes genuinely dynamic:

- action_mode: h_levels or delta_h_discrete
- h_max / delta_h range
- entropy coefficient
- cost penalty eta_cost
- LPM penalty strength/order/target
- PnL reward weight

The reporting layer is more thesis-oriented:
- incremental results.csv after every trial
- top_configs.csv and best_config.json
- report.md with interpretation notes
- plots for score, PnL/MDD, dynamic behavior, and action diversity
- final training command template for the selected config
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Defaults
# -----------------------------

DEFAULT_BASE_OUT = Path("rl_runs/TUNE_DYNAMIC_OPEC")
DEFAULT_SEED = 42


# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def fmt_money(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"${x:,.0f}"


def load_existing_results(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def parse_action_counts(series: pd.Series) -> Counter:
    c: Counter = Counter()
    for x in series.dropna():
        try:
            d = json.loads(x)
            c.update({str(k): int(v) for k, v in d.items()})
        except Exception:
            continue
    return c


def action_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = [v / total for v in counter.values() if v > 0]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def normalized_entropy(counter: Counter) -> float:
    k = len(counter)
    if k <= 1:
        return 0.0
    return float(action_entropy(counter) / math.log(k))


# -----------------------------
# Search space
# -----------------------------


def sample_config(i: int, rng: random.Random, args: argparse.Namespace) -> Dict[str, Any]:
    """Sample a deliberately small but behaviorally important search space."""
    action_mode = str(args.action_mode_fixed)

    cfg: Dict[str, Any] = {
        "trial": i,
        "timesteps": int(args.trial_timesteps),
        "action_mode": action_mode,
        "h_max": float(args.h_max_fixed),
        "mu_pnl": rng.choice([0.08, 0.12, 0.15, 0.20]),
        "lambda_lpm": rng.choice([3.0, 5.0, 6.0, 8.0, 10.0, 12.0]),
        "lpm_order": int(args.lpm_order_fixed) if int(args.lpm_order_fixed) in (1, 2) else rng.choice([1, 2]),
        "lpm_target": rng.choice([0.0, 0.001, 0.002]),
        "eta_cost": rng.choice([2.0, 5.0, 8.0, 10.0, 15.0, 25.0]),
        "ent_coef": rng.choice([0.03, 0.05, 0.08, 0.10, 0.15]),
        "learning_rate": rng.choice([3e-5, 5e-5, 8e-5, 1.2e-4]),
        "clip_range": rng.choice([0.16, 0.20, 0.25]),
    }

    # Fixed during each sweep for clean comparison. Run separate sweeps for
    # different delta_h_max values, e.g. 0.15 vs 0.50.
    cfg["delta_h_max"] = float(args.delta_h_max_fixed)
    cfg["delta_h_step"] = float(args.delta_h_step_fixed)

    return cfg


# -----------------------------
# Command construction
# -----------------------------


def build_train_command(cfg: Dict[str, Any], args: argparse.Namespace, out_dir: Path) -> List[str]:
    cmd = [
        "python", "-m", "rl.train_walkforward",
        "--cache", args.cache,
        "--out_dir", str(out_dir),
        "--exposure_id", args.exposure_id,
        "--scenario_dir", args.scenario_dir,
        "--train_mode", args.train_mode,
        "--year_start", str(args.year_start),
        "--year_end", str(args.year_end),
        "--max_windows", str(args.max_windows),
        "--device", args.device,
        "--vec", args.vec,
        "--n_envs", str(args.n_envs),
        "--timesteps", str(cfg["timesteps"]),
        "--eval_episodes", str(args.eval_episodes),
        "--max_train_scenarios", str(args.max_train_scenarios),
        "--max_eval_scenarios", str(args.max_eval_scenarios),
        "--n_steps", str(args.n_steps),
        "--batch_size", str(args.batch_size),
        "--ppo_gamma", str(args.ppo_gamma),
        "--ppo_gae_lambda", str(args.ppo_gae_lambda),
        "--ppo_ent_coef", str(cfg["ent_coef"]),
        "--ppo_vf_coef", str(args.ppo_vf_coef),
        "--ppo_n_epochs", str(args.ppo_n_epochs),
        "--ppo_learning_rate", str(cfg["learning_rate"]),
        "--ppo_clip_range", str(cfg["clip_range"]),
        "--h_max", str(cfg["h_max"]),
        "--delta_h_max", str(cfg["delta_h_max"]),
        "--delta_h_step", str(cfg["delta_h_step"]),
        "--action_mode", cfg["action_mode"],
        "--risk_mode", "lpm",
        "--mu_pnl", str(cfg["mu_pnl"]),
        "--lambda_lpm", str(cfg["lambda_lpm"]),
        "--lpm_order", str(cfg["lpm_order"]),
        "--lpm_target", str(cfg["lpm_target"]),
        "--eta_cost", str(cfg["eta_cost"]),
        "--parallel_windows", "1",
    ]

    if args.torch_threads is not None:
        cmd += ["--torch_threads", str(args.torch_threads)]
    if args.torch_interop_threads is not None:
        cmd += ["--torch_interop_threads", str(args.torch_interop_threads)]

    return cmd


def shell_command(cmd: Iterable[str]) -> str:
    return " \\\n  ".join(cmd)


# -----------------------------
# Metrics and scoring
# -----------------------------


def summarize_trial(df: pd.DataFrame, cfg: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    c = parse_action_counts(df.get("action_counts", pd.Series(dtype=object)))
    entropy = action_entropy(c)
    entropy_norm = normalized_entropy(c)
    unique_actions = len(c)

    mean_pnl = safe_float(df["pnl_net_sum"].mean()) if "pnl_net_sum" in df else float("nan")
    median_pnl = safe_float(df["pnl_net_sum"].median()) if "pnl_net_sum" in df else float("nan")
    mean_mdd = safe_float(df["mdd"].mean()) if "mdd" in df else float("nan")
    mean_cost = safe_float(df["cost_sum"].mean()) if "cost_sum" in df else float("nan")
    mean_h_std = safe_float(df["h_std"].mean()) if "h_std" in df else 0.0
    mean_h_abs = safe_float(df["h_abs_mean"].mean()) if "h_abs_mean" in df else 0.0
    mean_turnover_h = safe_float(df["turnover_h"].mean()) if "turnover_h" in df else 0.0
    mean_turnover_contract = safe_float(df["turnover_contract"].mean()) if "turnover_contract" in df else 0.0
    mean_cost_ratio = safe_float(mean_cost / (abs(mean_pnl) + 1.0), 0.0)

    q05_pnl = safe_float(df["pnl_net_sum"].quantile(0.05)) if "pnl_net_sum" in df else float("nan")
    prob_profit = safe_float((df["pnl_net_sum"] > 0).mean()) if "pnl_net_sum" in df else float("nan")

    # Dataset-specific diagnostics: useful because oracle_universe can dominate aggregate results.
    by_dataset = {}
    if "dataset" in df.columns:
        for dataset, g in df.groupby("dataset"):
            by_dataset[str(dataset)] = {
                "n": int(len(g)),
                "pnl_mean": safe_float(g["pnl_net_sum"].mean()) if "pnl_net_sum" in g else float("nan"),
                "mdd_mean": safe_float(g["mdd"].mean()) if "mdd" in g else float("nan"),
                "h_std_mean": safe_float(g["h_std"].mean()) if "h_std" in g else float("nan"),
                "turnover_h_mean": safe_float(g["turnover_h"].mean()) if "turnover_h" in g else float("nan"),
            }

    # Reject quasi-static or pathological policies. This is not the final score;
    # it prevents selecting a high-PnL no-hedge/full-hedge static policy.
    reject_reasons: List[str] = []
    if unique_actions < 2:
        reject_reasons.append("unique_actions<2")
    if mean_h_std < 0.02:
        reject_reasons.append("mean_h_std<0.02")
    if mean_turnover_h < 0.05:
        reject_reasons.append("mean_turnover_h<0.05")
    if mean_turnover_h > 3.0:
        reject_reasons.append("mean_turnover_h>3.0")
    if mean_cost_ratio > 0.20:
        reject_reasons.append("cost_ratio>20pct")

    # Score: risk-adjusted performance + small dynamic-policy bonus.
    # Units are dollar-ish; dynamic bonus is intentionally moderate.
    score = (
        mean_pnl
        - 0.65 * abs(mean_mdd)
        - 0.35 * mean_cost
        + 350_000.0 * entropy_norm
        + 1_250_000.0 * min(mean_h_std, 0.25)
        - 150_000.0 * max(mean_turnover_h - 1.25, 0.0)
    )
    if reject_reasons:
        score -= 10_000_000.0

    result = {
        **cfg,
        "rows": int(len(df)),
        "mean_pnl": mean_pnl,
        "median_pnl": median_pnl,
        "q05_pnl": q05_pnl,
        "prob_profit": prob_profit,
        "mean_mdd": mean_mdd,
        "mean_cost": mean_cost,
        "mean_cost_ratio": mean_cost_ratio,
        "mean_h_abs": mean_h_abs,
        "mean_h_std": mean_h_std,
        "mean_turnover_h": mean_turnover_h,
        "mean_turnover_contract": mean_turnover_contract,
        "unique_actions": unique_actions,
        "action_entropy": entropy,
        "action_entropy_norm": entropy_norm,
        "action_counts": json.dumps(dict(c), sort_keys=True),
        "by_dataset_json": json.dumps(by_dataset, sort_keys=True),
        "reject": bool(reject_reasons),
        "reject_reasons": ";".join(reject_reasons),
        "score": float(score),
        "out_dir": str(out_dir),
    }
    return result


# -----------------------------
# Trial execution
# -----------------------------


def run_trial(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir) / f"trial_{cfg['trial']:03d}"

    if out_dir.exists() and not args.force:
        p = out_dir / "results_all_windows.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            return summarize_trial(df, cfg, out_dir)
        shutil.rmtree(out_dir)
    elif out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = build_train_command(cfg, args, out_dir)
    (out_dir.parent / "commands").mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "commands" / f"trial_{cfg['trial']:03d}.sh").write_text(shell_command(cmd) + "\n", encoding="utf-8")

    log_path = out_dir / "tune_trial.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("COMMAND:\n" + shell_command(cmd) + "\n\n")
        logf.flush()
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)

    p = out_dir / "results_all_windows.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing trial result: {p}")
    df = pd.read_parquet(p)
    return summarize_trial(df, cfg, out_dir)


def elapsed_eta(start_ts: float, done: int, total: int) -> str:
    elapsed = max(time.time() - start_ts, 1e-9)
    if done <= 0:
        return f"elapsed={timedelta(seconds=int(elapsed))} ETA=--:--:--"
    rate = elapsed / done
    remaining = max(total - done, 0) * rate
    return f"elapsed={timedelta(seconds=int(elapsed))} ETA={timedelta(seconds=int(remaining))}"


def print_dashboard(results: List[Dict[str, Any]], total: int, start_ts: float) -> None:
    done = len(results)
    ok = sum(1 for r in results if not bool(r.get("failed", False)))
    fail = sum(1 for r in results if bool(r.get("failed", False)))
    accepted = sum(1 for r in results if (not bool(r.get("failed", False))) and (not bool(r.get("reject", True))))
    best_score = max([safe_float(r.get("score"), -1e18) for r in results], default=float("nan"))
    best_trial = None
    if results:
        best_row = max(results, key=lambda r: safe_float(r.get("score", -1e18)))
        best_trial = best_row.get("trial")
    print(
        f"[dashboard] done={done}/{total} ok={ok} accepted={accepted} fail={fail} "
        f"best_trial={best_trial} best_score={best_score:,.2f} {elapsed_eta(start_ts, done, total)}",
        flush=True,
    )


# -----------------------------
# Reporting
# -----------------------------


def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
        use_sns = True
    except Exception:
        sns = None
        use_sns = False

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=220)
        plt.close(fig)

    # Score distribution
    if "score" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        d = df[np.isfinite(pd.to_numeric(df["score"], errors="coerce"))]
        if use_sns:
            sns.histplot(d["score"], bins=25, kde=True, ax=ax)
        else:
            ax.hist(d["score"], bins=25)
        ax.set_title("Dynamic-policy tuner: score distribution")
        ax.set_xlabel("Composite score")
        ax.set_ylabel("Trial count")
        save(fig, "score_distribution.png")

    # PnL vs MDD colored by dynamic behavior
    needed = {"mean_pnl", "mean_mdd", "mean_h_std", "reject"}
    if needed.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        d = df.copy()
        d["accepted"] = ~d["reject"].astype(bool)
        if use_sns:
            sns.scatterplot(
                data=d,
                x="mean_pnl",
                y="mean_mdd",
                hue="accepted",
                size="mean_h_std",
                sizes=(40, 220),
                ax=ax,
            )
        else:
            ax.scatter(d["mean_pnl"], d["mean_mdd"], s=60)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Mean PnL vs MDD (marker size = h-std)")
        ax.set_xlabel("Mean net PnL (USD)")
        ax.set_ylabel("Mean MDD (USD; less negative is better)")
        save(fig, "pnl_vs_mdd_dynamic.png")

    # Dynamic behavior
    needed = {"mean_h_std", "mean_turnover_h", "action_entropy_norm", "reject"}
    if needed.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        d = df.copy()
        d["accepted"] = ~d["reject"].astype(bool)
        if use_sns:
            sns.scatterplot(
                data=d,
                x="mean_h_std",
                y="mean_turnover_h",
                hue="accepted",
                size="action_entropy_norm",
                sizes=(40, 220),
                ax=ax,
            )
        else:
            ax.scatter(d["mean_h_std"], d["mean_turnover_h"], s=60)
        ax.set_title("Dynamic behavior: hedge variability vs turnover")
        ax.set_xlabel("Mean h standard deviation")
        ax.set_ylabel("Mean hedge-ratio turnover")
        save(fig, "dynamic_behavior.png")

    # Score by action mode
    if {"action_mode", "score"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if use_sns:
            sns.boxplot(data=df, x="action_mode", y="score", ax=ax)
            sns.stripplot(data=df, x="action_mode", y="score", color="black", alpha=0.45, ax=ax)
        else:
            df.boxplot(column="score", by="action_mode", ax=ax)
        ax.set_title("Score by action mode")
        ax.set_xlabel("Action mode")
        ax.set_ylabel("Composite score")
        save(fig, "score_by_action_mode.png")

    # Heatmap-like pivot: eta_cost vs lambda_lpm median score
    if {"eta_cost", "lambda_lpm", "score"}.issubset(df.columns):
        piv = df.pivot_table(index="lambda_lpm", columns="eta_cost", values="score", aggfunc="median")
        if not piv.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            if use_sns:
                sns.heatmap(piv, annot=True, fmt=".0f", cmap="viridis", ax=ax)
            else:
                im = ax.imshow(piv.values, aspect="auto")
                ax.set_xticks(range(len(piv.columns)), labels=piv.columns)
                ax.set_yticks(range(len(piv.index)), labels=piv.index)
                fig.colorbar(im, ax=ax)
            ax.set_title("Median score by cost and LPM penalty")
            ax.set_xlabel("eta_cost")
            ax.set_ylabel("lambda_lpm")
            save(fig, "score_heatmap_eta_lpm.png")


def write_report(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# Dynamic RL policy tuning report\n")
    lines.append("## Purpose\n")
    lines.append(
        "This tuner searches for RL configurations that are not merely static hedge-ratio policies. "
        "The score rewards risk-adjusted PnL while adding a moderate bonus for genuine action/hedge-ratio diversity.\n"
    )
    lines.append("## Fixed experiment design\n")
    lines.append(f"- Exposure: `{args.exposure_id}`\n")
    lines.append(f"- Train mode: `{args.train_mode}`\n")
    lines.append(f"- Year range: {args.year_start} to {args.year_end}\n")
    lines.append(f"- Max windows: {args.max_windows}\n")
    lines.append(f"- Evaluation episodes: {args.eval_episodes}\n")
    lines.append(f"- Trials: {len(df)}\n")

    if df.empty:
        lines.append("\nNo results were produced.\n")
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return

    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)
    accepted = df_sorted[~df_sorted["reject"].astype(bool)].copy()
    lines.append("\n## Search summary\n")
    lines.append(f"- Accepted dynamic configs: {len(accepted)} / {len(df_sorted)}\n")
    lines.append(f"- Best score: {df_sorted['score'].max():,.2f}\n")
    lines.append(f"- Best accepted score: {accepted['score'].max():,.2f}\n" if len(accepted) else "- Best accepted score: none\n")

    best = accepted.iloc[0] if len(accepted) else df_sorted.iloc[0]
    lines.append("\n## Selected configuration\n")
    key_cols = [
        "trial", "score", "reject", "reject_reasons", "action_mode", "h_max", "delta_h_max", "delta_h_step",
        "mu_pnl", "lambda_lpm", "lpm_order", "lpm_target", "eta_cost", "ent_coef", "learning_rate", "clip_range",
        "mean_pnl", "mean_mdd", "mean_cost", "mean_h_std", "mean_turnover_h", "unique_actions", "action_entropy_norm",
    ]
    for c in key_cols:
        if c in best.index:
            lines.append(f"- {c}: `{best[c]}`\n")

    lines.append("\n## Top 10 accepted configs\n")
    show_cols = [c for c in key_cols if c in df.columns]
    if len(accepted):
        lines.append(accepted[show_cols].head(10).to_markdown(index=False))
        lines.append("\n")
    else:
        lines.append("No accepted configs. This means the current reward/action design still collapses to a near-static policy.\n")

    lines.append("\n## Interpretation notes\n")
    lines.append("- `mean_h_std` close to zero means the learned hedge ratio is effectively static.\n")
    lines.append("- High `unique_actions` alone is not sufficient; it must come with acceptable MDD, PnL, and turnover.\n")
    lines.append("- Rejected configs are not necessarily useless; they are rejected for the dynamic-policy objective of this tuner.\n")

    lines.append("\n## Generated files\n")
    lines.append("- `results.csv`: all trial-level results\n")
    lines.append("- `top_configs.csv`: sorted accepted configs first\n")
    lines.append("- `best_config.json`: selected config and final-train command\n")
    lines.append("- `plots/`: diagnostic plots\n")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_best_outputs(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    if df.empty:
        return
    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)
    accepted = df_sorted[~df_sorted["reject"].astype(bool)].copy()
    top = pd.concat([accepted, df_sorted[df_sorted["reject"].astype(bool)]], ignore_index=True)
    top.to_csv(out_dir / "top_configs.csv", index=False)

    best = accepted.iloc[0].to_dict() if len(accepted) else df_sorted.iloc[0].to_dict()
    cfg = {k: best[k] for k in [
        "timesteps", "action_mode", "h_max", "delta_h_max", "delta_h_step",
        "mu_pnl", "lambda_lpm", "lpm_order", "lpm_target", "eta_cost",
        "ent_coef", "learning_rate", "clip_range",
    ] if k in best}

    final_out = f"rl_runs/FINAL_{args.exposure_id}_DYNAMIC_SELECTED"
    cmd = build_train_command(
        cfg={**cfg, "trial": int(best.get("trial", -1)), "timesteps": args.final_timesteps},
        args=args,
        out_dir=Path(final_out),
    )
    payload = {
        "selected_trial": int(best.get("trial", -1)),
        "selected_by": "best accepted score if available, otherwise best score",
        "config": cfg,
        "metrics": {k: best.get(k) for k in [
            "score", "mean_pnl", "mean_mdd", "mean_cost", "mean_h_std", "mean_turnover_h",
            "unique_actions", "action_entropy_norm", "reject", "reject_reasons",
        ]},
        "final_train_command": shell_command(cmd),
    }
    (out_dir / "best_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "final_train_command.sh").write_text(shell_command(cmd) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tune dynamic RL hedge policy with thesis-grade diagnostics")
    ap.add_argument("--cache", default="rl_cache/precompute_OPEC_BASKET.npz")
    ap.add_argument("--scenario_dir", default="scenarios")
    ap.add_argument("--exposure_id", default="OPEC_BASKET")
    ap.add_argument("--out_dir", default=str(DEFAULT_BASE_OUT))
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true", help="Re-run trials even if result folders already exist")
    ap.add_argument("--parallel_trials", type=int, default=1, help="Number of tuning trials to run concurrently")
    ap.add_argument("--trial_timesteps", type=int, default=150_000, help="Fixed timesteps for every tuning trial; final selected config can use --final_timesteps")
    ap.add_argument("--lpm_order_fixed", type=int, default=2, help="Use fixed LPM order 1 or 2; set 0 to tune between 1 and 2")
    ap.add_argument("--action_mode_fixed", default="delta_h_discrete", choices=["delta_h_discrete", "h_levels"], help="Fixed action mode for the sweep")
    ap.add_argument("--h_max_fixed", type=float, default=0.78, help="Fixed hedge-ratio cap for all tuning trials")
    ap.add_argument("--delta_h_max_fixed", type=float, default=0.15, help="Fixed maximum one-step delta-h for delta_h_discrete sweeps")
    ap.add_argument("--delta_h_step_fixed", type=float, default=0.05, help="Fixed delta-h grid step for delta_h_discrete sweeps")

    # Training/evaluation scope
    ap.add_argument("--train_mode", default="hybrid_expanding")
    ap.add_argument("--year_start", type=int, default=2008)
    ap.add_argument("--year_end", type=int, default=2025)
    ap.add_argument("--max_windows", type=int, default=1)
    ap.add_argument("--eval_episodes", type=int, default=300)
    ap.add_argument("--max_train_scenarios", type=int, default=0)
    ap.add_argument("--max_eval_scenarios", type=int, default=0)

    # Fixed PPO/backtest controls
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vec", default="dummy")
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--ppo_gamma", type=float, default=0.97)
    ap.add_argument("--ppo_gae_lambda", type=float, default=0.97)
    ap.add_argument("--ppo_vf_coef", type=float, default=0.5)
    ap.add_argument("--ppo_n_epochs", type=int, default=5)
    ap.add_argument("--torch_threads", type=int, default=2)
    ap.add_argument("--torch_interop_threads", type=int, default=1)
    ap.add_argument("--final_timesteps", type=int, default=800_000)

    return ap.parse_args()



def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = ensure_dir(Path(args.out_dir))
    ensure_dir(out_dir / "commands")

    results_path = out_dir / "results.csv"
    previous = load_existing_results(results_path)
    done_trials = set(previous["trial"].astype(int).tolist()) if "trial" in previous.columns and not args.force else set()
    results: List[Dict[str, Any]] = previous.to_dict("records") if len(previous) and not args.force else []

    planned: List[Dict[str, Any]] = []
    for i in range(args.n_trials):
        cfg = sample_config(i, rng, args)
        if i in done_trials:
            print(f"[tune_dynamic] skip trial {i:03d} (already in results.csv)")
            continue
        planned.append(cfg)

    total_target = len(results) + len(planned)
    workers = max(1, int(args.parallel_trials))

    print(f"[tune_dynamic] out_dir={out_dir}")
    print(f"[tune_dynamic] trials={args.n_trials} | already_done={len(done_trials)} | to_run={len(planned)}")
    print(
        f"[tune_dynamic] parallel_trials={workers} | trial_timesteps={args.trial_timesteps} | "
        f"lpm_order_fixed={args.lpm_order_fixed} | action_mode_fixed={args.action_mode_fixed} | "
        f"h_max_fixed={args.h_max_fixed} | delta_h_max_fixed={args.delta_h_max_fixed} | "
        f"delta_h_step_fixed={args.delta_h_step_fixed}"
    )
    print("[tune_dynamic] trial stdout/stderr is written to each trial folder: tune_trial.log", flush=True)

    start = time.time()
    print_dashboard(results, total_target, start)

    def _execute(cfg: Dict[str, Any]) -> Dict[str, Any]:
        trial = int(cfg["trial"])
        print(
            f"[start] trial={trial:03d} action_mode={cfg['action_mode']} "
            f"h_max={cfg['h_max']} eta={cfg['eta_cost']} "
            f"lambda_lpm={cfg['lambda_lpm']} ent={cfg['ent_coef']}",
            flush=True,
        )
        try:
            res = run_trial(cfg, args)
            print(
                "score={score:,.2f} reject={reject} pnl={pnl} mdd={mdd} h_std={hstd:.4f} actions={actions}".format(
                    score=res["score"],
                    reject=res["reject"],
                    pnl=fmt_money(res["mean_pnl"]),
                    mdd=fmt_money(res["mean_mdd"]),
                    hstd=res["mean_h_std"],
                    actions=res["unique_actions"],
                ),
                flush=True,
            )
            return res
        except Exception as e:
            print(f"[fail] trial={trial:03d} error={e}", flush=True)
            return {**cfg, "failed": True, "error": str(e), "score": -1e18, "reject": True, "reject_reasons": "failed"}

    if planned:
        if workers == 1:
            for cfg in planned:
                results.append(_execute(cfg))
                df_now = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
                df_now.to_csv(results_path, index=False)
                write_best_outputs(df_now, args, out_dir)
                print_dashboard(results, total_target, start)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_execute, cfg) for cfg in planned]
                for fut in as_completed(futs):
                    results.append(fut.result())
                    df_now = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
                    df_now.to_csv(results_path, index=False)
                    write_best_outputs(df_now, args, out_dir)
                    print_dashboard(results, total_target, start)

    df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    df.to_csv(results_path, index=False)
    write_best_outputs(df, args, out_dir)
    save_plots(df, out_dir / "plots")
    write_report(df, args, out_dir)

    elapsed = time.time() - start
    print(f"[DONE] results: {results_path}")
    print(f"[DONE] report:  {out_dir / 'report.md'}")
    print(f"[DONE] best:    {out_dir / 'best_config.json'}")
    print(f"[DONE] elapsed: {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())