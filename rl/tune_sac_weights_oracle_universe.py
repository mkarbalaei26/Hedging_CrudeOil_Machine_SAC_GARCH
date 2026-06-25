from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

import numpy as np
import pandas as pd


def es95(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    q = np.quantile(x, 0.05)
    tail = x[x <= q]
    return float(tail.mean()) if len(tail) else float(q)


# ==== Helper functions for diagnostics and plotting ====

def safe_var(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    return float(x.var(ddof=1))


def hedging_effectiveness(hedged_pnl, nohedge_pnl):
    """Variance-reduction hedging effectiveness vs no hedge.

    HE = 1 - Var(hedged PnL) / Var(no-hedge PnL)
    Higher is better. Negative values mean the hedge increased variance.
    """
    var_h = safe_var(hedged_pnl)
    var_u = safe_var(nohedge_pnl)
    if not np.isfinite(var_h) or not np.isfinite(var_u) or var_u <= 0:
        return np.nan
    return float(1.0 - var_h / var_u)


def money_fmt(v, _pos=None):
    if pd.isna(v):
        return ""
    av = abs(float(v))
    if av >= 1e9:
        return f"{v/1e9:.1f}B"
    if av >= 1e6:
        return f"{v/1e6:.1f}M"
    if av >= 1e3:
        return f"{v/1e3:.0f}K"
    return f"{v:.0f}"


def generate_weight_grid(n=30):
    candidates = []
    for w_lpm in np.arange(0.25, 0.66, 0.05):
        for w_vol in np.arange(0.20, 0.56, 0.05):
            w_cost = 1.0 - w_lpm - w_vol
            if 0.10 <= w_cost <= 0.30:
                candidates.append({
                    "w_lpm": round(float(w_lpm), 2),
                    "w_volatility": round(float(w_vol), 2),
                    "w_cost": round(float(w_cost), 2),
                    "delta_h": None,
                    "gamma": None,
                    "timesteps": None,
                    "lpm_target": None,
                    "config_type": "reward_weight_grid",
                })

    candidates = sorted(
        candidates,
        key=lambda x: abs(x["w_lpm"] - 0.45) + abs(x["w_volatility"] - 0.35) + abs(x["w_cost"] - 0.20),
    )
    return candidates[:n]


def generate_policy_grid(args):
    candidates = []
    for delta_h in args.policy_delta_h_values:
        for gamma in args.policy_gamma_values:
            for timesteps in args.policy_timesteps_values:
                for lpm_target in args.policy_lpm_target_values:
                    candidates.append({
                        "w_lpm": round(float(args.fixed_reward_weight_lpm), 4),
                        "w_volatility": round(float(args.fixed_reward_weight_volatility), 4),
                        "w_cost": round(float(args.fixed_reward_weight_decision_cost), 4),
                        "delta_h": round(float(delta_h), 4),
                        "gamma": round(float(gamma), 5),
                        "timesteps": int(timesteps),
                        "lpm_target": round(float(lpm_target), 8),
                        "config_type": "policy_grid",
                    })
    return candidates


def generate_configs(args):
    if args.grid_type == "reward":
        return generate_weight_grid(args.n_configs)
    if args.grid_type == "policy":
        return generate_policy_grid(args)
    if args.grid_type == "both":
        return generate_weight_grid(args.n_configs) + generate_policy_grid(args)
    raise ValueError(args.grid_type)


def _run_one_config(args, cfg: dict, i: int, n_configs: int) -> dict:
    w_lpm = float(cfg["w_lpm"])
    w_vol = float(cfg["w_volatility"])
    w_cost = float(cfg["w_cost"])
    timesteps = int(cfg["timesteps"] if cfg.get("timesteps") is not None else args.timesteps_base + i)
    delta_h = float(cfg["delta_h"] if cfg.get("delta_h") is not None else args.delta_h)
    gamma = float(cfg["gamma"] if cfg.get("gamma") is not None else args.gamma)
    lpm_target = float(cfg["lpm_target"] if cfg.get("lpm_target") is not None else args.lpm_target)
    run_dir = expected_run_dir(args, cfg)

    # Resume source #1: already summarized, even if the heavy run folder was deleted.
    if args.resume and cfg_is_completed_in_summary(args, cfg):
        print("\n" + "=" * 100)
        print(f"[SKIP {i}/{n_configs}] already summarized in {summary_csv_path(args)}")
        print(
            f"config: weights=({w_lpm}, {w_vol}, {w_cost}) "
            f"delta_h={delta_h} gamma={gamma} timesteps={timesteps} lpm_target={lpm_target}"
        )
        print("=" * 100)
        return {"status": "skipped_summary", "run_dir": str(run_dir), "index": i}

    # Resume source #2: run folder exists and contains combined outputs.
    if args.resume and run_is_complete(run_dir):
        print("\n" + "=" * 100)
        print(f"[SKIP {i}/{n_configs}] already complete on disk: {run_dir}")
        print("=" * 100)
        ok = append_run_summary(args, run_dir)
        if ok and args.cleanup_runs:
            cleanup_run_dir(run_dir, enabled=True)
        return {"status": "skipped_disk", "run_dir": str(run_dir), "index": i}

    cmd = [
        "python", "-m", "rl.train_sac_portfolio_costaware",
        "--asset", args.asset,
        "--rolling-windows",
        "--timesteps", str(timesteps),
        "--parallel-windows", str(args.parallel_windows),
        "--torch-threads", str(args.torch_threads),
        "--blas-threads", str(args.blas_threads),
        "--h-min", str(args.h_min),
        "--h-max", str(args.h_max),
        "--delta-h", str(delta_h),
        "--gamma", str(gamma),
        "--lpm-target", str(lpm_target),
        "--reward-weight-lpm", str(w_lpm),
        "--reward-weight-volatility", str(w_vol),
        "--reward-weight-decision-cost", str(w_cost),
        "--plot-fraction", str(args.plot_fraction),
    ]

    print("\n" + "=" * 100)
    print(
        f"[RUN {i}/{n_configs}] type={cfg.get('config_type')} "
        f"weights=({w_lpm}, {w_vol}, {w_cost}) "
        f"delta_h={delta_h} gamma={gamma} timesteps={timesteps} lpm_target={lpm_target}"
    )
    print(" ".join(cmd))
    print("=" * 100)

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] training failed returncode={exc.returncode}: {run_dir}")

        # Even if the subprocess failed late because of disk/parquet/run_config,
        # useful CSV outputs may already exist. Try to ingest them.
        ok = append_run_summary(args, run_dir)
        if ok and args.cleanup_runs:
            cleanup_run_dir(run_dir, enabled=True)

        if args.continue_on_error:
            return {
                "status": "failed_continued",
                "run_dir": str(run_dir),
                "index": i,
                "elapsed": time.time() - start_time,
            }
        raise

    ok = append_run_summary(args, run_dir)
    if ok and args.cleanup_runs:
        cleanup_run_dir(run_dir, enabled=True)

    return {
        "status": "done",
        "run_dir": str(run_dir),
        "index": i,
        "elapsed": time.time() - start_time,
    }


def run_training(args, configs):
    ingest_existing_completed_runs(args, configs)

    if int(args.config_workers) <= 1:
        for i, cfg in enumerate(configs, 1):
            _run_one_config(args, cfg, i, len(configs))
        return

    print("\n" + "=" * 100)
    print(f"[CONFIG PARALLEL] running up to {int(args.config_workers)} configs concurrently")
    print("[WARNING] Reduce --parallel-windows / --torch-threads when using --config-workers > 1.")
    print("=" * 100)

    with ThreadPoolExecutor(max_workers=int(args.config_workers)) as ex:
        futures = [
            ex.submit(_run_one_config, args, cfg, i, len(configs))
            for i, cfg in enumerate(configs, 1)
        ]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                print(
                    f"[CONFIG DONE] index={result.get('index')} "
                    f"status={result.get('status')} run_dir={result.get('run_dir')}"
                )
            except Exception as exc:
                print(f"[CONFIG ERROR] {exc}")
                if not args.continue_on_error:
                    raise


def find_result_files(root: Path):
    patterns = [
        str(root / "**" / "results_all_windows.parquet"),
        str(root / "**" / "results_all_windows.csv"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(set(files))


def read_result_file(path):
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def expected_run_dir(args, cfg: dict) -> Path:
    asset = str(args.asset).lower()
    timesteps = int(cfg["timesteps"] if cfg.get("timesteps") is not None else args.timesteps_base)
    delta_h = float(cfg["delta_h"] if cfg.get("delta_h") is not None else args.delta_h)
    lpm_target = float(cfg["lpm_target"] if cfg.get("lpm_target") is not None else args.lpm_target)
    w_lpm = float(cfg["w_lpm"])
    w_vol = float(cfg["w_volatility"])
    w_cost = float(cfg["w_cost"])

    run_name = (
        f"{asset}_sac_lpm3aware_ROLL_T{timesteps}_"
        f"train2.0y_val6.0m_test6.0m_"
        f"hmin{args.h_min}_hmax{args.h_max}_dh{delta_h}_"
        f"wLPM{w_lpm}_wVOL{w_vol}_wCOST{w_cost}_tau{lpm_target}"
    )
    return Path(args.outputs_root) / run_name


def run_is_complete(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    markers = [
        run_dir / "results_all_windows.parquet",
        run_dir / "results_all_windows.csv",
        run_dir / "test_episode_summary_all_windows.parquet",
        run_dir / "test_episode_summary_all_windows.csv",
    ]
    return any(x.exists() and x.stat().st_size > 0 for x in markers)


def extract_config_from_run_name(run_dir: Path) -> dict:
    name = run_dir.name
    out = {
        "w_lpm": np.nan,
        "w_volatility": np.nan,
        "w_cost": np.nan,
        "delta_h": np.nan,
        "gamma": np.nan,
        "timesteps": np.nan,
        "lpm_target": np.nan,
        "h_min": np.nan,
        "h_max": np.nan,
    }

    try:
        m = re.search(r"_T(\d+)_", name)
        if m:
            out["timesteps"] = int(m.group(1))

        m = re.search(r"hmin([-+0-9.eE]+)_hmax([-+0-9.eE]+)_dh([-+0-9.eE]+)_", name)
        if m:
            out["h_min"] = float(m.group(1))
            out["h_max"] = float(m.group(2))
            out["delta_h"] = float(m.group(3))

        m = re.search(
            r"wLPM([-+0-9.eE]+)_wVOL([-+0-9.eE]+)_wCOST([-+0-9.eE]+)_tau([-+0-9.eE]+)",
            name,
        )
        if m:
            out["w_lpm"] = float(m.group(1))
            out["w_volatility"] = float(m.group(2))
            out["w_cost"] = float(m.group(3))
            out["lpm_target"] = float(m.group(4))
    except Exception:
        pass

    return out


def result_file_for_run(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "results_all_windows.parquet",
        run_dir / "results_all_windows.csv",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def summarize_run_dir(run_dir: Path) -> dict | None:
    rf = result_file_for_run(run_dir)
    if rf is None:
        return None

    try:
        df = read_result_file(rf)
    except Exception as exc:
        print(f"[WARN] cannot read result file {rf}: {exc}")
        return None

    if "scenario_kind" not in df.columns:
        print(f"[WARN] no scenario_kind in {rf}")
        return None

    d = df[df["scenario_kind"].astype(str).str.contains("oracle_universe", case=False, na=False)].copy()
    if d.empty:
        print(f"[WARN] no oracle_universe rows in {rf}")
        return None

    if "net_pnl_total" not in d.columns or "naive_pnl" not in d.columns:
        print(f"[WARN] missing net_pnl_total or naive_pnl in {rf}")
        return None

    run_cfg = extract_config_from_run(run_dir)
    # If run_config.json was not written because the process crashed late, recover from folder name.
    name_cfg = extract_config_from_run_name(run_dir)
    for k, v in name_cfg.items():
        if k not in run_cfg or pd.isna(run_cfg.get(k, np.nan)):
            run_cfg[k] = v

    sac_mean = float(pd.to_numeric(d["net_pnl_total"], errors="coerce").mean())
    naive_mean = float(pd.to_numeric(d["naive_pnl"], errors="coerce").mean())

    sac_es = es95(d["net_pnl_total"])
    naive_es = es95(d["naive_pnl"])

    if "no_hedge_pnl" in d.columns:
        nohedge_pnl = d["no_hedge_pnl"]
    elif "spot_pnl_total" in d.columns:
        nohedge_pnl = d["spot_pnl_total"]
    else:
        print(f"[WARN] no no-hedge PnL column in {rf}; skipping HE calculation")
        nohedge_pnl = pd.Series(np.nan, index=d.index)

    sac_he = hedging_effectiveness(d["net_pnl_total"], nohedge_pnl)
    naive_he = hedging_effectiveness(d["naive_pnl"], nohedge_pnl)

    mean_h = float(pd.to_numeric(d.get("mean_h", np.nan), errors="coerce").mean())
    turnover_h = float(pd.to_numeric(d.get("turnover_h", np.nan), errors="coerce").mean())

    flag_mean_pnl = sac_mean > naive_mean
    flag_es95 = sac_es > naive_es
    flag_he = sac_he > naive_he

    return {
        "run_dir": str(run_dir),
        "result_file": str(rf),
        "w_lpm": run_cfg["w_lpm"],
        "w_volatility": run_cfg["w_volatility"],
        "w_cost": run_cfg["w_cost"],
        "delta_h": run_cfg["delta_h"],
        "gamma": run_cfg["gamma"],
        "timesteps": run_cfg["timesteps"],
        "lpm_target": run_cfg["lpm_target"],
        "h_min": run_cfg["h_min"],
        "h_max": run_cfg["h_max"],
        "n_scenarios": int(len(d)),
        "sac_mean_pnl": sac_mean,
        "naive_mean_pnl": naive_mean,
        "diff_mean_pnl_vs_naive": sac_mean - naive_mean,
        "sac_es95": sac_es,
        "naive_es95": naive_es,
        "diff_es95_vs_naive": sac_es - naive_es,
        "sac_he": sac_he,
        "naive_he": naive_he,
        "diff_he_vs_naive": sac_he - naive_he,
        "mean_h_diagnostic": mean_h,
        "turnover_h": turnover_h,
        "flag_mean_pnl_better_than_naive": flag_mean_pnl,
        "flag_es95_better_than_naive": flag_es95,
        "flag_he_better_than_naive": flag_he,
        "flags_total": int(flag_mean_pnl) + int(flag_es95) + int(flag_he),
    }


def summary_csv_path(args) -> Path:
    return Path(args.out_dir) / "sac_weight_tuning_oracle_universe_summary.csv"


def load_incremental_summary(args) -> pd.DataFrame:
    path = summary_csv_path(args)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def config_key_from_values(w_lpm, w_volatility, w_cost, delta_h, gamma, timesteps, lpm_target) -> tuple:
    def rf(x, nd=10):
        try:
            if pd.isna(x):
                return np.nan
            return round(float(x), nd)
        except Exception:
            return np.nan

    def ri(x):
        try:
            if pd.isna(x):
                return np.nan
            return int(float(x))
        except Exception:
            return np.nan

    return (
        rf(w_lpm),
        rf(w_volatility),
        rf(w_cost),
        rf(delta_h),
        rf(gamma),
        ri(timesteps),
        rf(lpm_target),
    )


def config_key_from_cfg(args, cfg: dict) -> tuple:
    return config_key_from_values(
        cfg.get("w_lpm"),
        cfg.get("w_volatility"),
        cfg.get("w_cost"),
        cfg.get("delta_h") if cfg.get("delta_h") is not None else args.delta_h,
        cfg.get("gamma") if cfg.get("gamma") is not None else args.gamma,
        cfg.get("timesteps") if cfg.get("timesteps") is not None else args.timesteps_base,
        cfg.get("lpm_target") if cfg.get("lpm_target") is not None else args.lpm_target,
    )


def completed_keys_from_summary(args) -> set:
    df = load_incremental_summary(args)
    if df.empty:
        return set()

    required = ["w_lpm", "w_volatility", "w_cost", "delta_h", "gamma", "timesteps", "lpm_target"]
    for c in required:
        if c not in df.columns:
            return set()

    return {
        config_key_from_values(
            r["w_lpm"],
            r["w_volatility"],
            r["w_cost"],
            r["delta_h"],
            r["gamma"],
            r["timesteps"],
            r["lpm_target"],
        )
        for _, r in df.iterrows()
    }


def cfg_is_completed_in_summary(args, cfg: dict) -> bool:
    return config_key_from_cfg(args, cfg) in completed_keys_from_summary(args)


def save_incremental_summary(args, out: pd.DataFrame) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if out.empty:
        return

    # Deduplicate using the economically meaningful config key.
    key_cols = ["w_lpm", "w_volatility", "w_cost", "delta_h", "gamma", "timesteps", "lpm_target"]
    for c in key_cols:
        if c not in out.columns:
            out[c] = np.nan

    for c in key_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.drop_duplicates(subset=key_cols, keep="last").copy()

    out["rank_score"] = (
        out["diff_mean_pnl_vs_naive"].rank(ascending=False)
        + out["diff_es95_vs_naive"].rank(ascending=False)
        + out["diff_he_vs_naive"].rank(ascending=False)
        + out["turnover_h"].rank(ascending=True)
    )
    out = out.sort_values(["flags_total", "rank_score"], ascending=[False, True]).reset_index(drop=True)

    out.to_csv(args.out_dir / "sac_weight_tuning_oracle_universe_summary.csv", index=False)
    flagged = out[
        out[
            [
                "flag_mean_pnl_better_than_naive",
                "flag_es95_better_than_naive",
                "flag_he_better_than_naive",
            ]
        ].any(axis=1)
    ].copy()
    flagged.to_csv(args.out_dir / "flagged_better_than_naive.csv", index=False)

    try:
        out.to_parquet(args.out_dir / "sac_weight_tuning_oracle_universe_summary.parquet", index=False)
    except Exception as exc:
        print(f"[WARN] could not write summary parquet: {exc}")


def append_run_summary(args, run_dir: Path) -> bool:
    row = summarize_run_dir(run_dir)
    if row is None:
        return False

    old = load_incremental_summary(args)
    new = pd.DataFrame([row])
    out = pd.concat([old, new], ignore_index=True)
    save_incremental_summary(args, out)
    print(f"[SUMMARY] saved oracle_universe summary for: {run_dir.name}")
    return True


def cleanup_run_dir(run_dir: Path, *, enabled: bool) -> None:
    if not enabled:
        return
    if not run_dir.exists():
        return
    try:
        shutil.rmtree(run_dir)
        print(f"[CLEANUP] deleted run folder: {run_dir}")
    except Exception as exc:
        print(f"[WARN] could not delete {run_dir}: {exc}")


def ingest_existing_completed_runs(args, configs) -> None:
    if not args.ingest_existing:
        return

    print("\n" + "=" * 100)
    print("[INGEST] checking existing completed runs before starting")
    print("=" * 100)

    for cfg in configs:
        run_dir = expected_run_dir(args, cfg)
        if run_is_complete(run_dir):
            ok = append_run_summary(args, run_dir)
            if ok and args.cleanup_runs:
                cleanup_run_dir(run_dir, enabled=True)



def extract_config_from_run(run_dir: Path):
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return extract_config_from_run_name(run_dir)

    cfg = json.loads(cfg_path.read_text())
    args = cfg.get("args", cfg)

    def fget(key, fallback=np.nan):
        return args.get(key, cfg.get(key, fallback))

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    def to_int(x):
        try:
            return int(float(x))
        except Exception:
            return np.nan

    return {
        "w_lpm": to_float(fget("reward_weight_lpm")),
        "w_volatility": to_float(fget("reward_weight_volatility")),
        "w_cost": to_float(fget("reward_weight_decision_cost")),
        "delta_h": to_float(fget("delta_h")),
        "gamma": to_float(fget("gamma")),
        "timesteps": to_int(fget("timesteps")),
        "lpm_target": to_float(fget("lpm_target")),
        "h_min": to_float(fget("h_min")),
        "h_max": to_float(fget("h_max")),
    }


# ==== Plotting functions for tuning diagnostics ====

def _prepare_plot_dir(out_dir: Path) -> Path:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    return plot_dir


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tuning_results(out: pd.DataFrame, out_dir: Path) -> None:
    """Create thesis-ready diagnostic plots for reward-weight tuning."""
    if out.empty:
        return
    plot_dir = _prepare_plot_dir(out_dir)
    d = out.copy()
    d["config_label"] = d.apply(
        lambda r: (
            f"L{r['w_lpm']:.2f}/V{r['w_volatility']:.2f}/C{r['w_cost']:.2f} | "
            f"dh{r.get('delta_h', np.nan):.2f}/g{r.get('gamma', np.nan):.3f}/"
            f"T{int(r.get('timesteps', 0)) if pd.notna(r.get('timesteps', np.nan)) else 'NA'}/"
            f"tau{r.get('lpm_target', np.nan):.4g}"
        ),
        axis=1,
    )
    d["he_sac_pct"] = 100.0 * pd.to_numeric(d["sac_he"], errors="coerce")
    d["he_naive_pct"] = 100.0 * pd.to_numeric(d["naive_he"], errors="coerce")
    d["diff_he_pct"] = 100.0 * pd.to_numeric(d["diff_he_vs_naive"], errors="coerce")

    top_n = min(15, len(d))
    top = d.sort_values("rank_score").head(top_n).copy()

    # 1) Ranking bar: composite rank score.
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(data=top, x="rank_score", y="config_label", hue="flags_total", dodge=False, palette="viridis", ax=ax)
    ax.set_title("Top SAC Reward-Weight Configurations: Composite Rank")
    ax.set_xlabel("Composite rank score, lower is better")
    ax.set_ylabel("Reward weights: LPM / Volatility / Cost")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Flags won")
    _save_fig(fig, plot_dir / "rank_top_configs.png")

    # 2) Mean PnL vs ES95: risk-return tail trade-off.
    fig, ax = plt.subplots(figsize=(13, 8))
    sc = ax.scatter(
        d["sac_es95"],
        d["sac_mean_pnl"],
        c=d["diff_he_pct"],
        s=80 + 45 * d["flags_total"],
        cmap="coolwarm",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(d["naive_mean_pnl"].median(), color="black", linestyle="--", linewidth=1.2, label="Median Naive mean PnL")
    ax.axvline(d["naive_es95"].median(), color="gray", linestyle="--", linewidth=1.2, label="Median Naive ES95")
    ax.set_title("Risk-Return Tail Trade-off across SAC Weight Configurations")
    ax.set_xlabel("SAC ES95 / CVaR95 of Net PnL, higher is better")
    ax.set_ylabel("SAC Mean Net PnL, higher is better")
    ax.xaxis.set_major_formatter(FuncFormatter(money_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(money_fmt))
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("HE advantage over Naive, percentage points")
    ax.legend(loc="best")
    _save_fig(fig, plot_dir / "scatter_mean_pnl_vs_es95_colored_by_he.png")

    # 3) HE vs ES improvement.
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.scatterplot(
        data=d,
        x="diff_es95_vs_naive",
        y="diff_he_pct",
        size="diff_mean_pnl_vs_naive",
        hue="flags_total",
        palette="viridis",
        sizes=(80, 450),
        alpha=0.85,
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_title("SAC Improvement over Naive: ES95 vs Hedging Effectiveness")
    ax.set_xlabel("Δ ES95 vs Naive, higher is better")
    ax.set_ylabel("Δ Hedging Effectiveness vs Naive, percentage points")
    ax.xaxis.set_major_formatter(FuncFormatter(money_fmt))
    _save_fig(fig, plot_dir / "scatter_delta_es95_vs_delta_he.png")

    # 4) HE vs turnover/cost diagnostics.
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.scatterplot(
        data=d,
        x="turnover_h",
        y="he_sac_pct",
        hue="w_cost",
        size="sac_mean_pnl",
        palette="mako",
        sizes=(80, 450),
        alpha=0.85,
        ax=ax,
    )
    ax.axhline(d["he_naive_pct"].median(), color="black", linestyle="--", linewidth=1.2, label="Median Naive HE")
    ax.set_title("Hedging Effectiveness versus Hedge Turnover")
    ax.set_xlabel("Mean hedge-ratio turnover")
    ax.set_ylabel("SAC Hedging Effectiveness (%)")
    ax.legend(loc="best")
    _save_fig(fig, plot_dir / "scatter_he_vs_turnover.png")

    # 5) Pair plot for statistical overview.
    pair_cols = [
        "w_lpm", "w_volatility", "w_cost", "sac_mean_pnl", "sac_es95", "sac_he", "turnover_h", "flags_total"
    ]
    pair = d[pair_cols].dropna().copy()
    if len(pair) >= 3:
        g = sns.pairplot(
            pair,
            vars=["w_lpm", "w_volatility", "w_cost", "sac_mean_pnl", "sac_es95", "sac_he", "turnover_h"],
            hue="flags_total",
            palette="viridis",
            corner=True,
            diag_kind="hist",
            plot_kws={"alpha": 0.75, "s": 45, "edgecolor": "black", "linewidth": 0.3},
        )
        g.fig.suptitle("SAC Reward-Weight Tuning: Pairwise Statistical Diagnostics", y=1.02)
        g.fig.savefig(plot_dir / "pairplot_tuning_diagnostics.png", dpi=180, bbox_inches="tight")
        plt.close(g.fig)

    # 6) Heatmaps over LPM/VOL weights for the best cost slice(s).
    for metric, title, filename in [
        ("sac_mean_pnl", "Mean Net PnL", "heatmap_mean_pnl.png"),
        ("sac_es95", "ES95 / CVaR95", "heatmap_es95.png"),
        ("sac_he", "Hedging Effectiveness", "heatmap_he.png"),
        ("rank_score", "Composite Rank Score", "heatmap_rank_score.png"),
    ]:
        pivot = d.pivot_table(index="w_lpm", columns="w_volatility", values=metric, aggfunc="mean")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot.sort_index(ascending=False), annot=True, fmt=".2g", cmap="viridis", ax=ax)
        ax.set_title(f"Reward Weight Grid Heatmap: {title}")
        ax.set_xlabel("Volatility weight")
        ax.set_ylabel("LPM weight")
        _save_fig(fig, plot_dir / filename)

    # 7) 3D view: weight simplex diagnostics.
    try:
        fig = plt.figure(figsize=(13, 9))
        ax = fig.add_subplot(111, projection="3d")
        sc3 = ax.scatter(
            d["w_lpm"], d["w_volatility"], d["w_cost"],
            c=d["rank_score"], cmap="viridis_r", s=75 + 35 * d["flags_total"], alpha=0.9
        )
        ax.set_title("3D Weight Space: Composite Rank across SAC Configurations")
        ax.set_xlabel("LPM weight")
        ax.set_ylabel("Volatility weight")
        ax.set_zlabel("Cost weight")
        cb = fig.colorbar(sc3, ax=ax, pad=0.1)
        cb.set_label("Composite rank score, lower is better")
        _save_fig(fig, plot_dir / "3d_weight_space_rank.png")
    except Exception as exc:
        print(f"[WARN] 3D plot failed: {exc}")



def plot_policy_grid_extra(out: pd.DataFrame, out_dir: Path) -> None:
    if out.empty:
        return

    required_cols = ["delta_h", "gamma", "timesteps", "lpm_target"]
    for col in required_cols:
        if col not in out.columns:
            out[col] = np.nan

    if not (
        out["delta_h"].nunique(dropna=True) > 1
        or out["gamma"].nunique(dropna=True) > 1
        or out["timesteps"].nunique(dropna=True) > 1
        or out["lpm_target"].nunique(dropna=True) > 1
    ):
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    d = out.copy()

    # 1) Timesteps / delta_h / gamma diagnostic.
    if d["timesteps"].nunique(dropna=True) > 1:
        fig, ax = plt.subplots(figsize=(13, 8))
        sns.scatterplot(
            data=d,
            x="timesteps",
            y="rank_score",
            hue="delta_h",
            style="gamma",
            size="flags_total",
            sizes=(90, 420),
            palette="viridis",
            alpha=0.9,
            ax=ax,
        )
        ax.set_title("Policy Dynamics Grid: Timesteps, Delta-H, Gamma versus Rank")
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel("Composite rank score, lower is better")
        _save_fig(fig, plot_dir / "policy_grid_rank_vs_timesteps.png")

    # 2) Delta-H × Gamma heatmaps.
    if d["delta_h"].nunique(dropna=True) > 1 and d["gamma"].nunique(dropna=True) > 1:
        for metric, title, filename in [
            ("sac_he", "SAC Hedging Effectiveness", "policy_heatmap_he_by_delta_gamma.png"),
            ("sac_es95", "SAC ES95 / CVaR95", "policy_heatmap_es95_by_delta_gamma.png"),
            ("sac_mean_pnl", "SAC Mean PnL", "policy_heatmap_mean_pnl_by_delta_gamma.png"),
            ("rank_score", "Composite Rank Score", "policy_heatmap_rank_by_delta_gamma.png"),
        ]:
            if metric not in d.columns:
                continue
            pivot = d.pivot_table(index="delta_h", columns="gamma", values=metric, aggfunc="mean")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(11, 7))
            sns.heatmap(pivot.sort_index(ascending=False), annot=True, fmt=".2g", cmap="viridis", ax=ax)
            ax.set_title(f"Policy Grid Heatmap: {title}")
            ax.set_xlabel("Gamma")
            ax.set_ylabel("Delta-H")
            _save_fig(fig, plot_dir / filename)

    # 3) LPM target diagnostics.
    if d["lpm_target"].nunique(dropna=True) > 1:
        fig, ax = plt.subplots(figsize=(13, 8))
        sns.scatterplot(
            data=d,
            x="lpm_target",
            y="rank_score",
            hue="delta_h",
            style="timesteps",
            size="flags_total",
            sizes=(90, 420),
            palette="viridis",
            alpha=0.9,
            ax=ax,
        )
        ax.set_title("LPM Target Grid: Target Threshold versus Composite Rank")
        ax.set_xlabel("LPM target threshold on cumulative return")
        ax.set_ylabel("Composite rank score, lower is better")
        _save_fig(fig, plot_dir / "policy_grid_rank_vs_lpm_target.png")

        for metric, title, filename in [
            ("sac_he", "SAC Hedging Effectiveness", "policy_heatmap_he_by_lpm_target_delta.png"),
            ("sac_es95", "SAC ES95 / CVaR95", "policy_heatmap_es95_by_lpm_target_delta.png"),
            ("sac_mean_pnl", "SAC Mean PnL", "policy_heatmap_mean_pnl_by_lpm_target_delta.png"),
            ("rank_score", "Composite Rank Score", "policy_heatmap_rank_by_lpm_target_delta.png"),
        ]:
            if metric not in d.columns:
                continue
            pivot = d.pivot_table(index="lpm_target", columns="delta_h", values=metric, aggfunc="mean")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(pivot.sort_index(ascending=False), annot=True, fmt=".2g", cmap="viridis", ax=ax)
            ax.set_title(f"LPM Target × Delta-H Heatmap: {title}")
            ax.set_xlabel("Delta-H")
            ax.set_ylabel("LPM target")
            _save_fig(fig, plot_dir / filename)

    # 4) Timesteps × LPM target heatmaps.
    if d["timesteps"].nunique(dropna=True) > 1 and d["lpm_target"].nunique(dropna=True) > 1:
        for metric, title, filename in [
            ("sac_he", "SAC Hedging Effectiveness", "policy_heatmap_he_by_lpm_target_timesteps.png"),
            ("sac_es95", "SAC ES95 / CVaR95", "policy_heatmap_es95_by_lpm_target_timesteps.png"),
            ("sac_mean_pnl", "SAC Mean PnL", "policy_heatmap_mean_pnl_by_lpm_target_timesteps.png"),
            ("rank_score", "Composite Rank Score", "policy_heatmap_rank_by_lpm_target_timesteps.png"),
        ]:
            if metric not in d.columns:
                continue
            pivot = d.pivot_table(index="lpm_target", columns="timesteps", values=metric, aggfunc="mean")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(pivot.sort_index(ascending=False), annot=True, fmt=".2g", cmap="viridis", ax=ax)
            ax.set_title(f"LPM Target × Timesteps Heatmap: {title}")
            ax.set_xlabel("Training timesteps")
            ax.set_ylabel("LPM target")
            _save_fig(fig, plot_dir / filename)


def analyze_outputs(args):
    files = find_result_files(Path(args.outputs_root))
    if not files:
        raise FileNotFoundError(f"No results_all_windows files found under {args.outputs_root}")

    rows = []

    for f in files:
        p = Path(f)
        run_dir = p.parent
        df = read_result_file(p)

        if "scenario_kind" not in df.columns:
            continue

        d = df[df["scenario_kind"].astype(str).str.contains("oracle_universe", case=False, na=False)].copy()
        if d.empty:
            continue

        if "net_pnl_total" not in d.columns:
            continue
        if "naive_pnl" not in d.columns:
            print(f"[WARN] no naive_pnl in {p}")
            continue

        run_cfg = extract_config_from_run(run_dir)
        w_lpm = run_cfg["w_lpm"]
        w_vol = run_cfg["w_volatility"]
        w_cost = run_cfg["w_cost"]

        sac_mean = float(pd.to_numeric(d["net_pnl_total"], errors="coerce").mean())
        naive_mean = float(pd.to_numeric(d["naive_pnl"], errors="coerce").mean())

        sac_es = es95(d["net_pnl_total"])
        naive_es = es95(d["naive_pnl"])

        if "no_hedge_pnl" in d.columns:
            nohedge_pnl = d["no_hedge_pnl"]
        elif "spot_pnl_total" in d.columns:
            nohedge_pnl = d["spot_pnl_total"]
        else:
            print(f"[WARN] no no-hedge PnL column in {p}; skipping HE calculation")
            nohedge_pnl = pd.Series(np.nan, index=d.index)

        sac_he = hedging_effectiveness(d["net_pnl_total"], nohedge_pnl)
        naive_he = hedging_effectiveness(d["naive_pnl"], nohedge_pnl)

        mean_h = float(pd.to_numeric(d.get("mean_h", np.nan), errors="coerce").mean())
        turnover_h = float(pd.to_numeric(d.get("turnover_h", np.nan), errors="coerce").mean())

        flag_mean_pnl = sac_mean > naive_mean
        flag_es95 = sac_es > naive_es  # ES کمتر منفی یعنی بهتر
        flag_he = sac_he > naive_he

        rows.append({
            "run_dir": str(run_dir),
            "w_lpm": w_lpm,
            "w_volatility": w_vol,
            "w_cost": w_cost,
            "delta_h": run_cfg["delta_h"],
            "gamma": run_cfg["gamma"],
            "timesteps": run_cfg["timesteps"],
            "lpm_target": run_cfg["lpm_target"],
            "h_min": run_cfg["h_min"],
            "h_max": run_cfg["h_max"],
            "n_scenarios": int(len(d)),
            "sac_mean_pnl": sac_mean,
            "naive_mean_pnl": naive_mean,
            "diff_mean_pnl_vs_naive": sac_mean - naive_mean,
            "sac_es95": sac_es,
            "naive_es95": naive_es,
            "diff_es95_vs_naive": sac_es - naive_es,
            "sac_he": sac_he,
            "naive_he": naive_he,
            "diff_he_vs_naive": sac_he - naive_he,
            "mean_h_diagnostic": mean_h,
            "turnover_h": turnover_h,
            "flag_mean_pnl_better_than_naive": flag_mean_pnl,
            "flag_es95_better_than_naive": flag_es95,
            "flag_he_better_than_naive": flag_he,
            "flags_total": int(flag_mean_pnl) + int(flag_es95) + int(flag_he),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No analyzable oracle_universe rows found.")

    out["rank_score"] = (
        out["diff_mean_pnl_vs_naive"].rank(ascending=False)
        + out["diff_es95_vs_naive"].rank(ascending=False)
        + out["diff_he_vs_naive"].rank(ascending=False)
        + out["turnover_h"].rank(ascending=True)
    )

    out = out.sort_values(["flags_total", "rank_score"], ascending=[False, True]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "sac_weight_tuning_oracle_universe_summary.csv", index=False)
    out.to_parquet(args.out_dir / "sac_weight_tuning_oracle_universe_summary.parquet", index=False)

    flagged = out[
        out[
            [
                "flag_mean_pnl_better_than_naive",
                "flag_es95_better_than_naive",
                "flag_he_better_than_naive",
            ]
        ].any(axis=1)
    ].copy()
    flagged.to_csv(args.out_dir / "flagged_better_than_naive.csv", index=False)

    if not getattr(args, "no_plots", False):
        plot_tuning_results(out, args.out_dir)
        plot_policy_grid_extra(out, args.out_dir)

    print("\n" + "=" * 100)
    print("BEST OVERALL")
    print("=" * 100)
    print(out.head(10)[[
        "w_lpm", "w_volatility", "w_cost", "delta_h", "gamma", "timesteps", "lpm_target",
        "sac_mean_pnl", "naive_mean_pnl", "diff_mean_pnl_vs_naive",
        "sac_es95", "naive_es95", "diff_es95_vs_naive",
        "sac_he", "naive_he", "diff_he_vs_naive", "mean_h_diagnostic", "turnover_h",
        "flags_total", "run_dir"
    ]].to_string(index=False))

    print("\n" + "=" * 100)
    print("FLAGGED MODELS")
    print("=" * 100)
    print(flagged[[
        "w_lpm", "w_volatility", "w_cost", "delta_h", "gamma", "timesteps", "lpm_target",
        "flag_mean_pnl_better_than_naive",
        "flag_es95_better_than_naive",
        "flag_he_better_than_naive",
        "run_dir"
    ]].head(30).to_string(index=False))

    print(f"\nSaved to: {args.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["print-grid", "run", "analyze", "run-and-analyze"], default="analyze")
    ap.add_argument("--n-configs", type=int, default=30)
    ap.add_argument(
        "--grid-type",
        choices=["reward", "policy", "both"],
        default="reward",
        help="reward=tune reward weights; policy=tune delta_h/gamma/timesteps with fixed reward weights; both=run both grids.",
    )

    ap.add_argument("--asset", default="OPEC")
    ap.add_argument("--timesteps-base", type=int, default=10000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lpm-target", type=float, default=0.0)
    ap.add_argument("--parallel-windows", type=int, default=12)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--blas-threads", type=int, default=2)

    ap.add_argument("--h-min", type=float, default=0.0)
    ap.add_argument("--h-max", type=float, default=1.25)
    ap.add_argument("--delta-h", type=float, default=0.10)
    ap.add_argument("--plot-fraction", type=float, default=0.0)

    ap.add_argument("--fixed-reward-weight-lpm", type=float, default=0.45)
    ap.add_argument("--fixed-reward-weight-volatility", type=float, default=0.35)
    ap.add_argument("--fixed-reward-weight-decision-cost", type=float, default=0.20)
    ap.add_argument("--policy-delta-h-values", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--policy-gamma-values", type=float, nargs="+", default=[0.99, 0.995])
    ap.add_argument("--policy-timesteps-values", type=int, nargs="+", default=[2000, 4000])
    ap.add_argument("--policy-lpm-target-values", type=float, nargs="+", default=[0.0])

    ap.add_argument("--outputs-root", type=Path, default=Path("rl_outputs/sac_portfolio_lpm"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/sac_weight_tuning_oracle_universe"))
    ap.add_argument("--no-plots", action="store_true", help="Skip seaborn/matplotlib tuning plots during analyze mode.")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip configs whose output folder already contains combined results.")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume and rerun all configs.")
    ap.add_argument("--continue-on-error", action="store_true", help="Continue sweep if one config fails.")
    ap.add_argument("--cleanup-runs", action="store_true", help="After summarizing a completed run, delete its heavy output folder.")
    ap.add_argument("--ingest-existing", action="store_true", help="Before running, ingest existing completed run folders into the summary.")
    ap.add_argument(
        "--config-workers",
        type=int,
        default=1,
        help="Number of full config subprocesses to run concurrently. Use 1 for safest behavior; use 2 to overlap combine/save of one config with training of another.",
    )

    args = ap.parse_args()
    configs = generate_configs(args)

    if args.mode == "print-grid":
        for i, cfg in enumerate(configs, 1):
            print(i, cfg)
        return

    if args.mode in {"run", "run-and-analyze"}:
        run_training(args, configs)

    if args.mode in {"analyze", "run-and-analyze"}:
        if args.cleanup_runs and summary_csv_path(args).exists():
            out = load_incremental_summary(args)
            save_incremental_summary(args, out)
            if not getattr(args, "no_plots", False):
                plot_tuning_results(out, args.out_dir)
                plot_policy_grid_extra(out, args.out_dir)
            print(f"\nSaved incremental summary to: {args.out_dir}")
        else:
            analyze_outputs(args)


if __name__ == "__main__":
    main()