

"""Train and evaluate a simple SAC hedging agent with portfolio-LPM reward.

First clean experiment for the thesis project:
- Uses precomputed .npz files from rl_cache.
- Environment: SACPortfolioLPMEnv.
- Action: delta hedge ratio, starting from h0=1.
- Reward: LPM of total portfolio PnL plus decision-cost/smoothness penalties.
- Evaluation exports daily episode logs and summary metrics.

Run examples from project root:
    python -m rl.train_sac_portfolio_lpm --asset WTI --timesteps 50000
    python -m rl.train_sac_portfolio_lpm --asset BRENT --timesteps 50000
    python -m rl.train_sac_portfolio_lpm --asset OPEC --timesteps 50000

For a smoke run:
    python -m rl.train_sac_portfolio_lpm --asset WTI --timesteps 1000 --eval-episodes 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required. Install with: pip install stable-baselines3"
    ) from exc

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


from rl.SACPortfolioLPMEnv import SACPortfolioLPMConfig, SACPortfolioLPMEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "rl_cache"
OUTPUT_DIR = PROJECT_ROOT / "rl_outputs" / "sac_portfolio_lpm"


ASSET_TO_FILE = {
    "WTI": "precompute_WTI.npz",
    "BRENT": "precompute_BRENT.npz",
    "OPEC": "precompute_OPEC.npz",
}

ASSET_TO_EXPOSURE_ID = {
    "WTI": "WTI_SPOT",
    "BRENT": "BRENT_SPOT",
    "OPEC": "OPEC_BASKET",
}

SCENARIO_ROOT = PROJECT_ROOT / "scenarios"


SCENARIO_FILES_BY_KIND = {
    "baseline": ["baseline.parquet", "scenarios_baseline.parquet"],
    "company": ["companies.parquet", "company.parquet", "scenarios_company.parquet"],
    "oracle_universe": ["oracle_universe.parquet"],
    "oracle_all": ["oracle_all.parquet"],
}

# Columns to include in episode reports
SCENARIO_META_COLS_FOR_REPORT = [
    "scenario_id",
    "scenario_record_id",
    "exposure_id",
    "scenario_kind",
    "start_date",
    "end_date",
    "horizon_days",
    "horizon_days_target",
    "horizon_days_realized",
    "volume_bbl",
    "oracle_series",
    "oracle_pool",
    "oracle_freq",
    "label",
    "tag",
    "oracle_candidate",
    "oracle_bucket",
    "scenario_kind_original",
    "company_id",
    "company_size",
    "scenario_file",
    "scenario_source_file",
]


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def load_precompute(asset: str):
    asset = asset.upper()
    if asset not in ASSET_TO_FILE:
        raise ValueError(f"Unknown asset {asset!r}. Valid choices: {sorted(ASSET_TO_FILE)}")
    path = CACHE_DIR / ASSET_TO_FILE[asset]
    if not path.exists():
        raise FileNotFoundError(f"Precompute file not found: {path}")
    return np.load(path, allow_pickle=True), path


def decode_feature_names(pre) -> List[str]:
    if "feature_names" not in pre:
        return []
    names = []
    for x in np.asarray(pre["feature_names"]).tolist():
        if isinstance(x, bytes):
            names.append(x.decode("utf-8"))
        else:
            names.append(str(x))
    return names


def filter_feature_matrix(pre, mode: str) -> Dict[str, np.ndarray]:
    """Return a dict-like precompute object, optionally dropping NaN-heavy features.

    mode:
        all          -> keep all features, NaNs are handled by env nan_to_num
        core_no_nan  -> keep columns with finite values across the whole file
    """
    data = {k: pre[k] for k in pre.files}
    X = np.asarray(pre["feature_matrix"], dtype=np.float32)
    names = decode_feature_names(pre)

    if mode == "all":
        return data
    if mode != "core_no_nan":
        raise ValueError("feature_mode must be 'all' or 'core_no_nan'")

    finite_mask = np.all(np.isfinite(X), axis=0)
    X2 = X[:, finite_mask]
    names2 = [n for n, keep in zip(names, finite_mask) if keep]
    data["feature_matrix"] = X2.astype(np.float32)
    data["feature_names"] = np.asarray(names2, dtype=object)
    return data


# -----------------------------------------------------------------------------
# Scenario construction
# -----------------------------------------------------------------------------


def make_rolling_scenarios(
    dates_int: np.ndarray,
    tradable: np.ndarray,
    *,
    start_idx: int,
    end_idx: int,
    episode_len: int,
    stride: int,
    volume_bbl: float,
) -> List[Dict[str, float]]:
    """Create fixed-length physical trade scenarios using index windows.

    end_idx is exclusive. Each scenario has [i, i + episode_len).
    The environment then starts PnL at the first day after the initial hedge setup,
    so a 30-day scenario produces 29 daily decisions/returns.
    """
    n = len(dates_int)
    start_idx = max(0, int(start_idx))
    end_idx = min(n, int(end_idx))
    episode_len = int(episode_len)
    stride = int(stride)
    if episode_len < 2:
        raise ValueError("episode_len must be at least 2")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    scenarios: List[Dict[str, float]] = []
    last_start = end_idx - episode_len
    for i in range(start_idx, last_start + 1, stride):
        # Require at least some tradable days inside the scenario.
        if np.sum(np.asarray(tradable[i : i + episode_len], dtype=int) != 0) < max(2, episode_len // 2):
            continue
        scenario_id = f"GEN_{int(i)}_{int(i + episode_len)}"
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_record_id": int(len(scenarios)),
                "start_idx": int(i),
                "end_idx": int(i + episode_len),
                "volume_bbl": float(volume_bbl),
            }
        )
    if not scenarios:
        raise ValueError("No scenarios were generated. Check split indices and episode length.")
    return scenarios


# -----------------------------------------------------------------------------
# Scenario file helpers
# -----------------------------------------------------------------------------

def _date_to_precompute_int(x, dates_int: np.ndarray) -> int | None:
    """Convert a scenario date to the same integer date scale used by precompute.

    Existing precompute files in this project use small integers such as 12886,
    which are days since 1970-01-01, not Python ordinals. This helper detects the
    scale from dates_int and converts scenario start/end dates accordingly.
    """
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return None
    d = ts.to_pydatetime().date()
    max_di = int(np.nanmax(np.asarray(dates_int, dtype=np.int64)))

    # Common project format: days since Unix epoch, e.g. 12886.
    if max_di < 100_000:
        return int((pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days)

    # Python ordinal format, e.g. 738000+.
    if max_di < 10_000_000:
        return int(d.toordinal())

    # YYYYMMDD format.
    return int(pd.Timestamp(d).strftime("%Y%m%d"))


def _read_scenario_file(path: Path, exposure_id: str, kind: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "scenario_id" not in df.columns:
        raise KeyError(f"Scenario file {path} has no scenario_id column")
    if "start_date" not in df.columns or "end_date" not in df.columns:
        raise KeyError(f"Scenario file {path} must include start_date and end_date")
    if "volume_bbl" not in df.columns:
        df["volume_bbl"] = 1_000_000.0
    if "exposure_id" not in df.columns:
        df["exposure_id"] = exposure_id
    if "scenario_kind" not in df.columns:
        df["scenario_kind"] = kind

    df["scenario_id"] = df["scenario_id"].astype(str)
    df["scenario_kind"] = df["scenario_kind"].fillna(kind).astype(str)
    df["exposure_id"] = df["exposure_id"].fillna(exposure_id).astype(str)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["volume_bbl"] = pd.to_numeric(df["volume_bbl"], errors="coerce").fillna(1_000_000.0)
    df["scenario_source_file"] = str(path)
    df["scenario_file"] = path.name
    df["oracle_bucket"] = df.apply(lambda r: infer_oracle_bucket(r, default_kind=kind), axis=1)
    if kind == "oracle_all":
        df["scenario_kind_original"] = df["scenario_kind"]
        df["scenario_kind"] = df["oracle_bucket"]
    return df


def load_scenarios_from_root(
    scenario_root: Path,
    *,
    exposure_id: str,
    kinds: Sequence[str],
    dates_int: np.ndarray,
) -> pd.DataFrame:
    asset_dir = Path(scenario_root) / exposure_id
    if not asset_dir.exists():
        raise FileNotFoundError(f"Scenario asset directory not found: {asset_dir}")

    frames: List[pd.DataFrame] = []
    for kind in kinds:
        kind = str(kind).strip()
        if not kind:
            continue
        if kind not in SCENARIO_FILES_BY_KIND:
            raise ValueError(f"Unknown scenario kind {kind!r}. Valid: {sorted(SCENARIO_FILES_BY_KIND)}")
        found = False
        for filename in SCENARIO_FILES_BY_KIND[kind]:
            p = asset_dir / filename
            if p.exists():
                frames.append(_read_scenario_file(p, exposure_id, kind))
                found = True
                break
        if not found:
            print(f"[WARN] No file found for scenario kind {kind!r} under {asset_dir}")

    if not frames:
        raise ValueError(f"No scenario files loaded from {asset_dir}")

    meta = pd.concat(frames, ignore_index=True)
    meta["start_ord"] = meta["start_date"].map(lambda x: _date_to_precompute_int(x, dates_int))
    meta["end_ord"] = meta["end_date"].map(lambda x: _date_to_precompute_int(x, dates_int))
    meta = meta.dropna(subset=["start_ord", "end_ord"]).copy()
    meta["start_ord"] = meta["start_ord"].astype(int)
    meta["end_ord"] = meta["end_ord"].astype(int)

    meta["start_idx"] = np.searchsorted(dates_int, meta["start_ord"].to_numpy(), side="left")
    meta["end_idx"] = np.searchsorted(dates_int, meta["end_ord"].to_numpy(), side="left")
    meta["start_idx"] = meta["start_idx"].clip(0, len(dates_int) - 1).astype(int)
    meta["end_idx"] = meta["end_idx"].clip(1, len(dates_int)).astype(int)
    meta = meta[meta["end_idx"] > meta["start_idx"] + 1].copy()
    if meta.empty:
        print("[DEBUG] Scenario date conversion produced no valid rows.")
        print(f"[DEBUG] dates_int range: {int(np.min(dates_int))} -> {int(np.max(dates_int))}")
        raw_dates = pd.concat(frames, ignore_index=True)[["scenario_id", "start_date", "end_date"]].head(5)
        print(f"[DEBUG] sample scenario dates:\n{raw_dates}")
    meta = meta.drop_duplicates(subset=["scenario_id", "exposure_id", "scenario_kind"], keep="first")
    meta = meta.sort_values(["start_idx", "end_idx", "scenario_id"]).reset_index(drop=True)
    if meta.empty:
        raise ValueError("Scenario metadata became empty after date/index filtering")
    return meta


def scenario_meta_to_env_scenarios(df: pd.DataFrame) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for pos, (_, r) in enumerate(df.iterrows()):
        item: Dict[str, object] = {}
        for col, val in r.items():
            if isinstance(val, pd.Timestamp):
                item[col] = val.isoformat()
            elif isinstance(val, (list, tuple, dict, np.ndarray)):
                item[col] = val
            elif pd.isna(val):
                item[col] = None
            else:
                item[col] = val.item() if hasattr(val, "item") else val
        item["start_idx"] = int(r["start_idx"])
        item["end_idx"] = int(r["end_idx"])
        item["volume_bbl"] = float(r["volume_bbl"])
        item["scenario_id"] = str(r.get("scenario_id", f"GEN_{item['start_idx']}_{item['end_idx']}"))
        item["scenario_kind"] = str(r.get("scenario_kind", "unknown"))
        item["exposure_id"] = str(r.get("exposure_id", "unknown"))
        item["scenario_record_id"] = r.get("scenario_record_id", pos)
        out.append(item)
    return out


def split_scenario_meta_by_percentile(meta: pd.DataFrame, n_dates: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    train_cut, val_cut = default_percentile_splits(n_dates)
    train_meta = meta[meta["start_idx"] < train_cut].copy()
    val_meta = meta[(meta["start_idx"] >= train_cut) & (meta["start_idx"] < val_cut)].copy()
    test_meta = meta[meta["start_idx"] >= val_cut].copy()
    if train_meta.empty or val_meta.empty or test_meta.empty:
        raise ValueError(
            f"Scenario split produced empty partition: train={len(train_meta)}, val={len(val_meta)}, test={len(test_meta)}"
        )
    return train_meta.reset_index(drop=True), val_meta.reset_index(drop=True), test_meta.reset_index(drop=True), train_cut, val_cut


# -------------------------------------------------------------------------
# Rolling windows helpers
# -------------------------------------------------------------------------

def _filter_meta_by_start_idx(meta: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Select scenarios whose start date falls inside [start_idx, end_idx)."""
    return meta[(meta["start_idx"] >= int(start_idx)) & (meta["start_idx"] < int(end_idx))].copy().reset_index(drop=True)


def _safe_filename(x: object, max_len: int = 140) -> str:
    s = str(x)
    s = re.sub(r"[^A-Za-z0-9._=-]+", "_", s).strip("_")
    if not s:
        s = "unknown"
    return s[:max_len]


def _clean_token(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def infer_oracle_bucket(row, default_kind: str = "unknown") -> str:
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    fields = [
        get("scenario_kind", ""),
        get("tag", ""),
        get("label", ""),
        get("oracle_series", ""),
        get("oracle_candidate", ""),
        get("oracle_pool", ""),
        get("scenario_file", ""),
        get("scenario_source_file", ""),
    ]
    text = "__".join(_clean_token(x) for x in fields if x is not None)

    is_oracle_all = "oracle_all" in text or str(default_kind) == "oracle_all"
    if not is_oracle_all:
        return str(get("scenario_kind", default_kind) or default_kind)

    level = None
    direction = None

    if "feasible" in text:
        level = "feasible"
    elif "extreme" in text or "extereme" in text:
        level = "extreme"

    if "best" in text:
        direction = "best"
    elif "worst" in text:
        direction = "worst"

    if level and direction:
        return f"oracle_all_{level}_{direction}"
    if level:
        return f"oracle_all_{level}_unknown"
    if direction:
        return f"oracle_all_unknown_{direction}"
    return "oracle_all_unknown"


def _date_int_series_to_datetime(values):
    vals = pd.Series(values)
    vals_num = pd.to_numeric(vals, errors="coerce")
    max_v = vals_num.max(skipna=True)
    if pd.isna(max_v):
        return pd.to_datetime(vals, errors="coerce")
    if max_v < 100_000:
        return pd.to_datetime(vals_num, unit="D", origin="1970-01-01", errors="coerce")
    if max_v < 10_000_000:
        return vals_num.map(lambda x: pd.Timestamp.fromordinal(int(x)) if pd.notna(x) else pd.NaT)
    return pd.to_datetime(vals_num.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def episode_summary_to_report_rows(summary: pd.DataFrame, *, asset: str, window: int) -> pd.DataFrame:
    """Convert RL episode summary to a finalreport-compatible result table."""
    if summary.empty:
        return pd.DataFrame()
    exposure_fallback = ASSET_TO_EXPOSURE_ID.get(asset.upper(), asset.upper())
    out = pd.DataFrame({
        "scenario_id": summary.get("scenario_id", pd.Series([None] * len(summary))).astype(str),
        "exposure_id": summary.get("exposure_id", exposure_fallback),
        "scenario_kind": summary.get("scenario_kind", "unknown"),
        "strategy": "RL_SAC_LPM",
        "asset": asset.upper(),
        "window": int(window),
        "mode": "dynamic",
        "roll": "roll",
        "dynamic": 1,
        "start_date": pd.to_datetime(summary.get("start_date", pd.NaT), errors="coerce"),
        "end_date": pd.to_datetime(summary.get("end_date", pd.NaT), errors="coerce"),
        "volume_bbl": pd.to_numeric(summary.get("volume_bbl", np.nan), errors="coerce"),
        "spot_pnl_total": pd.to_numeric(summary.get("physical_pnl", np.nan), errors="coerce"),
        "fut_pnl_total": pd.to_numeric(summary.get("futures_pnl", np.nan), errors="coerce"),
        "cost_trade_total": pd.to_numeric(summary.get("decision_cost", np.nan), errors="coerce"),
        "cost_roll_total": pd.to_numeric(summary.get("roll_accounting_cost", np.nan), errors="coerce"),
        "net_pnl_total": pd.to_numeric(summary.get("total_pnl", np.nan), errors="coerce"),
        "turnover_h": pd.to_numeric(summary.get("turnover_h", np.nan), errors="coerce"),
        "turnover_contracts": pd.to_numeric(summary.get("turnover_contracts", np.nan), errors="coerce"),
        "trade_contracts": pd.to_numeric(summary.get("turnover_contracts", np.nan), errors="coerce"),
        "roll_contracts": np.nan,
        "max_abs_contracts": np.nan,
        "mdd_equity": pd.to_numeric(summary.get("mdd", np.nan), errors="coerce"),
        "mean_h": pd.to_numeric(summary.get("mean_h", np.nan), errors="coerce"),
        "min_h": pd.to_numeric(summary.get("min_h", np.nan), errors="coerce"),
        "max_h": pd.to_numeric(summary.get("max_h", np.nan), errors="coerce"),
        "no_hedge_pnl": pd.to_numeric(summary.get("no_hedge_pnl", np.nan), errors="coerce"),
        "naive_pnl": pd.to_numeric(summary.get("naive_pnl", np.nan), errors="coerce"),
        "steps": pd.to_numeric(summary.get("steps", np.nan), errors="coerce"),
    })

    for col in SCENARIO_META_COLS_FOR_REPORT:
        if col in summary.columns and col not in out.columns:
            out[col] = summary[col].values
    return out


def plot_fraction_of_test_episodes(
    test_log: pd.DataFrame,
    *,
    out_dir: Path,
    asset: str,
    window: int,
    fraction: float,
    max_plots: int,
) -> List[str]:
    """Plot approximately a fraction of test episodes into organized folders."""
    if test_log.empty or fraction <= 0:
        return []
    orders = sorted(test_log["eval_scenario_order"].dropna().astype(int).unique().tolist())
    if not orders:
        return []
    n_plots = int(math.ceil(len(orders) * float(fraction)))
    n_plots = max(1, min(len(orders), n_plots))
    if max_plots and max_plots > 0:
        n_plots = min(n_plots, int(max_plots))

    # Evenly spread plots across the test set instead of taking only the first rows.
    if n_plots >= len(orders):
        selected = orders
    else:
        selected = [orders[int(i)] for i in np.linspace(0, len(orders) - 1, n_plots)]
        selected = sorted(set(selected))

    made: List[str] = []
    plot_start_time = time.time()
    for order in selected:
        df_ep = test_log[test_log["eval_scenario_order"] == order]
        if df_ep.empty:
            continue
        first = df_ep.iloc[0]
        kind = _safe_filename(first.get("scenario_kind", "unknown"))
        sid = _safe_filename(first.get("scenario_id", f"scenario_{order}"))
        exposure_id = _safe_filename(first.get("exposure_id", ASSET_TO_EXPOSURE_ID.get(asset.upper(), asset.upper())))
        folder = out_dir / "charts" / f"window_{int(window):02d}" / kind
        filename = f"{asset.upper()}__{exposure_id}__w{int(window):02d}__{kind}__{sid}.png"
        path = folder / filename
        plot_sample_episode(test_log, path, scenario_order=int(order))
        if path.exists():
            made.append(str(path))
        if len(made) == 1 or len(made) % 25 == 0 or len(made) == len(selected):
            _progress_line(
                f"plot w{int(window):02d}",
                len(made),
                len(selected),
                plot_start_time,
                extra=f"latest={sid}",
            )
    return made



def _fmt_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.1f}h"


def _progress_line(prefix: str, current: int, total: int, start_time: float, extra: str = "") -> None:
    total = max(1, int(total))
    current = min(int(current), total)
    elapsed = time.time() - float(start_time)
    pct = 100.0 * current / total
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else 0.0
    msg = (
        f"[{prefix}] {current:,}/{total:,} ({pct:5.1f}%) | "
        f"elapsed={_fmt_seconds(elapsed)} | eta={_fmt_seconds(remaining)}"
    )
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)




def configure_compute_threads(torch_threads: int, blas_threads: int) -> None:
    torch_threads = max(1, int(torch_threads))
    blas_threads = max(1, int(blas_threads))

    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)

    if torch is not None:
        try:
            torch.set_num_threads(torch_threads)
            torch.set_num_interop_threads(max(1, min(torch_threads, 4)))
        except Exception as exc:
            print(f"[WARN] could not set torch thread counts: {exc}", flush=True)

    print(f"[compute] torch_threads={torch_threads} | blas_threads={blas_threads}", flush=True)


def reset_sac_replay_buffer(model) -> None:
    try:
        if getattr(model, "replay_buffer", None) is not None:
            model.replay_buffer.reset()
            print("[warm-start] replay buffer reset", flush=True)
    except Exception as exc:
        print(f"[WARN] could not reset replay buffer: {exc}", flush=True)


def split_indices_by_date(dates_int: np.ndarray, train_end: int, val_end: int) -> Tuple[int, int]:
    """Return index cutoffs for train/val/test based on integer dates.

    dates_int in these precomputes are Python ordinal dates, not YYYYMMDD.
    The command-line dates are also expected as ordinal integers for now.
    For this first script, defaults are percentile-based unless user passes cutoffs.
    """
    train_cut = int(np.searchsorted(dates_int, int(train_end), side="left"))
    val_cut = int(np.searchsorted(dates_int, int(val_end), side="left"))
    train_cut = int(np.clip(train_cut, 1, len(dates_int) - 2))
    val_cut = int(np.clip(val_cut, train_cut + 1, len(dates_int) - 1))
    return train_cut, val_cut


def default_percentile_splits(n: int) -> Tuple[int, int]:
    train_cut = int(n * 0.70)
    val_cut = int(n * 0.85)
    return train_cut, val_cut


# -----------------------------------------------------------------------------
# Evaluation and metrics
# -----------------------------------------------------------------------------


def run_policy_on_scenarios(
    model: SAC,
    pre_dict: Dict[str, np.ndarray],
    scenarios: Sequence[Dict[str, float]],
    cfg: SACPortfolioLPMConfig,
    *,
    deterministic: bool = True,
    max_episodes: int | None = None,
    progress_label: str | None = None,
    progress_every: int = 100,
) -> pd.DataFrame:
    eval_cfg = SACPortfolioLPMConfig(**asdict(cfg))
    eval_cfg.info_mode = "eval"
    env = SACPortfolioLPMEnv(pre_dict, scenarios, cfg=eval_cfg, seed=123)

    rows: List[Dict[str, float]] = []
    n_eval = len(scenarios) if max_episodes is None else min(len(scenarios), int(max_episodes))
    eval_start_time = time.time()
    if progress_label:
        _progress_line(progress_label, 0, n_eval, eval_start_time, extra="starting evaluation")
    for scenario_idx in range(n_eval):
        scenario_meta = dict(scenarios[scenario_idx])
        scenario_id = str(
            scenario_meta.get(
                "scenario_id",
                f"GEN_{int(scenario_meta.get('start_idx', -1))}_{int(scenario_meta.get('end_idx', -1))}",
            )
        )
        scenario_kind = str(scenario_meta.get("scenario_kind", "generated"))
        exposure_id = str(scenario_meta.get("exposure_id", "unknown"))
        scenario_record_id = scenario_meta.get("scenario_record_id", scenario_idx)

        report_meta = {
            col: scenario_meta.get(col)
            for col in SCENARIO_META_COLS_FOR_REPORT
            if col in scenario_meta
        }

        obs, reset_info = env.reset(options={"scenario_idx": scenario_idx})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            info = dict(info)
            info.update(report_meta)
            info["scenario_id"] = scenario_id
            info["scenario_kind"] = scenario_kind
            info["exposure_id"] = exposure_id
            info["scenario_record_id"] = int(scenario_record_id) if str(scenario_record_id).isdigit() else scenario_record_id
            info["eval_scenario_order"] = int(scenario_idx)
            info["reset_start_idx"] = int(reset_info["start_idx"])
            info["reset_end_idx"] = int(reset_info["end_idx"])
            rows.append(info)
            done = bool(terminated or truncated)
        if progress_label and (
            (scenario_idx + 1) % max(1, int(progress_every)) == 0 or (scenario_idx + 1) == n_eval
        ):
            _progress_line(
                progress_label,
                scenario_idx + 1,
                n_eval,
                eval_start_time,
                extra=f"latest_scenario={scenario_id}",
            )
    return pd.DataFrame(rows)


def compute_episode_summary(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return pd.DataFrame()

    if "scenario_id" not in log_df.columns:
        log_df = log_df.copy()
        log_df["scenario_id"] = (
            "GEN_"
            + log_df["reset_start_idx"].astype(str)
            + "_"
            + log_df["reset_end_idx"].astype(str)
        )
    if "scenario_kind" not in log_df.columns:
        log_df = log_df.copy()
        log_df["scenario_kind"] = "generated"
    if "exposure_id" not in log_df.columns:
        log_df = log_df.copy()
        log_df["exposure_id"] = "unknown"

    group_cols = [
        "scenario_id",
        "scenario_kind",
        "exposure_id",
        "scenario_idx",
        "eval_scenario_order",
        "reset_start_idx",
        "reset_end_idx",
    ]

    def _sum(col: str):
        return (col, "sum")

    meta_aggs = {
        col: (col, "first")
        for col in SCENARIO_META_COLS_FOR_REPORT
        if col in log_df.columns and col not in group_cols and col != "scenario_record_id"
    }
    if "scenario_record_id" in log_df.columns:
        meta_aggs["scenario_record_id"] = ("scenario_record_id", "first")

    summary = log_df.groupby(group_cols, dropna=False).agg(
        steps=("date_int", "count"),
        total_pnl=("portfolio_pnl_accounting", "sum"),
        total_reward=("reward", "sum"),
        physical_pnl=("pnl_phys", "sum"),
        futures_pnl=("pnl_fut", "sum"),
        decision_cost=("decision_cost", "sum"),
        roll_accounting_cost=("roll_accounting_cost", "sum"),
        mean_h=("h_after", "mean"),
        max_h=("h_after", "max"),
        min_h=("h_after", "min"),
        turnover_h=("action_delta_h_effective", lambda x: float(np.sum(np.abs(x)))),
        turnover_contracts=("delta_n", lambda x: float(np.sum(np.abs(x)))),
        roll_days=("roll_flag", "sum"),
        final_equity=("equity", "last"),
        mdd=("mdd", "min"),
        **meta_aggs,
    ).reset_index()

    no_hedge_sum = (
        log_df.groupby(group_cols, dropna=False)["pnl_phys"]
        .sum()
        .reset_index(name="no_hedge_pnl")
    )
    summary = summary.merge(no_hedge_sum, on=group_cols, how="left")

    # Exact naive h=1 benchmark: fixed short CL position from physical volume.
    # This must not depend on SAC's own position, otherwise it becomes biased
    # when the SAC policy closes the futures hedge.
    naive_rows = log_df.copy()
    naive_rows["n_naive"] = -np.rint(naive_rows["Q"] / 1000.0).astype(int)

    if "pnl_1c" not in naive_rows.columns:
        raise KeyError("Daily eval log must include pnl_1c to compute naive benchmark exactly.")

    naive_rows["naive_fut_pnl"] = naive_rows["n_naive"] * naive_rows["pnl_1c"]
    naive_rows["naive_pnl"] = naive_rows["pnl_phys"] + naive_rows["naive_fut_pnl"]

    naive_sum = naive_rows.groupby(group_cols, dropna=False).agg(
        naive_pnl=("naive_pnl", "sum"),
        naive_fut_pnl=("naive_fut_pnl", "sum"),
        n_naive=("n_naive", "first"),
    ).reset_index()

    summary = summary.merge(naive_sum, on=group_cols, how="left")
    return summary


def print_metric_block(summary: pd.DataFrame, label: str) -> None:
    if summary.empty:
        print(f"[{label}] no evaluation rows")
        return
    print(f"\n[{label}] episode metrics")
    print(f"episodes:              {len(summary)}")
    print(f"mean total pnl:        {summary['total_pnl'].mean():,.2f}")
    print(f"median total pnl:      {summary['total_pnl'].median():,.2f}")
    print(f"mean no-hedge pnl:     {summary['no_hedge_pnl'].mean():,.2f}")
    print(f"mean naive pnl:        {summary['naive_pnl'].mean():,.2f}")
    print(f"mean decision cost:    {summary['decision_cost'].mean():,.2f}")
    print(f"mean roll acct cost:   {summary['roll_accounting_cost'].mean():,.2f}")
    print(f"mean turnover h:       {summary['turnover_h'].mean():,.4f}")
    print(f"mean MDD:              {summary['mdd'].mean():,.2f}")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_sample_episode(log_df: pd.DataFrame, out_path: Path, scenario_order: int = 0) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.ticker import FuncFormatter
    except Exception:
        print("matplotlib/seaborn not available; skipping plots")
        return

    df = log_df[log_df["eval_scenario_order"] == scenario_order].copy()
    if df.empty:
        return

    df = df.sort_values("day_in_episode").reset_index(drop=True)
    df["plot_date"] = _date_int_series_to_datetime(df["date_int"])

    if df["plot_date"].isna().all():
        df["plot_date"] = pd.to_datetime(df["day_in_episode"], unit="D", origin="2000-01-01")

    sns.set_theme(style="whitegrid", context="notebook")

    def money_fmt(v, _pos):
        av = abs(v)
        if av >= 1e9:
            return f"{v/1e9:.1f}B"
        if av >= 1e6:
            return f"{v/1e6:.1f}M"
        if av >= 1e3:
            return f"{v/1e3:.0f}K"
        return f"{v:.0f}"

    fig, axes = plt.subplots(5, 1, figsize=(15, 16), sharex=True)

    title_bits = []
    for col in ["scenario_id", "scenario_kind", "oracle_bucket"]:
        if col in df.columns and pd.notna(df[col].iloc[0]):
            title_bits.append(f"{col}={df[col].iloc[0]}")
    if title_bits:
        fig.suptitle(" | ".join(title_bits), fontsize=11, y=0.995)

    # 1) Prices
    sns.lineplot(ax=axes[0], data=df, x="plot_date", y="spot", label="Spot", linewidth=2)
    if "f_mark" in df.columns:
        sns.lineplot(ax=axes[0], data=df, x="plot_date", y="f_mark", label="Futures mark", linewidth=2)

    for rdate in df.loc[df["roll_flag"] != 0, "plot_date"]:
        axes[0].axvline(rdate, linestyle="--", alpha=0.45)

    axes[0].set_title("Spot/Futures prices and roll dates")
    axes[0].set_ylabel("USD/bbl")
    axes[0].legend(loc="best")

    # 2) Hedge ratio only
    sns.lineplot(ax=axes[1], data=df, x="plot_date", y="h_after", label="Hedge ratio h", linewidth=2.2)
    axes[1].scatter(df["plot_date"], df["h_after"], s=22)
    axes[1].axhline(0, linewidth=1, alpha=0.35)
    axes[1].axhline(1, linewidth=1, alpha=0.35)
    axes[1].set_title("Hedge ratio decision path")
    axes[1].set_ylabel("h")
    axes[1].legend(loc="best")

    # 3) Contracts separate
    if "n_after" in df.columns:
        sns.lineplot(ax=axes[2], data=df, x="plot_date", y="n_after", label="Contracts after decision", linewidth=2)
    axes[2].bar(df["plot_date"], df["delta_n"], alpha=0.35, label="Daily Δ contracts", width=0.8)
    axes[2].axhline(0, linewidth=1, alpha=0.35)
    axes[2].set_title("CL contract position and daily changes")
    axes[2].set_ylabel("contracts")
    axes[2].legend(loc="best")

    # 4) Cumulative PnL
    df["cum_no_hedge"] = df["pnl_phys"].cumsum()
    df["cum_sac"] = df["portfolio_pnl_accounting"].cumsum()

    pnl_plot = df[["plot_date", "cum_no_hedge", "cum_sac"]].rename(
        columns={"cum_no_hedge": "No hedge", "cum_sac": "SAC"}
    )

    if "pnl_1c" in df.columns:
        n_naive = -int(np.rint(float(df["Q"].iloc[0]) / 1000.0))
        df["cum_naive"] = (df["pnl_phys"] + n_naive * df["pnl_1c"]).cumsum()
        pnl_plot["Naive h=1"] = df["cum_naive"]

    pnl_long = pnl_plot.melt(id_vars="plot_date", var_name="series", value_name="cum_pnl")
    sns.lineplot(ax=axes[3], data=pnl_long, x="plot_date", y="cum_pnl", hue="series", linewidth=2)
    axes[3].set_title("Cumulative PnL comparison")
    axes[3].set_ylabel("USD")
    axes[3].yaxis.set_major_formatter(FuncFormatter(money_fmt))
    axes[3].legend(loc="best")

    # 5) Equity/drawdown/cost
    sns.lineplot(ax=axes[4], data=df, x="plot_date", y="equity", label="Equity", linewidth=2)
    if "drawdown" in df.columns:
        sns.lineplot(ax=axes[4], data=df, x="plot_date", y="drawdown", label="Drawdown", linewidth=1.8)

    ax_cost = axes[4].twinx()
    ax_cost.bar(df["plot_date"], df["decision_cost"], alpha=0.25, label="Decision cost", width=0.8)

    axes[4].set_title("Equity, drawdown and decision cost")
    axes[4].set_ylabel("USD")
    axes[4].yaxis.set_major_formatter(FuncFormatter(money_fmt))
    ax_cost.set_ylabel("cost USD")
    ax_cost.yaxis.set_major_formatter(FuncFormatter(money_fmt))

    lines1, labels1 = axes[4].get_legend_handles_labels()
    lines2, labels2 = ax_cost.get_legend_handles_labels()
    axes[4].legend(lines1 + lines2, labels1 + labels2, loc="best")

    # X-axis: show day number and date clearly
    n_ticks = min(len(df), 12)
    tick_idx = np.linspace(0, len(df) - 1, n_ticks).round().astype(int)
    tick_dates = df.loc[tick_idx, "plot_date"]
    tick_labels = [
        f"D{int(df.loc[i, 'day_in_episode'])}\n{df.loc[i, 'plot_date'].strftime('%Y-%m-%d')}"
        for i in tick_idx
    ]

    axes[-1].set_xticks(tick_dates)
    axes[-1].set_xticklabels(tick_labels, rotation=0, ha="center")
    axes[-1].set_xlabel("Day in episode / date")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main training
# -----------------------------------------------------------------------------


def build_env(pre_dict: Dict[str, np.ndarray], scenarios: Sequence[Dict[str, float]], cfg: SACPortfolioLPMConfig, seed: int):
    def _make():
        env = SACPortfolioLPMEnv(pre_dict, scenarios, cfg=cfg, seed=seed)
        return Monitor(env)

    return DummyVecEnv([_make])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC with portfolio-LPM reward for crude-oil hedging.")
    p.add_argument("--asset", choices=sorted(ASSET_TO_FILE), default="WTI")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--episode-len", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--volume-bbl", type=float, default=1_000_000.0)
    p.add_argument("--feature-mode", choices=["all", "core_no_nan"], default="core_no_nan")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-episodes", type=int, default=20)

    # Reward / env parameters
    p.add_argument("--h-min", type=float, default=0.0)
    p.add_argument("--h-max", type=float, default=1.5)
    p.add_argument("--delta-h", type=float, default=0.20)
    p.add_argument("--lambda-lpm", type=float, default=1.0)
    p.add_argument("--eta-cost", type=float, default=1.0)
    p.add_argument("--lambda-smooth", type=float, default=0.01)

    p.add_argument(
        "--reward-cost-mode",
        choices=["legacy", "enhanced_cost"],
        default="legacy",
        help="legacy keeps the old reward; enhanced_cost adds explicit turnover/roll/position cost penalties.",
    )
    p.add_argument("--lambda-turnover-h", type=float, default=0.0)
    p.add_argument("--lambda-contract-turnover", type=float, default=0.0)
    p.add_argument("--lambda-roll-cost", type=float, default=0.0)
    p.add_argument("--lambda-position-size", type=float, default=0.0)

    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--reward-clip", type=float, default=50.0)

    # SAC hyperparameters
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--net-arch", type=str, default="128,128")

    p.add_argument(
        "--scenario-root",
        type=Path,
        default=SCENARIO_ROOT,
        help="Scenario root folder. Default uses project scenarios/.",
    )
    p.add_argument(
        "--scenario-kinds",
        type=str,
        default="oracle_universe,oracle_all,baseline",
        help="Comma-separated scenario kinds to load from scenarios/<EXPOSURE_ID>/.",
    )
    p.add_argument(
        "--generated-scenarios",
        action="store_true",
        help="Ignore scenario-root and use generated rolling scenarios instead.",
    )
    p.add_argument(
        "--rolling-windows",
        action="store_true",
        help="Run fixed rolling windows instead of one global train/val/test split.",
    )
    p.add_argument("--train-scenario-kinds", type=str, default="oracle_universe")
    p.add_argument("--eval-scenario-kinds", type=str, default="oracle_universe,oracle_all,baseline")
    p.add_argument("--train-years", type=float, default=2.0)
    p.add_argument("--val-months", type=float, default=6.0)
    p.add_argument("--test-months", type=float, default=6.0)
    p.add_argument("--step-months", type=float, default=6.0)
    p.add_argument("--max-windows", type=int, default=0, help="0 means run all possible windows.")
    p.add_argument("--plot-fraction", type=float, default=0.10)
    p.add_argument("--max-plots-per-window", type=int, default=0, help="0 means no cap.")
    p.add_argument("--device", type=str, default="auto", help="SB3 device: auto, cpu, cuda, or mps if supported.")
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--blas-threads", type=int, default=8)
    p.add_argument(
        "--warm-start",
        action="store_true",
        help="Use previous walk-forward window's trained SAC weights to initialize the next window.",
    )
    p.add_argument(
        "--keep-replay-buffer-on-warm-start",
        action="store_true",
        help="Keep old replay-buffer transitions when warm-starting. Default is to clear it.",
    )
    p.add_argument(
        "--parallel-windows",
        type=int,
        default=1,
        help="Number of walk-forward windows to train in parallel. Only works when --warm-start is not used.",
    )
    return p.parse_args()


def _train_eval_one_window_worker(task: Dict[str, object]) -> Dict[str, object]:
    """Train/evaluate one rolling window in a separate process."""
    worker_start = time.time()
    w = int(task["window"])
    asset = str(task["asset"]).upper()
    seed = int(task["seed"])
    out_dir = Path(str(task["out_dir"]))
    wdir = out_dir / f"window_{w:02d}"
    wdir.mkdir(parents=True, exist_ok=True)

    try:
        configure_compute_threads(int(task["torch_threads"]), int(task["blas_threads"]))
        raw_pre, _pre_path = load_precompute(asset)
        pre = filter_feature_matrix(raw_pre, str(task["feature_mode"]))

        cfg = SACPortfolioLPMConfig(
            initial_h=1.0,
            h_min=float(task["h_min"]),
            h_max=float(task["h_max"]),
            delta_h_bounds=(-float(task["delta_h"]), float(task["delta_h"])),
            lambda_lpm=float(task["lambda_lpm"]),
            eta_decision_cost=float(task["eta_cost"]),
            lambda_smooth=float(task["lambda_smooth"]),
            reward_cost_mode=str(task["reward_cost_mode"]),
            lambda_turnover_h=float(task["lambda_turnover_h"]),
            lambda_contract_turnover=float(task["lambda_contract_turnover"]),
            lambda_roll_cost=float(task["lambda_roll_cost"]),
            lambda_position_size=float(task["lambda_position_size"]),
            reward_scale=float(task["reward_scale"]),
            reward_clip=float(task["reward_clip"]),
            info_mode="train",
        )

        train_meta = pd.DataFrame(task["train_meta"])
        val_meta = pd.DataFrame(task["val_meta"])
        test_meta = pd.DataFrame(task["test_meta"])

        train_scenarios = scenario_meta_to_env_scenarios(train_meta)
        val_scenarios = scenario_meta_to_env_scenarios(val_meta)
        test_scenarios = scenario_meta_to_env_scenarios(test_meta)

        print(
            f"[parallel window {w:02d}] start | train={len(train_scenarios):,} "
            f"val={len(val_scenarios):,} test={len(test_scenarios):,}",
            flush=True,
        )

        env = build_env(pre, train_scenarios, cfg, seed=seed)
        net_arch = [int(x.strip()) for x in str(task["net_arch"]).split(",") if x.strip()]
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
            verbose=1,
            seed=seed,
            tensorboard_log=str(wdir / "tb"),
            device=str(task["device"]),
        )
        model.learn(total_timesteps=int(task["timesteps"]), progress_bar=True)
        model.save(wdir / "model.zip")

        val_log = run_policy_on_scenarios(
            model,
            pre,
            val_scenarios,
            cfg,
            deterministic=True,
            max_episodes=None,
            progress_label=f"eval val w{w:02d}",
            progress_every=max(1, len(val_scenarios) // 10),
        )
        test_log = run_policy_on_scenarios(
            model,
            pre,
            test_scenarios,
            cfg,
            deterministic=True,
            max_episodes=None,
            progress_label=f"eval test w{w:02d}",
            progress_every=max(1, len(test_scenarios) // 20),
        )
        val_log["window"] = int(w)
        test_log["window"] = int(w)

        val_summary = compute_episode_summary(val_log)
        test_summary = compute_episode_summary(test_log)
        val_summary["window"] = int(w)
        test_summary["window"] = int(w)
        report_rows = episode_summary_to_report_rows(test_summary, asset=asset, window=w)

        val_log.to_csv(wdir / "val_daily_log.csv", index=False)
        test_log.to_csv(wdir / "test_daily_log.csv", index=False)
        val_summary.to_csv(wdir / "val_episode_summary.csv", index=False)
        test_summary.to_csv(wdir / "test_episode_summary.csv", index=False)
        report_rows.to_csv(wdir / "results.csv", index=False)

        try:
            val_log.to_parquet(wdir / "val_daily_log.parquet", index=False)
            test_log.to_parquet(wdir / "test_daily_log.parquet", index=False)
            val_summary.to_parquet(wdir / "val_episode_summary.parquet", index=False)
            test_summary.to_parquet(wdir / "test_episode_summary.parquet", index=False)
            report_rows.to_parquet(wdir / "results.parquet", index=False)
        except Exception as exc:
            print(f"[WARN] parquet write failed in parallel window {w}: {exc}", flush=True)

        made_charts = plot_fraction_of_test_episodes(
            test_log,
            out_dir=out_dir,
            asset=asset,
            window=w,
            fraction=float(task["plot_fraction"]),
            max_plots=int(task["max_plots_per_window"]),
        )

        result = {
            "window": int(w),
            "status": "ok",
            "window_dir": str(wdir),
            "train_n": int(len(train_scenarios)),
            "val_n": int(len(val_scenarios)),
            "test_n": int(len(test_scenarios)),
            "charts_n": int(len(made_charts)),
            "duration_sec": float(time.time() - worker_start),
        }
        with open(wdir / "worker_status.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[parallel window {w:02d}] done | elapsed={_fmt_seconds(result['duration_sec'])}", flush=True)
        return result

    except Exception as exc:
        import traceback

        result = {
            "window": int(w),
            "status": "error",
            "window_dir": str(wdir),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "duration_sec": float(time.time() - worker_start),
        }
        with open(wdir / "worker_error.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[parallel window {w:02d}] ERROR: {exc}", flush=True)
        return result


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    configure_compute_threads(args.torch_threads, args.blas_threads)

    raw_pre, pre_path = load_precompute(args.asset)
    pre = filter_feature_matrix(raw_pre, args.feature_mode)
    dates_int = np.asarray(pre["dates_int"], dtype=np.int64)
    tradable = np.asarray(pre["tradable"], dtype=np.int8)
    n = len(dates_int)
    train_cut, val_cut = default_percentile_splits(n)

    if not args.generated_scenarios:
        exposure_id = ASSET_TO_EXPOSURE_ID[args.asset.upper()]
        kinds = [x.strip() for x in str(args.scenario_kinds).split(",") if x.strip()]
        scenario_meta = load_scenarios_from_root(
            Path(args.scenario_root),
            exposure_id=exposure_id,
            kinds=kinds,
            dates_int=dates_int,
        )
        train_meta, val_meta, test_meta, train_cut, val_cut = split_scenario_meta_by_percentile(scenario_meta, n)
        train_scenarios = scenario_meta_to_env_scenarios(train_meta)
        val_scenarios = scenario_meta_to_env_scenarios(val_meta)
        test_scenarios = scenario_meta_to_env_scenarios(test_meta)
        scenario_mode = "scenario_root"
        print(f"Loaded scenario kinds: {kinds}")
        print(f"Loaded scenario_id examples: {scenario_meta['scenario_id'].head(5).tolist()}")
    else:
        train_scenarios = make_rolling_scenarios(
            dates_int,
            tradable,
            start_idx=0,
            end_idx=train_cut,
            episode_len=args.episode_len,
            stride=args.stride,
            volume_bbl=args.volume_bbl,
        )
        val_scenarios = make_rolling_scenarios(
            dates_int,
            tradable,
            start_idx=max(0, train_cut - args.episode_len),
            end_idx=val_cut,
            episode_len=args.episode_len,
            stride=args.stride,
            volume_bbl=args.volume_bbl,
        )
        test_scenarios = make_rolling_scenarios(
            dates_int,
            tradable,
            start_idx=max(0, val_cut - args.episode_len),
            end_idx=n,
            episode_len=args.episode_len,
            stride=args.stride,
            volume_bbl=args.volume_bbl,
        )
        scenario_mode = "generated_rolling"

    cfg = SACPortfolioLPMConfig(
        initial_h=1.0,
        h_min=args.h_min,
        h_max=args.h_max,
        delta_h_bounds=(-float(args.delta_h), float(args.delta_h)),
        lambda_lpm=args.lambda_lpm,
        eta_decision_cost=args.eta_cost,
        lambda_smooth=args.lambda_smooth,
        reward_cost_mode=args.reward_cost_mode,
        lambda_turnover_h=args.lambda_turnover_h,
        lambda_contract_turnover=args.lambda_contract_turnover,
        lambda_roll_cost=args.lambda_roll_cost,
        lambda_position_size=args.lambda_position_size,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        info_mode="train",
    )

    if args.rolling_windows:
        if args.generated_scenarios:
            raise ValueError("--rolling-windows currently requires real scenario files; do not use --generated-scenarios.")

        asset = args.asset.upper()
        exposure_id = ASSET_TO_EXPOSURE_ID[asset]
        train_kinds = [x.strip() for x in str(args.train_scenario_kinds).split(",") if x.strip()]
        eval_kinds = [x.strip() for x in str(args.eval_scenario_kinds).split(",") if x.strip()]

        train_meta_all = load_scenarios_from_root(
            Path(args.scenario_root), exposure_id=exposure_id, kinds=train_kinds, dates_int=dates_int
        )
        eval_meta_all = load_scenarios_from_root(
            Path(args.scenario_root), exposure_id=exposure_id, kinds=eval_kinds, dates_int=dates_int
        )

        train_days = max(30, int(round(float(args.train_years) * 365.25)))
        val_days = max(20, int(round(float(args.val_months) * 30.4375)))
        test_days = max(20, int(round(float(args.test_months) * 30.4375)))
        step_days = max(1, int(round(float(args.step_months) * 30.4375)))

        min_start = int(max(0, min(train_meta_all["start_idx"].min(), eval_meta_all["start_idx"].min())))
        max_end = int(min(n, max(train_meta_all["start_idx"].max(), eval_meta_all["start_idx"].max()) + test_days + val_days + 1))

        hmin_tag = _safe_filename(f"hmin{args.h_min}")
        hmax_tag = _safe_filename(f"hmax{args.h_max}")
        dh_tag = _safe_filename(f"dh{args.delta_h}")
        run_name = (
            f"{asset.lower()}_sac_lpm_ROLL_T{args.timesteps}_"
            f"train{args.train_years}y_val{args.val_months}m_test{args.test_months}m_"
            f"{hmin_tag}_{hmax_tag}_{dh_tag}"
        )
        out_dir = OUTPUT_DIR / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 90)
        print("SAC Portfolio-LPM rolling-window training")
        print(f"asset/exposure:       {asset} / {exposure_id}")
        print(f"precompute:           {pre_path}")
        print(f"features:             {np.asarray(pre['feature_matrix']).shape[1]}")
        print(f"train scenario kinds: {train_kinds}")
        print(f"eval scenario kinds:  {eval_kinds}")
        print(f"train/val/test days:  {train_days} / {val_days} / {test_days}")
        print(f"step days:            {step_days}")
        print(f"output:               {out_dir}")
        print("=" * 90)

        all_val_logs: List[pd.DataFrame] = []
        all_test_logs: List[pd.DataFrame] = []
        all_val_summaries: List[pd.DataFrame] = []
        all_test_summaries: List[pd.DataFrame] = []
        all_report_rows: List[pd.DataFrame] = []
        window_rows: List[Dict[str, object]] = []
        window_candidate_rows: List[Dict[str, object]] = []
        warm_model = None

        if int(args.parallel_windows) > 1:
            if args.warm_start:
                raise ValueError("--parallel-windows > 1 cannot be used with --warm-start. Disable warm-start or set --parallel-windows 1.")

            print(
                f"[parallel] enabled | workers={int(args.parallel_windows)} | "
                f"torch_threads/process={int(args.torch_threads)} | blas_threads/process={int(args.blas_threads)}",
                flush=True,
            )

            tasks: List[Dict[str, object]] = []
            candidate_idx = 0
            train_cursor = min_start
            accepted_w = 0
            while True:
                train_end = train_cursor + train_days
                val_start = train_end
                val_end = val_start + val_days
                test_start = val_end
                test_end = test_start + test_days
                if test_start >= max_end or test_start >= n:
                    break
                test_end = min(test_end, n)

                train_meta = _filter_meta_by_start_idx(train_meta_all, train_cursor, train_end)
                val_meta = _filter_meta_by_start_idx(train_meta_all, val_start, val_end)
                test_meta = _filter_meta_by_start_idx(eval_meta_all, test_start, test_end)

                candidate_row = {
                    "candidate_index": int(candidate_idx),
                    "accepted_window": None,
                    "train_start_idx": int(train_cursor),
                    "train_end_idx": int(train_end),
                    "val_start_idx": int(val_start),
                    "val_end_idx": int(val_end),
                    "test_start_idx": int(test_start),
                    "test_end_idx": int(test_end),
                    "train_n": int(len(train_meta)),
                    "val_n": int(len(val_meta)),
                    "test_n": int(len(test_meta)),
                    "status": "accepted" if (not train_meta.empty and not val_meta.empty and not test_meta.empty) else "skipped_empty_partition",
                }
                window_candidate_rows.append(candidate_row)

                if candidate_row["status"] == "accepted":
                    candidate_row["accepted_window"] = int(accepted_w)

                    window_rows.append({
                        "window": int(accepted_w),
                        "train_start_idx": int(train_cursor),
                        "train_end_idx": int(train_end),
                        "val_start_idx": int(val_start),
                        "val_end_idx": int(val_end),
                        "test_start_idx": int(test_start),
                        "test_end_idx": int(test_end),
                        "train_n": int(len(train_meta)),
                        "val_n": int(len(val_meta)),
                        "test_n": int(len(test_meta)),
                        "charts_n": None,
                    })

                    tasks.append({
                        "window": int(accepted_w),
                        "asset": asset,
                        "seed": int(args.seed + accepted_w * 1009),
                        "out_dir": str(out_dir),
                        "feature_mode": args.feature_mode,
                        "train_meta": train_meta.to_dict(orient="records"),
                        "val_meta": val_meta.to_dict(orient="records"),
                        "test_meta": test_meta.to_dict(orient="records"),
                        "h_min": args.h_min,
                        "h_max": args.h_max,
                        "delta_h": args.delta_h,
                        "lambda_lpm": args.lambda_lpm,
                        "eta_cost": args.eta_cost,
                        "lambda_smooth": args.lambda_smooth,
                        "reward_cost_mode": args.reward_cost_mode,
                        "lambda_turnover_h": args.lambda_turnover_h,
                        "lambda_contract_turnover": args.lambda_contract_turnover,
                        "lambda_roll_cost": args.lambda_roll_cost,
                        "lambda_position_size": args.lambda_position_size,
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
                        "device": args.device,
                        "torch_threads": args.torch_threads,
                        "blas_threads": args.blas_threads,
                        "plot_fraction": args.plot_fraction,
                        "max_plots_per_window": args.max_plots_per_window,
                    })

                    accepted_w += 1

                    if int(args.max_windows) > 0 and accepted_w >= int(args.max_windows):
                        break

                else:
                    print(
                        f"[skip candidate {candidate_idx:02d}] train={len(train_meta)} val={len(val_meta)} test={len(test_meta)} "
                        f"idx={train_cursor}:{train_end}|{val_start}:{val_end}|{test_start}:{test_end}",
                        flush=True,
                    )

                candidate_idx += 1
                train_cursor += step_days

            if not tasks:
                raise RuntimeError("No valid windows were built for parallel execution.")

            print(f"[parallel] windows to run: {len(tasks)}", flush=True)
            worker_results: List[Dict[str, object]] = []
            parallel_start = time.time()
            with ProcessPoolExecutor(max_workers=int(args.parallel_windows)) as ex:
                futures = [ex.submit(_train_eval_one_window_worker, task) for task in tasks]
                for i, fut in enumerate(as_completed(futures), 1):
                    result = fut.result()
                    worker_results.append(result)
                    if result.get("status") == "ok":
                        for row in window_rows:
                            if int(row["window"]) == int(result["window"]):
                                row["charts_n"] = int(result.get("charts_n", 0))
                                row["duration_sec"] = float(result.get("duration_sec", np.nan))
                    _progress_line(
                        "parallel windows",
                        i,
                        len(tasks),
                        parallel_start,
                        extra=f"latest_window={result.get('window')} status={result.get('status')}",
                    )

            failed = [r for r in worker_results if r.get("status") != "ok"]
            if failed:
                print(f"[parallel] failed windows: {[r.get('window') for r in failed]}", flush=True)

            print("[combine] collecting per-window outputs", flush=True)
            all_val_logs = []
            all_test_logs = []
            all_val_summaries = []
            all_test_summaries = []
            all_report_rows = []
            for result in sorted(worker_results, key=lambda x: int(x.get("window", -1))):
                if result.get("status") != "ok":
                    continue
                wdir = Path(str(result["window_dir"]))
                try:
                    all_val_logs.append(pd.read_parquet(wdir / "val_daily_log.parquet"))
                    all_test_logs.append(pd.read_parquet(wdir / "test_daily_log.parquet"))
                    all_val_summaries.append(pd.read_parquet(wdir / "val_episode_summary.parquet"))
                    all_test_summaries.append(pd.read_parquet(wdir / "test_episode_summary.parquet"))
                    all_report_rows.append(pd.read_parquet(wdir / "results.parquet"))
                except Exception:
                    all_val_logs.append(pd.read_csv(wdir / "val_daily_log.csv"))
                    all_test_logs.append(pd.read_csv(wdir / "test_daily_log.csv"))
                    all_val_summaries.append(pd.read_csv(wdir / "val_episode_summary.csv"))
                    all_test_summaries.append(pd.read_csv(wdir / "test_episode_summary.csv"))
                    all_report_rows.append(pd.read_csv(wdir / "results.csv"))

            if not all_report_rows:
                raise RuntimeError("Parallel execution finished but produced no successful result rows.")

            combined_val_log = pd.concat(all_val_logs, ignore_index=True) if all_val_logs else pd.DataFrame()
            combined_test_log = pd.concat(all_test_logs, ignore_index=True) if all_test_logs else pd.DataFrame()
            combined_val_summary = pd.concat(all_val_summaries, ignore_index=True) if all_val_summaries else pd.DataFrame()
            combined_test_summary = pd.concat(all_test_summaries, ignore_index=True) if all_test_summaries else pd.DataFrame()
            combined_results = pd.concat(all_report_rows, ignore_index=True) if all_report_rows else pd.DataFrame()
            windows_df = pd.DataFrame(window_rows)
            window_candidates_df = pd.DataFrame(window_candidate_rows)
            worker_results_df = pd.DataFrame(worker_results)

            combined_val_log.to_csv(out_dir / "val_daily_logs_all_windows.csv", index=False)
            combined_test_log.to_csv(out_dir / "test_daily_logs_all_windows.csv", index=False)
            combined_val_summary.to_csv(out_dir / "val_episode_summary_all_windows.csv", index=False)
            combined_test_summary.to_csv(out_dir / "test_episode_summary_all_windows.csv", index=False)
            combined_results.to_csv(out_dir / "results_all_windows.csv", index=False)
            windows_df.to_csv(out_dir / "rolling_window_plan.csv", index=False)
            window_candidates_df.to_csv(out_dir / "rolling_window_candidates.csv", index=False)
            worker_results_df.to_csv(out_dir / "parallel_worker_results.csv", index=False)

            try:
                combined_val_log.to_parquet(out_dir / "val_daily_logs_all_windows.parquet", index=False)
                combined_test_log.to_parquet(out_dir / "test_daily_logs_all_windows.parquet", index=False)
                combined_val_summary.to_parquet(out_dir / "val_episode_summary_all_windows.parquet", index=False)
                combined_test_summary.to_parquet(out_dir / "test_episode_summary_all_windows.parquet", index=False)
                combined_results.to_parquet(out_dir / "results_all_windows.parquet", index=False)
                windows_df.to_parquet(out_dir / "rolling_window_plan.parquet", index=False)
                window_candidates_df.to_parquet(out_dir / "rolling_window_candidates.parquet", index=False)
                worker_results_df.to_parquet(out_dir / "parallel_worker_results.parquet", index=False)
            except Exception as exc:
                print(f"[WARN] combined parquet write failed: {exc}", flush=True)

            with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        "env_cfg": asdict(cfg),
                        "precompute_path": str(pre_path),
                        "scenario_mode": "rolling_windows_scenario_root_parallel",
                        "train_scenario_kinds": train_kinds,
                        "eval_scenario_kinds": eval_kinds,
                        "n_features": int(np.asarray(pre["feature_matrix"]).shape[1]),
                        "n_windows": int(len(windows_df)),
                        "n_window_candidates": int(len(window_candidates_df)),
                        "parallel_windows": int(args.parallel_windows),
                        "h_min": float(args.h_min),
                        "h_max": float(args.h_max),
                        "delta_h": float(args.delta_h),
                        "reward_cost_mode": str(args.reward_cost_mode),
                        "lambda_turnover_h": float(args.lambda_turnover_h),
                        "lambda_contract_turnover": float(args.lambda_contract_turnover),
                        "lambda_roll_cost": float(args.lambda_roll_cost),
                        "lambda_position_size": float(args.lambda_position_size),
                        "warm_start": False,
                        "device": str(args.device),
                        "torch_threads": int(args.torch_threads),
                        "blas_threads": int(args.blas_threads),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print("\n" + "=" * 90)
            print("PARALLEL ROLLING WINDOWS DONE")
            print(f"windows:       {len(windows_df)}")
            print(f"results rows:  {len(combined_results)}")
            print(f"report input:  {out_dir / 'results_all_windows.parquet'}")
            print(f"charts folder: {out_dir / 'charts'}")
            print("=" * 90)
            return

        if not all_report_rows:
            raise RuntimeError("No rolling windows were trained. Check train/val/test lengths and scenario availability.")

        print("[combine] combining logs and summaries across windows", flush=True)
        combined_val_log = pd.concat(all_val_logs, ignore_index=True) if all_val_logs else pd.DataFrame()
        combined_test_log = pd.concat(all_test_logs, ignore_index=True) if all_test_logs else pd.DataFrame()
        combined_val_summary = pd.concat(all_val_summaries, ignore_index=True) if all_val_summaries else pd.DataFrame()
        combined_test_summary = pd.concat(all_test_summaries, ignore_index=True) if all_test_summaries else pd.DataFrame()
        combined_results = pd.concat(all_report_rows, ignore_index=True) if all_report_rows else pd.DataFrame()
        windows_df = pd.DataFrame(window_rows)
        window_candidates_df = pd.DataFrame(window_candidate_rows)

        print("[combine] writing combined CSV/parquet outputs", flush=True)
        combined_val_log.to_csv(out_dir / "val_daily_logs_all_windows.csv", index=False)
        combined_test_log.to_csv(out_dir / "test_daily_logs_all_windows.csv", index=False)
        combined_val_summary.to_csv(out_dir / "val_episode_summary_all_windows.csv", index=False)
        combined_test_summary.to_csv(out_dir / "test_episode_summary_all_windows.csv", index=False)
        combined_results.to_csv(out_dir / "results_all_windows.csv", index=False)
        windows_df.to_csv(out_dir / "rolling_window_plan.csv", index=False)
        window_candidates_df.to_csv(out_dir / "rolling_window_candidates.csv", index=False)
        try:
            combined_val_log.to_parquet(out_dir / "val_daily_logs_all_windows.parquet", index=False)
            combined_test_log.to_parquet(out_dir / "test_daily_logs_all_windows.parquet", index=False)
            combined_val_summary.to_parquet(out_dir / "val_episode_summary_all_windows.parquet", index=False)
            combined_test_summary.to_parquet(out_dir / "test_episode_summary_all_windows.parquet", index=False)
            combined_results.to_parquet(out_dir / "results_all_windows.parquet", index=False)
            windows_df.to_parquet(out_dir / "rolling_window_plan.parquet", index=False)
            window_candidates_df.to_parquet(out_dir / "rolling_window_candidates.parquet", index=False)
        except Exception as exc:
            print(f"[WARN] combined parquet write failed: {exc}")

        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "env_cfg": asdict(cfg),
                    "precompute_path": str(pre_path),
                    "scenario_mode": "rolling_windows_scenario_root",
                    "train_scenario_kinds": train_kinds,
                    "eval_scenario_kinds": eval_kinds,
                    "n_features": int(np.asarray(pre["feature_matrix"]).shape[1]),
                    "n_windows": int(len(windows_df)),
                    "n_window_candidates": int(len(window_candidates_df)),
                    "h_min": float(args.h_min),
                    "h_max": float(args.h_max),
                    "delta_h": float(args.delta_h),
                    "reward_cost_mode": str(args.reward_cost_mode),
                    "lambda_turnover_h": float(args.lambda_turnover_h),
                    "lambda_contract_turnover": float(args.lambda_contract_turnover),
                    "lambda_roll_cost": float(args.lambda_roll_cost),
                    "lambda_position_size": float(args.lambda_position_size),
                    "warm_start": bool(args.warm_start),
                    "keep_replay_buffer_on_warm_start": bool(args.keep_replay_buffer_on_warm_start),
                    "device": str(args.device),
                    "torch_threads": int(args.torch_threads),
                    "blas_threads": int(args.blas_threads),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print("\n" + "=" * 90)
        print("ROLLING WINDOWS DONE")
        print(f"windows:       {len(windows_df)}")
        print(f"results rows:  {len(combined_results)}")
        print(f"report input:  {out_dir / 'results_all_windows.parquet'}")
        print(f"charts folder: {out_dir / 'charts'}")
        print("=" * 90)
        return

    run_name = f"{args.asset.lower()}_sac_lpm_T{args.timesteps}_L{args.episode_len}"
    out_dir = OUTPUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SAC Portfolio-LPM training")
    print(f"asset:          {args.asset}")
    print(f"precompute:     {pre_path}")
    print(f"features:       {np.asarray(pre['feature_matrix']).shape[1]}")
    print(f"scenario mode:  {scenario_mode}")
    print(f"train scenarios:{len(train_scenarios)}")
    print(f"val scenarios:  {len(val_scenarios)}")
    print(f"test scenarios: {len(test_scenarios)}")
    print(f"output:         {out_dir}")
    print("=" * 80)

    env = build_env(pre, train_scenarios, cfg, seed=args.seed)
    net_arch = [int(x.strip()) for x in args.net_arch.split(",") if x.strip()]
    policy_kwargs = {"net_arch": net_arch}

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=args.learning_starts,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(out_dir / "tb"),
        device=args.device,
    )
    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)

    model_path = out_dir / "model.zip"
    model.save(model_path)
    print(f"Saved model: {model_path}")

    # Evaluation logs
    val_log = run_policy_on_scenarios(
        model,
        pre,
        val_scenarios,
        cfg,
        deterministic=True,
        max_episodes=args.eval_episodes,
    )
    test_log = run_policy_on_scenarios(
        model,
        pre,
        test_scenarios,
        cfg,
        deterministic=True,
        max_episodes=args.eval_episodes,
    )

    val_log_path = out_dir / "val_daily_log.csv"
    test_log_path = out_dir / "test_daily_log.csv"
    val_log.to_csv(val_log_path, index=False)
    test_log.to_csv(test_log_path, index=False)

    val_summary = compute_episode_summary(val_log)
    test_summary = compute_episode_summary(test_log)
    val_summary.to_csv(out_dir / "val_episode_summary.csv", index=False)
    test_summary.to_csv(out_dir / "test_episode_summary.csv", index=False)

    print_metric_block(val_summary, "validation")
    print_metric_block(test_summary, "test")

    # Save config and run args
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "env_cfg": asdict(cfg),
                "precompute_path": str(pre_path),
                "scenario_mode": scenario_mode,
                "scenario_root": str(args.scenario_root),
                "scenario_kinds": args.scenario_kinds,
                "n_features": int(np.asarray(pre["feature_matrix"]).shape[1]),
                "train_cut": int(train_cut),
                "val_cut": int(val_cut),
                "n_total": int(n),
                "reward_cost_mode": str(args.reward_cost_mode),
                "lambda_turnover_h": float(args.lambda_turnover_h),
                "lambda_contract_turnover": float(args.lambda_contract_turnover),
                "lambda_roll_cost": float(args.lambda_roll_cost),
                "lambda_position_size": float(args.lambda_position_size),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if not test_log.empty:
        plot_sample_episode(test_log, out_dir / "sample_test_episode_0.png", scenario_order=0)
        print(f"Saved sample plot: {out_dir / 'sample_test_episode_0.png'}")


if __name__ == "__main__":
    main()