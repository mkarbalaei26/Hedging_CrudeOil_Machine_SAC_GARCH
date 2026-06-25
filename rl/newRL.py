#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regime-aware naive hedge override with the project's real hedging environment.

This version is intentionally aligned with the previous walk-forward pipeline:
- loads `rl_cache/precompute_<EXPOSURE>.npz`
- loads physical trading scenarios from `scenarios/<EXPOSURE>/*.parquet`
- reuses `OilHedgingDailyEnv`, so transaction costs, integer contracts, mark-to-market,
  rollover, physical PnL and margin/path mechanics remain exactly as in the project env.

The only new idea is the action interface:

    action -> multiplier in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    target_h = naive_h * multiplier

The wrapper converts this target hedge ratio to the closest delta-h action of the base env.
So the RL agent learns when to follow naive, reduce it, zero it, or increase it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise RuntimeError("gymnasium is required. Install stable-baselines3 dependencies.") from exc

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as exc:  # pragma: no cover
    raise RuntimeError("stable-baselines3 is required: pip install stable-baselines3") from exc

from rl.precompute import load_npz
from rl.env_daily import EnvConfig, OilHedgingDailyEnv
from rl.scenario_loader import ScenarioLoaderConfig, load_scenarios_from_parquet


# -----------------------------------------------------------------------------
# Date/scenario helpers
# -----------------------------------------------------------------------------


def _date_to_year(d: int) -> int:
    """Supports both YYYYMMDD integers and numpy datetime64[D] integer day counts."""
    d = int(d)
    if d > 10_000_000:  # YYYYMMDD
        return d // 10000
    return int(np.datetime64(d, "D").astype("datetime64[Y]").astype(int) + 1970)


def _bounds_for_years_like_cache(dates_int: np.ndarray, y0: int, y1: int) -> Tuple[int, int]:
    """Return inclusive date bounds matching the cache date encoding."""
    dates_int = np.asarray(dates_int, dtype=np.int64).reshape(-1)
    max_d = int(np.nanmax(dates_int))
    if max_d > 10_000_000:  # YYYYMMDD
        return int(y0 * 10000 + 101), int(y1 * 10000 + 1231)
    s = int(np.datetime64(f"{int(y0)}-01-01", "D").astype("int64"))
    e = int(np.datetime64(f"{int(y1)}-12-31", "D").astype("int64"))
    return s, e


def _available_years(dates_int: np.ndarray) -> Tuple[int, int]:
    years = np.asarray([_date_to_year(int(x)) for x in np.asarray(dates_int).reshape(-1)], dtype=np.int32)
    return int(np.nanmin(years)), int(np.nanmax(years))


def _load_window_scenarios(
    *,
    scenario_path: str,
    dates_int: np.ndarray,
    exposure_id: str,
    y0: int,
    y1: int,
    max_scenarios: int,
    seed: int,
    require_ok_coverage: bool = True,
    allow_shortened: bool = False,
) -> List[Dict[str, Any]]:
    start_int, end_int = _bounds_for_years_like_cache(dates_int, int(y0), int(y1))
    cfg = ScenarioLoaderConfig(
        require_ok_coverage=bool(require_ok_coverage),
        allow_shortened=bool(allow_shortened),
        max_scenarios=int(max_scenarios) if int(max_scenarios) > 0 else None,
        seed=int(seed),
    )
    scenarios = load_scenarios_from_parquet(
        scenario_path,
        dates_int=dates_int,
        exposure_id=str(exposure_id),
        window_start_day_int=int(start_int),
        window_end_day_int=int(end_int),
        require_full_containment=True,
        cfg=cfg,
    )
    if not scenarios:
        raise ValueError(
            f"No scenarios loaded from {scenario_path} for {exposure_id} in {y0}-{y1}. "
            "Check scenario_dir/exposure_id/date encoding."
        )
    return scenarios


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, np.nan)) for r in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# -----------------------------------------------------------------------------
# Override wrapper around the real project environment
# -----------------------------------------------------------------------------


@dataclass
class OverrideRewardConfig:
    multipliers: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50)
    naive_h: float = 1.0
    mu_rel: float = 1.0
    eta_cost_rel: float = 1.0
    kappa_turnover: float = 0.05
    lambda_downside: float = 3.0
    lambda_underperform: float = 1.0


class NaiveOverrideRealEnv(OilHedgingDailyEnv):
    """Use real OilHedgingDailyEnv mechanics, but expose discrete multiplier actions."""

    def __init__(
        self,
        pre: Any,
        scenarios: List[Dict[str, Any]],
        cfg: EnvConfig,
        override_cfg: OverrideRewardConfig,
        seed: Optional[int] = None,
    ):
        # Force base env into discrete delta-h mode. The wrapper action is multiplier;
        # it is converted to the nearest base delta-h action before calling super().step().
        cfg.action_mode = "delta_h_discrete"
        cfg.info_mode = "eval"
        super().__init__(pre, scenarios, cfg=cfg, seed=seed)
        self.override_cfg = override_cfg
        self._base_delta_grid = self._make_delta_grid(cfg)
        self._base_action_space = self.action_space
        self.action_space = spaces.Discrete(len(override_cfg.multipliers))
        self._naive_n_prev = 0
        self._naive_equity = 0.0
        self._naive_mdd = 0.0
        self._naive_peak = 0.0

    @staticmethod
    def _make_delta_grid(cfg: EnvConfig) -> np.ndarray:
        explicit_grid = getattr(cfg, "delta_h_grid", None)
        if explicit_grid is not None:
            grid = np.asarray(explicit_grid, dtype=np.float32).reshape(-1)
        else:
            dh_max = float(getattr(cfg, "delta_h_max", 0.10))
            dh_step = float(getattr(cfg, "delta_h_step", 0.05))
            if dh_step <= 0:
                dh_step = 0.05
            n_grid_steps = int(np.floor(dh_max / dh_step + 1e-9))
            grid = (np.arange(-n_grid_steps, n_grid_steps + 1, dtype=np.float32) * np.float32(dh_step)).astype(np.float32)
            if grid.size == 0 or not np.any(np.isclose(grid, 0.0)):
                grid = np.sort(np.unique(np.append(grid, np.float32(0.0)))).astype(np.float32)
        return grid

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._naive_n_prev = 0
        self._naive_equity = 0.0
        self._naive_mdd = 0.0
        self._naive_peak = 0.0
        return obs, info

    def _contracts_from_h_silent(self, h: float) -> int:
        return int(self._contracts_from_h(float(h)))

    @staticmethod
    def _extract_total_cost(cost_obj: Any) -> float:
        """Convert project cost-model output to a scalar total cost."""
        if isinstance(cost_obj, dict):
            for key in (
                "total",
                "total_cost",
                "cost",
                "transaction_cost_total",
                "execution_cost_total",
                "tcost",
                "value",
            ):
                if key in cost_obj:
                    try:
                        return float(cost_obj[key])
                    except Exception:
                        pass
            total = 0.0
            found = False
            for value in cost_obj.values():
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                    total += float(value)
                    found = True
            if found:
                return float(total)
            raise TypeError(f"Could not extract scalar total cost from cost dict keys={list(cost_obj.keys())}")
        return float(cost_obj)

    def _estimate_naive_step_pnl(self, *, t: int, roll_flag: bool) -> Tuple[float, float, int]:
        """Fixed naive benchmark using the same physical PnL, futures PnL and cost model."""
        h_b = float(np.clip(float(self.override_cfg.naive_h), -float(self.cfg.h_max), float(self.cfg.h_max)))
        n_b = self._contracts_from_h_silent(h_b)
        pnl_phys_arr = getattr(self, "pnl_phys", None)
        pnl_phys = float(pnl_phys_arr[int(t)]) if pnl_phys_arr is not None else 0.0
        pnl_fut = float(n_b * float(self.pnl_1c[int(t)]))
        cost_raw = self.cost_model.total_cost(int(self._naive_n_prev), int(n_b), bool(roll_flag))
        cost_b = self._extract_total_cost(cost_raw)
        pnl_net_b = float(pnl_phys + pnl_fut - cost_b)
        self._naive_n_prev = int(n_b)
        self._naive_equity += pnl_net_b
        self._naive_peak = max(float(self._naive_peak), float(self._naive_equity))
        self._naive_mdd = min(float(self._naive_mdd), float(self._naive_equity - self._naive_peak))
        return pnl_net_b, cost_b, n_b

    def step(self, action):
        action_int = int(np.asarray(action).reshape(-1)[0])
        action_int = int(np.clip(action_int, 0, len(self.override_cfg.multipliers) - 1))

        h_before = float(getattr(self, "_h", 0.0))
        mult = float(self.override_cfg.multipliers[action_int])
        target_h = float(np.clip(float(self.override_cfg.naive_h) * mult, -float(self.cfg.h_max), float(self.cfg.h_max)))
        desired_dh = float(target_h - h_before)
        base_action = int(np.argmin(np.abs(self._base_delta_grid - desired_dh)))

        obs, old_reward, terminated, truncated, info = super().step(base_action)

        h_after = float(getattr(self, "_h", h_before))
        t = int(info.get("t", max(int(getattr(self, "_t", 1)) - 1, 0)))
        roll_flag = bool(info.get("roll_flag", info.get("is_roll", False)))
        pnl_agent = float(info.get("pnl_net", 0.0))
        cost_agent = float(info.get("cost", 0.0))
        pnl_naive, cost_naive, n_naive = self._estimate_naive_step_pnl(t=t, roll_flag=roll_flag)

        # Normalize by physical notional. This follows the spirit of the SAC trainer.
        try:
            if t > 0:
                denom = float(self._Q * float(self.spot[t - 1]))
            else:
                denom = float(self._Q * float(self.spot[t]))
        except Exception:
            denom = 1.0
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            denom = 1.0

        r_agent = pnl_agent / denom
        r_naive = pnl_naive / denom
        r_rel = r_agent - r_naive
        delta_h = abs(float(h_after - h_before))
        downside = max(0.0, -r_agent)
        underperform = max(0.0, -r_rel)

        ocfg = self.override_cfg
        reward = (
            float(ocfg.mu_rel) * r_rel
            - float(ocfg.eta_cost_rel) * (cost_agent / denom)
            - float(ocfg.kappa_turnover) * delta_h
            - float(ocfg.lambda_downside) * (downside ** 2.0)
            - float(ocfg.lambda_underperform) * (underperform ** 2.0)
        )

        info.update(
            {
                "override_action": int(action_int),
                "override_multiplier": float(mult),
                "override_target_h": float(target_h),
                "override_base_action": int(base_action),
                "override_base_delta_h": float(self._base_delta_grid[base_action]),
                "naive_h": float(self.override_cfg.naive_h),
                "naive_n": int(n_naive),
                "naive_cost": float(cost_naive),
                "naive_pnl_net": float(pnl_naive),
                "naive_equity": float(self._naive_equity),
                "naive_mdd": float(self._naive_mdd),
                "relative_pnl_net": float(pnl_agent - pnl_naive),
                "relative_return": float(r_rel),
                "old_env_reward": float(old_reward),
            }
        )
        return obs, float(reward), terminated, truncated, info


# -----------------------------------------------------------------------------
# Training/evaluation
# -----------------------------------------------------------------------------


def _make_env(pre: Any, scenarios: List[Dict[str, Any]], env_cfg: EnvConfig, override_cfg: OverrideRewardConfig, seed: int):
    def _factory():
        return Monitor(NaiveOverrideRealEnv(pre, scenarios, cfg=env_cfg, override_cfg=override_cfg, seed=seed))

    return _factory


def run_episodes(env: NaiveOverrideRealEnv, model: Any, n_episodes: int, *, deterministic: bool = True, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if seed is not None:
        env.reset(seed=int(seed))

    for ep in range(int(n_episodes)):
        obs, info0 = env.reset()
        done = False
        reward_sum = 0.0
        pnl_sum = 0.0
        cost_sum = 0.0
        naive_pnl_sum = 0.0
        naive_cost_sum = 0.0
        relative_sum = 0.0
        turnover_h = 0.0
        last_h = None
        h_values: List[float] = []
        action_counts: Dict[int, int] = {}
        mdd = 0.0
        naive_mdd = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_int = int(np.asarray(action).reshape(-1)[0])
            action_counts[action_int] = action_counts.get(action_int, 0) + 1
            obs, r, terminated, truncated, inf = env.step(action)
            done = bool(terminated or truncated)
            reward_sum += float(r)
            pnl_sum += float(inf.get("pnl_net", 0.0))
            cost_sum += float(inf.get("cost", 0.0))
            naive_pnl_sum += float(inf.get("naive_pnl_net", 0.0))
            naive_cost_sum += float(inf.get("naive_cost", 0.0))
            relative_sum += float(inf.get("relative_pnl_net", 0.0))
            mdd = float(inf.get("mdd", mdd))
            naive_mdd = float(inf.get("naive_mdd", naive_mdd))
            h_now = float(inf.get("h", inf.get("h_new", inf.get("h_prev", 0.0))))
            h_values.append(h_now)
            if last_h is None:
                turnover_h += abs(h_now)
            else:
                turnover_h += abs(h_now - last_h)
            last_h = h_now
            steps += 1

        rows.append(
            {
                "episode_id": int(ep),
                "scenario_idx": int(info0.get("scenario_idx", -1)),
                "Q": float(info0.get("Q", np.nan)),
                "steps": int(steps),
                "reward_sum": float(reward_sum),
                "pnl_net_sum": float(pnl_sum),
                "cost_sum": float(cost_sum),
                "naive_pnl_net_sum": float(naive_pnl_sum),
                "naive_cost_sum": float(naive_cost_sum),
                "relative_pnl_net_sum": float(relative_sum),
                "win_vs_naive": int(relative_sum > 0.0),
                "mdd": float(mdd),
                "naive_mdd": float(naive_mdd),
                "turnover_h": float(turnover_h),
                "h_mean": float(np.mean(h_values)) if h_values else 0.0,
                "h_abs_mean": float(np.mean(np.abs(h_values))) if h_values else 0.0,
                "h_std": float(np.std(h_values)) if h_values else 0.0,
                "action_counts": json.dumps(action_counts, sort_keys=True),
            }
        )
    return rows


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQN policy that overrides naive hedge using real scenario/cost/roll env.")
    p.add_argument("--cache", required=True, help="Path to rl_cache/precompute_<EXPOSURE>.npz")
    p.add_argument("--scenario_dir", required=True, help="Scenario root directory, e.g. scenarios")
    p.add_argument("--exposure_id", required=True, help="WTI / BRENT / OPEC_BASKET etc. Must match scenario folder")
    p.add_argument("--out_dir", required=True)

    p.add_argument("--year_train_start", type=int, default=2008)
    p.add_argument("--year_train_end", type=int, default=2018)
    p.add_argument("--year_eval_start", type=int, default=2019)
    p.add_argument("--year_eval_end", type=int, default=2020)
    p.add_argument("--year_test_start", type=int, default=2021)
    p.add_argument("--year_test_end", type=int, default=2025)

    p.add_argument("--max_train_scenarios", type=int, default=5000)
    p.add_argument("--max_eval_scenarios", type=int, default=1000)
    p.add_argument("--eval_episodes", type=int, default=500)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="auto")

    p.add_argument("--naive_h", type=float, default=1.0)
    p.add_argument("--multipliers", default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    p.add_argument("--h_max", type=float, default=2.0)
    p.add_argument("--delta_h_max", type=float, default=0.50)
    p.add_argument("--delta_h_step", type=float, default=0.05)

    p.add_argument("--risk_mode", default="lpm")
    p.add_argument("--mu_pnl", type=float, default=0.10)
    p.add_argument("--lambda_var", type=float, default=0.0)
    p.add_argument("--lambda_lpm", type=float, default=1.0)
    p.add_argument("--lpm_order", type=int, default=2)
    p.add_argument("--lpm_target", type=float, default=0.0)
    p.add_argument("--eta_cost", type=float, default=1.0)

    p.add_argument("--mu_rel", type=float, default=1.0)
    p.add_argument("--eta_cost_rel", type=float, default=1.0)
    p.add_argument("--kappa_turnover", type=float, default=0.05)
    p.add_argument("--lambda_downside", type=float, default=3.0)
    p.add_argument("--lambda_underperform", type=float, default=1.0)

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--train_freq", type=int, default=4)
    p.add_argument("--gradient_steps", type=int, default=1)
    p.add_argument("--target_update_interval", type=int, default=2000)
    p.add_argument("--exploration_fraction", type=float, default=0.30)
    p.add_argument("--exploration_final_eps", type=float, default=0.05)
    p.add_argument("--eval_freq", type=int, default=10_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = load_npz(args.cache)
    dates_int = np.asarray(pre.dates_int, dtype=np.int64)
    y_min, y_max = _available_years(dates_int)
    print(f"Cache years detected: {y_min}..{y_max}")

    scenario_root = Path(args.scenario_dir) / str(args.exposure_id)
    p_train = scenario_root / "oracle_universe.parquet"
    p_oracle_all = scenario_root / "oracle_all.parquet"
    p_baseline = scenario_root / "baseline.parquet"
    for pth in [p_train, p_oracle_all, p_baseline]:
        if not pth.exists():
            raise FileNotFoundError(f"Scenario file not found: {pth}")

    train_scenarios = _load_window_scenarios(
        scenario_path=str(p_train),
        dates_int=dates_int,
        exposure_id=str(args.exposure_id),
        y0=args.year_train_start,
        y1=args.year_train_end,
        max_scenarios=args.max_train_scenarios,
        seed=args.seed + 11,
    )
    eval_scenarios = _load_window_scenarios(
        scenario_path=str(p_train),
        dates_int=dates_int,
        exposure_id=str(args.exposure_id),
        y0=args.year_eval_start,
        y1=args.year_eval_end,
        max_scenarios=args.max_eval_scenarios,
        seed=args.seed + 22,
    )
    test_scenarios = _load_window_scenarios(
        scenario_path=str(p_train),
        dates_int=dates_int,
        exposure_id=str(args.exposure_id),
        y0=args.year_test_start,
        y1=args.year_test_end,
        max_scenarios=args.max_eval_scenarios,
        seed=args.seed + 33,
    )
    test_scenarios_all = _load_window_scenarios(
        scenario_path=str(p_oracle_all),
        dates_int=dates_int,
        exposure_id=str(args.exposure_id),
        y0=args.year_test_start,
        y1=args.year_test_end,
        max_scenarios=args.max_eval_scenarios,
        seed=args.seed + 44,
        require_ok_coverage=True,
    )
    test_scenarios_baseline = _load_window_scenarios(
        scenario_path=str(p_baseline),
        dates_int=dates_int,
        exposure_id=str(args.exposure_id),
        y0=args.year_test_start,
        y1=args.year_test_end,
        max_scenarios=args.max_eval_scenarios,
        seed=args.seed + 55,
        require_ok_coverage=True,
    )

    env_cfg = EnvConfig(
        h_max=float(args.h_max),
        delta_h_max=float(args.delta_h_max),
        delta_h_step=float(args.delta_h_step),
        action_mode="delta_h_discrete",
        risk_mode=str(args.risk_mode),
        mu_pnl=float(args.mu_pnl),
        lambda_var=float(args.lambda_var),
        lambda_lpm=float(args.lambda_lpm),
        lpm_order=int(args.lpm_order),
        lpm_target=float(args.lpm_target),
        eta_cost=float(args.eta_cost),
        info_mode="eval",
    )
    multipliers = tuple(float(x.strip()) for x in str(args.multipliers).split(",") if x.strip())
    override_cfg = OverrideRewardConfig(
        multipliers=multipliers,
        naive_h=float(args.naive_h),
        mu_rel=float(args.mu_rel),
        eta_cost_rel=float(args.eta_cost_rel),
        kappa_turnover=float(args.kappa_turnover),
        lambda_downside=float(args.lambda_downside),
        lambda_underperform=float(args.lambda_underperform),
    )

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "env_cfg": asdict(env_cfg), "override_cfg": asdict(override_cfg)}, f, ensure_ascii=False, indent=2)

    train_env = DummyVecEnv([_make_env(pre, train_scenarios, env_cfg, override_cfg, seed=args.seed)])
    eval_env = DummyVecEnv([_make_env(pre, eval_scenarios, env_cfg, override_cfg, seed=args.seed + 1000)])

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(args.learning_rate),
        buffer_size=int(args.buffer_size),
        learning_starts=min(5000, max(500, int(args.total_timesteps) // 20)),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        train_freq=int(args.train_freq),
        gradient_steps=int(args.gradient_steps),
        target_update_interval=int(args.target_update_interval),
        exploration_fraction=float(args.exploration_fraction),
        exploration_final_eps=float(args.exploration_final_eps),
        policy_kwargs={"net_arch": [128, 128]},
        seed=int(args.seed),
        verbose=1,
        device=str(args.device),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=int(args.eval_freq),
        n_eval_episodes=min(int(args.eval_episodes), len(eval_scenarios)),
        deterministic=True,
        render=False,
    )

    print("=" * 100)
    print("Regime-aware naive override DQN using REAL OilHedgingDailyEnv")
    print(f"cache={args.cache}")
    print(f"scenario_root={scenario_root}")
    print(f"train scenarios={len(train_scenarios):,} | eval={len(eval_scenarios):,} | test={len(test_scenarios):,}")
    print(f"multipliers={multipliers} | naive_h={args.naive_h}")
    print("=" * 100)

    model.learn(total_timesteps=int(args.total_timesteps), callback=eval_callback, progress_bar=True)
    model.save(str(out_dir / "model_final.zip"))

    best_path = out_dir / "best_model" / "best_model.zip"
    if best_path.exists():
        model = DQN.load(str(best_path), env=train_env, device=str(args.device))

    def _eval_dataset(name: str, scenarios: List[Dict[str, Any]], seed: int) -> pd.DataFrame:
        env = NaiveOverrideRealEnv(pre, scenarios, cfg=env_cfg, override_cfg=override_cfg, seed=seed)
        n = min(int(args.eval_episodes), len(scenarios))
        rows = run_episodes(env, model, n_episodes=n, deterministic=True, seed=seed)
        df = pd.DataFrame(rows)
        df["dataset"] = str(name)
        return df

    dfs = [
        _eval_dataset("oracle_universe", test_scenarios, args.seed + 2020),
        _eval_dataset("oracle_all", test_scenarios_all, args.seed + 3030),
        _eval_dataset("baseline", test_scenarios_baseline, args.seed + 4040),
    ]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_parquet(out_dir / "test_episodes.parquet", index=False)
    df_all.to_csv(out_dir / "test_episodes.csv", index=False)

    summary_rows = []
    for name, g in df_all.groupby("dataset"):
        summary_rows.append(
            {
                "dataset": name,
                "n": int(len(g)),
                "reward_mean": float(g["reward_sum"].mean()),
                "pnl_net_mean": float(g["pnl_net_sum"].mean()),
                "naive_pnl_net_mean": float(g["naive_pnl_net_sum"].mean()),
                "relative_pnl_net_mean": float(g["relative_pnl_net_sum"].mean()),
                "win_rate_vs_naive": float(g["win_vs_naive"].mean()),
                "cost_mean": float(g["cost_sum"].mean()),
                "naive_cost_mean": float(g["naive_cost_sum"].mean()),
                "mdd_mean": float(g["mdd"].mean()),
                "naive_mdd_mean": float(g["naive_mdd"].mean()),
                "turnover_h_mean": float(g["turnover_h"].mean()),
                "h_abs_mean": float(g["h_abs_mean"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_by_dataset.csv", index=False)
    with open(out_dir / "summary_by_dataset.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("\nTest summary by dataset")
    print(summary.to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()