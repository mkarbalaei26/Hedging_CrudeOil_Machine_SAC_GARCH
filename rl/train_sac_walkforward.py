#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAC walk-forward trainer for crude-oil hedging.

This file is aligned with the existing PPO pipeline, but uses SAC with a
continuous delta-hedge action and a smoother reward-v2.

Run from the project root, for example:

python -m rl.train_sac_walkforward \
  --cache rl_cache/precompute_OPEC_BASKET.npz \
  --out_dir rl_runs/SAC_OPEC_SMOKE \
  --exposure_id OPEC_BASKET \
  --scenario_dir scenarios \
  --train_mode hybrid_expanding \
  --year_start 2008 \
  --year_end 2025 \
  --max_windows 1 \
  --total_timesteps 50000

Outputs:
- <out_dir>/<window_name>/model_best.zip
- <out_dir>/<window_name>/episodes_all.parquet
- <out_dir>/results_all_windows.parquet
- <out_dir>/run_config.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from rl.precompute import load_npz
from rl.scenario_loader import ScenarioLoaderConfig, load_scenarios_from_parquet
from rl.env_daily import EnvConfig, OilHedgingDailyEnv

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except Exception as e:  # pragma: no cover
    raise RuntimeError("stable-baselines3 is required for train_sac_walkforward.py") from e

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# -----------------------------------------------------------------------------
# SAC reward-v2 environment
# -----------------------------------------------------------------------------


@dataclass
class EnvConfigSAC(EnvConfig):
    """SAC-specific environment config.

    SAC uses a continuous action. For thesis-grade hedging evaluation we allow
    two semantics:

    - target:      h_t = naive_hedge_ratio + action_t
    - incremental: h_t = h_{t-1} + action_t

    The default is incremental because it makes the action a true hedge-ratio
    adjustment. However, the objective is not forced dynamic behaviour; it is
    risk-adjusted outperformance versus naive hedging after costs.
    """

    # Paper PnL signal
    mu_paper: float = 1.0

    # Total portfolio PnL signal. Keep this positive but usually smaller than
    # mu_paper: the futures leg is directly controlled by the agent, while the
    # physical leg is market-driven. Still, hedging must improve total portfolio
    # outcomes, not only paper PnL.
    mu_total: float = 0.25

    # Reward selection. "current" keeps the earlier multi-term reward.
    # "relative_naive" makes the objective explicitly benchmark-relative.
    reward_mode: str = "current"  # "current" | "relative_naive" | "dual_lpm"
    naive_benchmark_h: float = 0.5
    mu_rel: float = 1.0

    # Dual-LPM reward mode:
    # Paper leg gets a linear LPM penalty; total portfolio gets a quadratic LPM.
    lpm_paper_target: float = 0.0
    lpm_net_target: float = 0.0
    lambda_paper_lpm: float = 1.0
    lambda_net_lpm: float = 10.0
    mu_net_upside: float = 0.15
    boundary_penalty: float = 0.0
    boundary_zone: float = 0.03

    # Adaptive downside penalty on total portfolio return
    lambda_down: float = 10.0
    lambda_paper_loss: float = 1.0
    downside_order: int = 2
    dd_threshold: float = 0.3
    dd_beta: float = 3.0
    lambda_drawdown: float = 2.0

    # Cost and stability
    eta_cost_v2: float = 5.0
    kappa_stability: float = 0.5

    # Soft inaction penalty under stress
    omega_inaction: float = 5.0
    alpha_inaction: float = 10.0
    n_loss_max: int = 5
    dd_panic_threshold: float = 0.2

    # SAC action design
    action_mode: str = "delta_h_continuous"
    sac_action_semantics: str = "incremental"  # "incremental" | "target"
    delta_h_bounds: Tuple[float, float] = (-0.2, 0.2)
    min_action_change_threshold: float = 0.0

    # Rule-based risk overlay applied after SAC action.
    use_risk_overlay: bool = False
    overlay_loss_days_1: int = 3
    overlay_loss_days_2: int = 5
    overlay_reentry_profit_days: int = 5
    overlay_reentry_sum_threshold: float = 0.0
    overlay_cut_1: float = 0.50
    overlay_cut_2: float = 0.00
    overlay_profit_slope_days: int = 3
    overlay_profit_boost: float = 0.15
    overlay_dd_reduce_threshold: float = 0.15
    overlay_dd_cut: float = 0.50
    overlay_max_daily_abs_dh: float = 0.25
    overlay_use_dcc_garch: bool = True
    overlay_dcc_low: float = 0.35
    overlay_dcc_high: float = 0.75
    overlay_vol_high_z: float = 1.0
    overlay_hr_blend: float = 0.50

    # Full info is needed for reward-v2.
    info_mode: str = "eval"


class OilHedgingDailyEnvSAC(OilHedgingDailyEnv):
    """Base hedging env with SAC reward-v2.

    We reuse all market mechanics from OilHedgingDailyEnv: PnL, roll, cost,
    integer contracts, observations, and scenario handling. Only the reward is
    replaced.
    """

    def __init__(self, pre: Any, scenarios: List[Dict[str, Any]], cfg: EnvConfigSAC, seed: Optional[int] = None):
        cfg.action_mode = "delta_h_continuous"
        cfg.info_mode = "eval"
        super().__init__(pre, scenarios, cfg=cfg, seed=seed)
        self.cfg_sac = cfg
        self._n_loss_consecutive = 0
        if str(getattr(cfg, "sac_action_semantics", "incremental")) not in {"incremental", "target"}:
            raise ValueError("sac_action_semantics must be 'incremental' or 'target'")
        if str(getattr(cfg, "reward_mode", "current")) not in {"current", "relative_naive", "dual_lpm"}:
            raise ValueError("reward_mode must be 'current', 'relative_naive', or 'dual_lpm'")
        self._overlay_disabled = False
        self._paper_pnl_hist: List[float] = []
        self._net_pnl_hist: List[float] = []
        self._overlay_feature_idx: Dict[str, int] = self._build_overlay_feature_index(pre)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._n_loss_consecutive = 0
        self._overlay_disabled = False
        self._paper_pnl_hist = []
        self._net_pnl_hist = []
        return obs, info

    def _build_overlay_feature_index(self, pre: Any) -> Dict[str, int]:
        names = getattr(pre, "feature_names", None)
        if names is None and isinstance(getattr(pre, "meta", None), dict):
            names = pre.meta.get("feature_names") or pre.meta.get("features")
        if names is None:
            return {}
        idx: Dict[str, int] = {}
        for i, name in enumerate(list(names)):
            n = str(name).lower()
            if "dcc" in n and "corr" in n and "dcc_corr" not in idx:
                idx["dcc_corr"] = i
            if "ccc" in n and "corr" in n and "ccc_corr" not in idx:
                idx["ccc_corr"] = i
            if "garch" in n and ("vol" in n or "sigma" in n) and "garch_vol" not in idx:
                idx["garch_vol"] = i
            if "dcc" in n and ("hr" in n or "hedge" in n or "ratio" in n) and "dcc_hr" not in idx:
                idx["dcc_hr"] = i
            if "ols" in n and ("hr" in n or "hedge" in n or "ratio" in n) and "ols_hr" not in idx:
                idx["ols_hr"] = i
        return idx

    def _feature_value(self, key: str, t: int) -> Optional[float]:
        j = self._overlay_feature_idx.get(str(key))
        if j is None:
            return None
        try:
            x = float(self.X[int(t), int(j)])
            return x if np.isfinite(x) else None
        except Exception:
            return None

    @staticmethod
    def _recent_sum(xs: List[float], n: int) -> float:
        n = max(int(n), 1)
        if not xs:
            return 0.0
        return float(np.sum(xs[-n:]))

    @staticmethod
    def _consecutive_negative(xs: List[float], n: int) -> bool:
        n = max(int(n), 1)
        return len(xs) >= n and all(float(v) < 0.0 for v in xs[-n:])

    @staticmethod
    def _slope_positive(xs: List[float], n: int) -> bool:
        n = max(int(n), 2)
        if len(xs) < n:
            return False
        y = np.asarray(xs[-n:], dtype=np.float64)
        dy = np.diff(y)
        return bool(len(dy) > 0 and np.all(dy > 0.0) and float(dy[-1]) >= float(np.mean(dy)))

    def _apply_risk_overlay(self, *, h_model: float, h_before: float, t: int, denom: float, mdd: float) -> Tuple[float, Dict[str, Any]]:
        if not bool(getattr(self.cfg_sac, "use_risk_overlay", False)):
            return float(h_model), {"overlay_active": 0, "overlay_reason": "off"}

        cfg = self.cfg_sac
        reasons: List[str] = []
        h = float(h_model)
        h_max = float(getattr(cfg, "h_max", 1.0))
        denom = float(denom) if np.isfinite(denom) and abs(denom) > 1e-12 else 1.0
        dd_norm = abs(float(mdd)) / denom

        # Rule 1/2: stop-loss on paper leg.
        if self._consecutive_negative(self._paper_pnl_hist, int(cfg.overlay_loss_days_2)):
            self._overlay_disabled = True
            h = float(cfg.overlay_cut_2) * h
            reasons.append("paper_loss_lockout")
        elif self._consecutive_negative(self._paper_pnl_hist, int(cfg.overlay_loss_days_1)):
            h = float(cfg.overlay_cut_1) * h
            reasons.append("paper_loss_cut")

        # Rule 3: re-entry gate.
        if self._overlay_disabled:
            recent = self._recent_sum(self._paper_pnl_hist, int(cfg.overlay_reentry_profit_days))
            if recent > float(cfg.overlay_reentry_sum_threshold):
                self._overlay_disabled = False
                reasons.append("reentry_allowed")
            else:
                h = 0.0
                reasons.append("reentry_blocked")

        # Rule 4: profit momentum boost.
        if (not self._overlay_disabled) and self._recent_sum(self._paper_pnl_hist, int(cfg.overlay_profit_slope_days)) > 0.0:
            if self._slope_positive(np.cumsum(self._paper_pnl_hist).tolist(), int(cfg.overlay_profit_slope_days)):
                h = h + np.sign(h if abs(h) > 1e-12 else 1.0) * float(cfg.overlay_profit_boost)
                reasons.append("paper_profit_momentum_boost")

        # Rule 5: drawdown cut.
        if dd_norm > float(cfg.overlay_dd_reduce_threshold):
            h = float(cfg.overlay_dd_cut) * h
            reasons.append("drawdown_cut")

        # Rule 6/7/8: DCC/GARCH-aware controls, only if those features exist.
        if bool(getattr(cfg, "overlay_use_dcc_garch", True)):
            dcc_corr = self._feature_value("dcc_corr", t)
            if dcc_corr is None:
                dcc_corr = self._feature_value("ccc_corr", t)
            if dcc_corr is not None:
                if dcc_corr < float(cfg.overlay_dcc_low):
                    h = 0.50 * h
                    reasons.append("low_dcc_corr_cut")
                elif dcc_corr > float(cfg.overlay_dcc_high):
                    h = h + np.sign(h if abs(h) > 1e-12 else 1.0) * 0.05
                    reasons.append("high_dcc_corr_support")

            gvol = self._feature_value("garch_vol", t)
            if gvol is not None:
                try:
                    j = self._overlay_feature_idx.get("garch_vol")
                    hist = np.asarray(self.X[max(0, int(t) - 252): int(t) + 1, int(j)], dtype=np.float64)
                    mu, sig = float(np.nanmean(hist)), float(np.nanstd(hist))
                    z = (gvol - mu) / max(sig, 1e-12)
                    if np.isfinite(z) and z > float(cfg.overlay_vol_high_z):
                        h = 0.70 * h
                        reasons.append("high_garch_vol_cut")
                except Exception:
                    pass

            hr = self._feature_value("dcc_hr", t)
            if hr is None:
                hr = self._feature_value("ols_hr", t)
            if hr is not None and np.isfinite(hr):
                hr = float(np.clip(hr, -h_max, h_max))
                blend = float(np.clip(float(cfg.overlay_hr_blend), 0.0, 1.0))
                h = (1.0 - blend) * h + blend * hr
                reasons.append("blend_to_econometric_hr")

        # Rule 9: daily delta clamp.
        max_dh = float(getattr(cfg, "overlay_max_daily_abs_dh", 0.25))
        if max_dh > 0:
            h = float(np.clip(h, h_before - max_dh, h_before + max_dh))
            reasons.append("daily_dh_clamp")

        h = float(np.clip(h, -h_max, h_max))
        return h, {
            "overlay_active": int(bool(reasons)),
            "overlay_reason": ";".join(reasons) if reasons else "none",
            "overlay_disabled": int(bool(self._overlay_disabled)),
            "overlay_h_model": float(h_model),
            "overlay_h_final": float(h),
        }

    def _contracts_from_h_silent(self, h: float) -> int:
        """Convert hedge ratio to contracts without changing env state."""
        return int(self._contracts_from_h(float(h)))

    def _estimate_naive_pnl_for_current_step(self, *, t: int, cost_t: float) -> float:
        """Estimate same-day net PnL for a fixed naive benchmark hedge."""
        h_b = float(getattr(self.cfg_sac, "naive_benchmark_h", 0.5))
        h_b = float(np.clip(h_b, -float(self.cfg.h_max), float(self.cfg.h_max)))
        n_b = self._contracts_from_h_silent(h_b)
        pnl_phys_arr = getattr(self, "pnl_phys", None)
        if pnl_phys_arr is None:
            pnl_phys = 0.0
        else:
            pnl_phys = float(pnl_phys_arr[int(t)])
        pnl_fut_b = float(n_b * float(self.pnl_1c[int(t)]))
        return float(pnl_phys + pnl_fut_b - float(cost_t))

    def _reward_v2(
        self,
        *,
        pnl_net: float,
        pnl_fut: float,
        cost_t: float,
        denom: float,
        delta_h: float,
        mdd: float,
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self.cfg_sac
        denom = float(denom) if np.isfinite(denom) and abs(denom) > 1e-12 else 1.0

        # Attribution term: futures/paper leg net of cost. This rewards the
        # part most directly controlled by the hedge decision.
        paper_pnl = float(pnl_fut) - float(cost_t)
        r_paper = float(paper_pnl / denom)

        # Portfolio term: hedging is ultimately judged on total portfolio PnL.
        # This prevents the agent from maximizing futures PnL while worsening
        # the physical+futures portfolio.
        r_net = float(pnl_net / denom)

        reward_mode = str(getattr(cfg, "reward_mode", "current"))
        t_for_benchmark = int(getattr(self, "_t", 1)) - 1
        if t_for_benchmark < 0:
            t_for_benchmark = 0
        try:
            pnl_naive = self._estimate_naive_pnl_for_current_step(t=t_for_benchmark, cost_t=cost_t)
        except Exception:
            pnl_naive = 0.0
        r_naive = float(pnl_naive / denom)
        r_rel = float(r_net - r_naive)

        p = max(int(cfg.downside_order), 1)
        net_downside = max(0.0, -r_net)
        downside_term = net_downside if p == 1 else net_downside ** float(p)

        paper_loss = max(0.0, -r_paper)
        paper_loss_term = paper_loss if p == 1 else paper_loss ** float(p)

        if paper_pnl < 0:
            self._n_loss_consecutive += 1
        else:
            self._n_loss_consecutive = 0

        dd_norm = abs(float(mdd)) / denom
        phi = 1.0 + float(cfg.dd_beta) * max(0.0, dd_norm - float(cfg.dd_threshold))

        cost_norm = float(cost_t) / denom
        stability = float(delta_h) ** 2

        # Drawdown penalty is linear in normalized drawdown beyond threshold.
        # This is intentionally separate from one-step downside risk because MDD
        # is a path-risk metric, not a one-period return metric.
        drawdown_excess = max(0.0, dd_norm - float(cfg.dd_threshold))

        # Soft inaction penalty is now mild: it only discourages being frozen in
        # a stressed losing state. It is not the main objective because beating
        # naive is more important than forcing dynamic behaviour.
        loss_ratio = min(1.0, float(self._n_loss_consecutive) / max(float(cfg.n_loss_max), 1.0))
        panic_signal = max(0.0, dd_norm - float(cfg.dd_panic_threshold))
        psi = loss_ratio * panic_signal
        inaction = psi * float(cfg.omega_inaction) * float(np.exp(-float(cfg.alpha_inaction) * abs(float(delta_h))))

        if reward_mode == "relative_naive":
            reward = (
                float(cfg.mu_rel) * r_rel
                - phi * float(cfg.lambda_down) * downside_term
                - float(cfg.lambda_drawdown) * drawdown_excess
                - float(cfg.eta_cost_v2) * cost_norm
            )
        elif reward_mode == "dual_lpm":
            paper_lpm1 = max(0.0, float(cfg.lpm_paper_target) - r_paper)
            net_lpm2 = max(0.0, float(cfg.lpm_net_target) - r_net) ** 2.0
            # Pure dual-LPM objective: no explicit upside reward, no cost penalty,
            # and no boundary penalty. The agent is free to trade/speculate; it is
            # penalized only when paper PnL or total portfolio PnL falls below the
            # configured LPM targets.
            reward = (
                - float(cfg.lambda_paper_lpm) * paper_lpm1
                - float(cfg.lambda_net_lpm) * net_lpm2
            )
        else:
            paper_lpm1 = 0.0
            net_lpm2 = 0.0
            upside = 0.0
            boundary_term = 0.0
            reward = (
                float(cfg.mu_paper) * r_paper
                + float(cfg.mu_total) * r_net
                - phi * float(cfg.lambda_down) * downside_term
                - float(cfg.lambda_paper_loss) * paper_loss_term
                - float(cfg.lambda_drawdown) * drawdown_excess
                - float(cfg.eta_cost_v2) * cost_norm
                - float(cfg.kappa_stability) * stability
                - inaction
            )

        parts = {
            "r_paper": float(r_paper),
            "r_net": float(r_net),
            "r_naive": float(r_naive),
            "r_rel": float(r_rel),
            "pnl_naive_step": float(pnl_naive),
            "paper_lpm1": float(locals().get("paper_lpm1", 0.0)),
            "net_lpm2": float(locals().get("net_lpm2", 0.0)),
            "net_upside": float(locals().get("upside", 0.0)),
            "boundary_term": float(locals().get("boundary_term", 0.0)),
            "downside_term": float(downside_term),
            "paper_loss_term": float(paper_loss_term),
            "phi": float(phi),
            "dd_norm": float(dd_norm),
            "drawdown_excess": float(drawdown_excess),
            "cost_norm": float(cost_norm),
            "stability": float(stability),
            "psi": float(psi),
            "inaction": float(inaction),
            "reward_v2": float(reward),
        }
        return float(reward), parts

    def step(self, action):
        h_before = float(getattr(self, "_h", 0.0))
        n_before = int(getattr(self, "_n_prev", 0))

        # The base env's delta_h_continuous mode interprets action as:
        #     h_new = naive_hedge_ratio + action
        # For SAC we optionally reinterpret action as a true incremental change:
        #     h_new = h_prev + action
        # This is implemented by temporarily shifting naive_hedge_ratio before
        # delegating to the base env, so all mechanics/cost/roll/PnL code remain
        # unchanged.
        original_naive = float(getattr(self.cfg, "naive_hedge_ratio", 1.0))
        semantics = str(getattr(self.cfg_sac, "sac_action_semantics", "incremental"))
        if semantics == "incremental":
            self.cfg.naive_hedge_ratio = h_before
        try:
            obs, reward_old, terminated, truncated, info = super().step(action)
        finally:
            self.cfg.naive_hedge_ratio = original_naive

        h_after = float(getattr(self, "_h", h_before))
        n_after = int(getattr(self, "_n_prev", n_before))
        delta_h = float(h_after - h_before)

        overlay_info: Dict[str, Any] = {"overlay_active": 0, "overlay_reason": "off"}
        try:
            t_for_overlay = int(info.get("t", max(int(getattr(self, "_t", 1)) - 1, 0)))
            if t_for_overlay > 0:
                denom_overlay = float(self._Q * float(self.spot[t_for_overlay - 1]))
            else:
                denom_overlay = float(self._Q * float(self.spot[t_for_overlay]))
            h_overlay, overlay_info = self._apply_risk_overlay(
                h_model=h_after,
                h_before=h_before,
                t=t_for_overlay,
                denom=denom_overlay,
                mdd=float(info.get("mdd", 0.0)),
            )
            if bool(overlay_info.get("overlay_active", 0)):
                n_overlay = int(self._contracts_from_h_silent(h_overlay))
                if n_overlay != int(getattr(self, "_n_prev", n_after)):
                    extra_cost = float(self.cost_model.total_cost(int(getattr(self, "_n_prev", n_after)), n_overlay, roll_flag=False))
                    info["cost"] = float(info.get("cost", 0.0)) + extra_cost
                    info["pnl_net"] = float(info.get("pnl_net", 0.0)) - extra_cost
                    self._equity -= extra_cost
                    self._n_prev = int(n_overlay)
                    self._h = float(h_overlay)
                    h_after = float(h_overlay)
                    n_after = int(n_overlay)
                    delta_h = float(h_after - h_before)
        except Exception as e:
            overlay_info = {"overlay_active": 0, "overlay_reason": f"overlay_error:{type(e).__name__}"}

        pnl_net = float(info.get("pnl_net", 0.0))
        pnl_fut = float(info.get("pnl_fut", 0.0))
        cost_t = float(info.get("cost", 0.0))
        mdd = float(info.get("mdd", 0.0))
        self._paper_pnl_hist.append(float(pnl_fut) - float(cost_t))
        self._net_pnl_hist.append(float(pnl_net))

        t = int(info.get("t", max(int(getattr(self, "_t", 1)) - 1, 0)))
        if t > 0:
            denom = float(self._Q * float(self.spot[t - 1]))
        else:
            denom = float(self._Q * float(self.spot[t]))
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            denom = 1.0

        reward_v2, parts = self._reward_v2(
            pnl_net=pnl_net,
            pnl_fut=pnl_fut,
            cost_t=cost_t,
            denom=denom,
            delta_h=delta_h,
            mdd=mdd,
        )

        info["reward_old"] = float(reward_old)
        info["reward_v2"] = float(reward_v2)
        info["delta_h"] = float(delta_h)
        info["h_before"] = float(h_before)
        info["h_after"] = float(h_after)
        info["n_before"] = int(n_before)
        info["n_after"] = int(n_after)
        info["sac_action_semantics"] = str(getattr(self.cfg_sac, "sac_action_semantics", "incremental"))
        info.update(overlay_info)
        info["n_loss_consecutive"] = int(self._n_loss_consecutive)
        info.update(parts)
        return obs, reward_v2, terminated, truncated, info


def make_sac_env(pre: Any, scenarios: List[Dict[str, Any]], cfg: EnvConfigSAC, seed: Optional[int] = None):
    def _thunk():
        return OilHedgingDailyEnvSAC(pre, scenarios, cfg=cfg, seed=seed)
    return _thunk


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


class _TeeIO:
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


def _fmt_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    return str(timedelta(seconds=int(seconds)))


def _select_device(device_arg: str) -> str:
    device_arg = str(device_arg)
    if device_arg in ("cpu", "mps", "cuda"):
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def first_date_of_year(y: int) -> np.datetime64:
    return np.datetime64(f"{int(y):04d}-01-01", "D")


def last_date_of_year(y: int) -> np.datetime64:
    return np.datetime64(f"{int(y):04d}-12-31", "D")


def year_bounds(y0: int, y1: int) -> Tuple[int, int]:
    return int(first_date_of_year(y0).astype("int64")), int(last_date_of_year(y1).astype("int64"))


def build_walkforward_windows(
    *,
    year_start: int,
    year_end: int,
    train_mode: str,
    train_years_initial: int,
    expanding_max_years: int,
    step_years: int,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    train_years_initial = int(train_years_initial)
    expanding_max_years = int(expanding_max_years)
    step_years = max(int(step_years), 1)

    # First window: train_start..train_start+initial-1, val next year, test next year.
    anchor = int(year_start)
    train_y1 = anchor + train_years_initial - 1
    while True:
        val_y = train_y1 + 1
        test_y = train_y1 + 2
        if test_y > int(year_end):
            break

        if train_mode == "rolling":
            train_y0 = train_y1 - train_years_initial + 1
        elif train_mode == "expanding":
            train_y0 = anchor
        elif train_mode == "hybrid_expanding":
            train_y0 = max(anchor, train_y1 - expanding_max_years + 1)
        else:
            raise ValueError(f"unknown train_mode={train_mode}")

        ws_train, we_train = year_bounds(train_y0, train_y1)
        ws_val, we_val = year_bounds(val_y, val_y)
        ws_test, we_test = year_bounds(test_y, test_y)
        name = f"WF_train{train_y0}-{train_y1}_val{val_y}_test{test_y}"
        windows.append({
            "window_name": name,
            "train_y0": int(train_y0),
            "train_y1": int(train_y1),
            "val_y": int(val_y),
            "test_y": int(test_y),
            "train_start_int": int(ws_train),
            "train_end_int": int(we_train),
            "val_start_int": int(ws_val),
            "val_end_int": int(we_val),
            "test_start_int": int(ws_test),
            "test_end_int": int(we_test),
        })
        train_y1 += step_years
    return windows


def load_window_scenarios(
    *,
    scenario_path: str,
    dates_int: np.ndarray,
    exposure_id: str,
    start_int: int,
    end_int: int,
    max_scenarios: int,
    seed: int,
) -> List[Dict[str, Any]]:
    cfg = ScenarioLoaderConfig(
        require_ok_coverage=True,
        allow_shortened=False,
        max_scenarios=(int(max_scenarios) if int(max_scenarios) > 0 else None),
        seed=int(seed),
    )
    return load_scenarios_from_parquet(
        scenario_path,
        dates_int=dates_int,
        exposure_id=str(exposure_id),
        window_start_day_int=int(start_int),
        window_end_day_int=int(end_int),
        require_full_containment=True,
        cfg=cfg,
    )


def cap_episodes(n_req: int, n_avail: int) -> int:
    return int(n_avail) if int(n_req) <= 0 else int(min(int(n_req), int(n_avail)))


class SACProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, prefix: str = "SAC", print_freq: int = 10_000):
        super().__init__(verbose=0)
        self.total_timesteps = int(total_timesteps)
        self.prefix = str(prefix)
        self.print_freq = int(max(print_freq, 1))
        self.t0 = 0.0
        self.last_print = 0

    def _on_training_start(self) -> None:
        self.t0 = time.time()

    def _on_step(self) -> bool:
        n = int(self.model.num_timesteps)
        if n - self.last_print >= self.print_freq or n >= self.total_timesteps:
            self.last_print = n
            dt = max(time.time() - self.t0, 1e-9)
            fps = n / dt
            eta = max(self.total_timesteps - n, 0) / max(fps, 1e-9)
            print(f"[{self.prefix}] {n:,}/{self.total_timesteps:,} fps={fps:,.0f} ETA={_fmt_eta(eta)}", flush=True)
        return True


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def action_scalar(action: Any) -> float:
    try:
        return float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
    except Exception:
        return float("nan")


def run_eval_episodes(
    env: OilHedgingDailyEnvSAC,
    model: Any,
    n_episodes: int,
    *,
    deterministic: bool,
    seed: int,
    desc: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    env.reset(seed=int(seed))
    iterator: Iterable[int]
    if tqdm is not None:
        iterator = tqdm(range(int(n_episodes)), desc=desc, unit="ep", dynamic_ncols=True, file=sys.__stdout__)
    else:
        iterator = range(int(n_episodes))

    for _ in iterator:
        obs, info0 = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0
        ep_old = 0.0
        ep_v2 = 0.0
        ep_pnl = 0.0
        ep_cost = 0.0
        ep_turnover = 0.0
        h_turnover = 0.0
        last_n = None
        last_h = None
        h_values: List[float] = []
        n_values: List[int] = []
        actions: List[float] = []
        mdd = 0.0
        overlay_hits = 0

        while not done:
            action, _ = model.predict(obs, deterministic=bool(deterministic))
            actions.append(action_scalar(action))
            obs, r, terminated, truncated, inf = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            ep_reward += float(r)
            ep_old += float(inf.get("reward_old", 0.0))
            ep_v2 += float(inf.get("reward_v2", r))
            ep_pnl += float(inf.get("pnl_net", 0.0))
            ep_cost += float(inf.get("cost", 0.0))
            mdd = float(inf.get("mdd", mdd))
            overlay_hits += int(inf.get("overlay_active", 0))

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

        out.append({
            "scenario_idx": int(info0.get("scenario_idx", -1)),
            "Q": float(info0.get("Q", np.nan)),
            "steps": int(steps),
            "reward_sum": float(ep_reward),
            "reward_old_sum": float(ep_old),
            "reward_v2_sum": float(ep_v2),
            "pnl_net_sum": float(ep_pnl),
            "cost_sum": float(ep_cost),
            "turnover_contract": float(ep_turnover),
            "turnover_h": float(h_turnover),
            "h_abs_mean": float(np.mean(np.abs(h_values))) if h_values else 0.0,
            "h_mean": float(np.mean(h_values)) if h_values else 0.0,
            "h_std": float(np.std(h_values)) if h_values else 0.0,
            "h_nonzero_share": float(np.mean(np.abs(h_values) > 1e-6)) if h_values else 0.0,
            "n_abs_mean": float(np.mean(np.abs(n_values))) if n_values else 0.0,
            "action_mean": float(np.nanmean(actions)) if actions else 0.0,
            "action_std": float(np.nanstd(actions)) if actions else 0.0,
            "action_min": float(np.nanmin(actions)) if actions else 0.0,
            "action_max": float(np.nanmax(actions)) if actions else 0.0,
            "mdd": float(mdd),
            "overlay_hits": int(overlay_hits),
            "overlay_hit_share": float(overlay_hits / max(steps, 1)),
        })
    return out


def episodes_to_df(eps: List[Dict[str, Any]], dataset: str, scenarios: List[Dict[str, Any]], split: str) -> pd.DataFrame:
    df = pd.DataFrame(eps)
    if df.empty:
        df = pd.DataFrame(columns=["scenario_idx", "Q", "steps", "reward_sum", "pnl_net_sum", "cost_sum", "turnover_contract", "mdd"])

    def get_meta(i: int, key: str, default=None):
        if 0 <= int(i) < len(scenarios):
            return scenarios[int(i)].get(key, default)
        return default

    if "scenario_idx" not in df.columns:
        df["scenario_idx"] = -1
    df["scenario_idx"] = pd.to_numeric(df["scenario_idx"], errors="coerce").fillna(-1).astype(int)

    meta_cols = [
        "scenario_id", "tag", "oracle_series", "oracle_pool", "oracle_candidate",
        "company_id", "company_size", "start_idx", "end_idx", "start_date_int",
        "end_date_int", "horizon_days_target", "horizon_days_realized",
    ]
    idxs = df["scenario_idx"].to_numpy(dtype=int, copy=False)
    for col in meta_cols:
        df[col] = [get_meta(int(i), col, None) for i in idxs]

    df.insert(0, "split", str(split))
    df.insert(0, "dataset", str(dataset))
    return df


# -----------------------------------------------------------------------------
# Train/eval one window
# -----------------------------------------------------------------------------


def build_vec_env(pre: Any, scenarios: List[Dict[str, Any]], cfg: EnvConfigSAC, seed: int, n_envs: int, vec: str):
    thunks = [make_sac_env(pre, scenarios, cfg, seed=int(seed) + i) for i in range(max(int(n_envs), 1))]
    if int(n_envs) > 1 and str(vec) == "subproc":
        return SubprocVecEnv(thunks)
    return DummyVecEnv(thunks)


def train_eval_window(job: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(job["window_out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_sac.log"
    t0 = time.time()

    try:
        import torch
        if int(job.get("torch_threads", 0)) > 0:
            torch.set_num_threads(int(job["torch_threads"]))
        if int(job.get("torch_interop_threads", 0)) > 0:
            torch.set_num_interop_threads(int(job["torch_interop_threads"]))
    except Exception:
        pass

    live = bool(job.get("live", False))
    with open(log_path, "w", encoding="utf-8") as log_f:
        stdout_ctx = contextlib.redirect_stdout(_TeeIO(log_f, sys.__stdout__)) if live else contextlib.redirect_stdout(log_f)
        stderr_ctx = contextlib.nullcontext() if live else contextlib.redirect_stderr(log_f)
        with stdout_ctx, stderr_ctx:
            try:
                print("[SAC window job]", json.dumps({k: str(v) for k, v in job.items() if k not in {"env_cfg"}}, indent=2))
                pre = load_npz(str(job["cache"]))
                dates_int = pre.dates_int
                exposure_id = str(job["exposure_id"])
                scenario_dir = str(job["scenario_dir"])
                seed = int(job["seed"])

                p_univ = os.path.join(scenario_dir, exposure_id, "oracle_universe.parquet")
                p_all = os.path.join(scenario_dir, exposure_id, "oracle_all.parquet")
                p_base = os.path.join(scenario_dir, exposure_id, "baseline.parquet")

                train_scen = load_window_scenarios(
                    scenario_path=p_univ, dates_int=dates_int, exposure_id=exposure_id,
                    start_int=int(job["train_start_int"]), end_int=int(job["train_end_int"]),
                    max_scenarios=int(job["max_train_scenarios"]), seed=seed + 11,
                )
                val_scen = load_window_scenarios(
                    scenario_path=p_univ, dates_int=dates_int, exposure_id=exposure_id,
                    start_int=int(job["val_start_int"]), end_int=int(job["val_end_int"]),
                    max_scenarios=int(job["max_eval_scenarios"]), seed=seed + 22,
                )
                test_univ = load_window_scenarios(
                    scenario_path=p_univ, dates_int=dates_int, exposure_id=exposure_id,
                    start_int=int(job["test_start_int"]), end_int=int(job["test_end_int"]),
                    max_scenarios=int(job["max_eval_scenarios"]), seed=seed + 33,
                )
                test_all = load_window_scenarios(
                    scenario_path=p_all, dates_int=dates_int, exposure_id=exposure_id,
                    start_int=int(job["test_start_int"]), end_int=int(job["test_end_int"]),
                    max_scenarios=int(job["max_eval_scenarios"]), seed=seed + 44,
                )
                test_base = load_window_scenarios(
                    scenario_path=p_base, dates_int=dates_int, exposure_id=exposure_id,
                    start_int=int(job["test_start_int"]), end_int=int(job["test_end_int"]),
                    max_scenarios=int(job["max_eval_scenarios"]), seed=seed + 55,
                )

                if not train_scen:
                    raise RuntimeError("No train scenarios loaded for this window")

                env_cfg = EnvConfigSAC(**job["env_cfg"])
                (out_dir / "env_config_sac.json").write_text(json.dumps(asdict(env_cfg), indent=2), encoding="utf-8")

                device = _select_device(str(job["device"]))
                vec_env = build_vec_env(pre, train_scen, env_cfg, seed=seed, n_envs=int(job["n_envs"]), vec=str(job["vec"]))
                policy_kwargs = dict(net_arch=list(job["net_arch"]))

                model_path = out_dir / "model_best.zip"
                if bool(job.get("eval_only", False)):
                    if not model_path.exists():
                        raise FileNotFoundError(f"eval_only requested but checkpoint not found: {model_path}")
                    model = SAC.load(str(model_path), env=vec_env, device=device)
                else:
                    warm_start = str(job.get("warm_start_model") or "")
                    if warm_start and Path(warm_start).exists():
                        print(f"[warm_start] {warm_start}")
                        model = SAC.load(warm_start, env=vec_env, device=device)
                    else:
                        model = SAC(
                            "MlpPolicy",
                            vec_env,
                            learning_rate=float(job["learning_rate"]),
                            buffer_size=int(job["buffer_size"]),
                            learning_starts=int(job["learning_starts"]),
                            batch_size=int(job["batch_size"]),
                            tau=float(job["tau"]),
                            gamma=float(job["gamma"]),
                            train_freq=int(job["train_freq"]),
                            gradient_steps=int(job["gradient_steps"]),
                            ent_coef=str(job["ent_coef"]),
                            policy_kwargs=policy_kwargs,
                            verbose=0,
                            seed=seed,
                            device=device,
                        )
                    cb = SACProgressCallback(int(job["total_timesteps"]), prefix=f"SAC {job['window_name']}", print_freq=int(job["eta_print_freq"]))
                    model.learn(total_timesteps=int(job["total_timesteps"]), callback=cb, reset_num_timesteps=not bool(job.get("warm_start_model")))
                    model.save(str(model_path))

                try:
                    vec_env.close()
                except Exception:
                    pass

                n_eval = int(job["eval_episodes"])
                n_val = cap_episodes(n_eval, len(val_scen))
                n_univ = cap_episodes(n_eval, len(test_univ))
                n_all = cap_episodes(n_eval, len(test_all))
                n_base = cap_episodes(n_eval, len(test_base))

                frames: List[pd.DataFrame] = []
                for dataset, split, scenarios, n_ep, seed_off in [
                    ("oracle_universe", "val", val_scen, n_val, 1000),
                    ("oracle_universe", "test", test_univ, n_univ, 2000),
                    ("oracle_all", "test", test_all, n_all, 3000),
                    ("baseline", "test", test_base, n_base, 4000),
                ]:
                    if n_ep <= 0:
                        continue
                    env = OilHedgingDailyEnvSAC(pre, scenarios, cfg=env_cfg, seed=seed + seed_off)
                    eps = run_eval_episodes(env, model, n_ep, deterministic=True, seed=seed + seed_off + 1, desc=f"{job['window_name']} | {dataset}:{split}")
                    frames.append(episodes_to_df(eps, dataset, scenarios, split))

                if frames:
                    df = pd.concat(frames, ignore_index=True)
                else:
                    df = pd.DataFrame()
                df.insert(0, "strategy", "SAC")
                df.insert(0, "exposure_id", exposure_id)
                for key in ["window_name", "train_y0", "train_y1", "val_y", "test_y"]:
                    df[key] = job[key]
                df.to_parquet(out_dir / "episodes_all.parquet", index=False)

                summary = {
                    "ok": True,
                    "window_name": str(job["window_name"]),
                    "rows": int(len(df)),
                    "model_path": str(model_path),
                    "episodes_path": str(out_dir / "episodes_all.parquet"),
                    "elapsed_sec": float(time.time() - t0),
                    "n_train_scenarios": int(len(train_scen)),
                    "n_val_scenarios": int(len(val_scen)),
                    "n_test_universe": int(len(test_univ)),
                    "n_test_oracle_all": int(len(test_all)),
                    "n_test_baseline": int(len(test_base)),
                }
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                return summary
            except Exception as e:
                traceback.print_exc()
                summary = {
                    "ok": False,
                    "window_name": str(job.get("window_name")),
                    "error": repr(e),
                    "elapsed_sec": float(time.time() - t0),
                }
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                return summary


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC walk-forward trainer for crude-oil hedging")
    p.add_argument("--cache", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--exposure_id", required=True, choices=["WTI_SPOT", "BRENT_SPOT", "OPEC_BASKET"])
    p.add_argument("--scenario_dir", default="scenarios")
    p.add_argument("--train_mode", choices=["rolling", "expanding", "hybrid_expanding"], default="hybrid_expanding")
    p.add_argument("--year_start", type=int, default=2008)
    p.add_argument("--year_end", type=int, default=2025)
    p.add_argument("--train_years_initial", type=int, default=2)
    p.add_argument("--expanding_max_years", type=int, default=8)
    p.add_argument("--step_years", type=int, default=1)
    p.add_argument("--max_windows", type=int, default=0)
    p.add_argument("--parallel_windows", type=int, default=1)

    p.add_argument("--eval_only", action="store_true")

    # Optuna tuning mode. Reward coefficients are kept fixed; Optuna tunes only
    # SAC/action hyperparameters and maximizes validation reward_v2_sum.
    p.add_argument("--optuna_tune", action="store_true")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--trial_timesteps", type=int, default=100000)
    p.add_argument("--tune_max_windows", type=int, default=1)
    p.add_argument("--tune_eval_episodes", type=int, default=300)
    p.add_argument("--optuna_jobs", type=int, default=1, help="Parallel Optuna trials. Use cautiously with SB3 on macOS.")
    p.add_argument("--optuna_study_name", default="sac_hedging_reward_v2")
    p.add_argument("--optuna_storage", default="", help="Optional Optuna storage URL, e.g. sqlite:///sac_optuna.db")
    p.add_argument("--tune_reward", action="store_true", help="Tune selected reward coefficients using fixed external validation score.")
    p.add_argument("--run_best_after_tune", action="store_true", help="After Optuna finishes, train/evaluate the best configuration automatically.")
    p.add_argument("--best_timesteps", type=int, default=300000)
    p.add_argument("--best_max_windows", type=int, default=0, help="0 means all generated windows for the best run.")
    p.add_argument("--local_refine", action="store_true", help="Narrow Optuna search around the best SAC dual-LPM region found in prior trials.")

    p.add_argument("--device", default="auto")
    p.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--total_timesteps", type=int, default=200000)
    p.add_argument("--eval_episodes", type=int, default=300, help="0 means evaluate all loaded scenarios")
    p.add_argument("--max_train_scenarios", type=int, default=0)
    p.add_argument("--max_eval_scenarios", type=int, default=0)

    # SAC params
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--train_freq", type=int, default=1)
    p.add_argument("--gradient_steps", type=int, default=1)
    p.add_argument("--ent_coef", default="auto")
    p.add_argument("--net_arch", type=int, nargs="*", default=[256, 256])

    # Base env / action params
    p.add_argument("--h_max", type=float, default=1.25)
    p.add_argument("--naive_hedge_ratio", type=float, default=1.0)
    p.add_argument("--sac_action_semantics", choices=["incremental", "target"], default="incremental")
    p.add_argument("--delta_h_low", type=float, default=-0.2)
    p.add_argument("--delta_h_high", type=float, default=0.2)
    p.add_argument("--min_action_change_threshold", type=float, default=0.0)
    p.add_argument("--use_risk_overlay", action="store_true")
    p.add_argument("--overlay_loss_days_1", type=int, default=3)
    p.add_argument("--overlay_loss_days_2", type=int, default=5)
    p.add_argument("--overlay_reentry_profit_days", type=int, default=5)
    p.add_argument("--overlay_reentry_sum_threshold", type=float, default=0.0)
    p.add_argument("--overlay_cut_1", type=float, default=0.5)
    p.add_argument("--overlay_cut_2", type=float, default=0.0)
    p.add_argument("--overlay_profit_slope_days", type=int, default=3)
    p.add_argument("--overlay_profit_boost", type=float, default=0.15)
    p.add_argument("--overlay_dd_reduce_threshold", type=float, default=0.15)
    p.add_argument("--overlay_dd_cut", type=float, default=0.5)
    p.add_argument("--overlay_max_daily_abs_dh", type=float, default=0.25)
    p.add_argument("--overlay_use_dcc_garch", action="store_true")
    p.add_argument("--overlay_dcc_low", type=float, default=0.35)
    p.add_argument("--overlay_dcc_high", type=float, default=0.75)
    p.add_argument("--overlay_vol_high_z", type=float, default=1.0)
    p.add_argument("--overlay_hr_blend", type=float, default=0.5)
    p.add_argument("--cost_per_contract_trade_usd", type=float, default=10.0)

    # Reward v2 params
    p.add_argument("--mu_paper", type=float, default=1.0)
    p.add_argument("--mu_total", type=float, default=0.25)
    p.add_argument("--reward_mode", choices=["current", "relative_naive", "dual_lpm"], default="current")
    p.add_argument("--naive_benchmark_h", type=float, default=0.5)
    p.add_argument("--mu_rel", type=float, default=1.0)
    p.add_argument("--lpm_paper_target", type=float, default=0.0)
    p.add_argument("--lpm_net_target", type=float, default=0.0)
    p.add_argument("--lambda_paper_lpm", type=float, default=1.0)
    p.add_argument("--lambda_net_lpm", type=float, default=10.0)
    p.add_argument("--mu_net_upside", type=float, default=0.15)
    p.add_argument("--boundary_penalty", type=float, default=0.0)
    p.add_argument("--boundary_zone", type=float, default=0.03)
    p.add_argument("--lambda_down", type=float, default=10.0)
    p.add_argument("--lambda_paper_loss", type=float, default=1.0)
    p.add_argument("--downside_order", type=int, default=2)
    p.add_argument("--dd_threshold", type=float, default=0.3)
    p.add_argument("--dd_beta", type=float, default=3.0)
    p.add_argument("--lambda_drawdown", type=float, default=2.0)
    p.add_argument("--eta_cost_v2", type=float, default=5.0)
    p.add_argument("--kappa_stability", type=float, default=0.5)
    p.add_argument("--omega_inaction", type=float, default=5.0)
    p.add_argument("--alpha_inaction", type=float, default=10.0)
    p.add_argument("--n_loss_max", type=int, default=5)
    p.add_argument("--dd_panic_threshold", type=float, default=0.2)

    # System / logging
    p.add_argument("--torch_threads", type=int, default=2)
    p.add_argument("--torch_interop_threads", type=int, default=1)
    p.add_argument("--eta_print_freq", type=int, default=10_000)
    return p.parse_args()


def _summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compact SAC diagnostics for fast inspection."""
    if df.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["exposure_id", "dataset", "split", "strategy"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    def agg(g: pd.DataFrame) -> pd.Series:
        out: Dict[str, Any] = {}
        out["rows"] = int(len(g))
        out["n_scenarios"] = int(g["scenario_id"].nunique()) if "scenario_id" in g.columns else int(len(g))
        for col in [
            "pnl_net_sum", "cost_sum", "turnover_contract", "turnover_h",
            "overlay_hits", "overlay_hit_share",
            "h_abs_mean", "h_std", "h_nonzero_share", "n_abs_mean",
            "action_mean", "action_std", "action_min", "action_max", "mdd",
            "reward_sum", "reward_v2_sum", "reward_old_sum",
        ]:
            if col in g.columns:
                out[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
                out[f"{col}_median"] = float(pd.to_numeric(g[col], errors="coerce").median())
        if "pnl_net_sum" in g.columns:
            pnl = pd.to_numeric(g["pnl_net_sum"], errors="coerce")
            out["prob_profit"] = float((pnl > 0).mean())
            out["pnl_q05"] = float(pnl.quantile(0.05))
            out["pnl_q95"] = float(pnl.quantile(0.95))
        return pd.Series(out)

    return df.groupby(group_cols, dropna=False).apply(agg).reset_index()


def _write_ai_run_report(out_root: Path, args: argparse.Namespace, summaries: List[Dict[str, Any]], big: pd.DataFrame) -> None:
    """Write a readable markdown report about the SAC optimization run."""
    report_path = out_root / "sac_run_report.md"
    metrics_path = out_root / "sac_metrics_summary.csv"
    metrics = _summarize_results(big)
    if not metrics.empty:
        metrics.to_csv(metrics_path, index=False)

    lines: List[str] = []
    lines.append("# SAC RL Hedging Run Report\n")
    lines.append("## هدف اجرا\n")
    lines.append(
        "این اجرا از الگوریتم Soft Actor-Critic برای یادگیری نسبت پوشش ریسک پیوسته استفاده می‌کند. "
        "هر episode یک سناریوی معامله فیزیکی نفت است و عامل در هر روز نسبت پوشش ریسک را از طریق تغییر پیوسته‌ی delta-h تنظیم می‌کند.\n"
    )

    lines.append("## تنظیمات اصلی\n")
    for k in [
        "exposure_id", "train_mode", "year_start", "year_end", "train_years_initial",
        "expanding_max_years", "max_windows", "parallel_windows", "device", "vec", "n_envs",
        "total_timesteps", "eval_episodes", "learning_rate", "gamma", "tau",
        "buffer_size", "batch_size", "learning_starts", "train_freq", "gradient_steps", "ent_coef",
        "h_max", "naive_hedge_ratio", "sac_action_semantics", "delta_h_low", "delta_h_high",
        "use_risk_overlay", "overlay_loss_days_1", "overlay_loss_days_2", "overlay_reentry_profit_days",
        "overlay_cut_1", "overlay_cut_2", "overlay_profit_boost", "overlay_dd_reduce_threshold",
        "overlay_dd_cut", "overlay_max_daily_abs_dh", "overlay_use_dcc_garch",
        "reward_mode", "naive_benchmark_h", "mu_paper", "mu_total", "mu_rel",
        "lpm_paper_target", "lpm_net_target", "lambda_paper_lpm", "lambda_net_lpm", "mu_net_upside",
        "boundary_penalty", "boundary_zone", "lambda_down", "lambda_paper_loss", "downside_order",
        "dd_threshold", "dd_beta", "lambda_drawdown",
        "eta_cost_v2", "kappa_stability", "omega_inaction", "alpha_inaction",
        "n_loss_max", "dd_panic_threshold",
    ]:
        lines.append(f"- `{k}`: `{getattr(args, k, None)}`\n")

    lines.append("\n## تابع هدف استفاده‌شده\n")
    lines.append(
        "تابع پاداش نسخه دوم به‌صورت زیر تعریف شده است:\n\n"
        "```text\n"
        "R_t = mu_paper * r_paper + mu_total * r_net "
        "- phi(s_t) * lambda_down * max(0, -r_net)^p "
        "- lambda_paper_loss * max(0, -r_paper)^p "
        "- lambda_drawdown * max(0, dd_norm - dd_threshold) "
        "- eta_cost_v2 * cost_norm "
        "- kappa_stability * (Delta h)^2 "
        "- psi(s_t) * omega_inaction * exp(-alpha_inaction * |Delta h|)\n"
        "```\n\n"
        "در این فرمول، سیگنال سودآوری اصلی از بخش فیوچرز/کاغذی گرفته می‌شود، اما یک وزن مثبت کوچک‌تر برای بازده کل پورتفو نیز اضافه شده است. "
        "جریمه ریسک نزولی و افت سرمایه روی کل پورتفو اعمال می‌شود تا مدل صرفاً به سود فیوچرز تبدیل نشود و هدف اصلی یعنی بهترشدن پورتفوی هج‌شده نسبت به هج ساده حفظ شود.\n"
    )

    lines.append("## خلاصه پنجره‌های walk-forward\n")
    ok = sum(1 for s in summaries if bool(s.get("ok")))
    fail = sum(1 for s in summaries if not bool(s.get("ok")))
    lines.append(f"- پنجره‌های موفق: `{ok}`\n")
    lines.append(f"- پنجره‌های ناموفق: `{fail}`\n")
    if summaries:
        lines.append("\n")
        show = pd.DataFrame(summaries)
        keep = [c for c in ["window_name", "ok", "rows", "n_train_scenarios", "n_test_universe", "n_test_oracle_all", "n_test_baseline", "elapsed_sec", "error"] if c in show.columns]
        lines.append(show[keep].to_markdown(index=False))
        lines.append("\n")

    if not metrics.empty:
        lines.append("\n## خلاصه خروجی ارزیابی\n")
        lines.append(metrics.to_markdown(index=False))
        lines.append("\n")

    lines.append("\n## فایل‌های خروجی\n")
    lines.append("- `results_all_windows.parquet`: خروجی episodeها برای همه پنجره‌ها\n")
    lines.append("- `window_summaries.csv`: وضعیت train/eval هر پنجره\n")
    lines.append("- `sac_metrics_summary.csv`: خلاصه آماری سریع برای بررسی مدل\n")
    lines.append("- `<window>/train_sac.log`: لاگ آموزش هر پنجره\n")
    lines.append("- `<window>/model_best.zip`: مدل ذخیره‌شده SAC هر پنجره\n")

    report_path.write_text("".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Optuna tuning for SAC (fixed reward coefficients)
# -----------------------------------------------------------------------------

def _score_trial_from_results(df: pd.DataFrame) -> float:
    """Fixed external Optuna score, independent of reward coefficients.

    The score rewards validation PnL but strongly punishes path risk, tail loss,
    and boundary sticking. Cost is not explicitly penalized in this experiment.
    It is deliberately external to the training reward so
    reward coefficients can be tuned without making trial values incomparable.
    """
    if df.empty:
        return -1e18
    d = df.copy()
    if "dataset" in d.columns:
        d = d[d["dataset"].eq("oracle_universe")]
    if "split" in d.columns:
        d = d[d["split"].eq("val")]
    if d.empty:
        return -1e18

    pnl = pd.to_numeric(d.get("pnl_net_sum", pd.Series(dtype=float)), errors="coerce").dropna()
    mdd = pd.to_numeric(d.get("mdd", pd.Series(dtype=float)), errors="coerce").dropna()
    cost = pd.to_numeric(d.get("cost_sum", pd.Series(dtype=float)), errors="coerce").dropna()
    hstd = pd.to_numeric(d.get("h_std", pd.Series(dtype=float)), errors="coerce").dropna()
    habs = pd.to_numeric(d.get("h_abs_mean", pd.Series(dtype=float)), errors="coerce").dropna()

    if pnl.empty:
        return -1e18

    pnl_mean = float(pnl.mean())
    mdd_abs_mean = float(mdd.abs().mean()) if not mdd.empty else 0.0
    cost_mean = float(cost.mean()) if not cost.empty else 0.0
    hstd_mean = float(hstd.mean()) if not hstd.empty else 0.0

    var95 = float(pnl.quantile(0.05))
    cvar95 = float(pnl[pnl <= var95].mean()) if (pnl <= var95).any() else var95
    cvar_loss = max(0.0, -cvar95)

    boundary_pen = 0.0
    if not habs.empty:
        observed_max = float(habs.max())
        if observed_max > 0:
            boundary_pen = float((habs > 0.95 * observed_max).mean())

    # More conservative hedge-selection score:
    # keep PnL important, but penalize path risk and tail loss more strongly.
    # No h_std bonus: dynamic behaviour is not rewarded unless it improves PnL/risk.
    score = (
        pnl_mean / 1_000_000.0
        - 0.45 * (mdd_abs_mean / 1_000_000.0)
        - 0.40 * (cvar_loss / 1_000_000.0)
        - 0.20 * boundary_pen
    )
    return float(score)


def _trial_eval_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Diagnostics saved in Optuna trial attributes."""
    out: Dict[str, float] = {}
    if df.empty:
        return out
    d = df.copy()
    if "dataset" in d.columns:
        d = d[d["dataset"].eq("oracle_universe")]
    if "split" in d.columns:
        d = d[d["split"].eq("val")]
    if d.empty:
        return out
    for col in [
        "reward_v2_sum", "pnl_net_sum", "cost_sum", "turnover_h",
        "h_abs_mean", "h_std", "action_std", "mdd",
    ]:
        if col in d.columns:
            s = pd.to_numeric(d[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not s.empty:
                out[f"val_{col}_mean"] = float(s.mean())
                out[f"val_{col}_median"] = float(s.median())
    if "pnl_net_sum" in d.columns:
        pnl = pd.to_numeric(d["pnl_net_sum"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not pnl.empty:
            out["val_prob_profit"] = float((pnl > 0).mean())
    return out


def run_optuna_tuning(args: argparse.Namespace, windows: List[Dict[str, Any]], base_env_cfg: EnvConfigSAC) -> int:
    """Run Optuna tuning using fixed reward_v2_sum as validation objective.

    Reward coefficients must remain fixed across trials. Otherwise reward_v2_sum
    is not comparable. This tuner varies SAC/action hyperparameters only.
    """
    try:
        import optuna
    except Exception as e:
        raise RuntimeError("Optuna is required. Install it with: pip install optuna") from e

    out_root = Path(args.out_dir)
    tune_root = out_root / "optuna_trials"
    tune_root.mkdir(parents=True, exist_ok=True)

    tune_windows = windows[: max(1, int(args.tune_max_windows))]
    if not tune_windows:
        raise RuntimeError("No windows available for Optuna tuning")

    print(f"[Optuna] trials={args.n_trials} | tune_windows={len(tune_windows)}")
    print("[Optuna] objective = fixed external validation score on oracle_universe")
    print("[Optuna] score = pnl_mean - stronger MDD penalty - stronger CVaR penalty - boundary penalty; no explicit cost penalty")

    def objective(trial: Any) -> float:
        trial_dir = tune_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        if bool(getattr(args, "local_refine", False)):
            # Local refinement around the previously strong dual-LPM region.
            learning_rate = trial.suggest_float("learning_rate", 1.5e-5, 5e-5, log=True)
            gamma = trial.suggest_float("gamma", 0.94, 0.98)
            tau = trial.suggest_float("tau", 0.006, 0.014)
            batch_size = 256
            learning_starts = trial.suggest_categorical("learning_starts", [3000, 5000])
            train_freq = trial.suggest_categorical("train_freq", [2, 4])
            gradient_steps = trial.suggest_categorical("gradient_steps", [2, 4])
            ent_coef = trial.suggest_categorical("ent_coef", ["auto_0.1", "auto_0.2", "0.05", "0.1"])
            h_max = trial.suggest_categorical("h_max", [0.65, 0.7, 0.75])
            delta_abs = trial.suggest_categorical("delta_abs", [0.03, 0.05, 0.08])
        else:
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
            gamma = trial.suggest_float("gamma", 0.90, 0.995)
            tau = trial.suggest_float("tau", 0.003, 0.02, log=True)
            batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
            learning_starts = trial.suggest_categorical("learning_starts", [1000, 3000, 5000, 10000])
            train_freq = trial.suggest_categorical("train_freq", [1, 2, 4])
            gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
            ent_coef = trial.suggest_categorical("ent_coef", ["auto", "auto_0.05", "auto_0.1", "auto_0.2", "0.02", "0.05", "0.1"])
            h_max = trial.suggest_categorical("h_max", [0.6, 0.7, 0.8, 1.0])
            delta_abs = trial.suggest_categorical("delta_abs", [0.03, 0.05, 0.08, 0.10, 0.15])

        if bool(getattr(args, "tune_reward", False)):
            mu_rel = trial.suggest_float("mu_rel", 0.5, 2.0)
            lambda_down = trial.suggest_float("lambda_down", 2.0, 12.0)
            lambda_drawdown = trial.suggest_float("lambda_drawdown", 0.0, 6.0)
            eta_cost_v2 = trial.suggest_float("eta_cost_v2", 1.0, 12.0)
        else:
            mu_rel = float(base_env_cfg.mu_rel)
            lambda_down = float(base_env_cfg.lambda_down)
            lambda_drawdown = float(base_env_cfg.lambda_drawdown)
            eta_cost_v2 = float(base_env_cfg.eta_cost_v2)

        if bool(getattr(args, "tune_reward", False)) and str(getattr(args, "reward_mode", "current")) == "dual_lpm":
            if bool(getattr(args, "local_refine", False)):
                lambda_paper_lpm = trial.suggest_float("lambda_paper_lpm", 1.0, 3.0)
                lambda_net_lpm = trial.suggest_float("lambda_net_lpm", 12.0, 28.0)
            else:
                lambda_paper_lpm = trial.suggest_float("lambda_paper_lpm", 0.1, 8.0, log=True)
                lambda_net_lpm = trial.suggest_float("lambda_net_lpm", 2.0, 40.0)
            # These are inactive in pure dual-LPM reward, but kept as attrs for
            # backward-compatible config/reporting.
            mu_net_upside = float(base_env_cfg.mu_net_upside)
            boundary_penalty = float(base_env_cfg.boundary_penalty)
        else:
            lambda_paper_lpm = float(base_env_cfg.lambda_paper_lpm)
            lambda_net_lpm = float(base_env_cfg.lambda_net_lpm)
            mu_net_upside = float(base_env_cfg.mu_net_upside)
            boundary_penalty = float(base_env_cfg.boundary_penalty)

        env_cfg = EnvConfigSAC(**asdict(base_env_cfg))
        env_cfg.h_max = float(h_max)
        env_cfg.delta_h_bounds = (-float(delta_abs), float(delta_abs))
        env_cfg.sac_action_semantics = str(args.sac_action_semantics)
        env_cfg.mu_rel = float(mu_rel)
        env_cfg.lambda_down = float(lambda_down)
        env_cfg.lambda_drawdown = float(lambda_drawdown)
        env_cfg.eta_cost_v2 = float(eta_cost_v2)
        env_cfg.lambda_paper_lpm = float(lambda_paper_lpm)
        env_cfg.lambda_net_lpm = float(lambda_net_lpm)
        env_cfg.mu_net_upside = float(mu_net_upside)
        env_cfg.boundary_penalty = float(boundary_penalty)

        frames: List[pd.DataFrame] = []
        summaries: List[Dict[str, Any]] = []
        for i, w in enumerate(tune_windows):
            window_out = trial_dir / str(w["window_name"])
            job = {
                **w,
                "cache": str(args.cache),
                "out_dir": str(trial_dir),
                "window_out_dir": str(window_out),
                "scenario_dir": str(args.scenario_dir),
                "exposure_id": str(args.exposure_id),
                "seed": int(args.seed) + trial.number * 10000 + i * 1000,
                "env_cfg": asdict(env_cfg),
                "device": str(args.device),
                "vec": str(args.vec),
                "n_envs": int(args.n_envs),
                "total_timesteps": int(args.trial_timesteps),
                "eval_episodes": int(args.tune_eval_episodes),
                "max_train_scenarios": int(args.max_train_scenarios),
                "max_eval_scenarios": int(args.max_eval_scenarios),
                "learning_rate": float(learning_rate),
                "gamma": float(gamma),
                "tau": float(tau),
                "buffer_size": int(args.buffer_size),
                "batch_size": int(batch_size),
                "learning_starts": int(learning_starts),
                "train_freq": int(train_freq),
                "gradient_steps": int(gradient_steps),
                "ent_coef": str(ent_coef),
                "net_arch": tuple(int(x) for x in args.net_arch),
                "torch_threads": int(args.torch_threads),
                "torch_interop_threads": int(args.torch_interop_threads),
                "eta_print_freq": int(args.eta_print_freq),
                "eval_only": False,
                "live": False,
                "warm_start_model": "",
            }
            res = train_eval_window(job)
            summaries.append(res)
            if not bool(res.get("ok")):
                trial.set_user_attr("failed_window", str(res.get("window_name")))
                trial.set_user_attr("error", str(res.get("error")))
                return -1e18
            pth = window_out / "episodes_all.parquet"
            if pth.exists():
                frames.append(pd.read_parquet(pth))

        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.DataFrame()
        df.to_parquet(trial_dir / "results_all_windows.parquet", index=False)
        pd.DataFrame(summaries).to_csv(trial_dir / "window_summaries.csv", index=False)

        score = _score_trial_from_results(df)
        for k, v in _trial_eval_summary(df).items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("score_mean_val_reward_v2", float(score))
        return float(score)

    study = optuna.create_study(
        direction="maximize",
        study_name=str(args.optuna_study_name),
        storage=(str(args.optuna_storage) if str(args.optuna_storage).strip() else None),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=int(args.n_trials), n_jobs=int(args.optuna_jobs))

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_df.to_csv(out_root / "optuna_trials.csv", index=False)

    best = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "best_trial_number": int(study.best_trial.number),
        "best_trial_user_attrs": dict(study.best_trial.user_attrs),
        "note": "Optuna used a fixed external validation score. Reward coefficients were tuned only if --tune_reward was provided.",
    }
    (out_root / "optuna_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("[Optuna] best trial:", study.best_trial.number)
    print("[Optuna] best value:", study.best_value)
    print("[Optuna] best params:", study.best_params)
    print(f"[Optuna] wrote {out_root / 'optuna_trials.csv'}")
    print(f"[Optuna] wrote {out_root / 'optuna_best.json'}")

    if bool(getattr(args, "run_best_after_tune", False)):
        best_run_dir = out_root / "BEST_FULL_RUN"
        best_run_dir.mkdir(parents=True, exist_ok=True)
        best_params = dict(study.best_params)
        best_env_cfg = EnvConfigSAC(**asdict(base_env_cfg))

        if "h_max" in best_params:
            best_env_cfg.h_max = float(best_params["h_max"])
        if "delta_abs" in best_params:
            da = float(best_params["delta_abs"])
            best_env_cfg.delta_h_bounds = (-da, da)

        best_env_cfg.sac_action_semantics = str(args.sac_action_semantics)
        for attr in [
            "mu_rel", "lambda_down", "lambda_drawdown", "eta_cost_v2",
            "lambda_paper_lpm", "lambda_net_lpm", "mu_net_upside", "boundary_penalty"
        ]:
            if attr in best_params:
                setattr(best_env_cfg, attr, float(best_params[attr]))

        best_windows = windows[:]
        if int(getattr(args, "best_max_windows", 0)) > 0:
            best_windows = best_windows[: int(args.best_max_windows)]

        best_summaries = []
        for i, w in enumerate(best_windows):
            window_out = best_run_dir / str(w["window_name"])
            job = {
                **w,
                "cache": str(args.cache),
                "out_dir": str(best_run_dir),
                "window_out_dir": str(window_out),
                "scenario_dir": str(args.scenario_dir),
                "exposure_id": str(args.exposure_id),
                "seed": int(args.seed) + 900000 + i * 1000,
                "env_cfg": asdict(best_env_cfg),
                "device": str(args.device),
                "vec": str(args.vec),
                "n_envs": int(args.n_envs),
                "total_timesteps": int(args.best_timesteps),
                "eval_episodes": int(args.eval_episodes),
                "max_train_scenarios": int(args.max_train_scenarios),
                "max_eval_scenarios": int(args.max_eval_scenarios),
                "learning_rate": float(best_params.get("learning_rate", args.learning_rate)),
                "gamma": float(best_params.get("gamma", args.gamma)),
                "tau": float(best_params.get("tau", args.tau)),
                "buffer_size": int(args.buffer_size),
                "batch_size": int(best_params.get("batch_size", args.batch_size)),
                "learning_starts": int(best_params.get("learning_starts", args.learning_starts)),
                "train_freq": int(best_params.get("train_freq", args.train_freq)),
                "gradient_steps": int(best_params.get("gradient_steps", args.gradient_steps)),
                "ent_coef": str(best_params.get("ent_coef", args.ent_coef)),
                "net_arch": tuple(int(x) for x in args.net_arch),
                "torch_threads": int(args.torch_threads),
                "torch_interop_threads": int(args.torch_interop_threads),
                "eta_print_freq": int(args.eta_print_freq),
                "eval_only": False,
                "live": False,
                "warm_start_model": "",
            }
            best_summaries.append(train_eval_window(job))

        frames = []
        for w in best_windows:
            pth = best_run_dir / str(w["window_name"]) / "episodes_all.parquet"
            if pth.exists():
                frames.append(pd.read_parquet(pth))

        best_big = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        best_big.to_parquet(best_run_dir / "results_all_windows.parquet", index=False)
        pd.DataFrame(best_summaries).to_csv(best_run_dir / "window_summaries.csv", index=False)
        _write_ai_run_report(best_run_dir, args, best_summaries, best_big)
        print(f"[Optuna] best full run wrote {best_run_dir / 'results_all_windows.parquet'} rows={len(best_big):,}")

    return 0


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    windows = build_walkforward_windows(
        year_start=args.year_start,
        year_end=args.year_end,
        train_mode=args.train_mode,
        train_years_initial=args.train_years_initial,
        expanding_max_years=args.expanding_max_years,
        step_years=args.step_years,
    )
    if int(args.max_windows) > 0:
        windows = windows[: int(args.max_windows)]
    if not windows:
        print("[ERROR] no walk-forward windows generated")
        return 2

    env_cfg = EnvConfigSAC(
        h_max=float(args.h_max),
        naive_hedge_ratio=float(args.naive_hedge_ratio),
        sac_action_semantics=str(args.sac_action_semantics),
        delta_h_bounds=(float(args.delta_h_low), float(args.delta_h_high)),
        min_action_change_threshold=float(args.min_action_change_threshold),
        cost_per_contract_trade_usd=float(args.cost_per_contract_trade_usd),
        use_risk_overlay=bool(args.use_risk_overlay),
        overlay_loss_days_1=int(args.overlay_loss_days_1),
        overlay_loss_days_2=int(args.overlay_loss_days_2),
        overlay_reentry_profit_days=int(args.overlay_reentry_profit_days),
        overlay_reentry_sum_threshold=float(args.overlay_reentry_sum_threshold),
        overlay_cut_1=float(args.overlay_cut_1),
        overlay_cut_2=float(args.overlay_cut_2),
        overlay_profit_slope_days=int(args.overlay_profit_slope_days),
        overlay_profit_boost=float(args.overlay_profit_boost),
        overlay_dd_reduce_threshold=float(args.overlay_dd_reduce_threshold),
        overlay_dd_cut=float(args.overlay_dd_cut),
        overlay_max_daily_abs_dh=float(args.overlay_max_daily_abs_dh),
        overlay_use_dcc_garch=bool(args.overlay_use_dcc_garch),
        overlay_dcc_low=float(args.overlay_dcc_low),
        overlay_dcc_high=float(args.overlay_dcc_high),
        overlay_vol_high_z=float(args.overlay_vol_high_z),
        overlay_hr_blend=float(args.overlay_hr_blend),
        mu_paper=float(args.mu_paper),
        mu_total=float(args.mu_total),
        reward_mode=str(args.reward_mode),
        naive_benchmark_h=float(args.naive_benchmark_h),
        mu_rel=float(args.mu_rel),
        lpm_paper_target=float(args.lpm_paper_target),
        lpm_net_target=float(args.lpm_net_target),
        lambda_paper_lpm=float(args.lambda_paper_lpm),
        lambda_net_lpm=float(args.lambda_net_lpm),
        mu_net_upside=float(args.mu_net_upside),
        boundary_penalty=float(args.boundary_penalty),
        boundary_zone=float(args.boundary_zone),
        lambda_down=float(args.lambda_down),
        lambda_paper_loss=float(args.lambda_paper_loss),
        downside_order=int(args.downside_order),
        dd_threshold=float(args.dd_threshold),
        dd_beta=float(args.dd_beta),
        lambda_drawdown=float(args.lambda_drawdown),
        eta_cost_v2=float(args.eta_cost_v2),
        kappa_stability=float(args.kappa_stability),
        omega_inaction=float(args.omega_inaction),
        alpha_inaction=float(args.alpha_inaction),
        n_loss_max=int(args.n_loss_max),
        dd_panic_threshold=float(args.dd_panic_threshold),
    )

    if bool(args.optuna_tune):
        return run_optuna_tuning(args, windows, env_cfg)

    print(f"[SAC] exposure={args.exposure_id} windows={len(windows)} mode={args.train_mode}")
    print(f"[SAC] cache={args.cache}")
    print(f"[SAC] out_dir={out_root}")
    print(f"[SAC] device={args.device} vec={args.vec} n_envs={args.n_envs} parallel_windows={args.parallel_windows}")

    jobs: List[Dict[str, Any]] = []
    prev_model = ""
    for i, w in enumerate(windows):
        window_out = out_root / str(w["window_name"])
        job = {
            **w,
            "cache": str(args.cache),
            "out_dir": str(out_root),
            "window_out_dir": str(window_out),
            "scenario_dir": str(args.scenario_dir),
            "exposure_id": str(args.exposure_id),
            "seed": int(args.seed) + i * 1000,
            "env_cfg": asdict(env_cfg),
            "device": str(args.device),
            "vec": str(args.vec),
            "n_envs": int(args.n_envs),
            "total_timesteps": int(args.total_timesteps),
            "eval_episodes": int(args.eval_episodes),
            "max_train_scenarios": int(args.max_train_scenarios),
            "max_eval_scenarios": int(args.max_eval_scenarios),
            "learning_rate": float(args.learning_rate),
            "gamma": float(args.gamma),
            "tau": float(args.tau),
            "buffer_size": int(args.buffer_size),
            "batch_size": int(args.batch_size),
            "learning_starts": int(args.learning_starts),
            "train_freq": int(args.train_freq),
            "gradient_steps": int(args.gradient_steps),
            "ent_coef": str(args.ent_coef),
            "net_arch": tuple(int(x) for x in args.net_arch),
            "torch_threads": int(args.torch_threads),
            "torch_interop_threads": int(args.torch_interop_threads),
            "eta_print_freq": int(args.eta_print_freq),
            "eval_only": bool(args.eval_only),
            "live": bool(int(args.parallel_windows) <= 1),
            "warm_start_model": prev_model if int(args.parallel_windows) <= 1 else "",
        }
        jobs.append(job)
        prev_model = str(window_out / "model_best.zip")

    summaries: List[Dict[str, Any]] = []
    t0 = time.time()
    workers = max(1, int(args.parallel_windows))

    if workers == 1:
        for idx, job in enumerate(jobs, start=1):
            print(f"\n[SAC] ({idx}/{len(jobs)}) {job['window_name']}")
            res = train_eval_window(job)
            summaries.append(res)
            print(f"[SAC] done={idx}/{len(jobs)} ok={res.get('ok')} elapsed={_fmt_eta(time.time()-t0)}")
    else:
        print("[SAC] parallel windows enabled; warm-start is disabled for reproducibility.")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(train_eval_window, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                res = fut.result()
                summaries.append(res)
                done += 1
                ok = sum(1 for x in summaries if x.get("ok"))
                fail = sum(1 for x in summaries if not x.get("ok"))
                print(f"[SAC] progress done={done}/{len(jobs)} ok={ok} fail={fail} elapsed={_fmt_eta(time.time()-t0)}")

    pd.DataFrame(summaries).to_csv(out_root / "window_summaries.csv", index=False)

    frames: List[pd.DataFrame] = []
    for w in windows:
        pth = out_root / str(w["window_name"]) / "episodes_all.parquet"
        if pth.exists():
            try:
                frames.append(pd.read_parquet(pth))
            except Exception as e:
                print(f"[WARN] cannot read {pth}: {e}")
    if frames:
        big = pd.concat(frames, ignore_index=True)
    else:
        big = pd.DataFrame()

    big.to_parquet(out_root / "results_all_windows.parquet", index=False)
    _write_ai_run_report(out_root, args, summaries, big)

    print(f"[SAC] wrote {out_root / 'results_all_windows.parquet'} rows={len(big):,}")
    print(f"[SAC] wrote {out_root / 'sac_run_report.md'}")
    print(f"[SAC] wrote {out_root / 'sac_metrics_summary.csv'}")

    if summaries and any(not s.get("ok") for s in summaries):
        print("[SAC] completed with failures; inspect each window train_sac.log")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())