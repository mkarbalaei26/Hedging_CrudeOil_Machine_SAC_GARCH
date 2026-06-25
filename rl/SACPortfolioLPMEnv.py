

"""SAC environment for physical crude-oil hedging with portfolio-LPM reward.

This environment is intentionally simpler than ``env_daily.py``. It is meant as the
first clean SAC experiment:

- Episode = one physical crude trade scenario.
- Step = one trading day.
- Action = delta hedge ratio: h_t = h_{t-1} + delta_h_t.
- Initial hedge ratio is the naive full hedge: h_0 = 1.
- Physical exposure is long crude.
- Futures hedge is short CL contracts.
- Reward penalizes downside risk of the *total portfolio*:
      physical PnL + futures PnL - decision cost
  not futures-leg loss alone.
- Rollover is treated as an exogenous market mechanic. It can be logged as an
  accounting cost, but it is not included in the decision-cost penalty.

Expected precompute keys / attributes:
    dates_int, spot, dS, pnl_1c, roll_flag, tradable, feature_matrix
Optional keys / attributes:
    cl1, cl2, f_mark, feature_names

Scenario mappings may provide either:
    start_date_int, end_date_int, volume_bbl
or:
    start_idx, end_idx, volume_bbl

Date/index convention:
    end_idx / end_date_int is treated as end-exclusive when index-based and as the
    first date *after* the scenario when loaded by date. This matches the previous
    daily environment convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class SACPortfolioLPMConfig:
    """Configuration for the first clean SAC portfolio-LPM environment."""

    # Hedge/action design
    initial_h: float = 1.0
    h_min: float = -0.5
    h_max: float = 3
    delta_h_bounds: Tuple[float, float] = (-0.20, 0.20)
    min_action_change_threshold: float = 0.0
    apply_actions_only_on_tradable_days: bool = True

    # Contract/cost assumptions
    contract_size_bbl: float = 1000.0
    cost_per_contract_trade_usd: float = 10.0
    include_opening_cost_in_equity: bool = True
    include_opening_cost_in_first_reward: bool = False
    log_roll_accounting_cost: bool = True
    penalize_roll_in_reward: bool = False  # should remain False for this project step

    # Reward: normalized three-part episode-aware hedging objective.
    # Components are all dimensionless:
    #   1) downside/loss aversion: change in cumulative-trade LPM below lpm_target
    #   2) pure volatility/risk aversion: running volatility level of portfolio return
    #   3) cost aversion: cumulative agent decision transaction cost / initial notional
    # Default weights follow the project decision: 50% downside, 35% volatility, 15% cost.
    lpm_target: float = 0.0
    lpm_order: int = 2
    reward_weight_lpm: float = 0.50
    reward_weight_volatility: float = 0.35
    reward_weight_decision_cost: float = 0.15

    # Legacy names retained for CLI/backward compatibility. They are not used in
    # the new default reward unless external code explicitly reads them.
    lambda_lpm: float = 1.0
    eta_decision_cost: float = 1.0
    lambda_smooth: float = 0.0
    mu_portfolio_return: float = 0.0

    # Reward normalization / stability
    reward_scale: float = 100.0
    reward_clip: Optional[float] = 50.0
    cost_norm_floor: float = 1.0
    notional_floor: float = 1.0

    # Observation design
    include_position: bool = True
    include_time: bool = True
    include_elapsed: bool = True
    include_equity: bool = True
    include_prices: bool = False
    nan_to_num: float = 0.0

    # Info/log verbosity
    info_mode: str = "train"  # "train" | "eval"


# -----------------------------------------------------------------------------
# Scenario adapter
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    start_date_int: Optional[int] = None
    end_date_int: Optional[int] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    volume_bbl: float = 0.0

    @staticmethod
    def from_mapping(m: Dict[str, Any]) -> "ScenarioSpec":
        if "start_date_int" in m and "end_date_int" in m:
            return ScenarioSpec(
                start_date_int=int(m["start_date_int"]),
                end_date_int=int(m["end_date_int"]),
                volume_bbl=float(m["volume_bbl"]),
            )
        if "start_idx" in m and "end_idx" in m:
            return ScenarioSpec(
                start_idx=int(m["start_idx"]),
                end_idx=int(m["end_idx"]),
                volume_bbl=float(m["volume_bbl"]),
            )
        raise KeyError("Scenario mapping must provide start_date_int/end_date_int or start_idx/end_idx")


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------


class SACPortfolioLPMEnv(gym.Env):
    """Gymnasium-compatible SAC environment with portfolio-LPM reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        pre: Any,
        scenarios: Sequence[Dict[str, Any] | ScenarioSpec],
        *,
        cfg: Optional[SACPortfolioLPMConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required to use SACPortfolioLPMEnv")

        self.cfg = cfg or SACPortfolioLPMConfig()
        self.np_random = np.random.default_rng(seed)

        # Arrays from precompute; supports both object-like bundles and dict/np.load mappings.
        self.dates_int = self._get_array(pre, "dates_int", dtype=np.int64, required=True)
        self.spot = self._get_array(pre, "spot", dtype=np.float64, required=True)
        self.dS = self._get_array(pre, "dS", dtype=np.float64, required=True)
        self.pnl_1c = self._get_array(pre, "pnl_1c", dtype=np.float64, required=True)
        self.roll_flag = self._get_array(pre, "roll_flag", dtype=np.int8, required=True)
        self.tradable = self._get_array(pre, "tradable", dtype=np.int8, required=True)

        X = self._get_array(pre, "feature_matrix", dtype=np.float32, required=False)
        if X is None:
            X = np.zeros((len(self.spot), 0), dtype=np.float32)
        self.X = np.asarray(X, dtype=np.float32)

        self.cl1 = self._get_array(pre, "cl1", dtype=np.float64, required=False)
        self.cl2 = self._get_array(pre, "cl2", dtype=np.float64, required=False)
        self.f_mark = self._get_array(pre, "f_mark", dtype=np.float64, required=False)
        if self.f_mark is None:
            self.f_mark = self.cl1

        self.feature_names = self._get_feature_names(pre)

        n = len(self.spot)
        for name, arr in [
            ("dates_int", self.dates_int),
            ("dS", self.dS),
            ("pnl_1c", self.pnl_1c),
            ("roll_flag", self.roll_flag),
            ("tradable", self.tradable),
        ]:
            if len(arr) != n:
                raise ValueError(f"precompute array length mismatch for {name}: {len(arr)} != {n}")
        if self.X.shape[0] != n:
            raise ValueError(f"feature_matrix row count mismatch: {self.X.shape[0]} != {n}")

        self.scenarios: List[ScenarioSpec] = []
        for s in scenarios:
            if isinstance(s, ScenarioSpec):
                self.scenarios.append(s)
            else:
                self.scenarios.append(ScenarioSpec.from_mapping(dict(s)))
        if not self.scenarios:
            raise ValueError("No scenarios provided")

        # Observation space
        obs_dim = int(self.X.shape[1])
        if self.cfg.include_position:
            obs_dim += 3  # h_prev, n_prev_scaled, abs_n_prev_scaled
        if self.cfg.include_time:
            obs_dim += 3  # frac_remaining, is_roll_day, is_tradable
        if self.cfg.include_elapsed:
            obs_dim += 1
        if self.cfg.include_equity:
            obs_dim += 2  # equity_norm, drawdown_norm
        if self.cfg.include_prices:
            obs_dim += 3  # spot_return_lag/current proxy, basis if f_mark exists, spot level scaled

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        lo, hi = float(self.cfg.delta_h_bounds[0]), float(self.cfg.delta_h_bounds[1])
        if lo > hi:
            raise ValueError(f"delta_h_bounds must be ordered, got {self.cfg.delta_h_bounds}")
        self.action_space = spaces.Box(
            low=np.array([lo], dtype=np.float32),
            high=np.array([hi], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Episode state placeholders
        self._scenario_idx: int = -1
        self._i0: int = 0
        self._i1: int = 0
        self._t_start: int = 0
        self._t_end: int = 0
        self._t: int = 0
        self._Q: float = 0.0
        self._h: float = 0.0
        self._n_prev: int = 0
        self._equity: float = 0.0
        self._peak_equity: float = 0.0
        self._mdd: float = 0.0
        self._opening_cost: float = 0.0
        self._initial_notional: float = 1.0
        self._cum_portfolio_pnl_for_reward: float = 0.0
        self._prev_episode_lpm: float = 0.0
        self._running_volatility_level: float = 0.0
        self._cum_decision_cost_norm: float = 0.0

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_array(pre: Any, name: str, *, dtype: Any, required: bool) -> Optional[np.ndarray]:
        value = None
        if hasattr(pre, name):
            value = getattr(pre, name)
        else:
            try:
                if name in pre:
                    value = pre[name]
            except Exception:
                value = None
        if value is None:
            if required:
                raise KeyError(f"precompute bundle is missing required key/attribute: {name}")
            return None
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def _get_feature_names(pre: Any) -> List[str]:
        value = None
        if hasattr(pre, "feature_names"):
            value = getattr(pre, "feature_names")
        else:
            try:
                if "feature_names" in pre:
                    value = pre["feature_names"]
            except Exception:
                value = None
        if value is None:
            return []
        out: List[str] = []
        for x in np.asarray(value).tolist():
            if isinstance(x, bytes):
                out.append(x.decode("utf-8"))
            else:
                out.append(str(x))
        return out

    # ------------------------------------------------------------------
    # Scenario/date helpers
    # ------------------------------------------------------------------

    def _date_to_index(self, d_int: int) -> int:
        idx = int(np.searchsorted(self.dates_int, int(d_int), side="left"))
        if idx < 0:
            return 0
        if idx > len(self.dates_int):
            return len(self.dates_int)
        return idx

    def _scenario_to_indices(self, sc: ScenarioSpec) -> Tuple[int, int]:
        if sc.start_idx is not None and sc.end_idx is not None:
            i0, i1 = int(sc.start_idx), int(sc.end_idx)
        elif sc.start_date_int is not None and sc.end_date_int is not None:
            i0 = self._date_to_index(int(sc.start_date_int))
            # end is exclusive: first index strictly after / at end date, consistent with previous env convention.
            i1 = self._date_to_index(int(sc.end_date_int))
        else:
            raise ValueError("Scenario must have either date pair or index pair")

        i0 = int(np.clip(i0, 0, len(self.spot) - 1))
        i1 = int(np.clip(i1, i0 + 1, len(self.spot)))
        return i0, i1

    # ------------------------------------------------------------------
    # Contract/cost helpers
    # ------------------------------------------------------------------

    def _contracts_from_h(self, h: float) -> int:
        """Convert hedge ratio to signed CL contracts.

        Convention:
        - physical crude exposure is long;
        - hedge is short CL futures;
        - therefore positive h maps to negative contract count.
        """
        h_clip = float(np.clip(float(h), float(self.cfg.h_min), float(self.cfg.h_max)))
        raw_contracts = h_clip * (float(self._Q) / float(self.cfg.contract_size_bbl))
        return -int(np.rint(raw_contracts))

    def _decision_cost(self, n_old: int, n_new: int) -> float:
        """Cost caused by the agent changing the hedge position."""
        return float(abs(int(n_new) - int(n_old)) * float(self.cfg.cost_per_contract_trade_usd))

    def _roll_accounting_cost(self, n_position: int, roll: bool) -> float:
        """Optional accounting-only roll cost.

        This is not a decision penalty. If logged, it approximates close+open cost on
        the existing position at the mandatory roll date.
        """
        if not bool(roll) or not bool(self.cfg.log_roll_accounting_cost):
            return 0.0
        return float(2 * abs(int(n_position)) * float(self.cfg.cost_per_contract_trade_usd))

    def _notional(self, t: int) -> float:
        px_idx = max(int(t) - 1, 0)
        denom = float(self._Q) * float(self.spot[px_idx])
        if not np.isfinite(denom) or abs(denom) < float(self.cfg.notional_floor):
            denom = float(self.cfg.notional_floor)
        return abs(denom)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _make_obs(self) -> np.ndarray:
        x = np.asarray(self.X[self._t], dtype=np.float32)
        parts: List[np.ndarray] = [x]

        if self.cfg.include_position:
            denom_contracts = max(float(self._Q) / float(self.cfg.contract_size_bbl), 1.0)
            n_scaled = float(self._n_prev) / denom_contracts
            parts.append(np.array([float(self._h), n_scaled, abs(n_scaled)], dtype=np.float32))

        if self.cfg.include_time:
            total = max(int(self._t_end) - int(self._t_start), 1)
            remaining = max(int(self._t_end) - int(self._t), 0)
            frac_remaining = float(remaining) / float(total)
            is_roll = float(self.roll_flag[self._t] != 0)
            is_tradable = float(self.tradable[self._t] != 0)
            parts.append(np.array([frac_remaining, is_roll, is_tradable], dtype=np.float32))

        if self.cfg.include_elapsed:
            total = max(int(self._t_end) - int(self._t_start), 1)
            elapsed = max(int(self._t) - int(self._t_start), 0)
            parts.append(np.array([float(elapsed) / float(total)], dtype=np.float32))

        if self.cfg.include_equity:
            denom = self._notional(self._t)
            equity_norm = float(self._equity) / denom
            drawdown_norm = float(self._equity - self._peak_equity) / denom
            parts.append(np.array([equity_norm, drawdown_norm], dtype=np.float32))

        if self.cfg.include_prices:
            t = int(self._t)
            spot_prev = float(self.spot[t - 1]) if t > 0 else float(self.spot[t])
            spot_ret = float(self.dS[t]) / spot_prev if spot_prev not in (0.0, -0.0) else 0.0
            basis = 0.0
            if self.f_mark is not None and np.isfinite(self.f_mark[t]):
                basis = float(self.spot[t] - self.f_mark[t])
            spot_scaled = float(self.spot[t]) / 100.0
            parts.append(np.array([spot_ret, basis / 100.0, spot_scaled], dtype=np.float32))

        obs = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(
                obs,
                nan=float(self.cfg.nan_to_num),
                posinf=float(self.cfg.nan_to_num),
                neginf=float(self.cfg.nan_to_num),
            ).astype(np.float32, copy=False)
        return obs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if options and "scenario_idx" in options:
            self._scenario_idx = int(options["scenario_idx"])
        else:
            self._scenario_idx = int(self.np_random.integers(0, len(self.scenarios)))
        self._scenario_idx = int(np.clip(self._scenario_idx, 0, len(self.scenarios) - 1))

        sc = self.scenarios[self._scenario_idx]
        self._Q = float(sc.volume_bbl)
        if not np.isfinite(self._Q) or self._Q <= 0:
            raise ValueError(f"Scenario volume_bbl must be positive, got {self._Q}")

        i0, i1 = self._scenario_to_indices(sc)
        if i1 <= i0 + 1:
            raise ValueError("Scenario too short; need at least two daily observations")

        # First tradable day is used to establish the initial naive hedge.
        t0 = int(i0)
        if bool(self.cfg.apply_actions_only_on_tradable_days):
            while t0 < i1 and int(self.tradable[t0]) == 0:
                t0 += 1
        if t0 >= i1 - 1:
            t0 = int(i0)

        self._i0 = int(i0)
        self._i1 = int(i1)
        self._t_start = int(min(t0 + 1, i1 - 1))
        self._t_end = int(i1)
        self._t = int(self._t_start)

        self._h = float(np.clip(float(self.cfg.initial_h), float(self.cfg.h_min), float(self.cfg.h_max)))
        self._n_prev = int(self._contracts_from_h(self._h))

        self._initial_notional = max(float(self._notional(self._t_start)), float(self.cfg.notional_floor))
        self._cum_portfolio_pnl_for_reward = 0.0
        self._prev_episode_lpm = 0.0
        self._running_volatility_level = 0.0
        self._cum_decision_cost_norm = 0.0
        self._equity = 0.0
        self._peak_equity = 0.0
        self._mdd = 0.0
        self._opening_cost = self._decision_cost(0, self._n_prev)
        if bool(self.cfg.include_opening_cost_in_equity):
            self._equity -= float(self._opening_cost)
            self._peak_equity = self._equity
            self._mdd = min(0.0, self._equity - self._peak_equity)

        obs = self._make_obs()
        info = {
            "scenario_idx": int(self._scenario_idx),
            "Q": float(self._Q),
            "start_idx": int(self._i0),
            "end_idx": int(self._i1),
            "t_start": int(self._t_start),
            "t_end": int(self._t_end),
            "initial_h": float(self._h),
            "initial_n": int(self._n_prev),
            "opening_cost": float(self._opening_cost),
        }
        return obs, info

    def step(self, action):
        # Current day index. PnL uses the position carried into this day, N_{t-1}.
        # The action chosen at the end of this step determines N_t for the next move.
        t = int(self._t)
        h_before = float(self._h)
        n_used_for_pnl = int(self._n_prev)
        n_before_decision = int(self._n_prev)

        # PnL from the position that was already in place.
        pnl_phys = float(self._Q * float(self.dS[t]))
        pnl_fut = float(n_used_for_pnl * float(self.pnl_1c[t]))

        # Translate SAC action to delta-h.
        dh = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        lo, hi = float(self.cfg.delta_h_bounds[0]), float(self.cfg.delta_h_bounds[1])
        dh = float(np.clip(dh, lo, hi))
        if abs(dh) < float(self.cfg.min_action_change_threshold):
            dh = 0.0

        # On non-tradable days, the agent cannot rebalance. This avoids paying costs
        # on forward-filled CL prices.
        if bool(self.cfg.apply_actions_only_on_tradable_days) and int(self.tradable[t]) == 0:
            dh_effective = 0.0
            h_after = h_before
        else:
            dh_effective = float(dh)
            h_after = float(np.clip(h_before + dh_effective, float(self.cfg.h_min), float(self.cfg.h_max)))

        n_after = int(self._contracts_from_h(h_after))

        # Costs are explicitly split.
        decision_cost = self._decision_cost(n_before_decision, n_after)
        roll = bool(self.roll_flag[t] != 0)
        roll_accounting_cost = self._roll_accounting_cost(n_used_for_pnl, roll)

        reward_cost = float(decision_cost)
        if bool(self.cfg.penalize_roll_in_reward):
            reward_cost += float(roll_accounting_cost)

        # Total portfolio PnL for reward. This is the key fix: LPM is computed on
        # physical + futures as one hedged portfolio, not on the futures leg alone.
        portfolio_pnl_for_reward = float(pnl_phys + pnl_fut - reward_cost)
        portfolio_pnl_accounting = float(pnl_phys + pnl_fut - decision_cost - roll_accounting_cost)

        if bool(self.cfg.include_opening_cost_in_first_reward) and t == int(self._t_start):
            portfolio_pnl_for_reward -= float(self._opening_cost)
            portfolio_pnl_accounting -= float(self._opening_cost)

        denom = self._notional(t)
        portfolio_return = float(portfolio_pnl_for_reward / denom)
        decision_cost_norm = float(decision_cost / max(denom, float(self.cfg.cost_norm_floor)))

        # Episode-aware normalized components.
        # LPM is computed on cumulative trade return, so recovery from losses can
        # reduce the LPM penalty. Volatility and cost remain memory-based risk terms.
        self._cum_portfolio_pnl_for_reward += float(portfolio_pnl_for_reward)
        cumulative_portfolio_return = float(self._cum_portfolio_pnl_for_reward / max(self._initial_notional, float(self.cfg.notional_floor)))

        diff = float(self.cfg.lpm_target) - cumulative_portfolio_return
        downside = max(diff, 0.0)
        order = int(self.cfg.lpm_order)
        if order <= 1:
            episode_lpm = float(downside)
        else:
            episode_lpm = float(downside ** float(order))
        delta_lpm = float(episode_lpm - self._prev_episode_lpm)
        lpm = float(episode_lpm)

        self._running_volatility_level += float(portfolio_return * portfolio_return)
        volatility_penalty = float(self._running_volatility_level)

        self._cum_decision_cost_norm += float(decision_cost / max(self._initial_notional, float(self.cfg.cost_norm_floor)))
        cumulative_decision_cost_norm = float(self._cum_decision_cost_norm)

        # New normalized three-part reward:
        # 50% cumulative-LPM improvement/worsening + 35% running volatility level + 15% cumulative decision cost.
        smooth_pen = 0.0

        raw_w_lpm = max(float(getattr(self.cfg, "reward_weight_lpm", 0.50)), 0.0)
        raw_w_vol = max(float(getattr(self.cfg, "reward_weight_volatility", 0.35)), 0.0)
        raw_w_cost = max(float(getattr(self.cfg, "reward_weight_decision_cost", 0.15)), 0.0)
        weight_sum = raw_w_lpm + raw_w_vol + raw_w_cost
        if weight_sum <= 0.0:
            w_lpm, w_vol, w_cost = 0.50, 0.35, 0.15
        else:
            w_lpm = raw_w_lpm / weight_sum
            w_vol = raw_w_vol / weight_sum
            w_cost = raw_w_cost / weight_sum

        # Combine level + change in LPM to avoid the "freezing in loss" problem:
        # - episode_lpm keeps penalizing staying in a bad cumulative loss state
        # - delta_lpm rewards recovery and penalizes further deterioration
        lpm_combined = 0.70 * float(episode_lpm) + 0.30 * float(delta_lpm)
        reward_component_lpm = float(w_lpm * lpm_combined)
        reward_component_volatility = float(w_vol * volatility_penalty)
        reward_component_decision_cost = float(w_cost * cumulative_decision_cost_norm)

        reward_raw = -(
            reward_component_lpm
            + reward_component_volatility
            + reward_component_decision_cost
        )
        reward = reward_raw * float(self.cfg.reward_scale)
        if self.cfg.reward_clip is not None:
            reward = float(np.clip(reward, -float(self.cfg.reward_clip), float(self.cfg.reward_clip)))
        else:
            reward = float(reward)

        self._prev_episode_lpm = float(episode_lpm)

        # Equity path uses accounting PnL so evaluation can show roll cost separately.
        self._equity += portfolio_pnl_accounting
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity
        dd = float(self._equity - self._peak_equity)
        if dd < self._mdd:
            self._mdd = dd

        # Commit decision for next day.
        self._h = float(h_after)
        self._n_prev = int(n_after)

        terminated = bool(t >= (int(self._t_end) - 1))
        truncated = False
        if not terminated:
            self._t = int(t + 1)

        obs = self._make_obs()

        info_train = {
            "pnl_net": float(portfolio_pnl_accounting),
            "reward": float(reward),
            "h": float(self._h),
            "n_prev": int(self._n_prev),
            "cost": float(decision_cost),
            "mdd": float(self._mdd),
        }

        if str(self.cfg.info_mode).lower() != "eval":
            return obs, reward, terminated, truncated, info_train

        spot_t = float(self.spot[t])
        f_mark_t = float(self.f_mark[t]) if self.f_mark is not None and np.isfinite(self.f_mark[t]) else np.nan
        basis_t = float(spot_t - f_mark_t) if np.isfinite(f_mark_t) else np.nan

        info_eval = {
            **info_train,
            "scenario_idx": int(self._scenario_idx),
            "t": int(t),
            "date_int": int(self.dates_int[t]),
            "day_in_episode": int(t - self._t_start),
            "Q": float(self._Q),
            "spot": spot_t,
            "f_mark": f_mark_t,
            "basis": basis_t,
            "roll_flag": int(self.roll_flag[t]),
            "tradable": int(self.tradable[t]),
            "h_before": float(h_before),
            "action_delta_h_raw": float(dh),
            "action_delta_h_effective": float(dh_effective),
            "h_after": float(h_after),
            "n_used_for_pnl": int(n_used_for_pnl),
            "n_before_decision": int(n_before_decision),
            "n_after": int(n_after),
            "delta_n": int(n_after - n_before_decision),
            "pnl_phys": float(pnl_phys),
            "pnl_fut": float(pnl_fut),
            "pnl_1c": float(self.pnl_1c[t]),
            "decision_cost": float(decision_cost),
            "roll_accounting_cost": float(roll_accounting_cost),
            "portfolio_pnl_for_reward": float(portfolio_pnl_for_reward),
            "portfolio_pnl_accounting": float(portfolio_pnl_accounting),
            "portfolio_return": float(portfolio_return),
            "cumulative_portfolio_return": float(cumulative_portfolio_return),
            "decision_cost_norm": float(decision_cost_norm),
            "cumulative_decision_cost_norm": float(cumulative_decision_cost_norm),
            "lpm": float(lpm),
            "episode_lpm": float(episode_lpm),
            "delta_lpm": float(delta_lpm),
            "volatility_penalty": float(volatility_penalty),
            "running_volatility_level": float(self._running_volatility_level),
            "smooth_pen": float(smooth_pen),
            "reward_raw": float(reward_raw),
            "reward_weight_lpm": float(w_lpm),
            "reward_weight_volatility": float(w_vol),
            "reward_weight_decision_cost": float(w_cost),
            "reward_component_lpm": float(reward_component_lpm),
            "reward_component_volatility": float(reward_component_volatility),
            "reward_component_decision_cost": float(reward_component_decision_cost),
            "equity": float(self._equity),
            "peak_equity": float(self._peak_equity),
            "drawdown": float(self._equity - self._peak_equity),
            "mdd": float(self._mdd),
            "opening_cost": float(self._opening_cost),
        }
        return obs, reward, terminated, truncated, info_eval


# -----------------------------------------------------------------------------
# Factory helper
# -----------------------------------------------------------------------------


def make_sac_portfolio_lpm_env(
    pre: Any,
    scenarios: Sequence[Dict[str, Any] | ScenarioSpec],
    cfg: Optional[SACPortfolioLPMConfig] = None,
    seed: Optional[int] = None,
):
    """Return a thunk for SB3 DummyVecEnv/SubprocVecEnv."""

    def _thunk():
        return SACPortfolioLPMEnv(pre=pre, scenarios=scenarios, cfg=cfg, seed=seed)

    return _thunk


# Backward-friendly aliases for shorter imports if needed.
EnvConfig = SACPortfolioLPMConfig
OilHedgingSACPortfolioLPMEnv = SACPortfolioLPMEnv
make_env = make_sac_portfolio_lpm_env