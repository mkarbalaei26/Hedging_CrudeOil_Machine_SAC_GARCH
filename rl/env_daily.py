"""Daily-step RL environment for crude oil hedging with CL front-month futures.

Design goals (project-aligned)
------------------------------
- Episode = one physical trading scenario (Q barrels, start_date..end_date)
- Step = one trading day
- Hedge instrument: CL front-month (roll-aware series precomputed)
- Mandatory roll: enforced via precomputed `roll_flag` and roll-aware `pnl_1c`.
- Costs: tick-based proxy via `ExecutionCostModel` (per-contract per-trade cost).
- Tradable days: action ignored on days where CL1 settlement did not update
  (`tradable[t]==0`). This prevents re-hedging on forward-filled prices.

This environment is intentionally minimal and fast: it consumes NumPy arrays produced
by `rl/precompute.py` and scenario specs produced by `Scenario_Generator.py`.

Reward
------
Default daily reward is a risk-aware utility on normalized PnL:
    r_t = pnl_net_t / (Q * spot_{t-1})
    R_t = mu_pnl * r_t - lambda_rollvar * Var_L(r) - lambda_lpm * LPM(r_t) - eta_cost * (cost_t / (Q * spot_{t-1}))
    Var_L(r) is rolling variance over the last L steps inside the episode (default L=20).
    LPM uses max(0, target - r_t)^p.

NOTE: This module avoids any look-ahead. Observations at time t must only use
information available up to t (inclusive). The precompute pipeline must ensure
features are causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None

from cost_model import ExecutionCostConfig, ExecutionCostModel


# -------------------------
# Config
# -------------------------


@dataclass
class EnvConfig:
    # Position limits
    h_max: float = 2.0  # allow up to 200% hedge (conservative cap)
    delta_h_max: float = 0.10
    delta_h_step: float = 0.05

    # Action design (locked)
    # We learn a delta around a base hedge (naive) for faster, safer convergence.
    action_mode: str = "delta_h_continuous"  # locked
    base_policy: str = "naive"              # locked
    naive_hedge_ratio: float = 1.0           # locked (full hedge baseline)
    delta_h_bounds: Tuple[float, float] = (-2.0, 2.0)  # action bounds for delta_h
    min_action_change_threshold: float = 0.02          # ignore micro-churn
    apply_actions_only_on_tradable_days: bool = True

    # Optional alternative (kept for legacy experiments; not used when action_mode==delta_h_continuous)
    # Optional explicit grid. If None, the grid is generated from delta_h_max/delta_h_step.
    delta_h_grid: Optional[Tuple[float, ...]] = None
    h_levels: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)

    # Costs
    cost_per_contract_trade_usd: float = 10.0

    # Reward / risk
    # risk_mode:
    #   "none"        : reward = mu_pnl*r - eta_cost*c
    #   "quad"        : penalize r^2 (proxy)
    #   "lpm"         : penalize downside only (LPM)
    #   "rollvar_lpm" : penalize rolling variance (L) + downside LPM
    risk_mode: str = "rollvar_lpm"

    # Small incentive for profit (keep small for hedging)
    mu_pnl: float = 0.10

    # Symmetric quadratic proxy (legacy)
    lambda_var: float = 1.0  # used when risk_mode == "quad" (penalize r^2)

    # Rolling variance penalty
    lambda_rollvar: float = 5.0
    roll_var_L: int = 20

    # Lower Partial Moment (downside risk) penalty
    lambda_lpm: float = 2.0
    lpm_order: int = 2        # 1 or 2 are typical
    lpm_target: float = 0.0   # target return (0 = downside only)

    # Cost penalty
    # eta_cost should be interpretable across datasets/runs; reward uses standardized components (see norm_mode)
    eta_cost: float = 1.0
    cost_penalty_mult: float = 1.0

    # Reward component standardization (no look-ahead)
    #   "none" : use raw normalized return r and c_norm
    #   "ewma" : divide each component by an EWMA scale estimated online within the episode
    norm_mode: str = "ewma"          # "none" | "ewma"
    norm_ewma_alpha: float = 0.01    # smoothing for online scale estimates
    norm_eps: float = 1e-8
    norm_clip: float = 10.0          # clip standardized components to avoid rare blow-ups

    # Initial scales (priors) to keep early steps stable
    norm_init_r: float = 0.01        # ~1% daily notional move prior
    norm_init_cost: float = 1e-5     # cost as fraction of notional prior
    norm_init_var: float = 1e-6      # rolling var prior
    norm_init_lpm: float = 1e-6      # lpm prior

    # Info verbosity (train should be minimal for speed)
    info_mode: str = "train"  # "train" | "eval"

    # Observation
    include_position: bool = True
    include_time: bool = True
    include_equity: bool = True
    include_elapsed: bool = True
    nan_to_num: float = 0.0


# -------------------------
# Scenario adapter
# -------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    """Minimal scenario spec used by the RL env.

    Supports either date-based specs (start_date_int/end_date_int) or index-based
    specs (start_idx/end_idx). Exactly one pair must be provided.
    """

    # One of these pairs must be present
    start_date_int: Optional[int] = None
    end_date_int: Optional[int] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

    volume_bbl: float = 0.0

    @staticmethod
    def from_mapping(m: Dict[str, Any]) -> "ScenarioSpec":
        # Accept either date-int or index mapping.
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


# -------------------------
# Core env
# -------------------------


class OilHedgingDailyEnv(gym.Env):
    """Gymnasium-compatible daily hedging environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        pre: Any,
        scenarios: List[Dict[str, Any]],
        *,
        cfg: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required to use OilHedgingDailyEnv")

        self.cfg = cfg or EnvConfig()

        # Arrays from precompute (support both dict-like bundles and PrecomputeResult)
        if hasattr(pre, "dates_int"):
            # PrecomputeResult
            self.dates_int = np.asarray(pre.dates_int, dtype=np.int64)
            self.spot = np.asarray(pre.spot, dtype=np.float64)
            self.dS = np.asarray(pre.dS, dtype=np.float64)
            self.pnl_1c = np.asarray(pre.pnl_1c, dtype=np.float64)  # USD PnL per +1 contract
            self.roll_flag = np.asarray(pre.roll_flag, dtype=np.int8)
            self.tradable = np.asarray(pre.tradable, dtype=np.int8)
            self.X = np.asarray(getattr(pre, "feature_matrix", np.zeros((len(self.spot), 0), dtype=np.float32)), dtype=np.float32)
        else:
            # Mapping (dict / np.load)
            self.dates_int = np.asarray(pre["dates_int"], dtype=np.int64)
            self.spot = np.asarray(pre["spot"], dtype=np.float64)
            self.dS = np.asarray(pre.get("dS"), dtype=np.float64)
            self.pnl_1c = np.asarray(pre["pnl_1c"], dtype=np.float64)  # USD PnL per +1 contract
            self.roll_flag = np.asarray(pre["roll_flag"], dtype=np.int8)
            self.tradable = np.asarray(pre["tradable"], dtype=np.int8)
            self.X = np.asarray(pre.get("feature_matrix", np.zeros((len(self.spot), 0), dtype=np.float32)), dtype=np.float32)

        if len(self.dates_int) != len(self.spot):
            raise ValueError("precompute arrays length mismatch")
        if self.X.shape[0] != len(self.spot):
            raise ValueError("feature_matrix row count mismatch")

        # Build scenario list (convert to internal minimal spec)
        self.scenarios: List[ScenarioSpec] = []
        for s in scenarios:
            if isinstance(s, ScenarioSpec):
                self.scenarios.append(s)
            else:
                self.scenarios.append(ScenarioSpec.from_mapping(s))
        if not self.scenarios:
            raise ValueError("No scenarios provided")

        # Cost model
        self.cost_model = ExecutionCostModel(ExecutionCostConfig(cost_per_contract_trade_usd=float(self.cfg.cost_per_contract_trade_usd)))

        # RNG
        self.np_random = np.random.default_rng(seed)

        # Observation space
        obs_dim = int(self.X.shape[1])
        if self.cfg.include_position:
            obs_dim += 2  # h_prev, n_prev (scaled)
        if self.cfg.include_time:
            obs_dim += 2  # frac_remaining, is_roll_day
        if getattr(self.cfg, "include_elapsed", True):
            obs_dim += 1  # frac_elapsed
        if getattr(self.cfg, "include_equity", True):
            obs_dim += 2  # equity_norm, drawdown_norm

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action space
        if self.cfg.action_mode == "delta_h_continuous":
            lo, hi = float(self.cfg.delta_h_bounds[0]), float(self.cfg.delta_h_bounds[1])
            self.action_space = spaces.Box(low=np.array([lo], dtype=np.float32), high=np.array([hi], dtype=np.float32), shape=(1,), dtype=np.float32)
        elif self.cfg.action_mode == "delta_h_discrete":
            if getattr(self.cfg, "delta_h_grid", None) is not None:
                self._delta_grid = np.array(self.cfg.delta_h_grid, dtype=np.float32)
            else:
                # Configurable discrete hedge-ratio adjustment grid.
                # Example: delta_h_max=0.80 and delta_h_step=0.05 gives actions
                # from -0.80 to +0.80; the resulting hedge ratio is still clipped by h_max.
                dh_max = float(getattr(self.cfg, "delta_h_max", 0.10))
                dh_step = float(getattr(self.cfg, "delta_h_step", 0.05))
                if dh_step <= 0:
                    raise ValueError(f"delta_h_step must be positive, got {dh_step}")
                if dh_max < 0:
                    raise ValueError(f"delta_h_max must be non-negative, got {dh_max}")
                n_grid_steps = int(np.floor(dh_max / dh_step + 1e-9))
                self._delta_grid = (
                    np.arange(-n_grid_steps, n_grid_steps + 1, dtype=np.float32) * np.float32(dh_step)
                ).astype(np.float32)
                if self._delta_grid.size == 0 or not np.any(np.isclose(self._delta_grid, 0.0)):
                    self._delta_grid = np.sort(
                        np.unique(np.append(self._delta_grid, np.float32(0.0)))
                    ).astype(np.float32)
            self.action_space = spaces.Discrete(len(self._delta_grid))
        elif self.cfg.action_mode == "h_levels":
            self._h_levels = np.array(self.cfg.h_levels, dtype=np.float32)
            self.action_space = spaces.Discrete(len(self._h_levels))
        else:
            raise ValueError(f"Unknown action_mode: {self.cfg.action_mode}")

        # Episode state
        self._scenario_idx: int = -1
        self._i0: int = 0
        self._i1: int = 0
        self._t: int = 0
        self._Q: float = 0.0

        self._t_start: int = 0
        self._t_end: int = 0
        self._t0_hedge_day: int = 0

        self._h: float = 0.0
        self._n_prev: int = 0
        self._equity: float = 0.0
        self._peak_equity: float = 0.0
        self._mdd: float = 0.0

        # Rolling return buffer for Var_L within an episode
        self._rbuf: List[float] = []

        # Online scales for reward standardization (reset each episode)
        # Initialized here so attributes exist even before the first reset().
        self._scale_r: float = float(getattr(self.cfg, "norm_init_r", 0.01))
        self._scale_cost: float = float(getattr(self.cfg, "norm_init_cost", 1e-5))
        self._scale_var: float = float(getattr(self.cfg, "norm_init_var", 1e-6))
        self._scale_lpm: float = float(getattr(self.cfg, "norm_init_lpm", 1e-6))

    # -------------------------
    # Helpers
    # -------------------------

    def _date_to_index(self, d_int: int) -> int:
        """Map date int to array index (exact match required)."""
        # dates_int is sorted
        pos = int(np.searchsorted(self.dates_int, d_int))
        if pos < 0 or pos >= len(self.dates_int) or int(self.dates_int[pos]) != int(d_int):
            raise ValueError(f"Scenario date not found in precompute dates_int: {d_int}")
        return pos

    def _scenario_to_indices(self, sc: ScenarioSpec) -> Tuple[int, int]:
        if sc.start_idx is not None and sc.end_idx is not None:
            return int(sc.start_idx), int(sc.end_idx)
        if sc.start_date_int is not None and sc.end_date_int is not None:
            return self._date_to_index(int(sc.start_date_int)), self._date_to_index(int(sc.end_date_int))
        raise ValueError("ScenarioSpec missing start/end")

    def _contracts_from_h(self, h: float) -> int:
        """Convert hedge ratio to integer CL contracts.

        Convention: physical is long => hedge is short futures => N is negative.
        """
        h = float(np.clip(h, -float(self.cfg.h_max), float(self.cfg.h_max)))
        n = int(np.rint(h * (self._Q / 1000.0)))
        return -n

    def _make_obs(self) -> np.ndarray:
        x = self.X[self._t].astype(np.float32, copy=False)
        parts: List[np.ndarray] = [x]

        if self.cfg.include_position:
            # scale n_prev by (Q/1000) to keep magnitudes stable across scenario sizes
            denom = max(self._Q / 1000.0, 1.0)
            pos_vec = np.array([self._h, float(self._n_prev) / denom], dtype=np.float32)
            parts.append(pos_vec)

        if self.cfg.include_time:
            total = max(self._t_end - self._t_start, 1)
            remaining = max(self._t_end - self._t, 0)
            frac_remaining = float(remaining) / float(total)
            is_roll = float(self.roll_flag[self._t] != 0)
            t_vec = np.array([frac_remaining, is_roll], dtype=np.float32)
            parts.append(t_vec)

        if getattr(self.cfg, "include_elapsed", True):
            total = max(self._t_end - self._t_start, 1)
            elapsed = max(self._t - self._t_start, 0)
            frac_elapsed = float(elapsed) / float(total)
            parts.append(np.array([frac_elapsed], dtype=np.float32))

        if getattr(self.cfg, "include_equity", True):
            # Normalize by scenario notional at previous spot to keep scale stable.
            t = self._t
            if t > 0:
                denom = float(self._Q * float(self.spot[t - 1]))
            else:
                denom = float(self._Q * float(self.spot[t]))
            if denom == 0.0 or not np.isfinite(denom):
                denom = 1.0
            equity_norm = float(self._equity) / denom
            drawdown_norm = float(self._equity - self._peak_equity) / denom
            parts.append(np.array([equity_norm, drawdown_norm], dtype=np.float32))

        obs = np.concatenate(parts, axis=0).astype(np.float32, copy=False)

        # sanitize nans
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=float(self.cfg.nan_to_num), posinf=float(self.cfg.nan_to_num), neginf=float(self.cfg.nan_to_num)).astype(
                np.float32, copy=False
            )
        return obs

    # -------------------------
    # Gym API
    # -------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Sample scenario
        self._scenario_idx = int(self.np_random.integers(0, len(self.scenarios)))
        sc = self.scenarios[self._scenario_idx]

        self._Q = float(sc.volume_bbl)

        i0, i1 = self._scenario_to_indices(sc)
        # `i1` is treated as end-exclusive in this env.
        if i1 <= i0 + 1:
            raise ValueError("Scenario too short (need at least 2 indices: start..end)")

        self._i0 = i0
        self._i1 = i1
        self._t_end = i1

        # Choose the first day we can establish/adjust hedge
        t0 = i0
        if bool(getattr(self.cfg, "apply_actions_only_on_tradable_days", True)):
            while t0 < i1 and int(self.tradable[t0]) == 0:
                t0 += 1
        self._t0_hedge_day = int(t0)

        # Effective episode start is the day AFTER hedge establishment.
        # Ensure at least one step exists (t_start < t_end).
        self._t_start = int(min(t0 + 1, i1 - 1))
        self._t = int(self._t_start)

        # Position state: start from Naive hedge (locked)
        h0 = float(getattr(self.cfg, "naive_hedge_ratio", 1.0))
        self._h = float(np.clip(h0, -float(self.cfg.h_max), float(self.cfg.h_max)))
        self._n_prev = int(self._contracts_from_h(self._h))

        # Equity path (charge opening cost once at hedge establishment)
        self._equity = 0.0
        self._peak_equity = 0.0
        self._mdd = 0.0

        open_cost_out = self.cost_model.total_cost(0, self._n_prev, roll_flag=False)
        open_cost_breakdown = None
        if isinstance(open_cost_out, dict):
            open_cost_breakdown = dict(open_cost_out)
            if "cost_total" in open_cost_out:
                open_cost = float(open_cost_out["cost_total"])
            elif "total_cost" in open_cost_out:
                open_cost = float(open_cost_out["total_cost"])
            elif "total" in open_cost_out:
                open_cost = float(open_cost_out["total"])
            else:
                # Fallback: sum numeric values
                open_cost = float(
                    sum(float(v) for v in open_cost_out.values() if isinstance(v, (int, float, np.floating)))
                )
        else:
            open_cost = float(open_cost_out)
        self._equity -= open_cost
        self._peak_equity = self._equity

        self._rbuf = []
        # Reset online normalization scales per episode (no leakage)
        self._scale_r = float(getattr(self.cfg, "norm_init_r", 0.01))
        self._scale_cost = float(getattr(self.cfg, "norm_init_cost", 1e-5))
        self._scale_var = float(getattr(self.cfg, "norm_init_var", 1e-6))
        self._scale_lpm = float(getattr(self.cfg, "norm_init_lpm", 1e-6))

        obs = self._make_obs()
        info = {
            "scenario_idx": self._scenario_idx,
            "Q": self._Q,
            "i0": int(i0),
            "i1": int(i1),
            "t0_hedge_day": int(self._t0_hedge_day),
            "t_start": int(self._t_start),
            "t_end": int(self._t_end),
            "h0": float(self._h),
            "n0": int(self._n_prev),
            "opening_cost": float(open_cost),
            "opening_cost_breakdown": open_cost_breakdown,
        }
        return obs, info

    def step(self, action):
        # Current day index is self._t; PnL uses N_{t-1} on the move from t-1 -> t.
        # To keep consistency, we apply action to set N_t (for future moves), then compute today's PnL using N_prev.

        t = self._t
        n_used_for_pnl = int(self._n_prev)
        done = False

        # Determine new hedge ratio
        if self.cfg.action_mode == "delta_h_continuous":
            # SB3 passes ndarray shape (1,) for Box
            dh = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
            lo, hi = float(self.cfg.delta_h_bounds[0]), float(self.cfg.delta_h_bounds[1])
            dh = float(np.clip(dh, lo, hi))
            # micro-churn filter
            if abs(dh) < float(self.cfg.min_action_change_threshold):
                dh = 0.0
            h_new = float(self.cfg.naive_hedge_ratio + dh)
            h_new = float(np.clip(h_new, -float(self.cfg.h_max), float(self.cfg.h_max)))
        elif self.cfg.action_mode == "delta_h_discrete":
            dh = float(self._delta_grid[int(action)])
            h_new = float(np.clip(self._h + dh, -float(self.cfg.h_max), float(self.cfg.h_max)))
        else:  # h_levels
            h_new = float(np.clip(float(self._h_levels[int(action)]), -float(self.cfg.h_max), float(self.cfg.h_max)))

        # If not tradable, ignore re-hedge
        if bool(getattr(self.cfg, "apply_actions_only_on_tradable_days", True)) and int(self.tradable[t]) == 0:
            h_new = self._h

        # Convert to integer contracts
        n_new = self._contracts_from_h(h_new)

        # Costs (includes roll-cost on N_prev; cost_model has been patched)
        roll = bool(self.roll_flag[t] != 0)
        c_out = self.cost_model.total_cost(self._n_prev, n_new, roll_flag=roll)

        cost_breakdown = None
        if isinstance(c_out, dict):
            cost_breakdown = dict(c_out)
            if "cost_total" in c_out:
                cost_t = float(c_out["cost_total"])
            elif "total_cost" in c_out:
                cost_t = float(c_out["total_cost"])
            elif "total" in c_out:
                cost_t = float(c_out["total"])
            else:
                # Fallback: sum numeric values
                cost_t = float(
                    sum(float(v) for v in c_out.values() if isinstance(v, (int, float, np.floating)))
                )
        else:
            cost_t = float(c_out)

        # PnL components
        dS_t = float(self.dS[t])
        pnl_phys = float(self._Q * dS_t)
        pnl_fut = float(self._n_prev * float(self.pnl_1c[t]))

        pnl_net = pnl_phys + pnl_fut - cost_t

        # Normalize by notional (Q*spot_{t-1}). Use spot[t-1] if available; else spot[t].
        if t > 0:
            denom = float(self._Q * float(self.spot[t - 1]))
        else:
            denom = float(self._Q * float(self.spot[t]))
        if denom == 0.0 or not np.isfinite(denom):
            denom = 1.0

        r = pnl_net / denom
        c_norm = cost_t / denom
        # --- Risk / reward components ---
        mu = float(getattr(self.cfg, "mu_pnl", 0.10))

        # Update rolling buffer for rolling-variance modes
        L = int(getattr(self.cfg, "roll_var_L", 20))
        if L < 2:
            L = 2
        self._rbuf.append(float(r))
        if len(self._rbuf) > L:
            self._rbuf = self._rbuf[-L:]

        # Rolling variance over last L returns inside the episode
        if len(self._rbuf) >= 2:
            var_L = float(np.var(np.asarray(self._rbuf, dtype=np.float64), ddof=1))
        else:
            var_L = 0.0

        # Lower Partial Moment around target: max(0, target - r)^p
        diff = float(getattr(self.cfg, "lpm_target", 0.0)) - float(r)
        downside = max(diff, 0.0)
        p = int(getattr(self.cfg, "lpm_order", 2))
        if p <= 1:
            lpm = downside
        else:
            lpm = downside ** float(p)

        # ---- Standardize components (online, no look-ahead) ----
        norm_mode = str(getattr(self.cfg, "norm_mode", "ewma"))
        eps = float(getattr(self.cfg, "norm_eps", 1e-8))
        clipv = float(getattr(self.cfg, "norm_clip", 10.0))

        if norm_mode == "ewma":
            a = float(getattr(self.cfg, "norm_ewma_alpha", 0.01))
            a = float(np.clip(a, 1e-4, 0.25))

            # Use absolute magnitude as scale target
            self._scale_r = (1.0 - a) * float(self._scale_r) + a * float(abs(r))
            self._scale_cost = (1.0 - a) * float(self._scale_cost) + a * float(abs(c_norm))
            self._scale_var = (1.0 - a) * float(self._scale_var) + a * float(abs(var_L))
            self._scale_lpm = (1.0 - a) * float(self._scale_lpm) + a * float(abs(lpm))

            r_n = float(r) / (float(self._scale_r) + eps)
            c_n = float(c_norm) / (float(self._scale_cost) + eps)
            var_n = float(var_L) / (float(self._scale_var) + eps)
            lpm_n = float(lpm) / (float(self._scale_lpm) + eps)

            # Clip to keep reward numerically stable
            r_n = float(np.clip(r_n, -clipv, clipv))
            c_n = float(np.clip(c_n, -clipv, clipv))
            var_n = float(np.clip(var_n, 0.0, clipv))
            lpm_n = float(np.clip(lpm_n, 0.0, clipv))
        else:
            # "none": keep original magnitudes
            r_n, c_n, var_n, lpm_n = float(r), float(c_norm), float(var_L), float(lpm)

        # Risk penalty by mode (applied on standardized components)
        rm = str(getattr(self.cfg, "risk_mode", "rollvar_lpm"))
        if rm == "none":
            risk_pen = 0.0
        elif rm == "quad":
            risk_pen = float(getattr(self.cfg, "lambda_var", 1.0)) * (r_n * r_n)
        elif rm == "lpm":
            risk_pen = float(getattr(self.cfg, "lambda_lpm", 1.0)) * float(lpm_n)
        elif rm == "rollvar_lpm":
            risk_pen = float(getattr(self.cfg, "lambda_rollvar", 5.0)) * float(var_n) + float(getattr(self.cfg, "lambda_lpm", 2.0)) * float(lpm_n)
        else:
            risk_pen = 0.0

        # Cost penalty (standardized)
        cost_mult = float(getattr(self.cfg, "cost_penalty_mult", 1.0))
        cost_pen = cost_mult * float(getattr(self.cfg, "eta_cost", 1.0)) * float(c_n)

        # Final reward
        reward = float(mu * r_n - risk_pen - cost_pen)

        # Update equity path
        self._equity += pnl_net
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity
        dd = self._equity - self._peak_equity
        if dd < self._mdd:
            self._mdd = dd

        # Advance time
        self._h = h_new
        self._n_prev = int(n_new)

        # Terminal condition: scenario end index `i1` is end-exclusive.
        # We stop after processing t == (_t_end - 1).
        if t >= (self._t_end - 1):
            done = True
        else:
            self._t = t + 1

        obs = self._make_obs()

        if self.cfg.info_mode == "eval":
            info = {
                "t": t,
                "date_int": int(self.dates_int[t]),
                "Q": self._Q,
                "h": float(self._h),
                "n_used": int(n_used_for_pnl),
                "n_new": int(self._n_prev),
                "h": float(self._h),
                "h_new": float(self._h),
                "roll_flag": int(self.roll_flag[t]),
                "tradable": int(self.tradable[t]),
                "pnl_phys": pnl_phys,
                "pnl_fut": pnl_fut,
                "cost": cost_t,
                "cost_breakdown": cost_breakdown,
                "pnl_net": pnl_net,
                "equity": float(self._equity),
                "mdd": float(self._mdd),
                "reward_raw": float(r),
                "risk_pen": float(risk_pen),
                "var_L": float(var_L),
                "lpm": float(lpm),
                "reward": float(reward),
                "reward_raw_cost": float(c_norm),
                "reward_std": float(mu * r_n),
                "r_n": float(r_n),
                "c_n": float(c_n),
                "var_n": float(var_n),
                "lpm_n": float(lpm_n),
                "scale_r": float(self._scale_r),
                "scale_cost": float(self._scale_cost),
                "scale_var": float(self._scale_var),
                "scale_lpm": float(self._scale_lpm),
            }
        else:
            # Minimal info for training speed
            info = {
                "pnl_net": pnl_net,
                "cost": cost_t,
                "n_prev": int(self._n_prev),
                "h": float(self._h),
                "h_prev": float(self._h),
                "mdd": float(self._mdd),
            }

        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info


# -------------------------
# Factory helper
# -------------------------


def make_env(pre: Any, scenarios: List[Dict[str, Any]], cfg: Optional[EnvConfig] = None, seed: Optional[int] = None):
    """Small helper to create envs for vectorized training."""
    def _thunk():
        return OilHedgingDailyEnv(pre, scenarios, cfg=cfg, seed=seed)

    return _thunk