"""cost_model.py
-----------------
Locked execution cost model for CL futures hedging.

Design goals:
- Deterministic, strategy-agnostic costs (prevents AI "cheating")
- Simple tick-based proxy (bid/ask unavailable)

Contract facts (locked):
- CL contract size: 1000 bbl per contract
- Tick size: $0.01/bbl
- Tick value: $10 per contract

Cost assumption (locked default):
- half-spread = 1 tick => $10 per trade per contract

Definitions:
- Position is integer number of contracts N_t (can be long or short).
- A position change from N_{t-1} to N_t trades |N_t - N_{t-1}| contracts.
- Roll event (if enabled) adds an additional close+open on the *same* notional size:
  roll_cost_t = 2 * |N_{t-1}| * cost_per_contract
  This is applied only when roll_flag is True and N_{t-1} != 0.

Notes:
- This module does NOT decide N_t. Simulator will compute/round N_t.
- This module does NOT include slippage beyond the proxy tick cost.
- Provides vectorized helpers for batch simulation (NumPy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, Union, Optional

import numpy as np


@dataclass(frozen=True)
class ExecutionCostConfig:
    """Configuration for execution costs."""

    cost_per_contract_trade_usd: float = 10.0  # 1 tick half-spread
    rounding_mode: str = "bankers"  # "bankers" (half-to-even) or "away_from_zero"


class ExecutionCostModel:
    """Tick-based transaction cost model."""

    def __init__(self, cfg: ExecutionCostConfig | None = None):
        self.cfg = cfg if cfg is not None else ExecutionCostConfig()
        if self.cfg.rounding_mode not in ("bankers", "away_from_zero"):
            raise ValueError("rounding_mode must be 'bankers' or 'away_from_zero'")

    # ------------------------------------------------------------
    # Core cost calculations
    # ------------------------------------------------------------

    def trade_cost(self, n_prev: int, n_new: int) -> float:
        """Cost from changing position from n_prev to n_new."""
        dn = int(n_new) - int(n_prev)
        return abs(dn) * float(self.cfg.cost_per_contract_trade_usd)

    def roll_cost(self, n_prev: int, roll_flag: bool) -> float:
        """Additional cost on roll day for closing+opening positions.

        Charged on the position carried into the roll day (n_prev), i.e., the
        notional that must be closed and re-opened due to the mandatory roll.

        This avoids overcharging when a strategy enters a new position on a roll
        day (N_{t-1}=0, N_t!=0): in that case there is no position to roll, only
        an entry trade.
        """
        if not roll_flag:
            return 0.0
        n = abs(int(n_prev))
        if n == 0:
            return 0.0
        return 2.0 * n * float(self.cfg.cost_per_contract_trade_usd)

    def total_cost(self, n_prev: int, n_new: int, roll_flag: bool) -> Dict[str, float]:
        """Return a cost breakdown for a single day."""
        c_trade = self.trade_cost(n_prev, n_new)
        c_roll = self.roll_cost(n_prev, roll_flag)
        return {
            "cost_trade": float(c_trade),
            "cost_roll": float(c_roll),
            "cost_total": float(c_trade + c_roll),
        }

    # ------------------------------------------------------------
    # Vectorized helpers (NumPy) for batch simulation
    # ------------------------------------------------------------

    def trade_cost_vec(self, n_prev: np.ndarray, n_new: np.ndarray) -> np.ndarray:
        """Vectorized trade cost for arrays of previous/new positions."""
        n_prev = np.asarray(n_prev, dtype=int)
        n_new = np.asarray(n_new, dtype=int)
        if n_prev.shape != n_new.shape:
            raise ValueError("n_prev and n_new must have the same shape")
        dn = n_new - n_prev
        return np.abs(dn) * float(self.cfg.cost_per_contract_trade_usd)

    def roll_cost_vec(self, n_prev: np.ndarray, roll_flag: np.ndarray) -> np.ndarray:
        """Vectorized roll cost for arrays.

        Parameters
        ----------
        n_prev
            Integer position carried into the roll day (before re-hedge).
        roll_flag
            Boolean/int flag (True/1 means roll day).
        """
        n_prev = np.asarray(n_prev, dtype=int)
        rf = np.asarray(roll_flag).astype(bool)
        if n_prev.shape != rf.shape:
            raise ValueError("n_prev and roll_flag must have the same shape")
        n = np.abs(n_prev)
        out = np.zeros_like(n, dtype=float)
        mask = rf & (n != 0)
        if np.any(mask):
            out[mask] = 2.0 * n[mask] * float(self.cfg.cost_per_contract_trade_usd)
        return out

    def total_cost_vec(
        self,
        n_prev: np.ndarray,
        n_new: np.ndarray,
        roll_flag: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Vectorized cost breakdown.

        Returns dict of arrays: cost_trade, cost_roll, cost_total.
        """
        c_trade = self.trade_cost_vec(n_prev, n_new)
        c_roll = self.roll_cost_vec(n_prev, roll_flag)
        return {
            "cost_trade": c_trade.astype(float),
            "cost_roll": c_roll.astype(float),
            "cost_total": (c_trade + c_roll).astype(float),
        }

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------

    @staticmethod
    def _round_to_int(arr: np.ndarray, mode: str = "bankers") -> np.ndarray:
        """Round float array to int contracts.

        - bankers: half-to-even (NumPy rint)
        - away_from_zero: halves are rounded away from zero
        """
        x = np.asarray(arr, dtype=float)
        if mode == "bankers":
            return np.rint(x).astype(int)
        # away_from_zero
        s = np.sign(x)
        ax = np.abs(x)
        return (s * np.floor(ax + 0.5)).astype(int)

    @staticmethod
    def to_int_contracts(x: float) -> int:
        """Round to integer contracts (locked design decision).

        Uses vectorized rounding helper for consistency.
        """
        # Keep backward-compatible signature but route through the vectorized implementation
        return int(ExecutionCostModel._round_to_int(np.array([float(x)]), mode="bankers")[0])

    @staticmethod
    def to_int_contracts_vec(x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Vectorized rounding to integer contracts.

        Mirrors `to_int_contracts` behavior using bankers rounding by default.
        """
        arr = np.asarray(x, dtype=float)
        return ExecutionCostModel._round_to_int(arr, mode="bankers")

    def to_int_contracts_vec_mode(self, x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Vectorized rounding using the model's configured rounding_mode."""
        arr = np.asarray(x, dtype=float)
        return self._round_to_int(arr, mode=self.cfg.rounding_mode)

    @staticmethod
    def turnover_contracts(n_prev: np.ndarray, n_new: np.ndarray) -> np.ndarray:
        """Vectorized per-step contract turnover |ΔN|."""
        n_prev = np.asarray(n_prev, dtype=int)
        n_new = np.asarray(n_new, dtype=int)
        if n_prev.shape != n_new.shape:
            raise ValueError("n_prev and n_new must have the same shape")
        return np.abs(n_new - n_prev).astype(float)

    @staticmethod
    def turnover_h(h_prev: np.ndarray, h_new: np.ndarray) -> np.ndarray:
        """Vectorized per-step hedge-ratio turnover |Δh|."""
        a = np.asarray(h_prev, dtype=float)
        b = np.asarray(h_new, dtype=float)
        if a.shape != b.shape:
            raise ValueError("h_prev and h_new must have the same shape")
        return np.abs(b - a).astype(float)


def _demo() -> None:
    """Small self-test / demo. Not used by simulator."""
    cm = ExecutionCostModel()
    cm2 = ExecutionCostModel(ExecutionCostConfig(rounding_mode="away_from_zero"))
    print("round bankers:", cm.to_int_contracts_vec([0.5, 1.5, -0.5, -1.5]))
    print("round away  :", cm2.to_int_contracts_vec_mode([0.5, 1.5, -0.5, -1.5]))
    # open 3 contracts
    print(cm.total_cost(0, 3, roll_flag=False))
    # adjust down to 1
    print(cm.total_cost(3, 1, roll_flag=False))
    # roll day with 1 contract
    print(cm.total_cost(1, 1, roll_flag=True))
    # roll day entering a new position: should NOT pay roll cost (only trade cost)
    print(cm.total_cost(0, 1, roll_flag=True))

    n_prev = np.array([0, 3, 1])
    n_new = np.array([3, 1, 1])
    rf = np.array([0, 0, 1])
    print(cm.total_cost_vec(n_prev, n_new, rf))


if __name__ == "__main__":
    _demo()