"""strategy.naive

Naive (constant) hedge ratio baseline.

Returns a fixed hedge ratio h for all t.
- Sign conventions are handled by the simulator.
- Contract rounding and costs are handled by the simulator.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .base import HedgeStrategy


class NaiveConstantStrategy(HedgeStrategy):
    """Constant hedge ratio h_t = h."""

    name = "Naive"

    def __init__(
        self,
        h: float = 1.0,
        clip: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ):
        super().__init__(h=h, clip=clip, **kwargs)
        self.h = float(h)
        self.clip = clip

    def reset(self) -> None:
        # Stateless strategy
        pass

    def build_h_path(
        self,
        dS: np.ndarray,
        dF: np.ndarray,
        dates=None,
        scenario_meta: Optional[Dict[str, Any]] = None,
        spot: Optional[np.ndarray] = None,
        fut: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Preferred fast API: constant hedge ratio over the full window."""
        n = len(dS)
        h = np.full(n, float(self.h), dtype=float)
        if self.clip is not None:
            lo, hi = self.clip
            h = np.clip(h, float(lo), float(hi))
        return h

    def get_h(
        self,
        t_index: int,
        history_df: pd.DataFrame,
        scenario_meta: Dict[str, Any],
    ) -> float:
        h = float(self.h)
        if self.clip is not None:
            lo, hi = self.clip
            h = float(np.clip(h, lo, hi))
        return h


__all__ = ["NaiveConstantStrategy"]