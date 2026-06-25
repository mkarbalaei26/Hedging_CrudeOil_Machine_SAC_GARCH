"""strategy.nohedge

No-hedge baseline strategy.

Returns h_t = 0 for all t.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from .base import HedgeStrategy


class NoHedgeStrategy(HedgeStrategy):
    """No hedge: h_t = 0."""

    name = "NoHedge"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def reset(self) -> None:
        # stateless
        pass

    def build_h_path(
        self,
        dS,
        dF,
        dates=None,
        scenario_meta: Optional[Dict[str, Any]] = None,
        spot=None,
        fut=None,
    ):
        """Preferred fast API: zero hedge ratio over the full window."""
        n = len(dS)
        return np.zeros(n, dtype=float)

    def get_h(self, t_index: int, history_df: pd.DataFrame, scenario_meta: Dict[str, Any]) -> float:
        return 0.0
