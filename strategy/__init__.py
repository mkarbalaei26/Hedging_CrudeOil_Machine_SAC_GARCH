"""strategy package

This package contains hedge-ratio generators (h_t) used by `hedge_simulator.py`.

Design contract:
- Each strategy implements a consistent interface (see `strategy.base`).
- Strategies must NOT implement execution mechanics (rounding, costs, roll). Those are handled by the simulator.
- Strategies may implement build_h_path(...) for fast array-based evaluation; get_h(...) remains for compatibility.

This module provides:
- Convenient imports
- A small factory/registry to instantiate strategies by name

Usage example:
    from strategy import make_strategy
    strat = make_strategy("naive", h=1.0)

Naming convention:
- `nohedge`   : h_t = 0
- `naive`     : configurable constant h
- `ols_static`: MVHR/OLS one-shot at trade start
- `ols_roll`  : rolling OLS with window W
"""

from __future__ import annotations

from typing import Dict, Type, Any

from .base import HedgeStrategy

# NOTE: These imports will work once you create the corresponding modules.
# We intentionally import lazily inside `make_strategy` to avoid circular imports during development.


def available_strategies() -> Dict[str, str]:
    """Return supported strategy names and short descriptions."""
    return {
        "nohedge": "No hedge (h_t=0)",
        "naive": "Naive constant hedge ratio (configurable h)",
        "ols_static": "OLS/MVHR static hedge ratio (estimated once at trade start)",
        "ols_roll": "Rolling OLS hedge ratio with lookback window W",
        # GARCH-based strategies
        "ccc_garch": "CCC-GARCH proxy (uses precomputed h_ccc_proxy_{W} column or provided window array)",
        "dcc_garch": "DCC-GARCH (uses precomputed GARCH sigmas + DCC(1,1); supports fast build_h_path)",
    }


def make_strategy(name: str, **kwargs: Any):
    """Factory to instantiate a strategy by name.

    Parameters are passed to the strategy constructor.

    Examples:
        make_strategy("nohedge")
        make_strategy("naive", h=1.0)
        make_strategy("ols_roll", window=120)

    Raises:
        ValueError if the name is unknown.
    """

    key = str(name).strip().lower()

    if key == "nohedge":
        from .nohedge import NoHedgeStrategy
        return NoHedgeStrategy(**kwargs)

    if key == "naive":
        from .naive import NaiveConstantStrategy
        return NaiveConstantStrategy(**kwargs)

    if key in ("ols_static", "ols", "mvhr"):  # aliases
        from .ols_static import OLSStaticStrategy
        return OLSStaticStrategy(**kwargs)

    if key in ("ols_roll", "rolling_ols", "ols_rolling"):
        from .ols_rolling import OLSRollingStrategy
        return OLSRollingStrategy(**kwargs)

    if key in ("ccc_garch", "ccc", "cccgarch"):
        from .ccc_garch import CCCGarchProxyStrategy
        # CLI passes --window; map it to corr_window for this strategy.
        corr_window = int(kwargs.pop("window", 120))
        # NOTE: Do NOT set a default h_col here.
        # CCCGarchProxyStrategy derives the correct default column name internally as:
        #   <PREFIX>_h_ccc_proxy_<W>
        # based on exposure_id.
        # Remove params that belong to other strategies
        kwargs.pop("h", None)
        kwargs.pop("intercept", None)
        return CCCGarchProxyStrategy(corr_window=corr_window, **kwargs)

    if key in ("dcc_garch", "dcc", "dccgarch"):
        from .dcc_garch import DCCGarchStrategy
        # CLI may pass unrelated args used by other strategies
        kwargs.pop("h", None)
        kwargs.pop("intercept", None)
        # `window` is a valid arg for DCCGarchStrategy; ensure int
        if "window" in kwargs and kwargs["window"] is not None:
            kwargs["window"] = int(kwargs["window"])
        return DCCGarchStrategy(**kwargs)

    raise ValueError(
        f"Unknown strategy '{name}'. Available: {list(available_strategies().keys())}"
    )


__all__ = [
    "available_strategies",
    "make_strategy",
    "HedgeStrategy",
]