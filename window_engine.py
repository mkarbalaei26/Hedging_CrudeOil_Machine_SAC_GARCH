"""
window_engine.py
----------------
Builds per-scenario time windows for hedging simulation.

Assumptions (locked):
- N_t (number of contracts) will be integer-rounded later in simulator.
- Spot and futures prices are settlement-to-settlement.
- Futures price column in price_engine output is `F_mark`.

This engine:
- extracts [start_date, end_date] window
- merges spot + futures (F_mark)
- computes dS and dF
- optionally attaches extra feature columns
- enforces no future leakage (consumer must slice history up to t)

Note:
- The `history_up_to_t` method returns a view (no copy) for speed; callers must treat the output as read-only.
"""

from typing import List, Optional, Any, Dict
import pandas as pd
import numpy as np


class WindowEngine:
    def __init__(
        self,
        universe_df: pd.DataFrame,
        price_engine_df: pd.DataFrame,
        date_col: str = "Date",
        spot_col: str = "spot",
        futures_col: str = "F_mark",
        # Compatibility params (ignored here; the roll logic is embedded in price_engine_df)
        exposure_id: Optional[str] = None,
        mode_roll: Optional[bool] = None,
        **kwargs,
    ):
        """
        universe_df: must contain Date + spot (+ optional features)
        price_engine_df: must contain Date + F_mark (+ roll_flag optional)
        """
        # NOTE:
        # - `exposure_id`, `mode_roll` and any extra `kwargs` are accepted for compatibility with
        #   other modules that construct WindowEngine with different signatures.
        # - This WindowEngine always uses the provided `price_engine_df` (F_mark/roll_flag), so
        #   roll/no-roll is determined upstream.
        _ = exposure_id
        _ = mode_roll
        _ = kwargs
        self.date_col = date_col
        self.spot_col = spot_col
        self.futures_col = futures_col

        # normalize date (shallow copies to avoid mutating caller, but reduce memory churn)
        u = universe_df.copy(deep=False)
        p = price_engine_df.copy(deep=False)

        u[self.date_col] = pd.to_datetime(u[self.date_col]).dt.normalize()
        p[self.date_col] = pd.to_datetime(p[self.date_col]).dt.normalize()

        # keep needed columns
        keep_cols = [self.date_col, self.spot_col]
        self._universe = u[keep_cols + [c for c in u.columns if c not in keep_cols]].copy()

        self._price = p[[self.date_col, self.futures_col] +
                        (["roll_flag"] if "roll_flag" in p.columns else [])].copy()

        # merge once for efficiency
        self._merged = pd.merge(
            self._universe,
            self._price,
            on=self.date_col,
            how="inner",
        ).sort_values(self.date_col).reset_index(drop=True)

        # precompute date index map
        self._merged_dates = pd.to_datetime(self._merged[self.date_col]).dt.normalize()
        self._date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(self._merged_dates)}

        # cache arrays for fast slicing (avoids repeated pandas ops)
        self._spot_arr = pd.to_numeric(self._merged[self.spot_col], errors="coerce").to_numpy(dtype=float)
        self._fut_arr = pd.to_numeric(self._merged[self.futures_col], errors="coerce").to_numpy(dtype=float)
        self._roll_arr = (
            pd.to_numeric(self._merged["roll_flag"], errors="coerce").fillna(0).to_numpy(dtype=int)
            if "roll_flag" in self._merged.columns
            else None
        )

    # ------------------------------------------------------------
    # Core window builder
    # ------------------------------------------------------------

    def get_index_bounds(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[int, int]:
        """Return (i0, i1) inclusive indices into the merged arrays."""
        s = pd.Timestamp(start_date).normalize()
        e = pd.Timestamp(end_date).normalize()

        if s not in self._date_to_idx or e not in self._date_to_idx:
            raise ValueError("Start or end date not in merged price/universe data.")

        i0 = self._date_to_idx[s]
        i1 = self._date_to_idx[e]
        if i1 < i0:
            raise ValueError("End date before start date.")
        return i0, i1

    def build_window(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Returns a dataframe with columns:
        date, spot, fut, dS, dF, roll_flag?, + optional features
        """
        i0, i1 = self.get_index_bounds(start_date, end_date)

        # Slice arrays (fast) and build a minimal DataFrame
        idx = slice(i0, i1 + 1)
        dates = self._merged_dates.iloc[idx].to_numpy(dtype="datetime64[ns]")
        spot = self._spot_arr[idx]
        fut = self._fut_arr[idx]

        out: Dict[str, Any] = {
            "date": pd.to_datetime(dates).astype("datetime64[ns]"),
            "spot": spot,
            "fut": fut,
        }
        if self._roll_arr is not None:
            out["roll_flag"] = self._roll_arr[idx]

        # Attach extra feature columns (already aligned in self._merged)
        if feature_cols:
            for col in feature_cols:
                if col not in self._merged.columns:
                    raise ValueError(f"Feature column '{col}' not found in universe data.")
                out[col] = self._merged[col].iloc[idx].to_numpy(copy=False)

        df = pd.DataFrame(out)

        # compute daily differences via NumPy
        dS = np.empty(len(df), dtype=float)
        dF = np.empty(len(df), dtype=float)
        dS[:] = np.nan
        dF[:] = np.nan
        if len(df) > 0:
            dS[0] = 0.0
            dF[0] = 0.0
        if len(df) > 1:
            dS[1:] = spot[1:] - spot[:-1]
            dF[1:] = fut[1:] - fut[:-1]
        df["dS"] = dS
        df["dF"] = dF

        # Handle rare NaNs that may survive upstream forward-fill (e.g., leading NaNs or gaps)
        # Policy (locked): drop rows where spot or fut is NaN, effectively shortening the window.
        # This mirrors the project's earlier "shorten horizon" rule.
        na_mask = df[["spot", "fut"]].isna().any(axis=1)
        if na_mask.any():
            dropped = int(na_mask.sum())
            df = df.loc[~na_mask].reset_index(drop=True)
            if len(df) < 2:
                raise ValueError("Window becomes too short after dropping NaNs in spot/fut.")
            # Recompute diffs after dropping (NumPy)
            spot2 = df["spot"].to_numpy(dtype=float)
            fut2 = df["fut"].to_numpy(dtype=float)
            dS2 = np.empty(len(df), dtype=float)
            dF2 = np.empty(len(df), dtype=float)
            dS2[0] = 0.0
            dF2[0] = 0.0
            dS2[1:] = spot2[1:] - spot2[:-1]
            dF2[1:] = fut2[1:] - fut2[:-1]
            df["dS"] = dS2
            df["dF"] = dF2
            df["window_dropped_nan_rows"] = dropped
        else:
            df["window_dropped_nan_rows"] = 0

        return df.reset_index(drop=True)

    def build_window_arrays(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        feature_cols: Optional[List[str]] = None,
        drop_nan: bool = True,
    ) -> Dict[str, Any]:
        """Array-first window builder (preferred for speed).

        Returns a dict with keys:
            dates (datetime64[ns]), spot (float), fut (float), dS (float), dF (float),
            roll_flag (int, optional), window_dropped_nan_rows (int),
            and each requested feature column as a NumPy array.

        Notes
        -----
        - If `drop_nan` is True (default), rows where spot or fut is NaN are removed and diffs recomputed.
        - Output arrays are aligned and have equal length.
        """
        i0, i1 = self.get_index_bounds(start_date, end_date)
        idx = slice(i0, i1 + 1)

        dates = self._merged_dates.iloc[idx].to_numpy(dtype="datetime64[ns]")
        spot = self._spot_arr[idx]
        fut = self._fut_arr[idx]

        out: Dict[str, Any] = {
            "dates": dates,
            "spot": spot.astype(float, copy=False),
            "fut": fut.astype(float, copy=False),
        }
        if self._roll_arr is not None:
            out["roll_flag"] = self._roll_arr[idx]

        # Attach extra features
        if feature_cols:
            for col in feature_cols:
                if col not in self._merged.columns:
                    raise ValueError(f"Feature column '{col}' not found in universe data.")
                out[col] = self._merged[col].iloc[idx].to_numpy(copy=False)

        # diffs
        n = len(spot)
        dS = np.empty(n, dtype=float)
        dF = np.empty(n, dtype=float)
        if n > 0:
            dS[0] = 0.0
            dF[0] = 0.0
        if n > 1:
            dS[1:] = spot[1:] - spot[:-1]
            dF[1:] = fut[1:] - fut[:-1]
        out["dS"] = dS
        out["dF"] = dF

        dropped = 0
        if drop_nan:
            mask = np.isfinite(spot) & np.isfinite(fut)
            if not mask.all():
                dropped = int((~mask).sum())
                spot2 = spot[mask]
                fut2 = fut[mask]
                dates2 = dates[mask]

                n2 = len(spot2)
                if n2 < 2:
                    raise ValueError("Window becomes too short after dropping NaNs in spot/fut.")

                dS2 = np.empty(n2, dtype=float)
                dF2 = np.empty(n2, dtype=float)
                dS2[0] = 0.0
                dF2[0] = 0.0
                dS2[1:] = spot2[1:] - spot2[:-1]
                dF2[1:] = fut2[1:] - fut2[:-1]

                out["dates"] = dates2
                out["spot"] = spot2.astype(float, copy=False)
                out["fut"] = fut2.astype(float, copy=False)
                out["dS"] = dS2
                out["dF"] = dF2

                if "roll_flag" in out:
                    out["roll_flag"] = out["roll_flag"][mask]

                if feature_cols:
                    for col in feature_cols:
                        out[col] = out[col][mask]

        out["window_dropped_nan_rows"] = dropped
        return out

    # ------------------------------------------------------------
    # Helper for leakage-safe history slicing
    # ------------------------------------------------------------

    @staticmethod
    def history_up_to_t(window_df: pd.DataFrame, t_index: int) -> pd.DataFrame:
        """
        Returns window_df[0:t_index] inclusive.
        Used to prevent look-ahead bias in strategies.

        Note: returns a view (no copy) for speed; caller must treat as read-only.
        """
        if t_index < 0 or t_index >= len(window_df):
            raise ValueError("t_index out of range.")
        return window_df.iloc[: t_index + 1]

    # ------------------------------------------------------------
    # Pre-trade history (no leakage)
    # ------------------------------------------------------------

    def history_before_arrays(
        self,
        date: pd.Timestamp,
        max_rows: Optional[int] = None,
        feature_cols: Optional[List[str]] = None,
        drop_nan: bool = True,
    ) -> Dict[str, Any]:
        """Array-first history strictly BEFORE `date` (preferred for speed).

        Returns dict with keys:
            dates, spot, fut, dS, dF, roll_flag (optional),
            and requested feature columns (optional).

        Notes
        -----
        - If `drop_nan` is True, rows where spot or fut is NaN are removed and diffs recomputed.
        - This function does not allocate large pandas objects.
        """
        d = pd.Timestamp(date).normalize()
        if d not in self._date_to_idx:
            return {"dates": np.empty(0, dtype="datetime64[ns]"), "spot": np.empty(0, dtype=float), "fut": np.empty(0, dtype=float), "dS": np.empty(0, dtype=float), "dF": np.empty(0, dtype=float)}

        i1 = self._date_to_idx[d] - 1
        if i1 < 0:
            return {"dates": np.empty(0, dtype="datetime64[ns]"), "spot": np.empty(0, dtype=float), "fut": np.empty(0, dtype=float), "dS": np.empty(0, dtype=float), "dF": np.empty(0, dtype=float)}

        start_i = 0
        if max_rows is not None and max_rows > 0:
            start_i = max(0, i1 - int(max_rows) + 1)

        idx = slice(start_i, i1 + 1)
        dates = self._merged_dates.iloc[idx].to_numpy(dtype="datetime64[ns]")
        spot = self._spot_arr[idx]
        fut = self._fut_arr[idx]

        out: Dict[str, Any] = {
            "dates": dates,
            "spot": spot.astype(float, copy=False),
            "fut": fut.astype(float, copy=False),
        }
        if self._roll_arr is not None:
            out["roll_flag"] = self._roll_arr[idx]

        if feature_cols:
            for col in feature_cols:
                if col not in self._merged.columns:
                    raise ValueError(f"Feature column '{col}' not found in universe data.")
                out[col] = self._merged[col].iloc[idx].to_numpy(copy=False)

        # diffs
        n = len(spot)
        dS = np.empty(n, dtype=float)
        dF = np.empty(n, dtype=float)
        if n > 0:
            dS[0] = 0.0
            dF[0] = 0.0
        if n > 1:
            dS[1:] = spot[1:] - spot[:-1]
            dF[1:] = fut[1:] - fut[:-1]
        out["dS"] = dS
        out["dF"] = dF

        if drop_nan:
            mask = np.isfinite(spot) & np.isfinite(fut)
            if mask.any() and (not mask.all()):
                dates2 = dates[mask]
                spot2 = spot[mask]
                fut2 = fut[mask]

                n2 = len(spot2)
                if n2 == 0:
                    return {"dates": np.empty(0, dtype="datetime64[ns]"), "spot": np.empty(0, dtype=float), "fut": np.empty(0, dtype=float), "dS": np.empty(0, dtype=float), "dF": np.empty(0, dtype=float)}

                dS2 = np.empty(n2, dtype=float)
                dF2 = np.empty(n2, dtype=float)
                dS2[0] = 0.0
                dF2[0] = 0.0
                if n2 > 1:
                    dS2[1:] = spot2[1:] - spot2[:-1]
                    dF2[1:] = fut2[1:] - fut2[:-1]

                out["dates"] = dates2
                out["spot"] = spot2.astype(float, copy=False)
                out["fut"] = fut2.astype(float, copy=False)
                out["dS"] = dS2
                out["dF"] = dF2

                if "roll_flag" in out:
                    out["roll_flag"] = out["roll_flag"][mask]

                if feature_cols:
                    for col in feature_cols:
                        out[col] = out[col][mask]
            elif not mask.any():
                return {"dates": np.empty(0, dtype="datetime64[ns]"), "spot": np.empty(0, dtype=float), "fut": np.empty(0, dtype=float), "dS": np.empty(0, dtype=float), "dF": np.empty(0, dtype=float)}

        return out

    def history_before(
        self,
        date: pd.Timestamp,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Returns merged history strictly BEFORE `date`.

        Output columns:
            date, spot, fut, dS, dF, roll_flag? (if available)

        This is used by econometric strategies (e.g., OLS) to estimate
        hedge ratios using only past information.
        """
        h = self.history_before_arrays(date=date, max_rows=max_rows, feature_cols=None, drop_nan=True)
        if len(h.get("dates", [])) == 0:
            return pd.DataFrame(columns=["date", "spot", "fut", "dS", "dF"])

        df = pd.DataFrame({
            "date": pd.to_datetime(h["dates"]).astype("datetime64[ns]"),
            "spot": h["spot"],
            "fut": h["fut"],
            "dS": h["dS"],
            "dF": h["dF"],
        })
        if "roll_flag" in h:
            df["roll_flag"] = h["roll_flag"].astype(int)
        return df.reset_index(drop=True)
