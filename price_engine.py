# -*- coding: utf-8 -*-
"""price_engine.py

گام 3) Price Engine با سوییچ Roll/No-roll

این ماژول فقط «قیمت/settlement» و «PnL خام فیوچرز (mark-to-market)» را می‌سازد.
هیچ منطق hedge ratio، هزینه تراکنش، یا شبیه‌سازی معامله فیزیکی در این مرحله انجام نمی‌شود.

حالت‌ها:
- no_roll:   F_t = CL1_t
- roll:      ساخت مکانیزم «نگهداری قرارداد» با رول در RollDate

قانون رول (LOCKED):
- Last Trading Day (LTD): 3 روز کاری قبل از 25ام ماه تقویمیِ قبل از ماه قرارداد
- Roll Date: 2 روز کاری قبل از LTD

تعریف «روز کاری» در این پیاده‌سازی:
- روز کاری = روزی که settlement واقعاً آپدیت شده (tradable_mask=True)
  (همان چیزی که در DataAdapter و QC قفل کرده‌ایم)

نکته مهم درباره PnL در رول:
- PnL روز رول با قرارداد قدیمی محاسبه می‌شود (تا پایان روز رول هنوز قرارداد قبلی را داریم)
- در پایان روز رول، قرارداد بعدی باز می‌شود و قیمت مرجع برای PnL روز بعد می‌شود CL2 در همان روز رول

محدودیت داده:
- اگر فقط CL1 و CL2 داشته باشیم، پس از رول فرض می‌کنیم قرارداد جدید از روز بعد به بعد با CL1 دنبال می‌شود.
  این یک تقریب عملی است و باید در پایان‌نامه مستندسازی شود.

"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class RollEvent:
    month: str
    anchor_25: pd.Timestamp
    ltd: pd.Timestamp
    roll_date: pd.Timestamp


class RollCalendarBuilder:
    """Build roll calendar using a provided list of tradable dates."""

    def __init__(self, tradable_dates: pd.DatetimeIndex):
        if len(tradable_dates) == 0:
            raise ValueError("tradable_dates is empty")
        # ensure sorted unique
        self.tradable_dates = pd.DatetimeIndex(sorted(pd.unique(tradable_dates)))
        self._set = set(self.tradable_dates)

    def _prev_tradable(self, d: pd.Timestamp, n: int) -> pd.Timestamp:
        """Return the n-th previous tradable date strictly before or equal to d.

        Robust version:
        - If there is not enough history at the beginning of the sample,
        truncate to the earliest available tradable date instead of raising.
        - This avoids crashing in early Brent sample (e.g. 1987).
        - No look-ahead is introduced.
        """

        idx = self.tradable_dates.searchsorted(d, side="right") - 1

        # اگر هیچ تاریخ قابل معامله‌ای قبل از d نداریم
        if idx < 0:
            return pd.Timestamp(self.tradable_dates[0])

        idx2 = idx - n

        # اگر تاریخ کافی برای عقب رفتن نداریم
        if idx2 < 0:
            return pd.Timestamp(self.tradable_dates[0])

        return pd.Timestamp(self.tradable_dates[idx2])

    def build(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Build roll events table for months intersecting [start, end].

        Interpretation:
        - For a given calendar month M (e.g., 2019-06), anchor = 25th of M.
        - LTD = 3 tradable days before anchor.
        - RollDate = 2 tradable days before LTD.
        This corresponds to the contract month being M+1.
        """
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()

        months = pd.period_range(start=start, end=end, freq="M")
        events = []

        for p in months:
            m_start = p.to_timestamp(how="start")
            # anchor date is 25th of that month
            anchor = pd.Timestamp(year=m_start.year, month=m_start.month, day=25)

            # if anchor is outside data range by a lot, skip
            if anchor < self.tradable_dates.min() or anchor > self.tradable_dates.max():
                continue

            # LTD = 3 tradable days before anchor
            ltd = self._prev_tradable(anchor, n=3)
            roll = self._prev_tradable(ltd, n=2)

            # only keep roll dates that are within [start,end]
            if roll < start or roll > end:
                continue

            events.append({
                "month": str(p),
                "anchor_25": anchor,
                "ltd": ltd,
                "roll_date": roll,
            })

        if not events:
            return pd.DataFrame(columns=["month", "anchor_25", "ltd", "roll_date"])

        df = pd.DataFrame(events).sort_values("roll_date").reset_index(drop=True)
        return df


class PriceEngineCL:
    """WTI CL price engine with roll/no-roll switch.

    Inputs:
      df: dataframe containing Date, CL1, (optional) CL2
      date_col: name of date column
      cl1_col: name of front month settlement column
      cl2_col: name of second month settlement column
      tradable_mask: boolean Series aligned with df index (True when settlement updated)

    Outputs:
      get_series(mode): returns a DataFrame with columns:
        - Date
        - F_mark   (end-of-day mark price for the held position)
        - pnl_1c   (daily PnL for +1 contract)
        - roll_flag (1 on roll date when a roll occurs)
        - held_leg ("CL1" or "CL2" indicating the mark used for end-of-day)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
        cl1_col: str = "CL1",
        cl2_col: str = "CL2",
        contract_size_bbl: int = 1000,
        tradable_mask: Optional[pd.Series] = None,
        roll_calendar: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> None:
        self.date_col = date_col
        self.cl1_col = cl1_col
        self.cl2_col = cl2_col
        self.contract_size_bbl = int(contract_size_bbl)
        self.verbose = verbose

        d = df.copy()
        if date_col not in d.columns:
            raise KeyError(f"df must include '{date_col}'")
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        if cl1_col not in d.columns:
            raise KeyError(f"df must include '{cl1_col}'")
        if cl2_col not in d.columns:
            # allow missing CL2 for no-roll mode, but roll mode needs it at roll dates
            d[cl2_col] = np.nan

        self.df = d

        if tradable_mask is None:
            # default: tradable when CL1 changes (robust to forward-fill)
            cl1 = self.df[cl1_col].astype(float)
            tradable_mask = cl1.notna() & (cl1 != cl1.shift(1))
        else:
            if len(tradable_mask) != len(self.df):
                raise ValueError("tradable_mask length mismatch")
            tradable_mask = tradable_mask.astype(bool).reset_index(drop=True)

        self.tradable_mask = tradable_mask

        # roll calendar (lazy)
        self._roll_table = roll_calendar

    def get_roll_table(self) -> pd.DataFrame:
        if self._roll_table is not None:
            return self._roll_table.copy()

        tradable_dates = pd.DatetimeIndex(self.df.loc[self.tradable_mask, self.date_col])
        b = RollCalendarBuilder(tradable_dates)
        start = self.df[self.date_col].min()
        end = self.df[self.date_col].max()
        self._roll_table = b.build(start=start, end=end)
        return self._roll_table.copy()

    def get_series(self, mode: str = "no_roll") -> pd.DataFrame:
        mode = str(mode).lower().strip()
        if mode not in {"no_roll", "roll"}:
            raise ValueError("mode must be 'no_roll' or 'roll'")

        if mode == "no_roll":
            return self._series_no_roll()
        return self._series_roll()

    # -------------------------
    # Mode implementations
    # -------------------------

    def _series_no_roll(self) -> pd.DataFrame:
        d = self.df
        F = d[self.cl1_col].astype(float)
        pnl = (F - F.shift(1)) * self.contract_size_bbl
        out = pd.DataFrame({
            self.date_col: d[self.date_col],
            "F_mark": F,
            "pnl_1c": pnl,
            "roll_flag": 0,
            "held_leg": "CL1",
        })
        return out

    def _series_roll(self) -> pd.DataFrame:
        d = self.df
        roll_table = self.get_roll_table()

        dates = pd.to_datetime(d[self.date_col]).dt.normalize()
        cl1 = d[self.cl1_col].astype(float)
        cl2 = d[self.cl2_col].astype(float)

        # Vectorized roll flag aligned to `dates`
        if len(roll_table) > 0:
            roll_dates = pd.to_datetime(roll_table["roll_date"]).dt.normalize()
            is_roll_arr = dates.isin(roll_dates).to_numpy(dtype=bool)
        else:
            is_roll_arr = np.zeros(len(d), dtype=bool)

        F_mark = np.full(len(d), np.nan, dtype=float)
        pnl = np.full(len(d), np.nan, dtype=float)
        roll_flag = np.zeros(len(d), dtype=int)
        held_leg = np.array(["CL1"] * len(d), dtype=object)

        # Convert to NumPy once for speed
        dates_np = dates.to_numpy()
        cl1_np = cl1.to_numpy(dtype=float)
        cl2_np = cl2.to_numpy(dtype=float)

        # We compute PnL per +1 contract with explicit roll handling.
        # prev_price tracks the held contract price at the *end of previous day*.
        prev_price = np.nan

        n = len(d)
        for i in range(n):
            # Skip days where CL1 is missing; keep NaN
            c1 = cl1_np[i]
            if not np.isfinite(c1):
                F_mark[i] = np.nan
                pnl[i] = np.nan
                continue

            is_roll = bool(is_roll_arr[i])

            if i == 0:
                # initialize: assume we start holding CL1 at first observation
                prev_price = c1
                F_mark[i] = c1
                pnl[i] = np.nan
                continue

            if not np.isfinite(prev_price):
                # re-initialize if prev_price got lost
                c1_prev = cl1_np[i - 1]
                prev_price = c1_prev if np.isfinite(c1_prev) else c1

            # 1) daily PnL up to today's settlement using the contract held since yesterday
            pnl[i] = (c1 - prev_price) * self.contract_size_bbl

            if is_roll:
                roll_flag[i] = 1
                # 2) roll at end-of-day: close old (already marked by cl1), open new at cl2
                c2 = cl2_np[i]
                if np.isfinite(c2):
                    F_mark[i] = c2
                    held_leg[i] = "CL2"
                    # prev_price for next day becomes the new contract entry mark (cl2 at roll date)
                    prev_price = c2
                else:
                    # If CL2 missing on roll date, fall back to CL1 (documented limitation)
                    F_mark[i] = c1
                    held_leg[i] = "CL1"
                    prev_price = c1
            else:
                # normal day: end-of-day mark is cl1
                F_mark[i] = c1
                held_leg[i] = "CL1"
                prev_price = c1

        out = pd.DataFrame({
            self.date_col: d[self.date_col],
            "F_mark": F_mark,
            "pnl_1c": pnl,
            "roll_flag": roll_flag,
            "held_leg": held_leg,
        })
        return out


if __name__ == "__main__":
    # Minimal CLI smoke test
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default="MasterData.parquet")
    ap.add_argument("--date_col", type=str, default="Date")
    ap.add_argument("--cl1", type=str, default="CL1")
    ap.add_argument("--cl2", type=str, default="CL2")
    ap.add_argument("--mode", type=str, default="roll", choices=["roll", "no_roll"])
    ap.add_argument("--out", type=str, default="MasterData_price_engine.parquet")
    ap.add_argument("--fmt", type=str, default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--npz", type=str, default="", help="Optional NPZ output path for fast loading")
    args = ap.parse_args()

    master_path = str(args.master)
    if master_path.lower().endswith(".parquet"):
        df = pd.read_parquet(master_path)
    else:
        df = pd.read_csv(master_path, low_memory=False)

    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.dropna(subset=[args.date_col]).sort_values(args.date_col).reset_index(drop=True)

    # tradable: settlement update on CL1
    cl1 = pd.to_numeric(df[args.cl1], errors="coerce")
    tradable = cl1.notna() & (cl1 != cl1.shift(1))

    eng = PriceEngineCL(
        df=df[[args.date_col, args.cl1, args.cl2]].copy(),
        date_col=args.date_col,
        cl1_col=args.cl1,
        cl2_col=args.cl2,
        contract_size_bbl=1000,
        tradable_mask=tradable,
        verbose=True,
    )

    s = eng.get_series(mode=args.mode)
    rt = eng.get_roll_table()

    # Write to the same folder as this script by default
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path(__file__).resolve().parent / out_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.fmt == "csv" or str(out_path).lower().endswith(".csv"):
        s.to_csv(out_path, index=False)
    else:
        # Parquet is faster to re-load and preserves dtypes
        s.to_parquet(out_path, index=False)

    # Optional NPZ output for fast loading
    if args.npz:
        npz_path = Path(args.npz)
        if not npz_path.is_absolute():
            npz_path = (Path(__file__).resolve().parent / npz_path).resolve()
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(npz_path),
            Date=pd.to_datetime(s[args.date_col]).dt.normalize().to_numpy(dtype="datetime64[ns]"),
            F_mark=pd.to_numeric(s["F_mark"], errors="coerce").to_numpy(dtype=float),
            pnl_1c=pd.to_numeric(s["pnl_1c"], errors="coerce").to_numpy(dtype=float),
            roll_flag=pd.to_numeric(s["roll_flag"], errors="coerce").fillna(0).to_numpy(dtype=int),
            held_leg=s["held_leg"].astype(str).to_numpy(),
        )
        print("[PriceEngine] wrote NPZ:", str(npz_path))

    print("[PriceEngine] mode:", args.mode)
    print("[PriceEngine] rows:", len(s))
    print("[PriceEngine] roll events:", len(rt))
    if len(rt) > 0:
        print("[PriceEngine] first roll:", rt.iloc[0].to_dict())
        print("[PriceEngine] last  roll:", rt.iloc[-1].to_dict())
    print("[PriceEngine] wrote:", str(out_path))
    print("[PriceEngine] master:", master_path)