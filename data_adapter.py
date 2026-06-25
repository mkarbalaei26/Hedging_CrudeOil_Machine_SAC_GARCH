# -*- coding: utf-8 -*-
"""
DataAdapter
API یکتا برای همه مدل‌ها

وظایف:
- خواندن MasterData
- هم‌ترازی
- enforce کردن قوانین missing / zero / negative
- ساخت universe جدا برای هر exposure
- جلوگیری از leakage
"""

from __future__ import annotations

import os
import hashlib
import yaml
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path


class DataAdapter:

    def __init__(self, master_csv: str, config_path: Union[str, Dict[str, Any]], verbose: bool = True):
        # master_csv kept for backward compatibility; can be parquet/csv
        self.master_csv = master_csv
        # For logging/debugging only
        self.config_path = "<dict>" if isinstance(config_path, dict) else str(config_path)
        self.verbose = verbose

        if self.verbose:
            print(f"[DataAdapter] Loading master: {master_csv}")

        # -------------------------------------------------
        # Fast master loading (CSV -> cached Parquet)
        # -------------------------------------------------
        master_path = str(master_csv)
        lower = master_path.lower()

        # If a parquet is provided, always read parquet.
        if lower.endswith((".parquet", ".pq")):
            self.df_raw = pd.read_parquet(master_path)

        else:
            # CSV (or other text) -> try to use a stable cached parquet.
            # Cache key depends on absolute path + mtime + file size.
            abs_path = os.path.abspath(master_path)
            try:
                st = os.stat(abs_path)
                sig = f"{abs_path}|{int(st.st_mtime)}|{st.st_size}".encode("utf-8")
            except FileNotFoundError:
                # Let pandas raise a clearer error later
                sig = abs_path.encode("utf-8")

            cache_key = hashlib.md5(sig).hexdigest()[:12]
            cache_dir = os.path.join(os.path.dirname(abs_path), "._cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"MasterData_{cache_key}.parquet")

            if os.path.exists(cache_path):
                self.df_raw = pd.read_parquet(cache_path)

            else:
                # Robust CSV read (handles UTF-8/UTF-8-SIG/Latin1). Avoid pandas low_memory dtype churn.
                read_kwargs = dict(low_memory=False)
                try:
                    self.df_raw = pd.read_csv(abs_path, encoding="utf-8", **read_kwargs)
                except UnicodeDecodeError:
                    try:
                        self.df_raw = pd.read_csv(abs_path, encoding="utf-8-sig", **read_kwargs)
                    except UnicodeDecodeError:
                        # last resort (keeps all bytes); this is the safest for mixed encodings
                        self.df_raw = pd.read_csv(abs_path, encoding="latin1", **read_kwargs)

                # Write cache for subsequent runs
                try:
                    self.df_raw.to_parquet(cache_path, index=False)
                    if self.verbose:
                        print(f"[DataAdapter] Cached master parquet: {cache_path}")
                except Exception as e:
                    # Caching is a performance optimization; do not fail the run if it cannot be written.
                    if self.verbose:
                        print(f"[DataAdapter] WARN: could not write parquet cache: {e}")

        # Parse/sort dates robustly
        if "Date" not in self.df_raw.columns:
            raise ValueError("MasterData must contain a 'Date' column.")

        self.df_raw["Date"] = pd.to_datetime(self.df_raw["Date"], errors="coerce").dt.normalize()
        self.df_raw = (
            self.df_raw.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        )

        # -------------------------------
        # Load config
        # - Accept either a YAML path (str/pathlike) or an already-parsed dict
        # -------------------------------
        if isinstance(config_path, dict):
            self.cfg = config_path
        else:
            cfg_path = str(config_path)
            # Allow passing a Path-like
            cfg_path = str(Path(cfg_path))
            with open(cfg_path, "r") as f:
                self.cfg = yaml.safe_load(f)

        self.date_col = "Date"

        self.df_aligned = self._prepare_data()
        self._dates_np = pd.to_datetime(self.df_aligned[self.date_col]).dt.normalize().to_numpy(dtype="datetime64[ns]")
        # Cache direct column views for fast array slicing (no copies unless needed)
        self._col_view: Dict[str, np.ndarray] = {}
        for c in self.df_aligned.columns:
            if c == self.date_col:
                continue
            if pd.api.types.is_numeric_dtype(self.df_aligned[c]):
                self._col_view[c] = pd.to_numeric(self.df_aligned[c], errors="coerce").to_numpy(dtype=float)
            else:
                self._col_view[c] = self.df_aligned[c].to_numpy()

    # -------------------------------------------------
    # Core preprocessing
    # -------------------------------------------------

    def _prepare_data(self) -> pd.DataFrame:
        # Work on a shallow copy to keep df_raw intact but avoid deep churn.
        df = self.df_raw.copy(deep=False)

        # sort
        df = df.sort_values(self.date_col).reset_index(drop=True)

        # -------------------------------------------------
        # Update/Tradable flags (computed on RAW/pre-ffill data)
        # -------------------------------------------------
        # These flags are used to prevent re-hedging on days where settlement did not update.
        # Definition: updated = finite(raw) and raw != 0
        # NOTE: We compute on the raw dataframe BEFORE any ffill.
        override_cols = self.cfg.get("preprocess", {}).get("zero_rule_columns", None)
        core_cols_for_flags: List[str] = []
        if isinstance(override_cols, list) and override_cols:
            core_cols_for_flags = [c for c in override_cols if isinstance(c, str)]
        else:
            try:
                exposures = self.cfg.get("assets", {}).get("exposures", [])
                for e in exposures:
                    if isinstance(e, dict) and "price_column" in e:
                        core_cols_for_flags.append(e["price_column"])
            except Exception:
                pass
            try:
                hedge_cfg = self.cfg.get("assets", {}).get("hedge_instrument", {})
                cl1 = hedge_cfg.get("price_column_front") or hedge_cfg.get("price_column") or "CL1"
                cl2 = hedge_cfg.get("price_column_next") or hedge_cfg.get("price_column_second")
                core_cols_for_flags.append(cl1)
                if cl2:
                    core_cols_for_flags.append(cl2)
            except Exception:
                pass

        core_cols_for_flags = list(dict.fromkeys([c for c in core_cols_for_flags if c in df.columns and c != self.date_col]))

        updated_flags: Dict[str, np.ndarray] = {}
        for c in core_cols_for_flags:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            raw = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            upd = np.isfinite(raw) & (raw != 0.0)
            updated_flags[c] = upd

        # zero handling: drop zeros (set to NaN) BEFORE ffill so zeros don't get propagated
        zero_rule = self.cfg.get("preprocess", {}).get("zero_rule", "drop")
        if zero_rule == "drop":
            # Apply zero->NaN only to core price columns (spot + CL1/CL2), not to all numeric features.
            # Optional override: preprocess.zero_rule_columns: ["WTI", "Brent", "OPEC", "CL1", "CL2"]
            override_cols = self.cfg.get("preprocess", {}).get("zero_rule_columns", None)

            core_cols: List[str] = []
            if isinstance(override_cols, list) and override_cols:
                core_cols = [c for c in override_cols if isinstance(c, str)]
            else:
                # infer from config assets
                try:
                    exposures = self.cfg.get("assets", {}).get("exposures", [])
                    for e in exposures:
                        if isinstance(e, dict) and "price_column" in e:
                            core_cols.append(e["price_column"])
                except Exception:
                    pass

                try:
                    hedge_cfg = self.cfg.get("assets", {}).get("hedge_instrument", {})
                    cl1 = hedge_cfg.get("price_column_front") or hedge_cfg.get("price_column") or "CL1"
                    cl2 = hedge_cfg.get("price_column_next") or hedge_cfg.get("price_column_second")
                    core_cols.append(cl1)
                    if cl2:
                        core_cols.append(cl2)
                except Exception:
                    pass

            # keep unique existing numeric columns only
            core_cols = list(dict.fromkeys([c for c in core_cols if c in df.columns and c != self.date_col]))
            core_cols = [c for c in core_cols if pd.api.types.is_numeric_dtype(df[c])]

            if core_cols:
                zmask = (df[core_cols] == 0)
                df.loc[:, core_cols] = df[core_cols].mask(zmask, np.nan)

        # forward fill (global rule) with optional row-limit
        ffill_limit = self.cfg.get("preprocess", {}).get("ffill_limit_rows", None)
        if ffill_limit is None:
            df = df.ffill()
        else:
            try:
                ffill_limit_int = int(ffill_limit)
            except Exception:
                ffill_limit_int = None
            df = df.ffill(limit=ffill_limit_int)

        # Attach updated flags as int8 columns (NOT forward-filled)
        for c, upd in updated_flags.items():
            df[f"{c}__updated"] = upd.astype(np.int8)

        # Define a generic tradable flag for the hedge instrument (CL1) based on its raw update.
        # This is exposure-agnostic and useful for environments.
        try:
            hedge_cfg = self.cfg.get("assets", {}).get("hedge_instrument", {})
            cl1 = hedge_cfg.get("price_column_front") or hedge_cfg.get("price_column") or "CL1"
            if cl1 in updated_flags:
                df["__tradable"] = updated_flags[cl1].astype(np.int8)
            else:
                # fallback: tradable if cl1 is finite after alignment
                df["__tradable"] = np.isfinite(pd.to_numeric(df.get(cl1, np.nan), errors="coerce")).astype(np.int8)
        except Exception:
            df["__tradable"] = 0

        return df

    # -------------------------------------------------
    # Universe builder (SAFE VERSION)
    # -------------------------------------------------

    def get_universe(
        self,
        exposure_id: str,
        include_features: bool = True,
        feature_role: str = "both",
    ) -> pd.DataFrame:

        df = self.df_aligned

        exp = self._get_exposure(exposure_id)
        exp_col = exp["price_column"]

        hedge_cfg = self.cfg["assets"]["hedge_instrument"]
        cl1 = hedge_cfg.get("price_column_front") or hedge_cfg.get("price_column") or "CL1"
        # CL2 is optional; config key may be 'price_column_next' (preferred) or 'price_column_second'
        cl2 = hedge_cfg.get("price_column_next") or hedge_cfg.get("price_column_second")

        keep_cols = [self.date_col, exp_col, cl1]
        if cl2 is not None:
            keep_cols.append(cl2)
        # Updated/tradable flags (computed pre-ffill)
        for flag_col in (f"{exp_col}__updated", f"{cl1}__updated", "__tradable"):
            if flag_col in df.columns:
                keep_cols.append(flag_col)

        # Optional features: support multiple config shapes
        # - list of dicts: [{"column": "OVX"}, ...]
        # - list of strings: ["OVX", "VIX", ...]
        # - dict (recommended in this project):
        #     {"roles": {"baseline_only": [...], "ai_feature_only": [...]}, "columns": [...]}
        if include_features:
            features_cfg = self.cfg.get("features", None)
            feat_cols: List[str] = []

            if isinstance(features_cfg, list):
                for f in features_cfg:
                    if isinstance(f, str):
                        feat_cols.append(f)
                    elif isinstance(f, dict) and "column" in f:
                        feat_cols.append(f["column"])

            elif isinstance(features_cfg, dict):
                # explicit columns list
                cols = features_cfg.get("columns", [])
                if isinstance(cols, list):
                    feat_cols += [c for c in cols if isinstance(c, str)]

                # roles-based lists
                roles = features_cfg.get("roles", {})
                if isinstance(roles, dict):
                    # Apply role filter
                    if feature_role in ("both", "baseline", "baseline_only"):
                        v = roles.get("baseline_only", [])
                        if isinstance(v, list):
                            feat_cols += [c for c in v if isinstance(c, str)]
                    if feature_role in ("both", "ai", "ai_only", "ai_feature_only"):
                        v = roles.get("ai_feature_only", [])
                        if isinstance(v, list):
                            feat_cols += [c for c in v if isinstance(c, str)]

            # remove duplicates
            feat_cols = list(dict.fromkeys(feat_cols))

            # append
            for c in feat_cols:
                keep_cols.append(c)

        # keep only existing columns
        keep_cols = [c for c in keep_cols if c in df.columns]

        # remove duplicates safely
        keep_cols = list(dict.fromkeys(keep_cols))

        # SAFE selection
        out = df.loc[:, keep_cols]

        # SAFE mask building
        exp_sel = out[exp_col]
        if isinstance(exp_sel, pd.DataFrame):
            exp_sel = exp_sel.iloc[:, 0]

        cl1_sel = out[cl1]
        if isinstance(cl1_sel, pd.DataFrame):
            cl1_sel = cl1_sel.iloc[:, 0]

        mask = exp_sel.notna() & cl1_sel.notna()

        out = out.loc[mask].reset_index(drop=True)
        return out.copy()

    def get_universe_arrays(
        self,
        exposure_id: str,
        include_features: bool = True,
    ) -> Dict[str, Any]:
        """Return a universe as NumPy arrays for fast simulation.

        Output keys
        -----------
        - dates: np.ndarray[datetime64[ns]]
        - spot: np.ndarray[float]
        - cl1: np.ndarray[float]
        - cl2: Optional[np.ndarray[float]]
        - features: Dict[str, np.ndarray]
        """
        df = self.get_universe(exposure_id=exposure_id, include_features=include_features)

        exp = self._get_exposure(exposure_id)
        exp_col = exp["price_column"]

        hedge_cfg = self.cfg["assets"]["hedge_instrument"]
        cl1 = hedge_cfg.get("price_column_front") or hedge_cfg.get("price_column") or "CL1"
        cl2 = hedge_cfg.get("price_column_next") or hedge_cfg.get("price_column_second")

        dates = pd.to_datetime(df[self.date_col]).dt.normalize().to_numpy(dtype="datetime64[ns]")
        spot = pd.to_numeric(df[exp_col], errors="coerce").to_numpy(dtype=float)
        cl1_arr = pd.to_numeric(df[cl1], errors="coerce").to_numpy(dtype=float)
        cl2_arr = None
        if cl2 is not None and cl2 in df.columns:
            cl2_arr = pd.to_numeric(df[cl2], errors="coerce").to_numpy(dtype=float)

        feat_dict: Dict[str, np.ndarray] = {}
        if include_features:
            for c in df.columns:
                if c in (self.date_col, exp_col, cl1, cl2):
                    continue
                # preserve dtype but coerce numeric features to float where possible
                if pd.api.types.is_numeric_dtype(df[c]):
                    feat_dict[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                else:
                    feat_dict[c] = df[c].to_numpy()

        # Updated/tradable flags (int8)
        spot_upd_col = f"{exp_col}__updated"
        cl1_upd_col = f"{cl1}__updated"
        if spot_upd_col in df.columns:
            spot_updated = pd.to_numeric(df[spot_upd_col], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        else:
            spot_updated = np.ones(len(df), dtype=np.int8)

        if cl1_upd_col in df.columns:
            cl1_updated = pd.to_numeric(df[cl1_upd_col], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        else:
            cl1_updated = np.ones(len(df), dtype=np.int8)

        if "__tradable" in df.columns:
            tradable = pd.to_numeric(df["__tradable"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        else:
            tradable = cl1_updated.copy()

        return {
            "dates": dates,
            "spot": spot,
            "cl1": cl1_arr,
            "cl2": cl2_arr,
            "features": feat_dict,
            "exposure_id": exposure_id,
            "spot_col": exp_col,
            "cl1_col": cl1,
            "cl2_col": cl2,
            "spot_updated": spot_updated,
            "cl1_updated": cl1_updated,
            "tradable": tradable,
        }

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    def _get_exposure(self, exposure_id: str) -> dict:
        """Return exposure config dict for the given exposure_id.

        Robust matching rules (in order):
        1) exact match on exposure['id']
        2) case-insensitive match on exposure['id']
        3) exact match on exposure['price_column']
        4) case-insensitive match on exposure['price_column']

        This makes CLI usage forgiving (e.g., --exposure WTI) even if the config
        uses lowercase ids or only specifies price_column.
        """
        exposures = self.cfg.get("assets", {}).get("exposures", [])
        if not isinstance(exposures, list):
            exposures = []

        if not isinstance(exposure_id, str) or not exposure_id:
            raise ValueError("exposure_id must be a non-empty string")

        key = exposure_id.strip()
        key_lo = key.lower()

        # 1) exact id
        for e in exposures:
            if isinstance(e, dict) and e.get("id") == key:
                return e

        # 2) case-insensitive id
        for e in exposures:
            if isinstance(e, dict) and isinstance(e.get("id"), str) and e.get("id").lower() == key_lo:
                return e

        # 3) exact price_column
        for e in exposures:
            if isinstance(e, dict) and e.get("price_column") == key:
                return e

        # 4) case-insensitive price_column
        for e in exposures:
            if isinstance(e, dict) and isinstance(e.get("price_column"), str) and e.get("price_column").lower() == key_lo:
                return e

        # Helpful error message
        known = []
        for e in exposures:
            if not isinstance(e, dict):
                continue
            _id = e.get("id")
            _pc = e.get("price_column")
            if isinstance(_id, str):
                known.append(_id)
            if isinstance(_pc, str) and _pc not in known:
                known.append(_pc)

        raise ValueError(f"Exposure not found: {exposure_id}. Known exposures: {known}")

    def get_aligned_data(self):
        return self.df_aligned.copy(deep=False)

    def get_universe_arrays_fast(
        self,
        exposure_id: str,
        include_features: bool = True,
        feature_role: str = "both",
    ) -> Dict[str, Any]:
        """Array-first universe builder (FAST, no intermediate DataFrame).

        تفاوت با get_universe_arrays:
        - این تابع مستقیماً از df_aligned و viewهای numpy استفاده می‌کند.
        - هیچ DataFrame جدیدی نمی‌سازد (فقط آرایه‌ها را ماسک می‌کند).
        """
        exp = self._get_exposure(exposure_id)
        exp_col = exp["price_column"]

        hedge_cfg = self.cfg["assets"]["hedge_instrument"]
        cl1 = hedge_cfg.get("price_column_front") or hedge_cfg.get("price_column") or "CL1"
        cl2 = hedge_cfg.get("price_column_next") or hedge_cfg.get("price_column_second")

        if exp_col not in self._col_view:
            raise ValueError(f"Exposure price column not found in master: {exp_col}")
        if cl1 not in self._col_view:
            raise ValueError(f"Hedge front-month column not found in master: {cl1}")

        spot_all = np.asarray(self._col_view[exp_col], dtype=float)
        cl1_all = np.asarray(self._col_view[cl1], dtype=float)

        # SAFE mask: require spot and CL1 present
        mask = np.isfinite(spot_all) & np.isfinite(cl1_all)

        dates = self._dates_np[mask]
        spot = spot_all[mask]
        cl1_arr = cl1_all[mask]

        cl2_arr = None
        if cl2 is not None and cl2 in self._col_view:
            c2 = self._col_view[cl2]
            # ensure float for numeric
            try:
                c2 = np.asarray(c2, dtype=float)
            except Exception:
                c2 = pd.to_numeric(pd.Series(c2), errors="coerce").to_numpy(dtype=float)
            cl2_arr = c2[mask]

        feat_dict: Dict[str, np.ndarray] = {}
        if include_features:
            features_cfg = self.cfg.get("features", None)
            feat_cols: List[str] = []

            if isinstance(features_cfg, list):
                for f in features_cfg:
                    if isinstance(f, str):
                        feat_cols.append(f)
                    elif isinstance(f, dict) and "column" in f:
                        feat_cols.append(f["column"])

            elif isinstance(features_cfg, dict):
                cols = features_cfg.get("columns", [])
                if isinstance(cols, list):
                    feat_cols += [c for c in cols if isinstance(c, str)]

                roles = features_cfg.get("roles", {})
                if isinstance(roles, dict):
                    # Apply role filter if requested
                    if feature_role in ("both", "baseline", "baseline_only"):
                        v = roles.get("baseline_only", [])
                        if isinstance(v, list):
                            feat_cols += [c for c in v if isinstance(c, str)]
                    if feature_role in ("both", "ai", "ai_only", "ai_feature_only"):
                        v = roles.get("ai_feature_only", [])
                        if isinstance(v, list):
                            feat_cols += [c for c in v if isinstance(c, str)]

            # remove duplicates, keep existing only
            feat_cols = list(dict.fromkeys([c for c in feat_cols if c in self._col_view]))

            for c in feat_cols:
                arr = self._col_view[c]
                # numeric columns are already float; object columns keep dtype
                feat_dict[c] = arr[mask]

        # Updated/tradable flags (pre-ffill). If missing, default to ones.
        spot_upd_key = f"{exp_col}__updated"
        cl1_upd_key = f"{cl1}__updated"
        if spot_upd_key in self._col_view:
            spot_updated = np.asarray(self._col_view[spot_upd_key], dtype=np.int8)[mask]
        else:
            spot_updated = np.ones(mask.sum(), dtype=np.int8)

        if cl1_upd_key in self._col_view:
            cl1_updated = np.asarray(self._col_view[cl1_upd_key], dtype=np.int8)[mask]
        else:
            cl1_updated = np.ones(mask.sum(), dtype=np.int8)

        if "__tradable" in self._col_view:
            tradable = np.asarray(self._col_view["__tradable"], dtype=np.int8)[mask]
        else:
            tradable = cl1_updated.copy()

        return {
            "dates": dates,
            "spot": spot,
            "cl1": cl1_arr,
            "cl2": cl2_arr,
            "features": feat_dict,
            "exposure_id": exposure_id,
            "spot_col": exp_col,
            "cl1_col": cl1,
            "cl2_col": cl2,
            "spot_updated": spot_updated,
            "cl1_updated": cl1_updated,
            "tradable": tradable,
        }