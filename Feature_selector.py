"""Feature_selector.py

Step 8 (prep): Feature screening via correlation matrices.

What this script does
---------------------
- Loads MasterData.csv (expects a Date column)
- Computes correlation matrices for:
    1) Levels (raw series)
    2) Simple returns (pct_change)
- Produces both Pearson and Spearman correlations
- Saves:
    - CSV matrices
    - PNG heatmaps
    - A short markdown report with key takeaways (top correlations)

Important cautions (tell it like it is)
---------------------------------------
- Correlation on LEVELS is often spurious for trending time series.
  You should primarily use RETURNS correlations for feature selection.
- Many macro/uncertainty indices are weekly/monthly forward-filled to daily.
  This can inflate correlation in levels; returns mitigate but can still be noisy.
- This is a first-pass screening tool, not a final feature selection method.

Usage
-----
python Feature_selector.py \
  --master MasterData.csv \
  --out_dir results/feature_selection \
  --method pearson \
  --mode returns \
  --min_non_nan 0.80

Modes: levels | returns | both
Methods: pearson | spearman | both

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_keep_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _matches_any_pattern(col: str, patterns: List[str]) -> bool:
    for pat in patterns:
        try:
            if re.fullmatch(pat, col) is not None:
                return True
        except re.error:
            # treat as literal if regex invalid
            if pat == col:
                return True
    return False


def _load_master(path: str) -> pd.DataFrame:
    p = str(path)
    if p.lower().endswith(".parquet"):
        df = pd.read_parquet(p)
        # Ensure Date exists
        if "Date" not in df.columns:
            raise ValueError("MasterData parquet must contain a 'Date' column")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df = pd.read_csv(p, parse_dates=["Date"], low_memory=False)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Date"] = df["Date"].dt.normalize()
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out = out.drop(columns=["Date"])
    # coerce to numeric (strings -> NaN)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _filter_columns_by_coverage(X: pd.DataFrame, min_non_nan: float, *, keep_cols: Optional[List[str]] = None, keep_patterns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    if not (0 < min_non_nan <= 1):
        raise ValueError("min_non_nan must be in (0,1].")

    keep_cols = keep_cols or []
    keep_patterns = keep_patterns or []

    frac = X.notna().mean()

    # Base keep by coverage
    keep = set(frac[frac >= min_non_nan].index.tolist())

    # Force-keep explicit columns
    for c in keep_cols:
        if c in X.columns:
            keep.add(c)

    # Force-keep pattern matches (regex fullmatch)
    for c in X.columns:
        if _matches_any_pattern(c, keep_patterns):
            keep.add(c)

    keep_list = [c for c in X.columns if c in keep]
    dropped = sorted(set(X.columns) - set(keep_list))
    if dropped:
        print(f"[FeatureSelector] Dropped {len(dropped)} columns by coverage < {min_non_nan} (excluding keep-list/patterns)")
    return X[keep_list], dropped


def _simple_returns(X: pd.DataFrame) -> pd.DataFrame:
    # IMPORTANT: do not pad/fill in pct_change; we rely on existing preprocessing.
    return X.pct_change(fill_method=None)


def _walkforward_windows(
    dates: pd.Series,
    *,
    train_years: int,
    val_years: int,
    test_years: int,
    step_years: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return list of (train_start, train_end, test_start, test_end) with an internal val block.

    Convention:
      train = [t0, t1)
      val   = [t1, t2)
      test  = [t2, t3)

    Windows move forward by `step_years`.
    """
    dmin = pd.to_datetime(dates.min()).normalize()
    dmax = pd.to_datetime(dates.max()).normalize()

    t0 = dmin
    out = []
    while True:
        t1 = (t0 + pd.DateOffset(years=int(train_years))).normalize()
        t2 = (t1 + pd.DateOffset(years=int(val_years))).normalize()
        t3 = (t2 + pd.DateOffset(years=int(test_years))).normalize()
        if t3 > dmax:
            break
        out.append((t0, t1, t2, t3))
        t0 = (t0 + pd.DateOffset(years=int(step_years))).normalize()
    return out


def _coverage_by_window(
    df: pd.DataFrame,
    cols: List[str],
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """Compute coverage (non-NaN fraction) per column for each train/val/test segment."""
    rows = []
    for wi, (t0, t1, t2, t3) in enumerate(windows, start=1):
        segs = [
            ("train", t0, t1),
            ("val", t1, t2),
            ("test", t2, t3),
        ]
        for seg_name, a, b in segs:
            m = (df["Date"] >= a) & (df["Date"] < b)
            X = df.loc[m, cols]
            if len(X) == 0:
                continue
            frac = X.notna().mean()
            for c in cols:
                rows.append((wi, seg_name, a, b, c, float(frac.get(c, np.nan)), int(len(X))))

    out = pd.DataFrame(
        rows,
        columns=["window_id", "segment", "start", "end", "feature", "coverage", "n_rows"],
    )
    return out


def _corr(X: pd.DataFrame, method: str) -> pd.DataFrame:
    return X.corr(method=method, min_periods=30)


def _save_corr_csv(C: pd.DataFrame, out_path: Path) -> None:
    C.to_csv(out_path, float_format="%.6f")


def _heatmap(C: pd.DataFrame, title: str, out_png: Path, *, vmin: float = -1.0, vmax: float = 1.0) -> None:
    # Basic matplotlib heatmap (no seaborn dependency)
    fig = plt.figure(figsize=(max(10, 0.25 * len(C.columns)), max(8, 0.25 * len(C.columns))))
    ax = fig.add_subplot(111)

    data = C.values
    im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xticks(range(len(C.columns)))
    ax.set_yticks(range(len(C.index)))
    ax.set_xticklabels(C.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(C.index, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)



def _top_pairs(C: pd.DataFrame, k: int = 25, abs_min: float = 0.7) -> pd.DataFrame:
    """Return top correlated pairs (upper triangle), excluding diagonal."""
    cols = C.columns
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = C.iloc[i, j]
            if np.isfinite(v) and abs(v) >= abs_min:
                rows.append((cols[i], cols[j], float(v), abs(float(v))))
    if not rows:
        return pd.DataFrame(columns=["feature_i", "feature_j", "corr", "abs_corr"])
    out = pd.DataFrame(rows, columns=["feature_i", "feature_j", "corr", "abs_corr"]).sort_values(
        "abs_corr", ascending=False
    )
    return out.head(k).reset_index(drop=True)


# --- Correlation pruning and grouping helpers ---

def _connected_groups(C: pd.DataFrame, abs_thr: float) -> List[List[str]]:
    """Connected components where |corr| >= abs_thr (excluding diagonal)."""
    cols = list(C.columns)
    n = len(cols)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    vals = C.values
    for i in range(n):
        for j in range(i + 1, n):
            v = vals[i, j]
            if np.isfinite(v) and abs(v) >= abs_thr:
                union(i, j)

    groups: Dict[int, List[str]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(cols[i])

    # Only return groups with size >= 2, sorted by size desc
    out = [sorted(g) for g in groups.values() if len(g) >= 2]
    out.sort(key=lambda g: (-len(g), g[0]))
    return out


def _greedy_prune_by_corr(
    X: pd.DataFrame,
    C: pd.DataFrame,
    abs_thr: float,
    *,
    priority: pd.Series,
) -> Tuple[List[str], pd.DataFrame]:
    """Greedy correlation pruning.

    Keep features in descending priority order. Drop any feature whose |corr| with an already-kept
    feature is >= abs_thr.

    Returns:
      kept_features, dropped_df (feature, dropped_because_of, corr)
    """
    cols = list(C.columns)
    # Ensure priority index covers all columns
    pr = priority.reindex(cols)
    pr = pr.fillna(pr.min() - 1)
    ordered = list(pr.sort_values(ascending=False).index)

    kept: List[str] = []
    dropped_rows = []

    for f in ordered:
        if f not in cols:
            continue
        # Compare with kept
        drop = False
        for k in kept:
            v = C.loc[f, k]
            if np.isfinite(v) and abs(float(v)) >= abs_thr:
                dropped_rows.append((f, k, float(v)))
                drop = True
                break
        if not drop:
            kept.append(f)

    dropped_df = pd.DataFrame(dropped_rows, columns=["feature", "kept_feature", "corr"])
    dropped_df["abs_corr"] = dropped_df["corr"].abs()
    dropped_df = dropped_df.sort_values(["abs_corr", "feature"], ascending=[False, True]).reset_index(drop=True)
    return kept, dropped_df



def _write_report_md(
    out_dir: Path,
    mode: str,
    method: str,
    dropped_cols: List[str],
    used_cols: List[str],
    top_df: pd.DataFrame,
    *,
    prune_enabled: bool,
    prune_abs_thr: float,
    kept_features: Optional[List[str]] = None,
    dropped_corr_df: Optional[pd.DataFrame] = None,
    groups: Optional[List[List[str]]] = None,
) -> None:
    p = out_dir / "feature_correlation_report.md"
    with p.open("w", encoding="utf-8") as f:
        f.write(f"# گزارش غربال اولیه فیچرها با ماتریس همبستگی\n\n")
        f.write(f"**Mode:** {mode}  \\\n")
        f.write(f"**Method:** {method}  \\\n")
        f.write(f"**تعداد ستون‌های استفاده‌شده:** {len(used_cols)}  \\\n")
        f.write(f"**Correlation pruning:** {'ON' if prune_enabled else 'OFF'}  \\\n\n")

        if dropped_cols:
            f.write("## ستون‌های حذف‌شده به‌دلیل کمبود داده\n\n")
            f.write(f"min_non_nan فیلتر باعث حذف {len(dropped_cols)} ستون شد.\n\n")
            f.write("- " + "\n- ".join(dropped_cols) + "\n\n")

        f.write("## نکات کلیدی (هشدارهای روش‌شناسی)\n\n")
        f.write(
            "- همبستگی روی **سطح قیمت/شاخص** ممکن است کاذب باشد (روند و هم‌انباشتگی).\n"
            "- برای انتخاب فیچر، همبستگی روی **بازده** معمولاً معنادارتر است.\n"
            "- فیچرهای هفتگی/ماهانه که به روزانه forward-fill شده‌اند می‌توانند همبستگی‌های مصنوعی بسازند.\n\n"
        )

        f.write("## همبستگی‌های بزرگ (|corr| ≥ 0.70)\n\n")
        if top_df.empty:
            f.write("هیچ زوجی با |corr| ≥ 0.70 پیدا نشد (با فیلتر فعلی).\n")
        else:
            f.write(top_df.to_markdown(index=False))
            f.write("\n")

        f.write("\n## خروجی انتخاب فیچر برای LSTM (Correlation Pruning)\n\n")
        if not prune_enabled:
            f.write("Correlation pruning غیرفعال است.\n")
        else:
            f.write(f"آستانه حذف همبستگی: |corr| ≥ {prune_abs_thr:.2f}\n\n")
            if kept_features is None:
                f.write("لیست فیچرهای منتخب تولید نشد (خطا/عدم اجرا).\n")
            else:
                f.write(f"تعداد فیچرهای منتخب: {len(kept_features)}\n\n")

            if dropped_corr_df is not None and not dropped_corr_df.empty:
                f.write("### فیچرهای حذف‌شده به دلیل همبستگی بالا\n\n")
                f.write(dropped_corr_df.head(30).to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("هیچ فیچری به دلیل همبستگی بالا حذف نشد (با آستانه فعلی).\n\n")

            if groups:
                f.write("### گروه‌های همبستگی (Connected Components)\n\n")
                # show up to 15 groups
                for gi, g in enumerate(groups[:15], start=1):
                    f.write(f"- Group {gi} (size={len(g)}): " + ", ".join(g) + "\n")
                f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, default="MasterData.parquet")
    ap.add_argument("--out_dir", type=str, default="results/feature_selection")
    ap.add_argument("--mode", type=str, choices=["levels", "returns", "both"], default="both")
    ap.add_argument("--method", type=str, choices=["pearson", "spearman", "both"], default="both")
    ap.add_argument("--min_non_nan", type=float, default=0.80)
    ap.add_argument("--abs_min", type=float, default=0.70)
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--prune", action="store_true", help="Greedy prune highly correlated features and export LSTM-ready lists")
    ap.add_argument("--corr_thr", type=float, default=0.90, help="Absolute correlation threshold for pruning/groups")
    ap.add_argument(
        "--priority",
        type=str,
        choices=["coverage", "variance"],
        default="coverage",
        help="Priority for keeping features during pruning",
    )
    ap.add_argument(
        "--keep_cols",
        type=str,
        default="",
        help="Comma-separated column names to keep even if coverage is below min_non_nan (e.g., OPEC,OVX)",
    )
    ap.add_argument(
        "--keep_patterns",
        type=str,
        default="",
        help="Comma-separated regex patterns (fullmatch) of columns to keep even if sparse (e.g., '^OPEC.*$')",
    )

    ap.add_argument("--walkforward", action="store_true", help="Compute per-window coverage diagnostics for walk-forward training")
    ap.add_argument("--train_years", type=int, default=2)
    ap.add_argument("--val_years", type=int, default=1)
    ap.add_argument("--test_years", type=int, default=1)
    ap.add_argument("--step_years", type=int, default=1)
    ap.add_argument(
        "--min_non_nan_window",
        type=float,
        default=0.0,
        help="Optional: flag features with coverage below this threshold in ANY train/val/test segment (diagnostic only)",
    )
    args = ap.parse_args()

    out_dir = _ensure_dir(args.out_dir)

    print(f"[FeatureSelector] Loading: {args.master}")
    df = _load_master(args.master)
    X0 = _numeric_frame(df)

    # coverage filtering
    cov = X0.notna().mean().sort_values(ascending=False)
    cov.to_csv(out_dir / "coverage_by_column.csv", float_format="%.6f")

    keep_cols = _parse_keep_list(args.keep_cols)
    keep_patterns = _parse_keep_list(args.keep_patterns)

    X, dropped_cols = _filter_columns_by_coverage(X0, args.min_non_nan, keep_cols=keep_cols, keep_patterns=keep_patterns)
    used_cols = list(X.columns)

    if args.walkforward:
        windows = _walkforward_windows(
            df["Date"],
            train_years=int(args.train_years),
            val_years=int(args.val_years),
            test_years=int(args.test_years),
            step_years=int(args.step_years),
        )
        if not windows:
            print("[FeatureSelector] walkforward: no complete windows can be formed with the provided years.")
        else:
            cov_w = _coverage_by_window(df=df, cols=used_cols, windows=windows)
            cov_w.to_csv(out_dir / "coverage_by_window.csv", index=False)

            if args.min_non_nan_window and args.min_non_nan_window > 0:
                thr = float(args.min_non_nan_window)
                bad = cov_w[cov_w["coverage"] < thr].copy()
                bad.to_csv(out_dir / "low_coverage_flags_by_window.csv", index=False)

    # prepare datasets
    datasets: List[Tuple[str, pd.DataFrame]] = []
    if args.mode in ("levels", "both"):
        datasets.append(("levels", X.copy()))
    if args.mode in ("returns", "both"):
        datasets.append(("returns", _simple_returns(X)))

    methods = [args.method] if args.method != "both" else ["pearson", "spearman"]

    # compute and save
    all_top = []
    kept_features = None
    dropped_corr_df = None
    groups = None
    for mode_name, Xm in datasets:
        for method in methods:
            print(f"[FeatureSelector] Computing corr | mode={mode_name} | method={method}")
            C = _corr(Xm, method=method)

            csv_path = out_dir / f"corr_{mode_name}_{method}.csv"
            _save_corr_csv(C, csv_path)

            png_path = out_dir / f"corr_{mode_name}_{method}.png"
            _heatmap(C, title=f"Correlation ({mode_name}, {method})", out_png=png_path)

            top_df = _top_pairs(C, k=args.top_k, abs_min=args.abs_min)
            top_df.to_csv(out_dir / f"top_pairs_{mode_name}_{method}.csv", index=False)
            top_df["mode"] = mode_name
            top_df["method"] = method
            all_top.append(top_df)

            kept_features = None
            dropped_corr_df = None
            groups = None

            if args.prune:
                # Priority: coverage (default) or variance over Xm
                if args.priority == "coverage":
                    pr = Xm.notna().mean()
                else:
                    pr = Xm.var(skipna=True)

                # Build connected groups at the same threshold
                groups = _connected_groups(C, abs_thr=float(args.corr_thr))

                kept_features, dropped_corr_df = _greedy_prune_by_corr(
                    Xm,
                    C,
                    abs_thr=float(args.corr_thr),
                    priority=pr,
                )

                # Write selection artifacts
                sel_path = out_dir / f"selected_features_{mode_name}_{method}.csv"
                pd.DataFrame({"feature": kept_features}).to_csv(sel_path, index=False)

                drop_path = out_dir / f"dropped_correlated_{mode_name}_{method}.csv"
                dropped_corr_df.to_csv(drop_path, index=False)

                grp_rows = []
                for gid, g in enumerate(groups or [], start=1):
                    for feat in g:
                        grp_rows.append((gid, len(g), feat))
                grp_df = pd.DataFrame(grp_rows, columns=["group_id", "group_size", "feature"])
                grp_df.to_csv(out_dir / f"correlation_groups_{mode_name}_{method}.csv", index=False)

    # unified report
    top_all = pd.concat(all_top, ignore_index=True) if all_top else pd.DataFrame()
    if not top_all.empty:
        top_all.to_csv(out_dir / "top_pairs_all.csv", index=False)

    # write md report for the last computed block, plus general info
    # (If both modes/methods, report is still useful as a single file.)
    _write_report_md(
        out_dir=out_dir,
        mode=args.mode,
        method=args.method,
        dropped_cols=dropped_cols,
        used_cols=used_cols,
        top_df=(top_all.sort_values("abs_corr", ascending=False).head(args.top_k) if not top_all.empty else top_all),
        prune_enabled=bool(args.prune),
        prune_abs_thr=float(args.corr_thr),
        kept_features=(kept_features if "kept_features" in locals() else None),
        dropped_corr_df=(dropped_corr_df if "dropped_corr_df" in locals() else None),
        groups=(groups if "groups" in locals() else None),
    )

    print(f"[FeatureSelector] Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()