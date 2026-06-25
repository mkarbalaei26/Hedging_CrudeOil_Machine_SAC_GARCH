#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train RL agent on walkforward windows with multiple train/val/test splits.

The data is split into multiple windows, each with train/val/test years.
Training is done on each window separately.

The default mode is rolling windows: fixed train length, rolling forward.
The new mode 'hybrid_expanding' expands the training window from a fixed start.

Usage example:
python -m rl.train_walkforward --cache ... --train_mode hybrid_expanding ...

"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Other imports and RL training code omitted for brevity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--exposure_id", required=True)
    parser.add_argument("--scenario_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--train_years", type=int, default=2)
    parser.add_argument("--val_years", type=int, default=1)
    parser.add_argument("--test_years", type=int, default=1)

    parser.add_argument("--train_mode", default="rolling", choices=["rolling", "hybrid_expanding", "expanding"])

    # Other args omitted for brevity

    args = parser.parse_args()

    # Load scenario years etc. (code omitted)

    # Determine available years for walkforward
    years_available = sorted(get_available_years(args.scenario_dir))
    print(f"[walkforward] years available: {years_available}")

    train_years = args.train_years
    val_years = args.val_years
    test_years = args.test_years

    # Compute first_test_year as before:
    first_test_year = years_available[0] + train_years + val_years

    base_train_start = first_test_year - (train_years + val_years)

    windows = []

    for test_year in years_available:
        # Skip years before first_test_year
        if test_year < first_test_year:
            continue

        if args.train_mode == "rolling":
            train_start = test_year - (train_years + val_years)
            train_end = train_start + train_years - 1
            val_year = train_end + 1
            test_end = val_year + test_years - 1

        elif args.train_mode in ("hybrid_expanding", "expanding"):
            train_start = base_train_start
            train_end = test_year - val_years - 1
            val_year = train_end + 1
            test_end = val_year + test_years - 1

            if train_end < train_start:
                print(f"[walkforward] skipping window with train_end {train_end} < train_start {train_start}")
                continue

        else:
            print(f"Unknown train_mode {args.train_mode}", file=sys.stderr)
            sys.exit(1)

        label = f"WF_train{train_start}-{train_end}_val{val_year}_test{test_end}"
        print(f"[walkforward] window: {label}")

        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "val_year": val_year,
            "test_start": val_year,
            "test_end": test_end,
            "label": label,
        })

    # Proceed with training on each window (code omitted)

if __name__ == "__main__":
    main()