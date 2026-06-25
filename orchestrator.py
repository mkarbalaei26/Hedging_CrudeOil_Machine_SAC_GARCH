#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
orchestrator.py

High-throughput batch runner for hedge_simulator.py with:
- N concurrent processes (outer parallelism)
- each process can use --jobs (inner parallelism inside simulator)
- live progress per process by parsing tqdm lines
- smoke mode: run only first N scenario rows
- chunking mode: split scenario parquet into smaller parts, run each, then merge summaries
- resume: skip runs that already have output
- final merge: per (exposure, scenario_kind) into 12 final parquet files

Designed for macOS (no stdbuf needed).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ----------------------------
# CONFIG: scenarios & grid
# ----------------------------
SCENARIOS: Dict[str, Dict[str, str]] = {
    "WTI_SPOT": {
        "baseline": "scenarios/WTI_SPOT/baseline.parquet",
        "oracle_all": "scenarios/WTI_SPOT/oracle_all.parquet",
        "oracle_universe": "scenarios/WTI_SPOT/oracle_universe.parquet",
        "company": "scenarios/WTI_SPOT/companies.parquet",
    },
    "BRENT_SPOT": {
        "baseline": "scenarios/BRENT_SPOT/baseline.parquet",
        "oracle_all": "scenarios/BRENT_SPOT/oracle_all.parquet",
        "oracle_universe": "scenarios/BRENT_SPOT/oracle_universe.parquet",
        "company": "scenarios/BRENT_SPOT/companies.parquet",
    },
    "OPEC_BASKET": {
        "baseline": "scenarios/OPEC_BASKET/baseline.parquet",
        "oracle_all": "scenarios/OPEC_BASKET/oracle_all.parquet",
        "oracle_universe": "scenarios/OPEC_BASKET/oracle_universe.parquet",
        "company": "scenarios/OPEC_BASKET/companies.parquet",
    },
}

STR_BASE = ["nohedge", "naive", "ols_static"]
STR_ROLL = ["ols_roll", "ccc_garch", "dcc_garch"]

W_MAIN = [120, 252]
W_ALL = [30, 60, 120, 252]

# ----------------------------
# Scenario validation helper
# ----------------------------

def run_validate_pipeline(py: str, scenarios_root: str, exposure: str) -> None:
    """Run validate_pipeline.py as a subprocess and raise on failure."""
    cmd = [py, "validate_pipeline.py", "--scenarios_root", scenarios_root, "--exposure", exposure]
    print("[VALIDATE] " + " ".join(cmd))
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


@dataclass(frozen=True)
class RunSpec:
    exposure: str
    scenario_kind: str     # baseline/oracle_all/oracle_universe/company
    scenario_path: str
    strategy: str
    window: Optional[int]  # None for base strategies
    mode: str              # static/dynamic
    roll: str              # roll/noroll
    chunk_id: Optional[int] = None
    chunk_path: Optional[str] = None
    chunk_total: Optional[int] = None

    def key(self) -> str:
        w = "NA" if self.window is None else str(self.window)
        ch = "full" if self.chunk_id is None else f"chunk{self.chunk_id:04d}"
        return f"{self.exposure}|{self.scenario_kind}|{self.strategy}|w{w}|{self.mode}|{self.roll}|{ch}"

    def chunk_label(self) -> str:
        if self.chunk_id is None or not self.chunk_total or self.chunk_total <= 1:
            return ""
        # chunk_id is 0-based
        return f"chunk {self.chunk_id+1}/{self.chunk_total}"


def build_grid(include_company: bool) -> List[RunSpec]:
    """Build the full run list with strict global ordering.

    Ordering enforced:
      - exposures: OPEC_BASKET -> WTI_SPOT -> BRENT_SPOT
      - kinds: (baseline, oracle_all) for ALL exposures, then oracle_universe for ALL exposures,
              then company for ALL exposures (if enabled)

    Within each (exposure, kind) block we keep the same phase ordering as the bash grid.
    """
    runs: List[RunSpec] = []

    exposure_order = ["OPEC_BASKET", "WTI_SPOT", "BRENT_SPOT"]
    phase_kinds: List[List[str]] = [
        ["baseline", "oracle_all"],
        ["oracle_universe"],
    ]
    if include_company:
        phase_kinds.append(["company"])

    for kinds in phase_kinds:
        for exposure in exposure_order:
            d = SCENARIOS[exposure]
            for kind in kinds:
                scen = d[kind]

                # Phase 1 (important)
                for strat in STR_BASE:
                    runs.append(RunSpec(exposure, kind, scen, strat, None, "static", "roll"))

                for w in W_MAIN:
                    for strat in STR_ROLL:
                        runs.append(RunSpec(exposure, kind, scen, strat, w, "dynamic", "roll"))

                # Phase 2A
                for strat in STR_BASE:
                    runs.append(RunSpec(exposure, kind, scen, strat, None, "dynamic", "roll"))
                    runs.append(RunSpec(exposure, kind, scen, strat, None, "static", "noroll"))
                    runs.append(RunSpec(exposure, kind, scen, strat, None, "dynamic", "noroll"))

                # Phase 2B
                for w in W_ALL:
                    for strat in STR_ROLL:
                        runs.append(RunSpec(exposure, kind, scen, strat, w, "static", "roll"))
                        runs.append(RunSpec(exposure, kind, scen, strat, w, "dynamic", "roll"))
                        runs.append(RunSpec(exposure, kind, scen, strat, w, "static", "noroll"))
                        runs.append(RunSpec(exposure, kind, scen, strat, w, "dynamic", "noroll"))

    return runs


# ----------------------------
# Deduplicate runs helper
# ----------------------------

def dedupe_runs(runs: List[RunSpec]) -> List[RunSpec]:
    """Drop exact-duplicate RunSpec entries while preserving order.

    Duplicates can happen because the grid has a Phase-1 "main" set and a Phase-2 "all" set
    that may include overlapping combinations (e.g., w=120 dynamic roll appears in both).

    We consider a run identical if all execution-relevant fields match, including chunk id.
    """
    seen: set[tuple] = set()
    out: List[RunSpec] = []
    for r in runs:
        k = (
            r.exposure,
            r.scenario_kind,
            r.scenario_path,
            r.strategy,
            r.window,
            r.mode,
            r.roll,
            r.chunk_id,
            r.chunk_path,
        )
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


# ----------------------------
# Chunking / smoke
# ----------------------------

def load_scenarios(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)

    if p.suffix.lower() == ".npz":
        z = np.load(str(p), allow_pickle=True)
        out = {
            "scenario_id": z["scenario_id"],
            "start_date": pd.to_datetime(z["start_date"], errors="coerce").dt.normalize().astype("datetime64[ns]"),
            "end_date": pd.to_datetime(z["end_date"], errors="coerce").dt.normalize().astype("datetime64[ns]"),
            "volume_bbl": z["volume_bbl"].astype(int),
            "horizon_days_target": z["horizon_days_target"].astype(int),
            "horizon_days_realized": z["horizon_days_realized"].astype(int),
        }
        for k in ["company_id", "company_size", "oracle_series"]:
            if k in z.files:
                out[k] = z[k]
        return pd.DataFrame(out)

    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)

    # fallback
    return pd.read_parquet(p)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def make_variant_scenario_file(
    base_path: str,
    tmp_root: Path,
    exposure: str,
    kind: str,
    smoke_n: Optional[int],
    chunk_size: Optional[int],
) -> Tuple[int, List[Tuple[Optional[int], str]]]:
    """Create scenario variant parquet(s) once per (exposure, kind).

    Returns:
      (chunk_total, [(chunk_id, path), ...])

    - smoke_n: if set, keep only first N rows (after load)
    - chunk_size: if set, split into multiple parquet chunks
    """
    df = load_scenarios(base_path)

    if smoke_n is not None:
        df = df.head(smoke_n).copy()

    # if no chunking
    if not chunk_size or chunk_size <= 0 or len(df) <= chunk_size:
        out = tmp_root / exposure / kind / "scenarios_full.parquet"
        write_parquet(df, out)
        return 1, [(None, str(out))]

    chunks: List[Tuple[Optional[int], str]] = []
    n = len(df)
    k = 0
    for start in range(0, n, chunk_size):
        part = df.iloc[start : start + chunk_size].copy()
        out = tmp_root / exposure / kind / f"scenarios_chunk_{k:04d}.parquet"
        write_parquet(part, out)
        chunks.append((k, str(out)))
        k += 1

    return k, chunks


# ----------------------------
# Resume / output detection
# ----------------------------

# Fast resume indexer
def build_output_index(base_out: Path) -> Dict[Tuple[str, str], set[str]]:
    """Pre-index hedge_summary parquet filenames for fast resume checks.

    Returns a dict keyed by (exposure, scen_stem) -> set of filenames under the expected output dirs.
    """
    idx: Dict[Tuple[str, str], set[str]] = {}
    for exposure, kinds in SCENARIOS.items():
        for kind, scen_path in kinds.items():
            scen_stem = Path(scen_path).stem
            key = (exposure, scen_stem)
            idx.setdefault(key, set())
            for d in [base_out / exposure / scen_stem, base_out / scen_stem]:
                if not d.exists():
                    continue
                # Only index the filenames we care about
                for p in d.rglob("hedge_summary_*.parquet"):
                    idx[key].add(p.name)
    return idx

def run_tag(base_tag: str, spec: RunSpec) -> str:
    """
    CRITICAL: make tag unique per run to avoid overwriting across windows/modes.
    """
    w = "NA" if spec.window is None else str(spec.window)
    ch = "full" if spec.chunk_id is None else f"c{spec.chunk_id:04d}"
    return f"{base_tag}__{spec.scenario_kind}__{spec.strategy}__w{w}__{spec.mode}__{spec.roll}__{ch}"


def expected_out_dirs(base_out: Path, spec: RunSpec) -> List[Path]:
    """Return candidate output directories for this run.

    NOTE: Depending on simulator version/flags, outputs may be written either:
      1) <out_dir>/<exposure>/<scenario_stem>/...
      2) <out_dir>/<scenario_stem>/...

    We check both so resume/merge work reliably.
    """
    scen_stem = Path(spec.scenario_path).stem
    return [
        base_out / spec.exposure / scen_stem,
        base_out / scen_stem,
    ]


def output_exists(
    base_out: Path,
    base_tag: str,
    spec: RunSpec,
    *,
    any_tag: bool = False,
    index: Optional[Dict[Tuple[str, str], set[str]]] = None
) -> bool:
    """Check whether the output for this *exact* RunSpec already exists.

    Important fixes:
    - Only search within the expected scenario directory for this spec.
      Searching under base_out caused false positives across exposures/kinds.
    - In any_tag mode we still require the filename to contain the unique
      run signature tokens (kind/strategy/window/mode/roll + chunk if any).
      This prevents accidentally skipping other windows/modes/rolls.
    """

    scen_dirs = expected_out_dirs(base_out, spec)
    existing_dirs = [d for d in scen_dirs if d.exists()]
    # Fast path: if an index is provided, avoid filesystem walks
    scen_stem = Path(spec.scenario_path).stem
    if index is not None:
        names = index.get((spec.exposure, scen_stem), set())
        if not names:
            return False
    w = "NA" if spec.window is None else str(spec.window)
    chunk_token = "full" if spec.chunk_id is None else f"c{spec.chunk_id:04d}"

    # --- Exact tag mode -------------------------------------------------
    if not any_tag:
        tag = run_tag(base_tag, spec)
        suffix = f"_{tag}.parquet"
        if index is not None:
            return any(n.endswith(suffix) and n.startswith("hedge_summary_") for n in names)
        pattern = f"hedge_summary_*_{tag}.parquet"
        for d in existing_dirs:
            if any(d.rglob(pattern)):
                return True
        return False

    # --- Any-tag mode ---------------------------------------------------
    # We only treat it as existing if the file name contains the orchestrator
    # signature that encodes the run parameters.
    sig = f"__{spec.scenario_kind}__{spec.strategy}__w{w}__{spec.mode}__{spec.roll}__"

    iterable = names if index is not None else None
    if iterable is None:
        iterable = []
        for d in existing_dirs:
            for p in d.rglob("hedge_summary_*.parquet"):
                iterable.append(p.name)

    for name in iterable:
        if sig not in name:
            continue
        # Chunked runs must match the exact chunk id
        if spec.chunk_id is not None:
            if f"__{chunk_token}__" in name or name.endswith(f"__{chunk_token}.parquet"):
                return True
        else:
            # Non-chunked (full) run
            if "__full__" in name or name.endswith("__full.parquet"):
                return True

    return False


# ----------------------------
# Live progress parsing (tqdm)
# ----------------------------

TQDM_RE = re.compile(r"(\d{1,3})%\|")  # matches " 33%|" in tqdm line
TQDM_COUNTS_RE = re.compile(r"\b(\d{1,12})/(\d{1,12})\b")  # matches "104250/104250"
TQDM_RATE_RE = re.compile(r"(\d+(?:\.\d+)?)it/s")          # matches "51.07it/s"


def now_utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}h {m:02d}m {s:02d}s"


@dataclass
class ProcState:
    spec: RunSpec
    started: float
    pct: int = 0
    n_done: int = 0
    n_total: int = 0
    it_per_s: float = 0.0
    last_line: str = ""
    rc: Optional[int] = None
    log_path: Optional[str] = None


# ----------------------------
# Runner
# ----------------------------


async def run_one_process(
    py: str,
    sim: str,
    master: str,
    config: str,
    price_engine: str,
    base_out: Path,
    logs: Path,
    base_tag: str,
    jobs_inner: int,
    spec: RunSpec,
    live: Dict[str, ProcState],
) -> ProcState:
    state = ProcState(spec=spec, started=time.time())

    logs.mkdir(parents=True, exist_ok=True)
    log_name = spec.key().replace("|", "__").replace("/", "_").replace(":", "_")
    log_path = logs / f"run_{log_name}.log"
    state.log_path = str(log_path)

    tag = run_tag(base_tag, spec)
    scenario_file = spec.chunk_path or spec.scenario_path

    cmd = [
        py, sim,
        "--master", master,
        "--config", config,
        "--price_engine", price_engine,
        "--exposure", spec.exposure,
        "--scenarios", scenario_file,
        "--scenario_stem", Path(spec.scenario_path).stem,
        "--strategies", spec.strategy,
        "--out_dir", str(base_out),
        "--tag", tag,
        "--jobs", str(jobs_inner),
    ]

    if spec.window is not None:
        cmd += ["--window", str(spec.window)]
    cmd += ["--dynamic" if spec.mode == "dynamic" else "--static"]
    if spec.roll == "noroll":
        cmd += ["--no_roll"]

    # Start process
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    # stream to log and parse progress (robust against very long lines)
    buffer = ""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== START {now_utc()} ===\n")
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        assert proc.stdout is not None

        while True:
            chunk = await proc.stdout.read(8192)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            # tqdm often updates lines with '\r' (carriage return) without '\n'
            # Convert '\r' into '\n' so we can parse progress updates reliably.
            text = text.replace("\r", "\n")
            buffer += text

            # flush complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                f.write(line + "\n")

                state.last_line = line

                m = TQDM_RE.search(line)
                if m:
                    try:
                        state.pct = int(m.group(1))
                    except Exception:
                        pass

                m2 = TQDM_COUNTS_RE.search(line)
                if m2:
                    try:
                        state.n_done = int(m2.group(1))
                        state.n_total = int(m2.group(2))
                    except Exception:
                        pass

                m3 = TQDM_RATE_RE.search(line)
                if m3:
                    try:
                        state.it_per_s = float(m3.group(1))
                    except Exception:
                        pass

                live[spec.key()] = state

        # write any remaining buffered tail as one line
        tail = buffer.strip("\r\n")
        if tail:
            f.write(tail + "\n")
            state.last_line = tail
            live[spec.key()] = state

        rc = await proc.wait()
        state.rc = rc
        f.write(f"\n=== END {now_utc()} | rc={rc} ===\n")
    live[spec.key()] = state
    return state



def print_dashboard(
    total: int,
    done: int,
    running: List[ProcState],
    ok: int,
    fail: int,
    skipped: int,
    start_t: float,
) -> None:
    # Clearing the screen is expensive; caller controls whether to print dashboard.
    os.system("clear" if os.name != "nt" else "cls")
    elapsed = time.time() - start_t

    eta = "N/A"
    if done > 0:
        eta_s = elapsed * (total - done) / done
        eta = fmt_dur(eta_s)

    print(
        f"[{now_utc()}]  TOTAL={total}  DONE={done}  OK={ok}  FAIL={fail}  SKIP={skipped}  "
        f"ELAPSED={fmt_dur(elapsed)}  ETA={eta}"
    )
    print("-" * 120)

    if not running:
        print("(no running processes)")
        return

    # Header
    print(f"{'%':>3}  {'DUR':>10}  {'CHUNK':>10}  {'REC':>17}  {'RATE':>8}  RUN")
    print("-" * 120)

    for st in running[:25]:
        dur = fmt_dur(time.time() - st.started)
        ch = st.spec.chunk_label() or "-"
        rec = "-"
        if st.n_total > 0:
            rec = f"{st.n_done}/{st.n_total}"
        rate = "-"
        if st.it_per_s > 0:
            rate = f"{st.it_per_s:.2f}"
        run_name = st.spec.key()
        print(f"{st.pct:3d}  {dur:>10}  {ch:>10}  {rec:>17}  {rate:>8}  {run_name}")

        if st.last_line:
            ll = st.last_line
            if len(ll) > 160:
                ll = ll[:157] + "..."
            print(f"      {ll}")


async def run_batch(
    runs: List[RunSpec],
    *,
    py: str,
    sim: str,
    master: str,
    config: str,
    price_engine: str,
    out_dir: str,
    tag: str,
    logs_dir: str,
    procs_outer: int,
    jobs_inner: int,
    resume: bool,
    resume_any_tag: bool,
    no_dashboard: bool,
    dashboard_interval: float,
) -> None:
    base_out = Path(out_dir)
    logs = Path(logs_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    # Fast output index for resume
    out_index = build_output_index(base_out) if resume else None

    # queue with resume filtering
    queue: List[RunSpec] = []
    skipped = 0
    total_all = len(runs)
    for r in runs:
        if resume and output_exists(base_out, tag, r, any_tag=resume_any_tag, index=out_index):
            skipped += 1
            continue
        queue.append(r)

    total = len(queue)
    print(f"[INFO] queue={total} | skipped={skipped} | total_in_grid={total_all} | resume={resume} | any_tag={resume_any_tag}")
    ok = 0
    fail = 0
    done = 0
    start_t = time.time()

    # active tasks: list of (asyncio.Task, ProcState placeholder)
    active: List[Tuple[asyncio.Task, ProcState]] = []

    # live state for dashboard
    live: Dict[str, ProcState] = {}

    # seed initial
    def make_placeholder(spec: RunSpec) -> ProcState:
        return ProcState(spec=spec, started=time.time())

    async def launch(spec: RunSpec) -> None:
        nonlocal active
        ph = make_placeholder(spec)
        t = asyncio.create_task(run_one_process(
            py=py, sim=sim, master=master, config=config, price_engine=price_engine,
            base_out=base_out, logs=logs, base_tag=tag, jobs_inner=jobs_inner, spec=spec, live=live
        ))
        active.append((t, ph))

    # fill
    while queue and len(active) < procs_outer:
        await launch(queue.pop(0))

    last_dash = 0.0
    # main loop
    while active:
        # refresh dashboard from placeholders + completed states we have
        running_specs = [ph.spec for (_, ph) in active]
        running_states = []
        for sp in running_specs:
            running_states.append(live.get(sp.key(), ProcState(spec=sp, started=live.get(sp.key(), ProcState(spec=sp, started=time.time())).started)))
        if (not no_dashboard) and (time.time() - last_dash >= dashboard_interval):
            print_dashboard(total=total, done=done, running=running_states, ok=ok, fail=fail, skipped=skipped, start_t=start_t)
            last_dash = time.time()

        # wait for any to finish (short timeout so dashboard updates)
        done_tasks, _ = await asyncio.wait([t for (t, _) in active], timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
        if not done_tasks:
            continue

        new_active: List[Tuple[asyncio.Task, ProcState]] = []
        for t, ph in active:
            if t in done_tasks:
                st: ProcState = t.result()
                done += 1
                if st.rc == 0:
                    ok += 1
                else:
                    fail += 1
                # after completion, launch next if exists
                if queue:
                    await launch(queue.pop(0))
            else:
                # update placeholder with whatever we can glean? (keep last_line/pct from ph)
                new_active.append((t, ph))
        active = new_active

    # final
    elapsed = time.time() - start_t
    os.system("clear" if os.name != "nt" else "cls")
    print(f"✅ FINISHED  ok={ok} fail={fail} skipped={skipped} queue_done={done}/{total}  elapsed={fmt_dur(elapsed)}")
    print(f"out_dir={out_dir}")
    print(f"logs_dir={logs_dir}")


# ----------------------------
# Final merge (per 12 outputs)
# ----------------------------

def merge_outputs(
    out_dir: str,
    tag_prefix: str,
    out_merged_dir: str,
    include_company: bool,
) -> None:
    """Merge all hedge_summary outputs into exactly 12 parquet files (3 exposures × 4 kinds).

    We always write the merged files even if some buckets have no inputs (empty parquet).
    """
    base_out = Path(out_dir)
    merged_root = Path(out_merged_dir)
    merged_root.mkdir(parents=True, exist_ok=True)

    def _empty_summary_df() -> pd.DataFrame:
        cols = [
            "scenario_id", "exposure_id", "start_date", "end_date", "horizon_days", "volume_bbl",
            "strategy", "dynamic", "mode_roll",
            "spot_pnl_total", "fut_pnl_total", "cost_trade_total", "cost_roll_total", "net_pnl_total",
            "turnover_contracts", "turnover_h", "trade_contracts", "roll_contracts",
            "max_abs_contracts", "mdd_equity",
        ]
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    exposure_order = ["OPEC_BASKET", "WTI_SPOT", "BRENT_SPOT"]
    kind_order = ["baseline", "oracle_all", "oracle_universe", "company"]

    for exposure in exposure_order:
        for kind in kind_order:
            if kind == "company" and (not include_company):
                continue
            scen_path = SCENARIOS[exposure].get(kind)
            if scen_path is None:
                continue

            scen_stem = Path(scen_path).stem
            scen_dirs = [
                base_out / exposure / scen_stem,
                base_out / scen_stem,
            ]

            files: List[Path] = []
            for scen_dir in scen_dirs:
                if not scen_dir.exists():
                    continue
                # merge only files that belong to this tag_prefix
                files.extend(list(scen_dir.glob(f"hedge_summary_*_{tag_prefix}*.parquet")))

            # fallback: allow orchestrator-tagged outputs that may not start with tag_prefix
            if not files:
                for scen_dir in scen_dirs:
                    if not scen_dir.exists():
                        continue
                    files.extend(list(scen_dir.glob("hedge_summary_*__*.parquet")))

            # de-duplicate
            files = sorted(set(files))

            dfs: List[pd.DataFrame] = []
            for p in sorted(files):
                try:
                    df = pd.read_parquet(p)
                    dfs.append(df)
                except Exception as e:
                    print(f"[WARN] cannot read {p}: {e}")

            if dfs:
                big = pd.concat(dfs, ignore_index=True)

                # Strictly keep only rows that belong to the current target bucket.
                # This prevents cross-exposure contamination when fallback paths such as
                # <out_dir>/<scenario_stem>/ are shared across exposures.
                if "exposure_id" in big.columns:
                    big = big[big["exposure_id"].astype(str) == str(exposure)]
                if "scenario_kind" in big.columns:
                    big = big[big["scenario_kind"].astype(str) == str(kind)]

                if len(big) > 0 and "scenario_id" in big.columns:
                    big = big.drop_duplicates(
                        subset=["scenario_id", "strategy", "dynamic", "mode_roll"],
                        keep="last",
                    )

                if len(big) == 0:
                    big = _empty_summary_df()
            else:
                big = _empty_summary_df()

            outp = merged_root / exposure / kind / f"hedge_summary_{exposure}_{kind}_{tag_prefix}.parquet"
            outp.parent.mkdir(parents=True, exist_ok=True)
            big.to_parquet(outp, index=False)
            print(f"[MERGE] {exposure}/{kind}: files={len(files)} rows={len(big)} -> {outp}")


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="python")
    ap.add_argument("--sim", default="hedge_simulator.py")
    ap.add_argument("--master", default="MasterData.parquet")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--price_engine", default="MasterData_price_engine.parquet")

    ap.add_argument("--out_dir", default="results/BATCH_ALL")
    ap.add_argument("--logs_dir", default="results/BATCH_ALL/_orchestrator_logs")
    ap.add_argument("--tag", default="overnight_all_orchestrated")

    ap.add_argument("--procs", type=int, default=10, help="Outer parallelism: how many simulator processes in parallel")
    ap.add_argument("--jobs", type=int, default=12, help="Inner parallelism: passed to simulator --jobs")

    ap.add_argument("--resume", action="store_true", help="Skip runs that already have matching output files")
    ap.add_argument("--resume_any_tag", action="store_true", help="Skip if output exists regardless of tag (best-effort)")

    ap.add_argument("--include_company", action="store_true", help="Include company scenarios")

    ap.add_argument("--smoke", type=int, default=0, help="If >0, run only first N rows from each scenario file")
    ap.add_argument("--chunk", type=int, default=0, help="If >0, split scenario parquet into chunks of this size")

    ap.add_argument("--tmp_scen_dir", default="results/_tmp_scenarios", help="Temp scenario parquet location")

    ap.add_argument("--merge_only", action="store_true", help="Only merge existing outputs")
    ap.add_argument("--merged_out", default="results/FINAL_MERGED")

    # CLI filters for focused tests
    ap.add_argument("--only_exposure", choices=["WTI_SPOT", "BRENT_SPOT", "OPEC_BASKET"])
    ap.add_argument("--only_kind", choices=["baseline", "oracle_all", "oracle_universe", "company"])
    ap.add_argument("--only_strategy", choices=["nohedge", "naive", "ols_static", "ols_roll", "ccc_garch", "dcc_garch"])
    ap.add_argument("--only_window", type=int)
    ap.add_argument("--only_mode", choices=["static", "dynamic"])
    ap.add_argument("--only_roll", choices=["roll", "noroll"])

    ap.add_argument("--skip_validate", action="store_true", help="Skip running validate_pipeline.py before batch")
    ap.add_argument("--no_dashboard", action="store_true", help="Disable live dashboard printing")
    ap.add_argument("--dashboard_interval", type=float, default=0.75, help="Dashboard refresh interval in seconds")

    args = ap.parse_args()

    if args.merge_only:
        merge_outputs(
            out_dir=args.out_dir,
            tag_prefix=args.tag,
            out_merged_dir=args.merged_out,
            include_company=bool(args.include_company),
        )
        return 0

    # Fail-fast scenario validation (high value, low cost)
    scenarios_root = "scenarios"
    if os.path.isdir(scenarios_root):
        if getattr(args, "skip_validate", False) is False:
            run_validate_pipeline(py=args.py, scenarios_root=scenarios_root, exposure="ALL")
    else:
        print(f"[WARN] scenarios_root not found: {scenarios_root} (skipping validation)")

    # Build base grid (full specs)
    base_runs = build_grid(include_company=bool(args.include_company))

    # Expand to scenario variants: smoke/chunk -> rewrite spec.scenarios to temp files
    tmp_root = Path(args.tmp_scen_dir)
    expanded: List[RunSpec] = []

    # group by (exposure, kind) so we create variant scenario files once
    by_block: Dict[Tuple[str, str], List[RunSpec]] = {}
    for r in base_runs:
        by_block.setdefault((r.exposure, r.scenario_kind), []).append(r)

    for (exposure, kind), group in by_block.items():
        chunk_total, variants = make_variant_scenario_file(
            base_path=SCENARIOS[exposure][kind],
            tmp_root=tmp_root,
            exposure=exposure,
            kind=kind,
            smoke_n=(args.smoke if args.smoke > 0 else None),
            chunk_size=(args.chunk if args.chunk > 0 else None),
        )
        for chunk_id, chunk_path in variants:
            for r in group:
                expanded.append(RunSpec(
                    exposure=r.exposure,
                    scenario_kind=r.scenario_kind,
                    scenario_path=r.scenario_path,
                    strategy=r.strategy,
                    window=r.window,
                    mode=r.mode,
                    roll=r.roll,
                    chunk_id=chunk_id,
                    chunk_path=chunk_path,
                    chunk_total=chunk_total,
                ))

    # Optional filters for focused tests
    if getattr(args, "only_exposure", None):
        expanded = [r for r in expanded if r.exposure == args.only_exposure]
    if getattr(args, "only_kind", None):
        expanded = [r for r in expanded if r.scenario_kind == args.only_kind]
    if getattr(args, "only_strategy", None):
        expanded = [r for r in expanded if r.strategy == args.only_strategy]
    if getattr(args, "only_window", None) is not None:
        expanded = [r for r in expanded if r.window == args.only_window]
    if getattr(args, "only_mode", None):
        expanded = [r for r in expanded if r.mode == args.only_mode]
    if getattr(args, "only_roll", None):
        expanded = [r for r in expanded if r.roll == args.only_roll]

    # Drop exact duplicates (can arise from overlapping phase definitions)
    before = len(expanded)
    expanded = dedupe_runs(expanded)
    after = len(expanded)
    if after != before:
        print(f"[INFO] dedupe: {before} -> {after} runs")

    if not expanded:
        print("[ERROR] After applying filters, no runs remain.")
        return 2

    asyncio.run(run_batch(
        expanded,
        py=args.py,
        sim=args.sim,
        master=args.master,
        config=args.config,
        price_engine=args.price_engine,
        out_dir=args.out_dir,
        tag=args.tag,
        logs_dir=args.logs_dir,
        procs_outer=max(1, int(args.procs)),
        jobs_inner=max(1, int(args.jobs)),
        resume=bool(args.resume),
        resume_any_tag=bool(args.resume_any_tag),
        no_dashboard=bool(args.no_dashboard),
        dashboard_interval=float(args.dashboard_interval),
    ))

    # After run, merge
    merge_outputs(
        out_dir=args.out_dir,
        tag_prefix=args.tag,
        out_merged_dir=args.merged_out,
        include_company=bool(args.include_company),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())