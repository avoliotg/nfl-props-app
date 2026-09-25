"""rules_calibrate.py - measure the frozen constants rules.py refuses without.

Reports only. Writes nothing. Prints a block to paste.

WHY THIS EXISTS

    The receiving rule's trigger is an air-yards shock z-score standardized
    over the FULL 2023-2026 sample. rules.py cannot recompute that from a
    single week's props, because a quiet week with compressed air-yards
    variation would fire far more often than a volatile one and `z >= 1.0`
    would stop meaning what air_yards_retest.py measured.

    So the standardization has to be frozen. This script measures it once,
    from the same population and the same windows the validation used, and
    prints the CALIBRATION block for rules.py.

    rules.py refuses to evaluate the receiving rule until that block is
    pasted, rather than standardizing on whatever is to hand. Same principle
    as split_qb_rushing's min_match_rate defaulting to None: a threshold
    nobody has measured is not a threshold.

THREE CONSTANTS

    shock_mean, shock_sd   from (recent-3 mean minus base-8 mean) of
                           receiving_air_yards, over the population the rule
                           is served on. These convert the raw shock into the
                           z that was validated.

    dev_veto_at            the projection-minus-line value above which the
                           model vetoes. conjunction_followup.py found that
                           inside the rule cell, under-side ROI by dev
                           quintile runs +0.100, +0.079, +0.127, +0.086 and
                           then -0.117 in the top quintile. So the veto is
                           the boundary of that top quintile, measured on the
                           same rows.

A GATE, BECAUSE THIS SCRIPT DEFINES A LIVE RULE

    Section 1 reproduces air_yards_retest.py's lift at z >= 1.0 (+0.086 for
    receiving) using the constants it just measured. If the reproduction
    fails, the constants describe a different rule than the one that earned
    the interval, and the script aborts rather than printing a block that
    looks authoritative.

Run from the repo root:
    python rules_calibrate.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import eval_harness as eh
import input_weights as iw
import rules

MARKET = "receiving"
INPUT = "receiving_air_yards"
PUB_LIFT_Z1 = 0.0864          # air_yards_retest.py, receiving, z >= 1.0
LIFT_TOL = 0.02


def _rule(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 88)
    print("RULES CALIBRATION: the frozen constants rules.py needs")
    print(f"  seasons {seasons}   book {args.book}   market {MARKET}")
    print("=" * 88)

    # Reuse input_weights' loader so the population, the windows and the
    # standardization are identical to the validation rather than similar.
    p, _ = iw.build(types.SimpleNamespace(
        cache=args.cache, refresh=args.refresh, book=args.book), seasons)
    rc, bc = f"{INPUT}__recent", f"{INPUT}__base"
    need = [rc, bc, "line", "actual", "projection", "roi_under"]
    miss = [c for c in need if c not in p.columns]
    if miss:
        print(f"missing columns {miss}")
        sys.exit(1)

    g = p[(p["market"] == MARKET)].dropna(subset=need).copy()
    if len(g) < 2000:
        print(f"only {len(g)} usable {MARKET} rows; refusing to calibrate.")
        sys.exit(1)
    g["shock_raw"] = g[rc] - g[bc]
    g["dev"] = g["projection"] - g["line"]

    _rule("SECTION 1: THE CONSTANTS")
    shock_mean = float(g["shock_raw"].mean())
    shock_sd = float(g["shock_raw"].std(ddof=0))
    g["z"] = (g["shock_raw"] - shock_mean) / shock_sd
    cell = g[g["z"] >= rules.AIR_YARDS_Z_TRIGGER]
    if len(cell) < 300:
        print(f"rule cell is only {len(cell)} rows; refusing.")
        sys.exit(1)
    # the veto is the boundary of the TOP dev quintile inside the cell
    dev_veto_at = float(cell["dev"].quantile(0.80))

    print(f"  rows used                 {len(g):,}")
    print(f"  shock_mean                {shock_mean:+.6f}")
    print(f"  shock_sd                  {shock_sd:.6f}")
    print(f"  rule cell at z >= {rules.AIR_YARDS_Z_TRIGGER}     "
          f"{len(cell):,} rows "
          f"({len(cell) / 3.25:.0f}/season)")
    print(f"  dev_veto_at (80th pct)    {dev_veto_at:+.6f}")

    _rule("SECTION 2: GATE, reproduce the validated lift")
    base = float(g["roi_under"].mean())
    lift_all = float(cell["roi_under"].mean()) - base
    keep = cell[cell["dev"] <= dev_veto_at]
    veto = cell[cell["dev"] > dev_veto_at]
    lift_keep = float(keep["roi_under"].mean()) - base
    print(f"  {MARKET} baseline under ROI        {base:+.4f}")
    print(f"  cell, no veto      {len(cell):>6,} rows  lift {lift_all:+.4f}"
          f"   published {PUB_LIFT_Z1:+.4f}")
    print(f"  cell, veto applied {len(keep):>6,} rows  lift {lift_keep:+.4f}")
    print(f"  vetoed             {len(veto):>6,} rows  ROI  "
          f"{float(veto['roi_under'].mean()):+.4f}   "
          f"(should be clearly worse)")
    if abs(lift_all - PUB_LIFT_Z1) > LIFT_TOL:
        print()
        print("  ABORT: the lift does not reproduce air_yards_retest.py's")
        print(f"  {PUB_LIFT_Z1:+.4f}. These constants therefore describe a")
        print("  different rule than the one that earned the interval, and")
        print("  pasting them would wire up something unvalidated.")
        sys.exit(1)
    if lift_keep < lift_all:
        print()
        print("  WARNING: the veto makes the rule WORSE on this population.")
        print("  conjunction_followup.py found the top dev quintile at -0.117")
        print("  against roughly +0.10 elsewhere, so a reversal here means the")
        print("  quintile boundary is being computed on different rows.")
        print("  Investigate before pasting.")
    print("\n  GATE PASSED.")

    _rule("SECTION 3: PASTE THIS INTO rules.py")
    print("Replace the CALIBRATION dict with:")
    print()
    print("CALIBRATION = {")
    print('    "receiving": {')
    print(f'        "shock_mean": {shock_mean:.6f},')
    print(f'        "shock_sd": {shock_sd:.6f},')
    print(f'        "dev_veto_at": {dev_veto_at:.6f},')
    print("    },")
    print("}")
    print()
    print(f"# measured {len(g):,} receiving rows, seasons {seasons}, "
          f"book {args.book}")
    print(f"# cell at z >= {rules.AIR_YARDS_Z_TRIGGER}: {len(cell):,} rows, "
          f"lift {lift_all:+.4f} (published {PUB_LIFT_Z1:+.4f})")
    print(f"# with veto: {len(keep):,} rows, lift {lift_keep:+.4f}")
    print()
    print("  These are FROZEN. Recomputing them on a later sample changes")
    print("  what z >= 1.0 means and silently redefines the rule. If you ever")
    print("  do recompute, treat the result as a NEW rule needing its own")
    print("  out-of-sample test, not as a refresh of this one.")

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
