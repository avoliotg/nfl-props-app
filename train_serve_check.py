"""
OpalScales - train/serve agreement check for the EWMA features.

WHY THIS EXISTS
    Rolling features are computed in TWO places:
      build_dataset()        -> shift(1).ewm(halflife=h) per player, for training
      build_upcoming_week()  -> the bridge, for pre-kickoff projections
    If they disagree, the model trains on one feature definition and predicts on
    another. That is train/serve skew: no error is raised, the numbers just
    quietly stop meaning the same thing. Week 1 2026 runs ENTIRELY through the
    bridge, so this is the path the live board uses.

HOW IT TESTS
    Pretend 2025 has not happened. Call build_upcoming_week(2025, 1), which
    bridges from 2022-2024, and compare its features against the REAL week 1
    2025 rows in build_dataset(), which were computed by the shifted EWMA. Those
    two should be identical to floating-point noise for every player present in
    both.

    A perfect match means the bridge is a faithful reimplementation. Systematic
    differences mean skew, and the magnitude tells you how much.

HOW TO RUN
    python train_serve_check.py
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

TEST_SEASON = 2025          # pretend this season has not been played
TOL = 1e-6

MARKETS = {
    "receiving": ["targets_roll", "target_share_roll", "ypt_roll", "snap_roll"],
    "receptions": ["targets_roll", "target_share_roll", "snap_roll"],
    "rushing": ["carries_roll"],
    "qb_passing": ["attempts_roll"],
    "qb_rushing": ["rush_yds_roll", "carries_roll"],
}


def check(name, feats):
    print("\n" + "=" * 74)
    print(f"MARKET: {name}")
    print("=" * 74)
    mod = __import__(f"models.{name}", fromlist=["*"])
    if not hasattr(mod, "build_upcoming_week"):
        print("  no build_upcoming_week, skipped")
        return None

    truth = mod.build_dataset()
    truth = truth[(truth["season"] == TEST_SEASON) & (truth["week"] == 1)]
    have = [f for f in feats if f in truth.columns]
    truth = truth[["player_id"] + have].dropna(subset=have)
    print(f"  real week 1 {TEST_SEASON} rows from build_dataset: {len(truth):,}")

    try:
        bridged = mod.build_upcoming_week(TEST_SEASON, 1)
    except Exception as exc:
        print(f"  build_upcoming_week FAILED: {type(exc).__name__}: {exc}")
        return None
    bridged = bridged[["player_id"] + have].dropna(subset=have)
    print(f"  bridged rows from build_upcoming_week:              {len(bridged):,}")

    m = truth.merge(bridged, on="player_id", suffixes=("_true", "_bridge"))
    print(f"  players present in BOTH: {len(m):,}")
    if len(m) == 0:
        print("  nothing to compare")
        return None

    print("\n  feature            max_abs_diff   mean_abs_diff   n_mismatch")
    worst = 0.0
    bad_any = 0
    for f in have:
        a = m[f"{f}_true"].to_numpy(float)
        b = m[f"{f}_bridge"].to_numpy(float)
        d = np.abs(a - b)
        nbad = int((d > TOL).sum())
        bad_any += nbad
        worst = max(worst, float(np.nanmax(d)))
        print(f"  {f:18s} {np.nanmax(d):12.8f} {np.nanmean(d):15.8f} "
              f"{nbad:12d}")

    if worst <= TOL:
        print("\n  => EXACT MATCH. The bridge faithfully reproduces the training")
        print("     feature definition. No train/serve skew.")
    else:
        print(f"\n  => MISMATCH. Worst absolute difference {worst:.6f}.")
        print("     The model trains on one definition and predicts on another.")
        # show the biggest offenders so the cause is findable
        f0 = have[0]
        m["_d"] = (m[f"{f0}_true"] - m[f"{f0}_bridge"]).abs()
        top = m.nlargest(8, "_d")[["player_id", f"{f0}_true", f"{f0}_bridge", "_d"]]
        print(f"\n     largest {f0} differences:")
        print("     " + top.round(6).to_string().replace("\n", "\n     "))
    return worst


def main():
    print("=" * 74)
    print("TRAIN/SERVE AGREEMENT CHECK")
    print(f"comparing build_upcoming_week({TEST_SEASON}, 1) against the real")
    print(f"week 1 {TEST_SEASON} rows produced by build_dataset()")
    print("=" * 74)
    results = {}
    for name, feats in MARKETS.items():
        try:
            results[name] = check(name, feats)
        except Exception as exc:
            import traceback
            print(f"\n{name}: FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for k, v in results.items():
        if v is None:
            print(f"  {k:12s} not checked")
        elif v <= TOL:
            print(f"  {k:12s} MATCH")
        else:
            print(f"  {k:12s} SKEW, worst diff {v:.6f}")
    print("\n  Markets still on rolling(6) will show a mismatch, because their")
    print("  bridge takes a plain 6-game mean of the prior season only while")
    print("  build_dataset's window can span the offseason. That pre-existed")
    print("  this change; it is worth knowing about but is not new.")


if __name__ == "__main__":
    main()
