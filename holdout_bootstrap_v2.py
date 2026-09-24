"""
holdout_bootstrap_v2.py
=======================

Cluster bootstrap of the edge_threshold_v5 holdout protocol.

WHY v2 EXISTS
-------------
v1 reimplemented the protocol from its description and got a DIFFERENT answer
on the same input file: 50/16/18 bets at thresholds 0.095/0.115/0.115 against
v5's 71/67/38 at 0.090/0.090/0.095. Three concrete mismatches, found only by
reading v5's source:

  1. v1 required 30 bets on the training seasons. v5 requires **100**. With a
     floor of 30 the argmax chases thin high-edge cells and lands on 0.115.
  2. v5 DROPS a held-out season entirely if it yields fewer than 20 bets, so
     2026 contributes nothing. v1 pooled whatever it found.
  3. v1's grid stopped at 0.20. v5's GRID runs to 0.25.

The v1 output was shown with numbers attached before this was caught. The
smoke test only verified that the bootstrap detects a planted edge and rejects
a null, which it did. It never checked that the protocol reproduces v5 on the
same input, and that check costs one line.

SO THIS SCRIPT REFUSES TO RUN WITHOUT THAT CHECK
------------------------------------------------
protocol() is transcribed from edge_threshold_v5.holdout, and GRID plus the
clustered mean are IMPORTED from v5 rather than restated. Before any
resampling, the script runs the protocol on the unresampled data and compares
against --expect. A mismatch aborts.

WHAT IT MEASURES
----------------
v5's reported SE is computed on the bets the protocol SELECTED and carries no
variance from the threshold-selection step itself. This resamples games within
season and re-runs the ENTIRE protocol each time, so selection variance is
included. The resulting interval is wider, and it is the honest one.

Plus a season jackknife, and a placebo that shuffles the outcome within line
value. Note the protocol has a small POSITIVE bias under the placebo, because
it takes the best of 50 cutoffs on the training seasons, so expect slightly
above zero rather than exactly zero.

USAGE
-----
  python edge_threshold_v5.py --all-seasons --cache lines_cache.parquet \
      --markets receptions --line-value 2.5,3.5 --sigma-sqrt \
      --save-rows priced_2p5_3p5.csv

  python holdout_bootstrap_v2.py priced_2p5_3p5.csv \
      --expect 2023:71,2024:67,2025:38 --expect-roi 0.1475

  python holdout_bootstrap_v2.py priced_2p5_3p5.csv --no-check --boot 1000
"""

import argparse
import sys

import numpy as np
import pandas as pd

try:
    import edge_threshold_v5 as ET
except ImportError as e:
    sys.exit(f"could not import edge_threshold_v5: {e}\n"
             "run this from the repo root")

try:
    import eval_harness as eh
except ImportError as e:
    sys.exit(f"could not import eval_harness: {e}")

GRID = ET.GRID                      # imported, not restated
TRAIN_MIN = 100                     # edge_threshold_v5.holdout
TEST_MIN = 20                       # edge_threshold_v5.holdout


def protocol(s):
    """
    Transcribed from edge_threshold_v5.holdout. Returns (pooled_roi,
    total_bets, detail) where detail is a list of (season, chosen_t, n_bets,
    roi). A held-out season with fewer than TEST_MIN bets is DROPPED, matching
    v5, which is why 2026 does not appear.
    """
    seasons = sorted(s["season"].unique())
    if len(seasons) < 2:
        return np.nan, 0, []
    rows, detail = [], []
    for s_out in seasons:
        tr = s[s["season"] != s_out]
        te = s[s["season"] == s_out]
        bt, bv = None, -np.inf
        for t in GRID:
            g = tr[(tr["edge"] >= t) & (~tr["push"])]
            if len(g) < TRAIN_MIN:
                continue
            v = float(g["profit"].mean())
            if v > bv:
                bt, bv = t, v
        if bt is None:
            continue
        g = te[(te["edge"] >= bt) & (~te["push"])]
        if len(g) < TEST_MIN:
            detail.append((s_out, bt, len(g), np.nan))
            continue
        detail.append((s_out, bt, len(g), float(g["profit"].mean())))
        rows.append(g)
    if not rows:
        return np.nan, 0, detail
    allg = pd.concat(rows, ignore_index=True)
    return float(allg["profit"].mean()), len(allg), detail


def load(path):
    df = pd.read_csv(path)
    need = ["season", "edge", "profit", "event_id"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"missing columns {miss}. Present: {sorted(df.columns)}")
    d = df.copy()
    d["season"] = d["season"].astype(int)
    for c in ("edge", "profit"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "push" not in d.columns:
        print("  no `push` column, assuming none")
        d["push"] = False
    d["push"] = d["push"].astype(bool)
    if "won" not in d.columns:
        d["won"] = (d["profit"] > 0).astype(float)
    if "line_bet" in d.columns:
        d["line"] = d["line_bet"]
    elif "line_fanduel" in d.columns:
        d["line"] = d["line_fanduel"]
    d = d.dropna(subset=["edge", "profit"]).reset_index(drop=True)
    print("=" * 92)
    print(f"LOADED {path}")
    print("=" * 92)
    print(f"  {len(d)} rows, {d['event_id'].nunique()} game clusters, "
          f"seasons {sorted(d['season'].unique())}")
    print(f"  GRID imported from v5: {GRID[0]} to {GRID[-1]} "
          f"step {GRID[1] - GRID[0]:.3f}, {len(GRID)} cutoffs")
    print(f"  TRAIN_MIN {TRAIN_MIN}, TEST_MIN {TEST_MIN}\n")
    return d


def self_check(d, expect, expect_roi, tol=0.002):
    """Abort unless the protocol reproduces v5's printed holdout."""
    print("=" * 92)
    print("SELF-CHECK: does protocol() reproduce edge_threshold_v5?")
    print("=" * 92)
    roi, n, detail = protocol(d)
    print(f"  {'season':>8}{'chosen t':>10}{'bets':>7}{'ROI':>10}"
          f"{'expected bets':>15}{'':>4}")
    ok = True
    for s_out, bt, nb, r in detail:
        exp = expect.get(int(s_out))
        mark = ""
        if exp is not None:
            mark = "ok" if nb == exp else "MISMATCH"
            if nb != exp:
                ok = False
        rs = f"{r:+.4f}" if np.isfinite(r) else "dropped"
        print(f"  {int(s_out):>8}{bt:>10.3f}{nb:>7}{rs:>10}"
              f"{(exp if exp is not None else '-'):>15}   {mark}")
    print(f"  {'POOLED':>8}{'':>10}{n:>7}{roi:>+10.4f}", end="")
    if expect_roi is not None:
        d_roi = abs(roi - expect_roi)
        print(f"{expect_roi:>15.4f}   "
              f"{'ok' if d_roi < tol else 'MISMATCH'}")
        if d_roi >= tol:
            ok = False
    else:
        print()
    print()
    if not ok:
        sys.exit("SELF-CHECK FAILED. protocol() does not reproduce v5 on this\n"
                 "input, so any bootstrap of it would be a bootstrap of\n"
                 "something else. This is exactly how v1 went wrong. Fix the\n"
                 "protocol before proceeding, or pass --no-check if you\n"
                 "understand why the numbers differ.")
    print("  PASS. protocol() reproduces v5 exactly. Bootstrap is meaningful.\n")
    return roi


def boot(d, n_boot, seed=0, quiet=False):
    rng = np.random.default_rng(seed)
    by_season = {}
    for s, g in d.groupby("season"):
        pos = {}
        for gm, gg in g.groupby("event_id"):
            pos[gm] = gg.index.to_numpy()
        by_season[s] = (np.array(list(pos.keys()), dtype=object), pos)
    out = []
    for i in range(n_boot):
        rows = []
        for s, (games, pos) in by_season.items():
            pick = rng.choice(len(games), size=len(games), replace=True)
            rows.append(np.concatenate([pos[games[k]] for k in pick]))
        rep = d.loc[np.concatenate(rows)].reset_index(drop=True)
        roi, n, _ = protocol(rep)
        out.append(roi if n > 0 else np.nan)
        if not quiet and (i + 1) % 100 == 0:
            print(f"\r    {i + 1}/{n_boot}", end="", flush=True)
    if not quiet:
        print("\r" + " " * 30 + "\r", end="")
    a = np.array(out, float)
    return a[np.isfinite(a)]


def report_boot(d, n_boot, label, seed=0):
    roi, n, _ = protocol(d)
    a = boot(d, n_boot, seed)
    if len(a) < 20:
        print(f"  {label}: too few valid replicates\n")
        return
    lo, hi = np.percentile(a, [2.5, 97.5])
    print(f"  {label}")
    print(f"    point        {roi:+.4f}  on {n} bets")
    print(f"    boot median  {np.median(a):+.4f}")
    print(f"    95% CI       [{lo:+.4f}, {hi:+.4f}]")
    print(f"    share > 0    {(a > 0).mean() * 100:.1f}%")
    print(f"    valid reps   {len(a)} of {n_boot}")
    print()


def jackknife(d):
    print("=" * 92)
    print("SEASON JACKKNIFE. Drop one season entirely and recompute.")
    print("=" * 92)
    base, nb, _ = protocol(d)
    print(f"  {'dropped':>10}{'bets':>7}{'pooled ROI':>13}{'change':>10}")
    print(f"  {'none':>10}{nb:>7}{base:>+13.4f}{'':>10}")
    for s in sorted(d["season"].unique()):
        sub = d[d["season"] != s]
        if sub["season"].nunique() < 2:
            continue
        roi, n, _ = protocol(sub)
        ch = roi - base if np.isfinite(roi) else np.nan
        print(f"  {int(s):>10}{n:>7}{roi:>+13.4f}{ch:>+10.4f}")
    print("\n  Note this is NOT a clean influence measure: dropping a season")
    print("  also removes it from the TRAINING set for the other seasons, so")
    print("  the threshold moves too. Read it as sensitivity, not as one")
    print("  season's contribution.\n")


def placebo(d, n_boot, seed=1):
    print("=" * 92)
    print("PLACEBO. Outcome shuffled within line value.")
    print("=" * 92)
    rng = np.random.default_rng(seed)
    p = d.copy()
    key = "line" if "line" in p.columns else "season"
    for _, g in p.groupby(key):
        i = g.index.to_numpy()
        pr = p.loc[i, "profit"].to_numpy()
        perm = rng.permutation(len(i))
        p.loc[i, "profit"] = pr[perm]
        p.loc[i, "won"] = p.loc[i, "profit"].to_numpy() > 0
    print(f"  shuffled within `{key}`")
    report_boot(p, max(n_boot // 2, 200), "shuffled outcomes", seed)
    print("  The protocol takes the best of 50 cutoffs on the training")
    print("  seasons, so a SMALL positive placebo is expected. A large one")
    print("  means the protocol manufactures edge and nothing from it counts.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--expect", default=None,
                    help="season:bets pairs from v5's output, "
                         "e.g. 2023:71,2024:67,2025:38")
    ap.add_argument("--expect-roi", type=float, default=None)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--placebo", action="store_true")
    args = ap.parse_args()

    d = load(args.rows)

    expect = {}
    if args.expect:
        for part in args.expect.split(","):
            k, v = part.split(":")
            expect[int(k)] = int(v)

    if args.no_check:
        print("  --no-check passed, SKIPPING the reproduction check.\n")
        protocol(d)
    else:
        if not expect and args.expect_roi is None:
            sys.exit("pass --expect and/or --expect-roi with the numbers "
                     "edge_threshold_v5 printed, or --no-check if you "
                     "accept an unverified protocol.")
        self_check(d, expect, args.expect_roi)

    print("=" * 92)
    print("CLUSTER BOOTSTRAP, games resampled within season, protocol")
    print("re-run each replicate so THRESHOLD SELECTION variance is included")
    print("=" * 92)
    report_boot(d, args.boot, "full sample")

    jackknife(d)
    if args.placebo:
        placebo(d, args.boot)

    print("=" * 92)
    print("HOW TO CALL IT")
    print("=" * 92)
    print("  v5's SE is conditional on the bets it selected. This interval")
    print("  carries the selection variance too, so it is wider on purpose.")
    print("  A comfortable straddle of zero means the rule is not")
    print("  established and the app stays silent, which is J1.")
    print()


if __name__ == "__main__":
    main()
