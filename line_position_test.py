"""line_position_test.py - is a receptions line misplaced relative to mass?

Implements PREREG_line_position.md, written 2026-10-01 before any query
joining a receptions line to an outcome by line value. READ ONLY: fits no
model, changes no constant, writes nothing.

    python line_position_test.py --cache lines_cache.parquet

THE TEST OF RECORD IS THE RANK CORRELATION, NOT THE BEST CELL. A single
profitable line among nine proves nothing. Nine lines ordered by the
quantity the mechanism predicts is a different claim. Per-cell figures are
printed afterwards and are DESCRIPTIVE ONLY.

PRE-REGISTERED, and reproduced here so the output carries its own
predictions rather than requiring the reader to find the document:

    P1  Spearman rank correlation between line and under ROI is NEGATIVE
    P2  under ROI positive at lines 0.5-2.5, negative at 5.5+
    P3  pooled under ROI stays near the already-measured -0.0225
    P4  it will NOT clear after correction; the author expects failure

ALIVE ONLY IF ALL OF: permutation p on the full-sample correlation below
0.05; holdout correlation same sign; holdout low-line cells positive at the
ACTUAL price. Anything else is a null and the route closes.

WHY THE ACTUAL PRICE. receptions is asymmetric on 94.7 percent of rows with
an average absolute gap of 204.5 points. Assuming -110 would be badly wrong
here, which is why ROI is computed per row from the real under odds.

WHAT THIS MUST RECONCILE WITH. banded_under_test.py already measured
receptions unders at ROI -0.0225, permutation p 0.9502, over 7,500 bets,
MDE 0.0306. If the pooled figure here is materially different, the
population differs and that must be explained before anything else is read.
"""
import argparse
import sys

import numpy as np
import pandas as pd

MARKET = "receptions"
BOOK = "fanduel"
LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
LOW_LINES = [0.5, 1.5, 2.5]
HIGH_LINES = [5.5, 6.5, 7.5, 8.5]

# Fixed in the pre-registration: a devigged probability outside this band is
# treated as an alternate line rather than a main one.
DEVIG_LO, DEVIG_HI = 0.15, 0.85

DEV_SEASONS = [2023, 2024]
HOLDOUT_SEASONS = [2025, 2026]
N_PERM = 2000

# From banded_under_test.py, so the reconciliation is explicit.
PUBLISHED_POOLED_ROI = -0.0225
PUBLISHED_TOL = 0.010


def hr(t):
    print()
    print("=" * 76)
    print(t)
    print("=" * 76)


def american_to_dec_profit(odds):
    """Profit per 1 unit staked if the bet wins."""
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, o / 100.0, 100.0 / np.abs(o))


def devig_additive(over_odds, under_odds):
    """Two-sided additive devig -> P(over). None-safe, returns NaN."""
    def imp(o):
        o = np.asarray(o, dtype=float)
        return np.where(o > 0, 100.0 / (o + 100.0), np.abs(o) / (np.abs(o) + 100.0))
    po, pu = imp(over_odds), imp(under_odds)
    tot = po + pu
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, po / tot, np.nan)


def under_roi(df, win=None):
    """ROI per unit staked on the UNDER, at the actual under price.

    A push is impossible on a .5 line, so every row settles.

    `win` may be supplied to score a PERMUTED outcome against the real
    prices, which is what the null below needs. Taking it as an argument
    rather than recomputing from `actual` keeps the permutation from having
    to fabricate plausible stat lines.

    Returns (roi, n, win_rate, avg_odds).
    """
    if len(df) == 0:
        return np.nan, 0, np.nan, np.nan
    if win is None:
        win = (df["actual"] < df["line"]).to_numpy()
    win = np.asarray(win, dtype=bool)
    profit = american_to_dec_profit(df["under_odds"].to_numpy())
    units = np.where(win, profit, -1.0)
    return (float(units.mean()), int(len(df)), float(win.mean()),
            float(df["under_odds"].mean()))


def spearman(x, y):
    """Rank correlation, no scipy dependency."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def roi_by_line(df, win=None):
    """Under ROI per line, in LINES order. NaN where a line has no rows."""
    if win is None:
        win = (df["actual"] < df["line"]).to_numpy()
    win = np.asarray(win, dtype=bool)
    ln_arr = df["line"].to_numpy()
    out = []
    for ln in LINES:
        m = ln_arr == ln
        roi, n, wr, avg = under_roi(df[m], win=win[m])
        out.append((ln, roi, n, wr, avg))
    return out


def correlation_stat(df, win=None):
    rows = roi_by_line(df, win=win)
    lines = [r[0] for r in rows if np.isfinite(r[1])]
    rois = [r[1] for r in rows if np.isfinite(r[1])]
    return spearman(lines, rois), rows


def permute_p(df, observed, n_perm=N_PERM, seed=0):
    """Shuffle the win/loss outcome GLOBALLY, keeping each row's line+price.

    CORRECTED 2026-10-01, and the correction matters. The
    pre-registration specified shuffling the outcome WITHIN each line cell.
    That null is DEGENERATE: a within-cell shuffle leaves the cell's win
    rate unchanged, so every cell's ROI is identical to the observed one and
    the correlation statistic never moves. Measured on planted data the
    null had sd 0.0000 and returned p 1.0000 against a true correlation of
    -0.88.

    The association being tested is BETWEEN cells: does a line's VALUE
    predict its ROI. So the null has to break the link between a row's line
    and its outcome, which means permuting across the whole sample while
    each row keeps its own line and its own price.

    What this preserves: the number of rows per line, every row's actual
    under price, and the overall win rate. What it destroys: any
    relationship between line value and winning. That is H0 stated
    properly.
    """
    rng = np.random.default_rng(seed)
    win = (df["actual"] < df["line"]).to_numpy()
    stats = np.empty(n_perm)
    for b in range(n_perm):
        s, _ = correlation_stat(df, win=rng.permutation(win))
        stats[b] = s
    # one-sided: the prediction is a NEGATIVE correlation
    p = float(np.mean(stats <= observed))
    return p, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--perm", type=int, default=N_PERM)
    args = ap.parse_args()

    hr("LOADING")
    try:
        C = pd.read_parquet(args.cache)
    except Exception as e:
        print("  could not read %s: %s" % (args.cache, e))
        sys.exit(1)
    print("  %d rows in %s" % (len(C), args.cache))

    C = C[(C["market"] == MARKET) & (C["book"] == BOOK)].copy()
    print("  %d %s %s rows" % (len(C), BOOK, MARKET))
    if "snapshot_label" in C.columns:
        C = C[C["snapshot_label"] == "closing"]
        print("  %d closing" % len(C))

    # outcomes
    from models import data_utils, receptions as rcp
    seasons = sorted(C["season"].dropna().unique().astype(int).tolist())
    stats = data_utils.load_player_stats(seasons)
    stats = stats.to_pandas() if hasattr(stats, "to_pandas") else stats
    key = "receptions"
    s = stats[["player_display_name", "season", "week", key]].copy()
    s["_k"] = data_utils.norm_join_name(s["player_display_name"])
    C["_k"] = data_utils.norm_join_name(C["player"])
    before = len(C)
    C = C.merge(s[["_k", "season", "week", key]],
                on=["_k", "season", "week"], how="inner")
    C = C.rename(columns={key: "actual"})
    print("  joined %d of %d (%.1f%%)"
          % (len(C), before, 100.0 * len(C) / max(before, 1)))

    C = C[C["actual"].notna() & C["line"].notna()
          & C["under_odds"].notna() & C["over_odds"].notna()]
    print("  %d with a line, both prices and an outcome" % len(C))

    # alt-line exclusion, threshold fixed in the pre-registration
    C["book_p"] = devig_additive(C["over_odds"], C["under_odds"])
    n_before = len(C)
    C = C[(C["book_p"] >= DEVIG_LO) & (C["book_p"] <= DEVIG_HI)]
    print("  %d after excluding devigged p outside [%.2f, %.2f] (%d dropped)"
          % (len(C), DEVIG_LO, DEVIG_HI, n_before - len(C)))

    C = C[C["line"].isin(LINES)]
    print("  %d on the nine enumerated lines" % len(C))
    if len(C) == 0:
        print("  nothing to test")
        sys.exit(1)
    C = C.reset_index(drop=True)

    hr("P3 FIRST: DOES THE POOLED FIGURE RECONCILE")
    roi, n, wr, avg = under_roi(C)
    print("  pooled under ROI   %+.4f on %d bets, win rate %.4f, avg odds %.1f"
          % (roi, n, wr, avg))
    print("  banded_under_test  %+.4f (p 0.9502, 7,500+ bets, MDE 0.0306)"
          % PUBLISHED_POOLED_ROI)
    gap = roi - PUBLISHED_POOLED_ROI
    print("  difference         %+.4f  %s" % (gap,
          "consistent" if abs(gap) <= PUBLISHED_TOL else "DIVERGENT"))
    if abs(gap) > PUBLISHED_TOL:
        print()
        print("  THE POPULATIONS DIFFER. Read nothing below until this is")
        print("  explained: that test used its own loader and filters, and a")
        print("  pooled ROI that disagrees means these are not the same")
        print("  rows. P3 predicted agreement.")

    hr("DESCRIPTIVE: UNDER ROI BY LINE (not the test of record)")
    print("  %6s %10s %8s %10s %10s" % ("line", "ROI", "bets", "win rate",
                                        "avg odds"))
    print("  " + "-" * 50)
    for ln, r, nn, w, a in roi_by_line(C):
        if nn == 0:
            print("  %6.1f %10s %8d" % (ln, "no rows", 0))
            continue
        print("  %6.1f %+10.4f %8d %10.4f %10.1f" % (ln, r, nn, w, a))

    hr("P1, THE TEST OF RECORD: RANK CORRELATION")
    obs, rows = correlation_stat(C)
    print("  Spearman(line, under ROI) = %+.4f" % obs)
    print("  P1 predicted NEGATIVE: %s"
          % ("as predicted" if obs < 0 else "WRONG SIGN"))
    print()
    print("  running %d permutations, shuffling win/loss GLOBALLY while"
          % args.perm)
    print("  each row keeps its own line and its own price. The")
    print("  pre-registration said WITHIN each line; that null is")
    print("  degenerate, since a within-cell shuffle cannot change a cell's")
    print("  own win rate. Corrected before the first real run.")
    p, dist = permute_p(C, obs, n_perm=args.perm)
    print("  permutation null: mean %+.4f, sd %.4f" % (dist.mean(), dist.std()))
    print("  one-sided p (P[stat <= observed]) = %.4f" % p)
    print("  threshold 0.05: %s" % ("CLEARS" if p < 0.05 else "does not clear"))
    print()
    print("  MDE: with 9 cells and this null's sd of %.4f, the smallest")
    print("  correlation detectable at p<0.05 one-sided is about %+.4f."
          % (dist.std(), np.quantile(dist, 0.05)))
    print("  A null with |observed| well inside that is UNTESTED rather")
    print("  than cleared.")

    hr("P2: SIGN PATTERN AT THE ENDS")
    lowr, _, _, _ = under_roi(C[C["line"].isin(LOW_LINES)])
    highr, _, _, _ = under_roi(C[C["line"].isin(HIGH_LINES)])
    print("  lines 0.5-2.5  ROI %+.4f  (P2 predicted positive)" % lowr)
    print("  lines 5.5+     ROI %+.4f  (P2 predicted negative)" % highr)
    p2 = (lowr > 0) and (highr < 0)
    print("  P2 %s" % ("holds" if p2 else "FAILS"))

    hr("HOLDOUT: FROZEN DIRECTION ON LATER SEASONS")
    dev = C[C["season"].isin(DEV_SEASONS)]
    hold = C[C["season"].isin(HOLDOUT_SEASONS)]
    print("  dev     %s: %d rows" % (DEV_SEASONS, len(dev)))
    print("  holdout %s: %d rows" % (HOLDOUT_SEASONS, len(hold)))
    d_obs, _ = correlation_stat(dev)
    h_obs, _ = correlation_stat(hold)
    print("  dev correlation      %+.4f" % d_obs)
    print("  holdout correlation  %+.4f" % h_obs)
    same = np.isfinite(d_obs) and np.isfinite(h_obs) and (d_obs * h_obs > 0)
    print("  same sign: %s" % ("yes" if same else "NO"))
    hl, hn, _, _ = under_roi(hold[hold["line"].isin(LOW_LINES)])
    print("  holdout low-line ROI %+.4f on %d bets" % (hl, hn))

    hr("DECISION, AGAINST THE PRE-REGISTERED RULE")
    c1 = p < 0.05
    c2 = same
    c3 = np.isfinite(hl) and hl > 0
    print("  permutation p < 0.05 ........ %s" % ("yes" if c1 else "no"))
    print("  holdout correlation same sign %s" % ("yes" if c2 else "no"))
    print("  holdout low lines positive .. %s" % ("yes" if c3 else "no"))
    print()
    if c1 and c2 and c3:
        print("  ALIVE. All three conditions hold. P4 predicted failure and")
        print("  was wrong, which is the stronger form of a positive result.")
        print("  NOT a bet yet: report the average price per cell before")
        print("  sizing, since a 2.38-point hurdle at -110 is much larger")
        print("  at -175.")
    else:
        print("  NULL. The route closes. P4 predicted this.")
        print("  Record the MDE above beside the result: a null without it")
        print("  is not a null.")
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
