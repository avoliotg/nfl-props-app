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

BOOK = "fanduel"

# receptions is ENUMERABLE: nine distinct lines, so each is its own cell.
RECEPTIONS_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

# The yardage markets have 107 and 110 distinct lines, so binning is
# unavoidable. Ten-yard bands, FIXED IN THE PRE-REGISTRATION, chosen
# because they are round numbers. No edge sits near 46.5: aligning a bin to
# Rule 1's threshold would build the rule's own answer into the test.
YARDAGE_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 10000]

# A bin below this is reported but EXCLUDED from the correlation. In the
# receptions run a four-bet cell contributed +0.7758.
MIN_BETS = 100

STAT_COL = {"receptions": "receptions",
            "receiving": "receiving_yards",
            "rushing": "rushing_yards"}

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


def cells_for(market):
    """(label, midpoint, mask_fn) per cell. Fixed per market, never chosen.

    receptions enumerates its nine lines. The yardage markets use the
    ten-yard bands fixed in the pre-registration, and the MIDPOINT is what
    the rank correlation is computed against, so an open top bin gets a
    nominal midpoint rather than infinity.
    """
    if market == "receptions":
        return [("%.1f" % ln, ln, (lambda a, v=ln: a == v))
                for ln in RECEPTIONS_LINES]
    out = []
    for lo, hi in zip(YARDAGE_EDGES[:-1], YARDAGE_EDGES[1:]):
        if hi >= 10000:
            out.append(("70+", 75.0, (lambda a, l=lo: a >= l)))
        else:
            out.append(("%d-%d" % (lo, hi), (lo + hi) / 2.0,
                        (lambda a, l=lo, h=hi: (a >= l) & (a < h))))
    return out


def roi_by_line(df, win=None, market="receptions"):
    """Under ROI per cell. Returns (label, midpoint, roi, n, wr, avg)."""
    if win is None:
        win = (df["actual"] < df["line"]).to_numpy()
    win = np.asarray(win, dtype=bool)
    ln_arr = df["line"].to_numpy()
    out = []
    for label, mid, fn in cells_for(market):
        m = np.asarray(fn(ln_arr), dtype=bool)
        roi, n, wr, avg = under_roi(df[m], win=win[m])
        out.append((label, mid, roi, n, wr, avg))
    return out


def correlation_stat(df, win=None, market="receptions", min_bets=MIN_BETS):
    """Spearman(cell midpoint, cell under ROI) over cells clearing the floor.

    The bet floor is applied HERE so the same cells enter the observed
    statistic and every permuted one. Applying it only to the observed
    value would compare statistics computed over different cell sets.
    """
    rows = roi_by_line(df, win=win, market=market)
    mids = [r[1] for r in rows if np.isfinite(r[2]) and r[3] >= min_bets]
    rois = [r[2] for r in rows if np.isfinite(r[2]) and r[3] >= min_bets]
    return spearman(mids, rois), rows


def permute_p(df, observed, n_perm=N_PERM, seed=0,
              market="receptions"):
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
        s, _ = correlation_stat(df, win=rng.permutation(win),
                                market=market)
        stats[b] = s
    # one-sided: the prediction is a NEGATIVE correlation
    p = float(np.mean(stats <= observed))
    return p, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="rushing",
                    choices=["rushing", "receiving", "receptions"])
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--alpha", type=float, default=0.025,
                    help="PRE-REGISTERED at 0.025, not 0.05, because two "
                         "markets are tested as one family. Running two and "
                         "reporting whichever clears at 0.05 is the defect "
                         "this project keeps catching.")
    args = ap.parse_args()
    market = args.market

    hr("LINE POSITION TEST: %s" % market.upper())
    print("  pre-registered in PREREG_line_position_yardage.md")
    print("  statistic: Spearman(cell midpoint, cell under ROI)")
    print("  threshold: %.3f  (two-market family correction)" % args.alpha)
    print("  bet floor: %d per cell, below which a cell is reported but"
          % MIN_BETS)
    print("             excluded from the correlation")
    if market == "rushing":
        print()
        print("  R1 predicts a NEGATIVE correlation. This tests the MECHANISM")
        print("  behind Rule 1 (live, ROI +0.0589 at line <= 46.5), not a new")
        print("  rule. banded_under_test.py returned p 0.2587 for rushing on")
        print("  a best-of-grid test; a rank correlation is a narrower and")
        print("  more powerful question on the same rows.")
    elif market == "receiving":
        print()
        print("  V1 predicts NEGATIVE, V2 predicts it will NOT clear.")
        print("  banded_under_test.py closed receiving at -0.0350, p 0.9801,")
        print("  well powered, so a FLAT under bias is already dead.")

    hr("LOADING")
    try:
        C = pd.read_parquet(args.cache)
    except Exception as e:
        print("  could not read %s: %s" % (args.cache, e))
        sys.exit(1)
    print("  %d rows in %s" % (len(C), args.cache))
    C = C[(C["market"] == market) & (C["book"] == BOOK)].copy()
    print("  %d %s %s rows" % (len(C), BOOK, market))
    if "snapshot_label" in C.columns:
        C = C[C["snapshot_label"] == "closing"]
        print("  %d closing" % len(C))
    if len(C) == 0:
        print("  nothing to test")
        sys.exit(1)

    from models import data_utils
    seasons = sorted(C["season"].dropna().unique().astype(int).tolist())
    stats = data_utils.load_player_stats(seasons)
    stats = stats.to_pandas() if hasattr(stats, "to_pandas") else stats
    key = STAT_COL[market]
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
    C["book_p"] = devig_additive(C["over_odds"], C["under_odds"])
    n0 = len(C)
    C = C[(C["book_p"] >= DEVIG_LO) & (C["book_p"] <= DEVIG_HI)]
    print("  %d after the alt-line filter [%.2f, %.2f] (%d dropped)"
          % (len(C), DEVIG_LO, DEVIG_HI, n0 - len(C)))
    C = C.reset_index(drop=True)
    if len(C) == 0:
        print("  nothing left")
        sys.exit(1)

    hr("POOLED, FOR RECONCILIATION")
    roi, n, wr, avg = under_roi(C)
    print("  pooled under ROI %+.4f on %d bets, win rate %.4f, avg odds %.1f"
          % (roi, n, wr, avg))
    pub = {"rushing": None, "receiving": -0.0350, "receptions": -0.0225}
    if pub.get(market) is not None:
        print("  banded_under_test  %+.4f" % pub[market])
        print("  difference         %+.4f" % (roi - pub[market]))
    else:
        print("  banded_under_test reported rushing per BAND rather than")
        print("  pooled, so there is no single published figure to match.")

    hr("UNDER ROI BY CELL (descriptive)")
    print("  %-8s %10s %8s %10s %10s %s"
          % ("cell", "ROI", "bets", "win rate", "avg odds", ""))
    print("  " + "-" * 56)
    for label, mid, r, nn, w, a in roi_by_line(C, market=market):
        if nn == 0:
            print("  %-8s %10s %8d" % (label, "no rows", 0))
            continue
        flag = "" if nn >= MIN_BETS else "  <- below floor, excluded"
        print("  %-8s %+10.4f %8d %10.4f %10.1f%s"
              % (label, r, nn, w, a, flag))

    hr("TEST OF RECORD: RANK CORRELATION")
    obs, rows = correlation_stat(C, market=market)
    used = [r[0] for r in rows if np.isfinite(r[2]) and r[3] >= MIN_BETS]
    print("  cells used: %s" % used)
    print("  Spearman(midpoint, under ROI) = %+.4f" % obs)
    print("  predicted NEGATIVE: %s"
          % ("as predicted" if obs < 0 else "WRONG SIGN"))
    print()
    print("  %d permutations, win/loss shuffled GLOBALLY with each row"
          % args.perm)
    print("  keeping its own line and price")
    p, dist = permute_p(C, obs, n_perm=args.perm, market=market)
    print("  null: mean %+.4f, sd %.4f" % (dist.mean(), dist.std()))
    print("  one-sided p = %.4f" % p)
    print("  vs threshold %.3f: %s"
          % (args.alpha, "CLEARS" if p < args.alpha else "does not clear"))
    print()
    print("  MDE: null sd %.4f over %d cells, so the smallest correlation\n"
          "  detectable at p<%.3f one-sided is about %+.4f."
          % (dist.std(), len(used), args.alpha,
             np.quantile(dist, args.alpha)))

    hr("HOLDOUT: SIGN AND BINS FROZEN")
    dev = C[C["season"].isin(DEV_SEASONS)]
    hold = C[C["season"].isin(HOLDOUT_SEASONS)]
    print("  dev     %s: %d rows" % (DEV_SEASONS, len(dev)))
    print("  holdout %s: %d rows" % (HOLDOUT_SEASONS, len(hold)))
    d_obs, _ = correlation_stat(dev, market=market)
    h_obs, _ = correlation_stat(hold, market=market)
    print("  dev correlation     %+.4f" % d_obs)
    print("  holdout correlation %+.4f" % h_obs)
    same = np.isfinite(d_obs) and np.isfinite(h_obs) and (d_obs * h_obs > 0)
    print("  same sign: %s" % ("yes" if same else "NO"))

    lowcells = [r for r in roi_by_line(hold, market=market)
                if r[1] <= 25 and r[3] >= 30]
    lowroi = np.nan
    if lowcells:
        tot = sum(r[3] for r in lowcells)
        lowroi = sum(r[2] * r[3] for r in lowcells) / tot
        print("  holdout low-line ROI %+.4f on %d bets (cells %s)"
              % (lowroi, tot, [r[0] for r in lowcells]))

    hr("DECISION, AGAINST THE PRE-REGISTERED RULE")
    c1, c2 = p < args.alpha, same
    c3 = np.isfinite(lowroi) and lowroi > 0
    print("  permutation p < %.3f ......... %s" % (args.alpha,
                                                   "yes" if c1 else "no"))
    print("  holdout same sign ........... %s" % ("yes" if c2 else "no"))
    print("  holdout low lines positive .. %s" % ("yes" if c3 else "no"))
    print()
    if c1 and c2 and c3:
        print("  ALIVE for %s." % market)
        if market == "rushing":
            print("  This CORROBORATES Rule 1's mechanism independently of")
            print("  the band that defines it. It does NOT raise the rule's")
            print("  expected return, change its 46.5 threshold, or license")
            print("  widening it: the ROI is measured on its own band and a")
            print("  monotone pattern says nothing about where to cut.")
    else:
        print("  NULL for %s. That market closes." % market)
        print("  Record the MDE above beside this result.")
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
