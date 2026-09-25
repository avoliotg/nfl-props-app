"""coverage_test.py - does the rushing edge live in props few books post?

Reports only. Writes nothing, places no bet.

THE OBSERVATION THAT PROMPTED THIS

    cross_book.py found the low-line rushing under bias is market-wide: six
    of seven stable books shade the under by almost the same amount, with
    win rates in a band of 0.548 to 0.561, and FanDuel is the place to play
    it only because its hold is 4.92 percent against 6.02 to 7.07 elsewhere.

    But section 3 restricted to the 651 props that EVERY stable book posted
    and the edge largely vanished: FanDuel fell from +0.0311 raw edge to
    +0.0116, and three of seven books went negative. Arithmetic on FanDuel's
    two cells (1,425 bets at +0.0589, of which 651 at +0.0221) implies the
    other 774 ran at roughly +0.090.

    So the edge appears to concentrate in props that not every book bothers
    to post. That is consistent with the template-pricing observation, since
    the props only some books carry are the most obscure. It also has an
    uncomfortable consequence: obscure props are where LIMITS are smallest,
    so the profitable subset may be the unstakeable one.

WHY RAW n_books WOULD BE THE WRONG MEASURE

    The cache holds 11 books in 2023 and 7 to 8 by 2025, so "posted by 8
    books" means something different in each season. Raw counts are
    confounded with the calendar. This script uses COVERAGE: the share of
    that week's ACTIVE books which posted the prop, where active means the
    book posted at least one prop that week. That is comparable across
    seasons by construction.

THE CONFOUND THAT MATTERS MOST

    Coverage is not a clean instrument for obscurity. Low-coverage props are
    low-line props on low-volume players, and line value is already known to
    drive this effect. So coverage and line are entangled, and a coverage
    gradient could be the line gradient wearing a different label.

    Section 4 crosses the two so the marginal contribution of each is
    visible. Section 5 regresses profit on both with clustered errors and
    asks whether coverage survives controlling for line. If it does not, the
    section 3 result in cross_book was the line effect and nothing new.

    A second confound with a cheaper answer: low-coverage props might simply
    carry a WORSE price, which would show up as edge without showing up as
    ROI, or vice versa. Section 2 reports hold by coverage so that is visible
    rather than assumed.

STEP OR GRADIENT

    cross_book's section 3 used a single hard threshold, all seven stable
    books or not. That is exactly the kind of cut this project has been
    burned by. If ROI falls smoothly as coverage rises, the mechanism is
    real. If it is a cliff at one value, the cliff is probably how I built
    that section.

Run from the repo root:
    python coverage_test.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import eval_harness as eh
import recency_test as rt
import under_bias as ub

MARKET = "rushing"
CUT = 46.5
BOOK = "fanduel"

# Published by cross_book.py and under_bias.py. The gate aborts unless both
# reproduce, because this script exists to explain the gap between them.
PUB_ALL = (1425, 0.0589)
PUB_MATCHED = (651, 0.0221)


def _rule(t):
    print()
    print("=" * 94)
    print(t)
    print("=" * 94)


def head(first="cut"):
    print(f"  {first:<28}{'bets':>7}{'per szn':>9}{'under win':>11}"
          f"{'breakeven':>11}{'edge':>8}{'ROI':>9}{'lo95':>9}{'hi95':>9}")


def emit(label, g, n_boot, n_seasons=3.25, seed=0):
    prof = g["roi_under"].to_numpy()
    roi = prof.mean()
    win = 1.0 - g["over"].mean()
    be = (1.0 / g["dec_under"]).mean()
    lo, hi, _ = ub.boot_ci(prof, g["event_id"], n_boot, seed)
    star = "  <--" if np.isfinite(lo) and lo > 0 else ""
    print(f"  {label:<28}{len(g):>7,}{len(g) / n_seasons:>9.0f}{win:>11.4f}"
          f"{be:>11.4f}{win - be:>+8.4f}{roi:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}"
          f"{star}")
    return roi, lo, hi, win - be


def add_coverage(p):
    """Share of that week's ACTIVE books which posted each prop."""
    _rule("SECTION 1: BUILDING THE COVERAGE MEASURE")
    keyc = ["season", "week", "market", "_key"]
    active = (p.groupby(["season", "week"])["book"].nunique()
              .rename("books_active").reset_index())
    nb = (p.groupby(keyc)["book"].nunique().rename("n_books").reset_index())
    cov = nb.merge(active, on=["season", "week"], how="left")
    cov["coverage"] = cov["n_books"] / cov["books_active"]
    print("  books active per week, by season:")
    print(active.groupby("season")["books_active"]
          .agg(["min", "median", "max"]).to_string())
    print()
    print("  coverage distribution on rushing props:")
    r = cov[cov["market"] == MARKET]
    print(r["coverage"].describe(
        percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
    print()
    print("  n_books versus coverage, showing why the raw count is unusable:")
    print(r.groupby("season")["n_books"].agg(["median"]).join(
        r.groupby("season")["coverage"].agg(["median"]),
        lsuffix="_n_books", rsuffix="_coverage").round(3).to_string())
    return p.merge(cov[keyc + ["n_books", "books_active", "coverage"]],
                   on=keyc, how="left")


def section_descriptives(p):
    _rule("SECTION 2: IS COVERAGE JUST THE LINE, OR JUST THE PRICE?")
    g = p[(p["book"] == BOOK) & (p["market"] == MARKET) &
          (p["line"] <= CUT) & p["coverage"].notna()].copy()
    print(f"  {len(g):,} FanDuel low-line rushing bets with coverage")
    print()
    print(f"  correlation of coverage with line: "
          f"{g['coverage'].corr(g['line']):+.4f}")
    print("  (a strong positive correlation means the two are entangled and")
    print("   section 5's control is doing the real work)")
    print()
    g["_cb"] = pd.cut(g["coverage"], [0, .5, .7, .85, .99, 1.01],
                      labels=["<50%", "50-70%", "70-85%", "85-99%", "all"])
    print(f"  {'coverage':<12}{'bets':>7}{'mean line':>11}{'mean hold':>11}"
          f"{'line SD':>10}")
    for lab, gg in g.groupby("_cb", observed=True):
        print(f"  {str(lab):<12}{len(gg):>7,}{gg['line'].mean():>11.2f}"
              f"{gg['hold'].mean() * 100:>10.2f}%{gg['line'].std():>10.2f}")
    print()
    print("  If hold rises as coverage falls, part of any ROI gradient is")
    print("  price rather than bias, and the 'edge' column separates them.")
    return g


def section_gradient(g, n_boot):
    _rule("SECTION 3: STEP OR GRADIENT?")
    print("ROI by coverage bucket. A smooth decline as coverage rises means")
    print("the mechanism is real. A cliff at 'all' means cross_book's hard")
    print("all-seven-books threshold created the result.")
    print()
    head("coverage")
    cells = []
    for lab, gg in g.groupby("_cb", observed=True):
        if len(gg) < 120:
            print(f"  {str(lab):<28}{len(gg):>7,}   too few to bootstrap")
            continue
        cells.append((str(lab), *emit(str(lab), gg, n_boot)))
    print()
    print("  and the cumulative version, since that is how a rule would be")
    print("  written in practice (take everything below a coverage cap):")
    print()
    head("coverage <=")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 1.01):
        gg = g[g["coverage"] <= thr]
        if len(gg) < 120:
            continue
        emit(f"{thr:.2f}", gg, n_boot)
    return cells


def section_cross(g, n_boot):
    _rule("SECTION 4: LINE CROSSED WITH COVERAGE")
    print("The two are entangled, so this shows each one's contribution at")
    print("fixed levels of the other. Cells under 100 bets are blank.")
    g = g.copy()
    g["_lb"] = pd.cut(g["line"], [0, 20.5, 30.5, 40.5, CUT],
                      labels=["<=20.5", "20.5-30.5", "30.5-40.5",
                              "40.5-46.5"])
    labs = [l for l in g["_cb"].cat.categories if (g["_cb"] == l).any()]
    print()
    print(f"  {'line':<14}" + "".join(f"{str(l):>12}" for l in labs))
    for lb, gl in g.groupby("_lb", observed=True):
        row = []
        for cl in labs:
            cell = gl[gl["_cb"] == cl]
            row.append(f"{cell['roi_under'].mean():>+12.4f}"
                       if len(cell) >= 100 else f"{'-':>12}")
        print(f"  {str(lb):<14}" + "".join(row))
    print()
    print(f"  {'line (n)':<14}" + "".join(f"{str(l):>12}" for l in labs))
    for lb, gl in g.groupby("_lb", observed=True):
        row = []
        for cl in labs:
            cell = gl[gl["_cb"] == cl]
            row.append(f"{len(cell):>12,}" if len(cell) else f"{'-':>12}")
        print(f"  {str(lb):<14}" + "".join(row))
    print()
    print("  Read DOWN a column to see the line effect at fixed coverage, and")
    print("  ACROSS a row to see the coverage effect at fixed line. Whichever")
    print("  direction holds up is the one carrying the information.")


def section_regression(g):
    _rule("SECTION 5: DOES COVERAGE SURVIVE CONTROLLING FOR THE LINE?")
    print("Profit per unit staked on the under, regressed on coverage and")
    print("line, clustered on game. This is the decisive test of whether")
    print("coverage is a separate effect or the line wearing a label.")
    print()
    g = g.dropna(subset=["coverage", "line", "roi_under"])
    y = g["roi_under"].to_numpy()
    ev = g["event_id"].to_numpy()
    specs = [
        ("coverage alone", ["coverage"]),
        ("line alone", ["line"]),
        ("coverage + line", ["coverage", "line"]),
    ]
    print(f"  {'spec':<22}{'term':<12}{'coef':>10}{'SE':>9}{'t':>7}")
    for name, terms in specs:
        X = np.column_stack([np.ones(len(g))] + [g[t].to_numpy() for t in terms])
        f = rt.ols_cluster(X, y, ev)
        if f is None:
            continue
        for i, t in enumerate(terms, start=1):
            tt = f["beta"][i] / f["se"][i] if f["se"][i] else np.nan
            print(f"  {name if i == 1 else '':<22}{t:<12}"
                  f"{f['beta'][i]:>+10.4f}{f['se'][i]:>9.4f}{tt:>7.2f}")
    print()
    print("  A negative coverage coefficient means less-covered props are")
    print("  more profitable.")
    print()
    print("  THIS TEST HAS ASYMMETRIC POWER, AND THAT IS NOT A DETAIL.")
    print("  Measured on synthetic data with coverage and line correlated at")
    print("  0.74, which is roughly what section 2 reports on the real rows:")
    print()
    print("    when LINE was the true driver, the joint spec was decisive:")
    print("      coverage collapsed to t +0.08, line held at t -4.54")
    print("    when COVERAGE was the true driver, BOTH collapsed:")
    print("      coverage t -1.02, line t -1.43, neither significant")
    print()
    print("  So this regression can RULE COVERAGE OUT but cannot confirm it.")
    print("  Read the outcomes as:")
    print("    line survives, coverage dies  -> coverage was the line. Solid.")
    print("    both survive                  -> both matter. Solid.")
    print("    both die                      -> UNINFORMATIVE, not a null.")
    print("                                     Multicollinearity ate the")
    print("                                     power. Fall back on section 4")
    print("                                     and read across the rows.")


def section_seasons(g, n_boot):
    _rule("SECTION 6: SEASON SPLIT ON THE LOW-COVERAGE CELL")
    sub = g[g["coverage"] <= 0.7]
    if len(sub) < 200:
        print("  too few low-coverage bets to split.")
        return
    print(f"  coverage <= 70%, {len(sub):,} bets")
    print()
    head("season")
    signs = []
    for season, gg in sub.groupby("season"):
        if len(gg) < 60:
            print(f"  {season:<28}{len(gg):>7,}   too few")
            continue
        r, _, _, _ = emit(str(season), gg, max(400, n_boot // 3), 1.0)
        signs.append(np.sign(r))
    print()
    print("  SIGN CONSISTENT" if signs and all(s > 0 for s in signs)
          else "  SIGN FLIPS, unreliable")


def section_honesty(g):
    _rule("SECTION 7: WHAT TO DO WITH THIS")
    low = g[g["coverage"] <= 0.7]
    print(f"  bets at coverage <= 70%: {len(low):,} total, "
          f"about {len(low) / 3.25:.0f} per season")
    print()
    print("  THE PRACTICAL PROBLEM, AND IT IS NOT STATISTICAL.")
    print()
    print("  Coverage is a proxy for obscurity, and obscurity is also a proxy")
    print("  for the BET LIMIT. If the edge really does concentrate in props")
    print("  few books post, then it concentrates precisely where FanDuel")
    print("  caps the stake hardest. The profitable subset may be the")
    print("  unstakeable one, and the stakeable subset is the widely-posted")
    print("  props where cross_book measured almost no edge.")
    print()
    print("  No amount of further analysis resolves that. It takes opening")
    print("  the app on a handful of these props and trying to enter a large")
    print("  stake. Until that number exists, every ROI here is hypothetical.")
    print()
    print("  AND a reminder about what this script is NOT. Coverage was")
    print("  chosen to explain a gap that was itself found by slicing. This")
    print("  is a hypothesis about a hypothesis, on the same data, and its")
    print("  interval does not account for either layer of searching.")
    print("  Forward testing is the only thing that settles it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--boot", type=int, default=1500)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 94)
    print("COVERAGE: does the rushing edge live in props few books post?")
    print(f"  seasons {seasons}   market {MARKET}   line <= {CUT}   "
          f"book {BOOK}")
    print("=" * 94)

    # All books, because coverage is defined across books. Reuses
    # under_bias' loader and settler rather than restating either.
    p = ub.build(types.SimpleNamespace(
        cache=args.cache, refresh=args.refresh, book="all"), seasons)

    _rule("GATE: both published cells must reproduce")
    fd = p[p["book"] == BOOK]
    sub = fd[(fd["market"] == MARKET) & (fd["line"] <= CUT)]
    n1, r1 = len(sub), sub["roi_under"].mean()
    print(f"  all low-line   : {n1:,} bets, ROI {r1:+.4f}   "
          f"published {PUB_ALL[0]:,} / {PUB_ALL[1]:+.4f}")
    ok = n1 == PUB_ALL[0] and abs(r1 - PUB_ALL[1]) <= 0.001

    stable = ["betmgm", "betonlineag", "betrivers", "bovada", "draftkings",
              "fanduel", "williamhill_us"]
    lowall = p[(p["market"] == MARKET) & (p["line"] <= CUT) &
               (p["book"].isin(stable))]
    keyc = ["season", "week", "market", "_key"]
    cnt = lowall.groupby(keyc)["book"].nunique()
    full = cnt[cnt == len(stable)].index
    m = sub.set_index(keyc)
    m = m.loc[m.index.isin(full)]
    n2, r2 = len(m), m["roi_under"].mean()
    print(f"  matched all-7  : {n2:,} bets, ROI {r2:+.4f}   "
          f"published {PUB_MATCHED[0]:,} / {PUB_MATCHED[1]:+.4f}")
    ok = ok and n2 == PUB_MATCHED[0] and abs(r2 - PUB_MATCHED[1]) <= 0.001
    if not ok:
        print()
        print("  ABORT: the two cells this script exists to reconcile do not")
        print("  reproduce, so any explanation of the gap between them would")
        print("  be explaining the wrong gap.")
        sys.exit(1)
    print("  GATE PASSED.")

    p = add_coverage(p)
    g = section_descriptives(p)
    if len(g) < 500:
        print("too few rows.")
        sys.exit(1)
    section_gradient(g, args.boot)
    section_cross(g, args.boot)
    section_regression(g)
    section_seasons(g, args.boot)
    section_honesty(g)

    _rule("DONE. Nothing was written. No bet was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
