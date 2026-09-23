"""
ATTENTION STRATIFICATION v2: is the market soft where nobody looks?

WHAT v1 FOUND, AND IT IS THE FIRST MECHANISM THIS PROJECT HAS SEEN

Stratified by LINE LEVEL within market, beta falls monotonically as the line
rises, in two markets independently:

  qb_passing  Q1 +0.174 (med 192)  -> +0.123 -> +0.052 -> Q4 -0.303
  receiving   Q1 +0.122 (med 12.5) -> +0.088 -> +0.012 -> Q4 +0.012

Spearman -1.00 in both. And the single best-powered cell in the whole run was
receptions Q1: beta +0.216, t +4.83, on 3,687 rows at a median line of 2.5.

That is the low-line, low-volume population: the WR3s, the tight ends, the
backup backs. It is exactly the hypothesis that the pooled numbers were hiding
a soft pocket, and it now has a monotone trend behind it rather than one lucky
cell.

THREE BUGS IN v1, ALL MINE, ALL FIXED HERE

  1. THE n_books BINS DROPPED 19 PERCENT OF THE DATA. n_books reaches 10 in
     the data and the bins were [0,2,4,6,8], so every prop quoted by 9 or 10
     books became null and vanished. Receiving showed 7,007 rows across four
     buckets against 8,676 joined. Bins now cover the full observed range and
     the script ASSERTS that no rows are lost.
  2. RECEPTIONS GOT NO SPREAD TABLE AT ALL. Reception spreads are mostly 0,
     0.5 and 1.0, so qcut collapsed below four bins, a fixed label list
     mismatched, and a bare `except: pass` swallowed the error. Bucketing now
     uses whatever categories qcut actually produces and says how many it got.
  3. THE HOLDOUT TESTED THE MAX CELL RATHER THAN THE TREND. It picked
     receptions Q4, the highest single in-sample beta, and ignored the
     monotone Q1 signal entirely. Unsurprisingly it failed, because the
     maximum of 48 cells is high by construction. The holdout now tests a
     PRE-REGISTERED hypothesis taken from the trend.

WHY n_books WAS THE WRONG PROXY ANYWAY

Books auto-price every prop from their own models, so n_books measures
AGGREGATOR COVERAGE, not market attention. The median is 7, meaning almost
everything gets quoted. There is no obscurity pocket for it to find. Kept in
the output for completeness, but do not read much into it.

TWO BETTER PROXIES, BOTH ALREADY IN THE CACHE

  4. FANDUEL'S HOLD, computed from the actual over and under odds as
     P(breakeven over) + P(breakeven under) - 1. Books charge WIDER margins on
     props they are less confident about, because margin is how you protect
     yourself against adverse selection when you do not trust your own number.
     A high-hold prop is one FanDuel is nervous about, which is a direct read
     on their confidence rather than a proxy for it.

     It cuts both ways: a wider hold is also harder to beat, since you need a
     bigger edge to clear it. The interesting cell is high beta AND acceptable
     hold, so the table prints both together.

  5. PLAYER PROP FREQUENCY, how many player-weeks across the dataset that
     player has props for. A player quoted in 60 weeks is a household name the
     market models carefully. A player quoted in 4 weeks had a brief run of
     relevance, which is exactly when the market has least history to price
     from and the EWMA features have most to say. Unlike n_books this varies
     enormously, and it is the closest available stand-in for "nobody is
     looking at this player".

WHAT IS NOT TESTABLE YET

Line MOVEMENT is the best attention proxy there is: a prop whose line never
moves all week is one nobody is betting. But `historical_lines` holds one
closing snapshot per event, so there is no movement in it. Movement lives in
the `lines` table, which currently covers 2026 weeks 1 to 3 only, far too thin
for a beta with a usable standard error. THIS TEST UNLOCKS AS THE AUTOMATED
CAPTURE ACCUMULATES, and it should be the first thing run once 2026 has six or
eight weeks of multi-snapshot data.

Run from the repo root:
    python attention_strata.py --all-seasons --cache lines_cache.parquet
    python attention_strata.py --all-seasons --cache lines_cache.parquet --source consensus
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh

BREAKEVEN = 113.0 / 213.0


def implied(odds):
    o = np.asarray(odds, float)
    out = np.full(o.shape, np.nan)
    neg, pos = o < 0, o > 0
    out[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
    out[pos] = 100.0 / (o[pos] + 100.0)
    return out


# --------------------------------------------------------------------- data

def build_joined(args):
    sec = eh._secrets()
    box = {}

    def factory():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(x) for x in args.seasons.split(",") if x.strip()]
    else:
        seasons = [2025]

    lines = eh.load_lines(factory, seasons, markets, args.cache, args.refresh)
    if lines.empty:
        print("  no lines found")
        sys.exit(1)
    lines = lines[lines["week"].notna()]

    # n_books hit 10 in v1 against a backfill that used 8 books, which
    # usually means duplicate or inconsistent book naming. Print the roster.
    bk = lines["book"].value_counts()
    print(f"\n  {len(bk)} distinct book names in the cache:")
    for name, n in bk.items():
        print(f"    {str(name):<22}{n:>9}")

    props = eh.collapse_books(lines)
    print(f"\n  {len(props)} distinct player-week props")

    print("\nscoring models walk-forward")
    scored = []
    for m in markets:
        print(f"  {m}")
        sc = eh.score_market(m, seasons, mode="walk_forward",
                             population=args.score_population)
        if len(sc):
            scored.append(sc)
    if not scored:
        print("  nothing scored")
        sys.exit(1)
    proj = pd.concat(scored, ignore_index=True)

    from models import data_utils
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    props["week"] = props["week"].astype(int)
    proj["week"] = proj["week"].astype(int)

    j = props.merge(
        proj[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"\n  joined {len(j)} rows "
          f"({100.0 * len(j) / max(len(props), 1):.1f}% of props)")

    col = f"line_{args.source}"
    j = j.dropna(subset=[col, "projection", "actual", "n_books"]).copy()
    j["dev"] = j["projection"] - j[col]
    j["out"] = j["actual"] - j[col]
    j["line_used"] = j[col]

    oo = pd.to_numeric(j.get("over_odds"), errors="coerce")
    uo = pd.to_numeric(j.get("under_odds"), errors="coerce")
    j["hold"] = implied(oo) + implied(uo) - 1.0

    freq = j.groupby("_key").size().rename("player_freq")
    j = j.merge(freq, left_on="_key", right_index=True, how="left")
    return j


# ------------------------------------------------------------------- strata

def bucket(j, col, name, nq=4):
    """Quantile buckets using WHATEVER categories qcut produces, within market.

    v1 passed a fixed list of four labels. Where the underlying values were
    discrete enough that qcut collapsed to fewer bins, the label count
    mismatched, the exception was swallowed, and the market silently vanished
    from the table. Receptions lost its entire spread analysis that way.
    """
    out = pd.Series(pd.NA, index=j.index, dtype="object")
    rank = pd.Series(np.nan, index=j.index)
    for m, g in j.groupby("market"):
        if g[col].notna().sum() < 240:
            print(f"    note: {name} for {m}: only "
                  f"{int(g[col].notna().sum())} non-null, skipped")
            continue
        try:
            q = pd.qcut(g[col], nq, duplicates="drop")
        except Exception as e:
            print(f"    note: {name} bucketing failed for {m}: {e}")
            continue
        cats = list(q.cat.categories)
        if len(cats) < 2:
            print(f"    note: {name} for {m} collapsed to one bucket, skipped")
            continue
        if len(cats) < nq:
            print(f"    note: {name} for {m} gave {len(cats)} buckets not "
                  f"{nq} (discrete values)")
        lab = {c: f"B{i + 1}" for i, c in enumerate(cats)}
        codes = q.cat.codes
        out.loc[g.index] = q.map(lab).astype(object)
        rank.loc[g.index] = codes.replace(-1, np.nan).to_numpy()
    j[f"s_{name}"] = out
    j[f"r_{name}"] = rank
    return j


def add_strata(j):
    j = j.copy()

    # n_books bins must cover the FULL observed range. v1 used [0,2,4,6,8]
    # while the data reaches 10, silently dropping 19 percent of rows.
    hi = int(j["n_books"].max())
    edges = sorted(set([0, 4, 6, 7, 8] + ([hi] if hi > 8 else [])))
    labs = [f"{a + 1}" if b == a + 1 else f"{a + 1}-{b}"
            for a, b in zip(edges[:-1], edges[1:])]
    j["s_books"] = pd.cut(j["n_books"], edges, labels=labs)
    j["r_books"] = j["s_books"].cat.codes.replace(-1, np.nan).astype(float)
    lost = int(j["s_books"].isna().sum())
    print(f"\n  n_books buckets {labs}, rows outside the bins: {lost}")
    if lost:
        raise SystemExit("n_books bins do not cover the data, fix the edges")

    for col, name in (("line_used", "line"), ("book_spread", "spread"),
                      ("hold", "hold"), ("player_freq", "freq")):
        j = bucket(j, col, name)
    return j


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else np.nan


def report(j, key, label, min_rows=250):
    print("\n" + "=" * 100)
    print(f"STRATIFIED BY {label}")
    print("=" * 100)
    scheme, rankcol = f"s_{key}", f"r_{key}"
    if scheme not in j.columns:
        print("  not available")
        return
    for m, g in j.groupby("market"):
        sub_all = g[g[scheme].notna()]
        if not len(sub_all):
            print(f"\n  {m}: no buckets")
            continue
        print(f"\n  {m}   ({len(sub_all)} of {len(g)} rows bucketed)")
        print(f"  {'bucket':<8}{'n':>7}{'games':>7}{'beta':>8}{'SE':>7}"
              f"{'t':>7}{'over':>8}{'hold':>8}{'med line':>10}{'med freq':>10}")
        betas, ranks = [], []
        for b in sorted(sub_all[scheme].unique(), key=str):
            s = sub_all[sub_all[scheme] == b]
            if len(s) < min_rows:
                print(f"  {str(b):<8}{len(s):>7}   too few")
                continue
            f = eh.ols_clustered(s["dev"], s["out"], s["event_id"])
            if f is None:
                continue
            push = np.isclose(s["actual"], s["line_used"])
            won = (s["actual"] > s["line_used"]).astype(float)
            cm = eh.cluster_mean(won[~push].to_numpy(),
                                 s["event_id"].to_numpy()[~push])
            t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
            print(f"  {str(b):<8}{f['n']:>7}{f['clusters']:>7}"
                  f"{f['beta']:>8.3f}{f['se_beta']:>7.3f}{t:>+7.2f}"
                  f"{(cm['mean'] if cm else np.nan):>8.4f}"
                  f"{s['hold'].median():>8.4f}"
                  f"{s['line_used'].median():>10.1f}"
                  f"{s['player_freq'].median():>10.0f}")
            betas.append(f["beta"])
            ranks.append(float(np.nanmedian(s[rankcol].to_numpy())))
        if len(betas) >= 3:
            rho = spearman(ranks, betas)
            tag = "MONOTONE" if abs(rho) > 0.79 else "not monotone"
            print(f"    trend: spearman(bucket, beta) = {rho:+.2f}   {tag}")


# ------------------------------------------------------------- the holdout

def holdout(j, min_rows=150):
    """PRE-REGISTERED test, not a search over cells.

    v1 picked the single highest in-sample beta and tested that, which is the
    maximum of 48 cells and high by construction. The hypothesis here is fixed
    in advance from the TREND: the lowest line quartile beats the line by more
    than the highest. Two numbers per market, measured on 2025-2026, which the
    hypothesis never saw.
    """
    print("\n" + "=" * 100)
    print("HOLDOUT: pre-registered low-line hypothesis, measured on 2025-2026")
    print("=" * 100)
    print("  Hypothesis, fixed from the v1 trend and NOT searched over:")
    print("    beta is higher in the LOWEST line quartile than in the highest.")
    print()
    early = j[j["season"].isin([2023, 2024])]
    late = j[j["season"].isin([2025, 2026])]

    print(f"  {'market':<12}{'half':<10}{'Q1 beta':>10}{'SE':>7}{'t':>7}"
          f"{'Q4 beta':>10}{'SE':>7}{'Q1-Q4':>9}{'Q1 n':>7}")
    for m in sorted(j["market"].unique()):
        for tag, half in (("2023-24", early), ("2025-26", late)):
            g = half[(half["market"] == m) & half["r_line"].notna()]
            if not len(g):
                continue
            lo = g[g["r_line"] == g["r_line"].min()]
            hi = g[g["r_line"] == g["r_line"].max()]
            if len(lo) < min_rows or len(hi) < min_rows:
                print(f"  {m:<12}{tag:<10}  too few rows "
                      f"(Q1 {len(lo)}, Q4 {len(hi)})")
                continue
            fl = eh.ols_clustered(lo["dev"], lo["out"], lo["event_id"])
            fh = eh.ols_clustered(hi["dev"], hi["out"], hi["event_id"])
            if fl is None or fh is None:
                continue
            tl = fl["beta"] / fl["se_beta"] if fl["se_beta"] > 0 else np.nan
            print(f"  {m:<12}{tag:<10}{fl['beta']:>10.3f}{fl['se_beta']:>7.3f}"
                  f"{tl:>+7.2f}{fh['beta']:>10.3f}{fh['se_beta']:>7.3f}"
                  f"{fl['beta'] - fh['beta']:>+9.3f}{fl['n']:>7}")
        print()

    print("  READ IT LIKE THIS. The 2023-24 row is where the hypothesis came")
    print("  from, so it is not evidence. The 2025-26 row is the test. A Q1")
    print("  beta clearly above zero with a positive Q1-Q4 gap, in a market")
    print("  that showed the trend, is a real finding. The same sign in more")
    print("  than one market independently is stronger still, since two")
    print("  markets agreeing by chance is much less likely than one.")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--markets", default=",".join(eh.DEFAULT_MARKETS))
    ap.add_argument("--source", default="fanduel",
                    choices=["fanduel", "consensus"])
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--min-rows", type=int, default=250)
    ap.add_argument("--save-rows", default=None)
    args = ap.parse_args()
    if args.season and not args.seasons:
        args.seasons = str(args.season)

    print("=" * 100)
    print("ATTENTION STRATIFICATION v2")
    print(f"  measuring against line_{args.source}")
    print("=" * 100)

    j = build_joined(args)
    j = add_strata(j)

    print("\n  spread of each proxy:")
    print(f"    n_books      min {int(j['n_books'].min())}, "
          f"median {int(j['n_books'].median())}, max {int(j['n_books'].max())}")
    print(f"    book_spread  median {j['book_spread'].median():.2f}, "
          f"90th {j['book_spread'].quantile(0.90):.2f}")
    print(f"    hold         median {j['hold'].median():.4f}, "
          f"10th {j['hold'].quantile(0.10):.4f}, "
          f"90th {j['hold'].quantile(0.90):.4f}, "
          f"missing {int(j['hold'].isna().sum())}")
    print(f"    player_freq  median {j['player_freq'].median():.0f}, "
          f"10th {j['player_freq'].quantile(0.10):.0f}, "
          f"90th {j['player_freq'].quantile(0.90):.0f}")
    print("\n  bucket labels are B1 to B4, LOW value to HIGH value, per market")

    report(j, "line", "LINE LEVEL WITHIN MARKET  (the v1 finding)",
           args.min_rows)
    report(j, "freq", "PLAYER PROP FREQUENCY  (obscurity, new)", args.min_rows)
    report(j, "hold", "FANDUEL HOLD  (their own confidence, new)",
           args.min_rows)
    report(j, "spread", "DISAGREEMENT BETWEEN BOOKS", args.min_rows)
    report(j, "books", "NUMBER OF BOOKS  (coverage, weak proxy)",
           args.min_rows)

    holdout(j)

    if args.save_rows:
        j.to_csv(args.save_rows, index=False)
        print(f"\n  wrote {args.save_rows}")

    print("\n" + "=" * 100)
    print("WHAT TO DO WITH THIS")
    print("=" * 100)
    print("  If the low-line effect survives the holdout in receptions or")
    print("  receiving, the next step is NOT to ship it. It is to re-run")
    print("  edge_threshold.py restricted to that stratum, because a higher")
    print("  beta still has to clear a 6 percent hold before it is a bet.")
    print("  Watch the hold column while you are there: a soft prop with a")
    print("  wide margin can be beatable and still unprofitable.")
    print("  And remember what no dataset can show. If the pocket is real, it")
    print("  is real because LIMITS keep the big money out, which implies many")
    print("  small bets on obscure props, and only a bet slip confirms the")
    print("  stake is actually available.")


if __name__ == "__main__":
    main()
