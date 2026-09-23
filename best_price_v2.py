"""
BEST AVAILABLE PRICE: is the edge placeable, and at which book?

THE QUESTION THIS SETTLES

Pricing receptions 2.5 and 3.5 against the CONSENSUS line and settling there
produced a holdout ROI of +0.2991 at t +3.83. Settling at FanDuel gave
+0.1475. The consensus number is suspect because a median across eight books
is not a price anyone offered, and the settlement used FanDuel's odds applied
to a line FanDuel did not post.

But the pessimistic reading is not obviously right. If three books sit at the
consensus line on a given prop, betting that number is not hypothetical. The
way to find out is to stop using synthetic lines entirely and evaluate EVERY
BOOK SEPARATELY at its own line and its own odds, then take the best one.
That is a real bet at a named book.

HOW THE BEST PRICE IS CHOSEN, AND WHY NOT SIMPLY THE LOWEST LINE

For an over you want a low line, but you also want good odds, and books trade
those off against each other. A 2.5 at -140 can be worse than a 3.5 at +120.
So "best" cannot be read off the line alone.

Instead, for every book row the model probability is computed AT THAT BOOK'S
LINE, and the edge is that probability minus the breakeven implied by THAT
BOOK'S ODDS. The winner is the row with the highest edge. That is expected
value maximisation over the actual menu, which is what a bettor does.

The blended mean is anchored on the consensus line, because consensus remains
the best available estimate of the truth. Only the SETTLEMENT moves to a real
book.

WHAT ELSE THIS PRODUCES FOR FREE

Item 1 on the new-directions list: a book ranking. Two tables come out of the
same computation.

  - WHICH BOOK WINS. How often each book offers the best price, and the
    realized ROI of the bets it won. A book that wins often and loses money is
    a trap: it is offering the best price because its number is bad in a way
    the model likes and the market does not.
  - BOOK SHARPNESS. Beta measured against each book's own line. The book with
    the LOWEST beta is the sharpest, because it is the one leaving least
    information on the table. That book's line is the one worth treating as
    truth in future work.

TWO CAVEATS THAT THE NUMBERS CANNOT SHOW

  DEFUNCT BOOKS. The cache contains pointsbetus, barstool and unibet_us,
  which are 2023-era brands that no longer exist in the US. PointsBet's US
  business went to Fanatics and Barstool became ESPN Bet. If the best price
  keeps coming from those, the result is historically true and unplaceable
  today. The book table reports every winner so this is visible, and
  --books restricts the menu to ones you can actually use.

  LIMITS. A book offering the best price on an obscure prop may cap the stake
  at a small figure. No dataset shows this. Only a bet slip does.

Run from the repo root:
    python best_price.py --all-seasons --cache lines_cache.parquet \\
        --markets receptions --line-value 2.5,3.5

    python best_price.py --all-seasons --cache lines_cache.parquet \\
        --markets receptions --line-value 2.5,3.5 \\
        --books fanduel,draftkings,betmgm,betrivers
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh
import edge_threshold_v5 as et

BREAKEVEN = 113.0 / 213.0

# Brands that no longer operate in the US. Present in the 2023 data only.
DEFUNCT = {"pointsbetus", "barstool", "unibet_us"}


def implied(o):
    o = np.asarray(o, float)
    out = np.full(o.shape, np.nan)
    neg, pos = o < 0, o > 0
    out[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
    out[pos] = 100.0 / (o[pos] + 100.0)
    return out


def payout(o):
    o = np.asarray(o, float)
    out = np.full(o.shape, np.nan)
    neg, pos = o < 0, o > 0
    out[neg] = 100.0 / (-o[neg])
    out[pos] = o[pos] / 100.0
    return out


# --------------------------------------------------------------------- data

def load(args):
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

    raw = eh.load_lines(factory, seasons, markets, args.cache, args.refresh)
    if raw.empty:
        print("  no lines found")
        sys.exit(1)
    raw = raw[raw["week"].notna()].copy()
    raw["line"] = pd.to_numeric(raw["line"], errors="coerce")
    raw = raw.dropna(subset=["line"])
    raw["week"] = raw["week"].astype(int)

    # The REFERENCE set builds the consensus anchor and the SETTLEMENT menu
    # decides where the bet is placed. They must be separable: restricting
    # both to one book would price against the very line being settled at,
    # which is the mistake that made the synthetic-consensus run look good.
    ref = raw
    if args.reference_books:
        keep = [b.strip() for b in args.reference_books.split(",") if b.strip()]
        ref = raw[raw["book"].isin(keep)]
        print(f"\n  reference set for consensus: {ref['book'].nunique()} books")
    if args.books:
        keep = [b.strip() for b in args.books.split(",") if b.strip()]
        before = raw["book"].nunique()
        raw = raw[raw["book"].isin(keep)]
        print(f"\n  settlement menu: {raw['book'].nunique()} of {before} "
              f"books kept, consensus still from "
              f"{ref['book'].nunique()} books")

    props = eh.collapse_books(ref)
    print(f"\n  {len(props)} distinct player-week props, "
          f"{raw['book'].nunique()} books, {len(raw)} book rows")

    print("\nscoring models walk-forward")
    scored = []
    for m in markets:
        print(f"  {m}")
        sc = eh.score_market(m, seasons, mode="walk_forward",
                             population=args.score_population)
        if len(sc):
            scored.append(sc)
    if not scored:
        sys.exit(1)
    proj = pd.concat(scored, ignore_index=True)

    from models import data_utils
    props["_key"] = data_utils.norm_join_name(props["player"])
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    proj["week"] = proj["week"].astype(int)

    j = props.merge(
        proj[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"\n  joined {len(j)} props "
          f"({100.0 * len(j) / max(len(props), 1):.1f}%)")
    j["dev_cons"] = j["projection"] - j["line_consensus"]
    j["out_cons"] = j["actual"] - j["line_consensus"]
    j = j.dropna(subset=["dev_cons", "out_cons"])

    if args.line_value:
        vals = [round(float(x), 2) for x in args.line_value.split(",")]
        before = len(j)
        j = j[np.isin(np.round(j["line_consensus"], 2), vals)]
        print(f"  consensus line in {vals}: {len(j)} of {before} props")
    return j, raw, markets


# ------------------------------------------------------------- book pricing

def price_every_book(j, raw, params, sig):
    """Evaluate every book row at its own line and its own odds."""
    keys = ["season", "week", "market", "player"]
    anchor = j[keys + ["_key", "line_consensus", "projection", "actual",
                       "event_id"]].copy()

    a = [params.get((m, s), (np.nan,) * 4)[0]
         for m, s in zip(anchor["market"], anchor["season"])]
    b = [params.get((m, s), (np.nan,) * 4)[1]
         for m, s in zip(anchor["market"], anchor["season"])]
    anchor["alpha"], anchor["beta"] = a, b
    anchor = anchor.dropna(subset=["alpha", "beta"])
    anchor["blend"] = (anchor["line_consensus"] + anchor["alpha"]
                       + anchor["beta"] * (anchor["projection"]
                                           - anchor["line_consensus"]))
    sa = [sig.get((m, s), (np.nan, 0.5))[0]
          for m, s in zip(anchor["market"], anchor["season"])]
    anchor["sig_a"] = sa
    anchor = anchor.dropna(subset=["sig_a"])
    anchor["sigma"] = anchor["sig_a"] * np.sqrt(anchor["blend"].clip(lower=1e-6))

    bk = raw[keys + ["book", "line", "over_odds", "under_odds"]].copy()
    d = anchor.merge(bk, on=keys, how="inner")
    print(f"\n  {len(d)} book-level rows to price "
          f"({len(anchor)} props x {len(d) / max(len(anchor), 1):.1f} books)")

    po = np.full(len(d), np.nan)
    for m, idx in d.groupby("market").indices.items():
        po[idx] = et.p_over(m, d["blend"].to_numpy()[idx],
                            d["sigma"].to_numpy()[idx],
                            d["line"].to_numpy()[idx])
    d["p_over"] = po
    oo = pd.to_numeric(d["over_odds"], errors="coerce")
    uo = pd.to_numeric(d["under_odds"], errors="coerce")
    d["be_over"], d["be_under"] = implied(oo), implied(uo)
    d["pay_over"], d["pay_under"] = payout(oo), payout(uo)
    d["edge_over"] = d["p_over"] - d["be_over"]
    d["edge_under"] = (1.0 - d["p_over"]) - d["be_under"]

    # a book row is only usable for the side whose odds it actually quotes
    d["edge_over"] = d["edge_over"].where(np.isfinite(d["be_over"]), -9)
    d["edge_under"] = d["edge_under"].where(np.isfinite(d["be_under"]), -9)
    d["side_over"] = d["edge_over"] >= d["edge_under"]
    d["edge"] = np.maximum(d["edge_over"], d["edge_under"])
    d = d[d["edge"] > -1]

    push = np.isclose(d["actual"], d["line"])
    won = np.where(d["side_over"], d["actual"] > d["line"],
                   d["actual"] < d["line"])
    pay = np.where(d["side_over"], d["pay_over"], d["pay_under"])
    d["push"], d["won"] = push, won
    d["profit"] = np.where(push, 0.0, np.where(won, pay, -1.0))
    return d


def take_best(d):
    """One row per prop: the book offering the highest edge."""
    idx = d.groupby(["season", "week", "market", "player"])["edge"].idxmax()
    best = d.loc[idx].copy()
    print(f"  reduced to {len(best)} best-price bets")
    return best


# -------------------------------------------------------------- the reports

def book_winners(best, min_n=30):
    print("\n" + "=" * 96)
    print("WHICH BOOK OFFERS THE BEST PRICE, AND DOES IT PAY")
    print("=" * 96)
    print(f"  {'book':<18}{'wins':>7}{'share':>8}{'med line':>10}"
          f"{'win rate':>10}{'ROI':>9}{'SE':>8}{'defunct':>9}")
    tot = len(best)
    for bk, g in sorted(best.groupby("book"), key=lambda x: -len(x[1])):
        live = g[~g["push"]]
        cm = (eh.cluster_mean(live["profit"].to_numpy(),
                              live["event_id"].to_numpy())
              if len(live) >= min_n else None)
        print(f"  {str(bk):<18}{len(g):>7}{100.0 * len(g) / tot:>7.1f}%"
              f"{g['line'].median():>10.1f}"
              f"{live['won'].mean() if len(live) else np.nan:>10.4f}"
              f"{cm['mean'] if cm else np.nan:>+9.4f}"
              f"{cm['se'] if cm else np.nan:>8.4f}"
              f"{'YES' if bk in DEFUNCT else '':>9}")
    dead = best[best["book"].isin(DEFUNCT)]
    if len(dead):
        print(f"\n    {len(dead)} of {tot} best prices "
              f"({100.0 * len(dead) / tot:.1f}%) come from DEFUNCT books.")
        print("    Those bets are not placeable today. Re-run with --books "
              "restricted to")
        print("    the ones you hold accounts with to get a usable number.")


def book_sharpness(j, raw, min_n=400):
    """Beta against each book's own line. Lowest beta is the sharpest book."""
    print("\n" + "=" * 96)
    print("BOOK SHARPNESS: beta against each book's own line")
    print("=" * 96)
    print("  Lower beta means the book leaves LESS information on the table,")
    print("  so the lowest row is the sharpest price and the best candidate")
    print("  for a truth anchor in future work.")
    keys = ["season", "week", "market", "player"]
    base = j[keys + ["projection", "actual", "event_id"]]
    bk = raw[keys + ["book", "line"]]
    d = base.merge(bk, on=keys, how="inner")
    print(f"\n  {'book':<18}{'n':>8}{'games':>7}{'beta':>8}{'SE':>7}"
          f"{'t':>7}{'over rate':>11}")
    rows = []
    for b, g in d.groupby("book"):
        if len(g) < min_n:
            continue
        dev = g["projection"] - g["line"]
        out = g["actual"] - g["line"]
        f = eh.ols_clustered(dev, out, g["event_id"])
        if f is None:
            continue
        push = np.isclose(g["actual"], g["line"])
        won = (g["actual"] > g["line"]).astype(float)
        cm = eh.cluster_mean(won[~push].to_numpy(),
                             g["event_id"].to_numpy()[~push])
        rows.append((f["beta"], b, f, cm))
    for beta, b, f, cm in sorted(rows):
        t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
        print(f"  {str(b):<18}{f['n']:>8}{f['clusters']:>7}{f['beta']:>8.3f}"
              f"{f['se_beta']:>7.3f}{t:>+7.2f}"
              f"{cm['mean'] if cm else np.nan:>11.4f}")
    if rows:
        print(f"\n    sharpest: {rows[0][1]} at beta {rows[0][0]:+.3f}")
        print(f"    softest:  {rows[-1][1]} at beta {rows[-1][0]:+.3f}")


def line_profile(j, raw):
    """How each book POSTS its lines, not just where.

    A book whose beta is three times the field's is either pricing badly or
    posting differently, and those look identical in a beta. Things that would
    show up here:

      - integer rather than half-integer lines, which creates pushes and
        systematic deviation from a half-integer consensus
      - a much coarser or finer grid than the field
      - lines systematically above or below consensus, which is a lean rather
        than an error
      - unusual odds, since a book can express its opinion through the price
        instead of the number
    """
    print("\n" + "=" * 96)
    print("HOW EACH BOOK POSTS: line grid, lean, and odds")
    print("=" * 96)
    keys = ["season", "week", "market", "player"]
    cons = j[keys + ["line_consensus"]]
    d = raw[keys + ["book", "line", "over_odds", "under_odds"]].merge(
        cons, on=keys, how="inner")
    if not len(d):
        print("  no overlap")
        return
    d["dev"] = d["line"] - d["line_consensus"]
    d["is_half"] = np.isclose(np.mod(d["line"], 1.0), 0.5)
    print(f"  {'book':<18}{'n':>7}{'distinct':>10}{'% half-int':>12}"
          f"{'mean dev':>10}{'mean |dev|':>12}{'med over':>10}"
          f"{'med under':>11}{'med hold':>10}")
    for b, g in sorted(d.groupby("book")):
        oo = pd.to_numeric(g["over_odds"], errors="coerce")
        uo = pd.to_numeric(g["under_odds"], errors="coerce")
        hold = implied(oo) + implied(uo) - 1.0
        print(f"  {str(b):<18}{len(g):>7}{g['line'].nunique():>10}"
              f"{100.0 * g['is_half'].mean():>11.1f}%"
              f"{g['dev'].mean():>+10.3f}{g['dev'].abs().mean():>12.3f}"
              f"{oo.median():>10.0f}{uo.median():>11.0f}"
              f"{np.nanmedian(hold):>10.4f}")
    print("\n    A book with a low % half-int posts on a different grid and")
    print("    will produce pushes plus a structural deviation from a")
    print("    half-integer consensus. A large mean dev is a LEAN, which is")
    print("    a different thing from a large mean |dev|, which is noise.")


def curve(best, label):
    print("\n" + "=" * 96)
    print(f"THRESHOLD CURVE: {label}")
    print("=" * 96)
    weeks = best.groupby("season")["week"].nunique().sum()
    print(f"  {'min edge':>9}{'bets':>7}{'per wk':>8}{'win rate':>10}"
          f"{'ROI':>9}{'SE':>8}{'t':>7}{'95% CI':>20}")
    for t in np.round(np.arange(0, 0.201, 0.005), 4):
        g = best[(best["edge"] >= t) & (~best["push"])]
        if len(g) < 30:
            continue
        cm = eh.cluster_mean(g["profit"].to_numpy(), g["event_id"].to_numpy())
        if cm is None:
            continue
        tt = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {t:>9.3f}{len(g):>7}{len(g) / max(weeks, 1):>8.1f}"
              f"{g['won'].mean():>10.4f}{cm['mean']:>+9.4f}{cm['se']:>8.4f}"
              f"{tt:>+7.2f}  [{cm['mean'] - 1.96 * cm['se']:>+7.4f},"
              f"{cm['mean'] + 1.96 * cm['se']:>+8.4f}]")


def holdout(best):
    print("\n" + "=" * 96)
    print("HOLDOUT: threshold chosen on other seasons, spent on the held out one")
    print("=" * 96)
    seasons = sorted(best["season"].unique())
    grid = np.round(np.arange(0, 0.201, 0.005), 4)
    print(f"  {'held out':>9}{'chosen t':>10}{'bets':>7}{'per wk':>8}"
          f"{'win rate':>10}{'ROI':>9}{'SE':>8}{'t':>7}")
    rows = []
    for s in seasons:
        tr, te = best[best["season"] != s], best[best["season"] == s]
        bt, bv = None, -np.inf
        for t in grid:
            g = tr[(tr["edge"] >= t) & (~tr["push"])]
            if len(g) < 100:
                continue
            v = float(g["profit"].mean())
            if v > bv:
                bt, bv = t, v
        if bt is None:
            continue
        g = te[(te["edge"] >= bt) & (~te["push"])]
        cm = (eh.cluster_mean(g["profit"].to_numpy(), g["event_id"].to_numpy())
              if len(g) >= 20 else None)
        if cm is None:
            print(f"  {int(s):>9}{bt:>10.3f}{len(g):>7}   too few bets")
            continue
        wk = max(te["week"].nunique(), 1)
        tt = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        print(f"  {int(s):>9}{bt:>10.3f}{len(g):>7}{len(g) / wk:>8.1f}"
              f"{g['won'].mean():>10.4f}{cm['mean']:>+9.4f}{cm['se']:>8.4f}"
              f"{tt:>+7.2f}")
        rows.append(g)
    if rows:
        allg = pd.concat(rows, ignore_index=True)
        cm = eh.cluster_mean(allg["profit"].to_numpy(),
                             allg["event_id"].to_numpy())
        tt = cm["mean"] / cm["se"] if cm["se"] > 0 else np.nan
        lo = cm["mean"] - 1.96 * cm["se"]
        print(f"  {'POOLED':>9}{'':>10}{len(allg):>7}{'':>8}"
              f"{allg['won'].mean():>10.4f}{cm['mean']:>+9.4f}{cm['se']:>8.4f}"
              f"{tt:>+7.2f}")
        print()
        print("  COMPARE AGAINST:")
        print("    +0.1475 (t +1.74)  settling at FanDuel only")
        print("    +0.2991 (t +3.83)  settling at the synthetic consensus")
        print()
        if lo > 0:
            print("  The CI excludes zero at the BEST AVAILABLE REAL PRICE.")
            print("  Check the book table above before believing it: if the")
            print("  winners are defunct brands, this is not placeable.")
        else:
            print("  The CI includes zero. Shopping across real prices does")
            print("  not by itself establish the edge.")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--markets", default="receptions")
    ap.add_argument("--line-value", default=None,
                    help="restrict to these CONSENSUS line values, e.g. 2.5,3.5")
    ap.add_argument("--books", default=None,
                    help="restrict the SETTLEMENT menu to these books. The "
                         "consensus anchor still uses every book unless "
                         "--reference-books is given.")
    ap.add_argument("--reference-books", default=None,
                    help="books used to build the consensus anchor. Defaults "
                         "to all of them. Separate from --books on purpose.")
    ap.add_argument("--exclude-defunct", action="store_true",
                    help="drop pointsbetus, barstool and unibet_us")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.season and not args.seasons:
        args.seasons = str(args.season)
    if args.exclude_defunct:
        args.books = ",".join(sorted(
            set(["draftkings", "betonlineag", "williamhill_us", "betmgm",
                 "fanduel", "betrivers", "bovada", "fanatics"])))

    print("=" * 96)
    print("BEST AVAILABLE PRICE v2: reference set separate from settlement")
    print("=" * 96)

    j, raw, markets = load(args)
    if len(j) < 300:
        print("  too few props")
        return

    params = et.fit_params(j)
    if not params:
        print("  could not fit shrinkage parameters")
        return
    et.report_params(params)
    sig = et.fit_sigma_sqrt(j, params)
    print("\n  SIGMA = a * sqrt(mean), exponent imposed at 0.5")
    for (m, s) in sorted(sig):
        print(f"    {m} {int(s)}: a = {sig[(m, s)][0]:.3f}")

    book_sharpness(j, raw)
    line_profile(j, raw)

    d = price_every_book(j, raw, params, sig)
    best = take_best(d)

    book_winners(best)
    curve(best, "best available price")
    holdout(best)

    print("\n" + "=" * 96)
    print("HOW TO READ THIS")
    print("=" * 96)
    print("  Every bet here is at a REAL book's line and a REAL book's odds,")
    print("  so unlike the consensus run it is placeable in principle.")
    print("  Two things can still make it unplaceable in practice. Defunct")
    print("  brands, which the book table flags and --exclude-defunct removes.")
    print("  And limits, which no dataset can show and only a bet slip can.")
    print("  The sharpness table is the durable finding regardless of the ROI:")
    print("  the lowest-beta book is the best truth anchor available, and that")
    print("  is worth more to future work than any single threshold.")


if __name__ == "__main__":
    main()
