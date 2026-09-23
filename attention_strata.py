"""
ATTENTION STRATIFICATION: is the market soft where nobody is looking?

THE HYPOTHESIS

Every beta measured so far pools an entire market together. Receptions at
0.275 is the average across Travis Kelce and the WR3 on a 2.5 line. If the
market is sharp on the props the money cares about and soft on the ones it
does not, pooling averages a real positive against a real negative and hands
back something near zero.

So "the model measures nothing" was only ever entitled to be "the model
measures nothing ON AVERAGE ACROSS ALL PROPS". This script tests the other
claim.

THREE ATTENTION PROXIES, ALL FREE, ALL ALREADY IN THE CACHE

  n_books      how many books bothered to post the prop. A prop only one or
               two books will quote is close to a definition of one nobody is
               trying hard to price. Observed range in the data: 1 to 8.
  book_spread  max line minus min line across books. Disagreement between
               books is uncertainty made visible.
  line level   the bottom quartile of receiving lines is the WR3s and the
               backup running backs. The top quartile is the players the
               syndicates actually model.

collapse_books already computes n_books and book_spread and then the harness
throws them away at the join, which is the same mistake that hid n_train.

WHAT WOULD COUNT AS A POSITIVE RESULT

Not "one cell has a high beta". With four markets, three schemes and four
buckets there are 48 cells, so the best one is high by construction. Two
things are required:

  1. A MONOTONE TREND across the buckets. Beta rising as attention falls is a
     mechanism. Beta high in bucket 2 and low in buckets 1, 3 and 4 is noise.
     Spearman correlation between bucket rank and beta is printed for each
     scheme.
  2. It SURVIVES A HOLDOUT. The script picks the most promising stratum on
     2023 and 2024 only, then reports beta for that stratum on 2025 and 2026,
     which the choice never saw. That number is the only one to act on.

A CAUTION WORTH READING BEFORE GETTING EXCITED

The earlier subgroup work found within-tier correlation LOWER than pooled,
which cuts against this hypothesis. But that measured the model's correlation
with outcomes, not the model's edge over the line, and those two can move in
opposite directions. It is a caution, not an answer.

And the likelier mechanism if soft props do exist: books manage them with
LIMITS, not with sharp prices. A WR3 line may well be beatable and capped at
$100. That kills it for a syndicate and may be completely fine for you, which
would explain why the pocket stays soft instead of being arbitraged away.
This script cannot see limits. Only a real bet slip can.

Run from the repo root:
    python attention_strata.py --all-seasons --cache lines_cache.parquet
    python attention_strata.py --all-seasons --cache lines_cache.parquet --source consensus

Reuses eval_harness for credentials, fetching, the cache, the collapse and the
clustered estimators.
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh

BREAKEVEN = 113.0 / 213.0


# --------------------------------------------------------------------- data

def build_joined(args):
    """Load lines, score models walk-forward, join, KEEPING the attention
    columns that the harness discards."""
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
    props = eh.collapse_books(lines)
    print(f"  {len(props)} distinct player-week props")

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
    j = j.dropna(subset=[col, "projection", "actual", "n_books"])
    j["dev"] = j["projection"] - j[col]
    j["out"] = j["actual"] - j[col]
    j["line_used"] = j[col]
    return j


# ------------------------------------------------------------------- strata

def add_strata(j):
    """Four buckets per scheme, computed WITHIN market so the buckets mean the
    same thing across markets with different scales."""
    j = j.copy()

    # n_books is the most direct attention proxy, so use fixed edges rather
    # than quantiles: the categories are meaningful in themselves.
    j["s_books"] = pd.cut(j["n_books"], [0, 2, 4, 6, 8],
                          labels=["1-2 books", "3-4 books", "5-6 books",
                                  "7-8 books"])

    for name, col, labs in (
        ("s_spread", "book_spread", ["spread Q1 (tight)", "spread Q2",
                                     "spread Q3", "spread Q4 (wide)"]),
        ("s_level", "line_used", ["line Q1 (lowest)", "line Q2",
                                  "line Q3", "line Q4 (highest)"]),
    ):
        out = pd.Series(pd.NA, index=j.index, dtype="object")
        for m, g in j.groupby("market"):
            try:
                q = pd.qcut(g[col], 4, labels=labs, duplicates="drop")
                out.loc[g.index] = q.astype(object)
            except Exception:
                pass
        j[name] = out
    return j


def spearman(ranks, vals):
    ok = [(r, v) for r, v in zip(ranks, vals) if v is not None and np.isfinite(v)]
    if len(ok) < 3:
        return np.nan
    r = np.argsort(np.argsort([x[0] for x in ok]))
    v = np.argsort(np.argsort([x[1] for x in ok]))
    r, v = r - r.mean(), v - v.mean()
    d = np.sqrt((r @ r) * (v @ v))
    return float(r @ v / d) if d > 0 else np.nan


def report(j, scheme, label, min_rows=250):
    """Beta and over rate per bucket, per market, plus the trend."""
    print("\n" + "=" * 92)
    print(f"STRATIFIED BY {label}")
    print("=" * 92)
    order = [c for c in j[scheme].dropna().unique()]
    # keep the declared bucket order rather than first-seen order
    cats = j[scheme].dropna()
    if hasattr(cats, "cat"):
        order = [c for c in cats.cat.categories if c in set(cats)]
    else:
        order = sorted(set(cats), key=str)

    for m, g in j.groupby("market"):
        print(f"\n  {m}")
        print(f"  {'bucket':<20}{'n':>7}{'games':>7}{'beta':>8}{'SE':>7}"
              f"{'t':>7}{'over rate':>11}{'SE':>8}{'vs 0.5305':>11}"
              f"{'med line':>10}")
        betas, ranks = [], []
        for i, b in enumerate(order):
            sub = g[g[scheme] == b]
            if len(sub) < min_rows:
                if len(sub):
                    print(f"  {str(b):<20}{len(sub):>7}   too few")
                continue
            f = eh.ols_clustered(sub["dev"], sub["out"], sub["event_id"])
            won = (sub["actual"] > sub["line_used"]).astype(float)
            push = np.isclose(sub["actual"], sub["line_used"])
            cm = eh.cluster_mean(won[~push].to_numpy(),
                                 sub["event_id"].to_numpy()[~push])
            if f is None or cm is None:
                continue
            t = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
            print(f"  {str(b):<20}{f['n']:>7}{f['clusters']:>7}"
                  f"{f['beta']:>8.3f}{f['se_beta']:>7.3f}{t:>+7.2f}"
                  f"{cm['mean']:>11.4f}{cm['se']:>8.4f}"
                  f"{cm['mean'] - BREAKEVEN:>+11.4f}"
                  f"{sub['line_used'].median():>10.1f}")
            betas.append(f["beta"])
            ranks.append(i)
        if len(betas) >= 3:
            rho = spearman(ranks, betas)
            print(f"    trend: spearman(bucket order, beta) = {rho:+.2f}"
                  f"   {'MONOTONE' if abs(rho) > 0.79 else 'not monotone'}")
    print("\n    A monotone trend is a mechanism. A single high bucket with no")
    print("    trend is one of 48 cells and is high by construction.")


def holdout(j, schemes, min_rows=250):
    """Choose the most promising stratum on 2023-2024, score it on 2025-2026.

    This is the only number worth acting on. Everything above is in-sample
    with respect to which stratum got chosen.
    """
    print("\n" + "=" * 92)
    print("HOLDOUT: stratum chosen on 2023-2024, beta measured on 2025-2026")
    print("=" * 92)
    early = j[j["season"].isin([2023, 2024])]
    late = j[j["season"].isin([2025, 2026])]
    if len(early) < 1000 or len(late) < 500:
        print("  not enough rows in one of the halves")
        return

    best = None
    for scheme, label in schemes:
        for m, g in early.groupby("market"):
            for b in g[scheme].dropna().unique():
                sub = g[g[scheme] == b]
                if len(sub) < min_rows:
                    continue
                f = eh.ols_clustered(sub["dev"], sub["out"], sub["event_id"])
                if f is None:
                    continue
                if best is None or f["beta"] > best[0]:
                    best = (f["beta"], scheme, label, m, b, f["n"])
    if best is None:
        print("  no stratum had enough rows")
        return

    _, scheme, label, m, b, n_early = best
    print(f"  most promising on 2023-2024: {m}, {label} = {b}")
    print(f"    in-sample beta {best[0]:+.3f} on {n_early} rows "
          f"(SELECTED, biased upward)")

    sub = late[(late["market"] == m) & (late[scheme] == b)]
    if len(sub) < 100:
        print(f"    only {len(sub)} rows in 2025-2026, cannot confirm")
        return
    f = eh.ols_clustered(sub["dev"], sub["out"], sub["event_id"])
    if f is None:
        print("    fit failed on the holdout")
        return
    lo = f["beta"] - 1.96 * f["se_beta"]
    hi = f["beta"] + 1.96 * f["se_beta"]
    print(f"    HOLDOUT beta {f['beta']:+.3f} (SE {f['se_beta']:.3f}), "
          f"95% CI [{lo:+.3f}, {hi:+.3f}] on {f['n']} rows, "
          f"{f['clusters']} games")
    print()
    if lo > 0.20:
        print("    The soft-market hypothesis SURVIVES. This stratum beats the")
        print("    line by more than the pooled figure, out of sample.")
    elif lo > 0:
        print("    Weakly positive out of sample. Suggestive, underpowered.")
    else:
        print("    The holdout CI includes zero. The in-sample stratum was")
        print("    selection on noise, and the pooled number was not hiding")
        print("    a soft pocket.")


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

    print("=" * 92)
    print("ATTENTION STRATIFICATION: is the market soft where nobody looks?")
    print(f"  measuring against line_{args.source}")
    print("=" * 92)

    j = build_joined(args)
    j = add_strata(j)

    print(f"\n  attention spread in the data:")
    print(f"    n_books      min {int(j['n_books'].min())}, "
          f"median {int(j['n_books'].median())}, "
          f"max {int(j['n_books'].max())}")
    print(f"    book_spread  median {j['book_spread'].median():.2f}, "
          f"90th {j['book_spread'].quantile(0.90):.2f}")

    schemes = [("s_books", "NUMBER OF BOOKS QUOTING THE PROP"),
               ("s_spread", "DISAGREEMENT BETWEEN BOOKS"),
               ("s_level", "LINE LEVEL WITHIN MARKET")]
    for scheme, label in schemes:
        report(j, scheme, label, args.min_rows)

    holdout(j, schemes, args.min_rows)

    if args.save_rows:
        j.to_csv(args.save_rows, index=False)
        print(f"\n  wrote {args.save_rows}")

    print("\n" + "=" * 92)
    print("HOW TO READ THIS")
    print("=" * 92)
    print("  Look for a TREND, not a winner. Beta rising as the number of")
    print("  books falls is a mechanism. One high cell among 48 is noise.")
    print("  The HOLDOUT is the only number to act on.")
    print("  Even a positive result does not mean the bets are placeable.")
    print("  Books manage soft props with LIMITS rather than sharp prices, so")
    print("  a beatable WR3 line may be capped at a small stake. That is fine")
    print("  for you and fatal for a syndicate, which is the reason such a")
    print("  pocket could persist. Only a real bet slip can confirm it.")


if __name__ == "__main__":
    main()
