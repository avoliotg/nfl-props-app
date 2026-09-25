"""market_calibration.py - is the BOOK's price wrong anywhere?

A different question from everything else in this project. Every prior
measurement asked "does my model beat the line". This asks "is the devigged
price well calibrated", which is a much lower bar to clear and requires no
model at all.

WHY THE QUESTION CHANGED

    Receptions is 93 percent flat on the line, because FanDuel moves
    reception prices through the ODDS. So beta, which regresses actual minus
    line on projection minus line, spends both of its terms on the quantity
    that barely moves and discards the one carrying the market's actual
    information. The comparison "model versus market" has arguably never been
    run; what has been run is "model versus the grid the market rounds to".

    Devigging gives the market's own probability forecast. Once you have it,
    two things become measurable that were not before:

      1. WHICH devigging method matches reality. The project noted that
         methods differ by 1 to 2 points and never resolved which is right.
         Proper scoring rules resolve it empirically.

      2. WHERE the price is miscalibrated, if anywhere. Cut by line value, by
         integer versus half-integer, by how lopsided the price is, and by
         book. A pocket of laziness does not require out-forecasting sharp
         money, and there is already direct visual evidence of laziness:
         identical prices to the dollar for players in different roles on
         different teams at the bottom of the board.

WHAT WOULD CONSTITUTE A FINDING

    A bin where the realized rate differs from the devigged probability by
    more than the hold, with a cluster-robust interval excluding zero, and
    which survives being cut by season. Anything short of that is noise, and
    with this many bins some bins WILL look significant by chance. The script
    reports how many bins would be expected to deviate by luck alone, so the
    comparison is against that rather than against zero.

HONEST LIMITS

    Calibration is necessary, not sufficient. A perfectly calibrated price
    can still be beatable if you have information it lacks, and a
    miscalibrated price is only exploitable if the miscalibration exceeds the
    hold AND persists out of sample. Nothing here is a betting signal.

    Props within a game are correlated, so every interval is clustered on
    event_id. Binomial intervals would be too narrow and are not used.

Run from the repo root:
    python market_calibration.py --all-seasons --cache lines_cache.parquet
    python market_calibration.py --all-seasons --cache lines_cache.parquet --book fanduel
"""
import argparse
import sys

import numpy as np
import pandas as pd

import devig
import eval_harness as eh
from models import data_utils

# market -> the nflverse column holding the settled result.
# Taken straight from load_player_stats rather than any model module: every
# build_dataset applies filters (rushing gates carries >= 5) that condition on
# the outcome, and the market's calibration must be judged on the
# unconditional population.
ACTUAL_COL = {
    "receiving": "receiving_yards",
    "receptions": "receptions",
    "rushing": "rushing_yards",
    "qb_rushing": "rushing_yards",
    "qb_passing": "passing_yards",
}

# The three DISTINCT methods. shin is omitted because devig.py measures it as
# identical to additive for a two-outcome market, to within 1e-12 over 20,000
# price pairs. Including it would double count one assumption as two.
USE_METHODS = ("multiplicative", "additive", "power")

SEASONS_STATS = [2022, 2023, 2024, 2025, 2026]


def _rule(t):
    print()
    print("=" * 84)
    print(t)
    print("=" * 84)


def cluster_se_mean(values, clusters):
    """Cluster-robust SE of a mean. Props in one game are not independent.

    SE = sqrt( sum_g ( sum_{i in g} (x_i - xbar) )^2 ) / n
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return np.nan
    e = v - v.mean()
    df = pd.DataFrame({"e": e, "g": np.asarray(clusters)})
    per = df.groupby("g")["e"].sum().to_numpy()
    return float(np.sqrt((per ** 2).sum())) / n


def load_actuals():
    """(season, week, normalized name) -> every settled stat we need.

    Rows whose normalized name maps to more than one player_id are DROPPED,
    not guessed. name_resolve.py measured 21 such names in nflverse, the worst
    being `michael carter` at CB 51 rows against RB 47, which is close to a
    coin flip on a bettable player. Attaching the wrong outcome to a prop
    would corrupt the very thing being measured here.
    """
    ps = data_utils.load_player_stats(SEASONS_STATS)
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    cols = ["season", "week", "player_display_name"] + sorted(set(ACTUAL_COL.values()))
    have = [c for c in cols if c in ps.columns]
    missing = [c for c in cols if c not in ps.columns]
    if missing:
        print(f"  WARNING: load_player_stats has no {missing}; those markets "
              f"cannot be graded")
    ps = ps.dropna(subset=["player_display_name"]).copy()
    ps["_key"] = data_utils.norm_join_name(ps["player_display_name"])

    id_col = next((c for c in ("player_id", "gsis_id") if c in ps.columns), None)
    if id_col:
        nid = ps.groupby(["season", "week", "_key"])[id_col].transform("nunique")
        n_bad = int((nid > 1).sum())
        if n_bad:
            names = sorted(ps.loc[nid > 1, "_key"].unique())[:10]
            print(f"  dropped {n_bad} player-week rows whose normalized name "
                  f"maps to >1 player_id: {names}")
        ps = ps[nid == 1]

    # A player with a stat row was active, so a missing stat is a real zero.
    # Measured true for receptions (0 nulls in 22,540) and for QB rushing
    # (0 nulls in 2,817); applied here for consistency rather than effect.
    for c in set(ACTUAL_COL.values()):
        if c in ps.columns:
            ps[c] = ps[c].fillna(0.0)

    ps = ps.drop_duplicates(subset=["season", "week", "_key"], keep="first")
    return ps[have + ["_key"]]


def load_prices(args, seasons):
    _rule("LOADING PRICES")
    markets = sorted({eh.FETCH_MARKET.get(m, m) for m in ACTUAL_COL})
    sec = eh._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    lines = eh.load_lines(cf, seasons, markets, args.cache, args.refresh)
    if lines.empty:
        print("no lines found.")
        sys.exit(1)
    lines, rep = data_utils.split_qb_rushing(lines, min_match_rate=0.98)
    print(f"  split: {rep['reassigned']} rushing rows -> qb_rushing "
          f"({rep['reassigned_share']:.1%})")
    lines = lines[lines["week"].notna()].copy()
    lines["week"] = lines["week"].astype(int)
    print(f"  {len(lines):,} book-level price rows, "
          f"{lines['book'].nunique()} books, markets "
          f"{sorted(lines['market'].unique())}")
    if args.book:
        lines = lines[lines["book"] == args.book]
        print(f"  restricted to book={args.book}: {len(lines):,} rows")
    return lines


def devig_frame(lines):
    _rule("DEVIGGING")
    oo = lines["over_odds"].to_numpy()
    uo = lines["under_odds"].to_numpy()

    cols = {m: np.full(len(lines), np.nan) for m in USE_METHODS}
    raw = np.full(len(lines), np.nan)
    overround = np.full(len(lines), np.nan)
    reasons = {}

    for i in range(len(lines)):
        r = devig.devig_two_sided(oo[i], uo[i], "multiplicative")
        raw[i] = r["q_over"] if r["q_over"] is not None else np.nan
        overround[i] = r["overround"] if r["overround"] is not None else np.nan
        if not r["all_methods"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            continue
        for m in USE_METHODS:
            if m in r["all_methods"]:
                cols[m][i] = r["all_methods"][m]

    out = lines.copy()
    out["q_over_raw"] = raw
    out["overround"] = overround
    for m in USE_METHODS:
        out[f"p_{m}"] = cols[m]

    ok = out["p_multiplicative"].notna()
    print(f"  devigged {int(ok.sum()):,} of {len(out):,} rows "
          f"({ok.mean():.2%})")
    if reasons:
        print("  refusals:")
        for why, k in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {k:>7,}  {str(why)[:88]}")
    return out[ok].copy()


def attach_outcomes(priced, actuals):
    _rule("SETTLING")
    priced["_key"] = data_utils.norm_join_name(priced["player"])
    priced["actual"] = np.nan
    for market, col in ACTUAL_COL.items():
        if col not in actuals.columns:
            continue
        sel = priced["market"] == market
        if not sel.any():
            continue
        sub = priced.loc[sel, ["season", "week", "_key"]].merge(
            actuals[["season", "week", "_key", col]],
            on=["season", "week", "_key"], how="left")
        priced.loc[sel, "actual"] = sub[col].to_numpy()

    graded = priced["actual"].notna()
    print(f"  settled {int(graded.sum()):,} of {len(priced):,} priced rows "
          f"({graded.mean():.2%})")
    p = priced[graded].copy()

    p["push"] = p["actual"] == p["line"]
    n_push = int(p["push"].sum())
    print(f"  pushes excluded: {n_push:,} ({n_push / max(1, len(p)):.2%})")
    print(f"  integer lines: {int((p['line'] % 1 == 0).sum()):,}, "
          f"half-integer: {int((p['line'] % 1 != 0).sum()):,}")
    p = p[~p["push"]].copy()
    p["over"] = (p["actual"] > p["line"]).astype(int)
    print(f"  {len(p):,} settled non-push rows, overall over rate "
          f"{p['over'].mean():.4f}")
    return p


def section_hold(p):
    _rule("SECTION 1: THE HOLD, BY MARKET AND BOOK")
    print("Sanity check against the project's recorded figures: uniform odds")
    print("on yardage markets around -113/-114 with roughly 6.1 percent hold,")
    print("and asymmetric odds on receptions with roughly 6.9 percent.")
    print()
    p = p.copy()
    p["hold"] = p["overround"] / (1.0 + p["overround"])
    tab = p.groupby("market").agg(
        n=("hold", "size"), mean_hold=("hold", "mean"),
        median_hold=("hold", "median"), mean_overround=("overround", "mean"))
    print((tab * [1, 100, 100, 100]).round(2).to_string())
    print("  (hold and overround shown in percent)")
    if "fanduel" in set(p["book"]):
        print()
        print("  FanDuel only:")
        fd = p[p["book"] == "fanduel"].groupby("market")["hold"].agg(
            ["size", "mean", "median"])
        fd[["mean", "median"]] *= 100
        print(fd.round(2).to_string())


def score_methods(p):
    _rule("SECTION 2: WHICH DEVIGGING METHOD MATCHES REALITY?")
    print("Proper scoring rules, so this is decided by measurement rather than")
    print("by which method has the nicest story. Lower Brier and lower log")
    print("loss are better. 'bias' is mean(p) minus realized rate: positive")
    print("means the method says over too often.")
    print()
    print(f"  {'market':<13}{'method':<16}{'n':>8}{'mean p':>9}{'rate':>8}"
          f"{'bias':>9}{'clust SE':>10}{'t':>7}{'Brier':>9}{'logloss':>9}")
    best = {}
    for market, g in p.groupby("market"):
        rows = []
        for m in USE_METHODS:
            col = f"p_{m}"
            gg = g.dropna(subset=[col])
            if len(gg) < 200:
                continue
            pr = gg[col].to_numpy()
            y = gg["over"].to_numpy()
            bias = pr.mean() - y.mean()
            se = cluster_se_mean(pr - y, gg["event_id"])
            brier = float(np.mean((pr - y) ** 2))
            eps = 1e-12
            ll = float(-np.mean(y * np.log(np.clip(pr, eps, 1)) +
                                (1 - y) * np.log(np.clip(1 - pr, eps, 1))))
            t = bias / se if se and np.isfinite(se) and se > 0 else np.nan
            rows.append((m, len(gg), pr.mean(), y.mean(), bias, se, t, brier, ll))
            print(f"  {market:<13}{m:<16}{len(gg):>8,}{pr.mean():>9.4f}"
                  f"{y.mean():>8.4f}{bias:>+9.4f}{se:>10.4f}{t:>7.2f}"
                  f"{brier:>9.5f}{ll:>9.5f}")
        if rows:
            best[market] = min(rows, key=lambda r: r[8])[0]
        print()
    print("  lowest log loss by market:")
    for k, v in best.items():
        print(f"    {k:<13}{v}")
    print()
    print("  CAUTION. The methods differ by only 0.18 to 0.30 points on")
    print("  near-even props and by up to 2 points on lopsided ones, so on")
    print("  markets priced near even these scores will be nearly tied and")
    print("  the 'winner' is not meaningful. Treat a tie as a tie.")
    return best


def section_calibration(p, best):
    _rule("SECTION 3: IS THE PRICE CALIBRATED?")
    print("Devigged probability against realized rate, in bins, using each")
    print("market's best-scoring method. 'diff' is rate minus p, so POSITIVE")
    print("means the over hits more than the price implies. Intervals are")
    print("clustered on event_id.")
    edges = np.array([0, .2, .3, .4, .45, .5, .55, .6, .7, .8, 1.0])
    flagged = []
    for market, g in p.groupby("market"):
        m = best.get(market)
        if not m:
            continue
        col = f"p_{m}"
        g = g.dropna(subset=[col])
        print()
        print(f"  {market}  (method {m}, n={len(g):,})")
        print(f"    {'bin':<14}{'n':>8}{'mean p':>9}{'rate':>8}{'diff':>9}"
              f"{'clust SE':>10}{'t':>7}")
        g = g.assign(_bin=pd.cut(g[col], edges, include_lowest=True))
        for b, gg in g.groupby("_bin", observed=True):
            if len(gg) < 100:
                continue
            pr, y = gg[col].to_numpy(), gg["over"].to_numpy()
            diff = y.mean() - pr.mean()
            se = cluster_se_mean(y - pr, gg["event_id"])
            t = diff / se if se and np.isfinite(se) and se > 0 else np.nan
            star = "  <--" if abs(t) > 2.5 else ""
            print(f"    {str(b):<14}{len(gg):>8,}{pr.mean():>9.4f}"
                  f"{y.mean():>8.4f}{diff:>+9.4f}{se:>10.4f}{t:>7.2f}{star}")
            if abs(t) > 2.5:
                flagged.append((market, str(b), len(gg), diff, t))
    return flagged


def section_cuts(p, best):
    _rule("SECTION 4: WHERE THE PRICE MIGHT BE LAZY")
    print("Integer versus half-integer lines, and low versus high line values.")
    print("The template-pricing observation lives at the BOTTOM of the board,")
    print("so the low-line rows are the ones to watch.")
    for market, g in p.groupby("market"):
        m = best.get(market)
        if not m:
            continue
        col = f"p_{m}"
        g = g.dropna(subset=[col]).copy()
        print()
        print(f"  {market}  (method {m})")
        g["_int"] = np.where(g["line"] % 1 == 0, "integer", "half-integer")
        print(f"    {'cut':<22}{'n':>8}{'mean p':>9}{'rate':>8}{'diff':>9}"
              f"{'clust SE':>10}{'t':>7}")
        for label, gg in g.groupby("_int"):
            if len(gg) < 100:
                continue
            _emit(label, gg, col)
        try:
            g["_q"] = pd.qcut(g["line"], 4, duplicates="drop")
        except ValueError:
            continue
        for label, gg in g.groupby("_q", observed=True):
            if len(gg) < 100:
                continue
            _emit(f"line {label}", gg, col)


def _emit(label, gg, col):
    pr, y = gg[col].to_numpy(), gg["over"].to_numpy()
    diff = y.mean() - pr.mean()
    se = cluster_se_mean(y - pr, gg["event_id"])
    t = diff / se if se and np.isfinite(se) and se > 0 else np.nan
    star = "  <--" if abs(t) > 2.5 else ""
    print(f"    {str(label):<22}{len(gg):>8,}{pr.mean():>9.4f}{y.mean():>8.4f}"
          f"{diff:>+9.4f}{se:>10.4f}{t:>7.2f}{star}")


def section_multiplicity(p, best, flagged):
    _rule("SECTION 5: IS ANY OF THAT REAL?")
    n_tests = 0
    for market, g in p.groupby("market"):
        if best.get(market):
            n_tests += 10
    exp_false = n_tests * 0.0124  # two-sided |t| > 2.5 under the null
    print(f"  bins tested (approx)        {n_tests}")
    print(f"  expected |t| > 2.5 by luck  {exp_false:.1f}")
    print(f"  actually flagged            {len(flagged)}")
    print()
    if not flagged:
        print("  Nothing flagged. The devigged price is calibrated everywhere")
        print("  measured, which closes the 'is the price wrong' question in")
        print("  the negative and is a clean result rather than a null one.")
        return
    if len(flagged) <= exp_false:
        print("  Fewer flags than luck alone predicts. Treat as nothing.")
        return
    print("  More flags than chance predicts. Each still needs to survive a")
    print("  per-season split before it means anything, because the 2023")
    print("  anomaly has already produced pooled results that were one season")
    print("  in disguise. Flagged:")
    for market, b, n, diff, t in sorted(flagged, key=lambda r: -abs(r[4])):
        print(f"    {market:<13}{b:<14}n={n:<7,}diff {diff:+.4f}  t {t:+.2f}")
    print()
    print("  AND the bar is not zero. To be exploitable the deviation must")
    print("  exceed the hold on that market, which section 1 measured. A")
    print("  4-point miscalibration against a 6-point hold is still a loss.")


def section_by_season(p, best, flagged):
    if not flagged:
        return
    _rule("SECTION 6: DO THE FLAGS SURVIVE A SEASON SPLIT?")
    for market, b, n, diff, t in sorted(flagged, key=lambda r: -abs(r[4]))[:6]:
        m = best.get(market)
        col = f"p_{m}"
        g = p[(p["market"] == market)].dropna(subset=[col]).copy()
        lo, hi = _parse_bin(b)
        g = g[(g[col] > lo) & (g[col] <= hi)]
        print()
        print(f"  {market} {b}  (pooled diff {diff:+.4f}, t {t:+.2f})")
        print(f"    {'season':<10}{'n':>8}{'mean p':>9}{'rate':>8}{'diff':>9}"
              f"{'clust SE':>10}{'t':>7}")
        for season, gg in g.groupby("season"):
            if len(gg) < 50:
                continue
            _emit(season, gg, col)
        print("    A flag driven by ONE season is the 2023 pattern again.")


def _parse_bin(b):
    s = b.strip("()[]").split(",")
    return float(s[0]), float(s[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default=None,
                    help="restrict to one book, e.g. fanduel")
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 84)
    print("MARKET CALIBRATION: is the devigged price wrong anywhere?")
    print(f"  seasons {seasons}   book {args.book or 'all'}")
    print("=" * 84)

    lines = load_prices(args, seasons)
    priced = devig_frame(lines)
    actuals = load_actuals()
    p = attach_outcomes(priced, actuals)
    if len(p) < 1000:
        print("too few settled rows to measure calibration.")
        sys.exit(1)

    section_hold(p)
    best = score_methods(p)
    flagged = section_calibration(p, best)
    section_cuts(p, best)
    section_multiplicity(p, best, flagged)
    section_by_season(p, best, flagged)

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
