"""under_bias.py - real-odds ROI on the under side, and does the model add to it?

Reports only. Writes nothing, ships no constant, places no bet.

WHY THIS EXISTS

    market_calibration.py found that FanDuel's devigged price says OVER more
    often than the over actually hits, in every market except qb_passing. The
    harness had already measured the same thing and recorded it as "no
    model-free edge", because its 'vs breakeven' column only ever tested the
    OVER side. A deficit on the over is a surplus on the under, and nobody
    flipped it.

    Two corrections make the flip material:

      1. THE HOLD IS LOWER THAN RECORDED. FanDuel's measured median hold is
         4.55 percent on yardage markets and 5.73 on receptions. The project
         recorded 6.1 and 6.9. The 6.1 figure is the OVERROUND at -113, not
         the hold, and FanDuel is not pricing yardage at -113 anyway: a 4.55
         percent hold is -110. That moves the yardage breakeven from 0.5305
         to 0.5238, which is 0.67 points.

      2. RECEPTIONS ODDS ARE ASYMMETRIC, with a breakeven range of roughly
         0.417 to 0.610 per the project's own notes. So NO single breakeven
         is correct there, and the back-of-envelope ROI figures computed from
         a symmetric assumption are wrong for that market specifically.

    Hence this script: ROI from the ACTUAL posted odds on every row, no
    assumed price anywhere.

WHAT WOULD MAKE IT REAL

    Positive ROI whose cluster bootstrap interval excludes zero, that holds
    its SIGN across all seasons, and that survives being one of N buckets
    tested. The rushing low-line result already passed the season test in
    market_calibration (negative diff in all four seasons). This checks
    whether it pays after the real vig.

WHAT WOULD KILL IT

    An interval straddling zero, a sign flip in any season, or an edge that
    exists only where the posted odds are unusually short. The last one
    matters: an apparent edge concentrated on rows priced -140 is the book
    telling you something rather than the book being wrong.

THE CONJUNCTION TEST

    Section 5 asks the question the project's strategy actually rests on: in
    a bucket where the mechanical under pays, does it pay MORE when the
    model's projection also sits below the line? If yes, model and judgment
    compound and the screen has a measurable basis. If the two are
    independent, the model adds nothing to the pricing bias and the bias is
    the whole story.

    Note what this does NOT test. It cannot measure a human read, because a
    human read is not in the data. It only measures the model half of the
    conjunction.

Run from the repo root:
    python under_bias.py --all-seasons --cache lines_cache.parquet
    python under_bias.py --all-seasons --cache lines_cache.parquet --book all
    python under_bias.py --all-seasons --cache lines_cache.parquet --boot 2000
"""
import argparse
import sys

import numpy as np
import pandas as pd

import devig
import eval_harness as eh
from models import data_utils

ACTUAL_COL = {
    "receiving": "receiving_yards",
    "receptions": "receptions",
    "rushing": "rushing_yards",
    "qb_rushing": "rushing_yards",
    "qb_passing": "passing_yards",
}
SEASONS_STATS = [2022, 2023, 2024, 2025, 2026]

# From market_calibration.py, FanDuel only, all seasons. The gate aborts
# unless this script reproduces them, because it rebuilds the same settlement
# pipeline and a settlement bug would invent an edge out of nothing.
PUBLISHED_OVER_RATE = {
    "qb_passing": (0.5048, 1684),
    "qb_rushing": (0.4874, 1389),
    "receiving": (0.4903, 8771),
    "receptions": (0.4720, 7930),
    "rushing": (0.4665, 2819),
}
RATE_TOL = 0.003


def _rule(t):
    print()
    print("=" * 86)
    print(t)
    print("=" * 86)


def boot_ci(profit, clusters, n_boot, seed=0):
    """Cluster bootstrap of mean ROI, resampling GAMES not bets.

    Props inside one game share weather, pace, game script and blowout risk,
    so resampling rows would understate the interval badly.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"p": np.asarray(profit, float),
                       "g": np.asarray(clusters)})
    groups = [g["p"].to_numpy() for _, g in df.groupby("g")]
    if len(groups) < 10:
        return np.nan, np.nan, np.nan
    idx = np.arange(len(groups))
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        out[b] = np.concatenate([groups[i] for i in pick]).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return lo, hi, float(out.std())


def _boot_diff(g, flag_col, roi_col, n_boot, seed=0):
    """Cluster bootstrap of ROI(flag=True) minus ROI(flag=False).

    Resamples GAMES and recomputes both subsets inside each replicate, so the
    interval accounts for the fact that a game contributes rows to both
    sides and that the split itself is random.
    """
    rng = np.random.default_rng(seed)
    groups = [d for _, d in g.groupby("event_id")]
    if len(groups) < 20:
        return None
    idx = np.arange(len(groups))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        rep = pd.concat([groups[i] for i in pick], ignore_index=True)
        a = rep.loc[rep[flag_col], roi_col]
        b = rep.loc[~rep[flag_col], roi_col]
        if len(a) < 20 or len(b) < 20:
            continue
        out.append(a.mean() - b.mean())
    return np.array(out)


def load_actuals():
    ps = data_utils.load_player_stats(SEASONS_STATS)
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    cols = ["season", "week", "player_display_name"] + sorted(set(ACTUAL_COL.values()))
    have = [c for c in cols if c in ps.columns]
    ps = ps.dropna(subset=["player_display_name"]).copy()
    ps["_key"] = data_utils.norm_join_name(ps["player_display_name"])
    idc = next((c for c in ("player_id", "gsis_id") if c in ps.columns), None)
    if idc:
        nid = ps.groupby(["season", "week", "_key"])[idc].transform("nunique")
        n_bad = int((nid > 1).sum())
        if n_bad:
            print(f"  dropped {n_bad} rows on normalized-name collisions")
        ps = ps[nid == 1]
    for c in set(ACTUAL_COL.values()):
        if c in ps.columns:
            ps[c] = ps[c].fillna(0.0)
    return ps.drop_duplicates(subset=["season", "week", "_key"])[have + ["_key"]]


def build(args, seasons):
    _rule("LOADING AND SETTLING")
    markets = sorted({eh.FETCH_MARKET.get(m, m) for m in ACTUAL_COL})
    sec = eh._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    lines = eh.load_lines(cf, seasons, markets, args.cache, args.refresh)
    lines, rep = data_utils.split_qb_rushing(lines, min_match_rate=0.98)
    lines = lines[lines["week"].notna()].copy()
    lines["week"] = lines["week"].astype(int)
    if args.book != "all":
        lines = lines[lines["book"] == args.book]
    print(f"  {len(lines):,} rows, book={args.book}")

    # One row per prop per book. If the cache ever carries several snapshots
    # of the same prop, counting each as a bet would multiply a single
    # decision into many and shrink every interval spuriously.
    keys = ["season", "week", "market", "player", "book"]
    before = len(lines)
    if "captured_at" in lines.columns:
        lines = lines.sort_values("captured_at").drop_duplicates(keys, keep="last")
    else:
        lines = lines.drop_duplicates(keys, keep="last")
    if len(lines) != before:
        print(f"  deduped {before - len(lines):,} repeat snapshots of the "
              f"same prop")

    # real payouts, no assumed price anywhere
    lines["dec_over"] = [devig.american_to_decimal(o) for o in lines["over_odds"]]
    lines["dec_under"] = [devig.american_to_decimal(o) for o in lines["under_odds"]]
    ok = lines["dec_over"].notna() & lines["dec_under"].notna()
    print(f"  {int(ok.sum()):,} rows with both prices usable "
          f"({ok.mean():.2%})")
    lines = lines[ok].copy()

    actuals = load_actuals()
    lines["_key"] = data_utils.norm_join_name(lines["player"])
    lines["actual"] = np.nan
    for market, col in ACTUAL_COL.items():
        if col not in actuals.columns:
            continue
        sel = lines["market"] == market
        if not sel.any():
            continue
        sub = lines.loc[sel, ["season", "week", "_key"]].merge(
            actuals[["season", "week", "_key", col]],
            on=["season", "week", "_key"], how="left")
        lines.loc[sel, "actual"] = sub[col].to_numpy()

    p = lines[lines["actual"].notna()].copy()
    print(f"  settled {len(p):,} rows")

    p["push"] = p["actual"] == p["line"]
    n_push = int(p["push"].sum())
    p = p[~p["push"]].copy()
    p["over"] = (p["actual"] > p["line"]).astype(int)
    # profit per unit staked, from the posted price
    p["roi_under"] = np.where(p["over"] == 0, p["dec_under"] - 1.0, -1.0)
    p["roi_over"] = np.where(p["over"] == 1, p["dec_over"] - 1.0, -1.0)
    ov, hold = zip(*[devig.hold_from_odds(a, b)
                     for a, b in zip(p["over_odds"], p["under_odds"])])
    p["hold"] = hold
    print(f"  {n_push:,} pushes excluded, {len(p):,} decided bets, "
          f"over rate {p['over'].mean():.4f}")
    return p


def gate(p, args):
    _rule("GATE: reproduce market_calibration's settlement")
    if args.book != "fanduel":
        print(f"  book={args.book}, published figures are FanDuel only. "
              f"Gate skipped, and every number below is therefore ungated.")
        return
    ok = True
    print(f"  {'market':<14}{'n':>7}{'pub n':>7}{'over':>9}{'pub over':>10}"
          f"{'diff':>9}")
    for market, (want_rate, want_n) in PUBLISHED_OVER_RATE.items():
        g = p[p["market"] == market]
        if not len(g):
            print(f"  {market:<14} MISSING")
            ok = False
            continue
        rate = g["over"].mean()
        d = rate - want_rate
        print(f"  {market:<14}{len(g):>7,}{want_n:>7,}{rate:>9.4f}"
              f"{want_rate:>10.4f}{d:>+9.4f}")
        if abs(d) > RATE_TOL:
            ok = False
    if not ok:
        print()
        print("  ABORT: settlement does not match market_calibration, so any")
        print("  ROI below is computed on a different population than the")
        print("  finding it claims to test.")
        sys.exit(1)
    print("\n  GATE PASSED.")


def emit(label, g, col, n_boot, seed=0):
    prof = g[col].to_numpy()
    roi = prof.mean()
    lo, hi, sd = boot_ci(prof, g["event_id"], n_boot, seed)
    star = "  <--" if np.isfinite(lo) and lo > 0 else ""
    print(f"  {label:<26}{len(g):>7,}{g['over'].mean():>8.4f}"
          f"{roi:>+9.4f}{sd:>9.4f}{lo:>+9.4f}{hi:>+9.4f}{star}")
    return roi, lo, hi


def head():
    print(f"  {'':<26}{'bets':>7}{'over':>8}{'ROI':>9}{'bootSD':>9}"
          f"{'lo95':>9}{'hi95':>9}")


def section_market(p, n_boot):
    _rule("SECTION 1: MECHANICAL UNDER, REAL POSTED ODDS")
    print("Bet the under on every prop. No model, no selection, no judgement.")
    print("ROI is profit per unit staked using the ACTUAL under price.")
    print()
    head()
    res = {}
    for market, g in p.groupby("market"):
        res[market] = emit(market, g, "roi_under", n_boot)
    print()
    print("  and the over side, which should be roughly the mirror:")
    head()
    for market, g in p.groupby("market"):
        emit(market, g, "roi_over", n_boot)
    print()
    print("  mean hold by market (from the posted prices):")
    for market, g in p.groupby("market"):
        print(f"    {market:<14}{g['hold'].mean()*100:>6.2f}%  "
              f"median {g['hold'].median()*100:.2f}%")
    return res


def section_buckets(p, n_boot):
    _rule("SECTION 2: BY LINE BUCKET (quartiles within each market)")
    print("The template-pricing observation lives at the bottom of the board,")
    print("so the low-line buckets are the ones to watch.")
    flagged, all_buckets = [], []
    for market, g in p.groupby("market"):
        g = g.copy()
        try:
            g["_q"] = pd.qcut(g["line"], 4, duplicates="drop")
        except ValueError:
            continue
        print()
        print(f"  {market}")
        head()
        for b, gg in g.groupby("_q", observed=True):
            if len(gg) < 150:
                continue
            roi, lo, hi = emit(f"line {b}", gg, "roi_under", n_boot)
            rec = (market, str(b), len(gg), roi, lo, hi)
            all_buckets.append(rec)
            if np.isfinite(lo) and lo > 0:
                flagged.append(rec)
    return flagged, all_buckets


def pick_candidates(flagged, all_buckets, n_max=5):
    """Buckets worth carrying into the later sections.

    Sections 3 to 5 originally ran only on buckets whose bootstrap lower
    bound cleared zero. On the real data nothing cleared, so the conjunction
    test never ran and the question the project most cares about went
    unanswered. That is the wrong failure mode: "does the model add anything"
    is worth asking of a near miss, and a near miss is exactly where a second
    signal could matter.

    So the later sections now run on every bucket with POSITIVE ROI, capped
    at n_max, and those are labelled NEAR MISSES throughout. They are not
    findings. A best-of-20 bucket that fails a one-sided bound is not
    evidence, and section 6 keeps saying so.
    """
    if flagged:
        return flagged, "CLEARED the bootstrap bound"
    pos = [b for b in all_buckets if b[3] > 0]
    pos.sort(key=lambda r: -r[3])
    return pos[:n_max], ("NEAR MISSES, positive ROI but the bound includes "
                         "zero. NOT findings.")


def section_seasons(p, flagged, n_boot):
    if not flagged:
        return []
    _rule("SECTION 3: DOES EACH FLAG HOLD ITS SIGN EVERY SEASON?")
    print("A pooled edge that is one season in disguise has already happened")
    print("twice in this project. Sign consistency is the bar, not")
    print("significance in each season, since single seasons are small.")
    survivors = []
    for market, b, n, roi, lo, hi in flagged:
        g = p[p["market"] == market].copy()
        lo_e, hi_e = _parse_bin(b)
        g = g[(g["line"] > lo_e) & (g["line"] <= hi_e)]
        print()
        print(f"  {market} line {b}  (pooled ROI {roi:+.4f}, "
              f"95% [{lo:+.4f}, {hi:+.4f}])")
        head()
        signs = []
        for season, gg in g.groupby("season"):
            if len(gg) < 50:
                continue
            r, _, _ = emit(str(season), gg, "roi_under", max(200, n_boot // 4))
            signs.append(np.sign(r))
        if signs and all(s > 0 for s in signs):
            print("    SIGN CONSISTENT across every season with enough bets.")
            survivors.append((market, b, n, roi, lo, hi))
        else:
            print("    SIGN FLIPS. Treat the pooled figure as unreliable.")
    return survivors


def section_price(p, survivors, n_boot):
    if not survivors:
        return
    _rule("SECTION 4: IS THE EDGE JUST SHORT-PRICED ROWS?")
    print("If an apparent edge sits only where the under is priced unusually")
    print("SHORT, the book is telling you something rather than being wrong.")
    print("A real pricing bias should persist at ordinary prices.")
    for market, b, n, roi, lo, hi in survivors:
        g = p[p["market"] == market].copy()
        lo_e, hi_e = _parse_bin(b)
        g = g[(g["line"] > lo_e) & (g["line"] <= hi_e)].copy()
        g["_p"] = pd.cut(g["dec_under"], [0, 1.75, 1.87, 1.95, 99],
                         labels=["shorter than -133", "-133 to -115",
                                 "-115 to -105", "longer than -105"])
        print()
        print(f"  {market} line {b}")
        head()
        for lab, gg in g.groupby("_p", observed=True):
            if len(gg) < 80:
                continue
            emit(str(lab), gg, "roi_under", max(200, n_boot // 4))


def section_conjunction(p, survivors, seasons, n_boot):
    _rule("SECTION 5: DOES THE MODEL ADD TO THE PRICING BIAS?")
    print("The strategy is not 'bet every under'. It is 'bet where the model")
    print("and the read agree'. The read is not in the data, but the MODEL")
    print("half is testable: inside a bucket where the mechanical under pays,")
    print("does it pay MORE when the projection also sits below the line?")
    print()
    print("If the two are independent, the bias is the whole story and the")
    print("model contributes nothing here. If they compound, the screen has")
    print("a measurable basis.")

    if not survivors:
        print("\n  No bucket to test, not even a near miss with positive ROI.")
        return

    markets = sorted({m for m, *_ in survivors})
    proj = []
    for market in markets:
        try:
            s = eh.score_market(market, seasons, mode="walk_forward",
                                population="all")
        except Exception as e:
            print(f"\n  could not score {market}: {type(e).__name__}: {e}")
            continue
        if len(s):
            proj.append(s)
    if not proj:
        print("\n  no projections available.")
        return
    proj = pd.concat(proj, ignore_index=True)
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    proj["week"] = proj["week"].astype(int)
    proj = proj.drop_duplicates(subset=["season", "week", "market", "_key"])

    j = p.merge(proj[["season", "week", "market", "_key", "projection"]],
                on=["season", "week", "market", "_key"], how="inner")
    print(f"\n  {len(j):,} of {len(p):,} bets have a walk-forward projection")

    for market, b, n, roi, lo, hi in survivors:
        g = j[j["market"] == market].copy()
        lo_e, hi_e = _parse_bin(b)
        g = g[(g["line"] > lo_e) & (g["line"] <= hi_e)].copy()
        if len(g) < 150:
            continue
        g["model_says_under"] = g["projection"] < g["line"]
        print()
        print(f"  {market} line {b}   ({len(g):,} bets with a projection)")
        head()
        agree = g[g["model_says_under"]]
        dis = g[~g["model_says_under"]]
        if len(agree) >= 80:
            emit("model AGREES (proj<line)", agree, "roi_under",
                 max(200, n_boot // 4))
        if len(dis) >= 80:
            emit("model disagrees", dis, "roi_under", max(200, n_boot // 4))

        # The DIFFERENCE is the quantity of interest, and eyeballing two
        # overlapping intervals is not a test of it. Resample games and
        # recompute both sides inside each replicate.
        if len(agree) >= 80 and len(dis) >= 80:
            d = _boot_diff(g, "model_says_under", "roi_under",
                           max(400, n_boot // 2))
            if d is not None and len(d):
                lo2, hi2 = np.percentile(d, [2.5, 97.5])
                print(f"    agree minus disagree: "
                      f"{agree['roi_under'].mean() - dis['roi_under'].mean():+.4f}"
                      f"   95% [{lo2:+.4f}, {hi2:+.4f}]"
                      f"   share>0 {float((d > 0).mean()):.3f}")
                if lo2 > 0:
                    print("    THE MODEL ADDS. Agreement raises ROI beyond "
                          "sampling noise.")
                else:
                    print("    interval includes zero: on this bucket the")
                    print("    model and the pricing bias look independent,")
                    print("    so the bias is the whole story here.")

        print()
        print("    taking the model's OWN side rather than always the under:")
        head()
        if len(agree) >= 80:
            emit("under where proj<line", agree, "roi_under",
                 max(200, n_boot // 4))
        if len(dis) >= 80:
            emit("over where proj>line", dis, "roi_over",
                 max(200, n_boot // 4))


def section_honesty(p, flagged, survivors, all_buckets=None):
    _rule("SECTION 6: HOW MUCH OF THIS IS LUCK?")
    n_tests = sum(1 for _ in p.groupby("market")) * 4
    print(f"  buckets tested            {n_tests}")
    print(f"  expected 1-sided hits     {n_tests * 0.025:.1f}")
    print(f"  flagged (lo95 > 0)        {len(flagged)}")
    if all_buckets:
        best = max(all_buckets, key=lambda r: r[3])
        print(f"  best bucket               {best[0]} {best[1]}  "
              f"ROI {best[3]:+.4f}  lo95 {best[4]:+.4f}")
        if best[4] <= 0:
            print("  NOTE: the best of many buckets failing a one-sided bound")
            print("  is not evidence. Waiting for it to cross with another")
            print("  season of data is the threshold-chasing plan 5.1 already")
            print("  taught this project to distrust.")
    print(f"  survived the season test  {len(survivors)}")
    print()
    print("  A bucket that clears zero AND holds its sign every season is")
    print("  worth a paper-trade. It is not worth real money until it has")
    print("  been forward-tested, because every number here is in-sample with")
    print("  respect to the choice of bucket.")
    print()
    print("  PRACTICAL LIMITS, which no statistic here measures:")
    print("    low-line props are where FanDuel's limits are smallest and")
    print("    lines go stalest. A 7 percent edge you can stake 20 dollars on")
    print("    is a different proposition from one you can stake 500 on. That")
    print("    has to be checked by hand in the app, not here.")
    print()
    print("  AND recreational over-bias in player props is well documented in")
    print("  the literature. That supports the result being real rather than")
    print("  noise, and it also means FanDuel may already limit exactly these")
    print("  props.")


def _parse_bin(b):
    s = b.strip("()[]").split(",")
    return float(s[0]), float(s[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 86)
    print("UNDER-SIDE BIAS: real posted odds, cluster bootstrap")
    print(f"  seasons {seasons}   book {args.book}   boot {args.boot}")
    print("=" * 86)

    p = build(args, seasons)
    if len(p) < 1000:
        print("too few settled bets.")
        sys.exit(1)
    gate(p, args)
    section_market(p, args.boot)
    flagged, all_buckets = section_buckets(p, args.boot)
    candidates, why = pick_candidates(flagged, all_buckets)
    print()
    print(f"  carrying {len(candidates)} bucket(s) into sections 3 to 5: {why}")
    survivors = section_seasons(p, candidates, args.boot)
    if not survivors and candidates:
        print()
        print("  no bucket held its sign every season. Sections 4 and 5 will")
        print("  still run on the candidates, because 'does the model add'")
        print("  is worth answering even where the mechanical rule fails.")
        survivors = candidates
    section_price(p, survivors, args.boot)
    section_conjunction(p, survivors, seasons, args.boot)
    section_honesty(p, flagged, survivors, all_buckets)

    _rule("DONE. Nothing was written. No bet was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
