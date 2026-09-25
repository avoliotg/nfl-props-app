"""receptions_shipped_vs_price.py - does the SHIPPED price beat the market's?

Reports only. Writes nothing, changes no constant, places no bet.

WHY THIS IS THE DECISIVE TEST

    Receptions is the ONLY market where the app still displays Edge, Side and
    Tier. That verdict rests on beta 0.2261, measured with a game-clustered
    t of +6.5 and confirmed by four independent routes. The beta is real.

    But it is measured against the LINE, and receptions is 93 percent flat on
    the line because FanDuel moves reception prices through the ODDS. So the
    app's one remaining verdict is benchmarked against the stale quantity
    while the bet is settled against the sharp one.

    receptions_classifier.py already tested this and found nothing: every
    design was worse than the devigged market on out-of-sample log loss, and
    the encompassing t values ran 0.69 to 1.45. But that test used a LOGISTIC
    on LEAN_FEATS, which is a proxy for the pricing path rather than the path
    itself.

    This script closes that gap. It calls mc_pricing.p_over directly, which
    is the shipped function: blended mean via BLEND alpha 0.1491 and beta
    0.2261, negative binomial, calibrated sigma, zero gate. Whatever it says
    is a statement about the probability the app actually displays.

THE ASYMMETRY IN WHAT A RESULT MEANS, AND IT MATTERS

    The projections are walk-forward, from eval_harness.score_market, so no
    season is scored by a model trained on it.

    The PRICING CONSTANTS are not. BLEND's alpha and beta were fitted on
    2023-2026, so they are in-sample for every row here, and mc_pricing's
    sigma was calibrated on the same data. That biases this test IN FAVOUR
    of the shipped price.

    Therefore:

      a NULL is strong evidence. The price cannot beat the market even with
      its blend constants fitted on the very rows being scored.

      a POSITIVE is weak evidence and needs a walk-forward refit of the blend
      before anything is concluded from it.

    Stating that before running it, so the result cannot be read the
    convenient way afterwards.

TWO TESTS, ANSWERING DIFFERENT QUESTIONS

    LOG LOSS      is the shipped price better calibrated than the market's?
                  A proper scoring rule, no betting rule involved.

    ENCOMPASSING  does the shipped price carry information the market's does
                  NOT? Outcome regressed on book_p and on (p_shipped minus
                  book_p), clustered on game. This is the question betting
                  cares about, because a positive coefficient means combining
                  beats the price alone even if the shipped price is worse
                  overall.

    Measured power at this sample size, from receptions_classifier.py's
    synthetic checks (860 games, 9 props each):

        test                        model better   model worse
        encompassing coefficient      t +7.45        t +4.01
        log loss delta                t -3.21        t +4.92
        simple slope on disagree      t +0.62        t +0.40

    The simple slope is NOT used here. It has no power once book_p varies,
    and receptions' book_p has a standard deviation of 0.067 across 299
    distinct values, so it varies a great deal.

WHAT HAPPENS TO THE APP EITHER WAY

    If this is null, receptions should join the reference-only set and the app
    would then display no model verdict in ANY market, leaving only the two
    rules in rules.py. That is a big change to the product and it should be
    made on this test rather than on the proxy.

Run from the repo root:
    python receptions_shipped_vs_price.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd

import devig
import eval_harness as eh
import mc_pricing
import recency_test as rt
from models import data_utils

MARKET = "receptions"
DEVIG_METHOD = "additive"     # equals shin for a two-outcome market
EPS = 1e-12

# From market_calibration.py, FanDuel receptions, all seasons.
PUB_OVER_RATE = 0.4720
PUB_BOOK_LOGLOSS = 0.68371
RATE_TOL = 0.004


def _rule(t):
    print()
    print("=" * 92)
    print(t)
    print("=" * 92)


def logloss_vec(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_delta(a, b, clusters, n_boot, seed=0):
    """Cluster bootstrap of mean(a) - mean(b), paired on the same rows."""
    d = np.asarray(a, float) - np.asarray(b, float)
    agg = pd.DataFrame({"d": d, "g": np.asarray(clusters)}).groupby("g")["d"]
    agg = agg.agg(["sum", "size"])
    if len(agg) < 15:
        return np.nan, np.nan
    s, n = agg["sum"].to_numpy(), agg["size"].to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    return tuple(np.percentile(s[idx].sum(axis=1) / n[idx].sum(axis=1),
                               [2.5, 97.5]))


def load(args, seasons):
    _rule("LOADING")
    sec = eh._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    lines = eh.load_lines(cf, seasons, [MARKET], args.cache, args.refresh)
    lines = lines[(lines["book"] == args.book) & lines["week"].notna()].copy()
    lines["week"] = lines["week"].astype(int)
    k = ["season", "week", "market", "player", "book"]
    if "captured_at" in lines.columns:
        lines = lines.sort_values("captured_at").drop_duplicates(k, keep="last")
    lines["_key"] = data_utils.norm_join_name(lines["player"])

    po, do, du = [], [], []
    for oo, uo in zip(lines["over_odds"], lines["under_odds"]):
        r = devig.devig_two_sided(oo, uo, DEVIG_METHOD)
        po.append(r["p_over"] if r["valid"] else np.nan)
        do.append(devig.american_to_decimal(oo))
        du.append(devig.american_to_decimal(uo))
    lines["book_p"] = po
    lines["dec_over"] = do
    lines["dec_under"] = du
    lines = lines.dropna(subset=["book_p", "dec_over", "dec_under"]).copy()
    print(f"  {len(lines):,} {args.book} {MARKET} rows devigged")

    proj = eh.score_market(MARKET, seasons, mode="walk_forward",
                           population="all")
    proj["_key"] = data_utils.norm_join_name(proj["player"])
    proj["week"] = proj["week"].astype(int)
    proj = proj.drop_duplicates(subset=["season", "week", "_key"])
    print(f"  {len(proj):,} walk-forward projections, seasons "
          f"{sorted(proj['season'].unique().tolist())}")

    g = lines.merge(proj[["season", "week", "_key", "projection", "actual"]],
                    on=["season", "week", "_key"], how="inner")
    g = g[g["actual"] != g["line"]].copy()
    g["over"] = (g["actual"] > g["line"]).astype(int)
    print(f"  {len(g):,} settled props with a walk-forward projection")
    print(f"  realized over rate {g['over'].mean():.4f}   "
          f"published {PUB_OVER_RATE:.4f} (on a larger population)")
    return g


def price_shipped(g):
    _rule("SECTION 1: THE SHIPPED PRICE")
    print("CORRECTED 2026-09-25. The first version of this script passed the")
    print("RAW PROJECTION to mc_pricing.p_over. That function's argument is")
    print("the DISTRIBUTION MEAN, and blended_mean's own docstring says so in")
    print("as many words: 'This is the distribution mean. Pass the result to")
    print("p_over, not the raw projection.'")
    print()
    print("So the first run measured the PRE-PHASE-4.4 behaviour, which is")
    print("the exact defect 4.4 fixed, and reported a null about a code path")
    print("the app does not use. Its dispersion was the giveaway: unblended,")
    print("the mean deviates from the line about 4.4x too far, which is")
    print("1/0.2261.")
    print()
    print("The correct sequence is blended_mean -> p_over, and where mc.py")
    print("exposes a wrapper that is preferred, because that is literally")
    print("what app.py calls.")
    print()

    # THE EXPLICIT PATH IS PRIMARY, and the wrapper is a cross-check.
    #
    # The previous version preferred mc.prob_over and wrapped the call in a
    # bare try/except that set the probability to None. mc.prob_over has a
    # different signature than I assumed, so every one of 7,851 rows became
    # "unpriceable" and the script sailed on into NaN log losses and a
    # "GATE PASSED" on an empty frame. A swallowed TypeError reported as a
    # data property is the worst failure mode in this codebase and I wrote
    # it.
    #
    # blended_mean -> p_over is read straight from mc_pricing's source and
    # section 1b's 4.423 ratio confirms the blend is being applied, so it is
    # the reliable path. app.py passes GAMES_PLAYED = 0, so stage_mult is 1.0
    # and the two should agree; where they do not, the difference is reported
    # rather than hidden.
    def prob_explicit(p, l):
        m = mc_pricing.blended_mean(MARKET, float(p), float(l))
        if m is None:
            return None
        return mc_pricing.p_over(MARKET, m, float(l))

    path = "mc_pricing.blended_mean -> mc_pricing.p_over (stage_mult 1.0)"
    print(f"  pricing path: {path}")

    # Cross-check against the wrapper, with the signature discovered rather
    # than assumed, and any failure PRINTED.
    wrapper = None
    try:
        import inspect
        import mc as _mc
        if hasattr(_mc, "prob_over"):
            sig = inspect.signature(_mc.prob_over)
            print(f"  mc.prob_over signature: {sig}")
            n_req = sum(1 for prm in sig.parameters.values()
                        if prm.default is inspect._empty
                        and prm.kind in (prm.POSITIONAL_ONLY,
                                         prm.POSITIONAL_OR_KEYWORD))
            cands = [("(market, proj, line)", lambda p, l: _mc.prob_over(
                          MARKET, float(p), float(l))),
                     ("(market, proj, line, 0)", lambda p, l: _mc.prob_over(
                          MARKET, float(p), float(l), 0))]
            for label, fn in cands:
                try:
                    t = fn(5.0, 3.5)
                except Exception as e:
                    print(f"    {label}: {type(e).__name__}: {e}")
                    continue
                if t is None:
                    print(f"    {label}: returned None on a test row")
                    continue
                wrapper = fn
                print(f"    {label}: OK, returned {t:.4f}  "
                      f"({n_req} required args)")
                break
        else:
            print("  mc has no prob_over; cross-check skipped")
    except Exception as e:
        print(f"  mc unavailable for cross-check: {type(e).__name__}: {e}")

    vals, raw_vals, means, fails, wrap_vals = [], [], [], 0, []
    first_err = None
    for proj, line in zip(g["projection"], g["line"]):
        try:
            v = prob_explicit(proj, line)
        except Exception as e:
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            v = None
        if wrapper is not None:
            try:
                wv = wrapper(proj, line)
            except Exception:
                wv = None
            wrap_vals.append(wv if wv is not None else np.nan)
        # the WRONG path, kept so the size of my error is visible rather than
        # asserted, and so Phase 4.4's effect is quantified
        try:
            rv = mc_pricing.p_over(MARKET, float(proj), float(line))
        except Exception:
            rv = None
        m = mc_pricing.blended_mean(MARKET, float(proj), float(line))
        means.append(m if m is not None else np.nan)
        raw_vals.append(rv if rv is not None else np.nan)
        if v is None:
            fails += 1
            vals.append(np.nan)
        else:
            vals.append(float(v))

    g = g.copy()
    g["p_mc"] = vals
    g["p_raw"] = raw_vals
    g["blended"] = means
    if wrapper is not None and len(wrap_vals) == len(g):
        g["p_wrap"] = wrap_vals
    n_ok = int(g["p_mc"].notna().sum())
    print(f"  priced {n_ok:,} of {len(g):,} rows, {fails:,} unpriceable")
    if first_err:
        print(f"  first pricing exception: {first_err}")
    if n_ok == 0:
        print()
        print("  ABORT: nothing priced. Every section below would report NaN")
        print("  and the gate would pass on an empty frame, which is exactly")
        print("  what the previous version did. Fix the pricing call first.")
        sys.exit(1)
    if "p_wrap" in g.columns:
        cmp = g.dropna(subset=["p_mc", "p_wrap"])
        if len(cmp):
            d = (cmp["p_mc"] - cmp["p_wrap"]).abs()
            print(f"  cross-check vs mc.prob_over on {len(cmp):,} rows: "
                  f"max abs diff {d.max():.2e}")
            if d.max() > 1e-9:
                print("    NOTE: the wrapper and the explicit path DIFFER, so")
                print("    mc.py applies something beyond blended_mean and")
                print("    p_over. The wrapper is what the app uses; prefer")
                print("    its numbers and investigate the gap.")
    sub = g.dropna(subset=["p_mc"])

    _rule("SECTION 1b: HOW BIG WAS THE ERROR, AND WHAT 4.4 ACTUALLY DID")
    both = g.dropna(subset=["p_mc", "p_raw"])
    print(f"  {'quantity':<34}{'blended (shipped)':>19}{'raw (my error)':>16}")
    print(f"  {'mean probability':<34}{both['p_mc'].mean():>19.4f}"
          f"{both['p_raw'].mean():>16.4f}")
    print(f"  {'sd of probability':<34}{both['p_mc'].std():>19.4f}"
          f"{both['p_raw'].std():>16.4f}")
    print(f"  {'min':<34}{both['p_mc'].min():>19.4f}"
          f"{both['p_raw'].min():>16.4f}")
    print(f"  {'max':<34}{both['p_mc'].max():>19.4f}"
          f"{both['p_raw'].max():>16.4f}")
    print(f"  {'market book_p sd, for scale':<34}"
          f"{both['book_p'].std():>19.4f}")
    print()
    dev = (g["projection"] - g["line"]).dropna()
    mdev = (g["blended"] - g["line"]).dropna()
    print(f"  sd of (projection - line)      {dev.std():.4f}")
    print(f"  sd of (blended mean - line)    {mdev.std():.4f}")
    print(f"  ratio                          "
          f"{dev.std() / max(mdev.std(), 1e-9):.3f}   "
          f"(should be about 1/0.2261 = 4.42)")
    print()
    print("  If the blended sd is close to the market's, the shipped price is")
    print("  in the right ballpark and the first run's over-dispersion was")
    print("  entirely my error rather than a property of the app.")

    if sub["p_mc"].max() > 1.0:
        print("\n  ABORT: p_over returned above 1, so the scale is wrong.")
        sys.exit(1)
    if sub["book_p"].std() < 0.02:
        print("\n  ABORT: book_p barely varies; the encompassing test would")
        print("  be meaningless here as it was for qb_passing.")
        sys.exit(1)
    return sub


def gate(g):
    _rule("GATE: reproduce the market's own published score")
    if not len(g):
        print("  ABORT: empty frame, nothing to gate.")
        sys.exit(1)
    mkt = logloss_vec(g["book_p"], g["over"])
    print(f"  book_p log loss on these rows   {mkt.mean():.5f}")
    print(f"  published (larger population)   {PUB_BOOK_LOGLOSS:.5f}")
    print(f"  difference                      {mkt.mean() - PUB_BOOK_LOGLOSS:+.5f}")
    rate = g["over"].mean()
    print(f"  over rate {rate:.4f} vs published {PUB_OVER_RATE:.4f}")
    if abs(rate - PUB_OVER_RATE) > RATE_TOL:
        print()
        print("  NOTE: the over rate differs by more than the tolerance. This")
        print("  population is the intersection with walk-forward")
        print("  projections, so it is legitimately smaller, but check the")
        print("  row count before trusting anything below.")
    else:
        print("\n  GATE PASSED.")
    return mkt


def section_headline(g, mkt_ll, n_boot):
    _rule("SECTION 2: LOG LOSS, THE SHIPPED PRICE AGAINST THE MARKET")
    mc_ll = logloss_vec(g["p_mc"], g["over"])
    d = mc_ll.mean() - mkt_ll.mean()
    lo, hi = boot_delta(mc_ll, mkt_ll, g["event_id"], n_boot)
    print(f"  {'source':<28}{'n':>8}{'logloss':>10}{'Brier':>10}")
    print(f"  {'market (devigged book_p)':<28}{len(g):>8,}"
          f"{mkt_ll.mean():>10.5f}"
          f"{float(np.mean((g['book_p'] - g['over']) ** 2)):>10.5f}")
    print(f"  {'shipped mc_pricing.p_over':<28}{len(g):>8,}"
          f"{mc_ll.mean():>10.5f}"
          f"{float(np.mean((g['p_mc'] - g['over']) ** 2)):>10.5f}")
    print()
    print(f"  delta (shipped minus market)  {d:+.5f}   95% [{lo:+.5f}, {hi:+.5f}]")
    print("  NEGATIVE means the shipped price is better calibrated.")
    if np.isfinite(hi) and hi < 0:
        ll_verdict = "better"
        print("  -> the shipped price BEATS the market on a proper scoring rule")
    elif np.isfinite(lo) and lo > 0:
        ll_verdict = "worse"
        print("  -> the shipped price is WORSE than the market, and the")
        print("     interval excludes zero")
    else:
        ll_verdict = "tied"
        print("  -> indistinguishable from the market on this test")

    _rule("SECTION 3: ENCOMPASSING, THE TEST BETTING CARES ABOUT")
    print("Outcome ~ book_p + (p_shipped - book_p), clustered on game. A")
    print("positive coefficient on the disagreement means the shipped price")
    print("carries information the market's does not, so combining them beats")
    print("the price alone. That can be true even when section 2 says the")
    print("shipped price is worse overall.")
    print()
    dis = (g["p_mc"] - g["book_p"]).to_numpy()
    X = np.column_stack([np.ones(len(g)), g["book_p"].to_numpy(), dis])
    f = rt.ols_cluster(X, g["over"].to_numpy(), g["event_id"].to_numpy())
    if f is None:
        print("  fit failed.")
        return None
    t_dis = f["beta"][2] / f["se"][2] if f["se"][2] else np.nan
    print(f"  {'term':<22}{'coef':>10}{'SE':>9}{'t':>7}")
    print(f"  {'book_p':<22}{f['beta'][1]:>+10.4f}{f['se'][1]:>9.4f}"
          f"{f['beta'][1] / f['se'][1] if f['se'][1] else np.nan:>7.2f}")
    print(f"  {'disagree (shipped-book)':<22}{f['beta'][2]:>+10.4f}"
          f"{f['se'][2]:>9.4f}{t_dis:>7.2f}")
    print(f"  n {f['n']:,}   games {f['G']:,}")
    print()
    print("  A book_p coefficient near 1 means the market is well calibrated")
    print("  on these rows and there is no slack for the model to exploit.")
    print()
    print(f"  mean |disagreement| {np.abs(dis).mean():.4f}, "
          f"sd {dis.std():.4f}")
    print("  (a tiny disagreement cannot produce an edge no matter how")
    print("   informative it is, so this number bounds the upside)")
    return t_dis, ll_verdict


def section_calibration(g):
    _rule("SECTION 4: WHERE THE SHIPPED PRICE IS WRONG, IF ANYWHERE")
    print("Both probabilities binned against the realized rate, so a")
    print("miscalibration in the shipped price is visible even if the")
    print("aggregate tests are null.")
    edges = [0, .30, .40, .45, .50, .55, .60, .70, 1.0]
    for lab, col in (("shipped (blended mean)", "p_mc"),
                     ("market devigged", "book_p"),
                     ("raw projection (MY ERROR, for contrast)", "p_raw")):
        print()
        print(f"  {lab}")
        print(f"    {'bin':<14}{'n':>7}{'mean p':>9}{'actual':>9}{'bias':>9}")
        gg = g.assign(_b=pd.cut(g[col], edges, include_lowest=True))
        for b, s in gg.groupby("_b", observed=True):
            if len(s) < 60:
                continue
            print(f"    {str(b):<14}{len(s):>7,}{s[col].mean():>9.4f}"
                  f"{s['over'].mean():>9.4f}"
                  f"{s[col].mean() - s['over'].mean():>+9.4f}")


def section_verdict(t_dis, ll_verdict):
    _rule("SECTION 5: WHAT TO DO WITH THE APP")
    print("  Recall the asymmetry stated at the top: BLEND's alpha and beta")
    print("  were fitted on 2023-2026, so they are in-sample for every row")
    print("  scored here. This test is biased IN FAVOUR of the shipped price.")
    print()
    if t_dis is None or not np.isfinite(t_dis):
        print("  No usable coefficient. Nothing to conclude.")
        return
    if t_dis > 2 and ll_verdict == "better":
        print(f"  Encompassing t = {t_dis:+.2f} AND better calibrated than the")
        print("  market on log loss. That is the strongest possible reading:")
        print("  the shipped price both beats the market overall and carries")
        print("  information it lacks.")
        print()
        print("  BUT this is the WEAK direction of the asymmetry. Refit BLEND")
        print("  walk-forward and re-run before changing anything. A positive")
        print("  result from blend constants fitted on these very rows is")
        print("  exactly what overfitting looks like.")
        print()
        print("  KEEP the Edge column pending that refit.")
    elif t_dis > 2:
        print(f"  Encompassing t = {t_dis:+.2f}, but log loss says the shipped")
        print(f"  price is {ll_verdict.upper()} than the market overall.")
        print()
        print("  These are not contradictory. A worse-calibrated forecast can")
        print("  still carry independent information: measured on synthetic")
        print("  data, a deliberately WORSE model returned encompassing")
        print("  t +4.68 while losing on log loss by +0.0123.")
        print()
        print("  What it means in practice: the shipped probability should NOT")
        print("  be displayed as a price, because it is miscalibrated, but a")
        print("  COMBINATION of it and the market might beat the market. That")
        print("  is a different product than an Edge column and it needs its")
        print("  own out-of-sample test.")
        print()
        print("  So: withhold the verdict, and treat the disagreement as a")
        print("  candidate signal for rules.py rather than as a price.")
    elif t_dis < -2:
        print(f"  Encompassing t = {t_dis:+.2f}, NEGATIVE. The shipped price")
        print("  is anti-informative against the market: where it disagrees,")
        print("  it is wrong more often than right. That is worse than a")
        print("  null and the Edge column should come off immediately.")
    else:
        print(f"  Encompassing t = {t_dis:+.2f}. NULL, and it is the strong")
        print("  direction of the asymmetry: the shipped price cannot beat")
        print("  the market even with its blend constants fitted on these")
        print("  very rows.")
        print()
        print("  RECOMMENDATION. Receptions should join the reference-only")
        print("  set, the same way rushing, receiving, qb_passing and")
        print("  qb_rushing already have. The mechanism is to set")
        print("  BLEND['receptions']['beta'] to 0.0, which makes")
        print("  has_model_signal False and routes the market through the")
        print("  existing J1 path: probability still shown, Edge and Side and")
        print("  Tier withheld.")
        print()
        print("  That would leave the app displaying NO model verdict in any")
        print("  market, and only the two rules in rules.py. That is a large")
        print("  change to the product, so it is worth being clear about what")
        print("  it does and does not say:")
        print()
        print("    it does NOT say beta 0.2261 was wrong. That beta is real")
        print("    and measured four ways against the LINE.")
        print()
        print("    it DOES say the line is not what you bet against. FanDuel")
        print("    moves receptions through the ODDS, the line is 93 percent")
        print("    flat, and a model that improves on a stale quantity adds")
        print("    nothing to a sharp one.")
        print()
        print("  The two rules are unaffected. Neither uses the projection")
        print("  for direction: rushing ignores it entirely and receiving")
        print("  uses it only as a veto.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 92)
    print("RECEPTIONS: does the SHIPPED price beat the devigged market price?")
    print(f"  seasons {seasons}   book {args.book}   devig {DEVIG_METHOD}")
    print("=" * 92)

    g = load(args, seasons)
    if len(g) < 2000:
        print("too few rows.")
        sys.exit(1)
    g = price_shipped(g)
    mkt_ll = gate(g)
    t_dis, ll_verdict = section_headline(g, mkt_ll, args.boot)
    section_calibration(g)
    section_verdict(t_dis, ll_verdict)

    _rule("DONE. Nothing was written. No constant was changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
