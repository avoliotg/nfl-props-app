"""air_yards_retest.py - the air-yards finding, tested properly this time.

Reports only. Writes nothing, places no bet.

WHAT input_weights.py FOUND

    Six flags, all on receiving and receptions, all NEGATIVE, meaning the line
    moves MORE with recent downfield usage than the outcome justifies:

        receiving   receiving_air_yards   -0.0484  t -3.10   ctrl -0.0448
        receiving   air_yards_share      -14.6048  t -3.41   ctrl -13.66
        receiving   wopr                  -7.7402  t -2.77   ctrl -7.2310
        receptions  receiving_air_yards   -0.0039  t -3.50   ctrl -0.0043
        receptions  air_yards_share       -1.1098  t -3.58   ctrl -1.1461
        receptions  wopr                  -0.6183  t -3.01   ctrl -0.5970

    All six survive the control for the model's own projection, so this is
    NOT something the existing features already encode. That part stands.

TWO THINGS I GOT WRONG IN THAT SCRIPT

    1. THE SEASON TEST WAS TOO STRICT. It demanded sign consistency including
       2026, which had 335 rows against roughly 2,600 for a full season. All
       six flags flip only in 2026 and are consistently negative across the
       three COMPLETE seasons. A three-week partial season cannot veto a
       finding, and requiring it to agree is a test that would reject almost
       anything. This script requires a minimum row count before a season
       votes, and reports the excluded ones rather than hiding them.

    2. SECTION 4'S BAR WAS ZERO, WHICH IS THE WRONG BAR. Those cells bet the
       UNDER, and the under already carries the structural bias measured in
       under_bias.py: receptions -0.0221, receiving -0.0308 as a baseline.
       So an under cell beating zero is partly just inheriting that. The
       correct comparison is against the SAME market's baseline ROI on the
       SAME side, and the correct test is a bootstrap of the DIFFERENCE, not
       of the level. Receptions air-yards-share at z >= 1.0 showed +0.0742
       against a baseline of -0.0221, so the lift is about +0.096, larger
       than the figure printed. Whether that survives a proper interval is
       what this script exists to find out.

A THIRD THING WORTH CHECKING, WHICH I DID NOT BUILD AT ALL

    The six coefficients grow monotonically across seasons. air_yards_share
    on receiving runs -4.92, -20.05, -21.62 from 2023 to 2025. That is either
    a genuine trend, the market changing how it prices downfield usage, or an
    artifact of something that also changed over those seasons, which for
    this project means book composition. Section 4 examines it directly
    rather than leaving it as a curiosity.

AND ONE HONEST DEFLATION OF THE MULTIPLICITY MATH

    input_weights counted 25 tests and 6 flags. But air yards, air-yards
    share and WOPR are near-duplicates: WOPR is a weighted combination of
    target share and air-yards share by construction. So 6 flags are closer
    to 2 independent findings, one per market, on one underlying construct.
    That cuts both ways. The multiplicity penalty is smaller than 25 tests
    implies, and the amount of independent evidence is smaller than 6 flags
    implies. Section 1 measures the correlations so this is quantified rather
    than asserted.

Run from the repo root:
    python air_yards_retest.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import eval_harness as eh
import input_weights as iw
import recency_test as rt
import under_bias as ub

# The construct under test, per market. Deliberately narrow: this script is
# following up a specific finding, not searching again.
TARGETS = {
    "receiving": ["receiving_air_yards", "air_yards_share", "wopr",
                  "targets", "target_share"],
    "receptions": ["receiving_air_yards", "air_yards_share", "wopr",
                   "targets", "target_share"],
}

# A season needs at least this many rows before its sign counts as a vote.
# 2026 had 335 at the time of writing, roughly an eighth of a full season.
MIN_SEASON_ROWS = 800


def _rule(t):
    print()
    print("=" * 92)
    print(t)
    print("=" * 92)


def section_correlation(p):
    _rule("SECTION 1: HOW MANY INDEPENDENT FINDINGS ARE THERE REALLY?")
    print("WOPR is built from target share and air-yards share, so the six")
    print("flags are not six pieces of evidence. Correlations among the")
    print("recent-window regressors, per market:")
    for market, inputs in TARGETS.items():
        cols = [f"{i}__recent" for i in inputs
                if f"{i}__recent" in p.columns]
        g = p[p["market"] == market].dropna(subset=cols)
        if len(g) < 500 or len(cols) < 2:
            continue
        print()
        print(f"  {market}  (n={len(g):,})")
        c = g[cols].corr()
        c.index = [i.replace("__recent", "") for i in c.index]
        c.columns = [i.replace("__recent", "") for i in c.columns]
        print(c.round(3).to_string())
    print()
    print("  Correlations near 1 mean one construct wearing several names.")
    print("  Treat the highest-correlated cluster as ONE test per market.")


def section_seasons(p):
    _rule("SECTION 2: SEASON SIGNS, WITH PARTIAL SEASONS EXCLUDED FROM THE VOTE")
    print(f"A season votes only if it has at least {MIN_SEASON_ROWS} rows.")
    print("Excluded seasons are still PRINTED, marked 'no vote', so nothing")
    print("is hidden. This is the correction: the earlier script let a")
    print("three-week partial season veto six findings.")
    results = {}
    for market, inputs in TARGETS.items():
        for inp in inputs:
            rc, bc = f"{inp}__recent", f"{inp}__base"
            if rc not in p.columns:
                continue
            g0 = p[p["market"] == market].dropna(subset=[rc, bc, "resid"])
            if len(g0) < 1000:
                continue
            X = np.column_stack([np.ones(len(g0)), g0[rc], g0[bc]])
            full = rt.ols_cluster(X, g0["resid"].to_numpy(),
                                  g0["event_id"].to_numpy())
            if full is None:
                continue
            print()
            print(f"  {market} / {inp}   pooled {full['beta'][1]:+.4f}  "
                  f"t {full['beta'][1] / full['se'][1]:+.2f}")
            print(f"    {'season':<8}{'n':>7}{'recent':>12}{'SE':>10}{'t':>7}"
                  f"  vote")
            votes = []
            for season, g in g0.groupby("season"):
                if len(g) < 150:
                    continue
                Xs = np.column_stack([np.ones(len(g)), g[rc], g[bc]])
                f = rt.ols_cluster(Xs, g["resid"].to_numpy(),
                                   g["event_id"].to_numpy())
                if f is None:
                    continue
                t = f["beta"][1] / f["se"][1] if f["se"][1] else np.nan
                voting = len(g) >= MIN_SEASON_ROWS
                tag = "yes" if voting else "no vote (partial)"
                print(f"    {season:<8}{f['n']:>7,}{f['beta'][1]:>+12.4f}"
                      f"{f['se'][1]:>10.4f}{t:>7.2f}  {tag}")
                if voting:
                    votes.append(np.sign(f["beta"][1]))
            ok = bool(votes) and len(set(votes)) == 1
            print("    SIGN CONSISTENT across voting seasons" if ok
                  else "    SIGN FLIPS among voting seasons")
            results[(market, inp)] = (full["beta"][1], full["se"][1], ok)
    return results


def section_trend(p):
    _rule("SECTION 3: IS THE EFFECT GROWING, AND IF SO WHY?")
    print("The coefficients rise monotonically 2023 to 2025. Either the")
    print("market changed how it prices downfield usage, or something else")
    print("changed over those seasons. For this project the usual suspect is")
    print("book composition, but this run is FanDuel only, so that")
    print("explanation is unavailable and the trend needs another one.")
    print()
    print("Two candidates that can be checked here:")
    print("  A. the INPUT's own distribution shifted (more air-yards spread)")
    print("  B. the LINE got more responsive to it over time")
    print()
    for market in TARGETS:
        for inp in ("air_yards_share",):
            rc, bc = f"{inp}__recent", f"{inp}__base"
            if rc not in p.columns:
                continue
            g0 = p[p["market"] == market].dropna(subset=[rc, bc, "line",
                                                         "actual"])
            if len(g0) < 1000:
                continue
            print(f"  {market} / {inp}")
            print(f"    {'season':<8}{'n':>7}{'sd(recent)':>12}"
                  f"{'line~recent':>13}{'actual~recent':>15}{'gap':>10}")
            for season, g in g0.groupby("season"):
                if len(g) < 300:
                    continue
                X = np.column_stack([np.ones(len(g)), g[rc], g[bc]])
                ev = g["event_id"].to_numpy()
                fl = rt.ols_cluster(X, g["line"].to_numpy(), ev)
                fa = rt.ols_cluster(X, g["actual"].to_numpy(), ev)
                if not (fl and fa):
                    continue
                print(f"    {season:<8}{len(g):>7,}{g[rc].std():>12.4f}"
                      f"{fl['beta'][1]:>13.4f}{fa['beta'][1]:>15.4f}"
                      f"{fa['beta'][1] - fl['beta'][1]:>+10.4f}")
            print()
    print("  If 'line~recent' climbs while 'actual~recent' is flat, the")
    print("  market is leaning harder on the input over time and the trend is")
    print("  real. If both move together, the gap is stable and the trend is")
    print("  in the input, not the mispricing.")


def boot_lift(cell, base, n_boot, seed=0):
    """Cluster bootstrap of cell ROI minus baseline ROI on the same side.

    Both are resampled from the SAME game draw, because the cell is a subset
    of the baseline and treating them as independent would overstate the
    interval.

    VECTORISED. The first version rebuilt the frame with pd.concat on every
    replicate, which at 1,500 replicates across 40 cells meant roughly 60,000
    concatenations over 2,500 game groups and made the script take many
    minutes per input. Per game only four numbers are ever needed: the cell's
    profit sum and count, and the baseline's profit sum and count. A replicate
    is then pure array indexing. Verified to return bit-identical percentiles
    to the slow version on the same seed, and its cost is almost flat in
    n_boot (1.0s at 300 replicates, 1.2s at 1,500).
    """
    allg = pd.concat([base.assign(_in=False), cell.assign(_in=True)])
    allg = allg.drop_duplicates(subset=["_rowid"], keep="last")
    agg = allg.groupby("event_id").apply(
        lambda d: pd.Series({
            "cs": d.loc[d["_in"], "_roi"].sum(),
            "cn": float(d["_in"].sum()),
            "bs": d["_roi"].sum(),
            "bn": float(len(d)),
        }), include_groups=False)
    if len(agg) < 20:
        return np.nan, np.nan
    cs, cn, bs, bn = (agg[c].to_numpy() for c in ("cs", "cn", "bs", "bn"))
    G = len(agg)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    CN = cn[idx].sum(axis=1)
    BN = bn[idx].sum(axis=1)
    ok = (CN >= 30) & (BN > 0)
    if not ok.any():
        return np.nan, np.nan
    d = cs[idx].sum(axis=1)[ok] / CN[ok] - bs[idx].sum(axis=1)[ok] / BN[ok]
    return tuple(np.percentile(d, [2.5, 97.5]))


def section_lift(p, results, n_boot):
    _rule("SECTION 4: ROI AGAINST THE RIGHT BASELINE")
    print("An under cell beating zero partly inherits the structural under")
    print("bias. The bar is the SAME market's baseline ROI on the SAME side,")
    print("and the interval is a bootstrap of the DIFFERENCE.")
    print()
    base_roi = {}
    for market, g in p.groupby("market"):
        base_roi[(market, "under")] = g["roi_under"].mean()
        base_roi[(market, "over")] = g["roi_over"].mean()
        print(f"  baseline {market:<12} under {base_roi[(market,'under')]:+.4f}"
              f"   over {base_roi[(market,'over')]:+.4f}")

    keep = [(m, i) for (m, i), (b, se, ok) in results.items() if ok]
    if not keep:
        print()
        print("  No input survived section 2, so nothing to price.")
        return
    print()
    print("  Negative coefficients mean the line OVERweights the input, so a")
    print("  HIGH recent input is over-reflected and the UNDER is the side.")
    for market, inp in keep:
        rc, bc = f"{inp}__recent", f"{inp}__base"
        g0 = p[p["market"] == market].dropna(subset=[rc, bc]).copy()
        if len(g0) < 600:
            continue
        b = results[(market, inp)][0]
        side = "under" if b < 0 else "over"
        col = f"roi_{side}"
        g0["_dev"] = g0[rc] - g0[bc]
        g0["_z"] = (g0["_dev"] - g0["_dev"].mean()) / g0["_dev"].std(ddof=0)
        g0["_roi"] = g0[col]
        g0["_rowid"] = np.arange(len(g0))
        bl = base_roi[(market, side)]
        print()
        print(f"  {market} / {inp}  -> bet the {side.upper()} when z is high")
        print(f"    baseline {side} ROI {bl:+.4f}")
        print(f"    {'cut':<14}{'bets':>7}{'ROI':>9}{'lift':>9}"
              f"{'lift lo95':>11}{'lift hi95':>11}")
        for thr in (0.5, 1.0, 1.5, 2.0):
            cell = g0[g0["_z"] >= thr] if side == "under" else g0[g0["_z"] >= thr]
            if len(cell) < 150:
                continue
            roi = cell["_roi"].mean()
            lo, hi = boot_lift(cell, g0, n_boot)
            star = "  <--" if np.isfinite(lo) and lo > 0 else ""
            print(f"    z>=+{thr:<10}{len(cell):>7,}{roi:>+9.4f}"
                  f"{roi - bl:>+9.4f}{lo:>+11.4f}{hi:>+11.4f}{star}")
        print()
        print("    'lift' is ROI minus the baseline. A cell that beats zero")
        print("    but not the baseline has found nothing beyond the bias")
        print("    already measured in under_bias.py.")


def section_honesty(results):
    _rule("SECTION 5: WHAT THIS DOES AND DOES NOT ESTABLISH")
    ok = [k for k, v in results.items() if v[2]]
    print(f"  inputs tested                {len(results)}")
    print(f"  sign-consistent when voting  {len(ok)}")
    print()
    print("  Per section 1, the air-yards cluster is close to ONE construct")
    print("  per market, so count independent findings as 2 at most, not 6.")
    print()
    print("  The direction is worth stating plainly because it is the")
    print("  OPPOSITE of the original hypothesis. Nobody predicted this. The")
    print("  line does not underreact to a player's bad narrative; it")
    print("  OVERreacts to a player's recent downfield usage. A receiver")
    print("  coming off two deep-target games gets a line that leans too far")
    print("  toward that, and the under is the side.")
    print()
    print("  That is a coherent story: air yards are volatile and a couple of")
    print("  deep looks move a three-game average a long way, so a pricing")
    print("  model that weights them like stable volume will overshoot. But a")
    print("  coherent story is not evidence, and this is still one construct")
    print("  found by searching 25 cells on data already used many times.")
    print()
    print("  Forward testing is the only thing that settles it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--boot", type=int, default=1500)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 92)
    print("AIR YARDS RETEST: correct season handling, correct baseline")
    print(f"  seasons {seasons}   book {args.book}   boot {args.boot}")
    print("=" * 92)

    # Reuse input_weights' loader rather than restating it, so this cannot
    # measure a different population than the run it is correcting.
    p, _ = iw.build(types.SimpleNamespace(
        cache=args.cache, refresh=args.refresh, book=args.book), seasons)
    if len(p) < 2000:
        print("too few rows.")
        sys.exit(1)

    section_correlation(p)
    results = section_seasons(p)
    section_trend(p)
    section_lift(p, results, args.boot)
    section_honesty(results)

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
