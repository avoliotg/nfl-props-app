"""conjunction_followup.py - settle three open questions about the receiving conjunction.

Reports only. Writes nothing, places no bet.

WHAT conjunction_test.py LEFT OPEN

    Receiving, air-yards z >= 1.0, bet the under:
        model agrees (proj < line)   506 bets   +0.0982  [+0.0219, +0.1767]
        model disagrees              547 bets   +0.0147  [-0.0697, +0.0937]
        agree minus disagree                    +0.0835  [-0.0254, +0.1920]
    and at z >= 1.5 the agree cell was +0.1557 [+0.0501, +0.2684] on 275 bets.

    Three things were unresolved, and each has a specific test.

    1. THE BINARY AND CONTINUOUS VERSIONS DISAGREE. The agree/disagree gap is
       large (+0.0835) while the continuous devz coefficient is marginal
       (t -1.56). Those cannot both be right if the effect is linear in the
       model's dev. Only 27.7 percent of receiving rows have proj < line, so
       the model sits above the line most of the time and the binary split may
       be capturing a TAIL that a straight line dilutes.

       SECTION 1 puts the dummy and its interaction with z into the
       regression directly, alongside a binned version of devz that shows the
       shape without assuming one. If the effect is in the SIGN, the dummy
       carries it and the continuous term should stay weak. If the binned
       version is monotone, the continuous spec was simply underpowered.

    2. THE SEASON SPLIT WAS NEVER RUN on the conjunction cell. Every other
       finding in this project has been tested this way and two died from it.
       SECTION 2.

    3. THE BEST-OF-N PROBLEM IS UNQUANTIFIED. Twelve cells were reported and
       the headline is the largest. Bonferroni is too crude here because the
       cells overlap heavily.

       SECTION 3 does it properly with a PLACEBO. It shuffles the model's
       side within market and z-quintile, which destroys any relationship to
       the outcome while preserving the model's marginal frequency and its
       relationship to z, then recomputes the SAME grid of cells and takes
       the maximum statistic. Repeating that gives an empirical distribution
       for "the best cell out of this grid under the null", and the observed
       maximum is compared against it. That is the honest multiplicity
       correction for a search of this shape.

    SECTION 4 asks whether the result depends on the exact definition of
    agreement, since 'proj < line' is one arbitrary cut of a continuous
    quantity.

WHAT THIS SCRIPT CANNOT FIX

    The rule was found by searching 25 input-market cells on this data. The
    placebo in section 3 corrects for the conjunction grid, NOT for the
    search that produced the rule. Forward testing remains the only thing
    that addresses that.

Run from the repo root:
    python conjunction_followup.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import conjunction_test as ct
import eval_harness as eh
import input_weights as iw
import recency_test as rt

MARKETS = ("receiving", "receptions")
GRID_Z = (1.0, 1.5)
PUB = {("receiving", 1.0, True): (506, 0.0982),
       ("receiving", 1.0, False): (547, 0.0147)}


def _rule(t):
    print()
    print("=" * 94)
    print(t)
    print("=" * 94)


def gate(g):
    _rule("GATE: the cells being followed up must reproduce")
    ok = True
    print(f"  {'cell':<38}{'bets':>7}{'pub n':>7}{'ROI':>9}{'pub ROI':>9}"
          f"{'diff':>9}")
    for (market, thr, agree), (pn, pr) in PUB.items():
        cell = g[(g["market"] == market) & (g["z"] >= thr) &
                 (g["model_under"] == agree)]
        r = cell["roi_under"].mean()
        lab = f"{market} z>={thr} {'agree' if agree else 'disagree'}"
        print(f"  {lab:<38}{len(cell):>7,}{pn:>7,}{r:>+9.4f}{pr:>+9.4f}"
              f"{r - pr:>+9.4f}")
        if abs(r - pr) > 0.003 or abs(len(cell) - pn) > 5:
            ok = False
    if not ok:
        print("\n  ABORT: the cells do not reproduce.")
        sys.exit(1)
    print("\n  GATE PASSED.")


def section_dummy(g):
    _rule("SECTION 1: IS THE EFFECT IN THE SIGN OR IN THE MAGNITUDE?")
    print("Four specifications per market. 'under' is the dummy for")
    print("proj < line, 'ix' is its interaction with z. A large negative")
    print("continuous devz means magnitude matters; a large POSITIVE dummy")
    print("coefficient means only the SIGN matters.")
    print()
    print(f"  {'market':<12}{'spec':<26}{'term':<8}{'coef':>10}{'SE':>9}"
          f"{'t':>7}")
    for market in MARKETS:
        gm = g[g["market"] == market].dropna(
            subset=["z", "devz", "roi_under"]).copy()
        if len(gm) < 800:
            continue
        gm["under"] = gm["model_under"].astype(float)
        gm["ix"] = gm["under"] * gm["z"]
        gm["ixc"] = gm["devz"] * gm["z"]
        specs = [
            ("z + devz (continuous)", ["z", "devz"]),
            ("z + under (dummy)", ["z", "under"]),
            ("z + under + ix", ["z", "under", "ix"]),
            ("z + devz + under", ["z", "devz", "under"]),
        ]
        for name, terms in specs:
            X = np.column_stack([np.ones(len(gm))] +
                                [gm[t].to_numpy() for t in terms])
            f = rt.ols_cluster(X, gm["roi_under"].to_numpy(),
                               gm["event_id"].to_numpy())
            if f is None:
                continue
            for i, t in enumerate(terms, start=1):
                tt = f["beta"][i] / f["se"][i] if f["se"][i] else np.nan
                print(f"  {market if i == 1 else '':<12}"
                      f"{name if i == 1 else '':<26}{t:<8}"
                      f"{f['beta'][i]:>+10.4f}{f['se'][i]:>9.4f}{tt:>7.2f}")
        print()

    print("  And the SHAPE, without assuming linearity. Under-side ROI by")
    print("  devz quintile, inside the rule cell (z >= 1.0):")
    print()
    print(f"  {'market':<12}{'devz quintile':<16}{'bets':>7}{'mean devz':>11}"
          f"{'ROI':>9}")
    for market in MARKETS:
        cell = g[(g["market"] == market) & (g["z"] >= 1.0)].copy()
        if len(cell) < 300:
            continue
        try:
            cell["_q"] = pd.qcut(cell["devz"], 5, labels=False,
                                 duplicates="drop")
        except ValueError:
            continue
        for q in sorted(cell["_q"].dropna().unique()):
            sub = cell[cell["_q"] == q]
            print(f"  {market if q == 0 else '':<12}"
                  f"{f'q{int(q) + 1} (lowest dev first)' if q == 0 else f'q{int(q) + 1}':<16}"
                  f"{len(sub):>7,}{sub['devz'].mean():>11.3f}"
                  f"{sub['roi_under'].mean():>+9.4f}")
        print()
    print("  A monotone decline across quintiles means magnitude matters and")
    print("  the continuous spec was underpowered. A step between the lowest")
    print("  quintile and the rest means the effect is in the sign.")


def section_seasons(g, n_boot):
    _rule("SECTION 2: SEASON SPLIT ON THE CONJUNCTION CELL")
    print("Never run before on this cell. Two findings in this project have")
    print("died at exactly this step, so it is the real test.")
    for market in MARKETS:
        for thr in GRID_Z:
            cell = g[(g["market"] == market) & (g["z"] >= thr) &
                     g["model_under"]]
            if len(cell) < 150:
                continue
            print()
            print(f"  {market}  z >= {thr}  and model agrees   "
                  f"({len(cell):,} bets, pooled {cell['roi_under'].mean():+.4f})")
            print(f"    {'season':<8}{'bets':>7}{'under win':>11}{'ROI':>9}"
                  f"{'lo95':>9}{'hi95':>9}")
            signs = []
            for season, sub in cell.groupby("season"):
                if len(sub) < 40:
                    print(f"    {season:<8}{len(sub):>7,}   too few")
                    continue
                lo, hi = ct.boot_mean(sub["roi_under"], sub["event_id"],
                                      max(500, n_boot // 3))
                print(f"    {season:<8}{len(sub):>7,}"
                      f"{1 - sub['over'].mean():>11.4f}"
                      f"{sub['roi_under'].mean():>+9.4f}{lo:>+9.4f}"
                      f"{hi:>+9.4f}")
                signs.append(np.sign(sub["roi_under"].mean()))
            print("    SIGN CONSISTENT" if signs and all(s > 0 for s in signs)
                  else "    SIGN FLIPS")


def _grid_stats(g, flag):
    """The same grid of statistics conjunction_test reported."""
    out = {}
    for market in MARKETS:
        for thr in GRID_Z:
            cell = g[(g["market"] == market) & (g["z"] >= thr)]
            a = cell[cell[flag]]
            b = cell[~cell[flag]]
            if len(a) < 60 or len(b) < 60:
                continue
            out[(market, thr, "diff")] = (a["roi_under"].mean() -
                                          b["roi_under"].mean())
            out[(market, thr, "agree")] = a["roi_under"].mean()
    return out


def section_placebo(g, n_perm, seed=0):
    _rule("SECTION 3: PLACEBO NULL FOR THE BEST-OF-N PROBLEM")
    print("Twelve cells were reported and the headline is the largest, so the")
    print("interval on that cell is optimistic. Bonferroni is too crude")
    print("because the cells overlap heavily.")
    print()
    print("Instead: shuffle the model's side within market and z-quintile.")
    print("That destroys any relationship to the OUTCOME while keeping the")
    print("model's marginal frequency and its relationship to z intact. Then")
    print("recompute the same grid and take the MAXIMUM. Repeating gives the")
    print("distribution of 'best cell in this grid under the null'.")
    print()
    obs = _grid_stats(g, "model_under")
    if not obs:
        print("  nothing to test.")
        return
    obs_max_diff = max(v for k, v in obs.items() if k[2] == "diff")
    obs_max_agree = max(v for k, v in obs.items() if k[2] == "agree")
    print(f"  observed max agree-minus-disagree : {obs_max_diff:+.4f}")
    print(f"  observed max agree-cell ROI       : {obs_max_agree:+.4f}")

    gg = g.copy()
    gg["_zq"] = gg.groupby("market")["z"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    rng = np.random.default_rng(seed)
    md, ma = [], []
    for _ in range(n_perm):
        gg["_perm"] = gg.groupby(["market", "_zq"])["model_under"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        st = _grid_stats(gg, "_perm")
        if not st:
            continue
        md.append(max(v for k, v in st.items() if k[2] == "diff"))
        ma.append(max(v for k, v in st.items() if k[2] == "agree"))
    if not md:
        print("  placebo produced nothing.")
        return
    md, ma = np.array(md), np.array(ma)
    p_d = float((md >= obs_max_diff).mean())
    p_a = float((ma >= obs_max_agree).mean())
    print()
    print(f"  placebo permutations: {len(md)}")
    print(f"  {'statistic':<28}{'observed':>10}{'null 95th':>11}"
          f"{'null max':>10}{'p':>8}")
    print(f"  {'max agree-minus-disagree':<28}{obs_max_diff:>+10.4f}"
          f"{np.percentile(md, 95):>+11.4f}{md.max():>+10.4f}{p_d:>8.3f}")
    print(f"  {'max agree-cell ROI':<28}{obs_max_agree:>+10.4f}"
          f"{np.percentile(ma, 95):>+11.4f}{ma.max():>+10.4f}{p_a:>8.3f}")
    print()
    print("  This p-value IS corrected for choosing the best of the grid.")
    print("  It is NOT corrected for the 25-cell search that produced the")
    print("  air-yards rule in the first place.")
    print()
    if p_d > 0.10 and p_a > 0.10:
        print("  Neither statistic beats its own null. The conjunction is")
        print("  within what searching this grid produces by chance, and the")
        print("  headline cell should not be treated as a finding.")
    elif min(p_d, p_a) <= 0.05:
        print("  At least one statistic beats the best-of-grid null. That is")
        print("  the strongest form of evidence available from this data.")
    else:
        print("  Borderline. Suggestive, not established.")


def section_definition(g, n_boot):
    _rule("SECTION 4: DOES IT DEPEND ON HOW 'AGREEMENT' IS DEFINED?")
    print("'proj < line' is one arbitrary cut of a continuous quantity. If")
    print("the result only exists at exactly that cut, it is a threshold")
    print("artifact of the kind this project has already documented twice.")
    for market in MARKETS:
        cell = g[(g["market"] == market) & (g["z"] >= 1.0)]
        if len(cell) < 300:
            continue
        print()
        print(f"  {market}  z >= 1.0")
        print(f"    {'agreement definition':<28}{'bets':>7}{'ROI':>9}"
              f"{'lo95':>9}{'hi95':>9}")
        defs = [
            ("dev < 0 (as used)", cell["dev"] < 0),
            ("devz < -0.25", cell["devz"] < -0.25),
            ("devz < -0.50", cell["devz"] < -0.50),
            ("devz < +0.25", cell["devz"] < 0.25),
            ("devz below its median", cell["devz"] < cell["devz"].median()),
        ]
        for lab, mask in defs:
            sub = cell[mask]
            if len(sub) < 80:
                print(f"    {lab:<28}{len(sub):>7,}   too few")
                continue
            lo, hi = ct.boot_mean(sub["roi_under"], sub["event_id"], n_boot)
            star = "  <--" if np.isfinite(lo) and lo > 0 else ""
            print(f"    {lab:<28}{len(sub):>7,}"
                  f"{sub['roi_under'].mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}"
                  f"{star}")
    print()
    print("  A smooth profile across definitions is what a real effect looks")
    print("  like. A result that appears only at dev < 0 is not.")


def section_summary(g):
    _rule("SECTION 5: WHAT A FORWARD TEST WOULD LOG")
    print("Whatever the sections above say, this is the pre-registered shape")
    print("of the rule, fixed now so it cannot be tuned later:")
    print()
    print("    market     receiving (receiving yards)")
    print("    book       FanDuel")
    print("    signal     recent 3-game mean of receiving_air_yards minus")
    print("               the mean of the 8 games before those, standardized")
    print("               within market over the full history")
    print("    trigger    z >= 1.0")
    print("    filter     the walk-forward projection is below the line")
    print("    side       UNDER")
    print("    frequency  about 156 bets per season")
    print()
    for market in MARKETS:
        for thr in GRID_Z:
            cell = g[(g["market"] == market) & (g["z"] >= thr) &
                     g["model_under"]]
            if len(cell) < 100:
                continue
            print(f"    in-sample {market} z>={thr}: {len(cell):,} bets, "
                  f"ROI {cell['roi_under'].mean():+.4f}, "
                  f"under win {1 - cell['over'].mean():.4f}")
    print()
    print("  THREE FIELDS TO LOG PER BET, and the third is the one that")
    print("  cannot be recovered later:")
    print("    1. the model's projection and the line")
    print("    2. the rule's call (z value and side)")
    print("    3. your own read, plus one sentence of reasoning")
    print()
    print("  Your football read is not in any historical data, so the")
    print("  conjunction of rule and judgement can ONLY ever be measured")
    print("  going forward. Logging only the final decision destroys that")
    print("  information permanently.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--book", default="fanduel")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=400,
                    help="placebo permutations for the best-of-grid null")
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 94)
    print("CONJUNCTION FOLLOW-UP: sign or magnitude, seasons, best-of-N")
    print(f"  seasons {seasons}   book {args.book}   perm {args.perm}")
    print("=" * 94)

    p, _ = iw.build(types.SimpleNamespace(
        cache=args.cache, refresh=args.refresh, book=args.book), seasons)
    g = ct.prep(p)
    print(f"\n  {len(g):,} rows with air yards and a walk-forward projection")
    if len(g) < 2000:
        print("too few rows.")
        sys.exit(1)

    gate(g)
    section_dummy(g)
    section_seasons(g, args.boot)
    section_placebo(g, args.perm)
    section_definition(g, args.boot)
    section_summary(g)

    _rule("DONE. Nothing was written. No bet was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
