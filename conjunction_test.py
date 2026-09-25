"""conjunction_test.py - does the MODEL add to the air-yards rule?

Reports only. Writes nothing, places no bet.

WHY HERE AND NOT WHERE WE TRIED IT BEFORE

    under_bias.py ran this test on low-line rushing and found nothing: the
    under paid +0.0643 where the model DISAGREED and -0.0102 where it agreed,
    with the difference straddling zero. That result is now explainable rather
    than disappointing. Rushing's beta is -0.041 with a CI spanning zero, so
    the model has no directional signal there at all. Asking whether a
    signal-free model improves a rule is asking nothing.

    Receptions is different. Beta 0.226, game-clustered t +6.5, confirmed by
    four independent routes. Receiving is weaker at 0.066 but not empty. And
    air_yards_retest.py found the air-yards rule clears in BOTH of those
    markets, on the correct baseline, at 12 of 40 cells.

    So for the first time both halves of the conjunction have measured signal
    in the same market. This is the test the project's whole strategy rests
    on, and it has never been runnable until now.

THE RULE IS PRE-SPECIFIED, NOT RE-SEARCHED

    Taken from air_yards_retest.py's output, fixed before looking at anything
    here:

        PRIMARY    receiving_air_yards, z >= 1.0, bet the UNDER
                   receiving  lift +0.0864  [+0.0328, +0.1393]
                   receptions lift +0.0698  [+0.0154, +0.1301]

        SECONDARY  z >= 1.5 in both markets, reported but not the headline

    Re-tuning the threshold here would be searching the same data twice and
    would invalidate the interval. The gate checks these reproduce before
    anything else runs.

THE CONFOUND THAT COULD MAKE THIS UNINTERPRETABLE

    The receiving and receptions models carry targets_roll as a feature. A
    player with high recent air yards usually has high recent targets, so his
    projection is probably ABOVE the line, which means the model says OVER
    while the rule says UNDER. Agreement and the rule may therefore be
    mechanically opposed, leaving the agree cell small and unrepresentative.

    Section 2 measures that correlation before interpreting anything. If the
    two are strongly opposed, the binary agree/disagree split is close to
    meaningless and section 4's continuous version is the one to read.

FOUR WAYS OF ASKING IT, BECAUSE THE BINARY VERSION IS THE WEAKEST

    Section 3  binary: inside the rule cell, agree versus disagree
    Section 4  continuous: profit on z and on the model's dev jointly
    Section 5  the reverse direction: does air yards improve the MODEL's bets
    Section 6  the honest accounting

WHAT WOULD BE A FINDING

    The agree cell beating the disagree cell with a bootstrapped difference
    excluding zero, or the model's dev carrying a significant coefficient
    alongside z. Either would mean the two signals are not redundant and a
    screen has a measurable basis.

    A null here is also worth having. It would mean the air-yards rule stands
    alone and the model contributes nothing to it, which is useful to know
    before building a screen around both.

Run from the repo root:
    python conjunction_test.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import eval_harness as eh
import input_weights as iw
import recency_test as rt

INPUT = "receiving_air_yards"
MARKETS = ("receiving", "receptions")
PRIMARY_Z = 1.0

# From air_yards_retest.py. The gate aborts unless these reproduce.
PUB_LIFT = {
    ("receiving", 1.0): (1054, 0.0864),
    ("receptions", 1.0): (961, 0.0698),
}
BASELINE_TOL = 0.002


def _rule(t):
    print()
    print("=" * 94)
    print(t)
    print("=" * 94)


def boot_mean(vals, clusters, n_boot, seed=0):
    """Vectorised cluster bootstrap of a mean."""
    df = pd.DataFrame({"v": np.asarray(vals, float),
                       "g": np.asarray(clusters)})
    agg = df.groupby("g")["v"].agg(["sum", "size"])
    if len(agg) < 15:
        return np.nan, np.nan
    s, n = agg["sum"].to_numpy(), agg["size"].to_numpy().astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    N = n[idx].sum(axis=1)
    d = s[idx].sum(axis=1) / N
    return tuple(np.percentile(d, [2.5, 97.5]))


def boot_diff(g, flag_col, roi_col, n_boot, seed=0):
    """Vectorised cluster bootstrap of mean(flag=True) minus mean(flag=False).

    Resamples GAMES and recomputes both sides inside each replicate, so the
    interval accounts for a game contributing rows to both sides.
    """
    agg = g.groupby("event_id").apply(
        lambda d: pd.Series({
            "as_": d.loc[d[flag_col], roi_col].sum(),
            "an": float(d[flag_col].sum()),
            "bs": d.loc[~d[flag_col], roi_col].sum(),
            "bn": float((~d[flag_col]).sum()),
        }), include_groups=False)
    if len(agg) < 20:
        return np.nan, np.nan, np.nan
    a_s, an, bs, bn = (agg[c].to_numpy() for c in ("as_", "an", "bs", "bn"))
    G = len(agg)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    AN, BN = an[idx].sum(axis=1), bn[idx].sum(axis=1)
    ok = (AN >= 25) & (BN >= 25)
    if not ok.any():
        return np.nan, np.nan, np.nan
    d = a_s[idx].sum(axis=1)[ok] / AN[ok] - bs[idx].sum(axis=1)[ok] / BN[ok]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return lo, hi, float((d > 0).mean())


def prep(p):
    """Standardised air-yards shock and the model's view, per market."""
    rc, bc = f"{INPUT}__recent", f"{INPUT}__base"
    need = [rc, bc, "projection", "line", "roi_under", "roi_over"]
    g = p[p["market"].isin(MARKETS)].dropna(subset=need).copy()
    g["_dev_raw"] = g[rc] - g[bc]
    g["z"] = g.groupby("market")["_dev_raw"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    g["dev"] = g["projection"] - g["line"]
    g["devz"] = g.groupby("market")["dev"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    g["model_under"] = g["dev"] < 0
    return g


def gate(g, n_boot):
    _rule("GATE: the pre-specified rule must reproduce")
    print("Note the z-scores here are computed on rows that ALSO require a")
    print("projection, so counts can differ slightly from air_yards_retest.")
    print("The LIFT is what must match; the row count is reported for")
    print("transparency and allowed to move a little.")
    print()
    ok = True
    print(f"  {'market':<13}{'bets':>7}{'pub n':>7}{'base':>9}{'ROI':>9}"
          f"{'lift':>9}{'pub lift':>10}{'diff':>9}")
    for market in MARKETS:
        gm = g[g["market"] == market]
        base = gm["roi_under"].mean()
        cell = gm[gm["z"] >= PRIMARY_Z]
        lift = cell["roi_under"].mean() - base
        pn, pl = PUB_LIFT[(market, PRIMARY_Z)]
        print(f"  {market:<13}{len(cell):>7,}{pn:>7,}{base:>+9.4f}"
              f"{cell['roi_under'].mean():>+9.4f}{lift:>+9.4f}{pl:>+10.4f}"
              f"{lift - pl:>+9.4f}")
        if abs(lift - pl) > 0.02:
            ok = False
    if not ok:
        print()
        print("  ABORT: the rule this script tests does not reproduce the")
        print("  lift it was specified from. Conjunction results would be")
        print("  about a different rule.")
        sys.exit(1)
    print("\n  GATE PASSED.")


def section_confound(g):
    _rule("SECTION 2: ARE THE TWO SIGNALS MECHANICALLY OPPOSED?")
    print("The models carry targets_roll, and recent air yards correlate with")
    print("recent targets, so a high-z player probably projects ABOVE the")
    print("line. If so, the rule says UNDER while the model says OVER and the")
    print("binary split is close to meaningless.")
    print()
    print(f"  {'market':<13}{'n':>7}{'corr(z,devz)':>14}"
          f"{'model_under all':>17}{'model_under z>=1':>18}")
    for market in MARKETS:
        gm = g[g["market"] == market]
        cell = gm[gm["z"] >= PRIMARY_Z]
        print(f"  {market:<13}{len(gm):>7,}{gm['z'].corr(gm['devz']):>14.4f}"
              f"{gm['model_under'].mean():>17.1%}"
              f"{cell['model_under'].mean():>18.1%}")
    print()
    print("  A strongly NEGATIVE correlation, or a model_under share inside")
    print("  the cell far below the overall share, means the signals oppose")
    print("  each other and section 4 is the honest test.")
    print()
    print("  mean dev by z quintile, to see the shape rather than one number:")
    for market in MARKETS:
        gm = g[g["market"] == market].copy()
        try:
            gm["_q"] = pd.qcut(gm["z"], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        row = " ".join(f"{gm.loc[gm['_q'] == q, 'devz'].mean():+.3f}"
                       for q in sorted(gm["_q"].dropna().unique()))
        print(f"    {market:<13}{row}")


def section_binary(g, n_boot):
    _rule("SECTION 3: INSIDE THE RULE CELL, AGREE VERSUS DISAGREE")
    print(f"Rule: {INPUT} z >= {PRIMARY_Z}, bet the UNDER. Agreement means the")
    print("model's projection is also below the line.")
    print()
    print("DO NOT DRAW A CONCLUSION FROM THIS SECTION. It is reported because")
    print("it is the intuitive way to ask the question, and it is measurably")
    print("the WRONG way. On synthetic data with z and dev correlated at 0.55,")
    print("which is the confound section 2 measures:")
    print()
    print("    when the model genuinely ADDED   agree minus disagree -0.0392")
    print("    when the model was pure NOISE    agree minus disagree -0.1483")
    print()
    print("Both negative. The binary split returns the wrong SIGN even when")
    print("the model truly helps, because the confound makes the agree cell")
    print("small and selects it on low z. Section 4 separated the same two")
    print("worlds cleanly (devz t -5.23 against t -0.93).")
    print()
    print("So read section 4 for the answer and treat this as descriptive.")
    for thr in (PRIMARY_Z, 1.5):
        print()
        print(f"  --- z >= {thr} ---")
        print(f"  {'market':<13}{'split':<26}{'bets':>7}{'ROI':>9}"
              f"{'lo95':>9}{'hi95':>9}")
        for market in MARKETS:
            cell = g[(g["market"] == market) & (g["z"] >= thr)]
            if len(cell) < 150:
                continue
            for flag, lab in ((True, "model AGREES (proj<line)"),
                              (False, "model disagrees")):
                sub = cell[cell["model_under"] == flag]
                if len(sub) < 60:
                    print(f"  {market:<13}{lab:<26}{len(sub):>7,}"
                          f"   too few")
                    continue
                lo, hi = boot_mean(sub["roi_under"], sub["event_id"], n_boot)
                print(f"  {market:<13}{lab:<26}{len(sub):>7,}"
                      f"{sub['roi_under'].mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}")
            a = cell[cell["model_under"]]
            b = cell[~cell["model_under"]]
            if len(a) >= 60 and len(b) >= 60:
                d = a["roi_under"].mean() - b["roi_under"].mean()
                lo, hi, sh = boot_diff(cell, "model_under", "roi_under",
                                       n_boot)
                star = "  <--" if np.isfinite(lo) and lo > 0 else ""
                print(f"  {market:<13}{'AGREE MINUS DISAGREE':<26}"
                      f"{'':>7}{d:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}{star}")
                print(f"  {'':<13}{'  share of replicates > 0':<26}"
                      f"{'':>7}{sh:>9.3f}")


def section_continuous(g):
    _rule("SECTION 4: THE CONTINUOUS VERSION")
    print("Profit on the under, regressed on the air-yards shock and the")
    print("model's standardised dev, clustered on game. This avoids throwing")
    print("away the magnitude of either signal and is robust to the two being")
    print("correlated in a way that guts the binary split.")
    print()
    print("A NEGATIVE devz coefficient means the under pays more when the")
    print("model projects lower, which is the model adding information.")
    print()
    print(f"  {'market':<13}{'spec':<20}{'term':<8}{'coef':>10}{'SE':>9}"
          f"{'t':>7}")
    for market in MARKETS:
        gm = g[g["market"] == market].dropna(subset=["z", "devz", "roi_under"])
        if len(gm) < 800:
            continue
        y = gm["roi_under"].to_numpy()
        ev = gm["event_id"].to_numpy()
        for name, terms in (("z alone", ["z"]),
                            ("devz alone", ["devz"]),
                            ("z + devz", ["z", "devz"]),
                            ("z + devz + z*devz", ["z", "devz", "_ix"])):
            gg = gm.copy()
            gg["_ix"] = gg["z"] * gg["devz"]
            X = np.column_stack([np.ones(len(gg))] +
                                [gg[t].to_numpy() for t in terms])
            f = rt.ols_cluster(X, gg["roi_under"].to_numpy(),
                               gg["event_id"].to_numpy())
            if f is None:
                continue
            for i, t in enumerate(terms, start=1):
                tt = f["beta"][i] / f["se"][i] if f["se"][i] else np.nan
                print(f"  {market if i == 1 else '':<13}"
                      f"{name if i == 1 else '':<20}{t:<8}"
                      f"{f['beta'][i]:>+10.4f}{f['se'][i]:>9.4f}{tt:>7.2f}")
        print()
    print("  The interaction term asks whether the model helps MORE at high")
    print("  z. That is the precise form of 'the two compound'.")


def section_reverse(g, n_boot):
    _rule("SECTION 5: THE OTHER DIRECTION")
    print("Does the air-yards shock improve the MODEL's own bets? Take the")
    print("side the model prefers, then filter on z. Receptions is the market")
    print("where the model has real signal, so this is the version that")
    print("matters most for the existing candidate rule.")
    for market in MARKETS:
        gm = g[g["market"] == market].copy()
        if len(gm) < 800:
            continue
        gm["roi_model"] = np.where(gm["model_under"], gm["roi_under"],
                                   gm["roi_over"])
        print()
        print(f"  {market}")
        print(f"    {'cut':<30}{'bets':>7}{'ROI':>9}{'lo95':>9}{'hi95':>9}")
        lo, hi = boot_mean(gm["roi_model"], gm["event_id"], n_boot)
        print(f"    {'model side, all rows':<30}{len(gm):>7,}"
              f"{gm['roi_model'].mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}")
        # where the model says UNDER and air yards agrees, both point down
        for thr in (0.5, 1.0, 1.5):
            sub = gm[(gm["model_under"]) & (gm["z"] >= thr)]
            if len(sub) >= 100:
                lo, hi = boot_mean(sub["roi_under"], sub["event_id"], n_boot)
                star = "  <--" if np.isfinite(lo) and lo > 0 else ""
                print(f"    {f'both say UNDER, z>={thr}':<30}{len(sub):>7,}"
                      f"{sub['roi_under'].mean():>+9.4f}{lo:>+9.4f}"
                      f"{hi:>+9.4f}{star}")
        # and where they conflict, which side wins
        sub = gm[(~gm["model_under"]) & (gm["z"] >= PRIMARY_Z)]
        if len(sub) >= 100:
            lo, hi = boot_mean(sub["roi_under"], sub["event_id"], n_boot)
            print(f"    {'conflict: rule UNDER wins?':<30}{len(sub):>7,}"
                  f"{sub['roi_under'].mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}")
            lo, hi = boot_mean(sub["roi_over"], sub["event_id"], n_boot)
            print(f"    {'conflict: model OVER wins?':<30}{len(sub):>7,}"
                  f"{sub['roi_over'].mean():>+9.4f}{lo:>+9.4f}{hi:>+9.4f}")
            print("      In a conflict only one of these can be taken. The")
            print("      larger one tells you which signal to defer to.")


def section_honesty(g):
    _rule("SECTION 6: HONEST ACCOUNTING")
    for market in MARKETS:
        cell = g[(g["market"] == market) & (g["z"] >= PRIMARY_Z)]
        both = cell[cell["model_under"]]
        print(f"  {market:<13} rule fires {len(cell):>5,} times "
              f"({len(cell) / 3.25:>4.0f}/season), both agree "
              f"{len(both):>5,} ({len(both) / 3.25:>4.0f}/season)")
    print()
    print("  WHAT IS OUT OF SAMPLE HERE AND WHAT IS NOT.")
    print()
    print("  The RULE is pre-specified from air_yards_retest, so the")
    print("  thresholds were not tuned in this script. Good.")
    print()
    print("  But the rule itself was found by searching 25 input-market cells")
    print("  on this same data, and the decision to look at air yards came")
    print("  from that search. So the lift is in-sample with respect to the")
    print("  search even though it is out-of-sample with respect to this")
    print("  script. The conjunction result inherits that.")
    print()
    print("  Nothing here changes the conclusion that forward testing is the")
    print("  only thing that settles any of it. What this script CAN settle")
    print("  is whether building a screen around both signals is worth the")
    print("  effort, which is a design question rather than a betting one.")
    print()
    print("  ALSO: 2023 was weak in every air-yards input, roughly a quarter")
    print("  the magnitude of 2024 and 2025. If the mechanism is a stale")
    print("  market weight rather than a structural bias, it can un-stale")
    print("  itself at any time and the forward test is the only warning")
    print("  system.")


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

    print("=" * 94)
    print("CONJUNCTION: does the model add to the air-yards rule?")
    print(f"  seasons {seasons}   book {args.book}   input {INPUT}   "
          f"z >= {PRIMARY_Z}")
    print("=" * 94)

    p, _ = iw.build(types.SimpleNamespace(
        cache=args.cache, refresh=args.refresh, book=args.book), seasons)
    g = prep(p)
    print(f"\n  {len(g):,} rows with air yards AND a walk-forward projection")
    if len(g) < 2000:
        print("too few rows.")
        sys.exit(1)

    gate(g, args.boot)
    section_confound(g)
    section_binary(g, args.boot)
    section_continuous(g)
    section_reverse(g, args.boot)
    section_honesty(g)

    _rule("DONE. Nothing was written. No bet was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
