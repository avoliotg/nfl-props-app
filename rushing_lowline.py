"""rushing_lowline.py - the under bias on the LEAST selected cut available.

Reports only. Writes nothing, places no bet.

WHY NOT SIMPLY COMBINE THE TWO WINNING BUCKETS

    under_bias.py found rushing's two lowest quartiles both positive and both
    sign-consistent across seasons: +0.0632 on lines below 26.5 and +0.0544
    on 26.5 to 46.5, together 1,425 bets at roughly +0.059. Pooling them
    would very likely produce a lower bound above zero.

    It would also be selection. Those two cells were chosen BECAUSE they came
    out positive, out of twenty quartile cells tested. A bound computed on a
    group defined by its own result is not a bound.

    So this script does the opposite. It fixes ONE cut per market, chosen for
    a reason that has nothing to do with any result: the market's own MEDIAN
    line. Two cells per market, ten cells in total, every cell reported
    whether it helps or not. That is roughly half the multiplicity of the
    quartile run and none of the post-hoc grouping.

    The median split is also the a priori version of the actual hypothesis.
    The template-pricing observation is about the BOTTOM of the board, and
    "below the median line" is what that means without tuning.

    The quartile detail is still shown afterwards, clearly labelled as the
    selected version, plus a terciles cut so the answer's sensitivity to
    where the cut falls is visible rather than hidden.

WHAT WOULD MAKE THIS A RESULT

    The below-median rushing cell clearing zero on its bootstrap lower bound,
    holding its sign every season, surviving the ordinary-price restriction,
    and standing out against the one-in-forty that ten cells would produce by
    luck. Even then it is in-sample with respect to the decision to look at
    rushing at all, which came from an earlier run on the same data. Only
    forward testing fixes that, and nothing here substitutes for it.

Run from the repo root:
    python rushing_lowline.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import under_bias as ub


def _rule(t):
    print()
    print("=" * 84)
    print(t)
    print("=" * 84)


def head():
    print(f"  {'':<30}{'bets':>7}{'over':>8}{'ROI':>9}{'bootSD':>9}"
          f"{'lo95':>9}{'hi95':>9}")


def emit(label, g, col, n_boot, seed=0):
    prof = g[col].to_numpy()
    roi = prof.mean()
    lo, hi, sd = ub.boot_ci(prof, g["event_id"], n_boot, seed)
    star = "  <--" if np.isfinite(lo) and lo > 0 else ""
    print(f"  {label:<30}{len(g):>7,}{g['over'].mean():>8.4f}{roi:>+9.4f}"
          f"{sd:>9.4f}{lo:>+9.4f}{hi:>+9.4f}{star}")
    return roi, lo, hi


def section_median(p, n_boot):
    _rule("SECTION 1: MEDIAN SPLIT, EVERY MARKET, TEN CELLS")
    print("One cut per market at that market's own median line. Nothing about")
    print("this cut depends on any result. All ten cells are reported.")
    print()
    head()
    cells = []
    for market, g in p.groupby("market"):
        med = g["line"].median()
        for lab, gg in (("below median", g[g["line"] <= med]),
                        ("above median", g[g["line"] > med])):
            if len(gg) < 150:
                continue
            roi, lo, hi = emit(f"{market} {lab} ({med:g})", gg,
                               "roi_under", n_boot)
            cells.append((market, lab, med, len(gg), roi, lo, hi))
        print()
    hits = [c for c in cells if np.isfinite(c[5]) and c[5] > 0]
    print(f"  cells tested {len(cells)}, expected 1-sided hits "
          f"{len(cells) * 0.025:.1f}, cleared {len(hits)}")
    return cells, hits


def section_seasons(p, hits, n_boot):
    if not hits:
        _rule("SECTION 2: SEASON SPLIT")
        print("  Nothing cleared on the unselected cut, so there is nothing")
        print("  to split. That is the answer: on the least-selected version")
        print("  of this test, the under bias does not beat the vig.")
        return []
    _rule("SECTION 2: DOES EACH CLEARED CELL HOLD ITS SIGN EVERY SEASON?")
    survivors = []
    for market, lab, med, n, roi, lo, hi in hits:
        g = p[p["market"] == market]
        g = g[g["line"] <= med] if lab == "below median" else g[g["line"] > med]
        print()
        print(f"  {market} {lab}   pooled ROI {roi:+.4f} "
              f"95% [{lo:+.4f}, {hi:+.4f}]")
        head()
        signs = []
        for season, gg in g.groupby("season"):
            if len(gg) < 60:
                continue
            r, _, _ = emit(str(season), gg, "roi_under",
                           max(300, n_boot // 3))
            signs.append(np.sign(r))
        if signs and all(s > 0 for s in signs):
            print("    SIGN CONSISTENT")
            survivors.append((market, lab, med, n, roi, lo, hi))
        else:
            print("    SIGN FLIPS, unreliable")
    return survivors


def section_price(p, survivors, n_boot):
    if not survivors:
        return
    _rule("SECTION 3: ORDINARY PRICES ONLY")
    print("An edge that lives only on short-priced rows is the book pricing")
    print("information, not the book being wrong.")
    for market, lab, med, n, roi, lo, hi in survivors:
        g = p[p["market"] == market]
        g = g[g["line"] <= med] if lab == "below median" else g[g["line"] > med]
        g = g.copy()
        g["_p"] = pd.cut(g["dec_under"], [0, 1.75, 1.87, 1.95, 99],
                         labels=["shorter than -133", "-133 to -115",
                                 "-115 to -105", "longer than -105"])
        print()
        print(f"  {market} {lab}")
        head()
        for plab, gg in g.groupby("_p", observed=True):
            if len(gg) < 100:
                continue
            emit(str(plab), gg, "roi_under", max(300, n_boot // 3))


def section_sensitivity(p, n_boot):
    _rule("SECTION 4: HOW SENSITIVE IS THE ANSWER TO WHERE THE CUT FALLS?")
    print("Rushing only, since that is the market the earlier run pointed at.")
    print("If the edge appears at one cut and vanishes at a neighbouring one,")
    print("it is a threshold artifact of the kind plan 5.1 already documented.")
    g = p[p["market"] == "rushing"].copy()
    if len(g) < 500:
        print("  too few rushing bets.")
        return
    print()
    head()
    for thr in (20.5, 26.5, 30.5, 35.5, 40.5, 46.5, 50.5, 55.5):
        gg = g[g["line"] <= thr]
        if len(gg) < 200:
            continue
        emit(f"rushing line <= {thr:g}", gg, "roi_under", n_boot)
    print()
    print("  A smooth profile across neighbouring thresholds is what a real")
    print("  effect looks like. A spike at one value is not.")
    print()
    print("  and terciles, a third a priori cut:")
    head()
    try:
        g["_t"] = pd.qcut(g["line"], 3, duplicates="drop")
    except ValueError:
        return
    for b, gg in g.groupby("_t", observed=True):
        if len(gg) < 150:
            continue
        emit(f"rushing {b}", gg, "roi_under", n_boot)


def section_selected(p, n_boot):
    _rule("SECTION 5: THE SELECTED VERSION, FOR THE RECORD")
    print("Pooling the two quartile cells that came out positive. This IS")
    print("selection and the interval below is NOT a valid bound. It is here")
    print("so the number is on the record next to its own disclaimer rather")
    print("than being quietly recomputed later without one.")
    g = p[(p["market"] == "rushing") & (p["line"] <= 46.5)]
    if len(g) < 200:
        return
    print()
    head()
    emit("rushing <= 46.5 (SELECTED)", g, "roi_under", n_boot)
    print()
    print("  Compare this against section 1's below-median rushing cell. If")
    print("  the selected figure is much stronger than the unselected one,")
    print("  the difference is the selection, not the market.")


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
        import eval_harness as eh
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 84)
    print("UNDER BIAS on an UNSELECTED cut: median line split")
    print(f"  seasons {seasons}   book {args.book}   boot {args.boot}")
    print("=" * 84)

    # Reuse under_bias's loader and settler rather than restating them, so
    # this cannot silently measure a different population than the run it is
    # following up on.
    p = ub.build(types.SimpleNamespace(
        cache=args.cache, refresh=args.refresh, book=args.book), seasons)
    ub.gate(p, args)

    cells, hits = section_median(p, args.boot)
    survivors = section_seasons(p, hits, args.boot)
    section_price(p, survivors, args.boot)
    section_sensitivity(p, args.boot)
    section_selected(p, args.boot)

    _rule("DONE. Nothing was written. No bet was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
