"""cross_book.py - does the low-line rushing under bias exist at other books?

Reports only. Writes nothing, places no bet.

WHY THIS IS A REAL TEST AND NOT JUST MORE SLICING

    The rushing rule was FOUND on FanDuel. Every other book in the cache is
    therefore out of sample in the BOOK dimension, on data that played no
    part in choosing the rule. That is not full out-of-sample, because it is
    the same games, the same players and the same seasons, so the outcomes
    are shared and the tests are correlated. But it is a genuine independent
    check of one thing: whether the bias is a property of the MARKET or a
    property of FanDuel.

    The two answers have different consequences.

      MARKET-WIDE. If low-line rushing unders beat the vig at most books,
      this is a structural feature of how recreational props are priced, it
      is unlikely to be arbitraged away quickly, and it can be played
      wherever the limits are friendliest.

      FANDUEL-ONLY. If it is one book, it is that book's pricing model or
      customer mix. Entirely plausible, and it is where the account is, but
      it is more fragile: one model update removes it, and there is no
      corroboration from anyone else.

THE TWO CONFOUNDS THAT HAVE TO BE HANDLED

    1. HOLD DIFFERS BY BOOK. ROI mixes the bias with the price. A book with
       an 8 percent hold can have the same bias and a worse ROI. So the WIN
       RATE is reported next to the ROI everywhere, and the breakeven implied
       by each book's own measured hold is reported alongside. The win rate
       is the bias; the ROI is the bias net of that book's price.

    2. COVERAGE DIFFERS BY BOOK. Books post different props, so a per-book
       comparison on all available rows mixes the effect with the
       population. Section 3 therefore repeats the test on only the props
       where EVERY compared book posted a line, which holds the population
       fixed and makes the books directly comparable.

    And one that cannot be handled: three books appear only in 2023
    (barstool, pointsbetus, unibet_us) and one only from 2025 (fanatics).
    Those are reported but excluded from the headline comparison, because a
    book present for one season is confounded with that season.

FIXED THRESHOLD, NOT PER-BOOK MEDIAN

    The cut is rushing line <= 46.5, which is FanDuel's rushing median and
    the cut the finding was made on. Using each book's own median instead
    would change the population per book and confound the comparison with
    the very thing being tested. Section 4 varies the threshold anyway, so
    sensitivity is visible.

Run from the repo root:
    python cross_book.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys
import types

import numpy as np
import pandas as pd

import devig
import eval_harness as eh
import under_bias as ub

MARKET = "rushing"
CUT = 46.5          # FanDuel's rushing median, the cut the finding was made on
MIN_ROWS = 200
FULL_SEASON_BOOKS_MIN = 3   # seasons a book must appear in to join the headline


def _rule(t):
    print()
    print("=" * 92)
    print(t)
    print("=" * 92)


def head(first="book"):
    print(f"  {first:<24}{'bets':>7}{'under win':>11}{'breakeven':>11}"
          f"{'edge':>8}{'ROI':>9}{'lo95':>9}{'hi95':>9}")


def emit(label, g, n_boot, seed=0):
    prof = g["roi_under"].to_numpy()
    roi = prof.mean()
    win = 1.0 - g["over"].mean()
    # breakeven implied by this book's own measured prices on these rows
    be = (1.0 / g["dec_under"]).mean()
    lo, hi, _ = ub.boot_ci(prof, g["event_id"], n_boot, seed)
    star = "  <--" if np.isfinite(lo) and lo > 0 else ""
    print(f"  {label:<24}{len(g):>7,}{win:>11.4f}{be:>11.4f}"
          f"{win - be:>+8.4f}{roi:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}{star}")
    return roi, lo, hi, win - be


def section_coverage(p):
    _rule("SECTION 1: BOOK COVERAGE")
    g = p[(p["market"] == MARKET)]
    tab = g.pivot_table(index="book", columns="season", values="line",
                        aggfunc="size", fill_value=0)
    print(tab.to_string())
    seasons_per = (tab > 0).sum(axis=1)
    stable = sorted(seasons_per[seasons_per >= FULL_SEASON_BOOKS_MIN].index)
    dropped = sorted(seasons_per[seasons_per < FULL_SEASON_BOOKS_MIN].index)
    print()
    print(f"  books in >= {FULL_SEASON_BOOKS_MIN} seasons (headline set): {stable}")
    print(f"  excluded as season-confounded: {dropped}")
    print()
    print("  mean hold by book, from the posted prices on these rows:")
    for book, gg in g.groupby("book"):
        print(f"    {book:<18}{gg['hold'].mean() * 100:>6.2f}%   "
              f"median {gg['hold'].median() * 100:.2f}%   n={len(gg):,}")
    return stable


def section_per_book(p, stable, n_boot):
    _rule(f"SECTION 2: LOW-LINE RUSHING UNDER, PER BOOK (line <= {CUT})")
    print("'under win' is the raw bias. 'breakeven' is what that book's own")
    print("prices require. 'edge' is the difference, and is the number to")
    print("compare ACROSS books, because it is net of each book's price.")
    print()
    head()
    res = {}
    g = p[(p["market"] == MARKET) & (p["line"] <= CUT)]
    for book, gg in g.groupby("book"):
        if len(gg) < MIN_ROWS:
            continue
        tag = book if book in stable else f"{book} (partial)"
        res[book] = emit(tag, gg, n_boot)
    print()
    print("  and the ABOVE-cut rows, which should show no edge if the effect")
    print("  is specific to the bottom of the board:")
    print()
    head()
    for book, gg in p[(p["market"] == MARKET) & (p["line"] > CUT)].groupby("book"):
        if len(gg) < MIN_ROWS:
            continue
        emit(book if book in stable else f"{book} (partial)", gg, n_boot)

    print()
    hs = [b for b in res if b in stable and np.isfinite(res[b][1])
          and res[b][1] > 0]
    pos = [b for b in res if b in stable and res[b][3] > 0]
    print(f"  stable books tested        {len([b for b in res if b in stable])}")
    print(f"  positive raw edge          {len(pos)}  {sorted(pos)}")
    print(f"  bootstrap bound above zero {len(hs)}  {sorted(hs)}")
    print()
    print("  READ IT THIS WAY. Individual books have a quarter of FanDuel's")
    print("  row count or less, so most intervals will straddle zero even if")
    print("  the effect is real and identical everywhere. The informative")
    print("  quantity is how many books show a POSITIVE raw edge, and whether")
    print("  the edges cluster near FanDuel's or scatter around zero.")
    return res


def section_matched(p, stable, n_boot):
    _rule("SECTION 3: MATCHED PROPS ONLY, POPULATION HELD FIXED")
    print("Restricted to props where every stable book posted a line, so the")
    print("books are priced on identical player-weeks and any difference is")
    print("the pricing rather than the coverage.")
    g = p[(p["market"] == MARKET) & (p["line"] <= CUT) &
          (p["book"].isin(stable))].copy()
    if not len(g):
        print("  nothing to compare.")
        return
    keyc = ["season", "week", "market", "_key"]
    cnt = g.groupby(keyc)["book"].nunique()
    full = cnt[cnt == len(stable)].index
    g = g.set_index(keyc).loc[full].reset_index()
    props = len(full)
    print(f"  {props:,} props posted by all {len(stable)} stable books, "
          f"{len(g):,} book-rows")
    if props < 150:
        print("  too few matched props for this to say much.")
        return
    print()
    head()
    edges = {}
    for book, gg in g.groupby("book"):
        roi, lo, hi, edge = emit(book, gg, n_boot)
        edges[book] = edge
    print()
    print("  spread of raw edge across books on identical props: "
          f"{min(edges.values()):+.4f} to {max(edges.values()):+.4f}")
    print()
    print("  A tight cluster of positive edges means the bias is in the")
    print("  MARKET. FanDuel alone positive with the others near zero means")
    print("  it is FanDuel's pricing.")

    # pooled across books on matched props, one row per prop per book. This
    # is NOT independent evidence, it is the same games many times, so the
    # interval is reported with that warning attached.
    print()
    head("pooled (correlated)")
    emit("all stable books", g, n_boot)
    print("    NOT independent evidence: the same games appear once per book,")
    print("    so this interval is too narrow. It is here to show the central")
    print("    tendency, not to test it.")


def section_threshold(p, stable, n_boot):
    _rule("SECTION 4: THRESHOLD SENSITIVITY, PER BOOK")
    print("The FanDuel profile decayed smoothly from the bottom of the board")
    print("upward, which is what a real effect looks like. If other books")
    print("show the same shape, that is corroboration independent of whether")
    print("any single interval clears zero.")
    g = p[(p["market"] == MARKET) & (p["book"].isin(stable))]
    print()
    print(f"  {'book':<18}" + "".join(f"{t:>9}" for t in
                                      (20.5, 30.5, 40.5, 46.5, 55.5, 70.5)))
    print(f"  {'':<18}" + "".join(f"{'<=' + str(t):>9}" for t in
                                  (20.5, 30.5, 40.5, 46.5, 55.5, 70.5)))
    for book, gg in g.groupby("book"):
        row = []
        for thr in (20.5, 30.5, 40.5, 46.5, 55.5, 70.5):
            s = gg[gg["line"] <= thr]
            row.append(f"{s['roi_under'].mean():>+9.4f}" if len(s) >= 120
                       else f"{'-':>9}")
        print(f"  {book:<18}" + "".join(row))
    print()
    print("  Each row should decay left to right if the effect lives at the")
    print("  bottom of the board. Rows that are flat or rise are not showing")
    print("  the same phenomenon.")


def section_other_markets(p, stable, n_boot):
    _rule("SECTION 5: THE SAME CUT IN OTHER MARKETS, AS A CONTROL")
    print("If low-line unders beat the vig in every market at every book,")
    print("this is not a rushing finding, it is a general under bias and the")
    print("rushing result was just the loudest cell. FanDuel's own run said")
    print("otherwise, and this checks it more broadly.")
    print()
    for market, g0 in p[p["book"].isin(stable)].groupby("market"):
        med = g0["line"].median()
        g = g0[g0["line"] <= med]
        if len(g) < 400:
            continue
        print(f"  {market}  (below its own median, {med:g})")
        head()
        for book, gg in g.groupby("book"):
            if len(gg) < MIN_ROWS:
                continue
            emit(book, gg, max(400, n_boot // 3))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--boot", type=int, default=1200)
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 92)
    print("CROSS-BOOK: is the low-line rushing under bias market-wide?")
    print(f"  seasons {seasons}   cut line <= {CUT}   boot {args.boot}")
    print("=" * 92)

    # All books this time. Reuses under_bias' loader and settler so the
    # FanDuel numbers must come out identical to the run being followed up.
    p = ub.build(types.SimpleNamespace(
        cache=args.cache, refresh=args.refresh, book="all"), seasons)

    _rule("GATE: the FanDuel subset must reproduce the earlier run")
    fd = p[p["book"] == "fanduel"]
    ub.gate(fd, types.SimpleNamespace(book="fanduel"))
    sub = fd[(fd["market"] == MARKET) & (fd["line"] <= CUT)]
    roi = sub["roi_under"].mean()
    print(f"  FanDuel rushing <= {CUT}: {len(sub):,} bets, ROI {roi:+.4f}")
    print(f"  published: 1,425 bets, ROI +0.0589")
    if len(sub) != 1425 or abs(roi - 0.0589) > 0.001:
        print()
        print("  ABORT: the cell being generalised does not reproduce.")
        sys.exit(1)
    print("  GATE PASSED.")

    stable = section_coverage(p)
    section_per_book(p, stable, args.boot)
    section_matched(p, stable, args.boot)
    section_threshold(p, stable, args.boot)
    section_other_markets(p, stable, args.boot)

    _rule("DONE. Nothing was written. No bet was placed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
