"""qb_rushing_anchor.py - why is qb_rushing beta 0.655 at consensus and 0.094 at FanDuel?

Reports only. Writes nothing, fits no production constant.

THE OBSERVATION

  eval_harness, September 24, on 2023-2026:

    anchor      pooled beta            2023       2024       2025      2026
    fanduel     0.094  (t +1.02)      0.106     -0.068      0.260     0.003
    consensus   0.655  (t +2.81)      0.883     -0.059      0.247     0.023

  2024, 2025 and 2026 agree between anchors to within 0.02. The whole
  divergence is 2023, at t +6.61, which is the project's own tripwire for
  "this is a bug until proven otherwise".

  Two corroborating oddities in the same row. 2023's consensus SE is 0.134,
  TIGHTER than 2024's 0.153 and 2025's 0.161 on comparable n, which is
  backwards for a larger coefficient. And 2023's consensus alpha collapses to
  0.17 against 2.63, 1.97 and 1.89 in the other seasons.

THREE COMPETING EXPLANATIONS, AND THE TEST THAT SEPARATES THEM

  H1 POPULATION. FanDuel posts on only 1,380 of 1,705 props, so the two
     anchors are fitted on different row sets. Section 2 refits both anchors
     on the FanDuel-available rows ONLY. If 0.883 survives there, population
     is not the cause.

  H2 SYNTHETIC CONSENSUS. line_consensus is a MEDIAN over books, and a median
     over an even count averages the two middle values, landing on numbers no
     book offered. The project has already been burned by this: "settling on
     a line no book offered produces inflated apparent results." Section 3
     measures what share of consensus lines were actually available somewhere.

  H3 DEGENERATE LINE. If the 2023 consensus line is near-constant or on a
     very coarse grid, then dev and out both collapse to projection and
     actual minus roughly a constant. Beta then stops measuring what the
     model adds to the line and starts measuring the slope of actual on
     projection, which for OLS fitted values is approximately 1. A beta of
     0.883 with t(b=1) of -0.88 is precisely that signature.

     Section 5 tests it directly: it regresses actual on projection with no
     line involved at all, per season. If the 2023 consensus beta sits near
     that slope while other seasons do not, H3 is the answer and the 0.883 is
     an artifact of the anchor carrying no information rather than the model
     carrying a lot.

  These are not exclusive. H2 can cause H3.

Run from the repo root:
    python qb_rushing_anchor.py --all-seasons --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd

import eval_harness as eh
from models import data_utils

MARKET = "qb_rushing"

# Published by eval_harness on the run this script exists to explain. The gate
# aborts unless it reproduces them, because a script that reimplements part of
# an existing procedure must reproduce that procedure's output before it is
# allowed to produce a new number.
PUBLISHED = {
    ("fanduel", None): (0.094, 1380),
    ("consensus", None): (0.655, 1705),
    ("consensus", 2023): (0.883, 600),
    ("fanduel", 2023): (0.106, 424),
}
BETA_TOL = 0.004


def _rule(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def fit(sub, col):
    return eh.ols_clustered(sub["projection"] - sub[col],
                            sub["actual"] - sub[col], sub["event_id"])


def line(label, f):
    if f is None:
        print(f"  {label:<34} too few rows")
        return
    t0 = f["beta"] / f["se_beta"] if f["se_beta"] > 0 else np.nan
    t1 = (f["beta"] - 1) / f["se_beta"] if f["se_beta"] > 0 else np.nan
    print(f"  {label:<34}{f['n']:>6}{f['clusters']:>7}{f['beta']:>9.3f}"
          f"{f['se_beta']:>8.3f}{t0:>9.2f}{t1:>9.2f}{f['alpha']:>8.2f}")


def head():
    print(f"  {'':<34}{'n':>6}{'games':>7}{'beta':>9}{'SE':>8}"
          f"{'t(b=0)':>9}{'t(b=1)':>9}{'alpha':>8}")


def load(args, seasons):
    _rule("LOADING")
    stored = eh.FETCH_MARKET.get(MARKET, MARKET)
    sec = eh._secrets()
    box = {}

    def cf():
        if "c" not in box:
            box["c"] = eh.connect(sec)
        return box["c"]

    lines = eh.load_lines(cf, seasons, [stored], args.cache, args.refresh)
    if lines.empty:
        print("no lines found.")
        sys.exit(1)
    lines, rep = data_utils.split_qb_rushing(lines, min_match_rate=0.98)
    lines = lines[lines["market"] == MARKET].copy()
    lines = lines[lines["week"].notna()]
    print(f"{len(lines):,} book-level {MARKET} rows, "
          f"{lines['book'].nunique()} books")

    props = eh.collapse_books(lines)
    props = eh.report_lead(props, None)

    scored = eh.score_market(MARKET, seasons, mode="walk_forward",
                             population=args.score_population)
    if not len(scored):
        print("nothing scored.")
        sys.exit(1)

    props["_key"] = data_utils.norm_join_name(props["player"])
    scored["_key"] = data_utils.norm_join_name(scored["player"])
    props["week"] = props["week"].astype(int)
    scored["week"] = scored["week"].astype(int)
    joined = props.merge(
        scored[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"joined {len(joined):,} rows")
    return lines, joined


def gate(joined):
    _rule("GATE: reproduce eval_harness before explaining anything")
    head()
    ok = True
    for (src, season), (want_b, want_n) in PUBLISHED.items():
        col = f"line_{src}"
        sub = joined.dropna(subset=[col]).copy()
        if season is not None:
            sub = sub[sub["season"] == season]
        f = fit(sub, col)
        tag = f"{src}" + (f" {season}" if season else " pooled")
        line(tag, f)
        if f is None:
            ok = False
            continue
        db = abs(f["beta"] - want_b)
        dn = f["n"] - want_n
        if db > BETA_TOL or dn != 0:
            print(f"      MISMATCH: beta {f['beta']:.3f} vs published "
                  f"{want_b:.3f} (diff {db:.4f}), n {f['n']} vs {want_n}")
            ok = False
    if not ok:
        print()
        print("ABORT: this script does not reproduce the harness numbers it")
        print("claims to explain, so nothing below can be trusted. Check the")
        print("split, the lead filter and the score population first.")
        sys.exit(1)
    print("\nGATE PASSED.")


def section_population(joined):
    _rule("SECTION 2 (H1): SAME ROWS, BOTH ANCHORS")
    print("Restricted to props where FanDuel actually posted, so the anchor")
    print("is the only thing that differs. If 0.883 survives here, the extra")
    print("325 consensus-only rows are NOT the cause.")
    both = joined.dropna(subset=["line_fanduel", "line_consensus"]).copy()
    print(f"\nrows with both anchors: {len(both):,} of {len(joined):,}")
    print()
    head()
    for src in ("fanduel", "consensus"):
        line(f"{src} pooled, matched rows", fit(both, f"line_{src}"))
    print()
    for season, g in both.groupby("season"):
        for src in ("fanduel", "consensus"):
            line(f"{season} {src}", fit(g, f"line_{src}"))

    _rule("SECTION 2b: THE CONSENSUS-ONLY ROWS ON THEIR OWN")
    only = joined[joined["line_fanduel"].isna()].copy()
    print(f"{len(only):,} props with no FanDuel line")
    if len(only):
        print()
        print("by season:")
        print(only.groupby("season").size().to_string())
        print()
        head()
        line("consensus-only rows", fit(only, "line_consensus"))
        for season, g in only.groupby("season"):
            line(f"{season} consensus-only", fit(g, "line_consensus"))


def section_synthetic(lines, joined):
    _rule("SECTION 3 (H2): IS THE CONSENSUS LINE A PRICE ANY BOOK OFFERED?")
    print("line_consensus is a MEDIAN. Over an even number of books the median")
    print("averages the two middle values and can land on a number nobody")
    print("posted. Settling or fitting against such a line has already")
    print("produced one pure artifact in this project.")

    keys = ["season", "week", "market", "player"]
    offered = (lines.groupby(keys)["line"]
               .agg(lambda s: frozenset(s.dropna().tolist()))
               .rename("offered").reset_index())
    offered["_key"] = data_utils.norm_join_name(offered["player"])
    offered["week"] = offered["week"].astype(int)

    j = joined.merge(offered[["season", "week", "market", "_key", "offered"]],
                     on=["season", "week", "market", "_key"], how="left")
    j = j[j["offered"].notna()].copy()
    j["consensus_offered"] = [
        (c in o) for c, o in zip(j["line_consensus"], j["offered"])]

    print()
    print(f"  {'season':<8}{'props':>7}{'n_books':>9}{'consensus was':>15}"
          f"{'distinct':>10}{'line SD':>9}")
    print(f"  {'':<8}{'':>7}{'median':>9}{'offered':>15}{'lines':>10}{'':>9}")
    for season, g in j.groupby("season"):
        print(f"  {season:<8}{len(g):>7}{g['n_books'].median():>9.0f}"
              f"{g['consensus_offered'].mean():>14.1%}"
              f"{g['line_consensus'].nunique():>10}"
              f"{g['line_consensus'].std():>9.2f}")

    print()
    print("  A low 'consensus was offered' share in one season, alongside a")
    print("  beta that only appears in that season, is H2 confirmed.")

    # Beta split on whether the consensus line was a real price.
    if j["consensus_offered"].nunique() > 1:
        print()
        head()
        for flag, g in j.groupby("consensus_offered"):
            tag = "consensus WAS offered" if flag else "consensus was SYNTHETIC"
            line(tag, fit(g, "line_consensus"))
        for season in sorted(j["season"].unique()):
            gs = j[j["season"] == season]
            if gs["consensus_offered"].nunique() < 2:
                continue
            for flag, g in gs.groupby("consensus_offered"):
                tag = f"{season} " + ("real" if flag else "SYNTHETIC")
                line(tag, fit(g, "line_consensus"))


def section_spread(joined):
    _rule("SECTION 4: HOW FAR APART ARE THE TWO ANCHORS?")
    both = joined.dropna(subset=["line_fanduel", "line_consensus"]).copy()
    both["gap"] = both["line_consensus"] - both["line_fanduel"]
    print(f"  {'season':<8}{'n':>7}{'mean gap':>10}{'SD gap':>9}"
          f"{'|gap|>2':>9}{'fd SD':>8}{'cons SD':>9}{'books':>7}")
    for season, g in both.groupby("season"):
        print(f"  {season:<8}{len(g):>7}{g['gap'].mean():>+10.2f}"
              f"{g['gap'].std():>9.2f}{(g['gap'].abs() > 2).mean():>8.1%}"
              f"{g['line_fanduel'].std():>8.2f}"
              f"{g['line_consensus'].std():>9.2f}"
              f"{g['n_books'].median():>7.0f}")
    print()
    print("  A season where the consensus line has much LOWER SD than")
    print("  FanDuel's is a season where the median has flattened the line,")
    print("  which is H3.")


def section_degenerate(joined):
    _rule("SECTION 5 (H3): THE DECISIVE TEST")
    print("If the anchor carries no information, then dev and out both reduce")
    print("to projection and actual minus roughly a constant, and beta stops")
    print("measuring what the model adds to the line. It becomes the slope of")
    print("ACTUAL on PROJECTION, with no line involved, which for OLS fitted")
    print("values is approximately 1.")
    print()
    print("So: compare each season's anchored beta against the unanchored")
    print("slope of actual on projection. Where the anchored beta approaches")
    print("the unanchored slope, the line is contributing nothing.")
    print()
    print(f"  {'season':<8}{'n':>6}{'actual~proj':>13}{'SE':>8}"
          f"{'beta fd':>10}{'beta cons':>11}{'cons/slope':>12}")
    for season, g in joined.groupby("season"):
        raw = eh.ols_clustered(g["projection"], g["actual"], g["event_id"])
        gf = g.dropna(subset=["line_fanduel"])
        gc = g.dropna(subset=["line_consensus"])
        ff = fit(gf, "line_fanduel") if len(gf) > 20 else None
        fc = fit(gc, "line_consensus") if len(gc) > 20 else None
        if raw is None:
            continue
        ratio = (fc["beta"] / raw["beta"]) if (fc and raw["beta"]) else np.nan
        print(f"  {season:<8}{raw['n']:>6}{raw['beta']:>13.3f}"
              f"{raw['se_beta']:>8.3f}"
              f"{(ff['beta'] if ff else np.nan):>10.3f}"
              f"{(fc['beta'] if fc else np.nan):>11.3f}"
              f"{ratio:>12.2f}")
    print()
    print("  A ratio near 1.0 in one season and far below it in the others is")
    print("  H3 CONFIRMED for that season: the beta is measuring the model")
    print("  against nothing, not the model against the market.")


def section_books(lines, joined):
    _rule("SECTION 6: PER-BOOK LINES, AND THE ALL-SEASON BOOK SET")
    print("book presence by season (rows):")
    tab = lines.pivot_table(index="book", columns="season", values="line",
                            aggfunc="size", fill_value=0)
    print(tab.to_string())

    seasons = sorted(lines["season"].unique())
    always = [b for b in tab.index if (tab.loc[b] > 0).all()]
    print()
    print(f"books present in ALL seasons {seasons}: {always}")
    missing = [b for b in tab.index if b not in always]
    print(f"books NOT in every season: {missing}")

    if not always or len(always) == len(tab.index):
        print("\nnothing to restrict, skipping the re-fit.")
        return

    _rule("SECTION 6b: CONSENSUS REBUILT ON THE ALL-SEASON BOOK SET")
    print("Plan item 1.10, never run. line_consensus is a median over a")
    print("CHANGING book set, which is the leading explanation for 2023")
    print("behaving differently in several places at once.")
    sub = lines[lines["book"].isin(always)].copy()
    props2 = eh.collapse_books(sub)
    props2["_key"] = data_utils.norm_join_name(props2["player"])
    props2["week"] = props2["week"].astype(int)
    j2 = props2.merge(
        joined[["season", "week", "market", "_key", "projection", "actual"]],
        on=["season", "week", "market", "_key"], how="inner")
    print(f"\n{len(j2):,} rows on the restricted book set")
    print()
    head()
    for src in ("fanduel", "consensus"):
        col = f"line_{src}"
        if col in j2.columns:
            line(f"{src}, stable books", fit(j2.dropna(subset=[col]), col))
    for season, g in j2.groupby("season"):
        gg = g.dropna(subset=["line_consensus"])
        line(f"{season} consensus, stable books", fit(gg, "line_consensus"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    ap.add_argument("--all-seasons", action="store_true")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--score-population", default="all",
                    choices=["all", "board"])
    args = ap.parse_args()

    if args.all_seasons:
        seasons = list(eh.ALL_SEASONS)
    elif args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = [2025]

    print("=" * 78)
    print("qb_rushing: WHY 0.655 AT CONSENSUS AND 0.094 AT FANDUEL")
    print(f"  seasons {seasons}   population {args.score_population}")
    print("=" * 78)

    lines, joined = load(args, seasons)
    gate(joined)
    section_population(joined)
    section_synthetic(lines, joined)
    section_spread(joined)
    section_degenerate(joined)
    section_books(lines, joined)

    _rule("DONE. Nothing was written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
