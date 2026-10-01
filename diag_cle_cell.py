"""diag_cle_cell.py - what is the CLE opponent cell actually made of?

READ ONLY. Descriptive. No new hypothesis, no decision rule: this is
scrutiny of a result that already cleared, not another test.

WHAT CLEARED. trend_scan.py, 52 cells, max-statistic permutation over the
whole family: props against Cleveland returned under ROI +0.0944 on 685
bets, p 0.0230. Holdout conditions fixed in advance also held, dev +0.0812
on 423 bets and holdout +0.1155 on 262. The over tail cleared too, props
against Indianapolis at -0.1350, p 0.0180.

WHY SCRUTINY RATHER THAN CELEBRATION. Four specific ways this could be real
and uninteresting, or not real at all:

    1. NOT CLEVELAND, JUST THE BEST OF 32. The max statistic tests the best
       cell, so Houston at +0.0884 is untested. "The best opponent cell
       clears" is a weaker claim than "Cleveland is mispriced", and the
       second does not follow from the first.

    2. A TEAM NAME IS A LABEL, NOT A MECHANISM. Cleveland's 2023 defence
       and its 2026 defence are different units with different coordinators
       and personnel. If the effect is a mechanism it should track the
       defence's QUALITY, not the franchise.

    3. MARKET CONCENTRATION. If the cell is mostly rushing props below
       46.5, it is Rule 1 wearing a Cleveland jersey, and the two findings
       are one finding double-counted.

    4. SELECTION ON WHO GETS PROPS. Teams in low-scoring games attract
       props on different players. If the CLE cell is full of low lines on
       thin players, the effect may be a line-level or coverage artifact
       already measured elsewhere.

Each section below targets one of those.

Run:  python diag_cle_cell.py --cache lines_cache.parquet
"""
import argparse
import sys

import numpy as np
import pandas as pd

BOOK = "fanduel"
MARKETS = ["receiving", "receptions", "rushing", "qb_passing"]
STAT_COL = {"receiving": "receiving_yards", "receptions": "receptions",
            "rushing": "rushing_yards", "qb_passing": "passing_yards"}
FOCUS = ["CLE", "HOU", "IND", "TB"]


def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def profit_if_win(odds):
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, o / 100.0, 100.0 / np.abs(o))


def roi(d):
    if len(d) == 0:
        return np.nan, 0, np.nan
    w = (d["actual"] < d["line"]).to_numpy()
    p = profit_if_win(d["under_odds"].to_numpy())
    u = np.where(w, p, -1.0)
    return float(u.mean()), len(d), float(w.mean())


def boot_ci(d, n_boot=2000, seed=0):
    """Clustered on game, because props in one game share an outcome."""
    if len(d) < 30:
        return np.nan, np.nan
    w = (d["actual"] < d["line"]).to_numpy()
    p = profit_if_win(d["under_odds"].to_numpy())
    units = np.where(w, p, -1.0)
    key = (d["season"].astype(str) + "_" + d["week"].astype(str) + "_"
           + d["team"].astype(str)).to_numpy()
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(k, []).append(i)
    idx = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(idx), len(idx))
        sel = np.concatenate([idx[i] for i in pick])
        out[b] = units[sel].mean()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lines_cache.parquet")
    args = ap.parse_args()

    hr("LOADING")
    C = pd.read_parquet(args.cache)
    C = C[(C["book"] == BOOK) & (C["market"].isin(MARKETS))].copy()
    if "snapshot_label" in C.columns:
        C = C[C["snapshot_label"] == "closing"]
    from models import data_utils
    seasons = sorted(C["season"].dropna().unique().astype(int).tolist())
    stats = data_utils.load_player_stats(seasons)
    stats = stats.to_pandas() if hasattr(stats, "to_pandas") else stats
    stats["_k"] = data_utils.norm_join_name(stats["player_display_name"])
    C["_k"] = data_utils.norm_join_name(C["player"])

    frames = []
    for mk in MARKETS:
        col = STAT_COL[mk]
        s = stats[["_k", "season", "week", "team", "position", col]].copy()
        g = C[C["market"] == mk].merge(s, on=["_k", "season", "week"],
                                       how="inner")
        frames.append(g.rename(columns={col: "actual"}))
    D = pd.concat(frames, ignore_index=True)
    D = D[D["actual"].notna() & D["line"].notna()
          & D["under_odds"].notna() & D["over_odds"].notna()].copy()

    sched = data_utils.load_schedules(seasons)
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    rows = []
    for side, opp in (("home_team", "away_team"), ("away_team", "home_team")):
        t = sched[["season", "week", side, opp]].copy()
        t.columns = ["season", "week", "team", "opponent_team"]
        rows.append(t)
    TG = pd.concat(rows, ignore_index=True)
    D = D.merge(TG, on=["season", "week", "team"], how="left")
    D = D[D["opponent_team"].notna()].reset_index(drop=True)
    print("  %d rows with an opponent" % len(D))

    r, n, w = roi(D)
    print("  pooled under ROI %+.4f on %d bets, win rate %.4f" % (r, n, w))

    hr("1. IS IT CLEVELAND, OR JUST THE BEST OF 32?")
    print("  The max statistic tested the BEST cell. Every other opponent")
    print("  is untested, so the honest claim is about the maximum rather")
    print("  than about a particular team.")
    print()
    tab = []
    for team, g in D.groupby("opponent_team"):
        rr, nn, ww = roi(g)
        tab.append((team, rr, nn, ww))
    tab.sort(key=lambda x: -x[1])
    print("  %-6s %9s %7s %9s %22s" % ("opp", "ROI", "bets", "win rate",
                                       "95% CI clustered"))
    print("  " + "-" * 60)
    for team, rr, nn, ww in tab[:6] + [("...", np.nan, 0, np.nan)] + tab[-4:]:
        if team == "...":
            print("  ...")
            continue
        lo, hi = boot_ci(D[D["opponent_team"] == team])
        ci = ("[%+.4f, %+.4f]" % (lo, hi)) if np.isfinite(lo) else ""
        print("  %-6s %+9.4f %7d %9.4f %22s" % (team, rr, nn, ww, ci))
    print()
    spread = tab[0][1] - tab[-1][1]
    print("  spread between best and worst opponent: %.4f" % spread)
    print("  With 32 cells of ~700 bets, a spread of this size is what the")
    print("  permutation already accounted for. The question below is")
    print("  whether the TOP cell has a mechanism, not whether it is top.")

    hr("2. IS IT A MECHANISM OR A LABEL? SEASON BY SEASON")
    print("  A franchise name is not a defence. If this tracks something")
    print("  real it should persist across coordinator and personnel")
    print("  changes; if it is a label it will be one or two seasons")
    print("  carrying the whole thing.")
    print()
    for team in FOCUS:
        g = D[D["opponent_team"] == team]
        print("  %s:" % team)
        for s in seasons:
            sub = g[g["season"] == s]
            rr, nn, ww = roi(sub)
            if nn == 0:
                continue
            print("    %d  ROI %+.4f  %4d bets  win %.4f"
                  % (s, rr, nn, ww))
        print()

    hr("3. MARKET CONCENTRATION: IS IT RULE 1 IN DISGUISE?")
    print("  Rule 1 is rushing at line <= 46.5, ROI +0.0589. If the CLE")
    print("  cell is mostly those rows, the two results are one result")
    print("  counted twice.")
    print()
    for team in FOCUS:
        g = D[D["opponent_team"] == team]
        print("  %s, by market:" % team)
        for mk, sub in g.groupby("market"):
            rr, nn, ww = roi(sub)
            print("    %-12s ROI %+.4f  %4d bets  win %.4f"
                  % (mk, rr, nn, ww))
        rush_low = g[(g["market"] == "rushing") & (g["line"] <= 46.5)]
        rr, nn, _ = roi(rush_low)
        print("    rushing <= 46.5 (Rule 1 overlap): %d bets, %.1f%% of cell"
              % (nn, 100.0 * nn / max(len(g), 1)))
        # the cell with Rule 1's rows REMOVED
        rest = g.drop(rush_low.index)
        rr2, nn2, ww2 = roi(rest)
        lo, hi = boot_ci(rest)
        print("    cell EXCLUDING Rule 1 rows: ROI %+.4f on %d bets"
              % (rr2, nn2))
        if np.isfinite(lo):
            print("      95%% CI [%+.4f, %+.4f]" % (lo, hi))
        print()

    hr("4. SELECTION: WHAT KIND OF PROPS ARE THESE?")
    print("  If the cell is full of low lines on thin players, the effect")
    print("  may be the line-level or coverage artifact already measured.")
    print()
    print("  %-6s %9s %9s %9s %9s %9s"
          % ("opp", "mean line", "med line", "mean odds", "mean tot",
             "n games"))
    print("  " + "-" * 60)
    allmean = D["line"].mean()
    for team in FOCUS:
        g = D[D["opponent_team"] == team]
        ngames = g.groupby(["season", "week", "team"]).ngroups
        print("  %-6s %9.2f %9.2f %9.1f %9s %9d"
              % (team, g["line"].mean(), g["line"].median(),
                 g["under_odds"].mean(), "-", ngames))
    print("  %-6s %9.2f %9.2f %9.1f" % ("ALL", allmean, D["line"].median(),
                                        D["under_odds"].mean()))
    print()
    print("  and the CLE cell's line distribution against everyone else,")
    print("  per market, since a line is only comparable within a market:")
    for mk in MARKETS:
        a = D[(D["opponent_team"] == "CLE") & (D["market"] == mk)]["line"]
        b = D[(D["opponent_team"] != "CLE") & (D["market"] == mk)]["line"]
        if len(a) < 20:
            continue
        print("    %-12s CLE mean %7.2f   others %7.2f   diff %+.2f"
              % (mk, a.mean(), b.mean(), a.mean() - b.mean()))

    hr("5. THE OBVIOUS CONFOUND: ARE THESE JUST GOOD DEFENCES?")
    print("  If the effect is 'good defences are underpriced', it should")
    print("  correlate with yards allowed. If it does not, the opponent")
    print("  label is standing in for something else.")
    print()
    # points/yards allowed per team-season, from the stat rows themselves
    allowed = (D.groupby(["opponent_team", "season"])
               .apply(lambda g: pd.Series({
                   "u_roi": roi(g)[0], "bets": roi(g)[1]}),
                      include_groups=False)
               .reset_index())
    ya = (D.groupby(["opponent_team", "season", "week", "team"])["actual"]
          .sum().reset_index()
          .groupby(["opponent_team", "season"])["actual"].mean()
          .rename("prop_yards_allowed").reset_index())
    m = allowed.merge(ya, on=["opponent_team", "season"])
    m = m[m["bets"] >= 80]
    if len(m) > 10:
        c = np.corrcoef(m["prop_yards_allowed"], m["u_roi"])[0, 1]
        print("  team-seasons with 80+ bets: %d" % len(m))
        print("  correlation(prop yards allowed, under ROI) = %+.4f" % c)
        print()
        print("  NEGATIVE would mean stingier defences give better unders,")
        print("  which is the mechanism. Near zero means the opponent")
        print("  label is not acting through defensive quality.")
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
