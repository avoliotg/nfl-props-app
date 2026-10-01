"""def_quality_test.py - does LAGGED defensive quality predict under ROI?

READ ONLY. Writes nothing, changes no constant.

    python def_quality_test.py --cache lines_cache.parquet

WHY THIS EXISTS, AND THE DEFECT IT FIXES

    trend_scan.py found that props against Cleveland returned under ROI
    +0.0944 on 685 bets, p 0.0230 after correcting for all 52 cells, with
    the holdout conditions holding. diag_cle_cell.py then found something
    better than a team effect: the correlation between a defence's prop
    yards allowed and its under ROI was -0.7751 across 96 team-seasons. A
    smooth league-wide relationship, with Cleveland merely at one end and
    Indianapolis at the other.

    BUT THAT -0.7751 IS PARTLY DEFINITIONAL. "Prop yards allowed" was
    computed from the same `actual` values that decide whether the under
    won. A defence that allowed few yards mechanically produces unders
    that cashed. Some unknown share of that correlation is arithmetic
    rather than prediction.

    This script removes the circularity: defensive quality is measured
    from PRIOR games only, never from the game being scored.

PRE-REGISTERED, and honestly labelled

    The hypothesis came FROM this data, so a test on the same data cannot
    be confirmatory. What it can do is establish whether the effect
    survives de-circularisation at all, which is a necessary condition. The
    chronological holdout is the closest thing to confirmation available
    without waiting for new seasons.

    D1. The lagged correlation is NEGATIVE but materially WEAKER than
        -0.7751. Prediction: between -0.15 and -0.45. Reason: removing the
        definitional component must shrink it, and a market that prices
        defence at all should absorb most of the rest.

    D2. It SURVIVES as nonzero. Prediction: the quintile ordering has a
        permutation p below 0.05. This is the necessary condition.

    D3. IT DOES NOT SURVIVE CONTROLLING FOR THE LINE. Prediction: within
        line bands, the gradient shrinks sharply. Reason: a stingy defence
        produces lower lines, so defensive quality and line level are
        entangled exactly as coverage and line were in coverage_test.py,
        and the line is the market's own encoding of the matchup.

    D4. The holdout keeps the sign. Lower confidence than D2.

    ALIVE only if: permutation p below 0.05 on the full sample, the
    gradient survives within line bands, and the holdout keeps the sign.
    D3 predicts this fails at the second condition, which would mean the
    opponent effect is the line effect wearing a defensive label.

HOW QUALITY IS MEASURED

    For each defence, in each season, at each week: the mean per-game total
    of the prop-relevant stat allowed to opposing players, over that
    defence's PRIOR games in that season only. Minimum three prior games,
    so weeks 1 to 3 are dropped every season.

    Same season only, deliberately. Carrying across the offseason would
    mix personnel and coordinator changes into the measure, which is the
    very thing diag_cle_cell flagged about team labels.
"""
import argparse
import sys

import numpy as np
import pandas as pd

BOOK = "fanduel"
MARKETS = ["receiving", "receptions", "rushing", "qb_passing"]
STAT_COL = {"receiving": "receiving_yards", "receptions": "receptions",
            "rushing": "rushing_yards", "qb_passing": "passing_yards"}
MIN_PRIOR_GAMES = 3
N_QUINTILES = 5
N_PERM = 2000
MIN_BETS = 150
DEV = [2023, 2024]
HOLD = [2025, 2026]


def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def profit_if_win(odds):
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, o / 100.0, 100.0 / np.abs(o))


def roi_of(win, profit):
    if len(win) == 0:
        return np.nan
    return float(np.where(win, profit, -1.0).mean())


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def quintile_rois(df, win, profit, qcol="dq_bin", levels=None):
    levels = levels if levels is not None else sorted(
        [v for v in df[qcol].dropna().unique()])
    out = []
    arr = df[qcol].to_numpy()
    for lv in levels:
        m = arr == lv
        n = int(m.sum())
        out.append((lv, roi_of(win[m], profit[m]) if n >= MIN_BETS else np.nan,
                    n, float(win[m].mean()) if n else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--perm", type=int, default=N_PERM)
    args = ap.parse_args()

    hr("LAGGED DEFENSIVE QUALITY vs UNDER PROFITABILITY")
    print("  D1 predicts the lagged correlation is negative but much")
    print("     weaker than the circular -0.7751, between -0.15 and -0.45")
    print("  D2 predicts it survives as nonzero, permutation p < 0.05")
    print("  D3 predicts it does NOT survive controlling for the LINE,")
    print("     because a stingy defence produces lower lines and the line")
    print("     is the market's own encoding of the matchup")
    print("  D4 predicts the holdout keeps the sign")

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
        s = stats[["_k", "season", "week", "team", col]].copy()
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
    print("  %d prop rows with an opponent" % len(D))

    hr("1. BUILDING LAGGED DEFENSIVE QUALITY")
    print("  Per defence, per season, per week: mean per-game total of the")
    print("  prop-relevant stat allowed, over that defence's PRIOR games in")
    print("  that season only. Minimum %d prior games." % MIN_PRIOR_GAMES)
    print()
    print("  THIS IS THE WHOLE POINT. diag_cle_cell computed quality from")
    print("  the SAME rows it scored, so a defence that allowed few yards")
    print("  mechanically produced unders that cashed. Part of that")
    print("  -0.7751 was arithmetic rather than prediction.")

    # per defence-season-week, the total conceded in THAT game, per market
    gm = (D.groupby(["opponent_team", "season", "week", "market"])["actual"]
          .sum().reset_index()
          .rename(columns={"actual": "conceded"}))
    gm = gm.sort_values(["opponent_team", "market", "season", "week"])
    # expanding mean over PRIOR games only
    g = gm.groupby(["opponent_team", "market", "season"])["conceded"]
    gm["prior_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    gm["prior_n"] = g.transform(lambda s: s.shift(1).expanding().count())
    gm = gm[gm["prior_n"] >= MIN_PRIOR_GAMES]
    print()
    print("  %d defence-season-week-market cells with %d+ prior games"
          % (len(gm), MIN_PRIOR_GAMES))

    D = D.merge(gm[["opponent_team", "season", "week", "market",
                    "prior_mean", "prior_n"]],
                on=["opponent_team", "season", "week", "market"], how="inner")
    print("  %d prop rows survive the lag requirement" % len(D))
    print("  (weeks 1-%d of every season are necessarily dropped)"
          % MIN_PRIOR_GAMES)

    # quality is only comparable WITHIN a market, so quintiles are per market
    D["dq_bin"] = (D.groupby("market")["prior_mean"]
                   .transform(lambda s: pd.qcut(s, N_QUINTILES,
                                                labels=False,
                                                duplicates="drop")))
    D = D[D["dq_bin"].notna()].reset_index(drop=True)
    D["dq_bin"] = D["dq_bin"].astype(int)
    print("  quintiles assigned within market; bin 0 = stingiest")

    win = (D["actual"] < D["line"]).to_numpy()
    profit = profit_if_win(D["under_odds"].to_numpy())
    print()
    print("  pooled under ROI on this population %+.4f on %d bets"
          % (roi_of(win, profit), len(D)))

    hr("2. D1 AND D2: THE LAGGED GRADIENT")
    print("  %-6s %10s %8s %10s %12s"
          % ("bin", "ROI", "bets", "win rate", "mean allowed"))
    print("  " + "-" * 52)
    rows_q = quintile_rois(D, win, profit)
    for lv, r, n, w in rows_q:
        ma = D.loc[D["dq_bin"] == lv, "prior_mean"].mean()
        print("  %-6d %+10.4f %8d %10.4f %12.1f" % (lv, r, n, w, ma))
    obs = spearman([r[0] for r in rows_q if np.isfinite(r[1])],
                   [r[1] for r in rows_q if np.isfinite(r[1])])
    print()
    print("  Spearman(bin, under ROI) = %+.4f" % obs)
    print("  bin 0 is stingiest, so NEGATIVE means stingy defences give")
    print("  better unders, which is the mechanism.")
    print()
    print("  D1 range was -0.15 to -0.45: %s"
          % ("inside" if -0.45 <= obs <= -0.15 else "OUTSIDE"))

    print()
    print("  %d permutations, win/loss shuffled globally" % args.perm)
    rng = np.random.default_rng(0)
    dist = np.empty(args.perm)
    for b in range(args.perm):
        w = rng.permutation(win)
        rq = quintile_rois(D, w, profit)
        dist[b] = spearman([r[0] for r in rq if np.isfinite(r[1])],
                           [r[1] for r in rq if np.isfinite(r[1])])
    p = float(np.mean(dist <= obs))
    print("  null: mean %+.4f  sd %.4f" % (dist.mean(), dist.std()))
    print("  one-sided p = %.4f   %s" % (p, "CLEARS" if p < 0.05
                                         else "does not clear"))
    print("  MDE: detectable at p<0.05 only below rho %+.4f"
          % np.quantile(dist, 0.05))

    hr("3. D3: DOES IT SURVIVE CONTROLLING FOR THE LINE?")
    print("  A stingy defence produces LOWER lines, so defensive quality")
    print("  and line level are entangled exactly as coverage and line")
    print("  were in coverage_test.py. The line is the market's own")
    print("  encoding of the matchup, so this is the decisive control.")
    print()
    D["line_band"] = (D.groupby("market")["line"]
                      .transform(lambda s: pd.qcut(s, 4, labels=False,
                                                   duplicates="drop")))
    corr = D[["prior_mean", "line"]].corr().iloc[0, 1]
    print("  correlation(prior_mean, line) = %+.4f  (entanglement)" % corr)
    print()
    print("  under ROI by defensive quintile WITHIN each line quartile:")
    print("  %-12s %9s %9s %9s %9s %9s"
          % ("line band", "dq 0", "dq 1", "dq 2", "dq 3", "dq 4"))
    print("  " + "-" * 62)
    within = []
    for lb in sorted(D["line_band"].dropna().unique()):
        sub = D["line_band"].to_numpy() == lb
        cells = []
        for q in range(N_QUINTILES):
            m = sub & (D["dq_bin"].to_numpy() == q)
            n = int(m.sum())
            cells.append(roi_of(win[m], profit[m]) if n >= 60 else np.nan)
        rho = spearman(range(N_QUINTILES), cells)
        within.append(rho)
        print("  %-12s %s  rho %+.3f"
              % ("q%d" % int(lb),
                 " ".join(("%+9.4f" % c) if np.isfinite(c) else "        -"
                          for c in cells), rho))
    good = [r for r in within if np.isfinite(r)]
    print()
    print("  mean within-band rho %+.4f against pooled %+.4f"
          % (np.mean(good) if good else np.nan, obs))
    if good and np.mean(good) > obs + 0.15:
        print("  THE GRADIENT SHRINKS inside line bands, so much of the")
        print("  opponent effect is the LINE effect wearing a defensive")
        print("  label. D3 predicted this.")
    elif good:
        print("  the gradient largely SURVIVES the line control, so")
        print("  defensive quality is carrying information the line does")
        print("  not. D3 predicted otherwise.")

    hr("4. D4: HOLDOUT")
    for label, ss in (("dev %s" % DEV, DEV), ("holdout %s" % HOLD, HOLD)):
        m = D["season"].isin(ss).to_numpy()
        sub = D[m].reset_index(drop=True)
        rq = quintile_rois(sub, win[m], profit[m])
        rho = spearman([r[0] for r in rq if np.isfinite(r[1])],
                       [r[1] for r in rq if np.isfinite(r[1])])
        print("  %-18s rho %+.4f on %d bets" % (label, rho, len(sub)))
        for lv, r, n, w in rq:
            print("      bin %d  ROI %+.4f  %5d bets" % (lv, r, n))
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
