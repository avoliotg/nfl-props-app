"""opp_trend_test.py - does the LINE underreact to a defence's recent form?

READ ONLY. Writes nothing, changes no constant.

    python opp_trend_test.py --cache lines_cache.parquet

THE QUESTION, AND WHY IT IS NOT THE ONE ALREADY CLOSED

    recency_test.py measured FanDuel's weight on a PLAYER's last game
    against the weight that actually predicts his next one. They matched in
    all five markets, every interval covering zero, and zero of 40
    specifications cleared. So the line handles a player's own recent form
    correctly.

    Nothing has tested whether it handles the OPPONENT's recent form
    correctly. That is a different quantity and there is a reason to look:
    def_quality_test.py found a lagged, non-circular defensive-quality
    gradient of rho -0.9000, replicating in both periods, of which roughly
    HALF survived controlling for the line. If the line contained the whole
    of it, nothing should have survived.

THE REGIME-CHANGE FRAMING, which is the honest version of "that defence is
getting torched every week"

    A defence that loses two starting corners in week 5 is a different unit
    from week 6. You do not need 500 observations to believe that; you need
    the causal fact. A season-long average cannot see it, because it blends
    two different units.

    But "torched every week" is ALSO what three weeks of a normal defence
    looks like about one time in eight. With 32 teams you will see four or
    five apparent emerging trends every week and most will revert.

    So the measurable question is not "is this defence bad now". It is
    whether the LINE has already moved to reflect it. If a unit has been
    torched for three weeks and the lines against it have risen to match,
    the market has adjusted and there is nothing to take. If the lines have
    not moved, the market is pricing the old unit.

    That is testable on a large sample with no niche cells, which is the
    opposite of slicing by team and season.

TWO MEASURES, both lagged, neither able to see the game being scored

    baseline  the defence's mean allowance over ALL prior games this season
    recent    its mean over the last RECENT_N prior games
    surprise  recent minus baseline, standardised within market

    `surprise` is the regime-change proxy: positive means the defence has
    been worse lately than its own season to date.

PRE-REGISTERED

    O1. The line DOES respond to surprise. The correlation between
        surprise and the line set against that defence will be positive.
        Reason: books are not ignoring a defence falling apart.

    O2. The line responds INCOMPLETELY, so surprise still predicts the
        signed residual (outcome minus line) with a positive coefficient.
        This is the claim that would matter.

    O3. It does NOT convert to ROI. Prediction: under ROI by surprise
        quintile shows no ordering that clears a permutation null.
        Reason: def_quality_test found a perfect -0.90 ordering that still
        produced four losing bins out of five. An ordering is not an edge.

    ALIVE only if O2 holds AND a surprise-conditioned bet clears the
    permutation null at the actual price AND the holdout keeps the sign.
    O3 predicts failure at the second condition.
"""
import argparse
import sys

import numpy as np
import pandas as pd

BOOK = "fanduel"
MARKETS = ["receiving", "receptions", "rushing", "qb_passing"]
STAT_COL = {"receiving": "receiving_yards", "receptions": "receptions",
            "rushing": "rushing_yards", "qb_passing": "passing_yards"}
RECENT_N = 3
MIN_PRIOR = 4          # need a baseline longer than the recent window
N_QUINT = 5
N_PERM = 2000
MIN_CELL = 150
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
    return float(np.where(win, profit, -1.0).mean()) if len(win) else np.nan


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank().to_numpy() - (len(x) + 1) / 2
    ry = pd.Series(y).rank().to_numpy() - (len(y) + 1) / 2
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def ols_cluster(y, X, groups):
    """Coefficients with cluster-robust SEs. X includes no intercept."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        xg, ug = X[m], resid[m]
        s = xg.T @ ug
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    return beta, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--perm", type=int, default=N_PERM)
    args = ap.parse_args()

    hr("DOES THE LINE UNDERREACT TO A DEFENCE'S RECENT FORM?")
    print("  O1 the line DOES respond to a defence's recent surprise")
    print("  O2 it responds INCOMPLETELY, so surprise still predicts the")
    print("     signed residual. This is the claim that would matter.")
    print("  O3 it does NOT convert to ROI. def_quality_test found a")
    print("     perfect -0.90 ordering that still had four losing bins of")
    print("     five: an ordering is not an edge.")

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

    hr("1. BASELINE, RECENT, SURPRISE  (all lagged)")
    gm = (D.groupby(["opponent_team", "season", "week", "market"])["actual"]
          .sum().reset_index().rename(columns={"actual": "conceded"}))
    gm = gm.sort_values(["opponent_team", "market", "season", "week"])
    key = ["opponent_team", "market", "season"]
    g = gm.groupby(key)["conceded"]
    gm["baseline"] = g.transform(lambda s: s.shift(1).expanding().mean())
    gm["recent"] = g.transform(
        lambda s: s.shift(1).rolling(RECENT_N, min_periods=RECENT_N).mean())
    gm["prior_n"] = g.transform(lambda s: s.shift(1).expanding().count())
    gm = gm[(gm["prior_n"] >= MIN_PRIOR) & gm["recent"].notna()]
    gm["surprise_raw"] = gm["recent"] - gm["baseline"]
    print("  recent window %d games, baseline needs %d+ prior games"
          % (RECENT_N, MIN_PRIOR))
    print("  %d defence-season-week-market cells" % len(gm))

    D = D.merge(gm[key[:1] + ["season", "week", "market", "baseline",
                              "recent", "surprise_raw", "prior_n"]],
                on=["opponent_team", "season", "week", "market"],
                how="inner")
    # standardise WITHIN market: a 30-yard swing means something different
    # on qb_passing than on receptions
    D["surprise"] = (D.groupby("market")["surprise_raw"]
                     .transform(lambda s: (s - s.mean()) / s.std()))
    D = D[D["surprise"].notna()].reset_index(drop=True)
    print("  %d prop rows with a lagged surprise" % len(D))
    print("  surprise: mean %+.3f sd %.3f, min %+.2f max %+.2f"
          % (D["surprise"].mean(), D["surprise"].std(),
             D["surprise"].min(), D["surprise"].max()))

    win = (D["actual"] < D["line"]).to_numpy()
    profit = profit_if_win(D["under_odds"].to_numpy())
    print("  pooled under ROI %+.4f on %d bets" % (roi_of(win, profit),
                                                   len(D)))

    hr("2. O1: DOES THE LINE RESPOND AT ALL?")
    print("  Regressing the LINE on surprise, within market, so a positive")
    print("  coefficient means the book raises lines against a defence")
    print("  that has been worse lately than its own season to date.")
    print()
    print("  %-12s %10s %10s %8s %8s"
          % ("market", "coef", "SE", "t", "n"))
    print("  " + "-" * 52)
    for mk, g2 in D.groupby("market"):
        y = g2["line"].to_numpy(float)
        X = g2[["surprise"]].to_numpy(float)
        grp = (g2["season"].astype(str) + "_" + g2["week"].astype(str)
               ).to_numpy()
        b, se = ols_cluster(y, X, grp)
        print("  %-12s %+10.4f %10.4f %8.2f %8d"
              % (mk, b[1], se[1], b[1] / se[1] if se[1] else np.nan, len(g2)))

    hr("3. O2: DOES IT RESPOND ENOUGH?")
    print("  Regressing the SIGNED RESIDUAL (outcome minus line) on")
    print("  surprise. If the line already contained the whole of a")
    print("  defence's recent form, this coefficient should be ZERO.")
    print("  A positive coefficient means the line UNDERREACTS.")
    print()
    print("  %-12s %10s %10s %8s %8s"
          % ("market", "coef", "SE", "t", "n"))
    print("  " + "-" * 52)
    tstats = {}
    for mk, g2 in D.groupby("market"):
        y = (g2["actual"] - g2["line"]).to_numpy(float)
        X = g2[["surprise"]].to_numpy(float)
        grp = (g2["season"].astype(str) + "_" + g2["week"].astype(str)
               ).to_numpy()
        b, se = ols_cluster(y, X, grp)
        t = b[1] / se[1] if se[1] else np.nan
        tstats[mk] = t
        print("  %-12s %+10.4f %10.4f %8.2f %8d"
              % (mk, b[1], se[1], t, len(g2)))
    print()
    print("  O2 predicted POSITIVE coefficients. Markets with t above 2:")
    hits = [m for m, t in tstats.items() if np.isfinite(t) and t > 2]
    print("    %s" % (hits if hits else "none"))
    print("  NOTE: four markets tested, so a single t of 2.1 is not 0.05.")
    print("  Bonferroni across four gives a threshold of t about 2.5.")

    hr("4. O3: DOES IT CONVERT TO ROI?")
    D["sbin"] = (D.groupby("market")["surprise"]
                 .transform(lambda s: pd.qcut(s, N_QUINT, labels=False,
                                              duplicates="drop")))
    D = D[D["sbin"].notna()].reset_index(drop=True)
    D["sbin"] = D["sbin"].astype(int)
    win = (D["actual"] < D["line"]).to_numpy()
    profit = profit_if_win(D["under_odds"].to_numpy())

    def bins(w):
        out = []
        arr = D["sbin"].to_numpy()
        for q in range(N_QUINT):
            m = arr == q
            out.append(roi_of(w[m], profit[m]) if m.sum() >= MIN_CELL
                       else np.nan)
        return out

    obs_b = bins(win)
    print("  bin 0 = defence has been BETTER lately than its baseline")
    print("  bin 4 = defence has been WORSE lately")
    print()
    print("  %-6s %10s %8s %10s %12s"
          % ("bin", "under ROI", "bets", "win rate", "mean surprise"))
    print("  " + "-" * 52)
    for q in range(N_QUINT):
        m = D["sbin"].to_numpy() == q
        print("  %-6d %+10.4f %8d %10.4f %12.3f"
              % (q, obs_b[q], int(m.sum()), float(win[m].mean()),
                 float(D.loc[m, "surprise"].mean())))
    obs = spearman(range(N_QUINT), obs_b)
    print()
    print("  Spearman(bin, under ROI) = %+.4f" % obs)
    print("  If the line underreacted to a collapsing defence, the OVER")
    print("  would be the profitable side at high bins, so under ROI would")
    print("  FALL with the bin and this would be negative.")

    print()
    print("  %d permutations, win/loss shuffled globally" % args.perm)
    rng = np.random.default_rng(0)
    dist = np.empty(args.perm)
    for b in range(args.perm):
        dist[b] = spearman(range(N_QUINT), bins(rng.permutation(win)))
    p_lo = float(np.mean(dist <= obs))
    p_hi = float(np.mean(dist >= obs))
    print("  null sd %.4f" % dist.std())
    print("  one-sided p, negative tail %.4f   positive tail %.4f"
          % (p_lo, p_hi))
    print("  MDE: only |rho| beyond about %.3f is detectable at p<0.05"
          % abs(np.quantile(dist, 0.05)))
    print()
    best = max(range(N_QUINT), key=lambda q: obs_b[q]
               if np.isfinite(obs_b[q]) else -9)
    print("  best single bin: %d at %+.4f" % (best, obs_b[best]))
    print("  (five bins, so this is a best-of-five and not a p-value)")

    hr("5. HOLDOUT")
    for label, ss in (("dev %s" % DEV, DEV), ("holdout %s" % HOLD, HOLD)):
        m = D["season"].isin(ss).to_numpy()
        sub = D[m]
        w = win[m]
        pr = profit[m]
        vals = []
        for q in range(N_QUINT):
            mm = sub["sbin"].to_numpy() == q
            vals.append(roi_of(w[mm], pr[mm]) if mm.sum() >= 60 else np.nan)
        print("  %-18s rho %+.4f on %d bets"
              % (label, spearman(range(N_QUINT), vals), len(sub)))
        print("      %s" % " ".join(("%+.4f" % v) if np.isfinite(v) else "  -"
                                    for v in vals))

    hr("VERDICT")
    print("  O1 the line responds: see section 2")
    print("  O2 it underreacts:    see section 3, Bonferroni t about 2.5")
    print("  O3 no ROI conversion: rho %+.4f, p %.4f / %.4f"
          % (obs, p_lo, p_hi))
    if min(p_lo, p_hi) >= 0.05:
        print()
        print("  NULL on the ROI test. O3 predicted this. An ordering in")
        print("  the residual can be real while every bin still loses,")
        print("  which is exactly what def_quality_test showed.")
    else:
        print()
        print("  THE ROI ORDERING CLEARS. Check the holdout above and the")
        print("  bin levels: an ordering among losing bins is not a bet.")
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
