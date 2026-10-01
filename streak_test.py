"""streak_test.py - do prop streaks against a defence continue or revert?

READ ONLY. Writes nothing, changes no constant.

    python streak_test.py --cache lines_cache.parquet

THE REFRAME THAT SOLVES THE POWER PROBLEM

    "Cleveland in 2024" is one cell, about 60 props, and no amount of care
    gives it power. But "a defence whose last three games of receiving
    props all went over" is a TEMPLATE. It occurs hundreds of times across
    32 teams and four seasons. Each occurrence is weak; pooled, the
    template is thousands of props.

    So the question stops being "is this team on a trend" and becomes "do
    trends of this SHAPE continue". That is one well-powered test of a
    general claim, which is the "testable across seasons to become
    statistically powerful" idea stated formally.

WHY THIS IS NOT THE TEST THAT JUST FAILED

    opp_trend_test.py used a rolling MEAN of yards allowed. A mean averages
    a regime change away: a defence that was elite and then collapsed shows
    a middling average. A PATTERN does not. "Three consecutive games where
    the props went over" either happened or it did not, and it survives the
    averaging that destroyed the previous measure.

    It is also measured in the unit we bet in. Not yards allowed, but
    whether props CLEARED. "The last three receiving props against this
    defence went over" is directly bet-relevant in a way that yards
    allowed is not.

HOW A GAME IS CLASSIFIED

    For each defence, market and game: the share of that game's props that
    went OVER. The game is an "over game" if that share exceeds 0.5, an
    "under game" if below, and is skipped at exactly 0.5 so a tie cannot
    extend a streak in either direction. Games with fewer than MIN_PROPS
    props are skipped as too thin to classify.

THE TEMPLATE FAMILY, enumerated before looking

    streak length   2, 3, 4, 5, 6+       (6 or more pooled: the sample
                                          roughly halves per extra game)
    direction       consecutive overs, consecutive unders
    market          receiving, receptions, rushing, qb_passing

    40 templates. For each, the test is what happens on the NEXT game's
    props: continuation or reversion, as under ROI at the actual price.

    THE MONOTONICITY TEST IS THE POWERFUL ONE. Any single template is a
    modest sample, but if streaks mean anything the effect should GROW with
    streak length, and a rank correlation across lengths pools their
    evidence into one statistic. A single good cell among 40 is a
    best-of-40; an ordered gradient across lengths is not.

PRE-REGISTERED

    S1. REVERSION, not continuation. After a run of overs, the under is
        the better side. Reason: a streak of overs means the lines were
        too low, and books move lines; by the time a streak is visible the
        correction is in the price and may overshoot.

    S2. NO MONOTONE GRADIENT in streak length. Prediction: the rank
        correlation between streak length and under ROI after over-streaks
        does not clear a permutation null. Reason: everything measured
        today that looked like a gradient either was the line in disguise
        or produced only losing bins.

    S3. The 6+ cells will be too thin to interpret and will show the
        largest absolute ROI in the whole table, in whichever direction
        chance picks. Stated in advance so a big number there is read as
        noise rather than as the strongest signal.

    ALIVE only if: the max-statistic permutation over all 40 templates
    clears 0.05, AND the surviving template's holdout keeps the sign, AND
    it has 150+ next-game props in both periods.
"""
import argparse
import sys

import numpy as np
import pandas as pd

BOOK = "fanduel"
MARKETS = ["receiving", "receptions", "rushing", "qb_passing"]
STAT_COL = {"receiving": "receiving_yards", "receptions": "receptions",
            "rushing": "rushing_yards", "qb_passing": "passing_yards"}
MIN_PROPS = 3           # props needed to classify a game
LENGTHS = [2, 3, 4, 5, 6]   # 6 means "6 or more"
N_PERM = 2000
MIN_CELL = 60
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--cross-season", action="store_true",
                    help="let a streak carry across the offseason. OFF by "
                         "default: a franchise name is not a defence, and "
                         "diag_cle_cell flagged exactly that.")
    args = ap.parse_args()

    hr("DO PROP STREAKS AGAINST A DEFENCE CONTINUE OR REVERT?")
    print("  S1 predicts REVERSION: after a run of overs the under is the")
    print("     better side, because a streak of overs means lines were")
    print("     too low and books move lines")
    print("  S2 predicts NO monotone gradient in streak length")
    print("  S3 predicts the 6+ cells show the largest absolute ROI in the")
    print("     table and are too thin to mean anything. Stated now so a")
    print("     big number there is read as noise.")
    print()
    print("  streak scope: %s"
          % ("carries across seasons" if args.cross_season
             else "within season only"))

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
    D["over"] = (D["actual"] > D["line"]).astype(int)
    D["under"] = (D["actual"] < D["line"]).astype(int)
    print("  %d prop rows with an opponent" % len(D))

    hr("1. CLASSIFYING EACH DEFENCE-GAME")
    gm = (D.groupby(["opponent_team", "market", "season", "week"])
          .agg(n_props=("over", "size"), n_over=("over", "sum"),
               n_under=("under", "sum")).reset_index())
    gm = gm[gm["n_props"] >= MIN_PROPS].copy()
    gm["over_share"] = gm["n_over"] / gm["n_props"]
    gm["kind"] = np.where(gm["over_share"] > 0.5, "over",
                          np.where(gm["over_share"] < 0.5, "under", "tie"))
    print("  %d defence-game-market cells with %d+ props"
          % (len(gm), MIN_PROPS))
    print("  classified: %s" % gm["kind"].value_counts().to_dict())
    print("  ties are SKIPPED rather than counted either way, so a 50/50")
    print("  game cannot extend a streak in either direction")

    # streak entering each game, computed from PRIOR games only
    grp = (["opponent_team", "market"] if args.cross_season
           else ["opponent_team", "market", "season"])
    gm = gm.sort_values(grp + ["season", "week"])
    gm["run_over"] = 0
    gm["run_under"] = 0
    for _, idx in gm.groupby(grp).groups.items():
        ro = ru = 0
        for i in idx:
            gm.at[i, "run_over"] = ro
            gm.at[i, "run_under"] = ru
            k = gm.at[i, "kind"]
            if k == "over":
                ro, ru = ro + 1, 0
            elif k == "under":
                ro, ru = 0, ru + 1
            # a tie leaves both runs untouched
    print()
    print("  entering-streak distribution (over runs):")
    print("    %s" % gm["run_over"].value_counts().sort_index().head(9)
          .to_dict())
    print("  entering-streak distribution (under runs):")
    print("    %s" % gm["run_under"].value_counts().sort_index().head(9)
          .to_dict())

    # attach the entering streak to each prop in that game
    D = D.merge(gm[["opponent_team", "market", "season", "week",
                    "run_over", "run_under"]],
                on=["opponent_team", "market", "season", "week"],
                how="inner")
    print()
    print("  %d props sit in a classifiable game" % len(D))

    win = D["under"].to_numpy().astype(bool)
    profit = profit_if_win(D["under_odds"].to_numpy())
    print("  pooled under ROI %+.4f on %d bets" % (roi_of(win, profit),
                                                   len(D)))

    hr("2. THE TEMPLATE FAMILY")
    templates = []
    for mk in MARKETS:
        for L in LENGTHS:
            for direction in ("over", "under"):
                col = "run_over" if direction == "over" else "run_under"
                if L < max(LENGTHS):
                    m = ((D["market"] == mk) & (D[col] == L)).to_numpy()
                    lab = "%d" % L
                else:
                    m = ((D["market"] == mk) & (D[col] >= L)).to_numpy()
                    lab = "%d+" % L
                templates.append((mk, direction, lab, L, m))
    print("  %d templates: %d markets x %d lengths x 2 directions"
          % (len(templates), len(MARKETS), len(LENGTHS)))

    def table(w):
        return np.array([roi_of(w[m], profit[m]) if m.sum() >= MIN_CELL
                         else np.nan for *_, m in templates])

    obs = table(win)
    hr("3. THE TABLE  (uncorrected, so not evidence)")
    print("  'after N consecutive OVER games, bet the UNDER' is S1's claim")
    print()
    print("  %-12s %-6s %-5s %10s %7s %9s"
          % ("market", "after", "len", "under ROI", "bets", "win rate"))
    print("  " + "-" * 56)
    for i, (mk, d, lab, L, m) in enumerate(templates):
        n = int(m.sum())
        if n == 0:
            continue
        flag = "" if n >= MIN_CELL else "  thin"
        r = obs[i]
        rs = ("%+10.4f" % r) if np.isfinite(r) else "         -"
        print("  %-12s %-6s %-5s %s %7d %9s%s"
              % (mk, d, lab, rs, n,
                 ("%.4f" % float(win[m].mean())) if n else "-", flag))

    usable = np.isfinite(obs)
    print()
    print("  %d of %d templates clear the %d-bet floor"
          % (int(usable.sum()), len(templates), MIN_CELL))
    if not usable.any():
        print("  nothing to test")
        sys.exit(0)
    omax = float(np.nanmax(obs[usable]))
    imax = int(np.nanargmax(np.where(usable, obs, -np.inf)))
    print("  best template: %s after %s %s, ROI %+.4f on %d bets"
          % (templates[imax][0], templates[imax][2], templates[imax][1],
             omax, int(templates[imax][4].sum())))

    hr("4. MAX-STATISTIC PERMUTATION OVER ALL TEMPLATES")
    print("  %d permutations, win/loss shuffled globally with every prop"
          % args.perm)
    print("  keeping its own template memberships and its own price")
    rng = np.random.default_rng(0)
    maxes = np.empty(args.perm)
    for b in range(args.perm):
        t = table(rng.permutation(win))
        maxes[b] = np.nanmax(t[np.isfinite(t)])
        if (b + 1) % 500 == 0:
            print("\r    %d of %d" % (b + 1, args.perm), end="", flush=True)
    print()
    p = float(np.mean(maxes >= omax))
    print()
    print("  null maximum: median %+.4f  95th %+.4f  sd %.4f"
          % (np.median(maxes), np.quantile(maxes, 0.95), maxes.std()))
    print("  observed %+.4f -> p = %.4f   %s"
          % (omax, p, "CLEARS" if p < 0.05 else "does not clear"))
    print()
    print("  the null maximum is the yardstick: with %d templates, chance"
          % int(usable.sum()))
    print("  alone produces a best cell of about %+.4f." % np.median(maxes))

    hr("5. S2: THE MONOTONICITY TEST, which pools the lengths")
    print("  If streaks mean anything the effect should GROW with length.")
    print("  A rank correlation across lengths pools their evidence into")
    print("  one statistic, which a best-of-40 cannot do.")
    print()
    for mk in MARKETS:
        for d in ("over", "under"):
            rs, ls = [], []
            for i, (m2, d2, lab, L, m) in enumerate(templates):
                if m2 == mk and d2 == d and np.isfinite(obs[i]):
                    rs.append(obs[i])
                    ls.append(L)
            if len(rs) < 3:
                continue
            rho = spearman(ls, rs)
            print("  %-12s after %-6s rho %+.4f over %d lengths"
                  % (mk, d, rho, len(rs)))

    hr("6. HOLDOUT ON THE BEST TEMPLATE")
    mk, d, lab, L, m = templates[imax]
    sub = D[m]
    for label, ss in (("dev %s" % DEV, DEV), ("holdout %s" % HOLD, HOLD)):
        s2 = sub[sub["season"].isin(ss)]
        if len(s2) == 0:
            print("  %-18s no rows" % label)
            continue
        w = s2["under"].to_numpy().astype(bool)
        pr = profit_if_win(s2["under_odds"].to_numpy())
        print("  %-18s ROI %+.4f on %d bets  %s"
              % (label, roi_of(w, pr), len(s2),
                 "" if len(s2) >= 150 else "<- under the 150 floor"))

    hr("VERDICT")
    print("  S1 reversion: read the sign of the 'after over' rows above")
    print("  S2 no gradient: section 5")
    print("  S3 thin 6+ cells: section 3")
    print()
    if p >= 0.05:
        print("  NULL for the family. The best of %d templates is %+.4f and"
              % (int(usable.sum()), omax))
        print("  chance's best on this same data is %+.4f."
              % np.median(maxes))
        print()
        print("  Streak templates, as defined here, do not survive their")
        print("  own multiplicity. That is a result about the SHAPE of")
        print("  trend, not about one team, which is what makes it")
        print("  generalisable.")
    else:
        print("  THE FAMILY MAXIMUM CLEARS. Check section 6's holdout and")
        print("  the bet counts before treating it as a candidate.")
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
