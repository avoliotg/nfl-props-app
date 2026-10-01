"""streak_cond_test.py - do streaks mean something CONDITIONALLY?

READ ONLY. Writes nothing, changes no constant.

    python streak_cond_test.py --mode quality --cache lines_cache.parquet
    python streak_cond_test.py --mode bye     --cache lines_cache.parquet

WHAT ALREADY FAILED. streak_test.py tested 40 unconditional streak
templates. The best was +0.1206 on 115 bets; chance's best on the same data
was +0.1176, p 0.4735. Monotonicity across lengths gave -0.40, +0.80, +0.80,
-0.60, which is incoherent rather than weak. And the streak-length
distribution halved almost exactly, 2432 / 968 / 425 / 194 / 92 / 39 / 14 /
3, which is what independent coin flips produce.

So blind streak betting is noise. These two modes attack the assumption
that a streak means the same thing in every context.

MODE 'quality': STRUCTURAL versus RANDOM
    Theory: a good defence's bad run is variance. A structurally broken
    defence, say one that has lost its only competent run-stopper, gets
    exploited predictably until the personnel changes. Pooling the two
    washes out the second.

    Condition: the WORST quartile of lagged defensive quality, measured
    from prior games only, exactly as def_quality_test.py built it. That
    measure produced a real, non-circular rho of -0.9000 replicating in
    both periods, so it is a tested instrument rather than a new guess.

MODE 'bye': INTERRUPTED versus UNINTERRUPTED
    Theory: streaks are driven by temporary factors, a nagging injury or a
    coordinator's rut. A bye week is a hard reset for health and
    game-planning.

    Condition: did the DEFENCE have a bye immediately before this game.
    Prediction is specific and falsifiable: if streaks carry any
    information, it should collapse after a bye.

    Detected from the schedule as a gap of 2+ weeks since that defence's
    previous game, which is a pregame fact.

TWO THINGS THE ORIGINAL SUGGESTIONS GOT WRONG, worth recording

    OPENING LINES CANNOT BE TESTED HERE. historical_lines holds ONE closing
    price per prop per book, and its captured_at is DERIVED: six values per
    week, each a constant 44.4 minutes before its kickoff cluster. The
    `lines` table has genuine multi-capture paths but only from week 3 of
    2026. So "streaky on the field, efficient in the market" is the most
    plausible theory of the four and is a year of data away, not an
    afternoon.

    FOURTH-QUARTER MARGIN IS NOT A PREGAME FACT. Filtering to games still
    within one score entering the fourth quarter conditions on an in-game
    state, so any rule built on it could not be acted on. The salvageable
    version is the pregame SPREAD, which trend_scan.py already carried as a
    dimension and which did not surface.

MULTIPLICITY, stated before the run
    Two modes is two families. Threshold is 0.025 EACH, not 0.05. Both
    results are reported regardless of outcome.

PRE-REGISTERED
    C1 (quality). The worst-quartile restriction does NOT rescue the
       templates. Reason: restricting to a quarter of the data quarters
       the cell sizes, so the null maximum widens faster than any real
       effect could emerge.
    C2 (quality). If anything appears it will be in rushing, because a
       broken run defence is the most mechanically plausible version of
       the structural story.
    C3 (bye). Streak ROI after a bye will be indistinguishable from
       streak ROI without one. Reason: streak_test found the streaks
       themselves to be coin flips, and interrupting a coin flip changes
       nothing.
    C4 (bye). The post-bye cells will be thin enough that most fall below
       the floor, and the ones that survive will show large absolute
       numbers in whichever direction chance picks.
"""
import argparse
import sys

import numpy as np
import pandas as pd

BOOK = "fanduel"
MARKETS = ["receiving", "receptions", "rushing", "qb_passing"]
STAT_COL = {"receiving": "receiving_yards", "receptions": "receptions",
            "rushing": "rushing_yards", "qb_passing": "passing_yards"}
MIN_PROPS = 3
LENGTHS = [2, 3, 4]        # 4 means "4 or more": conditioning costs sample
MIN_PRIOR = 3
N_PERM = 2000
MIN_CELL = 50
ALPHA = 0.025              # two families
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


def load(cache):
    C = pd.read_parquet(cache)
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
    return D, sched


def add_streaks(D):
    gm = (D.groupby(["opponent_team", "market", "season", "week"])
          .agg(n_props=("over", "size"), n_over=("over", "sum"))
          .reset_index())
    gm = gm[gm["n_props"] >= MIN_PROPS].copy()
    gm["share"] = gm["n_over"] / gm["n_props"]
    gm["kind"] = np.where(gm["share"] > 0.5, "over",
                          np.where(gm["share"] < 0.5, "under", "tie"))
    gm = gm.sort_values(["opponent_team", "market", "season", "week"])
    gm["run_over"] = 0
    gm["run_under"] = 0
    for _, idx in gm.groupby(["opponent_team", "market", "season"]).groups \
            .items():
        ro = ru = 0
        for i in idx:
            gm.at[i, "run_over"] = ro
            gm.at[i, "run_under"] = ru
            k = gm.at[i, "kind"]
            if k == "over":
                ro, ru = ro + 1, 0
            elif k == "under":
                ro, ru = 0, ru + 1
    return D.merge(gm[["opponent_team", "market", "season", "week",
                       "run_over", "run_under"]],
                   on=["opponent_team", "market", "season", "week"],
                   how="inner")


def add_quality(D):
    """Lagged defensive quality, exactly as def_quality_test built it."""
    gm = (D.groupby(["opponent_team", "season", "week", "market"])["actual"]
          .sum().reset_index().rename(columns={"actual": "conceded"}))
    gm = gm.sort_values(["opponent_team", "market", "season", "week"])
    g = gm.groupby(["opponent_team", "market", "season"])["conceded"]
    gm["prior_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    gm["prior_n"] = g.transform(lambda s: s.shift(1).expanding().count())
    gm = gm[gm["prior_n"] >= MIN_PRIOR]
    D = D.merge(gm[["opponent_team", "season", "week", "market",
                    "prior_mean"]],
                on=["opponent_team", "season", "week", "market"], how="inner")
    # quartiles WITHIN market; 3 = most yards allowed = worst defence
    D["dq"] = (D.groupby("market")["prior_mean"]
               .transform(lambda s: pd.qcut(s, 4, labels=False,
                                            duplicates="drop")))
    return D[D["dq"].notna()].reset_index(drop=True)


def add_bye(D, sched):
    """Did the DEFENCE have a bye immediately before this game.

    A pregame fact: detected as a gap of 2+ weeks since that defence's
    previous game in the same season.
    """
    rows = []
    for side in ("home_team", "away_team"):
        t = sched[["season", "week", side]].copy()
        t.columns = ["season", "week", "team"]
        rows.append(t)
    G = pd.concat(rows, ignore_index=True).drop_duplicates()
    G = G.sort_values(["team", "season", "week"])
    G["prev_week"] = G.groupby(["team", "season"])["week"].shift(1)
    G["gap"] = G["week"] - G["prev_week"]
    G["post_bye"] = np.where(G["gap"].notna() & (G["gap"] >= 2),
                             "post bye", "normal")
    G = G.rename(columns={"team": "opponent_team"})
    return D.merge(G[["opponent_team", "season", "week", "post_bye"]],
                   on=["opponent_team", "season", "week"], how="left")


def run_family(D, templates, win, profit, perm, label):
    def table(w):
        return np.array([roi_of(w[m], profit[m]) if m.sum() >= MIN_CELL
                         else np.nan for *_, m in templates])
    obs = table(win)
    usable = np.isfinite(obs)
    print()
    print("  %-26s %-10s %10s %7s %9s"
          % ("template", "cond", "under ROI", "bets", "win rate"))
    print("  " + "-" * 66)
    for i, t in enumerate(templates):
        name, cond, m = t[0], t[1], t[-1]
        n = int(m.sum())
        if n == 0:
            continue
        rs = ("%+10.4f" % obs[i]) if np.isfinite(obs[i]) else "         -"
        print("  %-26s %-10s %s %7d %9s%s"
              % (name, cond, rs, n,
                 "%.4f" % float(win[m].mean()),
                 "" if n >= MIN_CELL else "  thin"))
    if not usable.any():
        print()
        print("  NO template clears the %d-bet floor. The conditioning\n"
              "  has cut the sample below what any test can use, which is\n"
              "  itself the answer." % MIN_CELL)
        return None
    omax = float(np.nanmax(obs[usable]))
    imax = int(np.nanargmax(np.where(usable, obs, -np.inf)))
    print()
    print("  %d of %d clear the floor" % (int(usable.sum()), len(templates)))
    print("  best: %s / %s at %+.4f on %d bets"
          % (templates[imax][0], templates[imax][1], omax,
             int(templates[imax][-1].sum())))
    rng = np.random.default_rng(0)
    maxes = np.empty(perm)
    for b in range(perm):
        t = table(rng.permutation(win))
        u = np.isfinite(t)
        maxes[b] = np.nanmax(t[u]) if u.any() else np.nan
    p = float(np.mean(maxes >= omax))
    print()
    print("  null maximum: median %+.4f  95th %+.4f"
          % (np.nanmedian(maxes), np.nanquantile(maxes, 0.95)))
    print("  observed %+.4f -> p = %.4f  vs threshold %.3f: %s"
          % (omax, p, ALPHA, "CLEARS" if p < ALPHA else "does not clear"))
    return templates[imax], omax, p, np.nanmedian(maxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["quality", "bye"])
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--perm", type=int, default=N_PERM)
    args = ap.parse_args()

    hr("CONDITIONAL STREAK TEST: mode=%s" % args.mode)
    print("  threshold %.3f (two families: quality and bye)" % ALPHA)
    if args.mode == "quality":
        print("  C1 predicts the worst-quartile restriction does NOT rescue")
        print("     the templates: quartering the data quarters the cells")
        print("  C2 predicts anything that appears will be in rushing")
    else:
        print("  C3 predicts post-bye streak ROI is indistinguishable from")
        print("     non-bye: interrupting a coin flip changes nothing")
        print("  C4 predicts most post-bye cells fall below the floor")

    hr("LOADING")
    D, sched = load(args.cache)
    print("  %d prop rows with an opponent" % len(D))
    D = add_streaks(D)
    print("  %d in a classifiable game" % len(D))

    if args.mode == "quality":
        D = add_quality(D)
        print("  %d with lagged defensive quality" % len(D))
        print("  quartile 3 = most yards allowed = WORST defences")
        conds = [("worst q", (D["dq"] == 3).to_numpy()),
                 ("best q", (D["dq"] == 0).to_numpy()),
                 ("all", np.ones(len(D), bool))]
    else:
        D = add_bye(D, sched)
        print("  %d with bye status" % len(D))
        print("  %s" % D["post_bye"].value_counts().to_dict())
        conds = [("post bye", (D["post_bye"] == "post bye").to_numpy()),
                 ("normal", (D["post_bye"] == "normal").to_numpy())]

    win = D["under"].to_numpy().astype(bool)
    profit = profit_if_win(D["under_odds"].to_numpy())
    print()
    print("  pooled under ROI %+.4f on %d bets" % (roi_of(win, profit),
                                                   len(D)))

    templates = []
    for mk in MARKETS:
        for L in LENGTHS:
            for direction in ("over", "under"):
                col = "run_over" if direction == "over" else "run_under"
                base = (D["market"] == mk)
                run = (D[col] >= L) if L == max(LENGTHS) else (D[col] == L)
                lab = "%d+" % L if L == max(LENGTHS) else "%d" % L
                for cname, cmask in conds:
                    m = (base & run).to_numpy() & cmask
                    templates.append(("%s after %s %s" % (mk, lab, direction),
                                      cname, m))

    hr("THE FAMILY")
    print("  %d templates = %d markets x %d lengths x 2 directions x %d"
          % (len(templates), len(MARKETS), len(LENGTHS), len(conds)))
    print("  conditions: %s" % [c[0] for c in conds])
    out = run_family(D, templates, win, profit, args.perm, args.mode)

    if out is None:
        print()
        print("  NOTHING WAS WRITTEN.")
        return
    best, omax, p, nullmed = out

    hr("CONDITION-BY-CONDITION, for the comparison the theory predicts")
    print("  The theories say the EFFECT should differ BY CONDITION, not")
    print("  that some cell somewhere is high. So this pools every streak")
    print("  template within each condition and compares them directly.")
    print()
    for cname, cmask in conds:
        anystreak = ((D["run_over"] >= 2) | (D["run_under"] >= 2)).to_numpy()
        m = cmask & anystreak
        if m.sum() < 30:
            print("  %-10s too few rows (%d)" % (cname, int(m.sum())))
            continue
        print("  %-10s under ROI %+.4f on %5d bets after any 2+ streak"
              % (cname, roi_of(win[m], profit[m]), int(m.sum())))
    print()
    for cname, cmask in conds:
        rs, ls = [], []
        for L in LENGTHS:
            m = cmask & ((D["run_over"] >= L) if L == max(LENGTHS)
                         else (D["run_over"] == L)).to_numpy()
            if m.sum() >= MIN_CELL:
                rs.append(roi_of(win[m], profit[m]))
                ls.append(L)
        if len(rs) >= 3:
            print("  %-10s monotonicity in OVER-streak length: rho %+.4f"
                  % (cname, spearman(ls, rs)))

    hr("VERDICT")
    if p >= ALPHA:
        print("  NULL for the %s family. Best %+.4f against a null maximum"
              % (args.mode, omax))
        print("  of %+.4f." % nullmed)
        print()
        if args.mode == "quality":
            print("  C1 predicted this. Conditioning on defensive quality")
            print("  does not rescue streak templates.")
        else:
            print("  C3 predicted this. A bye week does not change what a")
            print("  streak is worth, because the streaks were coin flips")
            print("  before the bye as well.")
    else:
        print("  CLEARS at %.3f. %s / %s, ROI %+.4f."
              % (ALPHA, best[0], best[1], omax))
        print("  Check the bet count and the condition comparison above:")
        print("  the theories predict a DIFFERENCE BETWEEN conditions, so a")
        print("  single high cell inside one condition is not the claim.")
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
