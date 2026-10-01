"""trend_scan.py - a corrected scan of pregame conditional cells.

Implements PREREG_trend_scan.md, written before any cell ROI was computed.
READ ONLY: fits nothing, writes nothing, changes no constant.

    python trend_scan.py --cache lines_cache.parquet

THE STATISTIC IS THE FAMILY MAXIMUM, NOT THE BEST CELL'S OWN p-VALUE.

    Shuffle win/loss globally, each row keeping its own cell memberships
    and its own price, recompute every cell, take the maximum. The p-value
    is the share of permutations whose maximum equals or exceeds the
    observed maximum.

    That is the only question a scan can honestly ask: given this many
    chances, is the best cell better than the best cell CHANCE ALONE would
    produce? It corrects for family size and for the overlap between cells
    at the same time, because each permuted family has exactly the same
    overlap structure as the real one. A Bonferroni cannot do the second
    part: "Browns on Thursday", "Browns at home" and "Browns in prime
    time" are not independent tests.

WHAT THIS CANNOT DO, STATED IN THE PRE-REGISTRATION RATHER THAN DISCOVERED
HERE. It cannot see an effect smaller than the null maximum. Rule 1's
measured edge is +0.0589, so if the null maximum lands near +0.08 as
predicted, THIS SCAN COULD NOT DETECT RULE 1 ITSELF. That is the
arithmetic cost of looking in 208 places, and the figure is arguably the
most useful thing the run produces: it sets the size of effect that
trend-hunting can establish at all.

EVERY DIMENSION IS KNOWABLE BEFORE KICKOFF. Nothing derived from the
outcome, nothing from the same week's result. `rest` is DERIVED rather than
read: it needs each team's previous game date, so it is computed from the
schedule per team-week.
"""
import argparse
import sys

import numpy as np
import pandas as pd

BOOK = "fanduel"
MARKETS = ["receiving", "receptions", "rushing", "qb_passing"]
STAT_COL = {"receiving": "receiving_yards", "receptions": "receptions",
            "rushing": "rushing_yards", "qb_passing": "passing_yards"}
MIN_BETS = 150
N_PERM = 2000
DEV_SEASONS = [2023, 2024]
HOLD_SEASONS = [2025, 2026]
SHORT_REST_DAYS = 5          # Thursday off a Sunday is 4 days


def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def profit_if_win(odds):
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, o / 100.0, 100.0 / np.abs(o))


def slot_of(kick):
    """Thu / Sun early / Sun late / SNF / MNF, from kickoff in US Eastern."""
    t = kick.dt.tz_convert("America/New_York")
    dow, hour = t.dt.dayofweek, t.dt.hour
    out = pd.Series("other", index=kick.index, dtype=object)
    out[dow == 3] = "Thu"
    out[(dow == 0)] = "MNF"
    sun = dow == 6
    out[sun & (hour < 15)] = "Sun early"
    out[sun & (hour >= 15) & (hour < 19)] = "Sun late"
    out[sun & (hour >= 19)] = "SNF"
    return out


def build_cells(df):
    """(dimension, level, boolean mask) for every cell. Fixed, not chosen."""
    cells = []
    for team, g in df.groupby("opponent_team"):
        cells.append(("opponent", str(team),
                      (df["opponent_team"] == team).to_numpy()))
    for v in ["home", "away"]:
        cells.append(("venue", v, (df["venue"] == v).to_numpy()))
    for s in ["Thu", "Sun early", "Sun late", "SNF", "MNF"]:
        cells.append(("slot", s, (df["slot"] == s).to_numpy()))
    for b in ["fav 6+", "fav 0-6", "dog 0-6", "dog 6+"]:
        cells.append(("spread", b, (df["spread_band"] == b).to_numpy()))
    for b in ["low", "mid", "high"]:
        cells.append(("total", b, (df["total_band"] == b).to_numpy()))
    for b in ["q1", "q2", "q3", "q4"]:
        cells.append(("line", b, (df["line_band"] == b).to_numpy()))
    for r in ["normal", "short"]:
        cells.append(("rest", r, (df["rest"] == r).to_numpy()))
    return cells


def cell_table(win, profit, cells, min_bets=MIN_BETS):
    """Under ROI per cell. Returns arrays aligned to `cells`."""
    rois = np.full(len(cells), np.nan)
    ns = np.zeros(len(cells), dtype=int)
    for i, (_, _, m) in enumerate(cells):
        n = int(m.sum())
        ns[i] = n
        if n < min_bets:
            continue
        units = np.where(win[m], profit[m], -1.0)
        rois[i] = units.mean()
    return rois, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lines_cache.parquet")
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    hr("TREND SCAN: corrected search over pregame conditional cells")
    print("  pre-registered in PREREG_trend_scan.md")
    print("  statistic: the FAMILY MAXIMUM under ROI, permutation-corrected")
    print("  bet floor: %d per cell" % MIN_BETS)
    print("  markets:   %s" % MARKETS)
    print()
    print("  T1 predicts the best cell looks impressive and does NOT clear.")
    print("  T2 predicts the NULL MAXIMUM is above +0.08, which is the")
    print("     figure that makes every published trend of that size")
    print("     uninformative. T2 is arguably the real deliverable.")
    print("  T3 predicts opponent cells dominate the raw list and survive")
    print("     least often.")

    hr("LOADING")
    try:
        C = pd.read_parquet(args.cache)
    except Exception as e:
        print("  could not read %s: %s" % (args.cache, e))
        sys.exit(1)
    print("  %d rows in %s" % (len(C), args.cache))
    C = C[(C["book"] == BOOK) & (C["market"].isin(MARKETS))].copy()
    if "snapshot_label" in C.columns:
        C = C[C["snapshot_label"] == "closing"]
    print("  %d %s closing rows in the four markets" % (len(C), BOOK))

    from models import data_utils
    seasons = sorted(C["season"].dropna().unique().astype(int).tolist())

    # outcomes, one market at a time because the stat column differs
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
        g = g.rename(columns={col: "actual"})
        frames.append(g)
    D = pd.concat(frames, ignore_index=True)
    print("  %d rows joined to an outcome" % len(D))

    D = D[D["actual"].notna() & D["line"].notna()
          & D["under_odds"].notna() & D["over_odds"].notna()].copy()
    print("  %d with a line, both prices and an outcome" % len(D))

    hr("BUILDING THE CELL DIMENSIONS")
    sched = data_utils.load_schedules(seasons)
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    need = ["season", "week", "home_team", "away_team", "spread_line",
            "total_line", "gameday", "gametime"]
    have = [c for c in need if c in sched.columns]
    sc = sched[have].copy()
    print("  schedule columns available: %s" % have)

    # kickoff, for the slot and rest dimensions
    if "gameday" in sc.columns and "gametime" in sc.columns:
        kick = pd.to_datetime(
            sc["gameday"].astype(str) + " " + sc["gametime"].astype(str),
            errors="coerce")
        # nflverse gametime is US Eastern wall clock
        sc["kick"] = (kick.dt.tz_localize("America/New_York",
                                          ambiguous="NaT",
                                          nonexistent="NaT")
                      .dt.tz_convert("UTC"))
    else:
        sc["kick"] = pd.NaT
    print("  kickoff resolved on %d of %d games"
          % (int(sc["kick"].notna().sum()), len(sc)))

    # REST IS DERIVED. It needs each team's PREVIOUS game date, which is
    # not a schedule column. Built per team-week here so that "short" means
    # the same thing everywhere.
    long_rows = []
    for side, opp in (("home_team", "away_team"), ("away_team", "home_team")):
        t = sc[["season", "week", side, opp, "kick", "spread_line",
                "total_line"]].copy()
        t.columns = ["season", "week", "team", "opponent_team", "kick",
                     "spread_line", "total_line"]
        t["venue"] = "home" if side == "home_team" else "away"
        t["team_spread"] = (t["spread_line"] if side == "home_team"
                            else -t["spread_line"])
        long_rows.append(t)
    TG = pd.concat(long_rows, ignore_index=True)
    TG = TG.sort_values(["team", "season", "week"])
    prev = TG.groupby(["team", "season"])["kick"].shift(1)
    TG["rest_days"] = (TG["kick"] - prev).dt.total_seconds() / 86400.0
    TG["rest"] = np.where(TG["rest_days"].notna()
                          & (TG["rest_days"] < SHORT_REST_DAYS),
                          "short", "normal")
    TG["slot"] = slot_of(TG["kick"])
    print("  rest: %d short, %d normal, %d unknown (week 1 has no prior game)"
          % (int((TG["rest"] == "short").sum()),
             int((TG["rest"] == "normal").sum()),
             int(TG["rest_days"].isna().sum())))
    print("  slot counts: %s" % TG["slot"].value_counts().to_dict())

    D = D.merge(TG[["season", "week", "team", "opponent_team", "venue",
                    "slot", "rest", "team_spread", "total_line"]],
                on=["season", "week", "team"], how="left",
                suffixes=("", "_sched"))
    before = len(D)
    D = D[D["opponent_team"].notna() & D["slot"].notna()]
    print("  %d rows with full game context (%d dropped)"
          % (len(D), before - len(D)))

    # bands. Spread is absolute and fixed; total and line are WITHIN-MARKET
    # quantiles so a 40-yard receiving line and a 3.5 reception line are
    # banded against their own market rather than against each other.
    D["spread_band"] = pd.cut(
        D["team_spread"], [-99, -6, 0, 6, 99],
        labels=["fav 6+", "fav 0-6", "dog 0-6", "dog 6+"])
    D["total_band"] = (D.groupby("market")["total_line"]
                       .transform(lambda s: pd.qcut(s, 3,
                                                    labels=["low", "mid",
                                                            "high"],
                                                    duplicates="drop")))
    D["line_band"] = (D.groupby("market")["line"]
                      .transform(lambda s: pd.qcut(s, 4,
                                                   labels=["q1", "q2", "q3",
                                                           "q4"],
                                                   duplicates="drop")))
    for c in ("spread_band", "total_band", "line_band"):
        D[c] = D[c].astype(object)
    D = D.reset_index(drop=True)

    hr("THE FAMILY")
    cells = build_cells(D)
    print("  %d cells over %d rows" % (len(cells), len(D)))
    by_dim = {}
    for dim, lvl, m in cells:
        by_dim[dim] = by_dim.get(dim, 0) + 1
    print("  by dimension: %s" % by_dim)

    win = (D["actual"] < D["line"]).to_numpy()
    profit = profit_if_win(D["under_odds"].to_numpy())
    pooled = float(np.where(win, profit, -1.0).mean())
    print("  pooled under ROI across everything: %+.4f on %d bets"
          % (pooled, len(D)))

    rois, ns = cell_table(win, profit, cells)
    usable = np.isfinite(rois)
    print("  %d cells clear the %d-bet floor, %d do not"
          % (int(usable.sum()), MIN_BETS, int((~usable).sum())))

    hr("RAW LEADERBOARD (uncorrected, and therefore not evidence)")
    order = np.argsort(-np.where(usable, rois, -np.inf))
    print("  %-10s %-14s %9s %8s %10s" % ("dim", "level", "ROI", "bets",
                                          "win rate"))
    print("  " + "-" * 56)
    for i in order[:15]:
        if not usable[i]:
            break
        dim, lvl, m = cells[i]
        wr = float(win[m].mean())
        print("  %-10s %-14s %+9.4f %8d %10.4f" % (dim, lvl, rois[i], ns[i],
                                                   wr))
    print()
    print("  and the worst, since a profitable OVER is the same finding")
    print("  mirrored and testing one tail only would hide half the space:")
    for i in order[::-1][:5]:
        if not usable[i]:
            continue
        dim, lvl, m = cells[i]
        print("  %-10s %-14s %+9.4f %8d" % (cells[i][0], cells[i][1],
                                            rois[i], ns[i]))

    obs_max = float(np.nanmax(rois[usable]))
    obs_min = float(np.nanmin(rois[usable]))
    best = cells[int(np.nanargmax(np.where(usable, rois, -np.inf)))]
    print()
    print("  observed family maximum %+.4f  (%s = %s)"
          % (obs_max, best[0], best[1]))

    hr("PERMUTATION: IS THE BEST CELL BETTER THAN CHANCE'S BEST CELL?")
    print("  %d permutations, win/loss shuffled globally with every row" % args.perm)
    print("  keeping its own cell memberships and its own price")
    rng = np.random.default_rng(0)
    maxes = np.empty(args.perm)
    mins = np.empty(args.perm)
    for b in range(args.perm):
        w = rng.permutation(win)
        r, _ = cell_table(w, profit, cells)
        u = np.isfinite(r)
        maxes[b] = np.nanmax(r[u])
        mins[b] = np.nanmin(r[u])
        if (b + 1) % 200 == 0:
            print("\r    %d of %d" % (b + 1, args.perm), end="", flush=True)
    print()
    p_max = float(np.mean(maxes >= obs_max))
    p_min = float(np.mean(mins <= obs_min))
    print()
    print("  NULL MAXIMUM distribution, which is the T2 deliverable:")
    print("    mean %+.4f   median %+.4f   sd %.4f"
          % (maxes.mean(), np.median(maxes), maxes.std()))
    print("    5th %+.4f   95th %+.4f"
          % (np.quantile(maxes, 0.05), np.quantile(maxes, 0.95)))
    print()
    print("    So chance alone produces a best cell of about %+.4f on this"
          % np.median(maxes))
    print("    family. ANY published trend at or below that size is")
    print("    uninformative, however impressive the win-loss record looks.")
    print()
    print("  observed max %+.4f -> p = %.4f   %s"
          % (obs_max, p_max,
             "CLEARS" if p_max < args.alpha else "does not clear"))
    print("  observed min %+.4f -> p = %.4f   %s   (the over tail)"
          % (obs_min, p_min,
             "CLEARS" if p_min < args.alpha else "does not clear"))

    hr("RULE 1 AGAINST THIS YARDSTICK")
    print("  Rule 1's measured edge is +0.0589 on 1,425 bets.")
    print("  The null maximum here is about %+.4f." % np.median(maxes))
    if np.median(maxes) > 0.0589:
        print("  So THIS SCAN COULD NOT HAVE FOUND RULE 1. A 208-cell")
        print("  search makes an effect that size invisible.")
        print()
        print("  Which says something about how Rule 1 was found: a NARROW")
        print("  search, with the threshold chosen in advance for a reason")
        print("  unrelated to any result (46.5 is FanDuel's own rushing")
        print("  median). Narrow searches can see small effects. Broad ones")
        print("  cannot, and no amount of care changes that arithmetic.")

    if p_max >= args.alpha:
        hr("VERDICT: NULL FOR THE FAMILY")
        print("  T1 predicted this. The best cell is %+.4f, and chance's"
              % obs_max)
        print("  best cell on this same data is %+.4f at the median and"
              % np.median(maxes))
        print("  %+.4f at the 95th percentile." % np.quantile(maxes, 0.95))
        print()
        print("  No cell in this family is alive. An interaction phase")
        print("  would be a SEPARATE pre-registered family, not an")
        print("  extension of this one.")
    else:
        hr("VERDICT: THE FAMILY MAXIMUM CLEARS")
        dim, lvl, m = best
        print("  %s = %s, ROI %+.4f on %d bets." % (dim, lvl, obs_max,
                                                    int(m.sum())))
        print()
        print("  Remaining conditions from the pre-registration, which must")
        print("  ALL hold before this is a candidate:")
        d = D[m]
        for label, seasons_ in (("dev 2023-2024", DEV_SEASONS),
                                ("holdout 2025-2026", HOLD_SEASONS)):
            sub = d[d["season"].isin(seasons_)]
            if len(sub) == 0:
                print("    %-20s no rows" % label)
                continue
            w2 = (sub["actual"] < sub["line"]).to_numpy()
            p2 = profit_if_win(sub["under_odds"].to_numpy())
            r2 = float(np.where(w2, p2, -1.0).mean())
            print("    %-20s ROI %+.4f on %d bets  %s"
                  % (label, r2, len(sub),
                     "" if len(sub) >= MIN_BETS else "<- under the floor"))
    print()
    print("  NOTHING WAS WRITTEN.")
    print()


if __name__ == "__main__":
    main()
