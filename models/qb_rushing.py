"""
QB Rushing Yards market module - self-contained engine.
Separate from rushing.py because QB rushing behaves fundamentally differently
from RB rushing (carries mix kneels/sneaks/scrambles, not homogeneous RB carries).
Validated 2025 OOS: corr 0.51 (best of all markets), 2-feature model.
Projects ALL QBs with valid rolling data - pocket passers included, since their
low-rushing lines are just as bettable (and just as easy for the model to call
correctly) as scramblers' high-rushing lines.

SEPTEMBER 24 2026: three changes, one of which is the point and two of which
are hygiene. All three came out of plan item 3.5, which made this market
measurable for the first time by deriving the qb_rushing label at analysis
time (data_utils.split_qb_rushing) rather than trying to import it.

  1. THE ONE THAT MATTERS: load_model's training window is now a PARAMETER.

     It hardcoded `season <= 2024`, which is verbatim the defect rushing.py's
     docstring note 5 was fixed for. A single fit pinned to 2024 cannot be
     scored leave-season-out, because scoring 2023 and 2024 would use a model
     trained on them. Leave-season-out is the discipline that makes every
     number in this project trustworthy, so this blocked an honest
     measurement of the market entirely. Signature now matches rushing.py's
     so the two cannot drift apart.

  2. MEASURED NO-OP, ADDED ANYWAY: fillna(0) on carries and rushing_yards
     before the rolling features.

     rushing.py does this on the reasoning that a player with a stat row was
     ACTIVE, so a missing rushing line is a genuine zero rather than absent
     data. Measured by qb_null_check.py on all 2,817 QB rows: ZERO nulls in
     either column, byte-identical model coefficients, and zero movement in
     the served projection. nflverse stores 0.0 for active QBs, same as it
     does for receptions.

     So this is a latent defect with no current effect, and it is recorded as
     a no-op rather than as a fix that moved something. It is here to remove
     the possibility if nflverse's storage convention ever changes.

     Note what it does NOT fix: 112 training rows (5.47 percent) still drop
     on a null FEATURE. Those are each QB's debut game, where shift(1) has
     nothing to shift. project_week drops the same rows, so the training
     population already matches the served population and that gap is
     correct rather than survivorship.

  3. actual_result WAS THE GRADING BUG IN PLAN ITEM 3.5.

     Three defects, all of which rushing.py had already fixed:
       - it returned None where a null stat means a genuine zero
       - it had no normalised-name fallback, so "Mitch Trubisky" against
         nflverse's "Mitchell Trubisky" never graded
       - it read build_dataset(), which is cached for the board and drops
         nothing on a null feature, so a debut game could not grade

     It now reads unfiltered stats and THEN filters to QB. That ordering is
     deliberate and is not what rushing.py does. name_resolve.py measured 21
     normalized names in nflverse that map to more than one player, and
     `anthony brown` is one of them: a CB with 11 rows and a QB with 2. An
     unfiltered name match can therefore return two rows and iloc[0] picks
     arbitrarily, which grades a bet against the wrong player. Filtering to
     QB removes that whole class here, and a collision guard catches whatever
     is left.

     OPEN, NOT FIXED HERE: rushing.py, receiving.py, receptions.py,
     qb_passing.py and anytime_td.py all take iloc[0] on an unfiltered name
     match and are exposed to the same 21 collisions. The worst case is
     `michael carter`, a CB with 51 rows against an RB with 47, which is
     close to a coin flip on a genuinely bettable player. That needs one
     shared helper rather than five copies of this guard, and it is blocking
     for plan item 5.2 (paper trading) but harmless while the app is silent.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from . import data_utils
from sklearn.linear_model import LinearRegression

SEASONS = [2022, 2023, 2024, 2025, 2026]
LEAN_FEATS = ["rush_yds_roll", "carries_roll"]

# Last season included in training by default. Parameterised so eval_harness
# can score walk-forward instead of always training through 2024. Same name
# and same default as rushing.DEFAULT_TRAIN_MAX_SEASON, deliberately, so the
# two modules cannot drift apart.
DEFAULT_TRAIN_MAX_SEASON = 2024

GAP_ANCHORS = [(0, 0.49), (3, 0.55), (7, 0.61), (15, 0.68), (30, 0.70)]


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_dataset():
    ps = data_utils.load_player_stats(SEASONS)
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    qb = ps[ps["position"] == "QB"].copy()

    # A QB with a stat row was active, so no carries means zero carries and
    # zero yards rather than unknown. Measured as a no-op on all 2,817 rows
    # (qb_null_check.py, September 24): nflverse already stores 0.0. Kept to
    # remove the possibility, not because it changed anything.
    qb["carries"] = qb["carries"].fillna(0)
    qb["rushing_yards"] = qb["rushing_yards"].fillna(0.0)

    qb = qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for col in ["rushing_yards", "carries"]:
        qb[f"{col}_roll"] = (qb.groupby("player_id")[col]
                             .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean()))
    qb = qb.rename(columns={"rushing_yards_roll": "rush_yds_roll"})

    games = data_utils.load_schedules(SEASONS)
    home = games[["season", "week", "home_team", "spread_line", "total_line"]].rename(
        columns={"home_team": "team"})
    home["team_spread"] = home["spread_line"]
    away = games[["season", "week", "away_team", "spread_line", "total_line"]].rename(
        columns={"away_team": "team"})
    away["team_spread"] = -away["spread_line"]
    team_game = pd.concat([home, away], ignore_index=True)[
        ["season", "week", "team", "team_spread", "total_line"]]
    qb = qb.merge(team_game, on=["season", "week", "team"], how="left")
    return qb


@st.cache_resource(show_spinner="Training model...")
def load_model(train_max_season=DEFAULT_TRAIN_MAX_SEASON):
    """Fit the two-feature model, with the training window as a parameter.

    train_max_season exists so eval_harness can score this market
    leave-season-out. Calling it with the default and then scoring 2023 or
    2024 measures a model trained on those rows, which is in-sample and
    biased upward. Any harness use must pass the held-out season's
    predecessor explicitly.

    No volume gate, by design: the module projects all QBs with valid rolling
    data, so unlike rushing.py there is no threshold that could drift between
    the training and serving populations. The only rows excluded are those
    with a null FEATURE, which are debut games, and project_week excludes
    exactly the same ones.
    """
    qb = build_dataset()
    train = qb[qb["season"] <= train_max_season].dropna(
        subset=LEAN_FEATS + ["rushing_yards"])
    model = LinearRegression().fit(train[LEAN_FEATS], train["rushing_yards"])
    return model, LEAN_FEATS


def available_seasons():
    return SEASONS


def available_weeks(season):
    rec = build_dataset()
    played = sorted(rec[rec["season"] == season]["week"].dropna().unique().tolist())
    if not played:
        return [1]
    # Also offer the NEXT week. Returning only played weeks meant that once
    # week 1 finished, week 2 vanished from the dropdown entirely and there was
    # no way to project an upcoming game.
    nxt = int(max(played)) + 1
    return played + ([nxt] if nxt <= 22 else [])


def project_week(season, week):
    model, feats = load_model()
    qb = build_dataset()
    wk = qb[(qb["season"] == season) & (qb["week"] == week)].copy()

    # Union played rows with the assembler at the FEATURE level, then score
    # once. Using the assembler only when the played set was EMPTY broke any
    # week where some games have finished and others have not: after the
    # Wednesday opener this returned only NE and SEA, so every other team
    # vanished from the board, from import_lines (which then stored null
    # projections) and from the export team map.
    up = build_upcoming_week(season, week)
    if len(up) > 0:
        wk = pd.concat([wk, up], ignore_index=True)
        if "player_id" in wk.columns:
            # played rows come first, so they win a duplicate: a real result
            # beats a bridged estimate
            wk = wk.drop_duplicates(subset=["player_id"], keep="first")

    wk = wk.dropna(subset=feats)

    if len(wk) == 0:
        return pd.DataFrame()

    wk["projection"] = model.predict(wk[feats]).round(1)
    cols = ["player_display_name", "team", "opponent_team", "position",
            "projection", "rush_yds_roll", "carries_roll"]
    cols = [c for c in cols if c in wk.columns]
    return wk[cols].sort_values("projection", ascending=False).reset_index(drop=True)


def tier_for_gap(gap):
    if gap is None or pd.isna(gap):
        return ""
    ag = abs(gap)
    if ag < 3:
        return "Pass"
    elif ag < 7:
        return "Lean"
    elif ag < 15:
        return "Strong"
    else:
        return "Max"


def confidence_for_gap(gap):
    if gap is None or pd.isna(gap):
        return None
    ag = abs(gap)
    xs = [a[0] for a in GAP_ANCHORS]
    ys = [a[1] for a in GAP_ANCHORS]
    hit = float(np.interp(ag, xs, ys))
    lo, hi = GAP_ANCHORS[0][1], GAP_ANCHORS[-1][1]
    return round(max(0, min(100, (hit - lo) / (hi - lo) * 100)))


def all_players(season):
    qb = build_dataset()
    p = qb[(qb["season"] == season)].dropna(subset=LEAN_FEATS)
    return sorted(p["player_display_name"].dropna().unique().tolist())


def player_history(season, player_name):
    model, feats = load_model()
    qb = build_dataset()
    p = qb[(qb["season"] == season) &
           (qb["player_display_name"] == player_name)].dropna(subset=feats)
    if len(p) == 0:
        return pd.DataFrame()
    p = p.copy()
    p["projection"] = model.predict(p[feats]).round(1)
    out = p[["week", "opponent_team", "projection", "rushing_yards",
             "rush_yds_roll", "carries_roll"]].copy()
    out = out.rename(columns={"rushing_yards": "actual"})
    return out.sort_values("week").reset_index(drop=True)


def actual_result(season, week, player_name):
    """Actual rushing yards for grading. Returns the number, or None.

    Now delegates to data_utils.actual_stat rather than carrying its own
    copy of the logic. The earlier version in this file was correct, but it
    was the SIXTH implementation of the same three rules plus a guard, and
    five copies of a guard is the "import, do not restate" failure that
    produced this project's two worst errors. One implementation, six
    callers.

    position="QB" is retained and still matters: it makes the `anthony brown`
    collision (a CB with 11 rows against a QB with 2) unreachable rather
    than merely guarded.

    Behaviour is unchanged from the September 24 version: exact name match,
    then a normalised fallback for spellings like "Mitch Trubisky" against
    nflverse's "Mitchell Trubisky", 0.0 rather than None when an active QB
    has a null rushing line, and None rather than a guess when a name
    resolves to more than one player.
    """
    return data_utils.actual_stat(season, week, player_name, "rushing_yards",
                                  seasons=SEASONS, position="QB")


# PERFORMANCE, September 25. No cache decorator existed here, and
# project_week calls this unconditionally on every call. Streamlit re-runs
# the script on each widget interaction, so every dropdown click re-read
# rosters, depth charts AND schedules from scratch. app.py renders the
# rushing board from both rushing.project_week and qb_rushing.project_week,
# so that board paid for two assemblers.
#
# TTL rather than a plain cache, deliberately: depth charts are the LIVE
# data, and a permanent cache would serve a stale QB1 after a starter is
# ruled out. This module is the most exposed to that of the six, because it
# selects on pos_rank == 1 at the latest snapshot.
@st.cache_data(ttl=1800, show_spinner=False)
def build_upcoming_week(season, week):
    """Manufacture QB player-week rows for a game not yet played (e.g. Week 1),
    bridging rushing yards and carries from the prior season."""
    import nflreadpy as nfl

    ros = nfl.load_rosters([season])
    ros = ros.to_pandas() if hasattr(ros, "to_pandas") else ros
    ros = ros[ros["position"] == "QB"]
    ros = ros[["gsis_id", "full_name", "team", "position"]].rename(
        columns={"gsis_id": "player_id"})
    ros = ros.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])

    # depth chart -> each team's current QB1 (most recent snapshot); keep only starters
    dc = nfl.load_depth_charts([season])
    dc = dc.to_pandas() if hasattr(dc, "to_pandas") else dc
    dc = dc[dc["pos_abb"] == "QB"].copy()
    latest_dt = dc["dt"].max()
    dc = dc[dc["dt"] == latest_dt]
    starters = dc[dc["pos_rank"] == 1][["gsis_id"]].rename(columns={"gsis_id": "player_id"})
    ros = ros.merge(starters, on="player_id", how="inner")

    qb = build_dataset()
    prior = qb[qb["season"] == season - 1].sort_values(["player_id", "week"])

    def _bridge(g):
        last6 = g.tail(6)
        return pd.Series({
            "rush_yds_roll": last6["rushing_yards"].mean(),
            "carries_roll": last6["carries"].mean(),
            "player_display_name": g.iloc[-1]["player_display_name"],
        })
    bridged = (prior.groupby("player_id", group_keys=False)
               .apply(_bridge, include_groups=False).reset_index())

    df = ros.merge(bridged, on="player_id", how="inner")

    sched = nfl.load_schedules([season])
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    wk = sched[sched["week"] == week]
    home = wk[["home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent_team"})
    away = wk[["away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent_team"})
    ctx = pd.concat([home, away], ignore_index=True)
    df = df.merge(ctx, on="team", how="left")

    df["season"] = season
    df["week"] = week
    df = df.dropna(subset=LEAN_FEATS)
    return df.reset_index(drop=True)
