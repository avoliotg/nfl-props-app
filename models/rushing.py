"""
Rushing Yards market module.

PHASE 3.1: the carries_roll survivorship fix, plus two further instances of
the same bug found in this file.

WHAT WAS WRONG

  1. carries_roll WAS COMPUTED AFTER THE VOLUME FILTER.
     build_dataset filtered `carries >= 5` and then took a 6-game rolling mean
     of carries, so a backup's workload feature was measured only over the
     weeks the starter was hurt. Measured inflation was +6.31 carries in the
     0-2 bucket, falling monotonically to +0.06 at 15 or more. Rolling
     features now come from build_all_rows, which applies no volume filter,
     exactly as anytime_td.build_all_rows does.

  2. THE TRAINING POPULATION DID NOT MATCH THE SERVED POPULATION.
     load_model trained on `carries_roll >= 5` while project_week serves
     `carries_roll >= 1.5`, so the model was fitted on established backs and
     then applied to players with a third of that workload. Surviving training
     rows averaged 54.0 rushing yards against a true unconditional 32.5, which
     inflated the intercept by roughly 66 percent for backups. This is the
     same lesson anytime_td variant C taught in the other direction: the
     training population must match the SERVED population, not the whole
     league and not a higher-volume subset. Both thresholds now read
     MIN_CARRIES_ROLL so they cannot drift apart again.

  3. build_upcoming_week BRIDGED FROM THE FILTERED FRAME.
     It read build_dataset() and averaged the last 6 games of carries, so
     every Week 1 projection inherited the same inflation. It now reads
     build_all_rows().

  4. Null rushing_yards for an active RB is a genuine ZERO, not missing data,
     so those rows were being dropped from training by dropna. A back with a
     stat row and no carries belongs in the unconditional population. Same bug
     family as Phase 3.4.

  5. load_model hardcoded `season <= 2024`, which prevented eval_harness from
     scoring the real model walk-forward. It is now a parameter.

WHAT DELIBERATELY DID NOT CHANGE

  build_dataset still applies `carries >= 5`, so the board, all_players and
  player_history behave exactly as before. Only the ROLLING features and the
  TRAINING population moved, which is the same split anytime_td uses.

CONSEQUENCES TO MEASURE, NOT ASSUME

  The projection distribution changes, so the shipped sigma 1.922*proj^0.7149
  and the floor 34.5 were fitted against the OLD projections and are now
  invalid. That is Phase 3.2 and it is not optional.

  The serving population also shifts: carries_roll now includes low-carry
  games, so every player's roll falls and some marginal backs drop below
  MIN_CARRIES_ROLL. Those are exactly the obscure edge-target players the
  strategy depends on, so run filter_diagnostics() and compare counts before
  concluding the threshold is still right.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from . import data_utils
from sklearn.linear_model import LinearRegression

SEASONS = [2022, 2023, 2024, 2025, 2026]
LEAN_FEATS = ["carries_roll", "team_spread", "total_line"]

# The board's display threshold AND the training threshold. One constant, so
# the population the model is fitted on cannot drift away from the population
# it is asked to score. See note 2 in the module docstring.
MIN_CARRIES_ROLL = 1.5

# Last season included in training by default. Parameterised so eval_harness
# can score walk-forward instead of always training through 2024.
DEFAULT_TRAIN_MAX_SEASON = 2024

# rushing yards - tiers in yards (like receiving, but rushing is noisier)
GAP_ANCHORS = [(0, 0.49), (3, 0.55), (7, 0.61), (15, 0.68), (30, 0.70)]


def _attach_schedule(rush):
    """Team spread and total from the schedule, one row per team-game."""
    games = data_utils.load_schedules(SEASONS)
    home = games[["season", "week", "home_team", "spread_line", "total_line"]].rename(
        columns={"home_team": "team"})
    home["team_spread"] = home["spread_line"]
    away = games[["season", "week", "away_team", "spread_line", "total_line"]].rename(
        columns={"away_team": "team"})
    away["team_spread"] = -away["spread_line"]
    team_game = pd.concat([home, away], ignore_index=True)[
        ["season", "week", "team", "team_spread", "total_line"]]
    return rush.merge(team_game, on=["season", "week", "team"], how="left")


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_all_rows():
    """Every active RB player-week, features included, NO volume filter.

    Position is a structural attribute of the player, so filtering on it
    cannot condition on the outcome and is safe here. Carries are an in-game
    result, so filtering on them before a rolling mean is what caused the bug.
    """
    ps = data_utils.load_player_stats(SEASONS)
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    rush = ps[ps["position"] == "RB"].copy()

    # An RB with a stat row was active. No carries means zero carries and zero
    # yards, not unknown, so these rows belong in the population rather than
    # being silently dropped later by dropna.
    rush["carries"] = rush["carries"].fillna(0)
    rush["rushing_yards"] = rush["rushing_yards"].fillna(0.0)

    rush = rush.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    rush["carries_roll"] = (rush.groupby("player_id")["carries"]
                            .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean()))
    return _attach_schedule(rush)


@st.cache_data(show_spinner=False)
def build_dataset():
    """Rows with real in-game volume. Unchanged semantics for the board."""
    rush = build_all_rows()
    return rush[rush["carries"] >= 5].copy()


@st.cache_resource(show_spinner="Training model...")
def load_model(train_max_season=DEFAULT_TRAIN_MAX_SEASON,
               train_min_carries_roll=MIN_CARRIES_ROLL):
    """Fit on the SERVED population, read from the unfiltered frame.

    `carries >= 5` conditions on the current game: a back who finishes with 2
    carries almost certainly gained few yards, so training on that population
    leaks the outcome and over-states the level. `carries_roll >= 1.5` uses
    only lagged history, which is the same rule project_week applies to decide
    who appears on the board.
    """
    rush = build_all_rows()
    train = rush[(rush["season"] <= train_max_season) &
                 (rush["carries_roll"] >= train_min_carries_roll)].dropna(
        subset=LEAN_FEATS + ["rushing_yards"])
    model = LinearRegression().fit(train[LEAN_FEATS], train["rushing_yards"])
    return model, LEAN_FEATS


def filter_diagnostics(season=2025):
    """Before and after counts, so the change is measured rather than assumed.

    Call from cmd or Colab:
        from models import rushing; print(rushing.filter_diagnostics())
    """
    allr = build_all_rows()
    filt = build_dataset()
    a = allr[allr["season"] == season]
    f = filt[filt["season"] == season]
    served = a[a["carries_roll"] >= MIN_CARRIES_ROLL]
    old_served = f[f["carries_roll"] >= MIN_CARRIES_ROLL]
    return pd.DataFrame([
        {"frame": "all active RB rows", "rows": len(a),
         "mean_carries_roll": round(a["carries_roll"].mean(), 2),
         "mean_rushing_yards": round(a["rushing_yards"].mean(), 1)},
        {"frame": "carries >= 5 (board)", "rows": len(f),
         "mean_carries_roll": round(f["carries_roll"].mean(), 2),
         "mean_rushing_yards": round(f["rushing_yards"].mean(), 1)},
        {"frame": f"served, carries_roll >= {MIN_CARRIES_ROLL}",
         "rows": len(served),
         "mean_carries_roll": round(served["carries_roll"].mean(), 2),
         "mean_rushing_yards": round(served["rushing_yards"].mean(), 1)},
        {"frame": "served within the filtered frame",
         "rows": len(old_served),
         "mean_carries_roll": round(old_served["carries_roll"].mean(), 2),
         "mean_rushing_yards": round(old_served["rushing_yards"].mean(), 1)},
    ])


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


def project_week(season, week, min_carries=MIN_CARRIES_ROLL):
    model, feats = load_model()
    rush = build_dataset()
    wk = rush[(rush["season"] == season) & (rush["week"] == week)].copy()

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

    wk = wk[wk["carries_roll"] >= min_carries].dropna(subset=feats)

    if len(wk) == 0:
        return pd.DataFrame()
    wk["projection"] = model.predict(wk[feats]).round(1)
    wk["carries_roll"] = wk["carries_roll"].round(1)
    cols = ["player_display_name", "team", "opponent_team", "position",
            "projection", "carries_roll"]
    cols = [c for c in cols if c in wk.columns]
    return wk[cols].sort_values("projection", ascending=False).reset_index(drop=True)


# PERFORMANCE, September 25. This function had NO cache decorator in any of
# the six market modules, and project_week calls it unconditionally on every
# call. Streamlit re-runs the whole script on each widget interaction, so
# every dropdown click re-read rosters and schedules (and depth charts, for
# the QB markets) from scratch, per market. app.py renders the rushing board
# from BOTH rushing.project_week and qb_rushing.project_week, so that board
# paid for two full assemblers.
#
# TTL RATHER THAN A PLAIN CACHE, DELIBERATELY. What this loads is the LIVE
# data: rosters and depth charts change mid-week when a starter is ruled out,
# and a permanent cache would serve a stale QB1 for the rest of the week.
# Thirty minutes keeps the board responsive while still picking up news
# within one refresh cycle. Lower it if that feels too slow to react; the
# cost of a miss is one rebuild.
@st.cache_data(ttl=1800, show_spinner=False)
def build_upcoming_week(season, week):
    """Manufacture player-week rows for a game not yet played (e.g. Week 1),
    bridging carries from the prior season. Fallback when build_dataset()
    has no rows for the requested week.

    Bridges from build_all_rows, NOT build_dataset. Reading the filtered frame
    averaged only the weeks a back got 5 or more carries, so every Week 1
    projection carried the same survivorship inflation the rolling feature
    used to have.
    """
    import nflreadpy as nfl

    ros = nfl.load_rosters([season])
    ros = ros.to_pandas() if hasattr(ros, "to_pandas") else ros
    ros = ros[ros["position"].isin(["RB"])]
    ros = ros[["gsis_id", "full_name", "team", "position"]].rename(
        columns={"gsis_id": "player_id"})
    ros = ros.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])

    rush = build_all_rows()
    prior = rush[rush["season"] == season - 1].sort_values(["player_id", "week"])

    def _bridge(g):
        last6 = g.tail(6)
        return pd.Series({
            "carries_roll": last6["carries"].mean(),
            "player_display_name": g.iloc[-1]["player_display_name"],
        })
    bridged = (prior.groupby("player_id", group_keys=False)
               .apply(_bridge, include_groups=False).reset_index())

    df = ros.merge(bridged, on="player_id", how="inner")

    sched = nfl.load_schedules([season])
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    wk = sched[sched["week"] == week]
    home = wk[["home_team", "away_team", "spread_line", "total_line"]].rename(
        columns={"home_team": "team", "away_team": "opponent_team"})
    home["team_spread"] = home["spread_line"]
    away = wk[["away_team", "home_team", "spread_line", "total_line"]].rename(
        columns={"away_team": "team", "home_team": "opponent_team"})
    away["team_spread"] = -away["spread_line"]
    ctx = pd.concat([home, away], ignore_index=True)[
        ["team", "opponent_team", "team_spread", "total_line"]]
    df = df.merge(ctx, on="team", how="inner")

    df["season"] = season
    df["week"] = week
    df = df.dropna(subset=LEAN_FEATS)
    return df.reset_index(drop=True)


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
    rush = build_dataset()
    p = rush[(rush["season"] == season) & (rush["carries_roll"] >= 5)]
    return sorted(p["player_display_name"].dropna().unique().tolist())


def player_history(season, player_name, min_carries=0.5):
    model, feats = load_model()
    rush = build_dataset()
    p = rush[(rush["season"] == season) &
             (rush["player_display_name"] == player_name) &
             (rush["carries_roll"] >= min_carries)].dropna(subset=feats)
    if len(p) == 0:
        return pd.DataFrame()
    p = p.copy()
    p["projection"] = model.predict(p[feats]).round(1)
    out = p[["week", "opponent_team", "projection", "rushing_yards", "carries_roll"]].copy()
    out = out.rename(columns={"rushing_yards": "actual"})
    return out.sort_values("week").reset_index(drop=True)


# NOW UNUSED. actual_result delegates to data_utils.actual_stat, which keeps
# its own cached frame with the collision guard attached. Left rather than
# deleted in case an external caller exists; nothing in this module uses it.
@st.cache_data(show_spinner=False)
def _all_player_stats():
    """Unfiltered player stats for grading.

    actual_result must NOT read build_dataset: that frame drops every week
    under 5 carries and every non-RB, so a backup who got 2 carries returned
    None and the bet silently never graded, and no QB rushing bet could ever
    grade at all. The lost rows average 5.0 rushing yards against 54.0 for the
    rows that survived, so the missing grades were disproportionately busts.
    """
    ps = data_utils.load_player_stats(SEASONS)
    return ps.to_pandas() if hasattr(ps, "to_pandas") else ps


def actual_result(season, week, player_name):
    """Actual rushing yards for grading. Returns the number, or None.

    Delegates to data_utils.actual_stat. This function already read unfiltered
    stats, already had the normalised fallback, and already treated a null
    stat on an active player as a genuine zero. What it lacked was the
    collision guard: name_resolve.py measured 21 normalized names in nflverse
    mapping to more than one player_id, and the WORST of them lands squarely
    in this market.

        michael carter   CB with 51 rows   against   RB with 47 rows

    Michael Carter the running back carries real rushing props. In any week
    both played, iloc[0] returned whichever row pandas happened to order
    first, so a bet could be graded against a cornerback's rushing yards.
    Close to a coin flip on a bettable player, and silent.

    POSITION LIST, AND WHY IT IS NOT JUST ["RB"].

    Since plan item 3.5, market = 'rushing' is the population AFTER
    quarterbacks are split out to qb_rushing, but it is still genuinely
    mixed: the harness's unjoined list for this market is led by Taysom Hill
    (35), Deebo Samuel (31), Xavier Worthy (26), Puka Nacua (18) and Jameson
    Williams (17), all receivers or hybrids taking carries. Filtering to RB
    alone would refuse to grade every one of them.

    So the list covers the offensive groups that take carries and excludes
    defensive backs, which is exactly what makes the michael carter collision
    unreachable while leaving the receivers gradeable. QB is deliberately
    absent: those rows belong to qb_rushing now.
    """
    return data_utils.actual_stat(season, week, player_name, "rushing_yards",
                                  seasons=SEASONS,
                                  position=["RB", "WR", "TE", "FB"])
