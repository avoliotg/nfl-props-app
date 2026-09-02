"""
QB Rushing Yards market module — self-contained engine.
Separate from rushing.py because QB rushing behaves fundamentally differently
from RB rushing (carries mix kneels/sneaks/scrambles, not homogeneous RB carries).
Validated 2025 OOS: corr 0.51 (best of all markets), 2-feature model.
Projects ALL QBs with valid rolling data — pocket passers included, since their
low-rushing lines are just as bettable (and just as easy for the model to call
correctly) as scramblers' high-rushing lines.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from . import data_utils
from sklearn.linear_model import LinearRegression

SEASONS = [2022, 2023, 2024, 2025, 2026]
LEAN_FEATS = ["rush_yds_roll", "carries_roll"]

GAP_ANCHORS = [(0, 0.49), (3, 0.55), (7, 0.61), (15, 0.68), (30, 0.70)]


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_dataset():
    ps = data_utils.load_player_stats(SEASONS)
    qb = ps[ps["position"] == "QB"].copy()
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
def load_model():
    qb = build_dataset()
    train = qb[qb["season"] <= 2024].dropna(subset=LEAN_FEATS + ["rushing_yards"])
    model = LinearRegression().fit(train[LEAN_FEATS], train["rushing_yards"])
    return model, LEAN_FEATS

def available_seasons():
    return SEASONS


def available_weeks(season):
    qb = build_dataset()
    played = sorted(qb[qb["season"] == season]["week"].dropna().unique().tolist())
    if played:
        return played
    return [1]


def project_week(season, week):
    model, feats = load_model()
    qb = build_dataset()
    wk = qb[(qb["season"] == season) & (qb["week"] == week)].copy()
    wk = wk.dropna(subset=feats)

    if len(wk) == 0:
        wk = build_upcoming_week(season, week)
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
    """Actual rushing yards for grading. Returns the number, or None."""
    df = build_dataset()
    m = df[(df["season"] == season) & (df["week"] == week) &
           (df["player_display_name"] == player_name)]
    if len(m) == 0:
        return None
    val = m.iloc[0]["rushing_yards"]
    return None if pd.isna(val) else float(val)


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

        # depth chart → each team's current QB1 (most recent snapshot); keep only starters
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