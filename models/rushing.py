"""
Rushing Yards market module.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from . import data_utils
from sklearn.linear_model import LinearRegression

SEASONS = [2022, 2023, 2024, 2025, 2026]
LEAN_FEATS = ["carries_roll", "team_spread", "total_line"]

# rushing yards — tiers in yards (like receiving, but rushing is noisier)
GAP_ANCHORS = [(0, 0.49), (3, 0.55), (7, 0.61), (15, 0.68), (30, 0.70)]


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_dataset():
    ps = data_utils.load_player_stats(SEASONS)
    rush = ps[ps["carries"].fillna(0) >= 5].copy()
    rush = rush.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    rush["carries_roll"] = (rush.groupby("player_id")["carries"]
                            .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean()))

    games = data_utils.load_schedules(SEASONS)
    home = games[["season", "week", "home_team", "spread_line", "total_line"]].rename(
        columns={"home_team": "team"})
    home["team_spread"] = home["spread_line"]
    away = games[["season", "week", "away_team", "spread_line", "total_line"]].rename(
        columns={"away_team": "team"})
    away["team_spread"] = -away["spread_line"]
    team_game = pd.concat([home, away], ignore_index=True)[
        ["season", "week", "team", "team_spread", "total_line"]]
    rush = rush.merge(team_game, on=["season", "week", "team"], how="left")
    return rush


@st.cache_resource(show_spinner="Training model...")
def load_model():
    rush = build_dataset()
    train = rush[(rush["season"] <= 2024) & (rush["carries_roll"] >= 5)].dropna(
        subset=LEAN_FEATS + ["rushing_yards"])
    model = LinearRegression().fit(train[LEAN_FEATS], train["rushing_yards"])
    return model, LEAN_FEATS


def available_seasons():
    return SEASONS


def available_weeks(season):
    rush = build_dataset()
    return sorted(rush[rush["season"] == season]["week"].dropna().unique().tolist())


def project_week(season, week, min_carries=1.5):
    model, feats = load_model()
    rush = build_dataset()
    wk = rush[(rush["season"] == season) & (rush["week"] == week)].copy()
    wk = wk[wk["carries_roll"] >= min_carries].dropna(subset=feats)
    if len(wk) == 0:
        return pd.DataFrame()
    wk["projection"] = model.predict(wk[feats]).round(1)
    wk["carries_roll"] = wk["carries_roll"].round(1)
    cols = ["player_display_name", "team", "opponent_team", "position",
            "projection", "carries_roll"]
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


def actual_result(season, week, player_name):
    """Actual rushing yards for grading. Returns the number, or None."""
    df = build_dataset()
    m = df[(df["season"] == season) & (df["week"] == week) &
           (df["player_display_name"] == player_name)]
    if len(m) == 0:
        return None
    val = m.iloc[0]["rushing_yards"]
    return None if pd.isna(val) else float(val)