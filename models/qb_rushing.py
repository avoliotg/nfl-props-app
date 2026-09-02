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