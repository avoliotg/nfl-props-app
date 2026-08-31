"""
QB Passing Yards market module.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from . import data_utils
from sklearn.linear_model import LinearRegression

SEASONS = [2022, 2023, 2024, 2025, 2026]
LEAN_FEATS = ["attempts_roll", "team_spread", "total_line", "wind_eff", "def_pass_roll"]

# passing yards are big numbers — tiers in yards, bigger gaps than receiving
GAP_ANCHORS = [(0, 0.49), (10, 0.54), (25, 0.60), (50, 0.66), (90, 0.70)]


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_dataset():
    ps = data_utils.load_player_stats(SEASONS)
    qb = ps[ps["position"] == "QB"].copy()
    qb = qb[qb["attempts"].fillna(0) >= 10].copy()
    qb = qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    qb["attempts_roll"] = (qb.groupby("player_id")["attempts"]
                           .transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean()))

    games = data_utils.load_schedules(SEASONS)
    home = games[["season", "week", "home_team", "spread_line", "total_line", "roof", "wind"]].rename(
        columns={"home_team": "team"})
    home["team_spread"] = home["spread_line"]
    away = games[["season", "week", "away_team", "spread_line", "total_line", "roof", "wind"]].rename(
        columns={"away_team": "team"})
    away["team_spread"] = -away["spread_line"]
    tg = pd.concat([home, away], ignore_index=True)[
        ["season", "week", "team", "team_spread", "total_line", "roof", "wind"]]
    qb = qb.merge(tg, on=["season", "week", "team"], how="left")

    qb["is_outdoor"] = qb["roof"].isin(["outdoors", "open"]).astype(int)
    qb["wind_eff"] = np.where(qb["is_outdoor"] == 1, qb["wind"].fillna(7), 0)

    # opponent pass defense: passing yards allowed, rolling
    qd = (qb.groupby(["season", "week", "opponent_team"])["passing_yards"].sum()
          .reset_index().rename(columns={"opponent_team": "defteam", "passing_yards": "pa"}))
    qd = qd.sort_values(["defteam", "season", "week"])
    qd["def_pass_roll"] = (qd.groupby("defteam")["pa"]
                           .transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean()))
    qb = qb.merge(qd[["season", "week", "defteam", "def_pass_roll"]].rename(
        columns={"defteam": "opponent_team"}), on=["season", "week", "opponent_team"], how="left")
    return qb


@st.cache_resource(show_spinner="Training model...")
def load_model():
    qb = build_dataset()
    train = qb[(qb["season"] <= 2024) & (qb["attempts_roll"] >= 10)].dropna(
        subset=LEAN_FEATS + ["passing_yards"])
    model = LinearRegression().fit(train[LEAN_FEATS], train["passing_yards"])
    return model, LEAN_FEATS


def available_seasons():
    return SEASONS


def available_weeks(season):
    qb = build_dataset()
    return sorted(qb[qb["season"] == season]["week"].dropna().unique().tolist())


def project_week(season, week, min_attempts=1.5):
    model, feats = load_model()
    qb = build_dataset()
    wk = qb[(qb["season"] == season) & (qb["week"] == week)].copy()
    wk = wk[wk["attempts_roll"] >= min_attempts].dropna(subset=feats)
    if len(wk) == 0:
        return pd.DataFrame()
    wk["projection"] = model.predict(wk[feats]).round(1)
    wk["attempts_roll"] = wk["attempts_roll"].round(1)
    cols = ["player_display_name", "team", "opponent_team", "position",
            "projection", "attempts_roll"]
    cols = [c for c in cols if c in wk.columns]
    return wk[cols].sort_values("projection", ascending=False).reset_index(drop=True)


def tier_for_gap(gap):
    if gap is None or pd.isna(gap):
        return ""
    ag = abs(gap)
    if ag < 10:
        return "Pass"
    elif ag < 25:
        return "Lean"
    elif ag < 50:
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
    p = qb[(qb["season"] == season) & (qb["attempts_roll"] >= 10)]
    return sorted(p["player_display_name"].dropna().unique().tolist())


def player_history(season, player_name, min_attempts=0.5):
    model, feats = load_model()
    qb = build_dataset()
    p = qb[(qb["season"] == season) &
           (qb["player_display_name"] == player_name) &
           (qb["attempts_roll"] >= min_attempts)].dropna(subset=feats)
    if len(p) == 0:
        return pd.DataFrame()
    p = p.copy()
    p["projection"] = model.predict(p[feats]).round(1)
    out = p[["week", "opponent_team", "projection", "passing_yards", "attempts_roll"]].copy()
    out = out.rename(columns={"passing_yards": "actual"})
    return out.sort_values("week").reset_index(drop=True)


def actual_result(season, week, player_name):
    """Actual passing yards for grading. Returns the number, or None."""
    df = build_dataset()
    m = df[(df["season"] == season) & (df["week"] == week) &
           (df["player_display_name"] == player_name)]
    if len(m) == 0:
        return None
    val = m.iloc[0]["passing_yards"]
    return None if pd.isna(val) else float(val)