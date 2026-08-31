"""
Receptions market module — mirrors receiving, predicts catches.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from . import data_utils
from sklearn.linear_model import LinearRegression

SEASONS = [2022, 2023, 2024, 2025, 2026]
LEAN_FEATS = ["target_share_roll", "targets_roll", "snap_roll",
              "team_spread", "total_line"]

# receptions are small numbers — tiers are in CATCHES, not yards
GAP_ANCHORS = [(0, 0.49), (0.5, 0.55), (1.0, 0.60), (2.0, 0.66), (4.0, 0.70)]


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_dataset():
    ps = data_utils.load_player_stats(SEASONS)
    rec = ps[ps["position"].isin(["WR", "TE", "RB"])].copy()
    rec = rec.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for col in ["targets", "target_share"]:
        rec[f"{col}_roll"] = (rec.groupby("player_id")[col]
                              .transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean()))

    snaps = data_utils.load_snap_counts(SEASONS)
    snaps_s = snaps[["season", "week", "team", "player", "offense_pct"]].copy()

    # normalize names for joining (stats and snap tables format names differently,
    # e.g. "DK Metcalf" vs "D.K. Metcalf") — strip periods, extra spaces, lowercase
    def _norm_name(s):
        return (s.astype(str)
                 .str.replace(".", "", regex=False)
                 .str.replace(r"\s+", " ", regex=True)
                 .str.strip()
                 .str.lower())

    rec["_join_name"] = _norm_name(rec["player_display_name"])
    snaps_s["_join_name"] = _norm_name(snaps_s["player"])
    snaps_s = snaps_s[["season", "week", "team", "_join_name", "offense_pct"]]

    rec = rec.merge(snaps_s, on=["season", "week", "team", "_join_name"], how="left")
    rec = rec.drop(columns=["_join_name"])
    rec["snap_roll"] = (rec.groupby("player_id")["offense_pct"]
                        .transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean()))

    games = data_utils.load_schedules(SEASONS)
    home = games[["season", "week", "home_team", "spread_line", "total_line"]].rename(
        columns={"home_team": "team"})
    home["team_spread"] = home["spread_line"]
    away = games[["season", "week", "away_team", "spread_line", "total_line"]].rename(
        columns={"away_team": "team"})
    away["team_spread"] = -away["spread_line"]
    team_game = pd.concat([home, away], ignore_index=True)[
        ["season", "week", "team", "team_spread", "total_line"]]
    rec = rec.merge(team_game, on=["season", "week", "team"], how="left")
    return rec


@st.cache_resource(show_spinner="Training model...")
def load_model():
    rec = build_dataset()
    train = rec[(rec["season"] <= 2024) & (rec["targets_roll"] >= 3)].dropna(
        subset=LEAN_FEATS + ["receptions"])
    model = LinearRegression().fit(train[LEAN_FEATS], train["receptions"])
    return model, LEAN_FEATS


def available_seasons():
    return SEASONS


def available_weeks(season):
    rec = build_dataset()
    return sorted(rec[rec["season"] == season]["week"].dropna().unique().tolist())


def project_week(season, week, min_targets=1.5):
    model, feats = load_model()
    rec = build_dataset()
    wk = rec[(rec["season"] == season) & (rec["week"] == week)].copy()
    wk = wk[wk["targets_roll"] >= min_targets].dropna(subset=feats)
    if len(wk) == 0:
        return pd.DataFrame()
    wk["projection"] = model.predict(wk[feats]).round(1)
    wk["snap_roll"] = (wk["snap_roll"] * 100).round(0)
    cols = ["player_display_name", "team", "opponent_team", "position",
            "projection", "targets_roll", "snap_roll"]
    cols = [c for c in cols if c in wk.columns]
    return wk[cols].sort_values("projection", ascending=False).reset_index(drop=True)


def tier_for_gap(gap):
    if gap is None or pd.isna(gap):
        return ""
    ag = abs(gap)
    if ag < 0.5:
        return "Pass"
    elif ag < 1.0:
        return "Lean"
    elif ag < 2.0:
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
    rec = build_dataset()
    p = rec[(rec["season"] == season) & (rec["targets_roll"] >= 3)]
    return sorted(p["player_display_name"].dropna().unique().tolist())


def player_history(season, player_name, min_targets=0.5):
    model, feats = load_model()
    rec = build_dataset()
    p = rec[(rec["season"] == season) &
            (rec["player_display_name"] == player_name) &
            (rec["targets_roll"] >= min_targets)].dropna(subset=feats)
    if len(p) == 0:
        return pd.DataFrame()
    p = p.copy()
    p["projection"] = model.predict(p[feats]).round(1)
    p["snap_roll"] = (p["snap_roll"] * 100).round(0)
    out = p[["week", "opponent_team", "projection", "receptions",
             "targets_roll", "snap_roll"]].copy()
    out = out.rename(columns={"receptions": "actual"})
    return out.sort_values("week").reset_index(drop=True)


def actual_result(season, week, player_name):
    """Actual receptions for grading. Returns the number, or None."""
    df = build_dataset()
    m = df[(df["season"] == season) & (df["week"] == week) &
           (df["player_display_name"] == player_name)]
    if len(m) == 0:
        return None
    val = m.iloc[0]["receptions"]
    return None if pd.isna(val) else float(val)