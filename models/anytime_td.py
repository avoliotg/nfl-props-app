"""
Anytime TD market module — CLASSIFICATION (probability), not regression.
Outputs a probability a player scores a TD, compared to the book's
implied probability from American odds. Edge = model prob − implied prob.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression

SEASONS = [2022, 2023, 2024, 2025]
FEATS = ["touches_roll", "td_rate_roll", "is_rb", "is_te"]

# This market is PROBABILITY-based. The shell checks this flag.
IS_PROBABILITY = True


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_dataset():
    ps = nfl.load_player_stats(SEASONS).to_pandas()
    s = ps[ps["position"].isin(["WR", "TE", "RB", "QB"])].copy()
    s["total_td"] = s["rushing_tds"].fillna(0) + s["receiving_tds"].fillna(0)
    s["scored"] = (s["total_td"] > 0).astype(int)
    s["touches"] = s["carries"].fillna(0) + s["targets"].fillna(0)
    s = s[s["touches"] >= 3].copy()
    s = s.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    s["td_rate_roll"] = (s.groupby("player_id")["scored"]
                         .transform(lambda x: x.shift(1).rolling(10, min_periods=4).mean()))
    s["touches_roll"] = (s.groupby("player_id")["touches"]
                         .transform(lambda x: x.shift(1).rolling(6, min_periods=3).mean()))
    s["is_rb"] = (s["position"] == "RB").astype(int)
    s["is_te"] = (s["position"] == "TE").astype(int)
    return s


@st.cache_resource(show_spinner="Training model...")
def load_model():
    s = build_dataset()
    train = s[(s["season"] <= 2024)].dropna(subset=FEATS + ["scored"])
    model = LogisticRegression(max_iter=1000).fit(train[FEATS], train["scored"])
    return model, FEATS


def available_seasons():
    return SEASONS


def available_weeks(season):
    s = build_dataset()
    return sorted(s[s["season"] == season]["week"].dropna().unique().tolist())


def project_week(season, week, min_touches=3):
    """Returns each player's model TD probability (as a %)."""
    model, feats = load_model()
    s = build_dataset()
    wk = s[(s["season"] == season) & (s["week"] == week)].copy()
    wk = wk[wk["touches_roll"] >= min_touches].dropna(subset=feats)
    if len(wk) == 0:
        return pd.DataFrame()
    wk["projection"] = (model.predict_proba(wk[feats])[:, 1] * 100).round(1)  # % chance
    wk["touches_roll"] = wk["touches_roll"].round(1)
    cols = ["player_display_name", "team", "opponent_team", "position",
            "projection", "touches_roll"]
    cols = [c for c in cols if c in wk.columns]
    return wk[cols].sort_values("projection", ascending=False).reset_index(drop=True)


# ---- odds <-> probability helpers ----
def american_to_prob(odds):
    """American odds -> implied probability (%)."""
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds > 0:
        return round(100 / (odds + 100) * 100, 1)
    else:
        return round(-odds / (-odds + 100) * 100, 1)


# ---- edge is in probability POINTS, so tiers differ ----
def tier_for_gap(gap):
    """gap here = model_prob% - implied_prob% (percentage points)."""
    if gap is None or pd.isna(gap):
        return ""
    ag = abs(gap)
    if ag < 3:
        return "Pass"
    elif ag < 7:
        return "Lean"
    elif ag < 12:
        return "Strong"
    else:
        return "Max"


def confidence_for_gap(gap):
    if gap is None or pd.isna(gap):
        return None
    # simple linear scale on the prob-point edge, capped
    ag = abs(gap)
    return round(min(100, ag / 15 * 100))


def all_players(season):
    s = build_dataset()
    p = s[(s["season"] == season) & (s["touches_roll"] >= 3)]
    return sorted(p["player_display_name"].dropna().unique().tolist())


def player_history(season, player_name, min_touches=3):
    model, feats = load_model()
    s = build_dataset()
    p = s[(s["season"] == season) &
          (s["player_display_name"] == player_name) &
          (s["touches_roll"] >= min_touches)].dropna(subset=feats)
    if len(p) == 0:
        return pd.DataFrame()
    p = p.copy()
    p["projection"] = (model.predict_proba(p[feats])[:, 1] * 100).round(1)
    p["scored_actual"] = p["scored"] * 100  # 0 or 100 (did/didn't score)
    out = p[["week", "opponent_team", "projection", "scored_actual", "touches_roll"]].copy()
    out = out.rename(columns={"scored_actual": "actual"})
    return out.sort_values("week").reset_index(drop=True)


def actual_result(season, week, player_name):
    """Did the player score a TD? Returns 100 (yes) or 0 (no), or None."""
    df = build_dataset()
    m = df[(df["season"] == season) & (df["week"] == week) &
           (df["player_display_name"] == player_name)]
    if len(m) == 0:
        return None
    return float(m.iloc[0]["scored"] * 100)