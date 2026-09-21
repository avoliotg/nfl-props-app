"""
Anytime TD market module — CLASSIFICATION (probability), not regression.
Outputs a probability a player scores a TD, compared to the book's
implied probability from American odds. Edge = model prob − implied prob.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from . import data_utils
from sklearn.linear_model import LogisticRegression

SEASONS = [2022, 2023, 2024, 2025, 2026]
FEATS = ["touches_roll", "td_rate_roll", "is_rb", "is_te"]

# This market is PROBABILITY-based. The shell checks this flag.
IS_PROBABILITY = True


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_all_rows():
    """Every active player-week at the relevant positions, features included,
    with NO volume filter applied.

    Rolling features are computed over every active game. Filtering first
    measured a low-usage player's TD rate over only his busiest weeks, so
    td_rate_roll read ~0.29 where the unconditional rate was ~0.25.
    """
    ps = data_utils.load_player_stats(SEASONS)
    s = ps[ps["position"].isin(["WR", "TE", "RB", "QB"])].copy()
    s["total_td"] = s["rushing_tds"].fillna(0) + s["receiving_tds"].fillna(0)
    s["scored"] = (s["total_td"] > 0).astype(int)
    s["touches"] = s["carries"].fillna(0) + s["targets"].fillna(0)
    s = s.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    s["td_rate_roll"] = (s.groupby("player_id")["scored"]
                         .transform(lambda x: x.shift(1).rolling(10, min_periods=4).mean()))
    s["touches_roll"] = (s.groupby("player_id")["touches"]
                         .transform(lambda x: x.shift(1).rolling(6, min_periods=3).mean()))
    s["is_rb"] = (s["position"] == "RB").astype(int)
    s["is_te"] = (s["position"] == "TE").astype(int)
    return s


@st.cache_data(show_spinner=False)
def build_dataset():
    """Rows with real in-game volume. Unchanged semantics for the board."""
    s = build_all_rows()
    return s[s["touches"] >= 3].copy()


@st.cache_resource(show_spinner="Training model...")
def load_model():
    # Train on the population the board actually serves, from the UNFILTERED
    # frame. `touches >= 3` conditions on the current game: a player who
    # finishes with 2 touches is likelier not to have scored, so training on
    # that population leaks the outcome and over-states P(score).
    # `touches_roll >= 3` uses only lagged history, which is the same rule the
    # board uses to decide who to display. Going fully unconditional overshoots
    # the other way, because 0-2 touch players have a far lower base rate.
    # Backtested on 2025: log loss 0.5452 vs 0.5537 shipped, overall gap
    # +0.3 pp (z +0.39) vs -2.1 (z -2.81).
    s = build_all_rows()
    train = s[(s["season"] <= 2024) & (s["touches_roll"] >= 3)].dropna(
        subset=FEATS + ["scored"])
    model = LogisticRegression(max_iter=1000).fit(train[FEATS], train["scored"])
    return model, FEATS


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


def project_week(season, week, min_touches=1.5):
    """Returns each player's model TD probability (as a %)."""
    model, feats = load_model()
    s = build_dataset()
    wk = s[(s["season"] == season) & (s["week"] == week)].copy()

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

    wk = wk[wk["touches_roll"] >= min_touches].dropna(subset=feats)

    if len(wk) == 0:
        return pd.DataFrame()
    wk["projection"] = (model.predict_proba(wk[feats])[:, 1] * 100).round(1)  # % chance
    wk["touches_roll"] = wk["touches_roll"].round(1)
    cols = ["player_display_name", "team", "opponent_team", "position",
            "projection", "touches_roll"]
    cols = [c for c in cols if c in wk.columns]
    return wk[cols].sort_values("projection", ascending=False).reset_index(drop=True)


def build_upcoming_week(season, week):
    """Manufacture player-week rows for a game not yet played (e.g. Week 1),
    bridging touches and TD rate from the prior season. Fallback when
    build_dataset() has no rows for the requested week. TD uses no schedule
    context for the model, but we attach opponent_team for the user's reference."""
    import nflreadpy as nfl

    ros = nfl.load_rosters([season])
    ros = ros.to_pandas() if hasattr(ros, "to_pandas") else ros
    ros = ros[ros["position"].isin(["WR", "TE", "RB", "QB"])]
    ros = ros[["gsis_id", "full_name", "team", "position"]].rename(
        columns={"gsis_id": "player_id"})
    ros = ros.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])

    s = build_dataset()
    prior = s[s["season"] == season - 1].sort_values(["player_id", "week"])

    def _bridge(g):
        last10 = g.tail(10)   # td_rate uses a 10-game window
        last6 = g.tail(6)     # touches uses a 6-game window
        return pd.Series({
            "td_rate_roll": last10["scored"].mean(),
            "touches_roll": last6["touches"].mean(),
            "player_display_name": g.iloc[-1]["player_display_name"],
        })
    bridged = (prior.groupby("player_id", group_keys=False)
               .apply(_bridge, include_groups=False).reset_index())

    df = ros.merge(bridged, on="player_id", how="inner")

    # position flags
    df["is_rb"] = (df["position"] == "RB").astype(int)
    df["is_te"] = (df["position"] == "TE").astype(int)

    # opponent lookup (for user reference only — TD model doesn't use it)
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
    df = df.dropna(subset=FEATS)
    return df.reset_index(drop=True)


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


def player_history(season, player_name, min_touches=0.5):
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


@st.cache_data(show_spinner=False)
def _all_player_stats():
    """Unfiltered player stats for grading.

    actual_result must NOT read build_dataset: that frame applies
    `touches >= 3`, so any player with 0, 1 or 2 touches has no row and
    returned None, and the bet silently never graded. Those players are
    almost all non-scorers, so the rows that vanished were overwhelmingly
    LOSSES, which flattered every TD calibration number.
    """
    ps = data_utils.load_player_stats(SEASONS)
    return ps.to_pandas() if hasattr(ps, "to_pandas") else ps


def actual_result(season, week, player_name):
    """Did the player score a TD? Returns 100 (yes) or 0 (no), or None.

    Three distinct cases, and conflating them is what caused the bug:
      - played and scored           -> 100
      - played and did not score    -> 0    (a real loss, previously invisible)
      - did not play at all         -> None (genuinely ungradeable)
    """
    ps = _all_player_stats()
    m = ps[(ps["season"] == season) & (ps["week"] == week) &
           (ps["player_display_name"] == player_name)]
    if len(m) == 0:
        # normalised fallback: FanDuel and nflverse spell names differently.
        # norm_join_name is vectorised, so both sides need a Series.
        wk = ps[(ps["season"] == season) & (ps["week"] == week)].copy()
        if len(wk):
            target = data_utils.norm_join_name(pd.Series([player_name])).iloc[0]
            wk["_norm"] = data_utils.norm_join_name(wk["player_display_name"])
            m = wk[wk["_norm"] == target]
    if len(m) == 0:
        return None  # no stat row at all, so the player was inactive
    row = m.iloc[0]
    rush_td = row.get("rushing_tds")
    rec_td = row.get("receiving_tds")
    rush_td = 0.0 if pd.isna(rush_td) else float(rush_td)
    rec_td = 0.0 if pd.isna(rec_td) else float(rec_td)
    return 100.0 if (rush_td + rec_td) > 0 else 0.0
