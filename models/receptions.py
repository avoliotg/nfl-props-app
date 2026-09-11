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

# Per-feature EWMA halflives, in games. Replaces rolling(6).mean().
#
# Selected on 2024 and confirmed ONCE on 2025: paired bootstrap -0.024 MAE,
# CI [-0.035, -0.013] vs the old 6-game window, with corr 0.431 -> 0.453.
#
# Note targets_roll wants 3 here while the receiving model wants 12, which is
# why each market carries its own dict rather than sharing one. Receptions is a
# count on a compressed scale, so recent volume matters more and long memory
# adds little.
HL = {
    "target_share_roll": 2.0,
    "targets_roll": 3.0,
    "snap_roll": 2.0,
}

# receptions are small numbers — tiers are in CATCHES, not yards
GAP_ANCHORS = [(0, 0.49), (0.5, 0.55), (1.0, 0.60), (2.0, 0.66), (4.0, 0.70)]


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_dataset():
    ps = data_utils.load_player_stats(SEASONS)
    rec = ps[ps["position"].isin(["WR", "TE", "RB"])].copy()
    rec = rec.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for col in ["targets", "target_share"]:
        hl = HL[f"{col}_roll"]
        rec[f"{col}_roll"] = (rec.groupby("player_id")[col]
                              .transform(lambda s, h=hl: s.shift(1)
                                         .ewm(halflife=h, min_periods=1).mean()))

    snaps = data_utils.load_snap_counts(SEASONS)
    snaps_s = snaps[["season", "week", "team", "player", "offense_pct"]].copy()

    # Shared normalizer: also strips suffixes and maps nicknames, which the old
    # local version did not. That silently dropped rows whose snap join failed.
    rec["_join_name"] = data_utils.norm_join_name(rec["player_display_name"])
    snaps_s["_join_name"] = data_utils.norm_join_name(snaps_s["player"])
    snaps_s = snaps_s[["season", "week", "team", "_join_name", "offense_pct"]]

    rec = rec.merge(snaps_s, on=["season", "week", "team", "_join_name"], how="left")
    rec = rec.drop(columns=["_join_name"])
    rec["snap_roll"] = (rec.groupby("player_id")["offense_pct"]
                        .transform(lambda s: s.shift(1)
                                   .ewm(halflife=HL["snap_roll"], min_periods=1).mean()))

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
    played = sorted(rec[rec["season"] == season]["week"].dropna().unique().tolist())
    if played:
        return played
    # no played games yet (pre-season) → only Week 1 is sensibly projectable
    return [1]


def project_week(season, week, min_targets=1.5):
    model, feats = load_model()
    rec = build_dataset()
    wk = rec[(rec["season"] == season) & (rec["week"] == week)].copy()

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

    wk = wk[wk["targets_roll"] >= min_targets].dropna(subset=feats)

    if len(wk) == 0:
        return pd.DataFrame()
    wk["projection"] = model.predict(wk[feats]).round(1)
    wk["snap_roll"] = (wk["snap_roll"] * 100).round(0)
    cols = ["player_display_name", "team", "opponent_team", "position",
            "projection", "targets_roll", "snap_roll"]
    cols = [c for c in cols if c in wk.columns]
    return wk[cols].sort_values("projection", ascending=False).reset_index(drop=True)


def build_upcoming_week(season, week):
    """Manufacture player-week rows for a game not yet played (e.g. Week 1),
    bridging rolling features from prior seasons. Fallback when build_dataset()
    has no rows for the requested week."""
    import nflreadpy as nfl

    ros = nfl.load_rosters([season])
    ros = ros.to_pandas() if hasattr(ros, "to_pandas") else ros
    ros = ros[ros["position"].isin(["WR", "TE", "RB"])]
    ros = ros[["gsis_id", "full_name", "team", "position"]].rename(
        columns={"gsis_id": "player_id"})
    ros = ros.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])

    # ALL prior seasons, not just one. The EWMA carries across the offseason,
    # and carry-over beat a per-season reset by +0.034 MAE (CI [+0.018,+0.050]).
    # Bridging from a single season was also the source of a train/serve
    # mismatch: 13-15 players of 263 had features that did not match what
    # build_dataset produced for the same week.
    rec = build_dataset()
    prior = rec[rec["season"] <= season - 1].sort_values(
        ["player_id", "season", "week"])

    def _ewm_last(g, col, key):
        """Final EWMA value over all prior games.

        build_dataset computes shift(1).ewm(...), so at a week-1 row that equals
        the unshifted EWMA through the last prior game. Computing it identically
        here is what keeps training and serving features the same; divergence
        would be train/serve skew, which yields plausible-looking wrong numbers
        rather than an error.
        """
        v = g[col].ewm(halflife=HL[key], min_periods=1).mean()
        return float(v.iloc[-1]) if len(v) else float("nan")

    def _bridge(g):
        return pd.Series({
            "targets_roll": _ewm_last(g, "targets", "targets_roll"),
            "target_share_roll": _ewm_last(g, "target_share", "target_share_roll"),
            "snap_roll": _ewm_last(g, "offense_pct", "snap_roll"),
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
    p = rec[(rec["season"] == season) & (rec["targets_roll"] >= 2)]
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