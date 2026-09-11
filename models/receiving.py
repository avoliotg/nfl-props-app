"""
Receiving Yards market module — self-contained engine.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from . import data_utils
from sklearn.linear_model import LinearRegression

SEASONS = [2022, 2023, 2024, 2025, 2026]
LEAN_FEATS = ["target_share_roll", "targets_roll", "snap_roll",
              "ypt_roll", "team_spread", "total_line"]

# Per-feature EWMA halflives, in games. Replaces rolling(6).mean().
#
# WHY: lagged-regression gate showed last week carries 2.2x the weight of six
# weeks ago (z = +9.13 on targets, +8.68 on target_share), monotone decline,
# spearman +1.00. Halflives selected on 2024 and confirmed ONCE on 2025:
# paired bootstrap -0.394 MAE, CI [-0.582, -0.204] vs the old 6-game window.
#
# The pattern is interpretable: fast-moving usage signals (share, snaps) want a
# short halflife, slower efficiency signals (volume, yards per target) a long
# one. An unbounded EWMA also uses a player's whole history rather than
# truncating at six games, and cross-season carry-over beat a per-season reset
# (+0.395 MAE, CI [+0.100, +0.690]), which independently validates the
# prior-season bridge.
HL = {
    "target_share_roll": 2.0,
    "targets_roll": 12.0,
    "snap_roll": 2.0,
    "ypt_roll": 12.0,
    "receiving_yards_roll": 6.0,   # computed but not in LEAN_FEATS
}

# gap (abs yards) -> historical hit rate anchors from proxy-line testing
GAP_ANCHORS = [(0, 0.49), (3, 0.56), (7, 0.64), (15, 0.70), (30, 0.72)]


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_dataset():
    ps = data_utils.load_player_stats(SEASONS)
    rec = ps[ps["position"].isin(["WR", "TE", "RB"])].copy()
    rec = rec.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    for col in ["receiving_yards", "targets", "target_share"]:
        hl = HL[f"{col}_roll"]
        rec[f"{col}_roll"] = (rec.groupby("player_id")[col]
                              .transform(lambda s, h=hl: s.shift(1)
                                         .ewm(halflife=h, min_periods=1).mean()))
    rec["ypt_game"] = rec["receiving_yards"] / rec["targets"].replace(0, np.nan)
    rec["ypt_roll"] = (rec.groupby("player_id")["ypt_game"]
                       .transform(lambda s: s.shift(1)
                                  .ewm(halflife=HL["ypt_roll"], min_periods=1).mean()))

    snaps = data_utils.load_snap_counts(SEASONS)
    snaps_s = snaps[["season", "week", "team", "player", "offense_pct"]].copy()

    # Shared normalizer: also strips suffixes, which the old local version did
    # not. That silently dropped 546 rows (4.1%) whose snap join failed.
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
        subset=LEAN_FEATS + ["receiving_yards"])
    model = LinearRegression().fit(train[LEAN_FEATS], train["receiving_yards"])
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
    bridging rolling features from prior seasons. Used as a fallback when
    build_dataset() has no rows for the requested week because games haven't
    been played yet."""
    import nflreadpy as nfl

    # 1. Player universe from current-season rosters (correct 2026 teams)
    ros = nfl.load_rosters([season])
    ros = ros.to_pandas() if hasattr(ros, "to_pandas") else ros
    ros = ros[ros["position"].isin(["WR", "TE", "RB"])]
    ros = ros[["gsis_id", "full_name", "team", "position"]].rename(
        columns={"gsis_id": "player_id"})
    ros = ros.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])

    # 2. Bridge features from ALL prior seasons, not just one. The EWMA carries
    #    across the offseason, and carry-over beat a per-season reset by
    #    +0.395 MAE (CI [+0.100, +0.690]).
    rec = build_dataset()
    prior = rec[rec["season"] <= season - 1].sort_values(
        ["player_id", "season", "week"])

    def _ewm_last(g, col, key):
        """Final EWMA value over all prior games.

        build_dataset computes shift(1).ewm(...), so at a week-1 row that
        equals the unshifted EWMA through the last prior game. Computing it the
        same way here keeps training and serving features identical; any
        divergence would be train/serve skew, which produces plausible-looking
        wrong numbers rather than an error.
        """
        v = g[col].ewm(halflife=HL[key], min_periods=1).mean()
        return float(v.iloc[-1]) if len(v) else float("nan")

    def _bridge(g):
        return pd.Series({
            "targets_roll": _ewm_last(g, "targets", "targets_roll"),
            "target_share_roll": _ewm_last(g, "target_share", "target_share_roll"),
            "ypt_roll": _ewm_last(g, "ypt_game", "ypt_roll"),
            "snap_roll": _ewm_last(g, "offense_pct", "snap_roll"),
            "player_display_name": g.iloc[-1]["player_display_name"],
        })
    bridged = (prior.groupby("player_id", group_keys=False)
               .apply(_bridge, include_groups=False).reset_index())

    # 3. Join rosters (teams) + bridged features
    df = ros.merge(bridged, on="player_id", how="inner")

    # 4. Attach upcoming-week schedule context (opponent, spread, total)
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

    # 5. Drop players missing features (rookies / thin data self-correct)
    df["season"] = season
    df["week"] = week
    df = df.dropna(subset=LEAN_FEATS)
    return df.reset_index(drop=True)


# ---- shared confidence currency ----
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
    """Normalized 0-100 relative confidence. NOT a win probability.
    Linearly stretches the historical hit-rate range onto 0-100."""
    if gap is None or pd.isna(gap):
        return None
    ag = abs(gap)
    xs = [a[0] for a in GAP_ANCHORS]
    ys = [a[1] for a in GAP_ANCHORS]
    hit = float(np.interp(ag, xs, ys))          # ~0.49 .. ~0.72
    lo, hi = GAP_ANCHORS[0][1], GAP_ANCHORS[-1][1]
    score = (hit - lo) / (hi - lo) * 100         # stretch to 0..100
    return round(max(0, min(100, score)))


def actual_yards(season, week, player_name):
    """Return a player's actual receiving yards for a season/week, or None."""
    df = build_dataset()
    m = df[(df["season"] == season) & (df["week"] == week) &
           (df["player_display_name"] == player_name)]
    if len(m) == 0:
        return None
    val = m.iloc[0]["receiving_yards"]
    return None if pd.isna(val) else float(val)


def player_history(season, player_name, min_targets=0.5):
    """All of a player's weekly projections vs actuals for a season."""
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
    out = p[["week", "opponent_team", "projection", "receiving_yards",
             "targets_roll", "snap_roll"]].copy()
    out = out.rename(columns={"receiving_yards": "actual"})
    return out.sort_values("week").reset_index(drop=True)


def all_players(season):
    """List of players who qualify in a season (for the search dropdown)."""
    rec = build_dataset()
    p = rec[(rec["season"] == season) & (rec["targets_roll"] >= 2)]
    return sorted(p["player_display_name"].dropna().unique().tolist())


def actual_result(season, week, player_name):
    """Actual receiving yards for grading. Returns the number, or None."""
    df = build_dataset()
    m = df[(df["season"] == season) & (df["week"] == week) &
           (df["player_display_name"] == player_name)]
    if len(m) == 0:
        return None
    val = m.iloc[0]["receiving_yards"]
    return None if pd.isna(val) else float(val)