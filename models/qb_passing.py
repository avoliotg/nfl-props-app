"""
QB Passing Yards market module.

PHASE 3.3: the attempts >= 10 survivorship fix, following the same shape as
the rushing fix and anytime_td before it.

WHAT WAS WRONG, AND IT IS THE SAME FOUR THINGS AS RUSHING

  1. attempts_roll WAS COMPUTED AFTER THE VOLUME FILTER.
     build_dataset filtered `attempts >= 10` and then took a 6-game rolling
     mean of attempts, so a QB's workload feature was measured only over the
     games he stayed in. Benched, injured and blowout games are deleted, which
     removes the LEFT TAIL of the outcome distribution. The handoff names this
     as the likely mechanism behind the +8.81 bias that appears in all sixteen
     subgroup cells, and the bias being slope-shaped (+1.06 to +10.48 across
     quintiles) is consistent with a workload feature that is inflated most
     for the QBs who get pulled most.

  2. THE TRAINING POPULATION DID NOT MATCH THE SERVED POPULATION.
     load_model trained on `attempts_roll >= 10`, project_week serves
     `attempts_roll >= 1.5`. Both now read MIN_ATTEMPTS_ROLL.

  3. build_upcoming_week BRIDGED FROM THE FILTERED FRAME, so every Week 1
     projection inherited the inflation. It now reads build_all_rows().

  4. actual_result READ build_dataset(), so a QB who threw 9 attempts had no
     row, returned None, and the bet silently never graded. Exactly the bug
     already fixed in rushing and anytime_td. It now reads unfiltered stats,
     and a null passing line for a QB with a stat row is treated as a genuine
     zero rather than as missing data.

  AND ONE THING RUSHING DID NOT HAVE:

  5. A FULL-SAMPLE STATISTIC LEAK.
     `qb["def_pass_roll"].fillna(qb["def_pass_roll"].mean())` took the mean
     over every season in the frame, so a 2023 row was filled with a number
     computed partly from 2025. eval_harness walk-forwards the model
     COEFFICIENTS but cannot undo a leak baked into a feature. The fill is now
     an expanding mean in chronological order, which uses only prior games.

     Residual caveat: the earliest rows of 2022 have no prior at all and fall
     back to the first available expanding value. That is a handful of rows in
     the oldest season and it cannot affect 2023 onward.

  Also improved while here: def_pass_roll is now computed from ALL QB rows
  rather than only 10-plus-attempt rows. Team passing yards allowed should
  count every pass thrown against that defense, including by a QB who left
  early.

WHAT DELIBERATELY DID NOT CHANGE

  build_dataset still applies `attempts >= 10`, so the board, all_players and
  player_history behave as before. Only the ROLLING features, the TRAINING
  population, the Week 1 bridge and GRADING moved.

CONSEQUENCES TO MEASURE, NOT ASSUME

  The projection distribution changes, so the shipped sigma 0.3531*proj and
  the floor 159.0 were fitted against the OLD projections.

  MIN_ATTEMPTS_ROLL at 1.5 is the serving threshold, so matching it is the
  principled choice. But anytime_td variant C showed that going too far toward
  the unconditional population OVERCORRECTS, because the wider population has
  a different base rate. A QB with attempts_roll near 1.5 is a wildcat or
  emergency arm, not a starter. Run filter_diagnostics() and compare the
  training mean against the unconditional mean before accepting 1.5.
"""
import numpy as np
import pandas as pd
import streamlit as st
import nflreadpy as nfl
from . import data_utils
from sklearn.linear_model import LinearRegression

SEASONS = [2022, 2023, 2024, 2025, 2026]
LEAN_FEATS = ["attempts_roll", "team_spread", "total_line", "wind_eff", "def_pass_roll"]

# The board's display threshold AND the training threshold. One constant, so
# the population the model is fitted on cannot drift from the one it scores.
MIN_ATTEMPTS_ROLL = 1.5

# Last season included in training by default. Parameterised so eval_harness
# can score walk-forward instead of always training through 2024.
DEFAULT_TRAIN_MAX_SEASON = 2024

# passing yards are big numbers - tiers in yards, bigger gaps than receiving
GAP_ANCHORS = [(0, 0.49), (10, 0.54), (25, 0.60), (50, 0.66), (90, 0.70)]


@st.cache_data(show_spinner="Pulling & preparing NFL data (first run only)...")
def build_all_rows():
    """Every active QB player-week, features included, NO attempts filter.

    Position is a structural attribute and safe to filter on. Attempts are an
    in-game result, so filtering on them before a rolling mean is the bug.
    """
    ps = data_utils.load_player_stats(SEASONS)
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    qb = ps[ps["position"] == "QB"].copy()

    # A QB with a stat row was active. No attempts means zero attempts and
    # zero yards, not unknown.
    qb["attempts"] = qb["attempts"].fillna(0)
    qb["passing_yards"] = qb["passing_yards"].fillna(0.0)

    qb = qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    qb["attempts_roll"] = (qb.groupby("player_id")["attempts"]
                           .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean()))

    games = data_utils.load_schedules(SEASONS)
    games = games.to_pandas() if hasattr(games, "to_pandas") else games
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

    # Opponent pass defense: passing yards allowed, rolling. Computed from ALL
    # QB rows, so a defence that faced a QB who left after 8 attempts is still
    # credited with those yards.
    qd = (qb.groupby(["season", "week", "opponent_team"])["passing_yards"].sum()
          .reset_index().rename(columns={"opponent_team": "defteam", "passing_yards": "pa"}))
    qd = qd.sort_values(["defteam", "season", "week"])
    qd["def_pass_roll"] = (qd.groupby("defteam")["pa"]
                           .transform(lambda s: s.shift(1).rolling(6, min_periods=3).mean()))
    qb = qb.merge(qd[["season", "week", "defteam", "def_pass_roll"]].rename(
        columns={"defteam": "opponent_team"}), on=["season", "week", "opponent_team"], how="left")

    # Thin or missing opponent-defence data falls back to the league average.
    # THE FALLBACK MUST NOT SEE THE FUTURE. The previous version used
    # qb["def_pass_roll"].mean() over all five seasons, so a 2023 row was
    # filled from data including 2025. An expanding mean in chronological order
    # uses only games already played.
    qb = qb.sort_values(["season", "week", "team"]).reset_index(drop=True)
    prior_mean = qb["def_pass_roll"].expanding().mean().shift(1)
    qb["def_pass_roll"] = qb["def_pass_roll"].fillna(prior_mean)
    # the very first rows of 2022 have no prior; use the earliest available
    # expanding value rather than the global mean
    first_valid = prior_mean.dropna()
    if len(first_valid):
        qb["def_pass_roll"] = qb["def_pass_roll"].fillna(first_valid.iloc[0])
    return qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_dataset():
    """Rows with real in-game volume. Unchanged semantics for the board."""
    qb = build_all_rows()
    return qb[qb["attempts"] >= 10].copy()


@st.cache_resource(show_spinner="Training model...")
def load_model(train_max_season=DEFAULT_TRAIN_MAX_SEASON,
               train_min_attempts_roll=MIN_ATTEMPTS_ROLL):
    """Fit on the SERVED population, read from the unfiltered frame.

    `attempts >= 10` conditions on the current game: a QB who finishes with 6
    attempts was benched, hurt or in a blowout, all of which depress passing
    yards, so training on that population deletes the left tail.
    `attempts_roll >= 1.5` uses only lagged history, which is the rule
    project_week applies to decide who appears on the board.
    """
    qb = build_all_rows()
    train = qb[(qb["season"] <= train_max_season) &
               (qb["attempts_roll"] >= train_min_attempts_roll)].dropna(
        subset=LEAN_FEATS + ["passing_yards"])
    model = LinearRegression().fit(train[LEAN_FEATS], train["passing_yards"])
    return model, LEAN_FEATS


def filter_diagnostics(season=2025):
    """Before and after counts, so the change is measured rather than assumed.

    Watch the unconditional mean against the training mean. anytime_td variant
    C overcorrected by training on a population with a materially lower base
    rate than the served one, and a QB at attempts_roll 1.5 is an emergency
    arm rather than a starter.
    """
    allq = build_all_rows()
    filt = build_dataset()
    a = allq[allq["season"] == season]
    f = filt[filt["season"] == season]
    rows = [
        ("all active QB rows", a),
        ("attempts >= 10 (board frame)", f),
        (f"NEW train/serve gate, roll >= {MIN_ATTEMPTS_ROLL}",
         a[a["attempts_roll"] >= MIN_ATTEMPTS_ROLL]),
        ("OLD train gate, roll >= 10 inside filtered",
         f[f["attempts_roll"] >= 10]),
    ]
    out = []
    for name, fr in rows:
        out.append({
            "frame": name,
            "rows": len(fr),
            "players": fr["player_id"].nunique(),
            "mean_attempts_roll": round(float(fr["attempts_roll"].mean()), 2)
            if len(fr) else None,
            "mean_pass_yds": round(float(fr["passing_yards"].mean()), 1)
            if len(fr) else None,
        })
    return pd.DataFrame(out)


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


def project_week(season, week, min_attempts=MIN_ATTEMPTS_ROLL):
    model, feats = load_model()
    qb = build_dataset()
    wk = qb[(qb["season"] == season) & (qb["week"] == week)].copy()

    # Union played rows with the assembler at the FEATURE level, then score
    # once. Using the assembler only when the played set was EMPTY broke any
    # week where some games have finished and others have not: after the
    # Wednesday opener this returned only NE and SEA, so every other team
    # vanished from the board, from import_lines (which stored null
    # projections), and from the export team map.
    up = build_upcoming_week(season, week)
    if len(up) > 0:
        wk = pd.concat([wk, up], ignore_index=True)
        if "player_id" in wk.columns:
            # played rows come first, so they win a duplicate: real results
            # beat a bridged estimate
            wk = wk.drop_duplicates(subset=["player_id"], keep="first")

    wk = wk[wk["attempts_roll"] >= min_attempts].dropna(subset=feats)

    if len(wk) == 0:
        return pd.DataFrame()
    wk["projection"] = model.predict(wk[feats]).round(1)
    wk["attempts_roll"] = wk["attempts_roll"].round(1)
    cols = ["player_display_name", "team", "opponent_team", "position",
            "projection", "attempts_roll"]
    cols = [c for c in cols if c in wk.columns]
    return wk[cols].sort_values("projection", ascending=False).reset_index(drop=True)


def build_upcoming_week(season, week):
    """Manufacture QB player-week rows for a game not yet played (e.g. Week 1),
    bridging attempts and opponent pass-defense from the prior season.
    Fallback when build_dataset() has no rows for the requested week.

    Bridges from build_all_rows, NOT build_dataset. Reading the filtered frame
    averaged only the games a QB threw 10 or more times, so every Week 1
    projection carried the same survivorship inflation as the rolling feature.
    """
    import nflreadpy as nfl

    ros = nfl.load_rosters([season])
    ros = ros.to_pandas() if hasattr(ros, "to_pandas") else ros
    ros = ros[ros["position"] == "QB"]
    ros = ros[["gsis_id", "full_name", "team", "position"]].rename(
        columns={"gsis_id": "player_id"})
    ros = ros.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])

    # depth chart -> each team's current QB1 (most recent snapshot); starters only
    dc = nfl.load_depth_charts([season])
    dc = dc.to_pandas() if hasattr(dc, "to_pandas") else dc
    dc = dc[dc["pos_abb"] == "QB"].copy()
    latest_dt = dc["dt"].max()
    dc = dc[dc["dt"] == latest_dt]
    starters = dc[dc["pos_rank"] == 1][["gsis_id"]].rename(columns={"gsis_id": "player_id"})
    ros = ros.merge(starters, on="player_id", how="inner")

    qb = build_all_rows()
    prior = qb[qb["season"] == season - 1].sort_values(["player_id", "week"])

    def _bridge(g):
        last6 = g.tail(6)
        return pd.Series({
            "attempts_roll": last6["attempts"].mean(),
            "player_display_name": g.iloc[-1]["player_display_name"],
        })
    bridged = (prior.groupby("player_id", group_keys=False)
               .apply(_bridge, include_groups=False).reset_index())

    df = ros.merge(bridged, on="player_id", how="inner")

    # schedule context incl. roof/wind for wind_eff
    sched = nfl.load_schedules([season])
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    wk = sched[sched["week"] == week]
    home = wk[["home_team", "away_team", "spread_line", "total_line", "roof", "wind"]].rename(
        columns={"home_team": "team", "away_team": "opponent_team"})
    home["team_spread"] = home["spread_line"]
    away = wk[["away_team", "home_team", "spread_line", "total_line", "roof", "wind"]].rename(
        columns={"away_team": "team", "home_team": "opponent_team"})
    away["team_spread"] = -away["spread_line"]
    ctx = pd.concat([home, away], ignore_index=True)[
        ["team", "opponent_team", "team_spread", "total_line", "roof", "wind"]]
    df = df.merge(ctx, on="team", how="inner")

    df["is_outdoor"] = df["roof"].isin(["outdoors", "open"]).astype(int)
    df["wind_eff"] = np.where(df["is_outdoor"] == 1, df["wind"].fillna(7), 0)

    # opponent pass defense: bridge each defence's end-of-prior-season value
    prior_def = qb[qb["season"] == season - 1].copy()
    def_last = (prior_def.sort_values(["opponent_team", "week"])
                .groupby("opponent_team")["def_pass_roll"].last().reset_index()
                .rename(columns={"opponent_team": "opp_ref", "def_pass_roll": "def_pass_roll_bridged"}))
    df = df.merge(def_last, left_on="opponent_team", right_on="opp_ref", how="left")
    df["def_pass_roll"] = df["def_pass_roll_bridged"].fillna(df["def_pass_roll_bridged"].mean())

    df["season"] = season
    df["week"] = week
    df = df.dropna(subset=LEAN_FEATS)
    return df.reset_index(drop=True)


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


@st.cache_data(show_spinner=False)
def _all_player_stats():
    """Unfiltered player stats for grading.

    actual_result must NOT read build_dataset: that frame applies
    `attempts >= 10`, so a QB who threw 9 times had no row, returned None, and
    the bet silently never graded. Those games are benched, injured and
    blowout games, which are disproportionately UNDERS, so the missing grades
    flattered the measured record. Same bug family as the rushing and
    anytime_td fixes.
    """
    ps = data_utils.load_player_stats(SEASONS)
    return ps.to_pandas() if hasattr(ps, "to_pandas") else ps


def actual_result(season, week, player_name):
    """Actual passing yards for grading. Returns the number, or None."""
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
    val = m.iloc[0]["passing_yards"]
    # a QB with a stat row was active, so a missing passing line is a genuine
    # zero, not an absence of data
    return 0.0 if pd.isna(val) else float(val)
