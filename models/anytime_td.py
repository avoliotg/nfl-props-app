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

# ONE constant for the training gate AND the board threshold. They were 3 and
# 1.5 respectively, so the model was fitted on established contributors and
# then asked to score players with half that workload. Measured on 2025: the
# 1.5-to-3 band predicted 0.1410 against a realized 0.0857, over-predicting by
# 65 percent in relative terms. Same defect as rushing.py note 2, where
# training on carries_roll >= 5 while serving 1.5 inflated the intercept by
# roughly 66 percent for backups.
#
# load_model's docstring previously ASSERTED that the two already matched.
# They did not. A single constant makes the claim true by construction.
MIN_TOUCHES_ROLL = 1.5

# Minimum prior games before a rolling feature means anything. These mirror
# the min_periods in build_all_rows, and build_upcoming_week now enforces
# them too. See _bridge.
MIN_GAMES_TD_RATE = 4
MIN_GAMES_TOUCHES = 3

# This market is PROBABILITY-based. The shell checks this flag.
IS_PROBABILITY = True

# EDGE, SIDE AND TIER ARE WITHHELD until the three September 25 fixes above
# have been re-measured on a full season. Plan item J1 suppresses those
# columns where BETA is clipped to zero, and this market has no beta, so
# nothing else stops the board recommending a bet built on a fabricated
# scoring rate.
#
# What was on screen: fringe players at 27 to 40 percent against market
# prices of 3 to 10, a one-sided disagreement of 22 to 34 points, every row
# tagged Max. market_calibration.py measured this book as well calibrated, so
# a gap that size is the model being wrong, not the market.
#
# Calibration on the served population is also genuinely poor in the range
# these rows land in: the (0.10, 0.15] bin predicted 0.1337 against a
# realized 0.0864, z +5.5.
#
# SET THIS TO FALSE once td_calibration.py shows the bias in the 1.5-to-3
# band and the (0.10, 0.15] bin has come down. The probability itself is
# still displayed; only the verdict is withheld.
CALIBRATION_SUSPENDED = True
SUSPENSION_REASON = (
    "Anytime TD is shown for reference only. Three defects found on "
    "September 25 2026 all inflated the model probability for low-volume "
    "players: the training gate did not match the board threshold, the "
    "week-1 bridge measured scoring rate over a player's busiest weeks "
    "only, and it enforced no minimum game count, so a single prior game "
    "with a touchdown became a 100 percent scoring rate. All three are "
    "fixed, and the verdict stays withheld until a full season "
    "re-measures the calibration."
)


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
    # `touches_roll` uses only lagged history, which is the same rule the
    # board uses to decide who to display. Going fully unconditional overshoots
    # the other way, because 0-2 touch players have a far lower base rate.
    #
    # CORRECTED September 25 2026. This comment used to say the gate was 3 and
    # that 3 was "the same rule the board uses". The board used 1.5. Both now
    # read MIN_TOUCHES_ROLL, so the sentence is true by construction rather
    # than by assertion. Refitting at the serving gate cut the 1.5-to-3 band's
    # bias from +0.0554 to +0.0376 on 2025.
    # Backtested on 2025: log loss 0.5452 vs 0.5537 shipped, overall gap
    # +0.3 pp (z +0.39) vs -2.1 (z -2.81).
    s = build_all_rows()
    train = s[(s["season"] <= 2024) &
              (s["touches_roll"] >= MIN_TOUCHES_ROLL)].dropna(
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


def project_week(season, week, min_touches=MIN_TOUCHES_ROLL):
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


# PERFORMANCE, September 25. This function had NO cache decorator in any of
# the six market modules, and project_week calls it unconditionally on every
# call. Streamlit re-runs the whole script on each widget interaction, so
# every dropdown click re-read rosters and schedules (and depth charts, for
# the QB markets) from scratch, per market. app.py renders the rushing board
# from BOTH rushing.project_week and qb_rushing.project_week, so that board
# paid for two full assemblers.
#
# TTL RATHER THAN A PLAIN CACHE, DELIBERATELY. What this loads is the LIVE
# data: rosters and depth charts change mid-week when a starter is ruled out,
# and a permanent cache would serve a stale QB1 for the rest of the week.
# Thirty minutes keeps the board responsive while still picking up news
# within one refresh cycle. Lower it if that feels too slow to react; the
# cost of a miss is one rebuild.
@st.cache_data(ttl=1800, show_spinner=False)
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

    # build_all_rows, NOT build_dataset. The filtered frame applies
    # `touches >= 3`, an IN-GAME quantity, so td_rate_roll was the mean of
    # `scored` over a player's BUSIEST weeks only. Measured on the 2024 bridge:
    # td_rate_roll read 0.2121 against an unfiltered 0.1847, and the inflation
    # was concentrated in the thinnest quartile (+0.0324, +0.0436, +0.0242,
    # +0.0068). touches_roll was inflated by +1.26, which matters twice over
    # since it also decides who clears MIN_TOUCHES_ROLL. Same defect as
    # rushing.py note 3.
    s = build_all_rows()
    prior = s[s["season"] == season - 1].sort_values(["player_id", "week"])

    def _bridge(g):
        """Bridge the rolling features, ENFORCING build_all_rows' minimums.

        THE BUG THIS FIXES, AND IT WAS THE WORST OF THE THREE.

        build_all_rows computes td_rate_roll as
        shift(1).rolling(10, min_periods=4) and touches_roll as
        shift(1).rolling(6, min_periods=3). This function applied NO minimum,
        and a mean over one row is that row's value. So a bridged player with
        a single qualifying prior game in which he scored received
        td_rate_roll = 1.0, handed straight to the model's strongest feature.

        Measured on the 2024 bridge: 136 of 460 players (29.6 percent) fell
        below the 4-game minimum, 14 of those carried td_rate_roll >= 0.5, and
        NINE carried exactly 1.0. Every one of the nine came from the one-game
        bucket. build_all_rows would have returned NaN for all of them and
        project_week's dropna would have removed them from the board.

        That is how the September 25 board showed fringe players at 27 to 40
        percent against market prices of 3 to 10, every row tagged Max.

        Returning NaN here is the point: it restores the drop that
        build_all_rows would have performed.
        """
        last10 = g.tail(10)   # td_rate uses a 10-game window
        last6 = g.tail(6)     # touches uses a 6-game window
        return pd.Series({
            "td_rate_roll": (last10["scored"].mean()
                             if len(last10) >= MIN_GAMES_TD_RATE
                             else float("nan")),
            "touches_roll": (last6["touches"].mean()
                             if len(last6) >= MIN_GAMES_TOUCHES
                             else float("nan")),
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


def actual_result(season, week, player_name):
    """Did the player score a TD? Returns 100 (yes) or 0 (no), or None.

    Three distinct cases, and conflating them is what caused the original bug:
      - played and scored           -> 100
      - played and did not score    -> 0    (a real loss, previously invisible)
      - did not play at all         -> None (genuinely ungradeable)

    Now delegates to data_utils.actual_stat, TWICE, because this market needs
    two columns summed and the helper returns one. That is fine: each call
    applies the same normalised fallback and the same collision guard, and
    both return 0.0 rather than None when the row exists with a null stat.

    So `both None` means no stat row at all, which is the ungradeable case,
    and anything else means the player was active.

    THE GUARD MATTERS MORE HERE THAN ANYWHERE. This market spans WR, TE, RB
    and QB, which is the widest population of the five, so it is the most
    exposed to the 21 duplicate normalized names name_resolve.py measured.
    The position list narrows it to the four offensive groups, and the guard
    refuses whatever ambiguity survives that.

    Kept deliberately: reading UNFILTERED rather than build_dataset(). That
    frame applies `touches >= 3`, so a player with 0 to 2 touches had no row
    and returned None, and the bet silently never graded. Those players are
    almost all non-scorers, so the rows that vanished were overwhelmingly
    LOSSES, which flattered every TD calibration number.
    """
    pos = ["WR", "TE", "RB", "QB"]
    rush = data_utils.actual_stat(season, week, player_name, "rushing_tds",
                                  seasons=SEASONS, position=pos)
    rec = data_utils.actual_stat(season, week, player_name, "receiving_tds",
                                 seasons=SEASONS, position=pos)
    if rush is None and rec is None:
        # No stat row, or an unresolved name collision. Either way this is not
        # gradeable, and a missing grade beats a wrong one.
        return None
    total = (rush or 0.0) + (rec or 0.0)
    return 100.0 if total > 0 else 0.0
