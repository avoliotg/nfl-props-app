"""
rules.py - the validated side-picking rules, evaluated against a week's props.

WHAT THIS IS AND WHAT IT IS NOT

    This module is SEPARATE from mc_pricing and does not touch it. The two
    answer different questions:

        mc_pricing  given a projection and a line, what is P(over)?
        rules.py    does a rule with measured out-of-sample support fire on
                    this prop, and on which side?

    That separation is deliberate, because two of the three surviving
    mechanisms are PRICING BIASES that involve no projection at all. Routing
    them through the pricing layer would mean inventing a projection-based
    story for an effect that has nothing to do with the projection, and
    rushing's beta is -0.041 precisely because the projection adds nothing
    there.

    This module makes no claim about probability, edge size, or stake. It
    answers: fired, side, which rule, and why not when it did not.

THE RULES, AND THEIR PROVENANCE

    Every constant below was measured, not chosen. Each rule carries the
    script that produced it, the sample, and the interval, so a future
    session can see what it would be overriding.

    1. RUSHING LOW-LINE UNDER            rushing_lowline.py
       market == 'rushing' and line <= 46.5, bet the UNDER.
       ROI +0.0589, 95% [+0.0115, +0.1075], about 438 bets/season.
       Positive in all four seasons (+0.0556, +0.0784, +0.0464, +0.0485).
       Smooth threshold decay from 20.5 to 55.5. Holds at ordinary prices.
       Market-wide: six of seven stable books show under win rates in a band
       of 0.548 to 0.561, so this is not one book's model. FanDuel is the
       place to play it because its hold is 4.92 percent against 6.02 to 7.07
       elsewhere.
       The MODEL IS NOT USED. The conjunction test found nothing, which is
       expected where beta spans zero.

    2. RECEIVING AIR-YARDS UNDER         air_yards_retest.py,
                                         conjunction_followup.py
       market == 'receiving', air-yards shock z >= 1.0, bet the UNDER,
       EXCEPT where the projection sits well above the line.
       Lift over the market's own under baseline: +0.086 at z >= 1.0,
       +0.113 at 1.5, +0.148 at 2.0. About 250 bets/season.
       Invariant across five definitions of the veto (+0.086 to +0.107).
       Best-of-grid placebo p = 0.072, which is BORDERLINE, and season ROI
       decays +0.165, +0.102, +0.070.
       The MODEL IS A VETO, NOT A CONFIRMATION. Inside the rule cell the
       under-side ROI by dev quintile runs +0.100, +0.079, +0.127, +0.086,
       -0.117. So the projection identifies one bad bucket and adds nothing
       directional elsewhere.

    NOT INCLUDED, DELIBERATELY: the qb_passing side-picker. It has a
    best-of-grid placebo p of 0.010, which is the strongest statistic here,
    but only TWO scorable seasons (+0.0301 and +0.1519, so the pooled figure
    is mostly one season), its ROI decays threefold as the cell floor rises
    from 200 to 400 bets, and its design omits the LINE even though the
    target is P(Y > L). Adding the line is a one-word change that should
    only help, and it should be retested before the rule is wired anywhere.

WHY THE CONSTANTS MUST BE FROZEN RATHER THAN COMPUTED PER WEEK

    The receiving rule's trigger is a z-score standardized over the FULL
    2023-2026 sample. If this module recomputed that z from whatever props
    happen to be on this week's board, `z >= 1.0` would mean a different
    thing every week and it would no longer be the rule that was measured.
    A quiet week with compressed air-yards variation would fire far more
    often than a volatile one.

    So the mean, the standard deviation and the veto cutoff are constants
    here, and they are currently None. This module REFUSES to evaluate the
    receiving rule until they are filled in, and says so. Run
    rules_calibrate.py once; it prints the block to paste.

    That is the same pattern as split_qb_rushing's min_match_rate defaulting
    to None: a threshold nobody has measured is not a threshold, and guessing
    one here would silently redefine a validated rule.

WHERE THIS FILE LIVES

    Repo ROOT, not models/. It parallels mc_pricing.py, which is also in the
    root: both are decision layers that consume market modules rather than
    being one. models/ holds the six market engines and data_utils, and
    nothing in it should need to know that rules exist.

NOTHING HERE PLACES A BET OR SIZES ONE

    The measured edges are 5 to 9 percent before any judgement is applied,
    and every figure is in-sample with respect to the SEARCH that found the
    rules even where it is out-of-sample within each test. Forward results
    are the only thing that settles them.
"""
import numpy as np
import pandas as pd

from models import data_utils

# ---------------------------------------------------------------------------
# RULE 1: RUSHING LOW-LINE UNDER. No calibration needed, the line is absolute.
# ---------------------------------------------------------------------------
RUSHING_MAX_LINE = 46.5          # FanDuel's own rushing median, the a priori cut

# ---------------------------------------------------------------------------
# RULE 2: RECEIVING AIR-YARDS UNDER. Needs frozen standardization.
# ---------------------------------------------------------------------------
AIR_YARDS_RECENT_GAMES = 3       # matches input_weights.py's recent window
AIR_YARDS_BASE_GAMES = 8         # matches its base window
AIR_YARDS_BASE_SHIFT = 4         # base excludes the recent window entirely
AIR_YARDS_MIN_RECENT = 3         # no mean over fewer games than the window
AIR_YARDS_MIN_BASE = 4           # matches input_weights' min_periods
AIR_YARDS_Z_TRIGGER = 1.0        # the validated threshold, do not tune here

# MEASURED CONSTANTS. Fill from rules_calibrate.py. None means "refuse".
#
#   shock_mean / shock_sd  standardize (recent3 - base8) into the z that
#                          air_yards_retest.py validated. Must come from the
#                          full history, never from one week.
#   dev_veto_at            the projection-minus-line value above which the
#                          model vetoes. This is the top dev quintile inside
#                          the rule cell, where under-side ROI measured
#                          -0.117 against roughly +0.10 elsewhere.
CALIBRATION = {
    "receiving": {
        "shock_mean": 0.510177,
        "shock_sd": 22.382350,
        "dev_veto_at": 7.528338,
    },
}
# Measured 2026-09-25 by rules_calibrate.py: 8,184 receiving rows, seasons
# 2023-2026, FanDuel. Its gate reproduced air_yards_retest.py's lift at
# z >= 1.0 as +0.0840 against a published +0.0864, so these constants
# describe the rule that earned the interval.
#
#   cell at z >= 1.0        1,053 rows (324/season)   lift +0.0840
#   with the veto applied     842 rows (259/season)   lift +0.1270
#   vetoed                    211 rows               ROI  -0.1171
#
# The vetoed figure matches conjunction_followup.py's top-quintile ROI of
# -0.1171 exactly, which confirms the boundary lands where intended. It is
# the SAME rows, so it is a consistency check and not independent evidence.
#
# ONE CAVEAT ON THE +0.1270. The veto's SHAPE was pre-specified: "skip the
# top dev quintile" came from conjunction_followup.py rather than being tuned
# here. But that quintile was identified as bad using this same data, and the
# boundary VALUE of +7.53 is the 80th percentile of these very rows. So the
# lift with the veto is more in-sample than the +0.0840 without it. Treat
# +0.0840 as the defensible figure and +0.1270 as the optimistic one.
#
# FROZEN. Recomputing these on a later sample changes what z >= 1.0 means and
# silently redefines the rule. If you ever do recompute, the result is a NEW
# rule needing its own out-of-sample test, not a refresh of this one.


def is_calibrated(market="receiving"):
    c = CALIBRATION.get(market) or {}
    return all(c.get(k) is not None
               for k in ("shock_mean", "shock_sd", "dev_veto_at"))


def _air_yards_history(seasons):
    """Per player-week air-yards shock, from lagged windows only.

    Windows do NOT overlap: base is shifted past the recent window, so the
    two regressors are not sharing observations. Minimums mirror
    input_weights.py, and a player below them gets NaN rather than a mean
    over one game. That last point is not pedantry: anytime_td's bridge
    omitted exactly this check and a single game became a 100 percent rate.
    """
    ps = data_utils.load_player_stats(list(seasons))
    ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    col = "receiving_air_yards"
    if col not in ps.columns:
        raise RuntimeError(
            f"{col} is absent from load_player_stats, so the receiving rule "
            f"cannot be evaluated. Columns: {sorted(ps.columns)[:30]}")
    ps = ps.dropna(subset=["player_display_name"]).copy()
    ps[col] = ps[col].fillna(0.0)
    ps["_key"] = data_utils.norm_join_name(ps["player_display_name"])

    idc = next((c for c in ("player_id", "gsis_id") if c in ps.columns), None)
    if idc:
        # Same collision guard as data_utils.actual_stat: a normalized name
        # mapping to two players cannot be attributed to either.
        nid = ps.groupby(["season", "week", "_key"])[idc].transform("nunique")
        ps = ps[nid == 1]
    key = idc if idc else "_key"

    ps = ps.sort_values([key, "season", "week"]).reset_index(drop=True)
    g = ps.groupby(key)[col]
    ps["_recent"] = g.transform(
        lambda s: s.shift(1).rolling(AIR_YARDS_RECENT_GAMES,
                                     min_periods=AIR_YARDS_MIN_RECENT).mean())
    ps["_base"] = g.transform(
        lambda s: s.shift(AIR_YARDS_BASE_SHIFT).rolling(
            AIR_YARDS_BASE_GAMES, min_periods=AIR_YARDS_MIN_BASE).mean())
    ps["shock_raw"] = ps["_recent"] - ps["_base"]
    ps["week"] = pd.to_numeric(ps["week"], errors="coerce").astype("Int64")
    return ps[["season", "week", "_key", "_recent", "_base", "shock_raw"]]


def evaluate(props, seasons=(2022, 2023, 2024, 2025, 2026)):
    """Which rules fire on these props, and on which side.

    props: a DataFrame with at least `market`, `player`, `line`, `season`,
    `week`. A `projection` column is required for the receiving rule's veto
    and its absence is reported rather than silently ignored.

    Returns (fired, report). `fired` has one row per firing with columns
    rule, market, player, line, side and a rule-specific `detail`. `report`
    is a dict recording how many props were considered per rule and why the
    rest did not fire, because a rule that fires zero times and a rule that
    could not be evaluated look identical in the output otherwise.
    """
    need = ["market", "player", "line", "season", "week"]
    missing = [c for c in need if c not in props.columns]
    if missing:
        raise KeyError(f"evaluate needs {need}; missing {missing}")

    p = props.copy()
    # Idempotent, so safe to apply even if the caller already split. Without
    # it a quarterback's rushing prop would be tested against a rule measured
    # on running backs and receivers.
    try:
        p, _ = data_utils.split_qb_rushing(p)
    except Exception as e:
        # Not fatal: if the caller already labelled qb_rushing, nothing is
        # lost. Recorded so a silent mislabel cannot hide.
        split_note = f"split_qb_rushing skipped: {type(e).__name__}: {e}"
    else:
        split_note = "split_qb_rushing applied"

    p["_key"] = data_utils.norm_join_name(p["player"])
    p["week"] = pd.to_numeric(p["week"], errors="coerce").astype("Int64")
    p["line"] = pd.to_numeric(p["line"], errors="coerce")

    out = []
    report = {"props_in": int(len(props)), "split": split_note, "rules": {}}

    # ---- RULE 1: rushing low-line under ----
    r = p[(p["market"] == "rushing") & p["line"].notna()]
    hit = r[r["line"] <= RUSHING_MAX_LINE]
    report["rules"]["rushing_low_line_under"] = {
        "eligible": int(len(r)),
        "fired": int(len(hit)),
        "reason": (f"line <= {RUSHING_MAX_LINE}"
                   if len(r) else "no rushing props with a line"),
    }
    for _, row in hit.iterrows():
        out.append({"rule": "rushing_low_line_under", "market": "rushing",
                    "player": row["player"], "line": row["line"],
                    "side": "UNDER",
                    "detail": f"line {row['line']:g} <= {RUSHING_MAX_LINE}"})

    # ---- RULE 2: receiving air-yards under ----
    rec = p[(p["market"] == "receiving") & p["line"].notna()]
    rr = {"eligible": int(len(rec)), "fired": 0}
    if not len(rec):
        rr["reason"] = "no receiving props with a line"
    elif not is_calibrated("receiving"):
        rr["reason"] = ("NOT CALIBRATED. shock_mean, shock_sd and dev_veto_at "
                        "are None, so z cannot be computed on the same scale "
                        "the rule was validated on. Run rules_calibrate.py "
                        "and paste the block it prints. Refusing rather than "
                        "standardizing on this week's props, which would be "
                        "a different rule.")
    elif "projection" not in rec.columns:
        rr["reason"] = ("no `projection` column, so the model veto cannot be "
                        "applied. The rule without its veto is NOT the "
                        "validated rule: the vetoed quintile measured -0.117 "
                        "against roughly +0.10 elsewhere. Refusing.")
    else:
        cal = CALIBRATION["receiving"]
        try:
            hist = _air_yards_history(seasons)
        except Exception as e:
            rr["reason"] = f"air-yards history unavailable: {type(e).__name__}: {e}"
        else:
            m = rec.merge(hist, on=["season", "week", "_key"], how="left")
            m["z"] = (m["shock_raw"] - cal["shock_mean"]) / cal["shock_sd"]
            m["dev"] = pd.to_numeric(m["projection"], errors="coerce") - m["line"]

            no_hist = int(m["shock_raw"].isna().sum())
            trig = m["z"] >= AIR_YARDS_Z_TRIGGER
            vetoed = trig & (m["dev"] > cal["dev_veto_at"])
            fires = trig & ~vetoed & m["dev"].notna()
            rr.update({
                "fired": int(fires.sum()),
                "no_air_yards_history": no_hist,
                "triggered_before_veto": int(trig.sum()),
                "vetoed_by_model": int(vetoed.sum()),
                "reason": (f"z >= {AIR_YARDS_Z_TRIGGER} and dev <= "
                           f"{cal['dev_veto_at']}"),
            })
            for _, row in m[fires].iterrows():
                out.append({
                    "rule": "receiving_air_yards_under",
                    "market": "receiving", "player": row["player"],
                    "line": row["line"], "side": "UNDER",
                    "detail": (f"z {row['z']:+.2f}, dev {row['dev']:+.1f} "
                               f"(veto above {cal['dev_veto_at']:+.1f})")})
    report["rules"]["receiving_air_yards_under"] = rr

    fired = (pd.DataFrame(out) if out else
             pd.DataFrame(columns=["rule", "market", "player", "line",
                                   "side", "detail"]))
    report["fired_total"] = int(len(fired))
    return fired, report


def describe():
    """Human-readable provenance, for a UI caption or a console check."""
    lines = [
        "RUSHING LOW-LINE UNDER",
        f"  market == 'rushing' and line <= {RUSHING_MAX_LINE} -> UNDER",
        "  ROI +0.0589, 95% [+0.0115, +0.1075], ~438 bets/season",
        "  positive in all four seasons; market-wide across six books",
        "  the model is NOT used (rushing beta -0.041)",
        "  source: rushing_lowline.py, cross_book.py",
        "",
        "RECEIVING AIR-YARDS UNDER",
        f"  market == 'receiving', z >= {AIR_YARDS_Z_TRIGGER} -> UNDER,",
        "  except where the projection sits above the veto cutoff",
        "  lift over the market's under baseline +0.086, ~250 bets/season",
        "  best-of-grid placebo p = 0.072 (BORDERLINE)",
        "  season ROI decays +0.165, +0.102, +0.070",
        "  the model is a VETO, not a confirmation",
        "  source: air_yards_retest.py, conjunction_followup.py",
        f"  calibrated: {is_calibrated('receiving')}",
        "",
        "NOT WIRED: qb_passing side-picker. Placebo p = 0.010 but only two",
        "  scorable seasons, ROI decays threefold as the cell floor rises,",
        "  and its design omits the line. Retest before using.",
    ]
    return "\n".join(lines)
