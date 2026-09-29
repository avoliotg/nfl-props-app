"""player_log.py - game logs and their summary statistics.

No streamlit, no database, no network. Pure frame work, so it can be tested
against planted answers.

WHY THE LOG COMES FIRST AND THE AVERAGE SECOND

    A game-log median is itself a forecast, and a weak one. The rushing
    projection is fitted by regression on rolling features and contributes
    a beta of exactly 0.000 against the closing line; a raw median of game
    logs is a cruder version of the same thing. So "his median is 38, the
    line is 43.5, take the under" is the inference the market already prices
    and that this project has spent three sessions disproving.

    The log is useful for the opposite reason: it is the raw material your
    own judgment works on. A bare average hides that he left one game in the
    second quarter, that two were blowouts where he sat the fourth, and that
    one came with the starter out. You know which games to discount. The
    model cannot. So the rows are the point and the summary is context.

WHY BOTH MEAN AND MEDIAN, AND WHAT THE GAP MEANS

    The MEDIAN is what the bet settles on. A line is placed so each side is
    close to even money, and "the over wins half the time" is a statement
    about the median.

    The MEAN is the distribution's centre of mass, and for these outcomes it
    sits ABOVE the median because they have a hard floor at zero and a long
    right tail. Measured market-wide at zero deviation, that offset is
    +4.0067 yards for rushing and +3.4623 for receiving.

    So the GAP is per-player skew. A large gap is a volatile player with
    occasional big games; a small gap is a steady one. For a given line the
    volatile player's under is more attractive than his mean suggests, which
    is the mechanism behind the one rule in this project that consults no
    model at all.

    The median is therefore the number to compare against the line, and the
    mean belongs beside it as a volatility read rather than as the
    comparison.

SAMPLE SIZE IS PART OF THE NUMBER

    Every summary carries its n. Three games in week 4 is not a median, and
    a figure quoted without its sample size will be read as a fact.
"""
import numpy as np
import pandas as pd

# The outcome column each market settles on, in nflverse player stats.
MARKET_STAT = {
    "receiving": "receiving_yards",
    "receptions": "receptions",
    "rushing": "rushing_yards",
    "qb_passing": "passing_yards",
    "qb_rushing": "rushing_yards",
}

# Volume and context columns worth showing beside the outcome, per market.
# These are what let a reader discount a game: six carries in a blowout is
# a different row from twenty in a close game.
MARKET_CONTEXT = {
    "receiving": ["targets", "receptions", "receiving_air_yards"],
    "receptions": ["targets", "receiving_yards"],
    "rushing": ["carries", "rushing_tds"],
    "qb_passing": ["attempts", "completions", "passing_tds"],
    "qb_rushing": ["carries", "rushing_tds"],
}

BASE_COLS = ["season", "week", "team", "opponent_team"]


def stat_col(market):
    return MARKET_STAT.get(market)


def build_log(stats, market, join_name, norm_fn, seasons=None):
    """Rows for one player in one market, most recent game first.

    `stats` is an nflverse player-stats frame. `join_name` is the player's
    normalised name and `norm_fn` is the same normaliser used to produce it,
    so the match here cannot disagree with the match used elsewhere.

    Returns an empty frame rather than raising when the player, the market
    or the columns are absent, because a missing log is a display gap and
    must not take a tab down with it.
    """
    col = stat_col(market)
    if col is None or stats is None or len(stats) == 0:
        return pd.DataFrame()
    if col not in stats.columns or "player_display_name" not in stats.columns:
        return pd.DataFrame()

    d = stats[stats["player_display_name"].notna()].copy()
    d["_key"] = norm_fn(d["player_display_name"])
    d = d[d["_key"] == join_name]
    if seasons:
        d = d[d["season"].isin(list(seasons))]
    if d.empty:
        return pd.DataFrame()

    keep = [c for c in BASE_COLS if c in d.columns]
    keep += [c for c in MARKET_CONTEXT.get(market, []) if c in d.columns]
    if col not in keep:
        keep.append(col)

    out = d[keep].copy()
    # A player with no stat line for a week did not play it. Dropping those
    # rows matters: a zero from a DNP is not a zero from a bad game, and
    # averaging them together understates every summary below.
    out = out[out[col].notna()]
    return out.sort_values(["season", "week"], ascending=[False, False])


def summarize(log, market, line=None):
    """Mean, median, their gap, and the hit rate against a line.

    Returns a dict. n is always present and always meaningful; every other
    figure is None when n is 0.

    The gap is mean minus median. Positive is the expected direction for a
    right-skewed outcome priced at the median, so a NEGATIVE gap is worth
    noticing: it means this player's big games are missing from the sample,
    usually because it is short.
    """
    col = stat_col(market)
    empty = {"n": 0, "mean": None, "median": None, "gap": None,
             "lo": None, "hi": None, "seasons": [],
             "over_line": None, "hit_rate": None, "line": line}
    if log is None or len(log) == 0 or col is None or col not in log.columns:
        return empty

    v = pd.to_numeric(log[col], errors="coerce").dropna()
    if v.empty:
        return empty

    mean = float(v.mean())
    med = float(v.median())
    res = {
        "n": int(len(v)),
        "mean": mean,
        "median": med,
        "gap": mean - med,
        "lo": float(v.min()),
        "hi": float(v.max()),
        "seasons": sorted(log["season"].unique().tolist())
        if "season" in log.columns else [],
        "over_line": None,
        "hit_rate": None,
        "line": line,
    }

    if line is not None and np.isfinite(line):
        # Pushes are excluded from the rate and not counted as either side,
        # matching how every ROI figure in this project is computed.
        nonpush = v[v != line]
        if len(nonpush):
            over = int((nonpush > line).sum())
            res["over_line"] = over
            res["hit_rate"] = over / len(nonpush)
            res["n_graded"] = int(len(nonpush))
    return res


def summary_line(s, market, unit=""):
    """One-line rendering, with the sample size attached to every figure.

    The sample size is not decoration. Three games in week 4 is not a
    median, and a figure quoted without its n will be read as a fact.
    """
    if s["n"] == 0:
        return "no games in the selected seasons"
    seasons = s.get("seasons") or []
    span = (str(seasons[0]) if len(seasons) == 1
            else "%s-%s" % (min(seasons), max(seasons)))
    bits = ["%s, n=%d" % (span, s["n"]),
            "median %.1f%s" % (s["median"], unit),
            "mean %.1f%s" % (s["mean"], unit),
            "gap %+.1f" % s["gap"],
            "range %.0f to %.0f" % (s["lo"], s["hi"])]
    if s.get("hit_rate") is not None:
        bits.append("cleared %.1f in %d of %d (%.0f%%)"
                    % (s["line"], s["over_line"], s["n_graded"],
                       100 * s["hit_rate"]))
    return "  ·  ".join(bits)


def compare_frame(logs, market):
    """Summaries for several players side by side, one column each.

    Two backs in the same game is the case this exists for: the comparison
    is the question, and two separate blocks make it harder than one frame.
    """
    rows = ["seasons", "games", "median", "mean", "gap", "low", "high"]
    data = {}
    for name, log in logs.items():
        s = summarize(log, market)
        if s["n"] == 0:
            data[name] = ["-"] * len(rows)
            continue
        seasons = s.get("seasons") or []
        span = (str(seasons[0]) if len(seasons) == 1
                else "%s-%s" % (min(seasons), max(seasons)))
        data[name] = [span, s["n"], round(s["median"], 1),
                      round(s["mean"], 1), round(s["gap"], 1),
                      round(s["lo"], 1), round(s["hi"], 1)]
    return pd.DataFrame(data, index=rows)
