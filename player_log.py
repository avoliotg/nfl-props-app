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
#
# anytime_td is ABSENT on purpose: it has no single column. The bet settles
# on whether the player scored a rushing OR receiving touchdown, so the
# outcome is derived from two columns by td_scored() below. Passing
# touchdowns do not count: the thrower is not the scorer.
MARKET_STAT = {
    "receiving": "receiving_yards",
    "receptions": "receptions",
    "rushing": "rushing_yards",
    "qb_passing": "passing_yards",
    "qb_rushing": "rushing_yards",
}

TD_MARKET = "anytime_td"
TD_COLS = ("rushing_tds", "receiving_tds")


def td_scored(rows):
    """Did the player score an anytime touchdown, per row. 1, 0 or NaN.

    Rushing plus receiving touchdowns. PASSING touchdowns are excluded
    because the passer is not the scorer, which is what the market settles
    on.

    NaN where neither column is present for that row, so a week the player
    did not appear in is not recorded as "did not score".
    """
    present = [c for c in TD_COLS if c in rows.columns]
    if not present:
        return pd.Series([np.nan] * len(rows), index=rows.index)
    tot = None
    any_known = None
    for c in present:
        v = pd.to_numeric(rows[c], errors="coerce")
        tot = v.fillna(0) if tot is None else tot + v.fillna(0)
        known = v.notna()
        any_known = known if any_known is None else (any_known | known)
    out = (tot > 0).astype(float)
    return out.where(any_known, np.nan)

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


def _span_of(s):
    """A season span string from either summary shape, or "" if neither."""
    seasons = s.get("seasons")
    if isinstance(seasons, (list, tuple)) and len(seasons):
        lo, hi = min(seasons), max(seasons)
    else:
        lo, hi = s.get("season_min"), s.get("season_max")
        if lo is None or hi is None:
            return ""
    return str(int(lo)) if int(lo) == int(hi) else "%d-%d" % (int(lo),
                                                              int(hi))


def summary_line(s, market, unit=""):
    """One-line rendering, with the sample size attached to every figure.

    The sample size is not decoration. Three games in week 4 is not a
    median, and a figure quoted without its n will be read as a fact.

    anytime_td gets a DIFFERENT sentence. Its outcome is 0 or 1, so the
    median is 0 or 1 and says nothing, the mean IS the scoring rate, and
    the mean-minus-median gap is not volatility but an artifact of which
    side of a half the rate sits on. Printing the yardage sentence there
    produced "median 0.0, mean 0.0, gap +0.0", which is true and useless.
    """
    if s["n"] == 0:
        return "no games in the selected seasons"
    # ACCEPTS EITHER SHAPE. summarize() supplies `seasons` as a list;
    # summarize_all() supplies season_min and season_max. Reading only one
    # of them raised a TypeError on the other.
    span = _span_of(s)
    if market == TD_MARKET:
        rate = s["mean"] or 0.0
        return ("%s: scored in %d of %d game(s), %.0f%%"
                % (span, int(round(rate * s["n"])), s["n"], 100.0 * rate))
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

def summarize_all(stats, market, norm_fn, seasons=None):
    """One row per player: n, mean, median and their gap, for one market.

    BULK ON PURPOSE. build_log filters the whole stats frame per player, and
    the Line Movement tab renders five markets with tens of players each, so
    calling it per player per market would rebuild the same frame hundreds
    of times on every widget interaction. This does one groupby.

    Returns a frame indexed by the NORMALISED name, so the caller joins
    without writing another name-matching implementation.

    Rows where the outcome is null are dropped before aggregating. That
    matters: a player who did not play has no stat line, and a zero from a
    DNP averaged together with real games understates every figure here.
    """
    is_td = (market == TD_MARKET)
    col = stat_col(market)
    if stats is None or len(stats) == 0:
        return pd.DataFrame()
    if "player_display_name" not in stats.columns:
        return pd.DataFrame()
    if not is_td and (col is None or col not in stats.columns):
        return pd.DataFrame()

    d = stats[stats["player_display_name"].notna()].copy()
    if seasons:
        d = d[d["season"].isin(list(seasons))]
    if is_td:
        # For this market the "median" is a SCORING RATE: the mean of a 0/1
        # outcome. The median of a 0/1 series is 0 or 1 and says nothing,
        # so a reader should look at the mean here and the median
        # everywhere else. summary_line names the market for that reason.
        d = d.assign(_td=td_scored(d))
        d = d[d["_td"].notna()]
        col = "_td"
    else:
        d = d[pd.to_numeric(d[col], errors="coerce").notna()]
    if d.empty:
        return pd.DataFrame()

    d["_key"] = norm_fn(d["player_display_name"])
    d["_v"] = pd.to_numeric(d[col], errors="coerce")

    g = d.groupby("_key")["_v"]
    out = pd.DataFrame({
        "n": g.size(),
        "mean": g.mean(),
        "median": g.median(),
        "lo": g.min(),
        "hi": g.max(),
    })
    # The GAP is mean minus median: per-player skew. Positive is the
    # expected direction for an outcome with a floor at zero and a long
    # right tail. A NEGATIVE gap usually means the sample is too short to
    # contain the player's big games rather than that he has none.
    out["gap"] = out["mean"] - out["median"]
    if "season" in d.columns:
        g2 = d.groupby("_key")["season"]
        # NAMED season_min / season_max, NOT "seasons". summarize() returns
        # `seasons` as a LIST and this returned it as a nunique COUNT, so
        # summary_line assumed a list and raised on an int. One field name
        # meaning two things is how a wrong number survives review.
        out["n_seasons"] = g2.nunique()
        out["season_min"] = g2.min()
        out["season_max"] = g2.max()
    return out

# Column labels per market, for the displayed log. The outcome column is
# named for what it is rather than "actual", because a log of receiving
# yards beside a log of receptions should not share a header.
STAT_LABEL = {
    "anytime_td": "Scored",
    "receiving": "Rec Yds",
    "receptions": "Rec",
    "rushing": "Rush Yds",
    "qb_passing": "Pass Yds",
    "qb_rushing": "Rush Yds",
}

# The USAGE stat that drives the market. It answers "did he get 2 targets or
# 12" when a yardage number looks low, which is the difference between a bad
# game and a game he barely played in.
VOLUME_COL = {
    # Touches, as the usage behind a scoring chance. carries alone would
    # miss a receiving back, so this uses carries and reports targets
    # separately via the second entry handled in display_log.
    "anytime_td": ("carries", "Car"),
    "receiving": ("targets", "Tgts"),
    "receptions": ("targets", "Tgts"),
    "rushing": ("carries", "Car"),
    "qb_passing": ("attempts", "Att"),
    "qb_rushing": ("carries", "Car"),
}


def american_to_prob(odds):
    """American odds -> implied probability as a PERCENT, or None.

    Mirrors models.anytime_td.american_to_prob deliberately rather than
    importing it, so this module keeps no dependency on the market modules
    and stays testable on its own. The two must agree: the Line Movement
    table's "Latest Prob%" comes from that function, and a log column
    produced a different way would silently disagree with the row above it.

    ONE-SIDED, so this INCLUDES the book's hold and overstates the true
    probability. anytime_td cannot be devigged, which devig.py refuses for
    that reason.
    """
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    if odds > 0:
        return round(100 / (odds + 100) * 100, 1)
    return round(-odds / (-odds + 100) * 100, 1)


def opponent_labels(stats_rows, sched):
    """Opponent with "@" prefix for away games, as a Series.

    nflverse player stats carry `opponent_team` but no home/away flag, so
    the side comes from the schedule: if the player's team is the game's
    home team it was a home game. Falls back to the bare opponent when the
    schedule is unavailable, rather than guessing a side.
    """
    opp = stats_rows.get("opponent_team")
    if opp is None:
        return pd.Series([""] * len(stats_rows), index=stats_rows.index)
    if sched is None or len(sched) == 0 or "game_id" not in stats_rows.columns:
        return opp.astype(str)
    cols = [c for c in ("game_id", "home_team") if c in sched.columns]
    if len(cols) < 2:
        return opp.astype(str)
    home = dict(zip(sched["game_id"], sched["home_team"]))
    team = stats_rows.get("team")
    if team is None:
        return opp.astype(str)
    out = []
    for gid, tm, op in zip(stats_rows["game_id"], team, opp):
        h = home.get(gid)
        if h is None or pd.isna(h):
            out.append(str(op))
        else:
            out.append(str(op) if str(tm) == str(h) else "@ " + str(op))
    return pd.Series(out, index=stats_rows.index)


def display_log(stats, market, join_name, norm_fn, sched=None,
                line_map=None, proj_map=None, train_max_season=None,
                seasons=None):
    """A player's game log, ready to render. Most recent game first.

    line_map and proj_map are {(season, week): value} dicts supplied by the
    caller. Passing them in keeps this module free of database and model
    imports, so it stays testable without either.

    THE PROJECTION IS WITHHELD ON TRAINING SEASONS. The market models train
    through train_max_season (2024 at the time of writing, hardcoded in
    receiving.load_model and exposed as DEFAULT_TRAIN_MAX_SEASON in rushing
    and qb_passing). A projection for a season the model was FITTED on is
    in-sample and would flatter it on exactly the weeks being inspected, so
    those cells are blanked. 2025 and 2026 are genuinely out-of-sample and
    are shown.

    Reading the cutoff from the model rather than hardcoding a year means
    the column corrects itself if training is ever extended.
    """
    is_td = (market == TD_MARKET)
    col = stat_col(market)
    if stats is None or len(stats) == 0:
        return pd.DataFrame()
    if "player_display_name" not in stats.columns:
        return pd.DataFrame()
    if not is_td and (col is None or col not in stats.columns):
        return pd.DataFrame()

    d = stats[stats["player_display_name"].notna()].copy()
    d["_key"] = norm_fn(d["player_display_name"])
    d = d[d["_key"] == join_name]
    if seasons:
        d = d[d["season"].isin(list(seasons))]
    if is_td:
        # DERIVED OUTCOME. NaN means he did not appear that week, which is
        # not the same as not scoring, so those rows are dropped like any
        # other missing stat line.
        d["_td"] = td_scored(d)
        d = d[d["_td"].notna()]
        col = "_td"
    else:
        # A week with no stat line is a week he did not play. Excluded, so
        # it never reads as a zero.
        d = d[pd.to_numeric(d[col], errors="coerce").notna()]
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values(["season", "week"], ascending=[False, False])

    stat_lab = STAT_LABEL.get(market, "Result")
    vol_col, vol_lab = VOLUME_COL.get(market, (None, None))

    out = pd.DataFrame({
        "Season": d["season"].astype(int),
        "Week": d["week"].astype(int),
        "Opp": opponent_labels(d, sched),
    })
    if vol_col and vol_col in d.columns:
        out[vol_lab] = pd.to_numeric(d[vol_col], errors="coerce")
    if is_td and "targets" in d.columns:
        # A receiving back scores through the air, so targets belong beside
        # carries for this market rather than instead of them.
        out["Tgts"] = pd.to_numeric(d["targets"], errors="coerce")
    if is_td:
        out[stat_lab] = ["yes" if v == 1 else "no"
                         for v in pd.to_numeric(d[col], errors="coerce")]
    else:
        out[stat_lab] = pd.to_numeric(d[col], errors="coerce")

    keys = list(zip(out["Season"], out["Week"]))
    if is_td:
        # anytime_td has NO LINE. Its price lives in over_odds alone, so
        # there is nothing to devig and a "Hit" column against a line would
        # be meaningless. Both the American price and the implied
        # percentage are shown: the odds are what the book displays, the
        # percentage is what compares to Proj.
        #
        # ONE-SIDED, so the percentage INCLUDES the hold and overstates the
        # true chance. A projection below it is therefore expected and is
        # not by itself a disagreement.
        if line_map:
            out["Price"] = [line_map.get(k) for k in keys]
            out["Closing%"] = [american_to_prob(line_map.get(k))
                               for k in keys]
        if proj_map:
            cut = train_max_season
            out["Proj%"] = [
                (proj_map.get(k) if (cut is None or k[0] > cut) else None)
                for k in keys]
        return out.reset_index(drop=True)
    if line_map:
        out["Line"] = [line_map.get(k) for k in keys]
        # Did the result clear the line that week. Blank where either side
        # is missing, and blank on a push rather than assigning it a side.
        res, ln = out[stat_lab].to_numpy(), out["Line"].to_numpy()
        hit = []
        for r, l in zip(res, ln):
            if l is None or pd.isna(l) or pd.isna(r):
                hit.append("")
            elif r > l:
                hit.append("over")
            elif r < l:
                hit.append("under")
            else:
                hit.append("push")
        out["Hit"] = hit
    if proj_map:
        cut = train_max_season
        out["Proj"] = [
            (proj_map.get(k) if (cut is None or k[0] > cut) else None)
            for k in keys]
    return out.reset_index(drop=True)
