"""app_cache.py - cached wrappers for the app's database reads.

WHY THIS EXISTS

    Streamlit re-runs the entire script top to bottom on every widget
    interaction. Tab switching is client-side and never re-runs, which is
    exactly why tabs feel instant while every ACTION feels slow. Measured on
    a warm local machine, all the compute in a board render is about 0.2
    seconds:

        six project_week calls, warm            0.09 s
        graded.apply(_row_edge) on 317 rows     0.07 s
        the five dict lookups at lines 301-314  0.003 s

    So compute is not the problem. The cost is NETWORK, and app.py had no
    cache decorators at all, so every re-run repeated every Supabase read.

    The worst offenders are two per-market LOOPS:

        app.py:803   for mkt_key, mkt_label in MARKETS.items():
                         mv = db.get_line_movement(...)      <- six calls
        app.py:1222  for mkt_name in x_markets:
                         mv = db.get_line_movement(...)      <- up to six more

    Six round trips per re-run on the Line Movement tab, and the game-filter
    widget triggers a re-run, which is precisely the "reducing the shown
    games resets and takes a while" symptom. Logging a bet ends in st.rerun()
    at app.py:650, so it pays the same cost again.

HOW INVALIDATION WORKS, AND WHY A PLAIN TTL IS NOT ENOUGH

    A pure TTL cache would serve stale data after a write: import a line
    batch, and the board would keep showing the old pool until the TTL
    expired. So the cache key includes a VERSION token held in session
    state. Reads are cached; any write calls bump(), the token changes, and
    the next read misses and refetches.

    That gives correct-after-write behaviour AND a fast re-run, which a TTL
    alone cannot do.

WHERE TO CALL bump()

    After anything that CHANGES what these two functions return:
      - db.import_lines (app.py around line 1169), which changes the pool
      - any hand-save of board rows into the capture table

    NOT needed after grading picks or saving a bet: those touch the bet log,
    not the lines table, so the line reads stay valid. If the bet log read is
    also slow, wrap it here the same way with its own bump.

THE user ARGUMENT IS DELIBERATELY UNDERSCORED

    Streamlit skips arguments whose names start with an underscore when
    computing a cache key. A Supabase user object is not reliably hashable,
    and hashing it would also mean a token refresh silently invalidated every
    cached read. The user still has to be PASSED, because db enforces access
    with it; it just does not participate in the key. Since the key includes
    season, week and market only, do not use this module in a context where
    two different users share one process.
"""
import streamlit as st

import db

_VER_KEY = "_db_cache_version"

# Short enough that an external capture (the GitHub Actions cron) shows up
# without a manual refresh, long enough that a burst of clicks is free.
TTL = 120


def version():
    """Current cache-busting token."""
    return st.session_state.get(_VER_KEY, 0)


def bump():
    """Invalidate every cached read. Call after any write to the lines table."""
    st.session_state[_VER_KEY] = version() + 1


@st.cache_data(ttl=TTL, show_spinner=False)
def _lines(season, week, market, _ver, _user):
    return db.get_lines(season, week, market, _user)


@st.cache_data(ttl=TTL, show_spinner=False)
def _movement(season, week, market, _ver, _user):
    return db.get_line_movement(season, week, market, _user)


def get_lines(season, week, market, user):
    """Drop-in for db.get_lines. Same arguments, same return."""
    return _lines(season, week, market, version(), user)


def get_line_movement(season, week, market, user):
    """Drop-in for db.get_line_movement. Same arguments, same return.

    This is the one that matters most: it is called once per market inside
    two separate loops, so caching it turns up to twelve round trips per
    widget change into zero.
    """
    return _movement(season, week, market, version(), user)


@st.cache_data(ttl=TTL, show_spinner=False)
def _projections(season, week, market):
    """market key -> {normalised player name: projection}.

    Takes the market KEY rather than a module because st.cache_data hashes
    its arguments and a module object is not usefully hashable.

    rushing and qb_rushing resolve to DIFFERENT modules on purpose. The Line
    Movement rows are already split by split_qb_rushing, so a running back
    must be looked up in rushing's board and a quarterback in qb_rushing's.
    Crossing them would put RB projections on QBs.

    Returns {} rather than raising when a market has no board yet, so a tab
    that shows movement fine does not go down because a model could not fit.
    """
    import db
    from models import (receiving, receptions, rushing, qb_passing,
                        anytime_td, qb_rushing)
    mods = {"receiving": receiving, "receptions": receptions,
            "rushing": rushing, "qb_passing": qb_passing,
            "anytime_td": anytime_td, "qb_rushing": qb_rushing}
    mod = mods.get(market)
    if mod is None or not hasattr(mod, "project_week"):
        return {}
    try:
        board = mod.project_week(season, week)
    except Exception:
        return {}
    if board is None or len(board) == 0:
        return {}
    if "player_display_name" not in board.columns \
            or "projection" not in board.columns:
        return {}
    out = {}
    for name, proj in zip(board["player_display_name"], board["projection"]):
        if name is None:
            continue
        out[db._norm_name(name)] = proj
    return out


def get_projections(season, week, market):
    """Cached {normalised name: projection} for one market and week.

    Not keyed on the user: a projection is a property of the model and the
    week, not of who is looking at it, unlike the lines reads above.
    """
    return _projections(season, week, market)


@st.cache_data(ttl=TTL, show_spinner=False)
def _log_summary(market, seasons):
    """market, seasons tuple -> {normalised name: summary dict}.

    seasons MUST be a tuple. st.cache_data hashes its arguments and a list
    is unhashable, which is the same trap that made rules.py's
    _air_yards_history need tuple(seasons) after it was given an lru_cache.

    Returns {} rather than raising if the stats load or the market is
    unknown, so a tab that renders movement fine does not go down because a
    game log could not be built.
    """
    import db
    import player_log
    from models import data_utils
    try:
        stats = data_utils.load_player_stats(list(seasons))
        stats = stats.to_pandas() if hasattr(stats, "to_pandas") else stats
    except Exception:
        return {}
    try:
        # db._norm_name, NOT data_utils.norm_join_name. The join in app.py
        # keys on mv["player"].map(db._norm_name), so these keys must come
        # from the same normaliser or nothing matches. summarize_all wants a
        # VECTORISED function, hence the map wrapper.
        summ = player_log.summarize_all(
            stats, market, lambda s: s.map(db._norm_name))
    except Exception:
        return {}
    if summ is None or len(summ) == 0:
        return {}
    # season_min / season_max must be here. player_log._span_of reads them
    # to build the "2025-2026" span in the log caption, and they were added
    # to summarize_all after this list was written, so the caption rendered
    # with a leading comma and no seasons.
    keep = [c for c in ("n", "mean", "median", "gap", "lo", "hi",
                        "seasons", "n_seasons", "season_min", "season_max")
            if c in summ.columns]
    return {k: {c: v[c] for c in keep} for k, v in summ[keep].iterrows()}


def get_log_summary(market, seasons):
    """Cached per-player game-log summary for one market.

    `seasons` is coerced to a tuple here so callers can pass a list without
    tripping the cache's hashing.
    """
    return _log_summary(market, tuple(seasons))


MODEL_TRAIN_MAX_FALLBACK = 2024


@st.cache_data(ttl=TTL, show_spinner=False)
def _player_log(market, join_name, seasons, _user=None):
    """One player's displayable game log. Empty frame on any failure.

    seasons MUST be a tuple: st.cache_data hashes its arguments.
    """
    import pandas as pd
    import db
    import player_log
    from models import data_utils

    mods = {}
    try:
        from models import (receiving, receptions, rushing, qb_passing,
                            anytime_td, qb_rushing)
        mods = {"receiving": receiving, "receptions": receptions,
                "rushing": rushing, "qb_passing": qb_passing,
                "anytime_td": anytime_td, "qb_rushing": qb_rushing}
    except Exception:
        pass

    try:
        stats = data_utils.load_player_stats(list(seasons))
        stats = stats.to_pandas() if hasattr(stats, "to_pandas") else stats
    except Exception:
        return pd.DataFrame()
    if stats is None or len(stats) == 0:
        return pd.DataFrame()

    def _norm(s):
        return s.map(db._norm_name)

    try:
        sched = data_utils.load_schedules(list(seasons))
        sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    except Exception:
        sched = None

    # NAME BRIDGE. join_name came from the book's spelling; player_history
    # wants nflverse's exact display name. Whichever display name normalises
    # to the same key is the one to ask for.
    disp = None
    if "player_display_name" in stats.columns:
        cand = stats[stats["player_display_name"].notna()]
        keys = _norm(cand["player_display_name"])
        hit = cand.loc[keys == join_name, "player_display_name"]
        if len(hit):
            disp = str(hit.iloc[0])

    # CLOSING LINES from the local cache. FanDuel only, to match the Move
    # column above, and the LAST capture of each week.
    line_map = {}
    try:
        C = pd.read_parquet("lines_cache.parquet")
        C = C[(C["market"] == market) & (C["book"] == db.DEFAULT_BOOK)]
        C = C[C["season"].isin(list(seasons))]
        if len(C):
            C = C.assign(_k=_norm(C["player"]))
            C = C[C["_k"] == join_name].sort_values("captured_at")
            for s, w, ln in zip(C["season"], C["week"], C["line"]):
                if pd.notna(w) and pd.notna(ln):
                    line_map[(int(s), int(w))] = float(ln)
    except Exception:
        line_map = {}

    # RECENT WEEKS from the LIVE table. The parquet is historical_lines and
    # ends at week 2 of 2026; `lines` carries week 3 onward. Merged second,
    # so the live value wins if the two ever overlap, which is the right
    # precedence: `lines` is what the capture observed.
    #
    # db.get_player_closing_lines applies the 20-to-90-minute window. That
    # is not a precaution: `lines` holds every capture including in-game
    # ones, and the latest capture per prop-week would BE the contaminated
    # row.
    try:
        live = db.get_player_closing_lines(join_name, market, list(seasons),
                                           _user)
        if live:
            line_map.update(live)
    except Exception:
        pass

    # PROJECTIONS, only for seasons after the training cutoff. display_log
    # blanks the rest, but not asking for them saves a model fit per season.
    mod = mods.get(market)
    cut = getattr(mod, "DEFAULT_TRAIN_MAX_SEASON",
                  MODEL_TRAIN_MAX_FALLBACK) if mod else \
        MODEL_TRAIN_MAX_FALLBACK
    proj_map = {}
    if mod is not None and disp and hasattr(mod, "player_history"):
        for s in seasons:
            if int(s) <= int(cut):
                continue
            try:
                h = mod.player_history(int(s), disp)
            except Exception:
                continue
            if h is None or len(h) == 0 or "projection" not in h.columns:
                continue
            for w, p in zip(h["week"], h["projection"]):
                if pd.notna(w) and pd.notna(p):
                    proj_map[(int(s), int(w))] = float(p)

    try:
        return player_log.display_log(
            stats, market, join_name, _norm, sched=sched,
            line_map=line_map or None, proj_map=proj_map or None,
            train_max_season=cut, seasons=list(seasons))
    except Exception:
        return pd.DataFrame()


def get_player_log(market, join_name, seasons, user=None):
    """Cached game log for one player in one market. Empty frame if absent.

    `user` is needed for the live closing-line read and is passed as _user
    so st.cache_data does not hash it, matching _lines and _movement.
    """
    return _player_log(market, join_name, tuple(seasons), _user=user)


def clear_all():
    """Hard reset, for a debug button. Prefer bump() in normal use."""
    _lines.clear()
    _movement.clear()
    bump()
