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


def clear_all():
    """Hard reset, for a debug button. Prefer bump() in normal use."""
    _lines.clear()
    _movement.clear()
    bump()
