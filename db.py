"""Database connection + helpers for OpalScales (Supabase).

Changes in this revision
------------------------
1. ONE authenticated client per session, not one per call. Supabase refresh
   tokens are single-use. The Line Movement tab calls get_authed_client once
   per market, so six set_session calls per rerun meant the later ones could
   be handed an already-rotated token, fail, and silently fall back to the
   anon client. Under an RLS policy scoped TO authenticated, the anon client
   reads return ZERO ROWS WITH NO ERROR, so the tab went blank for every week
   including ones that used to render. The client is now built once and reused.

2. Auth degradation is no longer silent. If the session dies we record it and
   the read helpers say so on screen instead of showing an empty table.

3. Queries are paginated. PostgREST caps an unbounded select at 1000 rows.
   Ordering captured_at ascending meant that past 1000 rows per market/week
   we kept the OLDEST snapshots and dropped the newest, so Line Movement would
   have frozen: latest line stops updating and every move reads flat. At ~150
   receiving players that is about seven captures.

4. Read failures raise into a visible warning rather than returning an empty
   DataFrame from a bare except.

5. compute_p_over added, so the save-to-log path can stop writing p_over=None.
"""
import streamlit as st
from supabase import create_client
import pandas as pd

# PostgREST returns at most this many rows per request by default.
_PAGE = 1000

# session_state keys
_CLIENT_KEY = "_opal_authed_client"
_CLIENT_TOKEN_KEY = "_opal_authed_client_token"
_DEGRADED_KEY = "_opal_auth_degraded"


# ---------- session_state access that survives bare (non-Streamlit) runs ----------

def _ss_get(key, default=None):
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default


def _ss_set(key, value):
    try:
        st.session_state[key] = value
    except Exception:
        pass


def _warn(msg):
    """Surface a problem instead of swallowing it. No-ops outside Streamlit."""
    try:
        st.warning(msg)
    except Exception:
        print(f"WARNING: {msg}")


# ---------- clients ----------

@st.cache_resource
def get_client():
    """Create a cached Supabase client from secrets (anon, no user session)."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def get_user_client(access_token, refresh_token):
    """Build a Supabase client carrying a specific user's session, so its
    queries run AS that authenticated user (making auth.uid() resolve at the DB).

    NOT cached by Streamlit: caching would leak one user's session across users.
    Reuse is handled per session in get_authed_client instead.

    Returns (client, fresh_access_token, fresh_refresh_token). If Supabase
    rotated the refresh token internally the caller MUST persist the fresh ones
    back to session_state or subsequent calls will fail.
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    client = create_client(url, key)
    try:
        client.auth.set_session(access_token, refresh_token)
    except Exception:
        # refresh token already used or invalid, session is dead
        return client, None, None
    try:
        session = client.auth.get_session()
    except Exception:
        session = None
    if session:
        return client, session.access_token, session.refresh_token
    return client, access_token, refresh_token


def get_authed_client(user):
    """Return an authenticated client for this session, building it at most once.

    A single client object refreshes its own session internally, so calling
    set_session repeatedly is not only unnecessary, it burns single-use refresh
    tokens and is what broke Line Movement. The built client is cached in
    session_state and reused for the life of the login.

    Falls back to the anon client if no valid tokens exist, and records that
    fallback in session_state so read helpers can tell the user why a table is
    empty rather than showing a blank grid.
    """
    if not (user and isinstance(user, dict)
            and user.get("access_token") and user.get("refresh_token")):
        _ss_set(_DEGRADED_KEY, "no session tokens (log out and back in)")
        return get_client()

    cached = _ss_get(_CLIENT_KEY)
    cached_token = _ss_get(_CLIENT_TOKEN_KEY)
    if cached is not None and cached_token == user.get("access_token"):
        _ss_set(_DEGRADED_KEY, None)
        return cached

    client, fresh_access, fresh_refresh = get_user_client(
        user["access_token"], user["refresh_token"])

    if fresh_access is None:
        # session dead. Degrade to anon, but say so.
        _ss_set(_CLIENT_KEY, None)
        _ss_set(_CLIENT_TOKEN_KEY, None)
        _ss_set(_DEGRADED_KEY, "session expired (log out and back in)")
        return get_client()

    if fresh_access != user["access_token"]:
        user["access_token"] = fresh_access
        user["refresh_token"] = fresh_refresh
        _ss_set("user", user)  # persist the rotation

    _ss_set(_CLIENT_KEY, client)
    _ss_set(_CLIENT_TOKEN_KEY, fresh_access)
    _ss_set(_DEGRADED_KEY, None)
    return client


def auth_degraded():
    """Reason string if the last client build fell back to anon, else None."""
    return _ss_get(_DEGRADED_KEY)


def clear_authed_client():
    """Drop the cached authenticated client. Call on logout."""
    _ss_set(_CLIENT_KEY, None)
    _ss_set(_CLIENT_TOKEN_KEY, None)
    _ss_set(_DEGRADED_KEY, None)


def test_connection():
    """Quick check: can we reach the database? Returns (ok, message)."""
    try:
        client = get_client()
        resp = client.table("lines").select("*").limit(1).execute()
        return True, f"Connected! ({len(resp.data)} rows)"
    except Exception as e:
        return False, f"Connection failed: {e} | URL used: {st.secrets['SUPABASE_URL']}"


# ---------- paginated reads ----------

def _fetch_lines_rows(client, sport, season, week, market=None, ascending=True):
    """Fetch ALL matching rows from 'lines', paging past the 1000-row cap.

    Raises on query failure so the caller can surface it. Returning [] on error
    is what let a dead session look like an empty week.
    """
    rows = []
    start = 0
    while True:
        q = (client.table("lines").select("*")
             .eq("sport", sport).eq("season", int(season)).eq("week", int(week)))
        if market is not None:
            q = q.eq("market", market)
        resp = (q.order("captured_at", desc=not ascending)
                 .range(start, start + _PAGE - 1).execute())
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < _PAGE:
            break
        start += _PAGE
        if start > 200000:  # runaway guard
            break
    return rows


# ---------- market + name normalisation ----------

# BUG PATTERN 2 HELPERS  (September 24 2026)
#
# The 'lines' table holds one row per BOOK per snapshot, and it ALSO holds
# hand-saved board rows written by the Save-to-Log path. Neither fact was true
# when this module was written: the original capture was FanDuel only, so
# counting rows was counting captures, and nothing else wrote to the table.
#
# Measured on 2026-09-24:
#   43,465 rows, 8 books
#   5,071 of them are hand-saved (book='fanduel' AND projection IS NOT NULL)
#
# There is no column marking a row as saved. The discriminator is that a
# CAPTURED row has no projection and no edge, because the capture has no model
# attached, while a SAVED row always carries both. That is an accident rather
# than a design, so if a future writer starts populating projection on capture
# these helpers break silently. Add an explicit source column if that ever
# happens.
#
# RULE: any function reading 'lines' must exclude saved rows and select or
# collapse on book before counting or differencing anything, and should carry
# n_books so the defect cannot recur invisibly.

DEFAULT_BOOK = "fanduel"


def _drop_saved_rows(df):
    """Keep only CAPTURED rows: no projection means it came from the API.

    Saved rows carry their own captured_at spread across whenever the board
    was saved, so they interleave with real captures and corrupt any first
    vs latest comparison.
    """
    if df.empty or "projection" not in df.columns:
        return df
    return df[df["projection"].isna()].copy()


def _select_book(df, book=DEFAULT_BOOK):
    """Restrict to one book, and report coverage.

    Returns (frame, n_books_seen, book_used).

    FanDuel is the default because it is the app's pricing anchor: the BLEND
    constants in mc_pricing were fitted on FanDuel closing lines, and FanDuel
    came out second sharpest of eleven books with the tightest hold at 0.0608.

    A MEDIAN across books is the right anchor for the research harness, where
    it estimates truth. It is the wrong anchor here, because a median across
    eight books is not a price any book offered, and settling or pricing
    against one produced a +0.2991 result at t +3.83 that was pure artifact.
    So this selects rather than averages, and a player with no FanDuel row
    drops out instead of being given a synthetic number.
    """
    if df.empty or "book" not in df.columns:
        return df, 0, None
    n_books = int(df["book"].nunique())
    sub = df[df["book"] == book]
    if sub.empty:
        return sub, n_books, None
    return sub.copy(), n_books, book

# Map various CSV market names -> the app's internal market keys
MARKET_MAP = {
    "receiving_yards": "receiving",
    "receiving yards": "receiving",
    "receiving": "receiving",
    "rec_yards": "receiving",
    "receptions": "receptions",
    "rushing": "rushing",
    "rushing_yards": "rushing",
    "rushing yards": "rushing",
    "rush_yards": "rushing",
    "passing": "qb_passing",
    "passing_yards": "qb_passing",
    "passing yards": "qb_passing",
    "qb_passing": "qb_passing",
    "pass_yards": "qb_passing",
    "anytime_td": "anytime_td",
    "anytime td": "anytime_td",
    "any_td": "anytime_td",
    "td": "anytime_td",
}


def _normalize_market(raw):
    """Translate a CSV market label into an app market key, or None if unrecognized."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return MARKET_MAP.get(key)


def _norm_name(name):
    """Normalize a player name for matching.

    Lowercase, strip whitespace, then remove punctuation and generational
    suffixes, because the same player is spelled differently by FanDuel and
    nflverse:
        'D.J. Moore'          vs 'DJ Moore'          (periods)
        'DeVon Achane'        vs "De'Von Achane"     (apostrophe)
        'Michael Pittman Jr.' vs 'Michael Pittman'   (suffix)
    Curly apostrophes from screenshot extraction are handled too.
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    for ch in (".", "'", "\u2019", "\u2018", "`", "-", ","):
        s = s.replace(ch, "")
    parts = [p for p in s.split() if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return " ".join(parts)


def _json_safe(d):
    """Replace NaN with None throughout a record, and coerce numpy scalars.

    NaN is not valid JSON and httpx serialises with allow_nan=False, so a
    single NaN anywhere in the record raises ValueError and the whole insert
    fails. Guarding individual fields kept missing cases, so the whole dict
    gets scrubbed here instead.
    """
    out = {}
    for k, v in d.items():
        if v is None or isinstance(v, (str, bool, int)):
            out[k] = v
            continue
        if isinstance(v, float):
            out[k] = None if v != v else v          # NaN != itself
            continue
        # numpy scalars (np.float64, np.int64) are not JSON-serialisable
        try:
            fv = float(v)
            out[k] = None if fv != fv else fv
        except (TypeError, ValueError):
            out[k] = v
    return out


def _to_num(v):
    if v is None or str(v).strip() == "" or str(v).strip().upper() == "UNCERTAIN":
        return None
    try:
        f = float(v)
    except Exception:
        return None
    # a NaN here would survive to the JSON encoder and crash the insert, so
    # reject it at the source as well
    return None if f != f else f


# ---------- model side + probability ----------

def _model_side(market, proj, line):
    """Which side the MODEL favours, decided by probability.

    Comparing projection to line is wrong under the gamma pricing. With the old
    symmetric normal the two agreed: a projection above the line meant P(over)
    above 0.5. The gamma puts the median well below the mean, so whenever the
    line sits between them the comparison says OVER while the probability says
    UNDER. That mislabelled the side while displaying a correct edge (Juwan
    Johnson at proj 43.9, line 43.5 showed OVER +8.5 when the model actually
    favoured the UNDER by 8.5), and a wrong side also grades wrong, which would
    have written false outcomes into the log.
    """
    import mc
    if proj is None or line is None or pd.isna(proj) or pd.isna(line):
        return ""
    p_over = mc.prob_over(market, float(proj), float(line), 0)
    if p_over is None:
        return "\u2014"
    return "OVER" if p_over > 0.5 else "UNDER"


def compute_p_over(market, proj, line, games_played=0):
    """P(over) for a yardage/count market, or None when unpriced.

    Exists so the save-to-log paths can store a real p_over instead of null.
    Below-floor projections return None by design and must stay null, never 0.
    """
    import mc
    if proj is None or line is None or pd.isna(proj) or pd.isna(line):
        return None
    try:
        p = mc.prob_over(market, float(proj), float(line), games_played)
    except Exception:
        return None
    if p is None or p != p:
        return None
    return float(p)


# ---------- imports ----------

def import_lines(rows, season, week, user, sport="NFL"):
    """Append a batch of imported lines as a NEW timestamped snapshot into 'lines'.
    Each import adds rows (never overwrites) so line movement is preserved for CLV.
    Also computes and stores the model's projection + edge AT IMPORT TIME, so the
    historical record reflects what the model actually said at that moment.

    Failed inserts are now counted and returned instead of raising mid-batch,
    so a single bad row cannot make a whole import look successful or dead.
    """
    from datetime import datetime, timezone
    import mc
    from models import receiving, receptions, rushing, qb_passing, anytime_td, qb_rushing

    MODULE_MAP = {
        "receiving": receiving, "receptions": receptions, "rushing": rushing,
        "qb_passing": qb_passing, "anytime_td": anytime_td,
    }
    GAMES_PLAYED = 0  # TODO: same placeholder as the Board; keep in sync

    client = get_authed_client(user)
    degraded = auth_degraded()
    if degraded:
        _warn(f"Importing without an authenticated session: {degraded}. "
              "Rows may be rejected by RLS.")

    captured_at = datetime.now(timezone.utc).isoformat()
    imported = 0
    failed = []
    by_market = {}
    bad_market = []

    boards_cache = {}

    def _board_for(module, mkt):
        """Played rows UNIONED with the upcoming-week assembler.

        project_week now unions these itself, so this is belt-and-braces.
        Played rows come first and win on a duplicate, because a played row
        carries the real current team and features.
        """
        frames = []
        for getter in ("project_week", "build_upcoming_week"):
            fn = getattr(module, getter, None)
            if fn is None:
                continue
            try:
                df = fn(season, week)
            except Exception:
                continue
            if df is not None and len(df) > 0:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if "player_display_name" in out.columns:
            out = out.drop_duplicates(subset=["player_display_name"], keep="first")
        return out

    def _get_board(mkt):
        if mkt not in boards_cache:
            module = MODULE_MAP.get(mkt)
            board = _board_for(module, mkt) if module else pd.DataFrame()
            if mkt == "rushing":
                qb_board = _board_for(qb_rushing, "qb_rushing")
                if len(qb_board) > 0:
                    qb_board = qb_board.copy()
                    qb_board["is_qb_model"] = True
                    board = board.copy()
                    board["is_qb_model"] = False
                    board = pd.concat([board, qb_board], ignore_index=True)
            boards_cache[mkt] = board
        return boards_cache[mkt]

    for r in rows:
        mkt = _normalize_market(r.get("market"))
        if mkt is None:
            bad_market.append(r.get("market"))
            continue

        player = str(r.get("player", "")).strip()
        if not player:
            continue

        line_val = _to_num(r.get("line"))
        over_odds = _to_num(r.get("over_odds"))
        under_odds = _to_num(r.get("under_odds"))

        projection = None
        edge = None
        board = _get_board(mkt)
        if len(board) > 0:
            match = board[board["player_display_name"].apply(_norm_name) == _norm_name(player)]
            if len(match) > 0:
                row = match.iloc[0]
                projection = float(row["projection"])
                if projection != projection:
                    projection = None

                if projection is None:
                    # no usable projection, so no edge: store nulls rather than
                    # crashing on None arithmetic
                    pass
                elif mkt == "anytime_td":
                    if over_odds is not None:
                        implied = anytime_td.american_to_prob(over_odds)
                        if implied is not None:
                            edge = round(projection - implied, 1)
                elif line_val is not None:
                    effective_mkt = "qb_rushing" if row.get("is_qb_model") else mkt
                    res = mc.edge_calc(effective_mkt, projection, line_val, GAMES_PLAYED,
                                       over_odds=over_odds, under_odds=under_odds)
                    if res:
                        edge = res["best_edge"]

        record = {
            "sport": sport, "season": int(season), "week": int(week), "market": mkt,
            "player": player, "line": line_val,
            "over_odds": over_odds, "under_odds": under_odds,
            "captured_at": captured_at,
            "projection": projection, "edge": edge,
        }
        try:
            client.table("lines").insert(_json_safe(record)).execute()
            imported += 1
            by_market[mkt] = by_market.get(mkt, 0) + 1
        except Exception as e:
            failed.append(f"{player} ({mkt}): {e}")

    return {"imported": imported, "by_market": by_market,
            "bad_market": [m for m in bad_market if m],
            "failed": failed}


# ---------- reads ----------

def get_lines(season, week, market, user, sport="NFL", book=DEFAULT_BOOK):
    """Fetch the pool of imported lines for a given market/week as a dict
    {player_norm: {'line':..., 'over_odds':..., 'under_odds':..., 'captured_at':...}}.

    Returns the most recent CAPTURED snapshot per player FROM ONE BOOK.

    FIXED September 24 2026. This used to filter on sport, season, week and
    market only, then overwrite a dict as rows came back oldest-first. With an
    eight-book capture that meant the prefilled line was whichever book
    happened to write last at the latest captured_at, which is arbitrary, and
    it also included hand-saved board rows. This function feeds the editable
    Line column, so an arbitrary book's line was what got priced.

    Symptom that found it: typed reception lines matched a captured FanDuel
    line on 92.4 percent of comparable props, but typed receiving, rushing and
    passing lines matched on only 44 to 47 percent. Reception lines are coarse
    enough that books agree, so an arbitrary book is usually FanDuel's number
    anyway; yardage lines differ between books by a yard or more routinely.

    The book matters because beta is highly sensitive to the anchor: across
    min, median, FanDuel and max it spans 0.24 to 0.37, which is 7 to 10
    standard errors. The mc_pricing.BLEND constants are fitted on FanDuel.
    """
    client = get_authed_client(user)
    degraded = auth_degraded()
    try:
        rows = _fetch_lines_rows(client, sport, season, week, market, ascending=True)
    except Exception as e:
        _warn(f"Line lookup failed for {market} wk{week}: {e}")
        return {}

    if not rows and degraded:
        _warn(f"No lines returned for {market} wk{week}, and the session is "
              f"degraded: {degraded}. This is probably auth, not missing data.")

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    df = _drop_saved_rows(df)
    sub, n_books, book_used = _select_book(df, book)

    if sub.empty and n_books > 0:
        # the book we want is not in this week's capture. Say so rather than
        # silently falling back to another book's numbers, because the pricing
        # constants are anchored on this one.
        _warn(f"No {book} rows for {market} wk{week} "
              f"({n_books} other book(s) present). Prefill left blank; enter "
              f"lines by hand and note that they are not {book}.")
        return {}

    if "captured_at" in sub.columns:
        sub = sub.sort_values("captured_at")

    out = {}
    for r in sub.to_dict("records"):
        out[_norm_name(r["player"])] = {"line": r.get("line"),
                                        "over_odds": r.get("over_odds"),
                                        "under_odds": r.get("under_odds"),
                                        "captured_at": r.get("captured_at"),
                                        "book": r.get("book"),
                                        "n_books": n_books}
    return out


def get_line_movement(season, week, market, user, sport="NFL",
                      book=DEFAULT_BOOK):
    """For a market/week, return a DataFrame with one row per player who has
    2+ captured snapshots FROM ONE BOOK: first vs. latest line/projection/edge,
    the movement between them, the latest tier, and whether the line moved
    TOWARD or AWAY from the model's read. TD (odds-only, no line) compares
    IMPLIED PROBABILITY instead of raw odds, since odds aren't linear. Players
    with only 1 snapshot are excluded (nothing to compare yet). Also carries
    the RAW latest line/odds (not just display values) so a save-to-log action
    can use them directly.

    FIXED September 24 2026, two independent contaminations:

    1. BOOK. 'lines' holds one row per book per snapshot, and this function
       treated every ROW as a snapshot. With an eight-book capture the counts
       ran about 8x high (Matthew Golden showed 55 "Line Captures" against 8
       real captured_at values), the sparkline was eight books interleaved at
       each timestamp rather than a time series, and first-row to last-row
       differencing meant Move and toward_away measured BOOK SPREAD rather
       than movement over time. Within one captured_at the book order is
       arbitrary, so the sign of Move was arbitrary too.

    2. HAND-SAVED ROWS. The Save-to-Log path writes into this same table,
       5,071 rows as of today, all labelled book='fanduel'. They carry their
       own captured_at spread across whenever a board was saved, so they
       interleaved with real captures. Excluded via projection IS NULL.

       Both defects predate the current capture and were silent: the original
       capture was FanDuel only and nothing else wrote to the table, so
       counting rows WAS counting captures.

    These are columns bets have been chosen from. n_books is now returned as
    an explicit column so the same defect cannot recur invisibly, which is the
    reason eval_harness could never hide it.
    """
    from models import anytime_td
    import mc

    client = get_authed_client(user)
    degraded = auth_degraded()

    try:
        rows = _fetch_lines_rows(client, sport, season, week, market, ascending=True)
    except Exception as e:
        _warn(f"Movement query failed for {market} wk{week}: {e}")
        return pd.DataFrame()

    if not rows:
        if degraded:
            _warn(f"No {market} rows returned for wk{week}, and the session is "
                  f"degraded: {degraded}. Log out and back in before concluding "
                  "the data is missing.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    n_saved = 0
    if "projection" in df.columns:
        n_saved = int(df["projection"].notna().sum())
    df = _drop_saved_rows(df)
    df, n_books, book_used = _select_book(df, book)

    if df.empty:
        _warn(f"No captured {book} rows for {market} wk{week} "
              f"({n_books} book(s) in the capture, {n_saved} hand-saved rows "
              f"excluded). Nothing to compare.")
        return pd.DataFrame()

    df["player_norm"] = df["player"].apply(_norm_name)
    is_td = (market == "anytime_td")

    def _side(row):
        return _model_side(market, row.get("projection"), row.get("line"))

    def _td_side(proj, implied):
        if proj is None or implied is None or pd.isna(proj) or pd.isna(implied):
            return ""
        if proj > implied:
            return "OVER"
        elif proj < implied:
            return "UNDER"
        return "\u2014"

    def _toward_away(first_side, latest_side, move):
        side = first_side if first_side in ("OVER", "UNDER") else latest_side
        if side not in ("OVER", "UNDER") or move is None or pd.isna(move):
            return ""
        if side == "OVER" and move > 0:
            return "toward"
        if side == "UNDER" and move < 0:
            return "toward"
        if move == 0:
            return "flat"
        return "away"

    out = []
    for pname, grp in df.groupby("player_norm"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("captured_at")
        first = grp.iloc[0]
        last = grp.iloc[-1]

        latest_edge = last.get("edge")
        latest_tier = (mc.tier_for_edge(latest_edge)
                       if latest_edge is not None and not pd.isna(latest_edge) else "")

        if is_td:
            first_implied = anytime_td.american_to_prob(first.get("over_odds"))
            latest_implied = anytime_td.american_to_prob(last.get("over_odds"))
            move = ((latest_implied - first_implied)
                    if (first_implied is not None and latest_implied is not None) else None)
            first_side = _td_side(first.get("projection"), first_implied)
            latest_side = _td_side(last.get("projection"), latest_implied)
            td_series = [anytime_td.american_to_prob(v) for v in grp["over_odds"].tolist()]
            td_series = [v for v in td_series if v is not None]
            out.append({
                "player": first["player"], "snapshots": len(grp),
                "first_line": first_implied, "latest_line": latest_implied,
                "line_move": move, "toward_away": _toward_away(first_side, latest_side, move),
                "first_edge": first.get("edge"), "first_side": first_side,
                "latest_edge": latest_edge, "latest_side": "", "latest_tier": latest_tier,
                "first_captured": first.get("captured_at"),
                "latest_captured": last.get("captured_at"),
                "raw_line": last.get("line"), "raw_over_odds": last.get("over_odds"),
                "raw_under_odds": last.get("under_odds"),
                "raw_projection": last.get("projection"),
                "p_over": None,
                "series": td_series,
                "n_books": n_books, "book": book_used,
            })
        else:
            first_side = _side(first)
            latest_side = _side(last)
            move = ((last.get("line") - first.get("line"))
                    if pd.notna(last.get("line")) and pd.notna(first.get("line")) else None)
            line_series = [v for v in grp["line"].tolist() if pd.notna(v)]
            out.append({
                "player": first["player"], "snapshots": len(grp),
                "first_line": first.get("line"), "latest_line": last.get("line"),
                "line_move": move, "toward_away": _toward_away(first_side, latest_side, move),
                "first_edge": first.get("edge"), "first_side": first_side,
                "latest_edge": latest_edge, "latest_side": latest_side,
                "latest_tier": latest_tier,
                "first_captured": first.get("captured_at"),
                "latest_captured": last.get("captured_at"),
                "raw_line": last.get("line"), "raw_over_odds": last.get("over_odds"),
                "raw_under_odds": last.get("under_odds"),
                "raw_projection": last.get("projection"),
                "p_over": compute_p_over(market, last.get("projection"), last.get("line")),
                "series": line_series,
                "n_books": n_books, "book": book_used,
            })
    return pd.DataFrame(out)


def diagnose_lines(season, week, user, sport="NFL"):
    """Row counts, distinct captures and multi-snapshot player counts per market.

    Exists so an empty Line Movement tab can be diagnosed from inside the app
    instead of from the Supabase SQL editor on a phone.
    """
    client = get_authed_client(user)
    degraded = auth_degraded()
    try:
        rows = _fetch_lines_rows(client, sport, season, week, None, ascending=True)
    except Exception as e:
        return pd.DataFrame(), f"query failed: {e}"

    if not rows:
        note = f"0 rows for {season} wk{week}"
        if degraded:
            note += f" | session degraded: {degraded}"
        return pd.DataFrame(), note

    df = pd.DataFrame(rows)
    df["player_norm"] = df["player"].apply(_norm_name)
    summary = []
    for mkt, grp in df.groupby("market"):
        per_player = grp.groupby("player_norm").size()
        summary.append({
            "market": mkt,
            "rows": len(grp),
            "players": int(per_player.shape[0]),
            "captures": int(grp["captured_at"].nunique()),
            "players_2plus": int((per_player >= 2).sum()),
            "null_projection": int(grp["projection"].isna().sum()),
            "null_edge": int(grp["edge"].isna().sum()),
        })
    note = f"{len(df)} rows, {df['captured_at'].nunique()} distinct captures"
    if degraded:
        note += f" | session degraded: {degraded}"
    return pd.DataFrame(summary).sort_values("market").reset_index(drop=True), note


# ---------- Authentication ----------

def sign_up(email, password):
    """Create a new user account. Returns (success, message)."""
    client = get_client()
    try:
        res = client.auth.sign_up({"email": email, "password": password})
        if res.user:
            return True, "Account created! You can now log in."
        return False, "Sign up failed, please try again."
    except Exception as e:
        return False, f"Sign up error: {e}"


def sign_in(email, password):
    """Log in an existing user. Returns (user_dict_or_None, message).
    The user dict carries the session tokens so an authenticated client can be
    built once per session.
    """
    client = get_client()
    try:
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            clear_authed_client()  # never reuse a previous user's client
            return {
                "id": res.user.id,
                "email": res.user.email,
                "access_token": res.session.access_token,
                "refresh_token": res.session.refresh_token,
            }, "Logged in!"
        return None, "Login failed, check your email and password."
    except Exception as e:
        return None, f"Login error: {e}"


def sign_out():
    """Drop the cached authenticated client so the next login builds a fresh one."""
    clear_authed_client()
    return True, "Logged out."


