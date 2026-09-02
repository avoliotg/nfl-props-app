"""Database connection + helpers for OpalScales (Supabase)."""
import streamlit as st
from supabase import create_client
import pandas as pd


@st.cache_resource
def get_client():
    """Create a cached Supabase client from secrets."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def get_user_client(access_token, refresh_token):
    """Build a Supabase client carrying a specific user's session, so its
    queries run AS that authenticated user (making auth.uid() resolve at the DB).
    Rebuilt per rerun from tokens stored in session_state. NOT cached — caching
    would leak one user's session across users.
    Returns (client, fresh_access_token, fresh_refresh_token) — if Supabase
    rotated the refresh token internally, the caller MUST persist the fresh
    ones back to session_state or subsequent calls will fail."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    client = create_client(url, key)
    try:
        client.auth.set_session(access_token, refresh_token)
    except Exception:
        # refresh token already used/invalid — session is dead, caller must re-login
        return client, None, None
    session = client.auth.get_session()
    if session:
        return client, session.access_token, session.refresh_token
    return client, access_token, refresh_token

def get_authed_client(user):
    """Given the session_state user dict, return an authenticated client,
    refreshing and persisting tokens back into `user` (in place) if Supabase
    rotated them. Falls back to the anon client if no valid tokens exist."""
    if not (user and isinstance(user, dict) and user.get("access_token") and user.get("refresh_token")):
        return get_client()
    client, fresh_access, fresh_refresh = get_user_client(user["access_token"], user["refresh_token"])
    if fresh_access is None:
        # session dead (refresh token already used/invalid) — degrade to anon;
        # user needs to log out and back in to restore authenticated access
        return get_client()
    if fresh_access != user["access_token"]:
        user["access_token"] = fresh_access
        user["refresh_token"] = fresh_refresh
        st.session_state.user = user  # persist the rotation so future calls use fresh tokens
    return client


def test_connection():
    """Quick check: can we reach the database? Returns (ok, message)."""
    try:
        url = st.secrets["SUPABASE_URL"]
        client = get_client()
        resp = client.table("lines").select("*").limit(1).execute()
        return True, f"Connected! ({len(resp.data)} rows)"
    except Exception as e:
        return False, f"Connection failed: {e} | URL used: {st.secrets['SUPABASE_URL']}"
    

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


def import_lines(rows, season, week, user, sport="NFL"):
    """Append a batch of imported lines as a NEW timestamped snapshot into 'lines'.
    Each import adds rows (never overwrites) so line movement is preserved for CLV.
    Also computes and stores the model's projection + edge AT IMPORT TIME, so the
    historical record reflects what the model actually said at that moment.
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

    captured_at = datetime.now(timezone.utc).isoformat()
    imported = 0
    by_market = {}
    bad_market = []

    boards_cache = {}
    def _get_board(mkt):
        if mkt not in boards_cache:
            module = MODULE_MAP.get(mkt)
            board = module.project_week(season, week) if module else pd.DataFrame()
            if mkt == "rushing":
                qb_board = qb_rushing.project_week(season, week)
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
            match = board[board["player_display_name"] == player]
            if len(match) > 0:
                row = match.iloc[0]
                projection = float(row["projection"])
                if mkt == "anytime_td":
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
        client.table("lines").insert(record).execute()
        imported += 1
        by_market[mkt] = by_market.get(mkt, 0) + 1

    return {"imported": imported, "by_market": by_market,
            "bad_market": [m for m in bad_market if m]}


def get_lines(season, week, market, user, sport="NFL"):
    """Fetch the pool of imported lines for a given market/week as a dict
    {player_name: {'line':..., 'over_odds':..., 'under_odds':...}}.
    Returns the MOST RECENT snapshot per player. Uses the authenticated user's
    client so RLS (read-for-authenticated) resolves correctly."""
    client = get_authed_client(user)
    try:
        resp = (client.table("lines").select("*")
                .eq("sport", sport).eq("season", int(season))
                .eq("week", int(week)).eq("market", market)
                .order("captured_at", desc=False).execute())
        out = {}
        for r in resp.data:
            out[r["player"]] = {"line": r.get("line"),
                                "over_odds": r.get("over_odds"),
                                "under_odds": r.get("under_odds"),
                                "captured_at": r.get("captured_at")}
        return out
    except Exception:
        return {}


def _to_num(v):
    if v is None or str(v).strip() == "" or str(v).strip().upper() == "UNCERTAIN":
        return None
    try:
        return float(v)
    except Exception:
        return None
    # ---------- Authentication ----------

def sign_up(email, password):
    """Create a new user account. Returns (success, message)."""
    client = get_client()
    try:
        res = client.auth.sign_up({"email": email, "password": password})
        if res.user:
            return True, "Account created! You can now log in."
        return False, "Sign up failed — please try again."
    except Exception as e:
        return False, f"Sign up error: {e}"


def sign_in(email, password):
    """Log in an existing user. Returns (user_dict_or_None, message).
    The user dict now also carries the session tokens so we can rebuild an
    authenticated client per rerun."""
    client = get_client()
    try:
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            return {
                "id": res.user.id,
                "email": res.user.email,
                "access_token": res.session.access_token,
                "refresh_token": res.session.refresh_token,
            }, "Logged in!"
        return None, "Login failed — check your email and password."
    except Exception as e:
        return None, f"Login error: {e}"