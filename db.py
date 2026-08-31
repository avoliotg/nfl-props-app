"""Database connection + helpers for OpalScales (Supabase)."""
import streamlit as st
from supabase import create_client


@st.cache_resource
def get_client():
    """Create a cached Supabase client from secrets."""
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def test_connection():
    """Quick check: can we reach the database? Returns (ok, message)."""
    try:
        url = st.secrets["SUPABASE_URL"]
        client = get_client()
        resp = client.table("lines").select("*").limit(1).execute()
        return True, f"Connected! ({len(resp.data)} rows)"
    except Exception as e:
        return False, f"Connection failed: {e} | URL used: {st.secrets['SUPABASE_URL']}"
    import pandas as pd

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


def import_lines(rows, season, week, sport="NFL"):
    """Append a batch of imported lines as a NEW timestamped snapshot into 'lines'.
    Each import adds rows (never overwrites) so line movement is preserved for CLV.
    rows: list of dicts with keys player, market, line, over_odds, under_odds.
    Returns a summary dict: imported count, per-market counts, and any problems.
    """
    from datetime import datetime, timezone
    client = get_client()
    captured_at = datetime.now(timezone.utc).isoformat()
    imported = 0
    by_market = {}
    bad_market = []
    for r in rows:
        mkt = _normalize_market(r.get("market"))
        if mkt is None:
            bad_market.append(r.get("market"))
            continue
        record = {
            "sport": sport,
            "season": int(season),
            "week": int(week),
            "market": mkt,
            "player": str(r.get("player", "")).strip(),
            "line": _to_num(r.get("line")),
            "over_odds": _to_num(r.get("over_odds")),
            "under_odds": _to_num(r.get("under_odds")),
            "captured_at": captured_at,
        }
        if not record["player"]:
            continue
        client.table("lines").insert(record).execute()   # append, don't overwrite
        imported += 1
        by_market[mkt] = by_market.get(mkt, 0) + 1
    return {"imported": imported, "by_market": by_market,
            "bad_market": [m for m in bad_market if m]}


def get_lines(season, week, market, sport="NFL"):
    """Fetch the pool of imported lines for a given market/week as a dict
    {player_name: {'line':..., 'over_odds':..., 'under_odds':...}}.
    Returns the MOST RECENT snapshot per player (movement history is preserved
    in the table; the Board shows the latest)."""
    client = get_client()
    try:
        resp = (client.table("lines").select("*")
                .eq("sport", sport).eq("season", int(season))
                .eq("week", int(week)).eq("market", market)
                .order("captured_at", desc=False).execute())
        out = {}
        for r in resp.data:
            out[r["player"]] = {"line": r.get("line"),
                                "over_odds": r.get("over_odds"),
                                "under_odds": r.get("under_odds")}
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
    """Log in an existing user. Returns (user_dict_or_None, message)."""
    client = get_client()
    try:
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            return {"id": res.user.id, "email": res.user.email}, "Logged in!"
        return None, "Login failed — check your email and password."
    except Exception as e:
        return None, f"Login error: {e}"