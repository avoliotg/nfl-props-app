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
    "rec_yards": "receiving",
    "receptions": "receptions",
    "rushing_yards": "rushing",
    "rushing yards": "rushing",
    "rush_yards": "rushing",
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
    """Upsert a batch of imported lines into the 'lines' table.
    rows: list of dicts with keys player, market, line, over_odds, under_odds.
    Returns a summary dict: imported count, per-market counts, and any problems.
    """
    client = get_client()
    imported = 0
    by_market = {}
    bad_market = []   # rows whose market wasn't recognized
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
        }
        if not record["player"]:
            continue
        # upsert on the unique key (sport+season+week+market+player)
        client.table("lines").upsert(
            record, on_conflict="sport,season,week,market,player"
        ).execute()
        imported += 1
        by_market[mkt] = by_market.get(mkt, 0) + 1
    return {"imported": imported, "by_market": by_market,
            "bad_market": [m for m in bad_market if m]}


def get_lines(season, week, market, sport="NFL"):
    """Fetch the pool of imported lines for a given market/week as a dict
    {player_name: {'line':..., 'over_odds':..., 'under_odds':...}}."""
    client = get_client()
    try:
        resp = (client.table("lines").select("*")
                .eq("sport", sport).eq("season", int(season))
                .eq("week", int(week)).eq("market", market).execute())
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