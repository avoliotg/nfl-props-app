"""
Bet log — persistent storage for graded picks.
Backed by the Supabase 'bets' table.

Note: the app uses the column name 'result_yards'; the database uses 'result'.
This module translates between them so app.py doesn't need to change.
"""
from datetime import datetime
import pandas as pd
import db

COLUMNS = ["logged_at", "market", "season", "week", "player",
           "projection", "line", "over_odds", "under_odds",
           "edge", "p_over", "side", "tier", "bet",
           "result_yards", "outcome"]

TABLE = "bets"


def _client():
    return db.get_client()


def load_log(user_id="admin"):
    """Return the full log as a DataFrame (empty if none yet)."""
    try:
        resp = _client().table(TABLE).select("*").eq("user_id", user_id).execute()
        rows = resp.data
    except Exception:
        rows = []
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    if "result" in df.columns:
        df = df.rename(columns={"result": "result_yards"})
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNS]


def append_entries(entries: pd.DataFrame, user_id="admin"):
    """Insert/replace graded picks in the DB, de-duplicating on
    market+season+week+player (latest entry wins)."""
    client = _client()
    for _, row in entries.iterrows():
        record = {
            "user_id": user_id,
            "logged_at": row.get("logged_at"),
            "market": row.get("market"),
            "season": int(row["season"]) if pd.notna(row.get("season")) else None,
            "week": int(row["week"]) if pd.notna(row.get("week")) else None,
            "player": row.get("player"),
            "projection": _num(row.get("projection")),
            "line": _num(row.get("line")),
            "over_odds": _num(row.get("over_odds")),
            "under_odds": _num(row.get("under_odds")),
            "edge": _num(row.get("edge")),
            "p_over": _num(row.get("p_over")),
            "side": row.get("side"),
            "tier": row.get("tier"),
            "bet": bool(row.get("bet")) if pd.notna(row.get("bet")) else False,
            "result": _num(row.get("result_yards")),
            "outcome": row.get("outcome") if pd.notna(row.get("outcome")) else None,
        }
        (client.table(TABLE)
               .delete()
               .eq("user_id", user_id)
               .eq("market", record["market"])
               .eq("season", record["season"])
               .eq("week", record["week"])
               .eq("player", record["player"])
               .execute())
        client.table(TABLE).insert(record).execute()
    return load_log(user_id)


def grade_log(actuals_lookup, is_prob=False, user_id="admin"):
    """Fill in result + outcome for logged picks that don't have a result yet.
    actuals_lookup(season, week, player) -> actual result, or None."""
    client = _client()
    log = load_log(user_id)
    if len(log) == 0:
        return log

    for _, row in log.iterrows():
        if pd.notna(row.get("outcome")) and str(row.get("outcome")).strip():
            continue
        actual = actuals_lookup(row["season"], row["week"], row["player"])
        if actual is None:
            continue

        if is_prob:
            outcome = "WIN" if actual >= 100 else "LOSS"
        else:
            line = float(row["line"])
            if actual == line:
                outcome = "PUSH"
            else:
                went_over = actual > line
                picked_over = (row["side"] == "OVER")
                outcome = "WIN" if went_over == picked_over else "LOSS"

        (client.table(TABLE)
               .update({"result": float(actual), "outcome": outcome})
               .eq("user_id", user_id)
               .eq("market", row["market"])
               .eq("season", int(row["season"]))
               .eq("week", int(row["week"]))
               .eq("player", row["player"])
               .execute())
    return load_log(user_id)


def now_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _num(v):
    """Coerce to float or None for DB numeric columns."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None