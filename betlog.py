"""
Bet log — persistent storage for graded picks.
Now backed by the Supabase 'bets' table (was CSV). Function names/signatures
kept identical so the rest of the app is unchanged.

Note: the app uses the column name 'result_yards'; the database uses 'result'.
This module translates between them so app.py doesn't need to change.
"""
from datetime import datetime
import pandas as pd
import db

COLUMNS = ["logged_at", "market", "season", "week", "player",
           "projection", "line", "gap", "confidence", "side",
           "tier", "bet", "result_yards", "outcome"]

TABLE = "bets"


def _client():
    return db.get_client()


def load_log():
    """Return the full log as a DataFrame (empty if none yet)."""
    try:
        resp = _client().table(TABLE).select("*").execute()
        rows = resp.data
    except Exception:
        rows = []
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    # translate DB 'result' -> app's 'result_yards'
    if "result" in df.columns:
        df = df.rename(columns={"result": "result_yards"})
    # ensure all expected columns exist (in case some are null/missing)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNS]


def append_entries(entries: pd.DataFrame):
    """Insert/replace graded picks in the DB, de-duplicating on
    market+season+week+player (latest entry wins)."""
    client = _client()
    for _, row in entries.iterrows():
        record = {
            "logged_at": row.get("logged_at"),
            "market": row.get("market"),
            "season": int(row["season"]) if pd.notna(row.get("season")) else None,
            "week": int(row["week"]) if pd.notna(row.get("week")) else None,
            "player": row.get("player"),
            "projection": _num(row.get("projection")),
            "line": _num(row.get("line")),
            "gap": _num(row.get("gap")),
            "confidence": _num(row.get("confidence")),
            "side": row.get("side"),
            "tier": row.get("tier"),
            "bet": bool(row.get("bet")) if pd.notna(row.get("bet")) else False,
            "result": _num(row.get("result_yards")),
            "outcome": row.get("outcome") if pd.notna(row.get("outcome")) else None,
        }
        # delete any existing row for this unique pick, then insert (latest wins)
        (client.table(TABLE)
               .delete()
               .eq("market", record["market"])
               .eq("season", record["season"])
               .eq("week", record["week"])
               .eq("player", record["player"])
               .execute())
        client.table(TABLE).insert(record).execute()
    return load_log()


def grade_log(actuals_lookup, is_prob=False):
    """Fill in result + outcome for logged picks that don't have a result yet.
    actuals_lookup(season, week, player) -> actual result, or None."""
    client = _client()
    log = load_log()
    if len(log) == 0:
        return log

    for _, row in log.iterrows():
        if pd.notna(row.get("outcome")) and str(row.get("outcome")).strip():
            continue  # already graded
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
               .eq("market", row["market"])
               .eq("season", int(row["season"]))
               .eq("week", int(row["week"]))
               .eq("player", row["player"])
               .execute())
    return load_log()


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