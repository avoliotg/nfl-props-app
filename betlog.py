"""
Bet log — persistent storage for graded picks.
Stores to a CSV so entries survive restarts. Storage logic lives
here so we can swap in a database later without touching the app.
"""
from pathlib import Path
from datetime import datetime
import pandas as pd

LOG_PATH = Path(__file__).parent / "data" / "bet_log.csv"

COLUMNS = ["logged_at", "market", "season", "week", "player",
           "projection", "line", "gap", "confidence", "side",
           "tier", "bet", "result_yards", "outcome"]


def load_log():
    """Return the full log as a DataFrame (empty if none yet)."""
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=COLUMNS)


def append_entries(entries: pd.DataFrame):
    """Append new graded picks to the log, de-duplicating on
    market+season+week+player (latest entry wins)."""
    existing = load_log()
    combined = pd.concat([existing, entries], ignore_index=True)
    # keep the most recent entry per unique pick
    combined = combined.drop_duplicates(
        subset=["market", "season", "week", "player"], keep="last"
    ).reset_index(drop=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(LOG_PATH, index=False)
    return combined


def now_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M")
def grade_log(actuals_lookup, is_prob=False):
    """Fill in result_yards + outcome for logged picks.
    actuals_lookup(season, week, player) -> the actual result, or None.
    - Yardage markets: actual is yards/count; WIN if the picked side (OVER/UNDER
      vs the line) was correct.
    - Probability markets (TD): actual is 100 (scored) or 0 (didn't); an anytime-TD
      bet wins if the player scored."""
    log = load_log()
    if len(log) == 0:
        return log

    log["outcome"] = log["outcome"].astype("object")
    log["result_yards"] = log["result_yards"].astype("object")

    for i, row in log.iterrows():
        if pd.notna(row.get("outcome")) and str(row.get("outcome")).strip():
            continue  # already graded
        actual = actuals_lookup(row["season"], row["week"], row["player"])
        if actual is None:
            continue  # game hasn't happened / no data yet
        log.at[i, "result_yards"] = actual

        if is_prob:
            # anytime-TD bet: win if the player scored (actual == 100)
            log.at[i, "outcome"] = "WIN" if actual >= 100 else "LOSS"
        else:
            line = float(row["line"])
            if actual == line:
                log.at[i, "outcome"] = "PUSH"
            else:
                went_over = actual > line
                picked_over = (row["side"] == "OVER")
                log.at[i, "outcome"] = "WIN" if went_over == picked_over else "LOSS"

    log.to_csv(LOG_PATH, index=False)
    return log