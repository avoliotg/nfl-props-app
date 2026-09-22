"""
Fill the null `week` column in historical_lines from the nflverse schedule.

The backfill deliberately stored `week` as null and kept the fetch dumb, since
a wrong week silently files a prop under the wrong game and would be invisible
later. This script does the matching separately, where it can be verified and
re-run.

The join is on kickoff time plus teams. The API uses full names ("Dallas
Cowboys") while nflverse uses abbreviations ("DAL"), so the map below is
required. It is checked against the schedule on every run, so a name the API
changes shows up as a loud failure rather than a silent miss.

Matching rule: a prop's event matches a scheduled game when the two teams are
the same pair and the kickoff times are within 6 hours. Time alone is not
enough, because a dozen games share a 1pm Sunday slot. Teams alone are not
enough either, because the same pair meets twice a season.

Run from the repo root with the venv active:
    python fill_weeks.py --dry-run          report match rate, write nothing
    python fill_weeks.py --run              write the week values
    python fill_weeks.py --run --season 2025
"""
import argparse
import os
import sys
from getpass import getpass

import pandas as pd

SECRETS_PATH = os.path.join(".streamlit", "secrets.toml")
TABLE = "historical_lines"
PAGE = 1000
MATCH_WINDOW_HOURS = 6

# The Odds API full team name -> nflverse abbreviation.
TEAM_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


# ------------------------------------------------------------------- plumbing

def _secrets():
    if not os.path.exists(SECRETS_PATH):
        return {}
    try:
        import tomllib
        with open(SECRETS_PATH, "rb") as fh:
            return tomllib.load(fh)
    except Exception as e:
        print(f"  could not parse the secrets file: {e}")
        return {}


def connect(sec):
    from supabase import create_client
    url = sec.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = sec.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    if not (url and key):
        print("  missing Supabase credentials")
        sys.exit(1)
    email = (os.environ.get("OPAL_EMAIL") or sec.get("OPAL_EMAIL")
             or sec.get("ADMIN_EMAIL") or input("email: ").strip())
    pw = (os.environ.get("OPAL_PASSWORD") or sec.get("OPAL_PASSWORD")
          or getpass("password (hidden): "))
    client = create_client(url, key)
    res = client.auth.sign_in_with_password({"email": email, "password": pw})
    if not res.user:
        print("  login failed")
        sys.exit(1)
    print(f"  authenticated as {res.user.email}")
    return client


def fetch_events(client, season=None):
    """Distinct events in historical_lines, paged past the 1000-row cap."""
    rows = []
    start = 0
    while True:
        q = (client.table(TABLE)
             .select("event_id,season,commence_time,home_team,away_team,week")
             .order("id", desc=False))
        if season is not None:
            q = q.eq("season", int(season))
        resp = q.range(start, start + PAGE - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        start += PAGE
        if start > 2000000:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # one row per event, keeping whether any row already has a week
    agg = (df.groupby(["event_id", "season"], as_index=False)
             .agg(commence_time=("commence_time", "first"),
                  home_team=("home_team", "first"),
                  away_team=("away_team", "first"),
                  week_filled=("week", lambda s: s.notna().any()),
                  n_rows=("event_id", "size")))
    return agg


# -------------------------------------------------------------------- joining

def load_schedule(seasons):
    from models import data_utils
    sched = data_utils.load_schedules(sorted(seasons))
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    keep = ["season", "week", "gameday", "gametime", "home_team", "away_team"]
    keep = [c for c in keep if c in sched.columns]
    sched = sched[keep].copy()
    # build a kickoff timestamp. gameday is a date and gametime a local
    # HH:MM in US Eastern, which is how nflverse publishes it.
    gd = pd.to_datetime(sched["gameday"], errors="coerce")
    if "gametime" in sched.columns:
        combined = gd.dt.strftime("%Y-%m-%d") + " " + sched["gametime"].fillna("13:00")
        kick = pd.to_datetime(combined, errors="coerce")
    else:
        kick = gd
    try:
        kick = (kick.dt.tz_localize("America/New_York", nonexistent="shift_forward",
                                    ambiguous="NaT").dt.tz_convert("UTC"))
    except Exception:
        kick = kick.dt.tz_localize("UTC")
    sched["kickoff"] = kick
    sched["pair"] = sched.apply(
        lambda r: tuple(sorted([str(r["home_team"]), str(r["away_team"])])), axis=1)
    return sched


def check_team_map(events, sched):
    """Any API team name missing from the map is a hard failure."""
    api_names = set(events["home_team"].dropna()) | set(events["away_team"].dropna())
    missing = sorted(n for n in api_names if n not in TEAM_ABBR)
    if missing:
        print("\n  >>> API team names not in TEAM_ABBR:")
        for n in missing:
            print(f"        {n}")
        print("  >>> Add them before running. A missing name silently drops")
        print("  >>> every prop for that team.")
        return False
    sched_abbrs = set(sched["home_team"].dropna()) | set(sched["away_team"].dropna())
    unknown = sorted(v for v in set(TEAM_ABBR.values()) if v not in sched_abbrs)
    if unknown:
        print(f"\n  note: mapped abbreviations not seen in the schedule: {unknown}")
        print("  (expected for relocations or a season a team did not play)")
    return True


def match_events(events, sched):
    """Attach a week to each event. Returns (matched, unmatched) frames."""
    ev = events.copy()
    ev["kick"] = pd.to_datetime(ev["commence_time"], errors="coerce", utc=True)
    ev["home_abbr"] = ev["home_team"].map(TEAM_ABBR)
    ev["away_abbr"] = ev["away_team"].map(TEAM_ABBR)
    ev["pair"] = ev.apply(
        lambda r: tuple(sorted([str(r["home_abbr"]), str(r["away_abbr"])])), axis=1)

    window = pd.Timedelta(hours=MATCH_WINDOW_HOURS)
    weeks = []
    reasons = []
    for _, r in ev.iterrows():
        cand = sched[(sched["season"] == r["season"]) & (sched["pair"] == r["pair"])]
        if len(cand) == 0:
            weeks.append(None)
            reasons.append("no scheduled game for that team pair")
            continue
        delta = (cand["kickoff"] - r["kick"]).abs()
        near = cand[delta <= window]
        if len(near) == 0:
            best = delta.min()
            weeks.append(None)
            reasons.append(f"nearest scheduled kickoff off by {best}")
            continue
        if len(near) > 1:
            # two meetings inside 6 hours is impossible, so take the nearest
            near = near.loc[[delta.idxmin()]]
        weeks.append(int(near.iloc[0]["week"]))
        reasons.append("")
    ev["week_matched"] = weeks
    ev["reason"] = reasons
    matched = ev[ev["week_matched"].notna()].copy()
    unmatched = ev[ev["week_matched"].isna()].copy()
    return matched, unmatched


# --------------------------------------------------------------------- update

def write_weeks(client, matched):
    """One update per event, which is a few hundred calls rather than 100k."""
    done = 0
    failed = 0
    for _, r in matched.iterrows():
        try:
            (client.table(TABLE)
             .update({"week": int(r["week_matched"])})
             .eq("event_id", r["event_id"]).execute())
            done += 1
        except Exception as e:
            failed += 1
            print(f"    update failed for {r['event_id']}: {type(e).__name__}")
        if done % 50 == 0 and done:
            print(f"    {done} events updated")
    return done, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="rewrite weeks even where already filled")
    args = ap.parse_args()
    if not (args.dry_run or args.run):
        print("pass --dry-run or --run")
        sys.exit(1)

    print("=" * 74)
    print("FILL WEEK NUMBERS IN historical_lines")
    print("=" * 74)

    sec = _secrets()
    client = connect(sec)

    events = fetch_events(client, args.season)
    if events.empty:
        print("  no rows found")
        return
    print(f"  {len(events)} distinct events, "
          f"{int(events['n_rows'].sum())} prop rows")
    print(f"  already filled: {int(events['week_filled'].sum())} events")

    seasons = sorted(events["season"].dropna().unique().astype(int).tolist())
    print(f"  seasons present: {seasons}")

    sched = load_schedule(seasons)
    print(f"  schedule rows loaded: {len(sched)}")

    if not check_team_map(events, sched):
        sys.exit(1)

    todo = events if args.force else events[~events["week_filled"]]
    print(f"  events to match: {len(todo)}")
    if len(todo) == 0:
        print("  nothing to do")
        return

    matched, unmatched = match_events(todo, sched)
    rate = 100.0 * len(matched) / len(todo)
    print(f"\n  matched {len(matched)} of {len(todo)} ({rate:.1f}%)")

    if len(matched):
        by_season = (matched.groupby("season")["week_matched"]
                     .agg(["count", "min", "max"]))
        print("\n  matched events by season:")
        print(by_season.to_string())
        rows_matched = int(matched["n_rows"].sum())
        print(f"\n  prop rows that would get a week: {rows_matched}")

    if len(unmatched):
        print(f"\n  >>> {len(unmatched)} unmatched events:")
        for _, r in unmatched.head(20).iterrows():
            print(f"      {r['season']} {r['away_team']} at {r['home_team']} "
                  f"{r['commence_time']}  ({r['reason']})")
        if len(unmatched) > 20:
            print(f"      and {len(unmatched) - 20} more")

    if args.dry_run:
        print("\n  dry run, nothing written.")
        print("  Re-run with --run if the match rate looks right.")
        return

    if len(matched) == 0:
        print("  nothing matched, so nothing to write")
        return

    print(f"\n  writing weeks for {len(matched)} events")
    done, failed = write_weeks(client, matched)
    print("\n" + "=" * 74)
    print(f"  updated {done} events, {failed} failed")
    print("  verify with:")
    print("    select season, week, count(*) from historical_lines")
    print("    group by 1,2 order by 1,2;")
    print("=" * 74)


if __name__ == "__main__":
    main()
