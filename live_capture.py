"""
Live prop line capture from The Odds API.

Replaces the screenshot-to-CSV transcription workflow. One command pulls every
upcoming game, every market, every US book, and writes a timestamped snapshot
to the `lines` table.

    live: markets x regions credits per event
          5 NFL markets x 1 region = 5 credits per game, so about 85 per slate

`regions=us` returns every US book in one call, so the multi-book data costs
nothing extra.

RUNS IDENTICALLY in three places, which is the point:
  cmd            credentials from .streamlit/secrets.toml
  Colab          credentials from environment variables
  GitHub Actions credentials from encrypted repo secrets

Environment variables take precedence over the secrets file:
    ODDS_API_KEY, SUPABASE_URL, SUPABASE_KEY, OPAL_EMAIL, OPAL_PASSWORD

SECURITY NOTE: the repo is public, so GitHub Actions logs are world readable.
Nothing here ever prints a key, a password, or a URL containing the key.

Usage:
    python live_capture.py                      capture NFL now
    python live_capture.py --dry-run            show what would be pulled, 0 credits
    python live_capture.py --sport nhl          NHL shots on goal
    python live_capture.py --hours-ahead 72     only games kicking off within 72h
    python live_capture.py --ceiling 500        hard stop on credits this run
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

BASE = "https://api.the-odds-api.com/v4"
SECRETS_PATH = os.path.join(".streamlit", "secrets.toml")

NFL = "americanfootball_nfl"
NHL = "icehockey_nhl"

NFL_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
]
NHL_MARKETS = ["player_shots_on_goal"]

# API market key -> the app's internal market name
MARKET_TO_APP = {
    "player_pass_yds": "qb_passing",
    "player_rush_yds": "rushing",
    "player_reception_yds": "receiving",
    "player_receptions": "receptions",
    "player_anytime_td": "anytime_td",
    "player_shots_on_goal": "shots_on_goal",
}

TABLE = "lines"


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


def cfg(name, sec, required=True, secret=True):
    """Environment first, then the secrets file. Never prints the value."""
    val = os.environ.get(name, "").strip() or str(sec.get(name, "")).strip()
    if not val and required:
        print(f"  missing {name}. Set it in the environment or "
              f"{SECRETS_PATH}.")
        sys.exit(1)
    if val:
        where = "environment" if os.environ.get(name) else SECRETS_PATH
        shown = "set" if secret else val
        print(f"  {name}: {shown} (from {where})")
    return val


class Api:
    def __init__(self, key, ceiling=None):
        self._key = key
        self.ceiling = ceiling
        self.used = None
        self.remaining = None
        self.last_cost = 0
        self.spent = 0
        self.calls = 0

    def get(self, path, **params):
        if self.ceiling is not None and self.spent >= self.ceiling:
            raise RuntimeError(
                f"credit ceiling reached ({self.spent} spent this run)")
        params["apiKey"] = self._key
        for attempt in range(4):
            try:
                r = requests.get(f"{BASE}{path}", params=params, timeout=45)
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"    network error, retrying in {wait}s ({type(e).__name__})")
                time.sleep(wait)
                continue
            self.calls += 1
            try:
                self.last_cost = int(r.headers.get("x-requests-last", 0))
            except ValueError:
                self.last_cost = 0
            self.spent += self.last_cost
            self.used = r.headers.get("x-requests-used", self.used)
            self.remaining = r.headers.get("x-requests-remaining", self.remaining)

            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"    rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            # never echo the response body: it can contain the request URL
            print(f"    HTTP {r.status_code} on {path}")
            return None
        print("    giving up after retries")
        return None

    def line(self):
        return (f"credits used {self.used}, remaining {self.remaining}, "
                f"this run {self.spent}, calls {self.calls}")


# ---------------------------------------------------------------------- parse

def parse_event(payload, sport, season, week, captured_at):
    """Flatten one event-odds payload into `lines` rows.

    Over and under for the same player are folded into one row, matching the
    schema the transcription workflow produced. anytime_td arrives as a single
    'Yes' outcome with no point, so line stays null and over_odds carries the
    price, which is how the app already stores it.
    """
    rows = []
    if not payload:
        return rows
    for bk in payload.get("bookmakers", []) or []:
        book = bk.get("key")
        for mkt in bk.get("markets", []) or []:
            app_market = MARKET_TO_APP.get(mkt.get("key"))
            if app_market is None:
                continue
            per_player = {}
            for o in mkt.get("outcomes", []) or []:
                player = o.get("description")
                if not player:
                    continue
                side = str(o.get("name", "")).lower()
                rec = per_player.setdefault(
                    player, {"line": None, "over_odds": None, "under_odds": None})
                if o.get("point") is not None:
                    rec["line"] = o.get("point")
                if side in ("over", "yes"):
                    rec["over_odds"] = o.get("price")
                elif side in ("under", "no"):
                    rec["under_odds"] = o.get("price")
            for player, rec in per_player.items():
                rows.append({
                    "sport": "NFL" if sport == NFL else "NHL",
                    "book": book,
                    "season": int(season),
                    "week": int(week) if week is not None else None,
                    "market": app_market,
                    "player": player,
                    "line": rec["line"],
                    "over_odds": rec["over_odds"],
                    "under_odds": rec["under_odds"],
                    "captured_at": captured_at,
                    "projection": None,
                    "edge": None,
                })
    return rows


def infer_season(now=None):
    """NFL seasons are labelled by their starting year, and run Sep to Feb."""
    t = now or datetime.now(timezone.utc)
    return t.year if t.month >= 3 else t.year - 1


def infer_week(sport, season, commence_time):
    """Week number from the nflverse schedule, or None if unavailable.

    Falls back to None rather than guessing, because a wrong week silently
    files a snapshot under the wrong game and would be invisible later.
    """
    if sport != NFL or not commence_time:
        return None
    try:
        from models import data_utils
        sched = data_utils.load_schedules([season])
        sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    except Exception:
        return None
    try:
        import pandas as pd
        ct = pd.to_datetime(commence_time, utc=True)
        gd = pd.to_datetime(sched["gameday"], errors="coerce").dt.tz_localize(
            "UTC", nonexistent="shift_forward")
        # a game kicking off late UTC belongs to the prior US date, so match
        # within a two-day window and take the nearest
        near = sched[(gd - ct).abs() <= pd.Timedelta(days=2)]
        if len(near) == 0:
            return None
        idx = (pd.to_datetime(near["gameday"], errors="coerce")
               .dt.tz_localize("UTC", nonexistent="shift_forward") - ct).abs().idxmin()
        return int(near.loc[idx, "week"])
    except Exception:
        return None


# -------------------------------------------------------------------- storage

def supa_client(sec):
    from supabase import create_client
    url = cfg("SUPABASE_URL", sec, secret=True)
    key = cfg("SUPABASE_KEY", sec, secret=True)
    email = cfg("OPAL_EMAIL", sec, secret=False)
    pw = cfg("OPAL_PASSWORD", sec, secret=True)
    client = create_client(url, key)
    res = client.auth.sign_in_with_password({"email": email, "password": pw})
    if not res.user:
        print("  Supabase login failed")
        sys.exit(1)
    print(f"  Supabase: authenticated as {res.user.email}")
    return client


def insert_rows(client, rows, chunk=400):
    written = 0
    for i in range(0, len(rows), chunk):
        batch = rows[i:i + chunk]
        try:
            client.table(TABLE).insert(batch).execute()
            written += len(batch)
        except Exception as e:
            # print the type only: the message can echo row contents
            print(f"    insert failed for {len(batch)} rows: {type(e).__name__}")
            print(f"    {str(e)[:200]}")
    return written


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="nfl", choices=["nfl", "nhl"])
    ap.add_argument("--dry-run", action="store_true",
                    help="list events and price the pull, no odds calls")
    ap.add_argument("--hours-ahead", type=int, default=240,
                    help="only games kicking off within this many hours")
    ap.add_argument("--ceiling", type=int, default=None)
    ap.add_argument("--season", type=int, default=None,
                    help="override the inferred season label")
    ap.add_argument("--no-week", action="store_true",
                    help="skip the nflverse week lookup (faster, week stays null)")
    args = ap.parse_args()

    sport = NFL if args.sport == "nfl" else NHL
    markets = NFL_MARKETS if sport == NFL else NHL_MARKETS
    season = args.season or infer_season()
    per_event = len(markets)

    print("=" * 70)
    print(f"LIVE CAPTURE  sport={args.sport}  season={season}")
    print("=" * 70)

    sec = _secrets()
    api = Api(cfg("ODDS_API_KEY", sec), ceiling=args.ceiling)

    events = api.get(f"/sports/{sport}/events")
    if not events:
        print("  no events returned")
        return

    cutoff = datetime.now(timezone.utc) + timedelta(hours=args.hours_ahead)
    keep = []
    for e in events:
        ct = e.get("commence_time")
        if not ct:
            continue
        try:
            t = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except ValueError:
            continue
        if t <= cutoff:
            keep.append(e)
    keep.sort(key=lambda e: e["commence_time"])

    print(f"\n  {len(events)} upcoming events, {len(keep)} within "
          f"{args.hours_ahead}h")
    print(f"  {len(markets)} markets x 1 region = {per_event} credits per event")
    print(f"  estimated cost: {len(keep) * per_event} credits")

    if args.dry_run:
        print("\n  events that would be pulled:")
        for e in keep:
            print(f"    {e['commence_time']}  {e.get('away_team')} at "
                  f"{e.get('home_team')}")
        print(f"\n  dry run only. {api.line()}")
        return

    if not keep:
        print("  nothing to capture")
        return

    client = supa_client(sec)

    # One timestamp for the whole run, so every row in this snapshot shares a
    # captured_at. Line Movement groups on that, and per-event timestamps
    # would fragment a single capture into many.
    captured_at = datetime.now(timezone.utc).isoformat()
    print(f"\n  snapshot timestamp: {captured_at}")

    week_cache = {}
    total = 0
    for n, e in enumerate(keep, 1):
        label = f"{e.get('away_team')} at {e.get('home_team')}"
        try:
            payload = api.get(f"/sports/{sport}/events/{e['id']}/odds",
                              regions="us", oddsFormat="american",
                              markets=",".join(markets))
        except RuntimeError as err:
            print(f"\n  {err}")
            break
        if not payload:
            print(f"  [{n}/{len(keep)}] {label}: no payload")
            continue

        week = None
        if not args.no_week:
            ct = e.get("commence_time")
            if ct not in week_cache:
                week_cache[ct] = infer_week(sport, season, ct)
            week = week_cache[ct]

        rows = parse_event(payload, sport, season, week, captured_at)
        if not rows:
            print(f"  [{n}/{len(keep)}] {label}: 0 rows "
                  f"(props may not be posted yet)")
            continue
        w = insert_rows(client, rows)
        total += w
        books = len({r["book"] for r in rows})
        wk = f"wk{week}" if week is not None else "wk?"
        print(f"  [{n}/{len(keep)}] {label} ({wk}): {w} rows, {books} books")

    print("\n" + "=" * 70)
    print(f"  wrote {total} rows at {captured_at}")
    print(f"  {api.line()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
