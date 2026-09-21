"""
PHASE 1: historical prop backfill from The Odds API.

This is the script that spends money, so it is built defensively.

  --dry-run   Enumerate every game using only the FREE events endpoint and
              print the exact credit cost. Costs about 1 credit per week
              queried. ALWAYS run this first.
  --run       Actually fetch odds and write to Supabase. Resumable, so an
              interrupted run picks up where it stopped.

Credit costs, from the docs:
    /v4/historical/sports/{sport}/events                    1 credit
    /v4/historical/sports/{sport}/events/{id}/odds          10 x markets x regions

With five NFL markets and one region that is 50 credits per event per snapshot,
so a restart without resumability is expensive. Completed work is recorded in a
local ledger file AND checked against Supabase, so the ledger being deleted
does not cause a re-spend.

`regions=us` returns every US book in one call, so all five books come at no
extra credit cost.

Snapshot labels:
    opening   the earliest snapshot available in the week the game appeared
    closing   40 minutes before commence_time

Usage:
    python odds_backfill.py --dry-run --season 2025
    python odds_backfill.py --run --season 2025 --snapshots closing
    python odds_backfill.py --run --season 2025 --snapshots opening,closing
    python odds_backfill.py --run --season 2025 --ceiling 30000

Set ODDS_API_KEY in the environment or .streamlit/secrets.toml.
Supabase credentials come from .streamlit/secrets.toml as usual.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from getpass import getpass

import requests

BASE = "https://api.the-odds-api.com/v4"
SECRETS_PATH = os.path.join(".streamlit", "secrets.toml")
LEDGER_PATH = "backfill_ledger.json"

NFL = "americanfootball_nfl"
NFL_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
]

NHL = "icehockey_nhl"
NHL_MARKETS = ["player_shots_on_goal"]

# Prop history begins here. Earlier dates return nothing for prop markets.
PROPS_AVAILABLE_FROM = datetime(2023, 5, 3, 6, 0, tzinfo=timezone.utc)

CLOSING_OFFSET_MIN = 40
TABLE = "historical_lines"

# Map the API's market keys onto the app's internal market names. anytime_td
# and the RB/QB rushing split are handled at parse time, not here.
MARKET_TO_APP = {
    "player_pass_yds": "qb_passing",
    "player_rush_yds": "rushing",
    "player_reception_yds": "receiving",
    "player_receptions": "receptions",
    "player_anytime_td": "anytime_td",
    "player_shots_on_goal": "shots_on_goal",
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
        print(f"  could not parse {SECRETS_PATH}: {e}")
        return {}


def get_api_key(sec):
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        return key
    key = str(sec.get("ODDS_API_KEY", "")).strip()
    if key:
        return key
    return getpass("ODDS_API_KEY (hidden): ").strip()


class Api:
    """Tracks credit usage and enforces a hard ceiling."""

    def __init__(self, key, ceiling=None):
        self.key = key
        self.ceiling = ceiling
        self.used = 0
        self.remaining = None
        self.last_cost = 0
        self.calls = 0
        self.spent_here = 0

    def get(self, path, **params):
        if self.ceiling is not None and self.spent_here >= self.ceiling:
            raise RuntimeError(
                f"credit ceiling reached: {self.spent_here} spent this run, "
                f"ceiling {self.ceiling}. Re-run to continue; progress is saved.")
        params["apiKey"] = self.key
        for attempt in range(4):
            try:
                r = requests.get(f"{BASE}{path}", params=params, timeout=45)
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"      network error ({e}), retrying in {wait}s")
                time.sleep(wait)
                continue
            self.calls += 1
            try:
                self.last_cost = int(r.headers.get("x-requests-last", 0))
            except ValueError:
                self.last_cost = 0
            self.spent_here += self.last_cost
            hdr_used = r.headers.get("x-requests-used")
            if hdr_used is not None:
                try:
                    self.used = int(hdr_used)
                except ValueError:
                    pass
            self.remaining = r.headers.get("x-requests-remaining", self.remaining)

            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"      rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            # 4xx other than 429 will not fix itself
            print(f"      HTTP {r.status_code}: {r.text[:200]}")
            return None
        print("      giving up on this call after retries")
        return None

    def line(self):
        return (f"credits used {self.used}, remaining {self.remaining}, "
                f"this run {self.spent_here}, calls {self.calls}")


# --------------------------------------------------------------------- ledger

def load_ledger():
    if not os.path.exists(LEDGER_PATH):
        return {"done": {}}
    try:
        with open(LEDGER_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        print(f"  ledger at {LEDGER_PATH} unreadable, starting a new one")
        return {"done": {}}


def save_ledger(led):
    tmp = LEDGER_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(led, fh, indent=1)
    os.replace(tmp, LEDGER_PATH)


# -------------------------------------------------------------- season layout

def season_snapshot_dates(season, sport=NFL):
    """Weekly timestamps at which to ask 'which games existed then'.

    For the NFL a season labelled 2025 runs from early September 2025 into
    mid-February 2026. Wednesday is used because books have posted the week's
    props by then and it predates most movement, which makes it a reasonable
    'opening' timestamp.
    """
    if sport == NFL:
        start = datetime(season, 9, 1, 18, 0, tzinfo=timezone.utc)
        weeks = 24  # regular season plus playoffs through the Super Bowl
    else:
        start = datetime(season, 10, 1, 18, 0, tzinfo=timezone.utc)
        weeks = 30
    out = []
    for i in range(weeks):
        d = start + timedelta(days=7 * i)
        if d < PROPS_AVAILABLE_FROM:
            continue
        if d > datetime.now(timezone.utc):
            break
        out.append(d)
    return out


def enumerate_events(api, sport, season):
    """Every event in the season, with the snapshot date it first appeared.

    Costs 1 credit per weekly call. The first appearance date doubles as the
    'opening' timestamp for that game.
    """
    events = {}
    for d in season_snapshot_dates(season, sport):
        iso = d.strftime("%Y-%m-%dT%H:%M:%SZ")
        data = api.get(f"/historical/sports/{sport}/events", date=iso)
        if not data:
            continue
        rows = data.get("data", data) or []
        new = 0
        for e in rows:
            eid = e.get("id")
            if not eid or eid in events:
                continue
            events[eid] = {
                "id": eid,
                "commence_time": e.get("commence_time"),
                "home_team": e.get("home_team"),
                "away_team": e.get("away_team"),
                "opening_date": data.get("timestamp") or iso,
            }
            new += 1
        print(f"    {iso}: {len(rows):>3} events, {new:>3} new "
              f"(running total {len(events)})")
    return events


def targets_for(ev, labels):
    """(label, iso_timestamp) pairs to fetch for one event."""
    out = []
    ct = ev.get("commence_time")
    if "opening" in labels and ev.get("opening_date"):
        out.append(("opening", ev["opening_date"]))
    if "closing" in labels and ct:
        try:
            t = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except ValueError:
            return out
        close = t - timedelta(minutes=CLOSING_OFFSET_MIN)
        out.append(("closing", close.strftime("%Y-%m-%dT%H:%M:%SZ")))
    return out


# ---------------------------------------------------------------------- parse

def parse_snapshot(payload, sport, season, label, ev):
    """Flatten one historical event-odds payload into database rows."""
    rows = []
    if not payload:
        return rows
    body = payload.get("data") or {}
    snap_ts = payload.get("timestamp")
    commence = body.get("commence_time") or ev.get("commence_time")
    event_id = body.get("id") or ev.get("id")

    for bk in body.get("bookmakers", []) or []:
        book = bk.get("key")
        for mkt in bk.get("markets", []) or []:
            app_market = MARKET_TO_APP.get(mkt.get("key"))
            if app_market is None:
                continue
            # collect over and under for the same player into one row
            per_player = {}
            for o in mkt.get("outcomes", []) or []:
                player = o.get("description")
                if not player:
                    continue
                side = str(o.get("name", "")).lower()
                rec = per_player.setdefault(player, {
                    "line": o.get("point"),
                    "over_odds": None,
                    "under_odds": None,
                })
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
                    "market": app_market,
                    "player": player,
                    "line": rec["line"],
                    "over_odds": rec["over_odds"],
                    "under_odds": rec["under_odds"],
                    "captured_at": snap_ts,
                    "event_id": event_id,
                    "commence_time": commence,
                    "home_team": body.get("home_team") or ev.get("home_team"),
                    "away_team": body.get("away_team") or ev.get("away_team"),
                    "snapshot_label": label,
                })
    return rows


# -------------------------------------------------------------------- storage

def supa_client(sec):
    from supabase import create_client
    url = sec.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = sec.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    if not (url and key):
        print("  missing Supabase credentials")
        sys.exit(1)
    client = create_client(url, key)
    email = os.environ.get("OPAL_EMAIL") or sec.get("ADMIN_EMAIL") or ""
    if not email:
        email = input("Supabase email: ").strip()
    pw = os.environ.get("OPAL_PASSWORD") or getpass("Supabase password (hidden): ")
    res = client.auth.sign_in_with_password({"email": email, "password": pw})
    if not res.user:
        print("  Supabase login failed")
        sys.exit(1)
    print(f"  Supabase: logged in as {res.user.email}")
    return client


def insert_rows(client, rows, chunk=400):
    """Insert in chunks. Returns the number written."""
    written = 0
    for i in range(0, len(rows), chunk):
        batch = rows[i:i + chunk]
        try:
            client.table(TABLE).insert(batch).execute()
            written += len(batch)
        except Exception as e:
            print(f"      insert failed for a batch of {len(batch)}: {e}")
    return written


def already_in_db(client, event_ids, label):
    """Event ids already present for this label, so a lost ledger does not
    cause a re-spend."""
    have = set()
    ids = list(event_ids)
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        try:
            resp = (client.table(TABLE).select("event_id")
                    .in_("event_id", chunk)
                    .eq("snapshot_label", label).execute())
            for r in resp.data or []:
                have.add(r["event_id"])
        except Exception as e:
            print(f"  could not check existing rows: {e}")
            return have
    return have


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate games and price the job, no odds calls")
    ap.add_argument("--run", action="store_true", help="fetch and store")
    ap.add_argument("--sport", default="nfl", choices=["nfl", "nhl"])
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--snapshots", default="closing",
                    help="comma separated: closing, opening, or both")
    ap.add_argument("--ceiling", type=int, default=None,
                    help="stop after spending this many credits this run")
    ap.add_argument("--limit-events", type=int, default=None,
                    help="only process the first N events, for a cheap test")
    args = ap.parse_args()

    if not (args.dry_run or args.run):
        print("pass --dry-run or --run")
        sys.exit(1)

    sport = NFL if args.sport == "nfl" else NHL
    markets = NFL_MARKETS if sport == NFL else NHL_MARKETS
    labels = [s.strip() for s in args.snapshots.split(",") if s.strip()]
    cost_per_call = 10 * len(markets) * 1

    sec = _secrets()
    api = Api(get_api_key(sec), ceiling=args.ceiling)

    print("=" * 74)
    print(f"BACKFILL  sport={args.sport}  season={args.season}  "
          f"snapshots={labels}")
    print(f"  {len(markets)} markets, 1 region, so {cost_per_call} credits "
          f"per event per snapshot")
    print("=" * 74)

    print("\nenumerating events (1 credit per weekly call)")
    events = enumerate_events(api, sport, args.season)
    if not events:
        print("  no events found. Check the season year and that prop history")
        print("  covers this period (props start 2023-05-03).")
        return

    if args.limit_events:
        keys = sorted(events, key=lambda k: events[k]["commence_time"] or "")
        events = {k: events[k] for k in keys[:args.limit_events]}
        print(f"  limited to the first {len(events)} events")

    total_calls = sum(len(targets_for(ev, labels)) for ev in events.values())
    print(f"\n  {len(events)} events, {total_calls} snapshot calls")
    print(f"  estimated cost: {total_calls} x {cost_per_call} = "
          f"{total_calls * cost_per_call} credits")
    print(f"  plus {api.spent_here} already spent enumerating")
    print(f"  {api.line()}")

    if args.dry_run:
        print("\n  dry run only, no odds calls made.")
        print("  Re-run with --run when the cost looks acceptable.")
        return

    client = supa_client(sec)
    led = load_ledger()
    done = led["done"]

    # reconcile with the database so a deleted ledger does not re-spend
    for label in labels:
        in_db = already_in_db(client, events.keys(), label)
        if in_db:
            print(f"  {len(in_db)} events already stored for '{label}', skipping")
        for eid in in_db:
            done[f"{eid}:{label}"] = "db"
    save_ledger(led)

    todo = []
    for eid, ev in events.items():
        for label, iso in targets_for(ev, labels):
            if done.get(f"{eid}:{label}"):
                continue
            todo.append((eid, ev, label, iso))
    print(f"\n  {len(todo)} calls remaining after skipping completed work")
    print(f"  that is {len(todo) * cost_per_call} credits\n")

    written_total = 0
    empty = 0
    for n, (eid, ev, label, iso) in enumerate(todo, 1):
        try:
            payload = api.get(
                f"/historical/sports/{sport}/events/{eid}/odds",
                date=iso, regions="us", oddsFormat="american",
                markets=",".join(markets))
        except RuntimeError as e:
            print(f"\n  {e}")
            break

        rows = parse_snapshot(payload, sport, args.season, label, ev)
        if not rows:
            # an empty response is not charged, but it must be visible rather
            # than silently skipped
            empty += 1
            print(f"  [{n}/{len(todo)}] {label} {ev.get('away_team')} at "
                  f"{ev.get('home_team')}: NO ROWS (cost {api.last_cost})")
            done[f"{eid}:{label}"] = "empty"
            if n % 10 == 0:
                save_ledger(led)
            continue

        w = insert_rows(client, rows)
        written_total += w
        done[f"{eid}:{label}"] = "ok"
        books = len({r["book"] for r in rows})
        print(f"  [{n}/{len(todo)}] {label} {ev.get('away_team')} at "
              f"{ev.get('home_team')}: {w} rows, {books} books "
              f"(cost {api.last_cost}, run total {api.spent_here})")
        if n % 10 == 0:
            save_ledger(led)

    save_ledger(led)
    print("\n" + "=" * 74)
    print(f"  wrote {written_total} rows")
    print(f"  {empty} snapshots returned nothing")
    print(f"  {api.line()}")
    print(f"  ledger: {LEDGER_PATH} (delete only if you want to re-check the DB)")
    print("=" * 74)


if __name__ == "__main__":
    main()
