"""
PHASE 0: validate The Odds API before spending anything.

Costs roughly 20 to 30 credits against the free 500. The point is to answer
four questions BEFORE paying for the historical backfill:

  1. Do FanDuel and DraftKings both appear as bookmaker keys for NFL props?
  2. Do all five NFL market keys return data?
       player_pass_yds, player_rush_yds, player_reception_yds,
       player_receptions, player_anytime_td
     (five keys cover all six app markets, because player_rush_yds contains
     both RB and QB rushing)
  3. Can the returned player names be matched to nflverse via
     data_utils.norm_join_name? An unmatched rate above about 5 percent is a
     real problem, because every downstream join depends on it.
  4. Does NHL player_shots_on_goal return data?

GATE: if FanDuel props are missing, or the unmatched name rate is high, stop
and reassess rather than buying the backfill.

Credit costs, from the docs:
    /sports and /events           free
    /events/{id}/odds             markets x regions
    /historical/.../odds          markets x regions x 10

Run from the repo root with the venv active:
    python odds_api_check.py

Set ODDS_API_KEY in the environment, or put ODDS_API_KEY in
.streamlit/secrets.toml, or paste it at the prompt.
"""
import os
import sys
import json
from getpass import getpass

import pandas as pd
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

# NHL keys seen in the wild. Coverage changes, so probe rather than assume.
NHL_MARKETS_TO_PROBE = [
    "player_shots_on_goal",
    "player_points",
    "player_assists",
    "player_power_play_points",
    "player_blocked_shots",
    "player_goals",
]

WANT_BOOKS = ("fanduel", "draftkings")


# ------------------------------------------------------------------- plumbing

def get_key():
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if key:
        print("  API key from environment")
        return key
    if os.path.exists(SECRETS_PATH):
        try:
            import tomllib
            with open(SECRETS_PATH, "rb") as fh:
                sec = tomllib.load(fh)
            key = str(sec.get("ODDS_API_KEY", "")).strip()
            if key:
                print(f"  API key from {SECRETS_PATH}")
                return key
        except Exception as e:
            print(f"  could not parse {SECRETS_PATH}: {e}")
    key = getpass("ODDS_API_KEY (hidden): ").strip()
    if not key:
        print("  no key supplied")
        sys.exit(1)
    return key


class Api:
    """Thin wrapper that tracks credit usage from the response headers."""

    def __init__(self, key):
        self.key = key
        self.used = None
        self.remaining = None
        self.last_cost = None
        self.calls = 0

    def get(self, path, **params):
        params["apiKey"] = self.key
        url = f"{BASE}{path}"
        r = requests.get(url, params=params, timeout=30)
        self.calls += 1
        self.used = r.headers.get("x-requests-used", self.used)
        self.remaining = r.headers.get("x-requests-remaining", self.remaining)
        self.last_cost = r.headers.get("x-requests-last", "?")
        if r.status_code != 200:
            body = r.text[:400]
            print(f"  HTTP {r.status_code} on {path}")
            print(f"     {body}")
            return None
        return r.json()

    def report(self):
        print(f"\n  credits: used {self.used}, remaining {self.remaining}, "
              f"last call cost {self.last_cost}, {self.calls} calls made")


def _outcomes(book_block, market_key):
    for m in book_block.get("markets", []):
        if m.get("key") == market_key:
            return m.get("outcomes", []) or []
    return []


# -------------------------------------------------------------------- checks

def check_sports(api):
    print("\n" + "=" * 74)
    print("0. IS THE KEY LIVE, AND WHICH SPORTS ARE IN SEASON?")
    print("=" * 74)
    data = api.get("/sports")
    if data is None:
        print("  could not list sports, so the key is probably wrong")
        sys.exit(1)
    keys = {s["key"]: s for s in data}
    for want in (NFL, NHL):
        s = keys.get(want)
        if s is None:
            print(f"  {want}: NOT FOUND in the sports list")
        else:
            print(f"  {want}: present, active={s.get('active')}")
    print(f"  ({len(keys)} sports total; this endpoint is free)")


def pick_event(api, sport):
    data = api.get(f"/sports/{sport}/events")
    if not data:
        return None
    ev = sorted(data, key=lambda e: e.get("commence_time", ""))[0]
    print(f"  {len(data)} upcoming events; using "
          f"{ev.get('away_team')} at {ev.get('home_team')} "
          f"({ev.get('commence_time')})")
    return ev


def check_nfl(api):
    print("\n" + "=" * 74)
    print("1. NFL PROPS: BOOKS AND MARKETS")
    print("=" * 74)
    ev = pick_event(api, NFL)
    if ev is None:
        print("  no upcoming NFL events returned")
        return None

    data = api.get(f"/sports/{NFL}/events/{ev['id']}/odds",
                   regions="us", oddsFormat="american",
                   markets=",".join(NFL_MARKETS))
    if data is None:
        return None
    print(f"  (this call cost {api.last_cost} credits)")

    books = data.get("bookmakers", []) or []
    print(f"\n  {len(books)} bookmakers returned: "
          f"{', '.join(sorted(b['key'] for b in books))}")

    for want in WANT_BOOKS:
        present = any(b["key"] == want for b in books)
        flag = "OK" if present else ">>> MISSING"
        print(f"    {want:<12}{flag}")

    print(f"\n  {'market':<24}{'books':>7}{'rows (fanduel)':>16}"
          f"{'sample':>34}")
    coverage = {}
    for mk in NFL_MARKETS:
        n_books = sum(1 for b in books if _outcomes(b, mk))
        fd = next((b for b in books if b["key"] == "fanduel"), None)
        fd_rows = _outcomes(fd, mk) if fd else []
        sample = ""
        if fd_rows:
            o = fd_rows[0]
            sample = (f"{o.get('description','?')} {o.get('name','')} "
                      f"{o.get('point','NA')} @ {o.get('price','?')}")
        coverage[mk] = (n_books, len(fd_rows))
        print(f"  {mk:<24}{n_books:>7}{len(fd_rows):>16}   {sample[:32]}")

    missing = [mk for mk, (nb, _n) in coverage.items() if nb == 0]
    if missing:
        print(f"\n  >>> NO BOOK RETURNED: {missing}")
        print("  >>> Market keys may have changed. Check the betting-markets")
        print("  >>> docs page before assuming the data is unavailable.")
    return data


def check_names(data):
    print("\n" + "=" * 74)
    print("2. DO THE PLAYER NAMES MATCH NFLVERSE?")
    print("=" * 74)
    if data is None:
        print("  skipped, no NFL odds payload")
        return

    api_names = set()
    for b in data.get("bookmakers", []) or []:
        for m in b.get("markets", []) or []:
            for o in m.get("outcomes", []) or []:
                d = o.get("description")
                if d:
                    api_names.add(d)
    print(f"  {len(api_names)} distinct player names in the payload")
    if not api_names:
        return

    try:
        from models import data_utils
    except Exception as e:
        print(f"  could not import data_utils: {e}")
        return

    try:
        ps = data_utils.load_player_stats([2025, 2026])
        ps = ps.to_pandas() if hasattr(ps, "to_pandas") else ps
    except Exception as e:
        print(f"  could not load player stats: {e}")
        return

    nfl_names = ps["player_display_name"].dropna().drop_duplicates()
    nfl_norm = set(data_utils.norm_join_name(nfl_names))
    api_norm = data_utils.norm_join_name(pd.Series(sorted(api_names)))

    matched = [n for n in api_norm if n in nfl_norm]
    unmatched_idx = [i for i, n in enumerate(api_norm) if n not in nfl_norm]
    src = sorted(api_names)
    rate = 100.0 * len(unmatched_idx) / len(api_norm)

    print(f"  matched {len(matched)} of {len(api_norm)} "
          f"({100 - rate:.1f}%), unmatched {len(unmatched_idx)} ({rate:.1f}%)")
    if unmatched_idx:
        print("  unmatched examples:")
        for i in unmatched_idx[:12]:
            print(f"    {src[i]}  ->  {api_norm.iloc[i]}")
    if rate > 5:
        print("\n  >>> GATE: unmatched rate above 5 percent. Inspect the list")
        print("  >>> above. Defenses, kickers and retired players are benign;")
        print("  >>> real skill players failing to match is not.")
    else:
        print("\n  name matching looks fine for a join.")


def check_nhl(api):
    print("\n" + "=" * 74)
    print("3. NHL: WHICH PROP MARKETS EXIST?")
    print("=" * 74)
    print("  Probing one market at a time so a single bad key does not void")
    print("  the whole call. Each probe costs 1 credit.\n")

    ev = pick_event(api, NHL)
    if ev is None:
        print("  No upcoming NHL events. The season may not have opened yet.")
        print("  Re-run this section once games are listed, or probe a")
        print("  historical event on a paid plan.")
        return

    print(f"  {'market':<28}{'books':>7}{'rows':>7}{'sample':>34}")
    for mk in NHL_MARKETS_TO_PROBE:
        data = api.get(f"/sports/{NHL}/events/{ev['id']}/odds",
                       regions="us", oddsFormat="american", markets=mk)
        if data is None:
            print(f"  {mk:<28}{'err':>7}")
            continue
        books = data.get("bookmakers", []) or []
        n_books = sum(1 for b in books if _outcomes(b, mk))
        fd = next((b for b in books if b["key"] == "fanduel"), None)
        rows = _outcomes(fd, mk) if fd else []
        sample = ""
        if rows:
            o = rows[0]
            sample = (f"{o.get('description','?')} {o.get('name','')} "
                      f"{o.get('point','NA')} @ {o.get('price','?')}")
        star = "  <<< the one we want" if mk == "player_shots_on_goal" and n_books else ""
        print(f"  {mk:<28}{n_books:>7}{len(rows):>7}   {sample[:30]}{star}")


def main():
    print("=" * 74)
    print("PHASE 0: ODDS API COVERAGE CHECK")
    print("=" * 74)
    key = get_key()
    api = Api(key)

    check_sports(api)
    nfl_data = check_nfl(api)
    check_names(nfl_data)
    check_nhl(api)
    api.report()

    # keep a copy of the payload so the backfill parser can be written and
    # tested offline without spending more credits
    if nfl_data is not None:
        out = "odds_api_sample.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(nfl_data, fh, indent=2)
        print(f"  saved a sample payload to {out} so the backfill parser can")
        print("  be written and tested offline without spending credits")

    print("\n" + "=" * 74)
    print("GATE")
    print("=" * 74)
    print("  PROCEED to Phase 1 if: fanduel and draftkings both appear, all")
    print("  five NFL markets return rows, the unmatched name rate is under")
    print("  about 5 percent, and player_shots_on_goal returns rows.")
    print("  RECONSIDER if FanDuel props are missing or names do not join,")
    print("  since every downstream join depends on that.")


if __name__ == "__main__":
    main()
