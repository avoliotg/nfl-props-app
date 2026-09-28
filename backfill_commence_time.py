"""Backfill commence_time on the automated-era `lines` rows. Plan item 1.5.

    python backfill_commence_time.py            gate and report, writes nothing
    python backfill_commence_time.py --apply    write

WHY THIS IS NEEDED. live_capture.py now stores commence_time on every new
row, but the 135k rows already captured have it null, so there is no way to
tell how long before kickoff any of them was priced. captured_at cannot
answer that: one snapshot spans games that kick off hours apart, so a single
Sunday afternoon capture holds finished 1pm players, live 4:05 players and
pregame Sunday night players at once.

THE ROUTE, in two tiers.

  tier 1  (join_name, week) -> game_id, straight from player_stats. Exact,
          because the stats row IS the game. Only works for weeks played.
  tier 2  join_name -> team from the latest week the player appears in,
          then (team, week) -> game_id from the schedule. Works for weeks
          not yet played, which is how week 4 rows resolve.

THE GATE, and this is the point of the script. lines_cache.parquet carries
commence_time on 100 percent of its 184,782 rows, including 5,799 from 2026.
So the route can be run against rows whose correct answer is already known
and required to reproduce it. That tests the whole chain end to end,
including the Eastern-to-UTC conversion, against a source this script had no
hand in producing.

The cache is also the HARDER population: its names are abbreviated
screenshot-era forms (J. Hill, M. Rudolph) while the database holds full
names from The Odds API. A route that clears the gate on initials is safe on
full names.

WHAT IT WILL NOT DO. Rows whose player never resolves keep a null
commence_time. Null means unknown, which is honest and filterable. Guessing
a kickoff would be worse than leaving the gap, because a wrong kickoff turns
into a wrong distance-to-kickoff and then into a silently wrong closing
population.

Rows that already have commence_time are never touched, so this is safe to
rerun and safe to interrupt.
"""
import os
import sys

import pandas as pd

from models import data_utils as du

import live_capture as lc

CACHE = "lines_cache.parquet"
AUTO_ERA_START = "2026-09-22T00:00:00+00:00"
SEASON = 2026
GATE_MIN_AGREE = 0.99
CHUNK = 400


def frame(x):
    return x.to_pandas() if hasattr(x, "to_pandas") else x


def hr(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


# --------------------------------------------------------------- build the map

hr("A. BUILD THE ROUTE")

sched = frame(du.load_schedules([SEASON]))
sched = sched[sched["season"] == SEASON].copy()
kick = pd.to_datetime(sched["gameday"].astype(str) + " "
                      + sched["gametime"].astype(str), errors="coerce")
sched["kickoff_utc"] = (kick.dt.tz_localize("America/New_York",
                                            nonexistent="shift_forward",
                                            ambiguous=True)
                            .dt.tz_convert("UTC"))
print("  %d scheduled %d games, %d with a parsed kickoff"
      % (len(sched), SEASON, int(sched["kickoff_utc"].notna().sum())))

stats = frame(du.load_player_stats([SEASON]))
stats = stats[(stats["season"] == SEASON)
              & stats["player_display_name"].notna()].copy()
stats["join_name"] = du.norm_join_name(stats["player_display_name"])

# tier 1: exact per-week game
t1 = (stats[["join_name", "week", "game_id"]]
      .drop_duplicates(subset=["join_name", "week"]))
t1 = t1.merge(sched[["game_id", "kickoff_utc"]], on="game_id", how="left")
t1 = t1.rename(columns={"kickoff_utc": "kick_t1", "game_id": "game_t1"})
print("  tier 1: %d (player, week) pairs from played weeks %s"
      % (len(t1), sorted(stats["week"].unique().tolist())))

# tier 2: latest team, then team-week from the schedule
tmap = (stats.sort_values("week")
             .drop_duplicates(subset=["join_name"], keep="last")
             [["join_name", "team"]])
tw = pd.concat([
    sched[["game_id", "week", "kickoff_utc", "away_team"]]
        .rename(columns={"away_team": "team"}),
    sched[["game_id", "week", "kickoff_utc", "home_team"]]
        .rename(columns={"home_team": "team"}),
], ignore_index=True).rename(columns={"kickoff_utc": "kick_t2",
                                      "game_id": "game_t2"})
print("  tier 2: %d players mapped to a team, %d team-week slots"
      % (len(tmap), len(tw)))


def resolve(df, player_col="player", week_col="week"):
    """Attach kickoff_utc to df, tier 1 first then tier 2."""
    d = df.copy()
    d["join_name"] = du.norm_join_name(d[player_col])
    d["_wk"] = pd.to_numeric(d[week_col], errors="coerce").astype("Int64")
    d = d.merge(t1, left_on=["join_name", "_wk"],
                right_on=["join_name", "week"], how="left",
                suffixes=("", "_t1"))
    d = d.merge(tmap, on="join_name", how="left")
    d = d.merge(tw, left_on=["team", "_wk"], right_on=["team", "week"],
                how="left", suffixes=("", "_t2"))
    d["kickoff_utc"] = d["kick_t1"].fillna(d["kick_t2"])
    d["tier"] = "none"
    d.loc[d["kick_t2"].notna(), "tier"] = "2"
    d.loc[d["kick_t1"].notna(), "tier"] = "1"
    return d


# ------------------------------------------------------------------- the gate

hr("B. GATE AGAINST KNOWN KICKOFFS IN THE CACHE")

if not os.path.exists(CACHE):
    print("  %s not found. The gate cannot run, so neither does the" % CACHE)
    print("  backfill. Nothing written.")
    sys.exit(1)

C = pd.read_parquet(CACHE)
G = C[(C["season"] == SEASON) & C["commence_time"].notna()].copy()
print("  %d cache rows from %d carry a known commence_time" % (len(G), SEASON))
if not len(G):
    print("  nothing to gate against. Nothing written.")
    sys.exit(1)

G["known"] = pd.to_datetime(G["commence_time"], utc=True, errors="coerce")
R = resolve(G)
have = R["kickoff_utc"].notna()
print("  route resolved %d of %d (%.2f%%)"
      % (int(have.sum()), len(R), 100 * have.mean()))
print("    tier 1 (exact game): %d" % int((R["tier"] == "1").sum()))
print("    tier 2 (team-week):  %d" % int((R["tier"] == "2").sum()))

cmp = R[have]
agree = (cmp["kickoff_utc"] == cmp["known"])
rate = float(agree.mean()) if len(cmp) else 0.0
print()
print("  of the resolved rows, %d of %d match the known kickoff EXACTLY "
      "(%.4f)" % (int(agree.sum()), len(cmp), rate))

if len(cmp) and not agree.all():
    bad = cmp[~agree]
    print()
    print("  disagreements, by size of error:")
    err = ((bad["kickoff_utc"] - bad["known"]).dt.total_seconds() / 3600.0)
    for h, n in err.round(2).value_counts().head(10).items():
        print("    %+7.2f hours   %5d rows" % (h, n))
    print()
    print("  A cluster at a whole number of hours is a timezone error. A")
    print("  scatter of odd values is a wrong-game match.")
    for _, r in bad.head(8).iterrows():
        print("    %-22s wk %-3s  route %s  known %s"
              % (str(r["player"])[:22], r["_wk"],
                 r["kickoff_utc"], r["known"]))

if rate < GATE_MIN_AGREE:
    print()
    print("  GATE FAILED. Agreement %.4f is below %.2f. The route does not"
          % (rate, GATE_MIN_AGREE))
    print("  reproduce kickoffs that are already known, so it must not be")
    print("  used to write kickoffs that are not. Nothing written.")
    sys.exit(1)
print("  GATE PASSED")


# ------------------------------------------------------------ pull target rows

hr("C. ROWS NEEDING A BACKFILL")

sec = lc._secrets()
client = lc.supa_client(sec)

rows, page = [], 0
while True:
    res = (client.table("lines")
           .select("id,season,week,market,player,captured_at,commence_time")
           .gte("captured_at", AUTO_ERA_START)
           .is_("commence_time", "null")
           .order("id")
           .range(page * 1000, page * 1000 + 999).execute())
    b = res.data or []
    rows.extend(b)
    if len(b) < 1000:
        break
    page += 1
    if page > 400:
        print("  stopped at 400 pages")
        break

L = pd.DataFrame(rows)
print("  %d rows with a null commence_time" % len(L))
if L.empty:
    print("  nothing to do")
    sys.exit(0)

T = resolve(L)
ok = T["kickoff_utc"].notna()
print("  resolvable   %6d (%.1f%%)" % (int(ok.sum()), 100 * ok.mean()))
print("    tier 1     %6d" % int((T["tier"] == "1").sum()))
print("    tier 2     %6d" % int((T["tier"] == "2").sum()))
print("  left null    %6d (%.1f%%)" % (int((~ok).sum()), 100 * (~ok).mean()))

print()
print("  by week:")
for wk, grp in T.groupby("_wk"):
    print("    week %-3s %6d rows, %6d resolvable"
          % (wk, len(grp), int(grp["kickoff_utc"].notna().sum())))

print()
print("  distinct kickoffs to be written: %d"
      % T.loc[ok, "kickoff_utc"].nunique())

if (~ok).any():
    print()
    print("  top unresolved players, left null on purpose:")
    for p, n in T.loc[~ok, "player"].value_counts().head(12).items():
        print("    %-30s %5d rows" % (str(p)[:30], n))

if "--apply" not in sys.argv:
    print()
    print("Dry run. Nothing written. Rerun with --apply.")
    sys.exit(0)


# ----------------------------------------------------------------- write it

hr("D. WRITING")

W = T[ok]
groups = list(W.groupby("kickoff_utc"))
print("  %d rows across %d kickoff groups" % (len(W), len(groups)))
print()

written, failed = 0, 0
for gi, (kt, grp) in enumerate(groups, 1):
    iso = pd.Timestamp(kt).isoformat()
    ids = [int(x) for x in grp["id"].tolist()]
    for i in range(0, len(ids), CHUNK):
        part = ids[i:i + CHUNK]
        try:
            (client.table("lines")
             .update({"commence_time": iso})
             .in_("id", part).execute())
            written += len(part)
        except Exception as e:
            failed += len(part)
            print("    chunk failed (%d rows): %s" % (len(part),
                                                      type(e).__name__))
    print("  [%d/%d] %s  %d rows  (running total %d)"
          % (gi, len(groups), iso, len(ids), written))

print()
print("  wrote %d rows, %d failed" % (written, failed))

# verify by reading back
res = (client.table("lines")
       .select("id", count="exact")
       .gte("captured_at", AUTO_ERA_START)
       .is_("commence_time", "null").execute())
left = res.count if hasattr(res, "count") else None
print("  rows still null in the automated era: %s" % left)
print("  expected: %d (the deliberately unresolved ones)" % int((~ok).sum()))
print()
