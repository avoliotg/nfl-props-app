"""Quantify the in-game capture contamination. READ ONLY, writes nothing.

This is plan item 1.1. It answers, for every row the automated capture has
written:

    how long before kickoff was this row priced

which `lines` cannot currently answer, because captured_at is one timestamp
for a whole run and a single run spans games that kick off hours apart.

ROUTE. lines has no team and no game id, so kickoff comes via:

    lines.player  ->  player_stats.player_display_name  ->  game_id
                  ->  schedules.gameday + gametime      ->  kickoff UTC

THE TIMEZONE GATE. gameday and gametime are strings in EASTERN. captured_at
is UTC. A naive comparison is off by four or five hours depending on DST,
which is enough to either hide the Thursday contamination entirely or invent
contamination across every Sunday 1pm slate. So before computing anything,
the script reproduces a kickoff whose UTC instant is known from an
independent source: The Odds API reported PHI at CHI as 2026-09-29T00:15:00Z,
and the schedule carries it as 2026-09-28 20:15 Eastern. If the conversion
does not return that instant, this aborts.

That gate is the point. Every number below is a subtraction of two
timestamps, so a timezone error would not look like an error. It would look
like a finding.

Run:  python diag_capture_timing.py
"""
import sys
from datetime import datetime, timezone

import pandas as pd

from models import data_utils as du

import live_capture as lc

# The automated GitHub Actions capture began here. Everything earlier is the
# hand-saved single-book era and a different instrument entirely.
AUTO_ERA_START = "2026-09-22T00:00:00+00:00"

# Independent ground truth for the timezone gate, from The Odds API.
GATE_GAME = "2026_03_PHI_CHI"
GATE_UTC = "2026-09-29T00:15:00+00:00"

# The historical closing-line instrument, per plan item 1.9.
WINDOW_LO, WINDOW_HI = 20, 90

# The rushing rule, for the exposure count.
RUSH_MAX_LINE = 46.5
RULE_BOOK = "fanduel"


def frame(x):
    return x.to_pandas() if hasattr(x, "to_pandas") else x


def hr(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


# ------------------------------------------------------- kickoffs, and the gate

hr("A. KICKOFFS, AND THE TIMEZONE GATE")

sched = frame(du.load_schedules([2026]))
sched = sched[sched["season"] == 2026].copy()

kick = pd.to_datetime(
    sched["gameday"].astype(str) + " " + sched["gametime"].astype(str),
    errors="coerce",
)
if kick.isna().any():
    print("  %d schedule rows have an unparseable gameday/gametime"
          % int(kick.isna().sum()))
sched["kickoff_utc"] = (kick.dt.tz_localize("America/New_York",
                                            nonexistent="shift_forward",
                                            ambiguous=True)
                            .dt.tz_convert("UTC"))

g = sched[sched["game_id"] == GATE_GAME]
if len(g) != 1:
    print("  GATE FAILED: %s not found in the 2026 schedule" % GATE_GAME)
    sys.exit(1)
got = g["kickoff_utc"].iloc[0]
want = pd.Timestamp(GATE_UTC)
print("  gate game     %s" % GATE_GAME)
print("  schedule says %s %s Eastern"
      % (g["gameday"].iloc[0], g["gametime"].iloc[0]))
print("  computed      %s" % got.isoformat())
print("  Odds API      %s" % want.isoformat())
if got != want:
    print()
    print("  GATE FAILED. The Eastern-to-UTC conversion does not reproduce a")
    print("  kickoff known from an independent source. Every number in this")
    print("  script is a difference of two timestamps, so this would not show")
    print("  up as an error further down. Aborting.")
    sys.exit(1)
print("  GATE PASSED")

sched_k = sched[["game_id", "week", "gameday", "gametime", "weekday",
                 "away_team", "home_team", "kickoff_utc"]]
print()
print("  week 3 kickoffs in UTC:")
w3k = sched_k[sched_k["week"] == 3].sort_values("kickoff_utc")
for _, r in w3k.iterrows():
    print("    %s  %-4s at %-4s  %s"
          % (r["kickoff_utc"].strftime("%Y-%m-%d %H:%M"), r["away_team"],
             r["home_team"], r["weekday"]))


# -------------------------------------------------- player to team to game

hr("B. PLAYER TO TEAM TO GAME")

# ROUTE CORRECTED 2026-09-28. The first version of this script built the map
# from WEEK 3 stats and joined on player name alone. That is week-blind, and
# `lines` contains week 4 rows, so a week 4 prop for a player who played in
# week 3 was matched to his WEEK 3 kickoff. That kickoff is in the past, so
# the row was counted as an in-game capture. The giveaway was a row reported
# as 844 minutes into a game, which is 14 hours, which is the distance back
# to the previous Sunday rather than anything real.
#
# The route now goes player -> team -> (team, week) -> game_id, which works
# for weeks that have not been played yet because the SCHEDULE carries them
# even when player stats do not.
#
# Team comes from the most recent week in which the player appears, so an
# in-season team change resolves to the current team rather than an old one.

stats = frame(du.load_player_stats([2026]))
stats = stats[(stats["season"] == 2026) & stats["player_display_name"].notna()]
stats = stats.sort_values("week")

tmap = (stats[["player_display_name", "team", "week"]]
        .drop_duplicates(subset=["player_display_name"], keep="last"))
tmap["join_name"] = du.norm_join_name(tmap["player_display_name"])
tmap = tmap.drop_duplicates(subset=["join_name"], keep="last")
tmap = tmap.rename(columns={"week": "last_seen_week"})
print("  %d players mapped to a team, from weeks %d to %d of 2026"
      % (len(tmap), int(stats["week"].min()), int(stats["week"].max())))

# One row per team per week, from the schedule. A team appears once as away
# and once as home, so this is a melt rather than a join.
tw = pd.concat([
    sched_k[["game_id", "week", "kickoff_utc", "away_team"]]
        .rename(columns={"away_team": "team"}),
    sched_k[["game_id", "week", "kickoff_utc", "home_team"]]
        .rename(columns={"home_team": "team"}),
], ignore_index=True)
print("  %d team-week slots across %d scheduled games"
      % (len(tw), sched_k["game_id"].nunique()))

# GATE. For week 3, which HAS been played, the team route must reproduce the
# same game_id that the direct stats game_id gives. If it does not, the team
# mapping is wrong and every distance below is wrong with it. This is the
# same discipline as the timezone gate: reproduce a known answer before
# producing a new one.
w3 = stats[stats["week"] == 3][["player_display_name", "team", "game_id"]].copy()
w3["join_name"] = du.norm_join_name(w3["player_display_name"])
chk = (w3.merge(tmap[["join_name", "team"]], on="join_name",
                how="inner", suffixes=("_stats", "_map"))
         .merge(tw[tw["week"] == 3][["team", "game_id"]]
                .rename(columns={"team": "team_map",
                                 "game_id": "game_id_route"}),
                on="team_map", how="left"))
agree = (chk["game_id"] == chk["game_id_route"])
print()
print("  GATE  week 3 team route vs direct stats game_id:")
print("        %d of %d agree (%.2f%%)"
      % (agree.sum(), len(chk), 100 * agree.mean()))
if len(chk) and agree.mean() < 0.98:
    print()
    print("        GATE FAILED. The team route does not reproduce week 3.")
    dis = chk[~agree].head(10)
    for _, r in dis.iterrows():
        print("          %-24s stats %s  route %s"
              % (str(r["player_display_name"])[:24], r["game_id"],
                 r["game_id_route"]))
    sys.exit(1)
print("        GATE PASSED")

# ------------------------------------------------------------- pull the rows

hr("C. LINES ROWS FROM THE AUTOMATED ERA")

sec = lc._secrets()
client = lc.supa_client(sec)

cols = "id,season,week,market,player,line,book,captured_at,commence_time"
rows, page, size = [], 0, 1000
while True:
    res = (client.table("lines").select(cols)
           .gte("captured_at", AUTO_ERA_START)
           .order("id")
           .range(page * size, page * size + size - 1).execute())
    batch = res.data or []
    rows.extend(batch)
    if len(batch) < size:
        break
    page += 1
    if page > 400:
        print("  stopped paginating at 400 pages, something is wrong")
        break

L = pd.DataFrame(rows)
print("  %d rows captured at or after %s" % (len(L), AUTO_ERA_START[:10]))
if L.empty:
    print("  nothing to analyse")
    sys.exit(0)

L["captured_at"] = pd.to_datetime(L["captured_at"], utc=True, format="mixed")
print("  %d distinct snapshots" % L["captured_at"].nunique())
print("  weeks present: %s" % sorted(L["week"].dropna().unique().tolist()))
print("  books present: %d" % L["book"].nunique())
print("  commence_time already populated on %d rows"
      % int(L["commence_time"].notna().sum()))


# ------------------------------------------------------------------- the join

hr("D. MATCH RATE")

# anytime_td returns a non-player outcome for the no-touchdown side. It is
# stored as a player and can never match. Excluded and reported, not counted
# as a failure.
NON_PLAYERS = {"no scorer", "no touchdown scorer", "none"}
L["join_name"] = du.norm_join_name(L["player"])
is_np = L["join_name"].str.lower().isin(NON_PLAYERS)
print("  %d rows are non-player outcomes (%s), excluded"
      % (int(is_np.sum()), ", ".join(sorted(NON_PLAYERS))))
L = L[~is_np].copy()

M = (L.merge(tmap[["join_name", "team", "last_seen_week"]],
             on="join_name", how="left")
      .merge(tw, on=["team", "week"], how="left"))

matched = M["kickoff_utc"].notna()
print("  matched   %6d rows (%.1f%%)" % (matched.sum(), 100 * matched.mean()))
print("  unmatched %6d rows (%.1f%%)" % ((~matched).sum(),
                                         100 * (~matched).mean()))

# Two failure modes, and only the first is a bug in our code.
#   no team   the name never resolved to a 2026 player at all
#   no game   the name resolved, but that team has no game in that week,
#             which is a bye week or a wrong week label on the row
if (~matched).any():
    U = M[~matched]
    no_team = U["team"].isna()
    print()
    print("  name never resolved to a 2026 player   %6d rows, %d names"
          % (int(no_team.sum()), U.loc[no_team, "player"].nunique()))
    print("  resolved, but no game in that week     %6d rows, %d names"
          % (int((~no_team).sum()), U.loc[~no_team, "player"].nunique()))
    print()
    print("  top unresolved names. A player who was priced but never recorded")
    print("  a 2026 stat line is unmatchable rather than wrong. A well known")
    print("  active player here IS a normalization failure and needs a look:")
    for name, n in U.loc[no_team, "player"].value_counts().head(20).items():
        print("    %-30s %5d rows" % (str(name)[:30], n))
    if (~no_team).any():
        print()
        print("  resolved but no game, by week and team:")
        z = (U[~no_team].groupby(["week", "team"]).size()
             .sort_values(ascending=False).head(15))
        for (wk, tm), n in z.items():
            print("    week %-3s %-4s %5d rows" % (wk, tm, n))

# --------------------------------------------------------- minutes to kickoff

hr("E. HOW LONG BEFORE KICKOFF WAS EACH ROW PRICED")

K = M[matched].copy()
K["mins"] = (K["kickoff_utc"] - K["captured_at"]).dt.total_seconds() / 60.0

post = K["mins"] < 0
inwin = (K["mins"] >= WINDOW_LO) & (K["mins"] <= WINDOW_HI)
late = (K["mins"] >= 0) & (K["mins"] < WINDOW_LO)
early = K["mins"] > WINDOW_HI

print("  IN GAME      %6d rows (%.2f%%)   captured after kickoff"
      % (post.sum(), 100 * post.mean()))
print("  too late     %6d rows (%.2f%%)   0 to %d min before"
      % (late.sum(), 100 * late.mean(), WINDOW_LO))
print("  IN WINDOW    %6d rows (%.2f%%)   %d to %d min before"
      % (inwin.sum(), 100 * inwin.mean(), WINDOW_LO, WINDOW_HI))
print("  too early    %6d rows (%.2f%%)   more than %d min before"
      % (early.sum(), 100 * early.mean(), WINDOW_HI))

if post.any():
    print()
    print("  in-game rows by snapshot:")
    s = (K[post].groupby("captured_at")
         .agg(rows=("id", "size"), players=("player", "nunique"),
              worst_mins=("mins", "min")).reset_index())
    for _, r in s.iterrows():
        et = r["captured_at"].tz_convert("America/New_York")
        print("    %s ET  %5d rows  %4d players  up to %.0f min into a game"
              % (et.strftime("%Y-%m-%d %a %H:%M"), r["rows"], r["players"],
                 -r["worst_mins"]))

    print()
    print("  in-game rows by market:")
    for mk, n in K[post]["market"].value_counts().items():
        print("    %-12s %5d" % (mk, n))

    # Printed because the week-blind join bug showed up ONLY as in-game rows
    # attributed to the wrong week. If a week appears here whose games had
    # not kicked off at capture time, the join has regressed.
    print()
    print("  in-game rows by the week label on the row:")
    for wk, n in K[post]["week"].value_counts().sort_index().items():
        print("    week %-3s %5d" % (wk, n))


# ------------------------------------------------- rule layer exposure

hr("F. RULE LAYER EXPOSURE")

print("  Rows the rushing low-line under rule could have fired on at a price")
print("  that was never available pregame. Rule is book=%s, market=rushing,"
      % RULE_BOOK)
print("  line <= %s, side UNDER." % RUSH_MAX_LINE)
print()
R = K[(K["market"] == "rushing") & (K["book"] == RULE_BOOK)
      & (K["line"].astype(float) <= RUSH_MAX_LINE)]
print("  qualifying rushing rows, all timings   %5d" % len(R))
print("  of those, captured IN GAME             %5d" % int((R["mins"] < 0).sum()))
print("  of those, captured IN WINDOW           %5d"
      % int(((R["mins"] >= WINDOW_LO) & (R["mins"] <= WINDOW_HI)).sum()))

if (R["mins"] < 0).any():
    print()
    print("  distinct player-lines affected:")
    bad = (R[R["mins"] < 0].groupby(["player", "line"])
           .agg(rows=("id", "size"), worst=("mins", "min"))
           .sort_values("worst").reset_index())
    for _, r in bad.head(30).iterrows():
        print("    %-24s line %5.1f   %3d rows   up to %.0f min in"
              % (str(r["player"])[:24], r["line"], r["rows"], -r["worst"]))
    print()
    print("  %d distinct player-lines total." % len(bad))


# ----------------------------------------------------------- salvage picture

hr("G. WHAT A BACKFILL WOULD SAVE")

if inwin.any():
    print("  Closing snapshots recoverable from week 3, by game:")
    z = (K[inwin].groupby("game_id")
         .agg(rows=("id", "size"), players=("player", "nunique"),
              mins=("mins", "median")).reset_index()
         .merge(sched_k[["game_id", "away_team", "home_team"]], on="game_id"))
    for _, r in z.sort_values("rows", ascending=False).iterrows():
        print("    %-4s at %-4s  %5d rows  %3d players  median %.0f min out"
              % (r["away_team"], r["home_team"], r["rows"], r["players"],
                 r["mins"]))
    print()
    print("  %d games with an in-window snapshot, out of %d week 3 games."
          % (z["game_id"].nunique(), len(w3k)))
    miss = w3k[~w3k["game_id"].isin(set(z["game_id"]))]
    if len(miss):
        print()
        print("  week 3 games with NO in-window snapshot:")
        for _, r in miss.iterrows():
            print("    %-4s at %-4s  %s %s ET"
                  % (r["away_team"], r["home_team"], r["weekday"],
                     r["gametime"]))
        print()
        print("  These are the games week 3 cannot contribute to a closing")
        print("  population no matter what the backfill does.")
else:
    print("  No rows fall in the %d to %d minute window. If that is the"
          % (WINDOW_LO, WINDOW_HI))
    print("  result, week 3 has no closing snapshot comparable to the")
    print("  historical instrument and should be excluded rather than fixed.")

print()
print("  NOTHING WAS WRITTEN. This script only reads.")
print()
