"""diag_movement.py - is line MOVEMENT measurable with what we have?

READ ONLY. Writes nothing, fits nothing, claims nothing.

THE IDEA BEING TESTED FOR FEASIBILITY

    Every route to beating the market's LOCATION using our features is
    closed. What is not closed is movement: the direction and size of the
    change in a prop's price between when it is posted and kickoff is not a
    fact about the player, it is a record of what the market learned after
    posting. That is new information by construction rather than a better
    arrangement of the same features, and it needs no external data source.

    This script does NOT test whether movement predicts anything. It asks
    the prior question: does the data contain movement at all, and if so how
    much, on how many props, with how many graded outcomes. Fitting anything
    before knowing that is how you get a result you cannot interpret.

MEASURE MOVEMENT IN DEVIGGED PROBABILITY, NOT IN THE LINE

    receptions is about 93 percent flat on the line because FanDuel moves
    reception prices through the ODDS. A movement measure defined as a
    change in the line would therefore find nothing in receptions by
    construction, and would understate movement everywhere that a book
    adjusts price before it adjusts the number. Devigged two-sided
    probability captures both.

TWO SOURCES, PROBABLY VERY DIFFERENT

    lines_cache.parquet is the historical instrument and is understood to be
    CLOSING lines: if it holds one snapshot per prop it contains no movement
    and cannot answer this at all.

    The `lines` table's automated era is six days old but has 36 snapshots
    across 8 books, and since today every row carries commence_time, so
    distance to kickoff is a subtraction.

Run:  python diag_movement.py
"""
import os
import sys

import numpy as np
import pandas as pd

import live_capture as lc

CACHE = "lines_cache.parquet"
AUTO_ERA_START = "2026-09-22T00:00:00+00:00"
KEY = ["season", "week", "market", "player", "book"]


def hr(t):
    print()
    print("=" * 76)
    print(t)
    print("=" * 76)


def dec(o):
    o = pd.to_numeric(o, errors="coerce")
    return np.where(o > 0, 1.0 + o / 100.0, 1.0 + 100.0 / (-o))


def devig(over_odds, under_odds):
    """Two-sided devigged P(over). NaN where either side is missing."""
    do, du = dec(over_odds), dec(under_odds)
    ro, ru = 1.0 / do, 1.0 / du
    tot = ro + ru
    return np.where(np.isfinite(tot) & (tot > 0), ro / tot, np.nan)


def snaps_per_prop(df, label):
    g = df.groupby(KEY, dropna=False).size()
    print("  %s: %d prop-book rows over %d distinct prop-books"
          % (label, len(df), len(g)))
    print("  snapshots per prop-book:")
    vc = g.value_counts().sort_index()
    shown = 0
    for k, v in vc.items():
        if shown >= 8:
            print("    %3d+ snapshots  %7d prop-books (remainder)"
                  % (k, int(vc[vc.index >= k].sum())))
            break
        print("    %3d  snapshots  %7d prop-books" % (k, v))
        shown += 1
    multi = int((g > 1).sum())
    print("  prop-books with MORE THAN ONE snapshot: %d of %d (%.1f%%)"
          % (multi, len(g), 100.0 * multi / max(len(g), 1)))
    return g


# ------------------------------------------------- A. the historical cache

hr("A. lines_cache.parquet: does the closing dataset contain movement")

if not os.path.exists(CACHE):
    print("  %s not found" % CACHE)
else:
    C = pd.read_parquet(CACHE)
    print("  %d rows, seasons %s"
          % (len(C), sorted(C["season"].dropna().unique().tolist())))
    if "snapshot_label" in C.columns:
        print("  snapshot_label values: %d distinct"
              % C["snapshot_label"].nunique())
        for k, v in C["snapshot_label"].value_counts().head(8).items():
            print("    %-28s %7d rows" % (str(k)[:28], v))
    print()
    snaps_per_prop(C, "cache")
    print()
    print("  captured_at values per (season, week): a closing-line dataset")
    print("  should show roughly one capture per slate.")
    cw = (C.groupby(["season", "week"])["captured_at"].nunique()
          .rename("n_captures").reset_index())
    print("    median captures per season-week: %.1f"
          % cw["n_captures"].median())
    print("    range: %d to %d" % (cw["n_captures"].min(),
                                   cw["n_captures"].max()))
    print()
    print("  VERDICT: if nearly every prop-book has exactly one snapshot,")
    print("  the cache is a closing-line dataset and carries no movement.")
    print("  Movement can then only come from the automated era, which is")
    print("  six days old, and this becomes a compounding asset rather")
    print("  than something testable now.")


# ----------------------------------------------------- B. the automated era

hr("B. the `lines` table, automated era")

sec = lc._secrets()
client = lc.supa_client(sec)

rows, page = [], 0
while True:
    res = (client.table("lines")
           .select("id,season,week,market,player,line,over_odds,under_odds,"
                   "book,captured_at,commence_time")
           .gte("captured_at", AUTO_ERA_START)
           .order("id")
           .range(page * 1000, page * 1000 + 999).execute())
    b = res.data or []
    rows.extend(b)
    if len(b) < 1000:
        break
    page += 1
    if page > 400:
        break

L = pd.DataFrame(rows)
print("  %d rows" % len(L))
if L.empty:
    print("  nothing to analyse")
    sys.exit(0)

L["captured_at"] = pd.to_datetime(L["captured_at"], utc=True, format="mixed")
L["commence_time"] = pd.to_datetime(L["commence_time"], utc=True,
                                    format="mixed", errors="coerce")
have_kick = L["commence_time"].notna()
print("  %d rows carry a kickoff (%.1f%%)"
      % (int(have_kick.sum()), 100.0 * have_kick.mean()))

L = L[have_kick].copy()
L["mins_out"] = ((L["commence_time"] - L["captured_at"])
                 .dt.total_seconds() / 60.0)
# Pregame only. An in-game price is not part of a movement path.
L = L[L["mins_out"] > 0].copy()
print("  %d pregame rows, lead time %.0f to %.0f minutes"
      % (len(L), L["mins_out"].min(), L["mins_out"].max()))
print()
g = snaps_per_prop(L, "automated era")


# -------------------------------------------- C. how much does price move

hr("C. HOW MUCH DOES THE PRICE ACTUALLY MOVE")

L["p_over"] = devig(L["over_odds"], L["under_odds"])
usable = L[L["p_over"].notna() & L["line"].notna()].copy()
print("  %d rows have a two-sided devigged price (%.1f%% of pregame rows)"
      % (len(usable), 100.0 * len(usable) / max(len(L), 1)))
print("  anytime_td has no under side, so it is absent here by design:")
for m, n in L.groupby("market")["p_over"].apply(
        lambda s: int(s.isna().sum())).items():
    print("    %-12s %6d rows without a two-sided price" % (m, n))

if usable.empty:
    print("  nothing further to measure")
    sys.exit(0)

usable = usable.sort_values("mins_out", ascending=False)
first = usable.groupby(KEY, dropna=False).first()
last = usable.groupby(KEY, dropna=False).last()
path = first.join(last, lsuffix="_first", rsuffix="_last")
path = path[path["mins_out_first"] > path["mins_out_last"]]
print()
print("  %d prop-books have two distinct pregame times" % len(path))
if path.empty:
    print("  no movement paths exist yet")
    sys.exit(0)

path["d_line"] = path["line_last"] - path["line_first"]
path["d_p"] = path["p_over_last"] - path["p_over_first"]
path["span_mins"] = path["mins_out_first"] - path["mins_out_last"]

print("  median span between first and last pregame price: %.0f minutes"
      % path["span_mins"].median())
print()
print("  %-12s %7s %11s %11s %11s %11s"
      % ("market", "paths", "line moved", "med |dline|", "price moved",
         "med |dp|"))
for m, grp in path.groupby(level="market"):
    lm = float((grp["d_line"].abs() > 1e-9).mean())
    pm = float((grp["d_p"].abs() > 1e-9).mean())
    print("  %-12s %7d %10.1f%% %11.2f %10.1f%% %11.4f"
          % (m, len(grp), 100 * lm, grp["d_line"].abs().median(),
             100 * pm, grp["d_p"].abs().median()))

print()
print("  'line moved' vs 'price moved' is the point. A market where the")
print("  line rarely moves but the price often does is one where movement")
print("  defined on the LINE would be invisible.")


# ----------------------------------------------- D. how much is gradeable

hr("D. WHAT COULD BE SCORED TODAY, AND POWER")

print("  Movement can only be scored where the game has been played.")
wk = path.reset_index().groupby("week").size()
for w, n in wk.items():
    print("    week %-3s %6d movement paths" % (w, n))
print()
print("  Week 3 is complete; week 4 is not. So the gradeable set is the")
print("  week 3 paths only, and they come from ONE slate of 16 games.")

n_w3 = int(wk.get(3, 0))
games = 16
print()
print("  ROUGH POWER, and the number that matters is GAMES not paths.")
print("  Props inside a game share game script, so the effective sample")
print("  for a movement slope is closer to the game count. With %d games"
      % games)
print("  a game-clustered slope has an MDE far above any plausible effect.")
print()
print("  For scale, the side-picker search needed about 1,100 out-of-sample")
print("  rows across 2 seasons to reach an MDE near 0.9 on a slope, and")
print("  that was already underpowered. One slate is not a test.")
print()
print("  ACCUMULATION RATE, if capture keeps running:")
per_week = n_w3 if n_w3 else 0
for weeks in (4, 9, 14, 18):
    ahead = max(weeks - 3, 0)
    print("    by week %-3s about %7d paths across %3d games"
          % (weeks, per_week * (ahead + 1), 16 * (ahead + 1)))
print()
print("  READ THIS AS A BUILD-OR-WAIT DECISION, not a result. If the paths")
print("  per week are large and the price genuinely moves, the honest plan")
print("  is to keep capturing, write the scoring script now, and run it")
print("  when the game count can support it. Fitting it on one slate would")
print("  produce a number that cannot be distinguished from noise and")
print("  would then be hard to unlearn.")
print()
print("  NOTHING WAS WRITTEN.")
print()
