"""promote_closing.py - move closing snapshots from `lines` into the research table.

    python promote_closing.py --week 3                dry run, writes nothing
    python promote_closing.py --week 3 --apply        insert

WHY THIS EXISTS. Two tables, and they are not the same instrument:

    historical_lines   the RESEARCH table. 184,782 rows, 2023 through week
                       2 of 2026, one closing price per prop per book.
                       eval_harness.fetch_lines reads it and filters
                       snapshot_label = 'closing'.

    lines              what live_capture writes. 135,522 rows in week 3
                       alone, up to nine captures per prop.

So the automated capture has been running since 22 September and the
research layer cannot see any of it. side_picker_search_v2.py and
banded_under_test.py still end at week 2. That is plan item 1.8 and it is
why every result in this project is measured on the same rows that
generated its hypotheses.

SELECTION IS BY TIME, NOT BY RECENCY, AND THAT IS THE WHOLE DESIGN.
`lines` holds every capture including 2,391 in-game rows. The LAST capture
per prop would BE those rows, since they are latest by definition, and a
post-kickoff price has absorbed part of the outcome. So this takes the last
capture 20 to 90 minutes before kickoff, which excludes them structurally
rather than by a filter that could be forgotten.

    Measured consequence, accepted: week 3 keeps about 66 percent of props.
    The 4:05 and 4:25 clusters and Monday night are lost because the old
    19:23 UTC cron drifted 50 minutes and fired after kickoff. Those props
    contribute NOTHING rather than contributing a different quantity, which
    is the point. The retimed crons fix it from week 4.

WRITTEN WITH A DIFFERENT LABEL, DELIBERATELY. snapshot_label is
'closing_live', not 'closing'. eval_harness filters on 'closing', so these
rows are INVISIBLE to every existing script until someone widens that
filter on purpose. The write is therefore safe by construction: nothing in
the project changes until a separate, visible, one-line decision is taken.

    That also means running this does NOT fix the research gap on its own.
    It stages the data. Reading it is the next decision.

event_id IS THE SCHEDULE'S game_id. The column is NOT NULL and week 3 was
captured before live_capture stored the API's own event id, which is zero
of 135,522 rows. The mapping is (join_name, week) -> game_id straight from
player_stats, which is backfill_commence_time.py's tier 1 and is exact
rather than inferred. Tier 2's team-week fallback is deliberately NOT used:
week 3 is played, so every player with an outcome has a game_id, and a prop
with no outcome is of no use to a research table anyway.

    So promoted rows carry a different event_id FORMAT from the historical
    hashes. Honest rather than ideal. From week 5 the real id will be there.

captured_at GOES THROUGH UNCHANGED. historical_lines' own captured_at is
DERIVED: six values per week, each a constant 44.4 minutes before its
kickoff cluster, identical :36 seconds. These rows carry a genuinely
observed time instead, which is the one thing the automated era has that
the historical era does not, and normalising it away would discard it.

anytime_td IS EXCLUDED. It was 2,116 of yesterday's 4,099 promotable rows
and has no line, so it would add bulk to the research table for a market
nothing currently measures.
"""
import argparse
import sys

import numpy as np
import pandas as pd

import live_capture as lc
from models import data_utils

TARGET = "historical_lines"
SOURCE = "lines"
SPORT = "NFL"
LABEL = "closing_live"
WINDOW_LO, WINDOW_HI = 20, 90
MARKETS = ["receiving", "receptions", "rushing", "qb_passing"]
KEY = ["season", "week", "market", "player", "book"]
BATCH = 500


def hr(t):
    print()
    print("=" * 76)
    print(t)
    print("=" * 76)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    hr("PROMOTE CLOSING SNAPSHOTS  season %d week %d" % (args.season,
                                                         args.week))
    print("  source %s -> target %s" % (SOURCE, TARGET))
    print("  window %d-%d minutes before kickoff" % (WINDOW_LO, WINDOW_HI))
    print("  label  %s   (eval_harness filters on 'closing', so these are" % LABEL)
    print("         invisible to it until that filter is widened)")
    print("  markets %s   (anytime_td excluded: no line)" % MARKETS)
    print()
    print("  MODE: %s" % ("APPLY, rows will be inserted" if args.apply
                          else "DRY RUN, nothing will be written"))

    sec = lc._secrets()
    client = lc.supa_client(sec)

    hr("1. TARGET STATE")
    res = (client.table(TARGET)
           .select("snapshot_label", count="exact")
           .eq("season", args.season).eq("week", args.week).execute())
    existing = res.count or 0
    print("  %s already holds %d rows for %d week %d"
          % (TARGET, existing, args.season, args.week))
    if existing:
        labels = {}
        r2 = (client.table(TARGET).select("snapshot_label")
              .eq("season", args.season).eq("week", args.week)
              .limit(5000).execute())
        for row in (r2.data or []):
            labels[row["snapshot_label"]] = labels.get(row["snapshot_label"], 0) + 1
        print("  labels present: %s" % labels)
        print()
        print("  ABORT. This week already has rows in the research table.")
        print("  A second insert would DOUBLE-COUNT every prop, and every")
        print("  figure computed from that table afterwards would be wrong")
        print("  in a way nothing would flag. Delete the existing rows")
        print("  deliberately if a re-promotion is really intended.")
        sys.exit(1)
    print("  empty, so this is an insert rather than a merge")

    hr("2. READING THE SOURCE")
    rows, page = [], 0
    cols = ("season,week,market,player,book,line,over_odds,under_odds,"
            "captured_at,commence_time,projection")
    while True:
        r = (client.table(SOURCE).select(cols)
             .eq("season", args.season).eq("week", args.week)
             .in_("market", MARKETS)
             .order("id")
             .range(page * 1000, page * 1000 + 999).execute())
        b = r.data or []
        rows.extend(b)
        if len(b) < 1000:
            break
        page += 1
        if page > 400:
            break
    L = pd.DataFrame(rows)
    print("  %d rows in the four markets" % len(L))
    if L.empty:
        print("  nothing to promote")
        sys.exit(1)

    before = len(L)
    if "projection" in L.columns:
        L = L[L["projection"].isna()]
        print("  %d after excluding hand-saved rows (%d dropped)"
              % (len(L), before - len(L)))

    L["captured_at"] = pd.to_datetime(L["captured_at"], utc=True,
                                      errors="coerce", format="mixed")
    L["commence_time"] = pd.to_datetime(L["commence_time"], utc=True,
                                        errors="coerce", format="mixed")
    n0 = len(L)
    L = L[L["captured_at"].notna() & L["commence_time"].notna()]
    print("  %d with both timestamps (%d dropped)" % (len(L), n0 - len(L)))

    hr("3. THE WINDOW")
    L["mins_out"] = ((L["commence_time"] - L["captured_at"])
                     .dt.total_seconds() / 60.0)
    bands = [("in game (after kickoff)", L["mins_out"] < 0),
             ("0 to %d min" % WINDOW_LO,
              (L["mins_out"] >= 0) & (L["mins_out"] < WINDOW_LO)),
             ("IN WINDOW", (L["mins_out"] >= WINDOW_LO)
              & (L["mins_out"] <= WINDOW_HI)),
             ("more than %d min" % WINDOW_HI, L["mins_out"] > WINDOW_HI)]
    for label, mask in bands:
        print("  %-26s %7d" % (label, int(mask.sum())))
    W = L[(L["mins_out"] >= WINDOW_LO) & (L["mins_out"] <= WINDOW_HI)].copy()
    print()
    print("  %d in-window rows" % len(W))
    if W.empty:
        print("  nothing in the window. If this week's games are not yet")
        print("  played that is expected: a prop four days out cannot have")
        print("  a capture 20 to 90 minutes before kickoff.")
        sys.exit(1)

    # ONE-SIDED QUOTES, DROPPED BEFORE SELECTION.
    #
    # Measured 2026-10-01: BetRivers posts RECEPTIONS as a one-sided
    # product, 2,930 week-3 rows across 171 players with an over and no
    # under, and 1,726 more in week 4. So it is how that book behaves, not
    # a capture defect: a broken parser would affect every book, and Bovada
    # shows 8 rows and BetMGM 1, which is noise.
    #
    # A quote with no under cannot be devigged and has no under to settle,
    # so it is unusable in a table whose entire purpose is two-sided
    # closing prices. Dropped with a count rather than aborting, because
    # this is expected book behaviour.
    #
    # NOTE FOR COVERAGE WORK: coverage_test.py counts the share of active
    # books that POSTED a prop. BetRivers counts as having posted a
    # receptions prop while supplying no under, so the coverage denominator
    # is inflated for that market. Rule 1 is rushing, so the gradient
    # measured this morning is unaffected, but coverage on receptions would
    # be measuring partly this artifact.
    one_sided = W["over_odds"].isna() | W["under_odds"].isna()
    if one_sided.any():
        print()
        print("  %d one-sided in-window rows dropped:" % int(one_sided.sum()))
        for (bk, mk), g in W[one_sided].groupby(["book", "market"]):
            miss = "no under" if g["under_odds"].isna().all() else (
                "no over" if g["over_odds"].isna().all() else "mixed")
            print("    %-14s %-12s %5d  (%s)" % (bk, mk, len(g), miss))
        W = W[~one_sided].copy()
        print("  %d two-sided in-window rows remain" % len(W))

    # LAST capture inside the window, per prop-book.
    #
    # Dropping one-sided rows FIRST matters: if a book's last in-window
    # capture were one-sided and an earlier one complete, selecting first
    # would discard a usable row in favour of an unusable one and then drop
    # it, losing the prop entirely.
    W = W.sort_values("captured_at")
    sel = W.drop_duplicates(subset=KEY, keep="last").copy()
    print("  collapse to %d prop-book rows" % len(sel))

    hr("4. event_id FROM player_stats, TIER 1 ONLY")
    stats = data_utils.load_player_stats([args.season])
    stats = stats.to_pandas() if hasattr(stats, "to_pandas") else stats
    st = stats[stats["week"] == args.week].copy()
    st["join_name"] = data_utils.norm_join_name(st["player_display_name"])
    t1 = (st[["join_name", "game_id"]]
          .dropna().drop_duplicates(subset=["join_name"]))
    print("  %d players with a week %d stat line and a game_id"
          % (len(t1), args.week))

    sel["join_name"] = data_utils.norm_join_name(sel["player"])
    n1 = len(sel)
    sel = sel.merge(t1, on="join_name", how="left")
    matched = sel["game_id"].notna()
    print("  %d of %d rows matched (%.1f%%)"
          % (int(matched.sum()), n1, 100.0 * matched.mean()))
    if (~matched).any():
        miss = (sel.loc[~matched, "player"].value_counts().head(10))
        print("  most frequent unmatched:")
        for nm, c in miss.items():
            print("    %-28s %d" % (nm[:28], c))
        print("  these are dropped: a prop with no outcome is of no use to")
        print("  a research table, and tier 2's team-week guess is not used")
        print("  on a played week.")
    sel = sel[matched].copy()

    # home / away from the schedule, via the same game_id
    sched = data_utils.load_schedules([args.season])
    sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
    sc = sched[["game_id", "home_team", "away_team"]].drop_duplicates()
    sel = sel.merge(sc, on="game_id", how="left")
    print("  home/away attached on %d of %d rows"
          % (int(sel["home_team"].notna().sum()), len(sel)))

    hr("5. GATES")
    ok = True
    neg = int((sel["mins_out"] < 0).sum())
    print("  no selected row post-kickoff ....... %d  %s"
          % (neg, "OK" if neg == 0 else "FAIL"))
    ok &= neg == 0
    dup = int(len(sel) - len(sel.drop_duplicates(subset=KEY)))
    print("  one row per prop-book .............. %d  %s"
          % (dup, "OK" if dup == 0 else "FAIL"))
    ok &= dup == 0
    noev = int(sel["game_id"].isna().sum())
    print("  every row has an event_id .......... %d  %s"
          % (noev, "OK" if noev == 0 else "FAIL"))
    ok &= noev == 0
    noline = int(sel["line"].isna().sum())
    print("  every row has a line ............... %d  %s"
          % (noline, "OK" if noline == 0 else "FAIL"))
    ok &= noline == 0
    # Still a hard gate. One-sided rows were dropped above with a named
    # reason, so anything reaching here is missing a price for a reason we
    # have not identified, and that should stop the run rather than be
    # filtered silently.
    nopx = int(sel["over_odds"].isna().sum() + sel["under_odds"].isna().sum())
    print("  both prices present ................ %d  %s"
          % (nopx, "OK" if nopx == 0 else "FAIL"))
    if nopx:
        print("    one-sided rows were already dropped above, so these are")
        print("    unexplained. Investigate rather than widening the drop.")
    ok &= nopx == 0
    if not ok:
        print()
        print("  A GATE FAILED. Nothing written.")
        sys.exit(1)

    hr("6. WHAT WOULD BE INSERTED")
    print("  %d rows" % len(sel))
    print()
    print("  %-12s %8s %8s %10s" % ("market", "rows", "props", "med min"))
    print("  " + "-" * 42)
    for mk, g in sel.groupby("market"):
        print("  %-12s %8d %8d %10.0f"
              % (mk, len(g), g["player"].nunique(), g["mins_out"].median()))
    print()
    print("  by book:")
    for bk, g in sel.groupby("book"):
        print("    %-16s %6d" % (bk, len(g)))

    uid = None
    try:
        uid = client.auth.get_user().user.id
    except Exception:
        pass
    print()
    print("  user_id: %s" % (uid or "COULD NOT RESOLVE"))
    if uid is None:
        print("  historical_lines.user_id is NOT NULL, so the insert would")
        print("  fail. Nothing written.")
        sys.exit(1)

    payload = []
    for _, r in sel.iterrows():
        payload.append({
            "user_id": uid,
            "sport": SPORT,
            "book": r["book"],
            "season": int(r["season"]),
            "week": int(r["week"]),
            "market": r["market"],
            "player": r["player"],
            "line": float(r["line"]),
            "over_odds": float(r["over_odds"]),
            "under_odds": float(r["under_odds"]),
            "captured_at": r["captured_at"].isoformat(),
            "event_id": str(r["game_id"]),
            "commence_time": r["commence_time"].isoformat(),
            "home_team": (None if pd.isna(r.get("home_team"))
                          else r.get("home_team")),
            "away_team": (None if pd.isna(r.get("away_team"))
                          else r.get("away_team")),
            "snapshot_label": LABEL,
        })
    print()
    print("  sample row:")
    for k, v in payload[0].items():
        print("    %-16s %s" % (k, v))

    if not args.apply:
        hr("DRY RUN")
        print("  Nothing written. Rerun with --apply to insert %d rows."
              % len(payload))
        print()
        print("  After inserting, NOTHING in the project changes: the label")
        print("  is '%s' and eval_harness filters on 'closing'." % LABEL)
        print("  Widening that filter is a separate decision.")
        print()
        return

    hr("7. INSERTING")
    done = 0
    for i in range(0, len(payload), BATCH):
        chunk = payload[i:i + BATCH]
        try:
            client.table(TARGET).insert(chunk).execute()
            done += len(chunk)
            print("\r  %d of %d" % (done, len(payload)), end="", flush=True)
        except Exception as e:
            print()
            print("  INSERT FAILED on batch starting %d: %s: %s"
                  % (i, type(e).__name__, e))
            print("  %d rows were already written. The target now holds a"
                  % done)
            print("  PARTIAL week, which is worse than none: delete rows")
            print("  with snapshot_label = '%s' for this week before" % LABEL)
            print("  retrying.")
            sys.exit(1)
    print()
    print("  inserted %d rows" % done)

    res = (client.table(TARGET).select("snapshot_label", count="exact")
           .eq("season", args.season).eq("week", args.week).execute())
    print("  target now holds %d rows for this week" % (res.count or 0))
    print()
    print("  Verify before trusting it:")
    print("    select market, count(*), min(captured_at), max(captured_at)")
    print("    from historical_lines")
    print("    where season = %d and week = %d and snapshot_label = '%s'"
          % (args.season, args.week, LABEL))
    print("    group by market;")
    print()


if __name__ == "__main__":
    main()
