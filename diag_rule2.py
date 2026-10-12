"""diag_rule2.py - why does the receiving rule fire zero times?

Touches nflverse only. No Supabase, no Odds API, no Streamlit. The point is
to isolate the air-yards carrier from everything the app wraps around it, so
a zero can be attributed rather than guessed at.

    python diag_rule2.py
    python diag_rule2.py --season 2026 --weeks 1,2,3,4,5,6,7

Reads nothing, writes nothing.
"""
import argparse

import numpy as np
import pandas as pd

import rules
from models import data_utils as du


def hr(t):
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", default="1,2,3,4,5,6,7")
    ap.add_argument("--seasons", default="2022,2023,2024,2025,2026",
                    help="must match rules.evaluate's default")
    args = ap.parse_args()

    seasons = tuple(int(s) for s in args.seasons.split(","))
    weeks = [int(w) for w in args.weeks.split(",")]

    hr("1. WHAT NFLVERSE ACTUALLY HAS")
    ps = du.load_player_stats(seasons)
    print("  %d rows, seasons %s"
          % (len(ps), sorted(ps["season"].dropna().unique().tolist())))
    col = "receiving_air_yards"
    if col not in ps.columns:
        print("  FATAL: no %s column. Candidates: %s"
              % (col, [c for c in ps.columns if "air" in c.lower()]))
        return
    cur = ps[ps["season"] == args.season]
    if len(cur):
        wk = sorted(cur["week"].dropna().unique().tolist())
        print("  season %d weeks present: %s" % (args.season, wk))
        print("  LAST COMPLETED WEEK in the data: %s" % (max(wk) if wk else "none"))
    else:
        print("  season %d has NO rows at all" % args.season)
    nz = cur[cur[col].notna() & (cur[col] != 0)]
    print("  season %d rows with a non-zero %s: %d" % (args.season, col, len(nz)))

    hr("2. CARRIER ROWS BY TARGET WEEK")
    print("  A carrier is only built for a week with NO stat rows. A week")
    print("  that has been played uses the exact merge instead, so 0 carrier")
    print("  rows there is CORRECT, not a failure.")
    print()
    print("  %6s %9s %9s %9s %9s %9s"
          % ("week", "played?", "carrier", "exact", "z>=1.0", "max z"))
    cal = rules.CALIBRATION["receiving"]
    for w in weeks:
        played = bool(len(ps[(ps["season"] == args.season) & (ps["week"] == w)]))
        rules._air_yards_history.cache_clear()
        h = rules._air_yards_history(seasons, ((args.season, w),))
        sub = h[(h["season"] == args.season) & (h["week"] == w)]
        shock = pd.to_numeric(sub["shock_raw"], errors="coerce")
        z = (shock - cal["shock_mean"]) / cal["shock_sd"]
        n_trig = int((z >= rules.AIR_YARDS_Z_TRIGGER).sum())
        mx = float(z.max()) if len(z.dropna()) else float("nan")
        print("  %6d %9s %9d %9d %9d %9.3f"
              % (w, "yes" if played else "NO",
                 0 if played else len(sub), len(sub) if played else 0,
                 n_trig, mx))

    hr("3. FULL evaluate() ON SYNTHETIC RECEIVING PROPS")
    print("  Real air-yards history, real calibration, synthetic lines and")
    print("  projections. Isolates the rule from get_line_movement.")
    names = (ps[ps["season"] == args.season]
             .groupby("player_display_name")[col].sum()
             .sort_values(ascending=False).head(60).index.tolist())
    if not names:
        print("  no season-%d receivers found, cannot build props" % args.season)
        return
    print("  using the top %d receivers by season air yards" % len(names))
    for w in weeks:
        props = pd.DataFrame({
            "market": "receiving", "player": names, "line": 55.0,
            "season": args.season, "week": w, "projection": 40.0,
        })
        rules._air_yards_history.cache_clear()
        try:
            fired, rep = rules.evaluate(props, seasons=seasons)
        except Exception as e:
            print("  week %-3d RAISED %s: %s" % (w, type(e).__name__, e))
            continue
        r = rep["rules"]["receiving_air_yards_under"]
        print("  week %-3d eligible %-4s no_hist %-4s trig %-4s vetoed %-4s "
              "fired %-4s" % (w, r.get("eligible"), r.get("no_air_yards_history"),
                              r.get("triggered_before_veto"),
                              r.get("vetoed_by_model"), r.get("fired")))
        if not r.get("fired"):
            print("        reason: %s" % (r.get("reason") or "none recorded"))

    hr("4. READING THIS")
    print("  no_hist equal to eligible      -> carrier not built. Check")
    print("                                    section 2: is the target week")
    print("                                    marked played? Is history deep")
    print("                                    enough for the base window?")
    print("  trig 0 with no_hist 0          -> carrier IS working and no")
    print("                                    receiver has z >= %.1f this"
          % rules.AIR_YARDS_Z_TRIGGER)
    print("                                    week. A real zero.")
    print("  trig > 0 but fired 0           -> the projection veto is eating")
    print("                                    every firing. Synthetic")
    print("                                    projections here are 40.0, so")
    print("                                    rerun with a real board before")
    print("                                    concluding that.")
    print("  RAISED                         -> the app would have caught this")
    print("                                    into lm_rule_errors and shown 0")


if __name__ == "__main__":
    main()
