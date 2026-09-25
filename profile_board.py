"""profile_board.py - where does the board's time actually go?

Reports only. Writes nothing.

THE COMPLAINT

    "A two minute task becomes thirty minutes" because the Streamlit board is
    slow to refresh.

THE DIAGNOSIS THIS SCRIPT CHECKS

    build_upcoming_week had NO cache decorator in any of the six market
    modules, and project_week calls it unconditionally on every call. Each one
    performs two to three uncached nflverse loads: rosters, schedules, and
    depth charts for the QB markets. Streamlit re-runs the entire script on
    every widget interaction, so each dropdown click re-read all of that from
    scratch, once per market. app.py renders the rushing board from BOTH
    rushing.project_week and qb_rushing.project_week, so that board paid for
    two full assemblers.

    A ttl=1800 cache is now on all six. This script measures whether that was
    in fact the bottleneck, because a guess about performance is still a
    guess.

HOW TO READ IT

    COLD is the first call, which pays the full cost. WARM is the second,
    which should be near zero if the cache is working. The ratio is the
    speedup a returning widget click now gets.

    If build_upcoming_week's COLD time is large and its WARM time is small,
    the fix worked. If COLD is already small, the bottleneck is elsewhere and
    the next suspect is app.py's render path rather than these modules.

    Run it OUTSIDE Streamlit. The cache decorators still function, with a
    "No runtime found" warning that can be ignored.

Run from the repo root:
    python profile_board.py
    python profile_board.py --season 2026 --week 3
"""
import argparse
import importlib
import sys
import time

MODULES = ["receiving", "receptions", "rushing", "qb_rushing", "qb_passing",
           "anytime_td"]


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    try:
        out = fn(*a, **k)
        n = len(out) if hasattr(out, "__len__") else None
        return time.perf_counter() - t0, n, None
    except Exception as e:
        return time.perf_counter() - t0, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=3)
    args = ap.parse_args()

    print("=" * 88)
    print(f"BOARD PROFILE   season {args.season}  week {args.week}")
    print("=" * 88)
    print()
    print(f"  {'module':<13}{'step':<22}{'cold s':>9}{'warm s':>9}"
          f"{'speedup':>10}{'rows':>8}")

    totals = {"cold": 0.0, "warm": 0.0}
    for name in MODULES:
        try:
            mod = importlib.import_module(f"models.{name}")
        except Exception as e:
            print(f"  {name:<13}import failed: {type(e).__name__}: {e}")
            continue

        # build_dataset first, so its cost is not attributed to the assembler
        c, n, err = timed(mod.build_dataset)
        w, _, _ = timed(mod.build_dataset)
        print(f"  {name:<13}{'build_dataset':<22}{c:>9.2f}{w:>9.2f}"
              f"{(c / w if w > 0.001 else float('inf')):>10.0f}"
              f"{(n if n is not None else 0):>8,}")
        if err:
            print(f"  {'':<13}  {err}")
            continue

        if hasattr(mod, "build_upcoming_week"):
            c, n, err = timed(mod.build_upcoming_week, args.season, args.week)
            w, _, _ = timed(mod.build_upcoming_week, args.season, args.week)
            flag = "  <-- the suspect" if c > 1.0 else ""
            print(f"  {name:<13}{'build_upcoming_week':<22}{c:>9.2f}{w:>9.2f}"
                  f"{(c / w if w > 0.001 else float('inf')):>10.0f}"
                  f"{(n if n is not None else 0):>8,}{flag}")
            if err:
                print(f"  {'':<13}  {err}")

        c, n, err = timed(mod.project_week, args.season, args.week)
        w, _, _ = timed(mod.project_week, args.season, args.week)
        print(f"  {name:<13}{'project_week':<22}{c:>9.2f}{w:>9.2f}"
              f"{(c / w if w > 0.001 else float('inf')):>10.0f}"
              f"{(n if n is not None else 0):>8,}")
        if err:
            print(f"  {'':<13}  {err}")
        totals["cold"] += c
        totals["warm"] += w
        print()

    print(f"  {'TOTAL project_week across six markets':<48}"
          f"{totals['cold']:>9.2f}{totals['warm']:>9.2f}")
    print()
    print("  The WARM total is what a repeat widget click costs once the")
    print("  caches are populated. The COLD total is what the first render")
    print("  after a deploy or a cache expiry costs.")
    print()
    print("  NOTE ON WHAT project_week's WARM TIME INCLUDES. project_week is")
    print("  itself uncached, so its warm time is the model predict, the")
    print("  sort, and the concat, on top of cached inputs. If that number is")
    print("  still large, caching project_week too is the next step. If it is")
    print("  small, the board is as fast as these modules can make it and any")
    print("  remaining slowness is in app.py's render path.")
    print()
    print("  AND A REMINDER THAT MAY MATTER MORE THAN ANY OF THIS. Logging a")
    print("  few bets does not need Streamlit at all. A CSV and a ten-line")
    print("  script from cmd is instant and has no refresh cycle to wait on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
