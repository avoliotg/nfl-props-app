"""
Why does load_player_stats return a different number of rows each call?

Observed, same code, adjacent runs:
    build_all_rows -> train mask after dropna: 6,289 then 9,927
    build_dataset  -> 15,422 then 15,019

The training mask covers 2022-2024 only. Those seasons are closed, so their
row count cannot legitimately change. A frame that shrinks means the fetch is
returning PARTIAL data some of the time, silently, and every model in the
project trains on that frame.

This script calls the loader repeatedly with the Streamlit cache cleared
between calls and reports per-season counts, so a partial fetch shows up as a
season that is short or missing rather than as a visible error.

Run from the repo root with the venv active:
    python data_fetch_check.py
"""
import inspect
import os
import sys

import pandas as pd
import streamlit as st

from models import data_utils

SEASONS = [2022, 2023, 2024, 2025, 2026]
N_CALLS = 4

# Roughly what a complete season should look like for all positions. Used only
# to flag a season that is obviously short, not as a hard assertion.
MIN_EXPECTED_FULL_SEASON = 9000


def _as_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def show_source():
    print("=" * 74)
    print("1. WHAT DOES THE LOADER ACTUALLY DO?")
    print("=" * 74)
    for fn_name in ("load_player_stats", "load_snap_counts", "load_schedules"):
        fn = getattr(data_utils, fn_name, None)
        if fn is None:
            print(f"\n  {fn_name}: not found")
            continue
        print(f"\n  ---- {fn_name} ----")
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            print("    source unavailable")
            continue
        print("\n".join("    " + ln for ln in src.splitlines()))
        low = src.lower()
        if "except" in low and "pass" in low:
            print("    >>> NOTE: contains an except block that may swallow a")
            print("    >>> failed fetch and return partial data.")
        if "try" in low and "return" in low and "except" in low:
            print("    >>> NOTE: has try/except around a return path. A partial")
            print("    >>> result here would look identical to a full one.")


def show_cache_dirs():
    print("\n" + "=" * 74)
    print("2. WHERE IS NFLREADPY CACHING, AND IS THE CACHE WRITABLE?")
    print("=" * 74)
    env_keys = [k for k in os.environ
                if "NFLREAD" in k.upper() or "NFL_" in k.upper()]
    if env_keys:
        for k in env_keys:
            print(f"  {k} = {os.environ[k]}")
    else:
        print("  no nflreadpy environment variables set")
    try:
        import nflreadpy
        print(f"  nflreadpy version: {getattr(nflreadpy, '__version__', 'unknown')}")
        cfg = getattr(nflreadpy, "config", None)
        if cfg is not None:
            for attr in dir(cfg):
                if attr.startswith("_"):
                    continue
                if "cache" in attr.lower() or "dir" in attr.lower():
                    try:
                        print(f"  config.{attr} = {getattr(cfg, attr)}")
                    except Exception:
                        pass
    except Exception as e:
        print(f"  could not inspect nflreadpy: {e}")
    try:
        import platformdirs
        d = platformdirs.user_cache_dir("nflreadpy")
        print(f"  likely cache dir: {d}")
        print(f"  exists: {os.path.isdir(d)}")
        if os.path.isdir(d):
            files = []
            for root, _dirs, names in os.walk(d):
                for n in names:
                    p = os.path.join(root, n)
                    try:
                        files.append((os.path.getsize(p), p))
                    except OSError:
                        pass
            print(f"  cached files: {len(files)}")
            for size, p in sorted(files, reverse=True)[:8]:
                print(f"    {size / 1e6:>8.2f} MB  {os.path.basename(p)}")
            tiny = [p for size, p in files if size < 1000]
            if tiny:
                print(f"  >>> {len(tiny)} suspiciously small cached file(s), which")
                print("  >>> is what a truncated download looks like:")
                for p in tiny[:5]:
                    print(f"    {os.path.basename(p)}")
    except Exception as e:
        print(f"  could not resolve cache dir: {e}")


def repeat_calls():
    print("\n" + "=" * 74)
    print("3. CALL THE LOADER REPEATEDLY, CACHE CLEARED BETWEEN CALLS")
    print("=" * 74)
    print("  Each call should return an identical frame. Any variation in a")
    print("  CLOSED season (2022-2024) is a partial fetch.\n")

    results = []
    for i in range(N_CALLS):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            ps = _as_pandas(data_utils.load_player_stats(SEASONS))
        except Exception as e:
            print(f"  call {i + 1}: RAISED {type(e).__name__}: {e}")
            results.append(None)
            continue
        by_season = ps.groupby("season").size().to_dict()
        results.append((len(ps), by_season))
        parts = "  ".join(f"{int(k)}:{v}" for k, v in sorted(by_season.items()))
        print(f"  call {i + 1}: {len(ps):>7} rows   {parts}")

    good = [r for r in results if r is not None]
    if len(good) < 2:
        print("\n  not enough successful calls to compare")
        return

    totals = {r[0] for r in good}
    print("\n" + "-" * 74)
    if len(totals) == 1:
        print("  STABLE across calls. The variation seen earlier was not")
        print("  reproduced here, so suspect a transient network failure or a")
        print("  cache write that was in progress at the time.")
    else:
        print(f"  UNSTABLE: {len(totals)} different row counts across "
              f"{len(good)} calls.")
        print(f"  counts seen: {sorted(totals)}")
        print("\n  per-season spread (a closed season varying is the smoking gun):")
        all_seasons = sorted({s for _n, bs in good for s in bs})
        for s in all_seasons:
            vals = sorted({bs.get(s, 0) for _n, bs in good})
            flag = ""
            if len(vals) > 1:
                flag = "  <<< VARIES"
                if int(s) <= 2024:
                    flag += " IN A CLOSED SEASON"
            short = [v for v in vals if 0 < v < MIN_EXPECTED_FULL_SEASON]
            if short and int(s) <= 2025:
                flag += "  (and looks short)"
            print(f"    {int(s)}: {vals}{flag}")


def main():
    print(f"python {sys.version.split()[0]}  pandas {pd.__version__}")
    show_source()
    show_cache_dirs()
    repeat_calls()

    print("\n" + "=" * 74)
    print("WHAT TO DO WITH THIS")
    print("=" * 74)
    print("  If section 3 is UNSTABLE, the loader needs a completeness check:")
    print("  fetch, then assert the expected seasons are present with a")
    print("  plausible row count, and raise rather than return a short frame.")
    print("  A silently short frame trains a silently different model, which is")
    print("  the same failure mode as the snap-count merge and the null")
    print("  projections: wrong answers with no error.")
    print("  If section 2 shows tiny cached files, delete the cache directory")
    print("  and re-run, since a truncated download will keep being reused.")
    print("  If section 3 is STABLE, record the row counts somewhere and treat")
    print("  any future deviation as a fetch problem rather than a code change.")


if __name__ == "__main__":
    main()
