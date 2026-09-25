"""smoke_actual_stat.py - planted-answer rig for data_utils.actual_stat.

No network, no database: load_player_stats is replaced with a synthetic frame
carrying the real collisions measured by name_resolve.py.

Run from the repo root:
    python smoke_actual_stat.py
"""
import sys

import numpy as np
import pandas as pd


def _frame():
    """Synthetic nflverse stats with the traps that matter.

    - Michael Carter appears TWICE in 2025 week 1 under two player_ids, a CB
      and an RB. This is the real collision and the worst one: grading must
      REFUSE rather than pick a side.
    - Anthony Brown is the same problem across QB and CB.
    - Lamar Jackson collides too, but only the QB played in week 2, so that
      week is unambiguous and must grade normally.
    - Mitchell Trubisky is spelled the nflverse way, so grading a bet written
      "Mitch Trubisky" must resolve through the alias.
    - Zero Runner has a null stat and must grade as 0.0.
    - Solo Player is uncontested and must just work.
    """
    r = [
        # season, week, name, position, id, rush, rec, recyd
        (2025, 1, "Michael Carter", "RB", "id-rb", 42.0, 3.0, 25.0),
        (2025, 1, "Michael Carter", "CB", "id-cb", 0.0, 0.0, 0.0),
        (2025, 1, "Anthony Brown", "QB", "id-qb", 11.0, 0.0, 0.0),
        (2025, 1, "Anthony Brown", "CB", "id-cb2", 0.0, 0.0, 0.0),
        (2025, 1, "Lamar Jackson", "QB", "id-lj", 55.0, 0.0, 0.0),
        (2025, 1, "Lamar Jackson", "CB", "id-lj2", 0.0, 0.0, 0.0),
        (2025, 2, "Lamar Jackson", "QB", "id-lj", 61.0, 0.0, 0.0),
        (2025, 1, "Mitchell Trubisky", "QB", "id-mt", 7.0, 0.0, 0.0),
        (2025, 1, "Zero Runner", "QB", "id-zr", np.nan, np.nan, np.nan),
        (2025, 1, "Solo Player", "WR", "id-sp", 0.0, 6.0, 88.0),
    ]
    return pd.DataFrame(r, columns=["season", "week", "player_display_name",
                                    "position", "player_id", "rushing_yards",
                                    "receptions", "receiving_yards"])


def main():
    try:
        from models import data_utils
    except ImportError:
        import importlib.util
        import os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "models", "data_utils.py")
        spec = importlib.util.spec_from_file_location("data_utils", p)
        data_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_utils)

    data_utils.load_player_stats = lambda seasons: _frame()
    if hasattr(data_utils._stats_frame, "cache_clear"):
        data_utils._stats_frame.cache_clear()

    S = [2025]
    A = lambda *a, **k: data_utils.actual_stat(*a, seasons=S, **k)
    f = []

    def chk(name, got, want):
        if got != want and not (got is None and want is None):
            f.append(f"  {name}: got {got!r}, expected {want!r}")

    # --- the collision guard, which is the whole point ---
    chk("Michael Carter refused", A(2025, 1, "Michael Carter",
                                    "rushing_yards"), None)
    chk("Anthony Brown refused", A(2025, 1, "Anthony Brown",
                                   "rushing_yards"), None)
    chk("Lamar Jackson wk1 refused", A(2025, 1, "Lamar Jackson",
                                       "rushing_yards"), None)

    # --- but an unambiguous week of a colliding name must still grade ---
    chk("Lamar Jackson wk2 grades", A(2025, 2, "Lamar Jackson",
                                      "rushing_yards"), 61.0)

    # --- position narrowing makes a collision unreachable, not just guarded.
    # This is why qb markets should pass position='QB'.
    chk("Anthony Brown as QB grades", A(2025, 1, "Anthony Brown",
                                        "rushing_yards", position="QB"), 11.0)
    chk("Lamar Jackson wk1 as QB grades",
        A(2025, 1, "Lamar Jackson", "rushing_yards", position="QB"), 55.0)
    chk("Michael Carter as RB grades",
        A(2025, 1, "Michael Carter", "rushing_yards", position="RB"), 42.0)

    # --- a LIST of positions, which is what receiving/receptions need to
    # preserve today's behaviour (their build_dataset filters to WR/TE/RB,
    # so the CB half of michael carter is already invisible there) ---
    chk("Michael Carter as WR/TE/RB grades",
        A(2025, 1, "Michael Carter", "rushing_yards",
          position=["WR", "TE", "RB"]), 42.0)
    chk("Solo Player in WR/TE/RB grades",
        A(2025, 1, "Solo Player", "receptions",
          position=["WR", "TE", "RB"]), 6.0)
    chk("QB not in WR/TE/RB", A(2025, 1, "Lamar Jackson", "rushing_yards",
                                position=["WR", "TE", "RB"]), None)

    # --- the alias path ---
    chk("Mitch Trubisky via alias", A(2025, 1, "Mitch Trubisky",
                                      "rushing_yards"), 7.0)

    # --- a null stat on an active player is a real zero ---
    chk("null grades as 0.0", A(2025, 1, "Zero Runner", "rushing_yards"), 0.0)

    # --- ordinary cases ---
    chk("solo receptions", A(2025, 1, "Solo Player", "receptions"), 6.0)
    chk("solo receiving", A(2025, 1, "Solo Player", "receiving_yards"), 88.0)

    # --- things that must return None ---
    chk("unknown player", A(2025, 1, "Nobody At All", "rushing_yards"), None)
    chk("unplayed week", A(2025, 9, "Solo Player", "receptions"), None)
    chk("missing column", A(2025, 1, "Solo Player", "passing_yards"), None)
    chk("wrong position filter",
        A(2025, 1, "Solo Player", "receptions", position="QB"), None)

    # --- the audit helper must find exactly the planted collisions ---
    col = data_utils.colliding_names(S)
    names = set(col["_norm"]) if len(col) else set()
    want = {"michael carter", "anthony brown", "lamar jackson"}
    if names != want:
        f.append(f"  colliding_names returned {sorted(names)}, "
                 f"expected {sorted(want)}")

    if f:
        print(f"SMOKE FAIL ({len(f)})")
        print("\n".join(f))
        return 1
    print("SMOKE PASS: collisions refused, unambiguous weeks still grade, "
          "position narrowing works, alias resolves, null is zero, audit "
          "finds all three")
    return 0


if __name__ == "__main__":
    sys.exit(main())
