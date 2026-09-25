"""smoke_qb_rushing.py - planted-answer rig for the September 24 qb_rushing changes.

No network and no database: data_utils.load_player_stats is replaced with a
synthetic QB frame carrying planted traps, and the Streamlit caches are
cleared between fits so each assertion sees the data it expects.

Covers the three things that changed, and nothing else:

  1. load_model's train_max_season actually restricts the training window.
     The whole point of the change is that the harness can score
     leave-season-out, so a parameter that is accepted and ignored would be
     worse than no parameter at all.
  2. actual_result returns 0.0 rather than None for a QB who did not run.
  3. actual_result refuses on a name collision rather than grading the wrong
     player, and still resolves a spelling variant through norm_join_name.

Run from the repo root:
    python smoke_qb_rushing.py
"""
import sys

import numpy as np
import pandas as pd


def _frame():
    """Synthetic QB player-weeks with four planted traps.

    - hi / lo are ordinary QBs whose seasons differ in level, so restricting
      the training window MUST move the coefficients.
    - "Mitch Trubisky" is spelled "Mitchell Trubisky" here, so grading a bet
      written the FanDuel way exercises the normalised fallback.
    - "Zero Runner" has a null rushing_yards in 2025 week 1, which must grade
      as 0.0 and not as None.
    - "Anthony Brown" appears TWICE in 2025 week 1 under two player_ids, the
      real collision measured in nflverse. Grading must refuse.
    """
    rows = []
    for pid, name, lvl in (("p1", "Hi Runner", 60.0), ("p2", "Lo Runner", 4.0)):
        for season in (2023, 2024, 2025):
            # 2025 is deliberately a different level, so a fit through 2024
            # and a fit through 2025 cannot coincide.
            mult = 3.0 if season == 2025 else 1.0
            for wk in range(1, 9):
                rows.append({"player_id": pid, "player_display_name": name,
                             "position": "QB", "season": season, "week": wk,
                             "rushing_yards": lvl * mult, "carries": lvl / 6.0,
                             "team": "BUF"})

    rows.append({"player_id": "p3", "player_display_name": "Mitchell Trubisky",
                 "position": "QB", "season": 2025, "week": 1,
                 "rushing_yards": 17.0, "carries": 3.0, "team": "PIT"})
    rows.append({"player_id": "p4", "player_display_name": "Zero Runner",
                 "position": "QB", "season": 2025, "week": 1,
                 "rushing_yards": np.nan, "carries": 0.0, "team": "TB"})
    rows.append({"player_id": "p5", "player_display_name": "Anthony Brown",
                 "position": "QB", "season": 2025, "week": 1,
                 "rushing_yards": 12.0, "carries": 2.0, "team": "BAL"})
    rows.append({"player_id": "p6", "player_display_name": "Anthony Brown",
                 "position": "QB", "season": 2025, "week": 1,
                 "rushing_yards": 99.0, "carries": 9.0, "team": "DAL"})
    # A non-QB that must never be visible to this module at all.
    rows.append({"player_id": "p7", "player_display_name": "Some Corner",
                 "position": "CB", "season": 2025, "week": 1,
                 "rushing_yards": 0.0, "carries": 0.0, "team": "NYJ"})
    return pd.DataFrame(rows)


def _schedules():
    out = []
    for season in (2023, 2024, 2025):
        for wk in range(1, 9):
            out.append({"season": season, "week": wk, "home_team": "BUF",
                        "away_team": "PIT", "spread_line": -3.0,
                        "total_line": 45.0})
            out.append({"season": season, "week": wk, "home_team": "TB",
                        "away_team": "DAL", "spread_line": 1.0,
                        "total_line": 41.0})
            out.append({"season": season, "week": wk, "home_team": "BAL",
                        "away_team": "NYJ", "spread_line": -7.0,
                        "total_line": 44.0})
    return pd.DataFrame(out)


def main():
    from models import data_utils, qb_rushing

    data_utils.load_player_stats = lambda seasons: _frame()
    data_utils.load_schedules = lambda seasons: _schedules()
    # _qb_player_stats is gone: actual_result now delegates to
    # data_utils.actual_stat, which keeps its own cache. Clear that one too or
    # the planted frame below is ignored and every grading case fails.
    for fn in (qb_rushing.build_dataset,):
        if hasattr(fn, "clear"):
            fn.clear()
    if hasattr(data_utils._stats_frame, "cache_clear"):
        data_utils._stats_frame.cache_clear()
    if hasattr(qb_rushing.load_model, "clear"):
        qb_rushing.load_model.clear()

    failures = []

    # ---- 1. the train window parameter must actually bind ----
    import inspect
    target = getattr(qb_rushing.load_model, "__wrapped__", qb_rushing.load_model)
    if "train_max_season" not in inspect.signature(target).parameters:
        failures.append("  load_model has no train_max_season parameter")

    m24, _ = qb_rushing.load_model(train_max_season=2024)
    if hasattr(qb_rushing.load_model, "clear"):
        qb_rushing.load_model.clear()
    m25, _ = qb_rushing.load_model(train_max_season=2025)

    same_int = abs(m24.intercept_ - m25.intercept_) < 1e-9
    same_coef = np.allclose(m24.coef_, m25.coef_, atol=1e-9)
    print(f"fit through 2024: intercept {m24.intercept_:+.4f} coef {m24.coef_}")
    print(f"fit through 2025: intercept {m25.intercept_:+.4f} coef {m25.coef_}")
    if same_int and same_coef:
        failures.append(
            "  train_max_season did NOT change the fit. The parameter is "
            "accepted and ignored, which is worse than not having it, because "
            "the harness would silently score in-sample.")

    # The default must still be the shipped behaviour.
    if qb_rushing.DEFAULT_TRAIN_MAX_SEASON != 2024:
        failures.append(
            f"  DEFAULT_TRAIN_MAX_SEASON is "
            f"{qb_rushing.DEFAULT_TRAIN_MAX_SEASON}, expected 2024")

    # ---- 2. a QB who did not run grades as zero, not None ----
    got = qb_rushing.actual_result(2025, 1, "Zero Runner")
    if got != 0.0:
        failures.append(f"  null rushing_yards graded as {got!r}, expected 0.0")

    # ---- 3. the normalised fallback, and the collision guard ----
    got = qb_rushing.actual_result(2025, 1, "Mitch Trubisky")
    if got != 17.0:
        failures.append(
            f"  'Mitch Trubisky' against 'Mitchell Trubisky' graded {got!r}, "
            f"expected 17.0 via the normalised fallback")

    got = qb_rushing.actual_result(2025, 1, "Anthony Brown")
    if got is not None:
        failures.append(
            f"  a two-player name collision graded {got!r}; it must return "
            f"None rather than pick one arbitrarily")

    # ---- guards: things that must NOT work ----
    if qb_rushing.actual_result(2025, 1, "Some Corner") is not None:
        failures.append("  a CB graded through the QB module")
    if qb_rushing.actual_result(2025, 17, "Hi Runner") is not None:
        failures.append("  a week with no stat row did not return None")
    if qb_rushing.actual_result(2025, 1, "Nobody At All") is not None:
        failures.append("  an unknown player did not return None")

    # ---- the ordinary case still works ----
    if qb_rushing.actual_result(2025, 1, "Hi Runner") != 180.0:
        failures.append(
            f"  ordinary grade returned "
            f"{qb_rushing.actual_result(2025, 1, 'Hi Runner')!r}, expected 180.0")

    print()
    if failures:
        print(f"SMOKE FAIL ({len(failures)})")
        print("\n".join(failures))
        return 1
    print("SMOKE PASS: train window binds, zero grades as zero, spelling "
          "fallback works, collision refuses, CB invisible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
