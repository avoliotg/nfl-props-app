"""smoke_rules.py - planted-answer rig for models/rules.py.

No network, no database: the air-yards history loader is replaced with a
synthetic frame, so the real evaluate() is exercised rather than a paraphrase.

Run from the repo root:
    python smoke_rules.py
"""
import sys

import numpy as np
import pandas as pd


def _props():
    """Props with the traps that matter.

    - two rushing lines straddling 46.5, so the boundary is tested as
      inclusive (<=) rather than strict
    - a QB rushing prop labelled 'rushing', which split_qb_rushing must
      relabel so the RB rule does not fire on a quarterback
    - a receiving prop whose shock clears z=1.0 and whose dev is below the
      veto: must fire
    - a receiving prop that clears z but whose dev is ABOVE the veto: must be
      vetoed, because the top dev quintile measured -0.117
    - a receiving prop with no air-yards history: must not fire
    - a receptions prop, which no rule covers
    """
    return pd.DataFrame([
        ("rushing", "RB OnTheLine", 46.5, 2026, 3, 40.0),
        ("rushing", "RB Below", 30.5, 2026, 3, 28.0),
        ("rushing", "RB Above", 55.5, 2026, 3, 60.0),
        ("rushing", "Kyler Murray", 20.5, 2026, 3, 22.0),
        ("receiving", "WR Fires", 55.5, 2026, 3, 50.0),
        ("receiving", "WR Vetoed", 55.5, 2026, 3, 90.0),
        ("receiving", "WR NoHistory", 40.5, 2026, 3, 38.0),
        ("receptions", "WR Fires", 4.5, 2026, 3, 4.0),
    ], columns=["market", "player", "line", "season", "week", "projection"])


def _hist():
    """Air-yards shock history. Planted so z is computable by hand.

    With shock_mean 0 and shock_sd 10, a shock_raw of 15 gives z = 1.5 and a
    shock_raw of 2 gives z = 0.2.
    """
    return pd.DataFrame([
        (2026, 3, "wr fires", 60.0, 45.0, 15.0),
        (2026, 3, "wr vetoed", 60.0, 45.0, 15.0),
        (2026, 3, "rb ontheline", 10.0, 8.0, 2.0),
    ], columns=["season", "week", "_key", "_recent", "_base", "shock_raw"])


def _positions():
    return pd.DataFrame([
        {"season": 2026, "player_norm": "kyler murray", "position": "QB"},
        {"season": 2026, "player_norm": "rb ontheline", "position": "RB"},
        {"season": 2026, "player_norm": "rb below", "position": "RB"},
        {"season": 2026, "player_norm": "rb above", "position": "RB"},
        {"season": 2026, "player_norm": "wr fires", "position": "WR"},
        {"season": 2026, "player_norm": "wr vetoed", "position": "WR"},
        {"season": 2026, "player_norm": "wr nohistory", "position": "WR"},
    ])


def main():
    try:
        import rules
        from models import data_utils
    except ImportError as e:
        print(f"could not import: {e}")
        print("rules.py belongs in the REPO ROOT, next to mc_pricing.py, "
              "not in models/.")
        return 1

    f = []
    props = _props()
    # Capture the SHIPPED constants before section 1 blanks them, so the
    # describe() call at the end reports the real state. Capturing after the
    # blanking meant describe() reported "calibrated: False" on a file that
    # is in fact calibrated, which is exactly the kind of misleading output a
    # smoke test should not produce.
    orig_cal = dict(rules.CALIBRATION["receiving"])

    # ---- 1. UNCALIBRATED: the receiving rule must REFUSE, not guess ----
    rules.CALIBRATION["receiving"] = {"shock_mean": None, "shock_sd": None,
                                      "dev_veto_at": None}
    orig_hist = rules._air_yards_history
    orig_pos = getattr(data_utils, "_position_rows", None)
    data_utils._position_rows = lambda key: _positions()
    rules._air_yards_history = lambda seasons: _hist()

    fired, rep = rules.evaluate(props)
    rec = rep["rules"]["receiving_air_yards_under"]
    if rec["fired"] != 0:
        f.append(f"  uncalibrated: receiving fired {rec['fired']}, expected 0")
    if "NOT CALIBRATED" not in (rec.get("reason") or ""):
        f.append(f"  uncalibrated: reason was {rec.get('reason')!r}, expected "
                 f"it to say NOT CALIBRATED")
    # rushing must still work: it needs no calibration
    rush = rep["rules"]["rushing_low_line_under"]
    # TWO, not three: 46.5 fires because the bound is inclusive, 30.5 fires,
    # 55.5 is above the cut, and Kyler Murray is relabelled to qb_rushing by
    # split_qb_rushing so the RB rule cannot reach him. An earlier version of
    # this rig expected 3 and was simply miscounted.
    if rush["fired"] != 2:
        f.append(f"  uncalibrated: rushing fired {rush['fired']}, expected 2 "
                 f"(46.5 inclusive and 30.5; NOT 55.5, NOT the QB)")

    # ---- 2. CALIBRATED ----
    rules.CALIBRATION["receiving"] = {"shock_mean": 0.0, "shock_sd": 10.0,
                                      "dev_veto_at": 10.0}
    fired, rep = rules.evaluate(props)
    got = {(r["rule"], r["player"]) for _, r in fired.iterrows()}

    want = {
        ("rushing_low_line_under", "RB OnTheLine"),   # 46.5 is inclusive
        ("rushing_low_line_under", "RB Below"),
        ("receiving_air_yards_under", "WR Fires"),     # z 1.5, dev -5.5
    }
    for w in want:
        if w not in got:
            f.append(f"  expected firing missing: {w}")
    forbidden = {
        ("rushing_low_line_under", "RB Above"),        # 55.5 > 46.5
        ("rushing_low_line_under", "Kyler Murray"),    # relabelled qb_rushing
        ("receiving_air_yards_under", "WR Vetoed"),    # dev +34.5 > veto
        ("receiving_air_yards_under", "WR NoHistory"),  # no shock
    }
    for b in forbidden:
        if b in got:
            f.append(f"  rule fired where it must not: {b}")

    rec = rep["rules"]["receiving_air_yards_under"]
    if rec.get("vetoed_by_model") != 1:
        f.append(f"  vetoed_by_model was {rec.get('vetoed_by_model')}, "
                 f"expected 1")
    if rec.get("triggered_before_veto") != 2:
        f.append(f"  triggered_before_veto was "
                 f"{rec.get('triggered_before_veto')}, expected 2")
    if rec.get("no_air_yards_history") != 1:
        f.append(f"  no_air_yards_history was "
                 f"{rec.get('no_air_yards_history')}, expected 1")

    # every firing must be an UNDER; neither rule has an over side
    if len(fired) and set(fired["side"]) != {"UNDER"}:
        f.append(f"  sides were {set(fired['side'])}, expected only UNDER")

    # ---- 3. missing projection must REFUSE, not skip the veto ----
    fired3, rep3 = rules.evaluate(props.drop(columns=["projection"]))
    rec3 = rep3["rules"]["receiving_air_yards_under"]
    if rec3["fired"] != 0:
        f.append(f"  no projection: receiving fired {rec3['fired']}, "
                 f"expected 0")
    if "projection" not in (rec3.get("reason") or ""):
        f.append(f"  no projection: reason was {rec3.get('reason')!r}")
    if rep3["rules"]["rushing_low_line_under"]["fired"] != 2:
        f.append("  no projection: rushing should be unaffected (expected 2)")

    # ---- 4. an empty frame must not raise ----
    try:
        fired4, rep4 = rules.evaluate(props.iloc[0:0])
        if len(fired4):
            f.append("  empty input produced firings")
    except Exception as e:
        f.append(f"  empty input raised {type(e).__name__}: {e}")

    # ---- 5. a missing required column must raise clearly ----
    try:
        rules.evaluate(props.drop(columns=["line"]))
        f.append("  missing `line` did not raise")
    except KeyError:
        pass
    except Exception as e:
        f.append(f"  missing `line` raised {type(e).__name__}, expected KeyError")

    rules._air_yards_history = orig_hist
    rules.CALIBRATION["receiving"] = orig_cal   # so describe() below reports
    if orig_pos is not None:                    # the SHIPPED state, not the
        data_utils._position_rows = orig_pos    # test's planted values

    print()
    print(rules.describe())
    print()
    if f:
        print(f"SMOKE FAIL ({len(f)})")
        print("\n".join(f))
        return 1
    print("SMOKE PASS: boundary inclusive, QB excluded, veto applied, "
          "uncalibrated refuses, missing projection refuses, empty safe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
