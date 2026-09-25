"""smoke_qb_split.py - planted-answer rig for data_utils.split_qb_rushing.

No network, no database, no repo data. The position frame is injected, so this
exercises the real function rather than a paraphrase of it.

Run from the repo root:
    python smoke_qb_split.py

Every case has a KNOWN answer, including known-zero ones. A tool that
reassigns a running back, or that reassigns nothing at all, is broken.
"""
import sys
import pandas as pd

try:
    from models import data_utils
except ImportError:
    # models/__init__.py does not exist yet (plan item J4), so fall back to
    # loading the file directly rather than failing on a packaging detail.
    import importlib.util
    import os
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "models", "data_utils.py")
    _spec = importlib.util.spec_from_file_location("data_utils", _p)
    data_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(data_utils)


def _positions():
    """Planted position table. Note the traps:

    - Kyler Murray is a QB in both seasons: the ordinary case.
    - Taysom Hill is listed QB in 2023 and TE in 2024, so the per-season map
      must win over the pooled mode.
    - Bijan Robinson is an RB and must never be touched.
    - Jalen Hurts has no 2025 stat row, so he must resolve through the POOLED
      fallback rather than being left unmatched.
    - Marcus Mariota's line spells him with a suffix the stat table lacks.
    """
    return pd.DataFrame([
        {"season": 2023, "player_norm": "kyler murray", "position": "QB"},
        {"season": 2025, "player_norm": "kyler murray", "position": "QB"},
        {"season": 2023, "player_norm": "taysom hill", "position": "QB"},
        {"season": 2024, "player_norm": "taysom hill", "position": "TE"},
        {"season": 2024, "player_norm": "taysom hill", "position": "TE"},
        {"season": 2023, "player_norm": "bijan robinson", "position": "RB"},
        {"season": 2025, "player_norm": "bijan robinson", "position": "RB"},
        {"season": 2023, "player_norm": "jalen hurts", "position": "QB"},
        {"season": 2023, "player_norm": "marcus mariota", "position": "QB"},
    ])


def _lines():
    return pd.DataFrame([
        # market, season, player, expected market after the split
        ("rushing",    2023, "Kyler Murray",      "qb_rushing"),
        ("rushing",    2025, "Kyler Murray",      "qb_rushing"),
        ("rushing",    2023, "Taysom Hill",       "qb_rushing"),
        ("rushing",    2024, "Taysom Hill",       "rushing"),
        ("rushing",    2023, "Bijan Robinson",    "rushing"),
        ("rushing",    2025, "Bijan Robinson",    "rushing"),
        ("rushing",    2025, "Jalen Hurts",       "qb_rushing"),
        ("rushing",    2023, "Marcus Mariota Jr.", "qb_rushing"),
        ("rushing",    2023, "Nobody At All",     "rushing"),
        ("receptions", 2023, "Kyler Murray",      "receptions"),
        ("receiving",  2023, "Bijan Robinson",    "receiving"),
        ("qb_rushing", 2023, "Kyler Murray",      "qb_rushing"),
    ], columns=["market", "season", "player", "expected"])


def main():
    lines = _lines()
    expected = lines.pop("expected")

    out, report = data_utils.split_qb_rushing(
        lines, position_rows=_positions())

    got = out["market"].tolist()
    want = expected.tolist()

    failures = []
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            failures.append(
                f"  row {i}: {lines.iloc[i]['player']} "
                f"({lines.iloc[i]['season']}) got '{g}', expected '{w}'")

    # Planted report values. These are the checks that catch a function which
    # relabels correctly but miscounts, which is the class of defect that
    # produced a wrong number with figures attached.
    checks = [
        ("rows_in_from_market", 9),
        ("reassigned", 5),
        ("unmatched", 1),
        ("distinct_qbs", 4),
        ("matched_pooled", 1),
    ]
    for key, want_val in checks:
        if report.get(key) != want_val:
            failures.append(
                f"  report['{key}'] = {report.get(key)!r}, expected {want_val!r}")

    # Idempotence: a second pass must be a no-op.
    twice, report2 = data_utils.split_qb_rushing(out, position_rows=_positions())
    if twice["market"].tolist() != got:
        failures.append("  second pass changed the frame; not idempotent")
    if report2["reassigned"] != 0:
        failures.append(
            f"  second pass reassigned {report2['reassigned']} rows, expected 0")

    # The input frame must not be mutated in place.
    if lines["market"].tolist() != _lines()["market"].tolist():
        failures.append("  the input frame was mutated in place")

    # The gate must ABORT, not warn, when the match rate is below the bar.
    try:
        data_utils.split_qb_rushing(
            _lines().drop(columns=["expected"]),
            position_rows=_positions(), min_match_rate=0.99)
    except RuntimeError:
        pass
    else:
        failures.append("  min_match_rate=0.99 did not raise on an 0.889 rate")

    print("report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print()

    if failures:
        print("SMOKE FAIL")
        print("\n".join(failures))
        return 1
    print("SMOKE PASS: 12 rows, 5 reassigned, 1 unmatched, idempotent, gate aborts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
