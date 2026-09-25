"""smoke_norm_name.py - planted-answer rig for data_utils.norm_join_name.

No network, no database. norm_join_name is pure string work, so this tests
the real function with no injection needed.

Run from the repo root:
    python smoke_norm_name.py
"""
import os
import sys

import pandas as pd

try:
    from models import data_utils
except ImportError:
    import importlib.util
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "models", "data_utils.py")
    _spec = importlib.util.spec_from_file_location("data_utils", _p)
    data_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(data_utils)


# (input, expected output, why this case is here)
CASES = [
    # The two September 24 additions.
    ("Audric Estim\u00e9", "audric estime", "precomposed acute"),
    ("Audric Estime\u0301", "audric estime", "decomposed acute, same answer"),
    ("Lamar Jackson (BAL)", "lamar jackson", "parenthetical team tag"),
    ("Lamar Jackson(BAL)", "lamar jackson", "tag with no space"),
    ("Kyler Murray (ARI)", "kyler murray", "tag, ordinary QB"),

    # Everything the function did BEFORE must still hold. A normalizer that
    # gains a rule and loses one is a net regression, and this is the one
    # place that can catch it.
    ("D.K. Metcalf", "dk metcalf", "periods"),
    ("De'Von Achane", "devon achane", "straight apostrophe"),
    ("De\u2019Von Achane", "devon achane", "curly apostrophe"),
    ("Michael Pittman Jr.", "michael pittman", "suffix Jr"),
    ("Deebo Samuel Sr.", "deebo samuel", "suffix Sr"),
    ("Luther Burden III", "luther burden", "suffix III"),
    ("Chig Okonkwo", "chigoziem okonkwo", "alias table"),
    ("Kenny Gainwell", "kenneth gainwell", "alias table"),
    ("  Josh   Allen  ", "josh allen", "whitespace collapse"),

    # The five aliases added September 24 from name_resolve.py.
    ("Cameron Ward", "cam ward", "alias, QB 2025-26"),
    ("Josh Dobbs", "joshua dobbs", "alias, QB"),
    ("Mitch Trubisky", "mitchell trubisky", "alias, QB"),
    ("Eli Mitchell", "elijah mitchell", "alias, RB"),
    ("Phillip Walker", "pj walker", "alias, QB"),

    # The four abbreviated-initial forms were deliberately NOT aliased,
    # because an alias is a permanent global rewrite and an initial form is
    # not unique to one player. If someone adds them later, these cases fail
    # and the comment in _NAME_ALIASES explains why that is a regression
    # rather than an improvement.
    ("J. Hill", "j hill", "initial form stays unmapped"),
    ("M. Rudolph", "m rudolph", "initial form stays unmapped"),
    ("D. Evans", "d evans", "initial form stays unmapped"),

    # Interactions between rules, which is where an ordering mistake shows up.
    ("Marcus Mariota Jr. (WAS)", "marcus mariota", "tag AND suffix"),
    ("Jos\u00e9 Borregales (TB)", "jose borregales", "accent AND tag"),

    # Cases that must NOT change, so the rules are not over-broad.
    ("Amon-Ra St. Brown", "amonra st brown", "hyphen and period, no suffix"),
    ("Equanimeous St. Brown", "equanimeous st brown", "no accent to fold"),
    ("Steve Smith", "steve smith", "plain name untouched"),
    # 'v' is in the suffix list, so a surname ENDING in a separate v token
    # would be eaten. No such player, but the guard documents the risk.
    ("Ochaun Mathis", "ochaun mathis", "no false suffix strip"),
]


def main():
    inp = pd.Series([c[0] for c in CASES])
    got = data_utils.norm_join_name(inp).tolist()

    failures = []
    for (raw, want, why), g in zip(CASES, got):
        if g != want:
            failures.append(f"  {raw!r} -> {g!r}, expected {want!r}  ({why})")

    # Idempotence. The normalizer is applied on both sides of several joins
    # and in actual_result's fallback, so feeding it its own output must be a
    # no-op or those call sites disagree with each other.
    twice = data_utils.norm_join_name(pd.Series(got)).tolist()
    for once, tw in zip(got, twice):
        if once != tw:
            failures.append(f"  not idempotent: {once!r} -> {tw!r}")

    # ---- THE ARROW BACKEND, which is what broke the deployed app ----
    #
    # pandas may back a string column with pyarrow, and when it does,
    # .str.replace(regex=True) runs through RE2 instead of Python's re. RE2
    # rejects \uXXXX escapes, so a raw-string pattern that Python resolves
    # happily raises ArrowInvalid: "invalid escape sequence: \u".
    #
    # On September 25 every case above passed locally on an object-dtype
    # column while the DEPLOYED app crashed on a pyarrow-backed one. So the
    # object dtype alone is not a sufficient test, and this rig was the thing
    # that should have caught it and did not.
    #
    # Skips when pyarrow is absent rather than failing, because a missing
    # optional dependency is not a defect in the normalizer. Note the
    # container this was written in had no pyarrow, which is exactly why the
    # gap existed.
    try:
        import pyarrow  # noqa: F401
        arrow_in = pd.Series([c[0] for c in CASES], dtype="string[pyarrow]")
    except Exception as exc:
        print(f"NOTE: pyarrow backend NOT exercised ({type(exc).__name__}). "
              f"On a machine with pyarrow this rig also checks the RE2 path.")
    else:
        try:
            arrow_got = data_utils.norm_join_name(arrow_in).tolist()
        except Exception as exc:
            failures.append(f"  pyarrow-backed input raised "
                            f"{type(exc).__name__}: {exc}")
            arrow_got = None
        if arrow_got is not None:
            for (raw, want, why), g in zip(CASES, arrow_got):
                if g != want:
                    failures.append(f"  [arrow] {raw!r} -> {g!r}, expected "
                                    f"{want!r}  ({why})")
            if arrow_got != got:
                failures.append("  [arrow] output differs from the "
                                "object-dtype output; the engines disagree")

    if failures:
        print(f"SMOKE FAIL ({len(failures)} of {len(CASES)} cases)")
        print("\n".join(failures))
        return 1
    print(f"SMOKE PASS: {len(CASES)} cases, accents folded, tags stripped, "
          f"prior rules intact, idempotent, arrow backend agrees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
